import itertools
import json
import os

import evaluate
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from .config_loader import Loader
from .samplers import get_task_sampler
from .utils import build_model


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, correction=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def compute_attns(
    models,
    prompting_strategies,
    num_samples,
):
    ...


def compute_evals(models, eval_methods, metric_path):

    all_metrics = {}

    pbar = tqdm(range(0, len(eval_methods) * len(models)))
    for method, kwargs in eval_methods.items():
        metrics = {}
        for model in models:
            metrics[model.name] = eval_model(model, **kwargs)
            pbar.update(1)
        all_metrics[method] = metrics

    if metric_path is not None:
        with open(metric_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

    return all_metrics


def compute_metrics(eval_preds):
    output, labels = eval_preds
    metric = evaluate.load("mse")
    all_metrics = metric.compute(
        predictions=flatten(output[0]),
        references=flatten(labels),
    )

    return {"mse_loss": all_metrics["mse"]}


def eval_attn(
    model,
    task,
):
    ...


def eval_batch(model, task, xs, xp, **kwargs):
    if isinstance(model, torch.nn.Module):
        device = next(model.parameters()).device
    else:
        device = torch.device("cpu")
    with torch.no_grad():
        ys = task.evaluate(xs)
        loss_func = task.get_training_metric()
        if xp is None:
            pred = model(xs.to(device), ys.to(device), loss_func=loss_func)
            loss = task.get_metric()(pred.logits.cpu(), ys).detach().cpu().numpy()
        else:
            b_size, n_points, _ = xs.shape
            loss = np.zeros((b_size, n_points))
            for i in range(n_points):
                xs_comb = torch.cat((xs[:, :i, :], xp[:, i:, :]), dim=1)
                ys = task.evaluate(xs_comb)

                pred = model(xs_comb.to(device), ys.to(device), inds=[i], loss_func=loss_func)
                loss[:, i] = task.get_metric()(pred.logits.cpu(), ys).detach().cpu().numpy()[:, i]

    return torch.tensor(loss)


def eval_model(
    model,
    task,
    data,
    input_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    **kwargs,
):

    assert num_eval_examples % batch_size == 0
    task_sampler = get_task_sampler(
        data=data,
        name=task,
        n_dims=input_dims,
        batch_size=batch_size,
        n_points=n_points,
        **kwargs,
    )

    metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for _ in range(num_eval_examples // batch_size):
        xs, xp = generating_func(
            task_sampler(),
            n_points,
            batch_size,
            **kwargs,
        )
        metric = eval_batch(
            model,
            task_sampler(),
            xs,
            xp,
        )
        metrics.append(metric)

    return aggregate_metrics(torch.cat(metrics, dim=0))


def flatten(array):
    return list(itertools.chain.from_iterable(array))


def gen_standard(task_sampler, n_points, batch_size, **kwargs):
    xs = task_sampler.sample_xs(n_points, batch_size)

    return xs, None


def gen_ood_length(task_sampler, n_points, batch_size, ood_size=4, **kwargs):
    xs = task_sampler.sample_xs(n_points * ood_size, batch_size)

    return xs, None


def gen_opposite_quadrants(task_sampler, n_points, b_size, ood_size=1, **kwargs):
    xs = task_sampler.sample_xs(n_points * ood_size, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(task_sampler, n_points, b_size, ood_size=1, **kwargs):
    xs = task_sampler.sample_xs(n_points * ood_size, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size, ood_size=1, **kwargs):
    xs = data_sampler.sample_xs(n_points * ood_size, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i:i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection)
        xs_test_post_i_normalized = (xs_test_post_i_orthogonalized * xs_test_post_i.norm(dim=2).unsqueeze(2) /
                                     xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2))

        xs_test_post[:, i:i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size, **kwargs):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i:i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def get_model_from_run(run_path, step=-1, only_conf=False, **kwargs):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader)
    if only_conf:
        return None, config

    model = build_model(
        config["model"],
        **kwargs,
    )

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, config
