import os

import fire
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch

from src.patch import Patch
from src.eval import get_model_from_run
from src.samplers import get_task_sampler
from src.plot import rename

batch_size = 100
test_max_points = 400


def aggregate_metrics(metrics, bootstrap_trials=batch_size):
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


def main(run_path="./models/linear_regression/dim_4-16/main/azi2cja7"):

    _, config = get_model_from_run(run_path, only_conf=True)

    if config.get("patch"):
        patch = Patch(config["patch"]["name"])
        patch.apply()

    input_dims = config["task"]["curriculum"]["dims"]["end"]
    n_points = config["task"]["curriculum"]["points"]["end"]

    kwargs = {
        "task": config["task"]["name"],
        "max_length": config["max_length"],
        "n_in_dims": input_dims,
        "max_points": n_points,
        "patch": config["patch"] if config.get("patch") else None,
    }
    model, _ = get_model_from_run(run_path, **kwargs)
    model.cuda().eval()
    device = next(model.parameters()).device
    if config["model"]["type"] == "ssm":
        model_name = config["model"]["family"]
    elif config["model"]["type"] == "decoder":
        if config["model"].get("rope_scaling"):
            model_name = config["model"]["rope_scaling"]["type"]
        else:
            model_name = "rope"
    model_name = rename(model_name)

    task_sampler = get_task_sampler(
        **config["task"],
        n_dims=input_dims,
        batch_size=batch_size,
    )
    task = task_sampler()
    loss_func = task.get_training_metric()
    metric = task.get_metric()

    xs = task.sample_xs(
        b_size=batch_size,
        n_points=n_points,
        n_dims_truncated=input_dims,
    )
    ys = task.evaluate(xs)

    instruction_x = xs[:, :input_dims * 2, :]
    ans_x = xs[:, -input_dims:, :]
    target_x = xs[:, -input_dims // 2:, :]

    noise_type = "duplicated"
    noise_x = instruction_x[:, -1, :].unsqueeze(1)

    # log duplicated atk
    if not os.path.exists(os.path.join(run_path, f"{noise_type}")):
        os.makedirs(os.path.join(run_path, f"{noise_type}"))

    for noise_count in range(5, 21):
        x_test = torch.cat(
            (
                xs[:, :xs.shape[1] - noise_count, :],
                noise_x.repeat(1, noise_count, 1),
                target_x,
            ),
            dim=1,
        )
        y_test = task.evaluate(x_test)

        with torch.no_grad():
            pred = model(x_test.to(device), y_test.to(device), loss_func=loss_func)

        test = metric(pred.logits.cpu(), y_test).detach().cpu()
        test = aggregate_metrics(test)

        x = np.arange(len(test["mean"]))
        start = xs.shape[1] - noise_count
        plt.plot(test["mean"], "-", label="Random points", color="xkcd:blue grey", lw=2, zorder=1)
        plt.plot(x[start + noise_count:], test["mean"][start + noise_count:], "-", color="xkcd:red", lw=3, zorder=2)
        plt.scatter(x[start], test["mean"][start], c="red", marker="|", s=30, zorder=3)
        plt.annotate(
            f"{start}",
            xy=(start, test["mean"][start]),
            xytext=(start, test["mean"][start] + 0.5),
            fontsize=12,
            color="red",
            ha='center',
        )
        low = test["bootstrap_low"]
        high = test["bootstrap_high"]
        plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=1)

        plt.axhline(input_dims, ls="--", color="xkcd:greyish", label="zero estimator")
        plt.axvspan(-1, n_points, color="gray", alpha=0.1)
        plt.xlabel("# of in-context examples")
        plt.ylabel("squared error")
        plt.xlim(-1, len(low) + 0.1)
        plt.ylim(-0.5, 20)
        plt.legend()
        plt.tight_layout()
        plt.title(f"{noise_type} attack on {model_name} - noise points: {noise_count}")
        plt.savefig(os.path.join(run_path, f"{noise_type}", f"noise_points={noise_count}.png"))
        plt.clf()

    # Inst + noise + tar
    baseline_x = task.sample_xs(
        b_size=batch_size,
        n_points=test_max_points,
        n_dims_truncated=input_dims,
    )
    baseline_y = task.evaluate(baseline_x)

    with torch.no_grad():
        baseline_pred = model(baseline_x.to(device), baseline_y.to(device), loss_func=loss_func)

    baseline = metric(baseline_pred.logits.cpu(), baseline_y).detach().cpu()
    baseline = aggregate_metrics(baseline)

    begin_x = torch.cat(
        (
            instruction_x,
            noise_x.repeat(1, xs.shape[1] - instruction_x.shape[1], 1),
            target_x,
        ),
        dim=1,
    )
    begin_y = task.evaluate(begin_x)

    with torch.no_grad():
        begin_pred = model(begin_x.to(device), begin_y.to(device), loss_func=loss_func)

    begin = metric(begin_pred.logits.cpu(), begin_y).detach().cpu()
    begin = aggregate_metrics(begin)

    middle_x = torch.cat(
        (
            instruction_x,
            noise_x.repeat(1, (test_max_points - instruction_x.shape[1] - target_x.shape[1]) // 2, 1),
            target_x,
            # noise_x.repeat(1, (test_max_points - instruction_x.shape[1] - target_x.shape[1]) // 2, 1),
        ),
        dim=1,
    )
    middle_y = task.evaluate(middle_x)

    with torch.no_grad():
        middle_pred = model(middle_x.to(device), middle_y.to(device), loss_func=loss_func)

    middle = metric(middle_pred.logits.cpu(), middle_y).detach().cpu()
    middle = aggregate_metrics(middle)

    end_x = torch.cat(
        (
            instruction_x,
            noise_x.repeat(1, test_max_points - instruction_x.shape[1] - target_x.shape[1], 1),
            target_x,
        ),
        dim=1,
    )
    end_y = task.evaluate(end_x)

    with torch.no_grad():
        end_pred = model(end_x.to(device), end_y.to(device), loss_func=loss_func)

    end = metric(end_pred.logits.cpu(), end_y).detach().cpu()
    end = aggregate_metrics(end)

    plt.plot(baseline["mean"], "-", label="Random points", color="xkcd:charcoal", lw=2, zorder=1)
    low = baseline["bootstrap_low"]
    high = baseline["bootstrap_high"]
    plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=1)

    x = np.arange(test_max_points)
    start, endp = xs.shape[1], xs.shape[1] + target_x.shape[1]
    plt.plot(begin["mean"], "-", label="target dist = 50", color="xkcd:green", lw=2, zorder=4)
    plt.plot(x[start:endp], begin["mean"][start:endp], "-", color="xkcd:forest", lw=3, zorder=5)
    low = begin["bootstrap_low"]
    high = begin["bootstrap_high"]
    # plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    start = instruction_x.shape[1] + (test_max_points - instruction_x.shape[1] - target_x.shape[1]) // 2
    endp = start + target_x.shape[1]
    plt.plot(middle["mean"], "-", label="target dist = 212", color="xkcd:tangerine", lw=2, zorder=3)
    plt.plot(x[start:endp], middle["mean"][start:endp], "-", color="xkcd:orange red", lw=3, zorder=5)
    low = middle["bootstrap_low"]
    high = middle["bootstrap_high"]
    # plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    start = instruction_x.shape[1] + (test_max_points - instruction_x.shape[1] - target_x.shape[1])
    plt.plot(end["mean"], "-", label="target dist = 392", color="xkcd:cerulean", lw=2, zorder=2)
    plt.plot(x[start:], end["mean"][start:], "-", color="xkcd:prussian blue", lw=3, zorder=5)
    low = end["bootstrap_low"]
    high = end["bootstrap_high"]
    # plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    plt.axhline(input_dims, ls="--", color="xkcd:greyish", label="zero estimator")
    plt.axvspan(-1, n_points, color="gray", alpha=0.1)
    plt.xlabel("# of in-context examples")
    plt.xlim(-1, len(low) + 0.1)
    if config["task"]["data"] == "gaussian":
        plt.ylabel("squared error")
        plt.ylim(-0.1, 30)
    elif config["task"]["data"] == "boolean":
        plt.ylabel("accuracy")
        plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.title(f"{noise_type} attack on {model_name}")
    plt.savefig(os.path.join(run_path, f"{noise_type}", f"inst+noise+tar.png"))
    plt.clf()

    # change inst position
    baseline_x = torch.cat((noise_x.repeat(1, test_max_points, 1),), dim=1)
    baseline_y = task.evaluate(baseline_x)

    with torch.no_grad():
        baseline_pred = model(baseline_x.to(device), baseline_y.to(device), loss_func=loss_func)

    baseline = metric(baseline_pred.logits.cpu(), baseline_y).detach().cpu()
    baseline = aggregate_metrics(baseline)

    begin_x = torch.cat((
        instruction_x,
        noise_x.repeat(1, test_max_points - instruction_x.shape[1], 1),
    ), dim=1)
    begin_y = task.evaluate(begin_x)

    with torch.no_grad():
        begin_pred = model(begin_x.to(device), begin_y.to(device), loss_func=loss_func)

    begin = metric(begin_pred.logits.cpu(), begin_y).detach().cpu()
    begin = aggregate_metrics(begin)

    middle_x = torch.cat(
        (
            noise_x.repeat(1, (test_max_points - instruction_x.shape[1]) // 2, 1),
            instruction_x,
            noise_x.repeat(1, (test_max_points - instruction_x.shape[1]) // 2, 1),
        ),
        dim=1,
    )
    middle_y = task.evaluate(middle_x)

    with torch.no_grad():
        middle_pred = model(middle_x.to(device), middle_y.to(device), loss_func=loss_func)

    middle = metric(middle_pred.logits.cpu(), middle_y).detach().cpu()
    middle = aggregate_metrics(middle)

    end_x = torch.cat(
        (
            noise_x.repeat(1, test_max_points - instruction_x.shape[1], 1),
            instruction_x,
        ),
        dim=1,
    )
    end_y = task.evaluate(end_x)

    with torch.no_grad():
        end_pred = model(end_x.to(device), end_y.to(device), loss_func=loss_func)

    end = metric(end_pred.logits.cpu(), end_y).detach().cpu()
    end = aggregate_metrics(end)

    plt.plot(baseline["mean"], "-", label="All duplicated", color="xkcd:charcoal", lw=2, zorder=1)
    low = baseline["bootstrap_low"]
    high = baseline["bootstrap_high"]
    plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=1)

    plt.axhline(input_dims, ls="--", color="xkcd:greyish", label="zero estimator")
    plt.axvspan(-1, n_points, color="gray", alpha=0.1)
    plt.xlabel("# of in-context examples")
    plt.xlim(-1, len(low) + 0.1)
    if config["task"]["data"] == "gaussian":
        plt.ylabel("squared error")
        plt.ylim(-0.1, 30)
    elif config["task"]["data"] == "boolean":
        plt.ylabel("accuracy")
        plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.title(f"{noise_type} attack on {model_name}")
    plt.savefig(os.path.join(run_path, f"{noise_type}", f"baseline_all_dup.png"))
    plt.clf()

    plt.plot(baseline["mean"], "-", label="All duplicated", color="xkcd:charcoal", lw=2, zorder=1)
    low = baseline["bootstrap_low"]
    high = baseline["bootstrap_high"]
    plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=1)

    x = np.arange(test_max_points)

    endp = instruction_x.shape[1]
    plt.plot(begin["mean"], "-", label="instruction at begin", color="xkcd:green", lw=2, zorder=4)
    plt.axvspan(0, endp, color="xkcd:light red", alpha=0.1)
    low = begin["bootstrap_low"]
    high = begin["bootstrap_high"]
    # plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    start = (test_max_points - instruction_x.shape[1]) // 2
    endp = start + instruction_x.shape[1]
    plt.plot(middle["mean"], "-", label="instruction at middle", color="xkcd:tangerine", lw=2, zorder=3)
    plt.axvspan(start, endp, color="xkcd:light red", alpha=0.1)
    low = middle["bootstrap_low"]
    high = middle["bootstrap_high"]
    # plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    start = test_max_points - instruction_x.shape[1]
    plt.plot(end["mean"], "-", label="instruction at end", color="xkcd:cerulean", lw=2, zorder=2)
    plt.axvspan(start, test_max_points, color="xkcd:light red", alpha=0.1)
    low = end["bootstrap_low"]
    high = end["bootstrap_high"]
    # plt.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    plt.axhline(input_dims, ls="--", color="xkcd:greyish", label="zero estimator")
    plt.axvspan(-1, n_points, color="gray", alpha=0.1)
    plt.xlabel("# of in-context examples")
    plt.xlim(-1, len(low) + 0.1)
    if config["task"]["data"] == "gaussian":
        plt.ylabel("squared error")
        plt.ylim(-0.1, 30)
    elif config["task"]["data"] == "boolean":
        plt.ylabel("accuracy")
        plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.title(f"Changing inst on {model_name}")
    plt.savefig(os.path.join(run_path, f"{noise_type}", f"change_inst_position.png"))
    plt.clf()

    # # lost in the middle test
    begin_x = torch.cat(
        (
            instruction_x,
            ans_x,
            noise_x.repeat(1, test_max_points - instruction_x.shape[1] - ans_x.shape[1] - target_x.shape[1], 1),
            target_x,
        ),
        dim=1,
    )
    begin_y = task.evaluate(begin_x)

    with torch.no_grad():
        begin_pred = model(begin_x.to(device), begin_y.to(device), loss_func=loss_func)

    begin = metric(begin_pred.logits.cpu(), begin_y).detach().cpu()
    begin = aggregate_metrics(begin)

    middle_x = torch.cat(
        (
            instruction_x,
            noise_x.repeat(1, (test_max_points - instruction_x.shape[1] - ans_x.shape[1] - target_x.shape[1]) // 2, 1),
            ans_x,
            noise_x.repeat(1, (test_max_points - instruction_x.shape[1] - ans_x.shape[1] - target_x.shape[1]) // 2, 1),
            target_x,
        ),
        dim=1,
    )
    middle_y = task.evaluate(middle_x)

    with torch.no_grad():
        middle_pred = model(middle_x.to(device), middle_y.to(device), loss_func=loss_func)

    middle = metric(middle_pred.logits.cpu(), middle_y).detach().cpu()
    middle = aggregate_metrics(middle)

    end_x = torch.cat(
        (
            instruction_x,
            noise_x.repeat(1, test_max_points - instruction_x.shape[1] - ans_x.shape[1] - target_x.shape[1], 1),
            ans_x,
            target_x,
        ),
        dim=1,
    )
    end_y = task.evaluate(end_x)

    with torch.no_grad():
        end_pred = model(end_x.to(device), end_y.to(device), loss_func=loss_func)

    end = metric(end_pred.logits.cpu(), end_y).detach().cpu()
    end = aggregate_metrics(end)

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(5, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[1:4, 0])
    zoom = fig.add_subplot(gs[:, 1])

    x = np.arange(test_max_points)
    start = test_max_points - target_x.shape[1]
    ax.axvspan(start, test_max_points, color="xkcd:light red", alpha=0.1)
    ax.plot(begin["mean"], "-", label="ans at begin", color="xkcd:green", lw=2, zorder=4)
    low = begin["bootstrap_low"]
    high = begin["bootstrap_high"]
    # ax.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    ax.plot(middle["mean"], "-", label="ans at middle", color="xkcd:tangerine", lw=2, zorder=3)
    low = middle["bootstrap_low"]
    high = middle["bootstrap_high"]
    # ax.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    ax.plot(end["mean"], "-", label="ans at end", color="xkcd:cerulean", lw=2, zorder=2)
    low = end["bootstrap_low"]
    high = end["bootstrap_high"]
    # ax.fill_between(range(len(low)), low, high, alpha=0.3, zorder=2)

    ax.axhline(input_dims, ls="--", color="xkcd:greyish", label="zero estimator")
    ax.axvspan(-1, n_points, color="gray", alpha=0.1)
    ax.set_xlabel("# of in-context examples")
    ax.set_xlim(-1, len(low) + 0.1)

    zoom_start = test_max_points - target_x.shape[1] - 15
    zoom.plot(x[zoom_start:], begin["mean"][zoom_start:], "-", color="xkcd:green", lw=2, zorder=4)
    zoom.plot(x[zoom_start:], middle["mean"][zoom_start:], "-", color="xkcd:tangerine", lw=2, zorder=3)
    zoom.plot(x[zoom_start:], end["mean"][zoom_start:], "-", color="xkcd:cerulean", lw=2, zorder=2)
    zoom.axhline(input_dims, ls="--", color="xkcd:greyish")
    zoom.axvspan(zoom_start, test_max_points, color="gray", alpha=0.1)
    zoom.axvspan(start, test_max_points, color="xkcd:light red", alpha=0.1)
    zoom.set_xlim(zoom_start, test_max_points)

    if config["task"]["data"] == "gaussian":
        y_max = 30
        ax.set_ylabel("squared error")
    elif config["task"]["data"] == "boolean":
        y_max = 1.1
        ax.set_ylabel("accuracy")

    ax.set_ylim(-0.1, y_max)
    zoom.set_ylim(-0.1, y_max)

    ax.legend()
    ax.set_title(f"{config['task']['name']} - lost in middle test on {model_name}")
    zoom.set_title("Zoomed In")

    con1 = ConnectionPatch(
        xyA=(start, -0.1),
        coordsA=ax.transData,
        xyB=(zoom_start, -0.1),
        coordsB=zoom.transData,
        color="red",
    )
    con2 = ConnectionPatch(
        xyA=(start, y_max),
        coordsA=ax.transData,
        xyB=(zoom_start, y_max),
        coordsB=zoom.transData,
        color="red",
    )
    fig.add_artist(con1)
    fig.add_artist(con2)

    plt.tight_layout()
    plt.savefig(os.path.join(run_path, f"{noise_type}", f"lost_in_mid.png"))
    plt.clf()


if __name__ == "__main__":
    sns.set_theme('notebook', 'darkgrid')
    palette = sns.color_palette('colorblind')
    fire.Fire(main)
