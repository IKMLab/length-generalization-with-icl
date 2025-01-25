import gc
import os

import fire
import seaborn as sns
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.patch import Patch
from src.eval import get_model_from_run
from src.samplers import get_task_sampler
from src.plot import rename


def main(run_path="./models/linear_regression/dim_4-16/onify6up"):
    num_samples = 64
    batch_size = 32
    test_max_points = 200

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

    task_sampler = get_task_sampler(
        **config["task"],
        n_dims=input_dims,
        batch_size=batch_size,
    )

    task = task_sampler()
    loss_func = task.get_training_metric()
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

    attns = []
    for batch in range(num_samples // batch_size):
        xs = task.sample_xs(
            b_size=batch_size,
            n_points=n_points,
            n_dims_truncated=input_dims,
        )
        ys = task.evaluate(xs)
        with torch.no_grad():
            pred = model(xs.to(device), ys.to(device), loss_func=loss_func, output_attentions=True)
            # attn shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
            attns.append(tuple(t.cpu() for t in pred.attentions))

        del xs, ys, pred
        gc.collect()
        torch.cuda.empty_cache()

    attns = np.concatenate(attns, axis=1)

    layer = -1
    head = 0
    fig, ax = plt.subplots()
    ax.plot(attns[layer, :, head, -2, ::2].mean(axis=0), label="Head 0")
    ax.plot(attns[layer, :, head + 1, -2, ::2].mean(axis=0), label="Head 1")
    ax.plot(attns[layer, :, head + 2, -2, ::2].mean(axis=0), label="Head 2")
    ax.plot(attns[layer, :, head + 3, -2, ::2].mean(axis=0), label="Head 3")
    ax.set_xlim()
    ax.legend(fancybox=True, shadow=True)
    ax.figure.set_size_inches(6, 4)
    ax.set(xlabel="Token position", ylabel="Attention weight")
    ax.set_title(f"{config.get('task', {}).get('name', '')} - {model_name} - Layer {np.shape(attns)[0]}")
    plt.tight_layout()

    save_path = os.path.join(
        run_path, "attn", f"attetion_layer={np.shape(attns)[0]}_head={head}-{head+3}_length={np.shape(attns)[3]}.png")
    plt.savefig(save_path)
    plt.clf()

    if np.shape(attns)[2] > 4:
        fig, ax = plt.subplots()
        ax.plot(attns[layer, :, head + 4, -2, ::2].mean(axis=0), label="Head 4")
        ax.plot(attns[layer, :, head + 5, -2, ::2].mean(axis=0), label="Head 5")
        ax.plot(attns[layer, :, head + 6, -2, ::2].mean(axis=0), label="Head 6")
        ax.plot(attns[layer, :, head + 7, -2, ::2].mean(axis=0), label="Head 7")
        ax.legend(fancybox=True, shadow=True)
        ax.figure.set_size_inches(6, 4)
        ax.set(xlabel="Token position", ylabel="Attention weight")
        ax.set_title(f"{config.get('task', {}).get('name', '')} - {model_name} - Layer {np.shape(attns)[0]}")
        plt.tight_layout()

        save_path = os.path.join(
            run_path, f"attetion_layer={np.shape(attns)[0]}_head={head+4}-{head+7}_length={np.shape(attns)[3]}.png")
        plt.savefig(save_path)
        plt.clf()

    attns = []
    for batch in range(num_samples // batch_size):
        xs = task.sample_xs(
            b_size=batch_size,
            n_points=test_max_points,
            n_dims_truncated=input_dims,
        )
        ys = task.evaluate(xs)
        with torch.no_grad():
            pred = model(xs.to(device), ys.to(device), loss_func=loss_func, output_attentions=True)
            # attn shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
            attns.append(tuple(t.cpu() for t in pred.attentions))

        del xs, ys, pred
        gc.collect()
        torch.cuda.empty_cache()

    attns = np.concatenate(attns, axis=1)

    fig, ax = plt.subplots()
    ax.plot(attns[layer, :, head, -2, ::2].mean(axis=0), label="Head 0")
    ax.plot(attns[layer, :, head + 1, -2, ::2].mean(axis=0), label="Head 1")
    ax.plot(attns[layer, :, head + 2, -2, ::2].mean(axis=0), label="Head 2")
    ax.plot(attns[layer, :, head + 3, -2, ::2].mean(axis=0), label="Head 3")
    ax.legend(fancybox=True, shadow=True)
    ax.figure.set_size_inches(6, 4)
    ax.set(xlabel="Token position", ylabel="Attention weight")
    ax.set_title(f"{config.get('task', {}).get('name', '')} - {model_name} - Layer {np.shape(attns)[0]}")
    plt.tight_layout()

    save_path = os.path.join(
        run_path, "attn", f"attetion_layer={np.shape(attns)[0]}_head={head}-{head+3}_length={np.shape(attns)[3]}.png")
    plt.savefig(save_path)
    plt.clf()

    if np.shape(attns)[2] > 4:
        fig, ax = plt.subplots()
        ax.plot(attns[layer, :, head + 4, -2, ::2].mean(axis=0), label="Head 4")
        ax.plot(attns[layer, :, head + 5, -2, ::2].mean(axis=0), label="Head 5")
        ax.plot(attns[layer, :, head + 6, -2, ::2].mean(axis=0), label="Head 6")
        ax.plot(attns[layer, :, head + 7, -2, ::2].mean(axis=0), label="Head 7")
        ax.legend(fancybox=True, shadow=True)
        ax.figure.set_size_inches(6, 4)
        ax.set(xlabel="Token position", ylabel="Attention weight")
        ax.set_title(f"{config.get('task', {}).get('name', '')} - {model_name} - Layer {np.shape(attns)[0]}")
        plt.tight_layout()

        save_path = os.path.join(
            run_path, f"attetion_layer={np.shape(attns)[0]}_head={head+4}-{head+7}_length={np.shape(attns)[3]}.png")
        plt.savefig(save_path)
        plt.clf()

    layers = range(attns.shape[0])
    # normalized token position
    x = np.arange(test_max_points) / (test_max_points - 1)
    fig, ax = plt.subplots()
    for layer in layers:
        ax.plot(x, attns[layer, :, :, -2, 1::2].mean(axis=0).mean(axis=0), label=f"Layer {layer+1}")

    ax.set(xlabel="Normalized token position", ylabel="Average attention weight")
    ax.set_ylim(0, 0.02)
    ax.figure.set_size_inches(9, 5)
    ax.legend(bbox_to_anchor=(1.23, 1), loc='upper right', fancybox=True, shadow=True)
    ax.set_title(f"{config.get('task', {}).get('name', '')} - {model_name}")
    plt.tight_layout()
    save_path = os.path.join(run_path, f"avg_attention_score")
    plt.savefig(save_path)
    plt.clf()

    attns = []
    for batch in range(num_samples // batch_size):
        xs = task.sample_xs(
            b_size=batch_size,
            n_points=n_points,
            n_dims_truncated=input_dims,
        )
        xs = xs.repeat(1, 4, 1)
        ys = task.evaluate(xs)
        with torch.no_grad():
            pred = model(xs.to(device), ys.to(device), loss_func=loss_func, output_attentions=True)
            # attn shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
            attns.append(tuple(t.cpu() for t in pred.attentions))

        del xs, ys, pred
        gc.collect()
        torch.cuda.empty_cache()

    attns = np.concatenate(attns, axis=1)

    fig, ax = plt.subplots()
    ax.plot(attns[layer, :, head, -2, ::2].mean(axis=0), label="Head 0")
    ax.plot(attns[layer, :, head + 1, -2, ::2].mean(axis=0), label="Head 1")
    ax.plot(attns[layer, :, head + 2, -2, ::2].mean(axis=0), label="Head 2")
    ax.plot(attns[layer, :, head + 3, -2, ::2].mean(axis=0), label="Head 3")
    ax.legend(fancybox=True, shadow=True)
    ax.figure.set_size_inches(6, 4)
    ax.set(xlabel="Token position", ylabel="Attention weight")
    ax.set_title(f"{config.get('task', {}).get('name', '')} - {model_name} - Layer {np.shape(attns)[0]}")
    plt.tight_layout()

    save_path = os.path.join(
        run_path, "attn",
        f"attetion_layer={np.shape(attns)[0]}_head={head}-{head+3}_length={np.shape(attns)[3]}_duplicated-data.png")
    plt.savefig(save_path)
    plt.clf()

    if np.shape(attns)[2] > 4:
        fig, ax = plt.subplots()
        ax.plot(attns[layer, :, head + 4, -2, ::2].mean(axis=0), label="Head 4")
        ax.plot(attns[layer, :, head + 5, -2, ::2].mean(axis=0), label="Head 5")
        ax.plot(attns[layer, :, head + 6, -2, ::2].mean(axis=0), label="Head 6")
        ax.plot(attns[layer, :, head + 7, -2, ::2].mean(axis=0), label="Head 7")
        ax.legend(fancybox=True, shadow=True)
        ax.figure.set_size_inches(6, 4)
        ax.set(xlabel="Token position", ylabel="Attention weight")
        ax.set_title(f"{config.get('task', {}).get('name', '')} - {model_name} - Layer {np.shape(attns)[0]}")
        plt.tight_layout()

        save_path = os.path.join(
            run_path,
            f"attetion_layer={np.shape(attns)[0]}_head={head+4}-{head+7}_length={np.shape(attns)[3]}_duplicated-data.png"
        )
        plt.savefig(save_path)
        plt.clf()


if __name__ == "__main__":
    sns.set_theme('notebook', 'darkgrid')
    palette = sns.color_palette('colorblind')
    fire.Fire(main)
