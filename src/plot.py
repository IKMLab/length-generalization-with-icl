import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


def basic_plot(metrics, models=None, task="gaussian", trivial=1.0, n_points=100):
    fig, ax = plt.subplots(1, 1)

    if models is not None:
        metrics = {k.name: metrics[k.name] for k in models}

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    plt.axvline(n_points, ls="-", color="red")
    plt.annotate("Bound", xy=(n_points + 5, 20), color="r", rotation=0)
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=rename(name), color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    ax.set_xlabel("in-context examples")

    if task == "gaussian":

        ax.set_ylabel("squared error")
        ax.set_xlim(-1, len(low) + 0.1)
        ax.set_ylim(-0.1, 30)
        legend = ax.legend(loc="upper right")

    elif task == "boolean":

        ax.set_ylabel("accuracy")
        ax.set_xlim(-1, len(low) + 0.1)
        ax.set_ylim(-0.1, 1.1)
        legend = ax.legend(loc="lower right")

    fig.set_size_inches(6, 5)
    for line in legend.get_lines():
        line.set_linewidth(4)

    return fig, ax


def rename(name):
    if "OLS" in name:
        return "Least Squares"
    elif "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    elif "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return "Lasso"
    elif "fire" in name:
        return "FIRE"
    elif "yarn" in name:
        name = name.replace("dynamic-", "Dynamic ")
        return name.replace("yarn", "YaRN")
    elif "nope" in name:
        return "NoPE"
    elif "linear" in name:
        return "PI"
    elif "dynamic" in name:
        return "Dynamic NTK"
    elif "alibi" in name:
        return "ALiBi"
    elif "RoPE" in name:
        return name
    else:
        return name.capitalize()
