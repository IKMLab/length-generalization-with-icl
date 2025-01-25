from datasets import Dataset

from .samplers import get_data_sampler, get_task_sampler


def load_dataset(args, curriculum) -> Dataset:
    data_sampler = get_data_sampler(**args)
    task_sampler = get_task_sampler(**args)
    task = task_sampler()

    xs = data_sampler.sample_xs(
        b_size=args["batch_size"],
        n_points=curriculum.n_points,
        n_dims=curriculum.n_dims_truncated,
    )
    ys = task.evaluate(xs)

    def generator():
        for x, y in zip(xs, ys):
            yield {"xs": x, "ys": y, "labels": y}

    return Dataset.from_generator(generator)
