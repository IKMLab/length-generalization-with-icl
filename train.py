import datetime
import json
import os

import fire
import torch
import matplotlib.pyplot as plt
import wandb
import yaml
from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler

from src.config_loader import Loader
from src.curriculum import Curriculum
from src.eval import compute_evals, get_model_from_run
from src.patch import Patch
from src.plot import basic_plot
from src.samplers import get_task_sampler, sample_transformation
from src.utils import build_model, get_relevant_baselines


class Trainer:

    training: bool = False

    def __init__(
        self,
        config: dict,
        save_path: str = None,
        *args,
        **kwargs,
    ) -> None:

        self.config = config

        if config.get("patch"):
            self.patch = Patch(config["patch"]["name"])
            self.patch.apply()

        self.curriculum = Curriculum(config["task"]["curriculum"])
        self.input_dims = config["task"]["curriculum"]["dims"]["end"]
        self.batch_size = config["training"]["per_device_train_batch_size"]
        self.task_sampler = get_task_sampler(
            n_dims=self.input_dims,
            batch_size=self.batch_size,
            **config["task"],
        )

        if isinstance(config["max_length"], int):
            self.max_length = config["max_length"]
        else:
            self.max_length = float(config["max_length"])

        self.model_args = {
            "task": self.config["task"]["name"],
            "max_length": self.max_length,
            "n_in_dims": self.input_dims,
            "max_points": config["task"]["curriculum"]["points"]["end"],
            "patch": self.config.get("patch"),
        }

        if save_path is not None:
            self.save_path = save_path
            self.model, _ = get_model_from_run(self.save_path, **self.model_args)
        else:
            self.save_path = os.path.join(
                "./models",
                self.config["task"]["name"],
                f"dim_{self.config['task']['curriculum']['dims']['start']}-{self.config['task']['curriculum']['dims']['end']}",
                datetime.datetime.now().strftime("%Y%m%d"),
                self.config["wandb_id"],
            )

            self.model = build_model(self.config["model"], **self.model_args)
            self.model.cuda()

        if config["model"]["type"] == "decoder":
            self.model.name = "RoPE"
            if config["model"].get("rope_scaling"):
                self.model.name = self.config["model"]["rope_scaling"]["type"]
        elif config["model"]["type"] == "ssm":
            self.model.name = self.config["model"]["family"]
        else:
            raise ValueError(f"Model type {config['model']['type']} not recognized")

        self.optimizer = AdamW(self.model.parameters(), lr=config["training"]["learning_rate"])
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=config["training"]["max_steps"] * config["training"]["warmup_ratio"],
            num_training_steps=config["training"]["max_steps"],
        )

    def train(self) -> None:

        self.training = True

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        state_path = os.path.join(self.save_path, "state.pt")

        starting_step = 0
        pbar = tqdm(range(starting_step, self.config["training"]["max_steps"]))
        for step in range(self.config["training"]["max_steps"]):
            self.model.train()

            task = self.task_sampler()
            loss_func = task.get_training_metric()
            xs = task.sample_xs(
                self.curriculum.n_points,
                self.batch_size,
                self.curriculum.n_dims_truncated,
            )
            ys = task.evaluate(xs)

            output = self.model(xs.cuda(), ys.cuda(), loss_func=loss_func)
            loss = output.loss
            loss.backward()

            point_wise_tags = list(range(self.curriculum.n_points))
            point_wise_loss_func = task.get_metric()
            point_wise_loss = point_wise_loss_func(output.logits, ys.cuda()).mean(dim=0)
            point_wise_loss = point_wise_loss.cpu().detach().numpy()
            baseline_loss = (
                sum(max(self.curriculum.n_dims_truncated - ii, 0) for ii in range(self.curriculum.n_points)) /
                self.curriculum.n_points)

            if step % self.config["training"]["logging_steps"] == 0:
                wandb.log(
                    {
                        "overall_loss": loss.item(),
                        "excess_loss": loss.item() / baseline_loss,
                        "point_wise/loss": dict(zip(point_wise_tags, point_wise_loss)),
                        "n_points": self.curriculum.n_points,
                        "n_dims": self.curriculum.n_dims_truncated,
                    },
                    step=step,
                )

            self.curriculum.update()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            pbar.update(1)

            pbar.set_description(f"loss {loss}")
            if step % self.config["training"]["save_steps"] == 0:
                training_state = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_step": step,
                }
                torch.save(training_state, state_path)

            config_path = os.path.join(self.save_path, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(self.config, f)

    def evaluate(self) -> None:

        metric_path = os.path.join(self.save_path, f"metrics.json")
        state_path = os.path.join(self.save_path, "state.pt")

        recompute = True
        if os.path.exists(metric_path):
            checkpoint_created = os.path.getmtime(state_path)
            cache_created = os.path.getmtime(metric_path)
            if checkpoint_created < cache_created:
                recompute = False

        self.model.cuda().eval()
        models = [self.model]
        models += get_relevant_baselines(self.config["task"]["name"])
        if not recompute:
            with open(metric_path, "r") as f:
                metrics = json.load(f)
        else:
            eval_methods = {
                "standard": {
                    "prompting_strategy": "standard"
                },
                "ood_length": {
                    "prompting_strategy": "ood_length",
                    "ood_size": 8
                },
            }

            base_kwargs = {
                "task": self.config["task"]["name"],
                "data": self.config["task"]["data"],
                "input_dims": self.input_dims,
                "n_points": self.config["task"]["curriculum"]["points"]["end"],
                "prompting_strategy": "ood_length",
                "ood_size": 8
            }

            if self.config["task"]["data"] == "gaussian":
                for method in ["half_subspace", "skewed"]:
                    if "subspace" in method:
                        eigenvals = torch.zeros(self.input_dims)
                        eigenvals[:self.input_dims // 2] = 1
                    else:
                        eigenvals = 1 / (torch.arange(self.input_dims) + 1)

                    scale = sample_transformation(eigenvals, normalize=True)
                    eval_methods[f"{method}"] = {
                        "data_sampler_kwargs": {
                            "scale": scale
                        },
                    }

                for dim in ["x", "y"]:
                    for scale in [0.333, 0.5, 2, 3]:
                        if dim == "x":
                            eigenvals = scale * torch.ones(self.input_dims)
                            t = sample_transformation(eigenvals)
                            scaling_args = {"task_sampler_kwargs": {"scale": t}}
                        else:
                            scaling_args = {"task_sampler_kwargs": {"scale": scale}}

                        eval_methods[f"scale-{dim}={scale}"] = scaling_args

            for name, kwargs in eval_methods.items():
                # allow kwargs to override base_kwargs values
                eval_methods[name] = base_kwargs.copy()
                eval_methods[name].update(kwargs)

            metrics = compute_evals(
                models,
                eval_methods,
                metric_path,
            )

        for metric_name, log in metrics.items():
            basic_plot(
                log,
                models,
                task=self.config["task"]["data"],
                trivial=self.input_dims,
                n_points=self.config["task"]["curriculum"]["points"]["end"],
            )
            plt.title(
                f"{self.config['task']['name'].capitalize()} - n_dim {self.input_dims}\n max_data_points {self.config['task']['curriculum']['points']['end']} - max_model_points {self.max_length / 2}"
            )
            plt.savefig(os.path.join(self.save_path, f"{metric_name}.png"))
            wandb.log({metric_name: wandb.Image(plt)})
            plt.clf()


def main(
    config_path: str = None,
    eval_only: bool = False,
) -> None:
    # Solution sourced from https://stackoverflow.com/a/9577670
    # See https://gist.github.com/joshbode/569627ced3076931b02f for python3 implementation
    Loader.add_constructor("!include", Loader.construct_include)

    if not eval_only:

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader)

        for length in config["model"]["max_length"]:
            patch_name = config["patch"]["name"] if config.get("patch") else "default"

            config["max_length"] = length
            wandb.init(
                **config["wandb"],
                config=config,
                tags=[
                    config["task"]["name"],
                    config["model"]["family"],
                    patch_name,
                ],
                name=f"{config['task']['name']}_{config['model']['family']}_{patch_name}",
            )

            config["wandb_id"] = wandb.run.id
            trainer = Trainer(config=config)
            trainer.train()
            trainer.evaluate()

    else:

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader)

        if config.get("wandb_id"):
            wandb_id = config["wandb_id"]
        else:
            wandb_id = config_path.split("/")[-2]

        wandb.init(
            **config["wandb"],
            id=wandb_id,
            resume="must",
        )
        trainer = Trainer(config=config, save_path="/".join(config_path.split("/")[:-1]))
        trainer.evaluate()

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
