import torch

from transformers import (
    GPT2Config,
    LlamaConfig,
    MambaConfig,
)

from .modeling.baselines import (
    LeastSquaresModel,
    NNModel,
    AveragingModel,
    LassoModel,
    NullClassifier,
)
from .modeling.decoder import (
    GPT2ForTokenClassification,
    GPT2ForInContextLearning,
    LlamaForTokenizeICL,
    LlamaForInContextLearning,
)
from .modeling.ssm import MambaForInContextLearning


def build_model(config, **kwargs) -> torch.nn.Module:
    if config["family"] == "GPT2":

        config["n_in_dims"] = kwargs["n_in_dims"]
        config["n_positions"] = kwargs["max_length"]

        if kwargs.get("patch"):
            for key in kwargs["patch"]:
                config[key] = kwargs["patch"][key]

        cfg = GPT2Config(**config)
        name2class = {
            "token_classification": GPT2ForTokenClassification,
            "linear_regression": GPT2ForInContextLearning,
            "sparse_linear_regression": GPT2ForInContextLearning,
            "linear_classification": GPT2ForInContextLearning,
            "noisy_linear_regression": GPT2ForInContextLearning,
            "quadratic_regression": GPT2ForInContextLearning,
            "relu_2nn_regression": GPT2ForInContextLearning,
            "decision_tree": GPT2ForInContextLearning,
            "conjunction": GPT2ForInContextLearning,
            'teach_biconjunction': GPT2ForInContextLearning,
            "mono_conjunction": GPT2ForInContextLearning,
            "teach_conjunction": GPT2ForInContextLearning,
            "disjunction": GPT2ForInContextLearning,
            "sparse_disjunction": GPT2ForInContextLearning,
            "nearest_neighbours": GPT2ForInContextLearning,
            "parity": GPT2ForInContextLearning,
            "sparse_parity": GPT2ForInContextLearning,
            "majority": GPT2ForInContextLearning,
            "int_halfspace": GPT2ForInContextLearning,
            "dnf": GPT2ForInContextLearning,
            "teach_dnf": GPT2ForInContextLearning,
            "cnf": GPT2ForInContextLearning,
            'sparse_thres': GPT2ForInContextLearning,
        }

    elif config["family"] == "Llama2":

        config["n_in_dims"] = kwargs["n_in_dims"]
        config["max_position_embeddings"] = kwargs["max_length"]
        config["intermediate_size"] = config["hidden_size"] * 4

        if kwargs.get("patch"):
            for key in kwargs["patch"]:
                config[key] = kwargs["patch"][key]

        if config.get("rope_scaling"):
            config["rope_scaling"]["factor"] = config["rope_scaling"].get(
                "factor",
                kwargs["max_length"] / kwargs["max_points"] / 2 * 8,
            )

        cfg = LlamaConfig(**config)
        name2class = {
            "token_icl": LlamaForTokenizeICL,
            "linear_regression": LlamaForInContextLearning,
            "sparse_linear_regression": LlamaForInContextLearning,
            "linear_classification": LlamaForInContextLearning,
            "noisy_linear_regression": LlamaForInContextLearning,
            "quadratic_regression": LlamaForInContextLearning,
            "relu_2nn_regression": LlamaForInContextLearning,
            "decision_tree": LlamaForInContextLearning,
            "conjunction": LlamaForInContextLearning,
            'teach_biconjunction': LlamaForInContextLearning,
            "mono_conjunction": LlamaForInContextLearning,
            "teach_conjunction": LlamaForInContextLearning,
            "disjunction": LlamaForInContextLearning,
            "sparse_disjunction": LlamaForInContextLearning,
            "nearest_neighbours": LlamaForInContextLearning,
            "parity": LlamaForInContextLearning,
            "sparse_parity": LlamaForInContextLearning,
            "majority": LlamaForInContextLearning,
            "int_halfspace": LlamaForInContextLearning,
            "dnf": LlamaForInContextLearning,
            "teach_dnf": LlamaForInContextLearning,
            "cnf": LlamaForInContextLearning,
            'sparse_thres': LlamaForInContextLearning,
        }

    elif config["family"] == "Mamba":

        config["n_in_dims"] = kwargs["n_in_dims"]
        cfg = MambaConfig(**config)
        name2class = {
            "linear_regression": MambaForInContextLearning,
            "sparse_linear_regression": MambaForInContextLearning,
            "linear_classification": MambaForInContextLearning,
            "noisy_linear_regression": MambaForInContextLearning,
            "quadratic_regression": MambaForInContextLearning,
            "relu_2nn_regression": MambaForInContextLearning,
            "decision_tree": MambaForInContextLearning,
            "conjunction": MambaForInContextLearning,
            'teach_biconjunction': MambaForInContextLearning,
            "mono_conjunction": MambaForInContextLearning,
            "teach_conjunction": MambaForInContextLearning,
            "disjunction": MambaForInContextLearning,
            "sparse_disjunction": MambaForInContextLearning,
            "nearest_neighbours": MambaForInContextLearning,
            "parity": MambaForInContextLearning,
            "sparse_parity": MambaForInContextLearning,
            "majority": MambaForInContextLearning,
            "int_halfspace": MambaForInContextLearning,
            "dnf": MambaForInContextLearning,
            "teach_dnf": MambaForInContextLearning,
            "cnf": MambaForInContextLearning,
            'sparse_thres': MambaForInContextLearning,
        }

    else:
        raise ValueError(f"Model {config['family']} not supported")

    if kwargs["task"] in name2class:
        model = name2class[kwargs["task"]](cfg)
    else:
        raise ValueError(f"Task {kwargs['task']} not supported")

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
        ],
        "quadratic_regression": [],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (LassoModel, {
                "alpha": 0.01
            }),
        ],
        "noisy_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
        ],
        "cnf": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        "conjunction": [
            (NNModel, {
                "n_neighbors": 1
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        "disjunction": [
            (NNModel, {
                "n_neighbors": 1
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        "dnf": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        "int_halfspace": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        "majority": [
            (NNModel, {
                "n_neighbors": 1
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        # "nearest_neighbours": [
        #     (NNModel, {
        #         "n_neighbors": 1
        #     }),
        #     (AveragingModel, {}),
        #     (NullClassifier, {}),
        # ],
        "parity": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        "sparse_disjunction": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        "sparse_parity": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
        "sparse_thres": [
            (NNModel, {
                "n_neighbors": 3
            }),
            (AveragingModel, {}),
            (NullClassifier, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models
