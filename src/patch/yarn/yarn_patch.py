import transformers

from .configuration_llama_yarn import rope_scaling_validation
from .modeling_llama_yarn import init_rope


def apply_yarn_rope_patch() -> None:
    transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation = rope_scaling_validation
    transformers.models.llama.modeling_llama.LlamaAttention._init_rope = init_rope
