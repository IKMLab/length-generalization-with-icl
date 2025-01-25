import transformers

from .configuration_llama_fire import rope_scaling_validation
from .modeling_llama_fire import init_rope, forward, sdpa_forward


def apply_fire_patch() -> None:
    transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation = rope_scaling_validation
    transformers.models.llama.modeling_llama.LlamaAttention._init_rope = init_rope
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = sdpa_forward
