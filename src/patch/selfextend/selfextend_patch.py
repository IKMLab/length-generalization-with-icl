import transformers

from .llama_selfextend import self_extend_forward


def apply_selfextend_attn_patch() -> None:
    transformers.models.llama.modeling_llama.LlamaAttention.forward = self_extend_forward
