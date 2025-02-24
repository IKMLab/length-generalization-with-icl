# A modified version of jquesnelle/yarn for transformers version 4.40.0
import math

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations,
    dim,
    base=10000,
    max_position_embeddings=2048,
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot,
    high_rot,
    dim,
    base=10000,
    max_position_embeddings=2048,
):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class LlamaYaRNScalingRotaryEmbedding(LlamaRotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        extrapolation_factor=1,
        attn_factor=1,
        beta_fast=32,
        beta_slow=1,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__(
            dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
            scaling_factor=scaling_factor,
        )

        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        pos_freqs = self.base**(torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float()) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq)
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * self.attn_factor)

    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.scaling_factor * self.max_position_embeddings:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        cos, sin = super().forward(x, position_ids)
        return cos * self.mscale, sin * self.mscale
