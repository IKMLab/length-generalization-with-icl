import math

import torch
from einops import repeat


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


class ResonanceRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()

        assert dim % 2 == 0, 'dim must be multiple of 2 for Resonance RoPE.'

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (self.base**(torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        r_wavelengths = torch.round(2 * math.pi / inv_freq)
        r_inv_freq = 2 * math.pi / r_wavelengths
        self.register_buffer("r_inv_freq", r_inv_freq, persistent=False)
        self.register_buffer("r_wavelengths", r_wavelengths, persistent=False)

    def compute_freqs(self, position_ids):
        seq_len = torch.max(position_ids) + 1
        r_inv_freq_expanded = self.r_inv_freq[None, :].float().expand(position_ids.shape[0], -1)
        position_ids_expanded = position_ids.float()

        freqs_list = list()
        for i in range(self.dim // 2):
            if seq_len >= self.r_wavelengths[i].item():
                current_freq = repeat(
                    position_ids_expanded[:, :self.r_wavelengths[i].int()] * r_inv_freq_expanded[:, i],
                    'b l -> b (repeat l)',
                    repeat=math.ceil(seq_len / self.r_wavelengths[i].item()),
                )[:, :seq_len]
            else:
                current_freq = position_ids_expanded * r_inv_freq_expanded[:, i]

            freqs_list.append(current_freq)

        return torch.stack(freqs_list, dim=2)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = self.compute_freqs(position_ids)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class ResonanceLinearScalingRotaryEmbedding(ResonanceRotaryEmbedding):

    def forward(self, x, position_ids):
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class ResonanceNTKScalingRotaryEmbedding(ResonanceRotaryEmbedding):

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        base = base * scaling_factor**(dim / (dim - 2))
        super().__init__(dim, max_position_embeddings, base, device, scaling_factor)


class ResonanceYaRNScalingRotaryEmbedding(ResonanceRotaryEmbedding):

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

        super().__init__(dim, max_position_embeddings, base, device, scaling_factor)
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
        r_wavelengths = torch.round(2 * math.pi / inv_freq)
        r_inv_freq = 2 * math.pi / r_wavelengths
        self.register_buffer("r_inv_freq", r_inv_freq, persistent=False)
        self.register_buffer("r_wavelengths", r_wavelengths, persistent=False)
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
