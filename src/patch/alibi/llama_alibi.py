import math
import torch


class ALiBi:
    """code from mpt"""

    def __init__(
        self,
        n_heads: int,
        alibi_bias_max: int = 8,
    ):
        self.n_heads = n_heads
        self.alibi_bias_max = alibi_bias_max

    def __call__(self, q_len, device):
        alibi = torch.arange(1 - q_len, 1, dtype=torch.int32, device=device).view(1, 1, 1, q_len)
        num_heads_power_of_2 = 2**math.ceil(math.log2(self.n_heads))

        base = torch.arange(1, num_heads_power_of_2 + 1, dtype=torch.int64, device=device).float()
        base = base * (self.alibi_bias_max / num_heads_power_of_2)

        slopes = 1.0 / torch.pow(2, base)
        slopes = slopes.view(1, num_heads_power_of_2, 1, 1)

        if num_heads_power_of_2 != self.n_heads:
            slopes = torch.concat([slopes[:, 1::2, ...], slopes[:, ::2, ...]], dim=1)[:, :self.n_heads, ...]

        alibi = alibi * slopes
        return alibi.squeeze(0)
