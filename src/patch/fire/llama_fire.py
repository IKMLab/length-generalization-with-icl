import torch


class FIRE(torch.nn.Module):
    """
    https://github.com/mcleish7/arithmetic/blob/86022a57d38c0fde46444d62e8dcbebcc0af614c/cramming/architectures/embeddings.py#L419
    """

    def __init__(self, num_heads=12, mlp_width=32, init_c=0.1, init_L=512., eps=1e-6, max_length=0):
        """
        FIRE attention bias module (https://arxiv.org/abs/2310.04418).

        Args:
            num_heads: number of attention heads.
            mlp_width: Width of MLP.
            init_c: initial value of log transformation parameter
            init_L: initial value of thresholding parameter
            eps: small constant for numerical stability
        """
        super(FIRE, self).__init__()
        self.max_length = max_length  # using random PE

        # Define the MLP layers
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, mlp_width),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_width, num_heads),
        )

        # Initialize c (log transformation parameter)
        self.c = torch.nn.Parameter(torch.tensor(init_c))

        # Initialize L (threshold)
        self.init_L = torch.nn.Parameter(torch.tensor(init_L), requires_grad=False)
        self.L_multiplier = torch.nn.Parameter(torch.tensor(1.0))  # learn a multiplier to L

        self.eps = eps

    def forward(self, x):
        """
        Compute FIRE attention bias (https://arxiv.org/abs/2310.04418).

        Args:
            x: input sequence, shape [bsz, seq_len, hidden_dim]

        Returns:
            attention bias of shape [1, num_heads, seq_len, seq_len]
        """
        seq_length = x.size(1)
        # max length will be increased to max sequnece length if max length is short
        if (seq_length > self.max_length) or (not self.training):
            max_length = seq_length
        else:
            max_length = self.max_length

        # take a subset (of length seq_length) of a random permutation of length max_length, then sort it to
        positions = torch.sort(torch.randperm(max_length, dtype=torch.float, device=x.device)[:seq_length]).values
        relative_distances = positions[:, None] - positions[None, :]

        # Thresholding the normalizer for short sequence modeling
        threshold = torch.abs(self.L_multiplier * self.init_L)
        position_normalizer = torch.max(positions, threshold)
        position_normalizer = position_normalizer[:, None]

        # Amplifying differences among local positions with log transform
        relative_distances = torch.log(torch.abs(self.c * relative_distances) + 1)
        position_normalizer = torch.log(torch.abs(self.c * position_normalizer) + 1)

        # Progressive interpolation
        normalized_distances = relative_distances / (position_normalizer + self.eps)
        fire_bias = self.mlp(normalized_distances.unsqueeze(-1)).unsqueeze(0)
        fire_bias = fire_bias.permute(0, 3, 1, 2)

        return fire_bias
