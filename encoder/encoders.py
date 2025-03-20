import torch
import torch.nn as nn

from layers import MLP, MeanPooledFC, MedianPooledFC, SelfAttention

class DistributionEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size):
        super().__init__()
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        # self.positional_embedding = nn.Embedding(set_size, latent_dim)

        self.latent_act = nn.SELU()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        enc = self.encoder(x)
        # generate compressed latent by mean pooling
        enc_mean = torch.mean(enc, dim=1)
        # enc_mean = torch.median(enc, dim=1).values
        lat = self.latent_act(self.latent_proj(enc_mean))
        return lat

class DistributionEncoderTx(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, heads=4):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            *[SelfAttention(hidden_dim, heads) for _ in range(layers)]
        )

class DistributionEncoderGNN(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            MLP(in_dim, hidden_dim, hidden_dim, fc_layers),
            *[MeanPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)]
        )

class DistributionEncoderHybrid(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2, heads=4):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            MLP(in_dim, hidden_dim, hidden_dim, fc_layers),
            *[MeanPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)]
        )

class DistributionEncoderMedianGNN(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            MLP(in_dim, hidden_dim, hidden_dim, fc_layers),
            *[MedianPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)]
        )
