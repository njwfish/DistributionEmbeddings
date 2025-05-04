import torch
import torch.nn as nn
import numpy as np

from layers import MLP, MeanPooledFC, MedianPooledFC, SelfAttention

class WormholeEncoder(nn.Module):
    def __init__(self, data_shape, latent_dim, hidden_dim, set_size, layers=2, heads=4):
        super().__init__()

        in_dim = np.prod(data_shape)

        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        # self.positional_embedding = nn.Embedding(set_size, latent_dim)

        self.latent_act = nn.SELU()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            *[SelfAttention(hidden_dim, heads) for _ in range(layers)]
        )

    def forward(self, x):
        enc = self.encoder(x.flatten(start_dim=2))
        # generate compressed latent by mean pooling
        enc_mean = torch.mean(enc, dim=1)
        lat = self.latent_act(self.latent_proj(enc_mean))
        return lat