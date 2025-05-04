import numpy as np
import torch
import torch.nn as nn
from layers import SelfAttention
from generator.losses import pairwise_sinkhorn
from geomloss import SamplesLoss

class DistributionDecoderTx(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden_dim, set_size, layers=2, heads=4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SELU(),
            *[SelfAttention(hidden_dim, heads) for _ in range(layers)],
            nn.Linear(hidden_dim, out_dim)
        )
        self.multiplier = nn.Linear(latent_dim, latent_dim*set_size)
        self.set_size = set_size

    def forward(self, latent):
        expanded_latent = self.multiplier(latent).view(latent.shape[0], self.set_size, -1)
        return self.decoder(expanded_latent)

class WormholeGenerator(nn.Module):
    def __init__(self, latent_dim, data_shape, hidden_dim, 
                 set_size, layers=1, heads=4, scaling=0.9):
        super().__init__()
        out_dim = np.prod(data_shape)
        self.model = DistributionDecoderTx(latent_dim, out_dim, hidden_dim, set_size, layers, heads)
        self.sinkhorn = SamplesLoss("sinkhorn", p=2, scaling=scaling)

    def forward(self, latent):
        return self.model(latent)

    def loss(self, x, latent):
        rec = self(latent)
        x = x.reshape(rec.shape)

        # x_min = x.amin(dim=(0,1), keepdim=True)
        # x_max = x.amax(dim=(0,1), keepdim=True)
        # x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1

        # rec_min = rec.amin(dim=(0,1), keepdim=True)
        # rec_max = rec.amax(dim=(0,1), keepdim=True)
        # rec_scaled = 2 * (rec - rec_min) / (rec_max - rec_min) - 1
        
        rec_loss = self.sinkhorn(rec, x)
        input_w2 = pairwise_sinkhorn(x, self.sinkhorn)
        latent_d = torch.cdist(latent, latent)

        return rec_loss.mean() + ((input_w2 - latent_d)**2).mean()/2
    
    def sample(self, latent, num_samples=None):
        return self(latent)