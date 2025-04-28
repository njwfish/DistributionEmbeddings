import torch
import torch.nn as nn
import numpy as np

class KMEEncoder(nn.Module):
    def __init__(self, data_shape, gamma=0.1, d=32, seed=None, device='cuda'):
        super().__init__()
        self.gamma = gamma
        self.d = d//2
        self.seed = seed
        self.N_dims = np.prod(data_shape)
        self.scale = 1/self.gamma
        self.W = torch.randn(self.N_dims, self.d) * self.scale
        self.W = self.W.to(device)

    def forward(self, x):
        # x: (batch, set_size, N_dims)

        x = x.flatten(start_dim=2) # for mnist for now

        batch, set_size, N_dims = x.shape

        XW = torch.matmul(x, self.W)  # (batch, set_size, d)
        cos = torch.cos(XW)
        sin = torch.sin(XW)
        phi = torch.cat([cos, sin], dim=-1)  # (batch, set_size, 2d)

        # mean across the set
        return phi.mean(dim=1)  # (batch, 2d)
