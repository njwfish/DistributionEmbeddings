import torch
import torch.nn as nn
import numpy as np

class KMEEncoder(nn.Module):
    def __init__(self, gamma=0.1, d=32, seed=None, device='cuda'):
        super().__init__()
        self.gamma = gamma
        self.d = d//2
        self.seed = seed  # optional! for reproducibility

    def forward(self, x):
        # x: (batch, set_size, N_dims)

        x = x.flatten(start_dim=2) # for mnist for now

        batch, set_size, N_dims = x.shape

        # RFF 
        if self.seed is not None:
            torch.manual_seed(self.seed)
        scale = 1.0 / self.gamma
        W = torch.randn(N_dims, self.d, device=x.device) * scale  # (N_dims, d)

        XW = torch.matmul(x, W)  # (batch, set_size, d)
        cos = torch.cos(XW)
        sin = torch.sin(XW)
        phi = torch.cat([cos, sin], dim=-1)  # (batch, set_size, 2d)

        # mean across the set
        return phi.mean(dim=1)  # (batch, 2d)
