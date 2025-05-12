import torch
import torch.nn as nn
import numpy as np

class KMEEncoder(nn.Module):
    def __init__(self, data_shape, gamma=1., d=32, seed=None, train_W=False):
        super().__init__()
        self.gamma = gamma
        self.d = d//2
        self.seed = seed
        self.N_dims = np.prod(data_shape)
        self.scale = torch.nn.Parameter(torch.tensor(1/self.gamma))
        self.W = torch.nn.Parameter(torch.randn(self.N_dims, self.d), requires_grad=train_W)

    def forward(self, x):
        # x: (batch, set_size, N_dims)

        x = x.flatten(start_dim=2) # for mnist for now

        batch, set_size, N_dims = x.shape

        XW = torch.matmul(x, self.scale * self.W)  # (batch, set_size, d)
        cos = torch.cos(XW)
        sin = torch.sin(XW)
        phi = torch.cat([cos, sin], dim=-1)  # (batch, set_size, 2d)

        # mean across the set
        return phi.mean(dim=1)  # (batch, 2d)
