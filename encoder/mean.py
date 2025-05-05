import torch
import torch.nn as nn
import numpy as np

class MeanEncoder(nn.Module):
    def __init__(self, data_shape, d=32, seed=None):
        super().__init__()
        self.seed = seed
        self.N_dims = np.prod(data_shape)
        self.proj = torch.nn.Parameter(torch.randn(self.N_dims, self.d))

    def forward(self, x):
        # x: (batch, set_size, N_dims)
        x = x.flatten(start_dim=2) # for mnist for now
        x = x.mean(dim=2)
        x = torch.matmul(x, self.proj)
        return x
