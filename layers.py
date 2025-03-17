import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.SELU()
        ])
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.SELU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MeanPooledFC(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.fc = MLP(2 * in_dim, hidden_dim, out_dim, layers)

    def forward(self, x):
        pooled_rep = x.mean(dim=1)
        x = torch.cat([x, pooled_rep.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=2)
        return self.fc(x)
    
class MedianPooledFC(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.fc = MLP(2 * in_dim, hidden_dim, out_dim, layers)

    def forward(self, x):
        pooled_rep = torch.median(x, dim=1).values
        x = torch.cat([x, pooled_rep.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=2)
        return self.fc(x)