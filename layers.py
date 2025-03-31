import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim, layers=2):
        super().__init__()
        if isinstance(in_dims, int):
            in_dims = [in_dims]
        
        self.in_dims = in_dims
        total_in_dim = sum(in_dims)
        
        self.layers = nn.ModuleList([
            nn.Linear(total_in_dim, hidden_dim),
            nn.SELU()
        ])
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.SELU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, *inputs): 
        if isinstance(self.in_dims, int):
            inputs = [inputs]
        
        assert len(inputs) == len(self.in_dims), f"Expected {len(self.in_dims)} inputs, got {len(inputs)}"
        
        # check input dimensions
        for i, (x, d) in enumerate(zip(inputs, self.in_dims)):
            assert x.shape[-1] == d, f"Input {i} has wrong dimension: expected {d}, got {x.shape[-1]}"
        
        x = torch.cat(inputs, dim=-1) 
        
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
    
class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)