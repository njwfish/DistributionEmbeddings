import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)

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

class SetAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size):
        super().__init__()
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        # self.positional_embedding = nn.Embedding(set_size, latent_dim)

        self.latent_act = nn.SELU()
        self.encoder = None
        self.decoder = None

    def set_encoder(self, x):
        enc = self.encoder(x)
        # generate compressed latent by mean pooling
        enc_mean = torch.mean(enc, dim=1)
        # enc_mean = torch.median(enc, dim=1).values
        lat = self.latent_act(self.latent_proj(enc_mean))
        return lat
    
    def forward(self, x):
        lat = self.set_encoder(x)

        # sample gaussian noise
        pos = torch.randn(x.shape[0], x.shape[1], self.latent_dim, device=x.device)

        # reshape latent to each element of each set
        lat_rep = lat.unsqueeze(1).repeat(1, x.shape[1], 1)

        # combine pos and latent 
        lat_pos = torch.cat([lat_rep, pos], dim=2)

        rec = self.decoder(lat_pos)

        return lat, rec

class SetAutoencoderTx(SetAutoencoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, heads=4):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            *[SelfAttention(hidden_dim, heads) for _ in range(layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.SELU(),
            *[SelfAttention(hidden_dim, heads) for _ in range(layers)],
            nn.Linear(hidden_dim, in_dim)
        )

class SetAutoencoderGNN(SetAutoencoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            MLP(in_dim, hidden_dim, hidden_dim, fc_layers),
            *[MeanPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)]
        )
        self.decoder = nn.Sequential(
            MLP(2 * latent_dim, hidden_dim, hidden_dim, fc_layers),
            *[MeanPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)],
            MLP(hidden_dim, hidden_dim, in_dim, fc_layers)
        )

class SetAutoencoderMedianGNN(SetAutoencoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            MLP(in_dim, hidden_dim, hidden_dim, fc_layers),
            *[MedianPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)]
        )
        self.decoder = nn.Sequential(
            MLP(2 * latent_dim, hidden_dim, hidden_dim, fc_layers),
            *[MedianPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)],
            MLP(hidden_dim, hidden_dim, in_dim, fc_layers)
        )

def train_dist_ae(
        dist_ae, optimizer, train_loader, loss_fn, 
        n_epochs = 100, device = 'cpu'
):
    """
    Train the distance autoencoder.

    dist_ae: SetAutoencoder
    optimizer: torch.optim.Optimizer
    train_loader: torch.utils.data.DataLoader
    loss_fn: a distributional loss function, e.g. sliced wasserstein distance, mmd, sinkhorn
    n_epochs: int
    device: str
    """
    dist_ae = dist_ae.train()
    dist_ae = dist_ae.to(device)
    
    for epoch in range(n_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            _, rec = dist_ae(batch)
            loss = loss_fn(rec, batch)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return dist_ae