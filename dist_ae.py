import torch
import torch.nn as nn

from layers import MLP, MeanPooledFC, MedianPooledFC

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)

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

class SetAutoencoderHybrid(SetAutoencoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2, heads=4):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            MLP(in_dim, hidden_dim, hidden_dim, fc_layers),
            *[MeanPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.SELU(),
            *[SelfAttention(hidden_dim, heads) for _ in range(layers)],
            nn.Linear(hidden_dim, in_dim)
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

def train_w_stop(dist_ae, optimizer, train_loader, val_loader, loss_fn, max_epochs=100,
                   device='cuda', patience=5, verbose=True):
    dist_ae.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        dist_ae.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            _, rec = dist_ae(batch.to(device))
            loss = loss_fn(rec, batch.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        dist_ae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                _, rec = dist_ae(batch.to(device))
                val_loss += loss_fn(rec, batch.to(device))
        
        val_loss /= len(val_loader)
        if verbose:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered! 🛑")
                break
    
    return dist_ae