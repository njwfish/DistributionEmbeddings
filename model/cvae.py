import torch
import torch.nn as nn
from layers import MLP

class SimpleVAE(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim, vae_latent_dim, fc_layers=2):
        super(SimpleVAE, self).__init__()
        # Encoder outputs mu and logvar
        self.encoder_hidden = MLP(in_dim + latent_dim, hidden_dim, hidden_dim, fc_layers)
        self.encoder_mu = nn.Linear(hidden_dim, vae_latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, vae_latent_dim)
        
        # Decoder
        self.decoder = MLP(vae_latent_dim + latent_dim, hidden_dim, in_dim, fc_layers)
        
    def encode(self, x, c):
        # Concatenate input and context
        h = torch.cat([x, c], dim=-1)
        h = self.encoder_hidden(h)
        
        # Get mu and logvar
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        
        return mu, logvar
    
    def decode(self, z, c):
        # Concatenate latent and context
        h = torch.cat([z, c], dim=-1)
        return self.decoder(h)

        
        
        
        
        