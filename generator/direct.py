import torch
import torch.nn as nn
from generator.losses import sliced_wasserstein_distance, mmd, sinkhorn

class DirectGenerator(nn.Module):
    def __init__(self, model, loss_type='swd', loss_params=None, noise_dim=100):
        super().__init__()
        self.noise_dim = noise_dim
        self.model = model
        self.loss_type = loss_type
        self.loss_params = loss_params

        def loss_fn(x, y):
            if self.loss_type == 'swd':
                return torch.vmap(sliced_wasserstein_distance, randomness='different')(x, y, **self.loss_params).mean()
            elif self.loss_type == 'mmd':
                return torch.vmap(mmd, randomness='different')(x, y, **self.loss_params).mean()
            elif self.loss_type == 'sinkhorn':
                return torch.vmap(sinkhorn, randomness='different')(x, y, **self.loss_params).mean()
            
        self.loss_fn = loss_fn

    def forward(self, latent, num_samples=1):
        eps = torch.randn(latent.shape[0], num_samples, self.noise_dim).to(latent.device)
        latent = latent.unsqueeze(1).repeat(1, num_samples, 1)
        return self.model(latent, eps)

    def loss(self, x, latent):
        samples_per_set = x.shape[0] // latent.shape[0]
        recon = self(latent, num_samples=samples_per_set)
        x = x.reshape(recon.shape)
        return self.loss_fn(recon, x)
    
    def sample(self, latent, num_samples):
        return self(latent, num_samples)
    
    