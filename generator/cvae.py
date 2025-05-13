import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(
        self, 
        model, 
        latent_dim=100, 
        noise_shape=None,
        beta=1.0,  # Weight of the KL term (beta-VAE parameter)
        kl_anneal_steps=0,  # Number of steps for KL annealing (0 means no annealing)
        kl_anneal_start=0.0,  # Starting value for KL annealing
    ):
        super(CVAE, self).__init__()
        self.model = model  # model should output mean and logvar
        self.latent_dim = latent_dim
        self.noise_shape = noise_shape
        
        # Training hyperparameters
        self.beta = beta
        self.kl_anneal_steps = kl_anneal_steps
        self.kl_anneal_start = kl_anneal_start
        
        # Training state
        self.register_buffer('current_step', torch.tensor(0))
        
    def get_kl_weight(self):
        """
        Calculate the current KL weight based on annealing schedule
        """
        if self.kl_anneal_steps == 0:
            return self.beta
            
        alpha = (self.current_step.float() / self.kl_anneal_steps).clamp(0.0, 1.0)
        return self.beta * (self.kl_anneal_start + (1 - self.kl_anneal_start) * alpha)
        
    def compute_kl_loss(self, mu, logvar):
        """
        Compute KL divergence loss
        """
        # Compute KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        """
        Forward pass for training. Returns reconstruction loss and KL divergence.
        """
        if isinstance(x, dict):
            x = x['encoder_inputs']
            batch_size, set_size, seq_len, channels = x.shape
            x = x.view(batch_size * set_size, seq_len, channels)
        # Expand condition to match batch size if needed
        c = c.unsqueeze(1).repeat(1, x.shape[0] // c.shape[0], 1).view(-1, c.shape[-1])
        
        # Get mean and logvar from encoder part of the model
        mu, logvar = self.model.encode(x, c)
        
        # Sample latent using reparameterization
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.model.decode(z, c)
        
        # Compute losses
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = self.compute_kl_loss(mu, logvar)
        
        # Apply KL annealing
        kl_weight = self.get_kl_weight()
        
        # Increment step counter
        self.current_step += 1
        
        return recon_loss + kl_weight * kl_loss
    
    def loss(self, x, c):
        return self.forward(x, c)

    def sample(self, context, num_samples, return_texts=False):
        """
        Sample from the CVAE given context vectors.
        """
        device = context.device
        n_sets = context.shape[0]
        
        # Expand context for the number of samples
        context = context.unsqueeze(1).repeat(1, num_samples, 1).view(-1, context.shape[-1])
        
        # Sample from prior
        z = torch.randn(context.shape[0], self.latent_dim).to(device)
        
        # Decode
        with torch.no_grad():
            samples = self.model.decode(z, context)

        if return_texts:
            # this does not work
            import numpy as np
            vocab = np.array(['A', 'C', 'G', 'T', 'N'])
            # sample along last axis to get one sample
            _, seq_len, vocab_size = samples.shape
            samples = torch.argmax(samples, dim=-1).detach().cpu().numpy()
            print("samples shape", samples)
            all_texts = vocab[samples]
            # convert to string
            all_texts = [''.join(text) for text in all_texts]
            # reshape to batch_size, num_samples
            all_texts = np.array(all_texts).reshape(n_sets, num_samples)
            samples = torch.tensor(samples).reshape(n_sets, num_samples, seq_len)
            return samples, all_texts
        
        # Reshape to match the expected output format
        samples = samples.view(n_sets, num_samples, -1)
        
        return samples 