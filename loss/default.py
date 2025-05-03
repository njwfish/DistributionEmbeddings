import torch

class LossManager:
    def __init__(self, mask_context_prob=0.0):
        self.mask_context_prob = mask_context_prob

    def loss(self, encoder, generator, batch, device):
        losses = {}
        loss = 0

        if isinstance(batch['samples'], torch.Tensor):
            samples = batch['samples'].to(device)
            latent = encoder(samples)  # latent is num samples x num sets x latent dim
            
            if self.mask_context_prob > 0:
                context_mask = torch.bernoulli(torch.zeros(latent.shape[0])+self.mask_context_prob).to(latent.device)
                latent = latent * context_mask[:, None]
            recon_loss = generator.loss(samples.view(-1, *samples.shape[2:]), latent)
        else:
            # For dictionary samples (like PubMed dataset), move tensors to device
            samples = {}
            for key, value in batch['samples'].items():
                if isinstance(value, torch.Tensor):
                    samples[key] = value.to(device)
                else:
                    samples[key] = value

            # Encode samples to latent space
            latent = encoder(samples)
            if self.mask_context_prob > 0:
                context_mask = torch.bernoulli(torch.zeros(latent.shape[0])+self.mask_context_prob).to(latent.device)
                latent = latent * context_mask

            recon_loss = generator.loss(samples, latent)

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        return loss, losses