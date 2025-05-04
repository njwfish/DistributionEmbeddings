import torch
from loss.utils import to_device

def pert_pred_loss_fn(encoder, curr_latent, next_samples):
    next_latent = encoder(next_samples)
    delta_latent_pred = encoder.next_predictor(curr_latent)

    latent_delta = next_latent - curr_latent

    diff = latent_delta - delta_latent_pred
    loss = torch.mean(torch.square(diff))
    return loss

class OrderedLossManager:
    def __init__(self, mask_context_prob=0.0, next_latent_loss_weight=1.0):
        self.mask_context_prob = mask_context_prob
        self.next_latent_loss_weight = next_latent_loss_weight

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

        if self.next_latent_loss_weight > 0:
            next_samples = to_device(batch, device, key='next_set_samples')
            loss += self.next_latent_loss_weight * pert_pred_loss_fn(encoder, latent, next_samples)
            losses['next_latent_loss'] = self.next_latent_loss_weight * pert_pred_loss_fn(encoder, latent, next_samples)

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        return loss, losses