import torch

def classification_loss_fn(encoder, latent, classes):
    latent_pred = encoder.classifier(latent)
    loss = torch.nn.functional.cross_entropy(latent_pred, classes)
    return loss

    
class ClassificationLossManager:
    def __init__(self, classification_loss_weight=1.0, mask_context_prob=0.0):
        self.classification_loss_weight = classification_loss_weight
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

            # get random subset of set indices 
            set_indices = torch.randperm(samples['hyena_input_ids'].shape[1])[:1000]
            samples_for_reconstruction = {
                'hyena_input_ids': samples['hyena_input_ids'][:, set_indices],
                'hyena_attention_mask': samples['hyena_attention_mask'][:, set_indices],
            }
            recon_loss = generator.loss(samples_for_reconstruction, latent)
        
        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss

        if self.classification_loss_weight > 0:
            classes = batch['classes'].to(device).mean(dim=1)
            classification_loss = classification_loss_fn(
                encoder, latent, classes
            )
            loss += self.classification_loss_weight * classification_loss
            losses['classification_loss'] = classification_loss

        return loss, losses