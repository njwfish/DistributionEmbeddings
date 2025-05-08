import torch

def pert_pred_loss_fn(encoder, pert_latent, pert_embedding):
    pert_latent_pred = encoder.pert_predictor(pert_embedding)
    diff = pert_latent - pert_latent_pred
    loss = torch.mean(torch.square(diff))
    return loss


class PerturbSpatialLossManager:
    def __init__(self, pert_pred_loss_weight=1.0, mask_context_prob=0.0):
        self.pert_pred_loss_weight = pert_pred_loss_weight
        self.mask_context_prob = mask_context_prob
    
    def loss(self, encoder, generator, batch, device):
        losses = {}
        loss = 0
        samples = batch['samples'].to(device)

        latent = encoder(samples)  # latent is num samples x num sets x latent dim
        recon_loss = generator.loss(samples.view(-1, *samples.shape[2:]), latent)
        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss

        if self.pert_pred_loss_weight > 0:
            
            pert_set_idx = ~batch['is_control'].to(device)

            if pert_set_idx.sum() != 0:

                pert_embedding = batch['pert_embedding'].to(device)

                pert_latent = latent[pert_set_idx]
                pert_embedding = pert_embedding[pert_set_idx]

                pert_pred_loss = pert_pred_loss_fn(
                    encoder, pert_latent, pert_embedding
                )
                loss += self.pert_pred_loss_weight * pert_pred_loss
                losses['pert_pred_loss'] = pert_pred_loss
            else:
                losses['pert_pred_loss'] = 0.0

        return loss, losses