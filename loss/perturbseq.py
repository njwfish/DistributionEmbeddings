import torch

def mean_loss_fn(encoder, samples, latent):
    samples_mean = torch.mean(samples, dim=1)
    mean_pred = encoder.mean_predictor(latent)
    diff = samples_mean - mean_pred
    loss =  torch.mean(torch.square(diff))
    return loss

def pert_pred_loss_fn(encoder, ctrl_samples, pert_samples, pert_embedding):
    ctrl_latent = encoder(ctrl_samples)
    delta_latent_pred = encoder.pert_predictor(ctrl_latent, pert_embedding)

    delta_true = pert_samples.mean(dim=1) - ctrl_samples.mean(dim=1)
    delta_pred = encoder.mean_predictor(delta_latent_pred)

    diff = delta_true - delta_pred
    loss = torch.mean(torch.square(diff))
    return loss


class PerturbSeqLossManager:
    def __init__(self, mean_loss_weight=1.0, pert_pred_loss_weight=1.0, mask_context_prob=0.0):
        self.mean_loss_weight = mean_loss_weight
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

        if self.mean_loss_weight > 0:
            mean_loss = mean_loss_fn(encoder, samples, latent)
            loss += self.mean_loss_weight * mean_loss
            losses['mean_loss'] = mean_loss
        if self.pert_pred_loss_weight > 0:
            
            pert_set_idx = ~batch['is_control'].to(device)

            if pert_set_idx.sum()!=0:

                pert_embedding = batch['pert_embedding'].to(device)
                ctrl_samples = batch['ctrl_samples'].to(device)
                pert_samples = samples[pert_set_idx]
                ctrl_samples_for_pert = ctrl_samples[pert_set_idx]
                pert_embedding = pert_embedding[pert_set_idx]

                pert_pred_loss = pert_pred_loss_fn(
                    encoder, ctrl_samples_for_pert, pert_samples, pert_embedding
                )
                loss += self.pert_pred_loss_weight * pert_pred_loss
                losses['pert_pred_loss'] = pert_pred_loss
            else:
                losses['pert_pred_loss'] = 0.0

        return loss, losses