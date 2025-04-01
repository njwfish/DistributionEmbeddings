''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is based on,
https://github.com/TeaPearce/Conditional_Diffusion_MNIST

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

import torch
import torch.nn as nn
import numpy as np



def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, model, betas, n_T, drop_prob=0.1, noise_shape=None):
        super(DDPM, self).__init__()
        self.model = model
        self.noise_shape = noise_shape

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        c = c.unsqueeze(1).repeat(1, x.shape[0] // c.shape[0], 1).view(-1, c.shape[-1]) 

        # x = x.reshape(-1, 1, 28, 28)

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(c.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x).to(c.device)  # eps ~ N(0, 1)
        x_t = (
            self.sqrtab.to(c.device)[_ts][(...,) + (None,) * (x.ndim - 1)] * x
            + self.sqrtmab.to(c.device)[_ts][(...,) + (None,) * (x.ndim - 1)] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        # context_mask = torch.bernoulli(torch.zeros(c.shape[0])+self.drop_prob).to(self.device)
        
        # sample random indices across features
        indices = torch.randperm(x.shape[1])[:1_000].to(x.device)
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(
            noise[:, indices], self.model(x_t, c, _ts[:, None] / self.n_T, node_indices=indices)
        )
    
    def loss(self, x, c):
        return self.forward(x, c)

    def sample(self, context, num_samples, return_trajectory=False):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance
        device = context.device
        n_sets = context.shape[0]
        context = context.unsqueeze(1).repeat(1, num_samples, 1).view(-1, context.shape[-1]) 
        n_sample = context.shape[0]

        x = torch.randn(n_sample, *self.noise_shape).to(device)  # x_T ~ N(0, 1), sample initial noise

        x_trajectory = [] # keep track of generated steps in case want to plot something 
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, *([1] * (x.ndim - 1)))

            z = torch.randn_like(x).to(device) if i > 1 else 0

            eps = self.model(x, context, t_is)

            x = (
                self.oneover_sqrta[i] * (x - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            
            if return_trajectory:
                x_trajectory.append(x.detach().cpu().numpy())
        
        x = x.view(n_sets, num_samples, *self.noise_shape)
        if return_trajectory:
            x_trajectory = np.array(x_trajectory)
            return x, x_trajectory
        else:
            return x
