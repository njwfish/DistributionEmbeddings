import torch
import torch.nn as nn
import torch.autograd as autograd

class ConditionalWGANDiscriminator(nn.Module):
    def __init__(self, in_dim, cond_dim, hidden_dim, out_dim=1):
        super(ConditionalWGANDiscriminator, self).__init__()
        # Process the input data
        self.data_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU()
        )
        
        # Process the condition
        self.cond_layer = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SELU()
        )
        
        # Combined processing
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, condition):
        # Process data and condition separately
        data_features = self.data_layer(x)
        cond_features = self.cond_layer(condition)
        
        # Concatenate features along the last dimension
        combined = torch.cat([data_features, cond_features], dim=-1)
        
        # Process combined features
        return self.combined_layer(combined)

def compute_gradient_penalty(discriminator, real_samples, fake_samples, condition, device):
    """
    Compute gradient penalty for WGAN-GP
    
    Args:
        discriminator: discriminator to use
        real_samples: tensor of real samples
        fake_samples: tensor of generated samples
        condition: tensor of conditions
        device: device to use
    """
    # Random weight for interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # Interpolated samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Calculate discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated, condition)
    
    # Calculate gradients
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

def wgan_discriminator_loss(discriminator, real_samples, fake_samples, condition, device, lambda_gp=10):
    """
    WGAN-GP discriminator loss function
    
    Args:
        discriminator: discriminator to use
        real_samples: tensor of real samples
        fake_samples: tensor of generated samples
        condition: tensor of conditions
        device: device to use
        lambda_gp: gradient penalty coefficient
    """
    # Standard WGAN loss
    d_real = discriminator(real_samples, condition)
    d_fake = discriminator(fake_samples, condition)
    wasserstein_loss = torch.mean(d_fake) - torch.mean(d_real)
    
    # Gradient penalty
    gp = compute_gradient_penalty(discriminator, real_samples, fake_samples, condition, device)
    
    # Total loss
    return wasserstein_loss + lambda_gp * gp, wasserstein_loss.item(), gp.item()

def wgan_generator_loss(discriminator, fake_samples, condition):
    """
    WGAN generator loss function
    
    Args:
        discriminator: discriminator to use
        fake_samples: tensor of generated samples
        condition: tensor of conditions
    """
    # Generator wants to minimize: -E[D(fake, cond)]
    d_fake = discriminator(fake_samples, condition)
    return -torch.mean(d_fake)

def train_wgan(
        model, discriminator, optimizer, discriminator_optimizer, train_loader, 
        n_epochs=100, n_critic=5, lambda_gp=10, device='cuda'
):
    """
    Train the conditional WGAN-GP model
    
    Args:
        model: model to train (generator)
        discriminator: discriminator (critic) to use
        optimizer: optimizer for the generator
        discriminator_optimizer: optimizer for the discriminator
        train_loader: data loader providing data
        n_epochs: number of epochs to train
        n_critic: number of discriminator updates per generator update
        lambda_gp: gradient penalty coefficient
        device: device to train on
    """
    model.to(device)
    discriminator.to(device)
    model.train()
    discriminator.train()
    
    # For tracking progress
    losses = {
        'd_loss': [], 'wasserstein_loss': [], 'gp_loss': [], 'g_loss': []
    }
    
    for epoch in range(n_epochs):
        total_d_loss = 0
        total_wasserstein_loss = 0
        total_gp_loss = 0
        total_g_loss = 0
        d_steps = 0
        g_steps = 0
        
        for batch_data in train_loader:
            # Move data to device
            data = batch_data.to(device)
            
            # Train discriminator
            for _ in range(n_critic):
                discriminator_optimizer.zero_grad()
                
                # Generate fake samples using the condition
                with torch.no_grad():
                    latent, fake_samples = model(data)
                    # Ensure latent has the right shape for conditioning
                    latent = latent.unsqueeze(1).repeat(1, data.shape[1], 1)
                
                # Calculate discriminator loss with gradient penalty
                d_loss, wasserstein_loss, gp_loss = wgan_discriminator_loss(
                    discriminator, data, fake_samples, latent, device, lambda_gp
                )
                d_loss.backward()
                discriminator_optimizer.step()
                
                total_d_loss += d_loss.item()
                total_wasserstein_loss += wasserstein_loss
                total_gp_loss += gp_loss
                d_steps += 1
            
            # Train generator
            optimizer.zero_grad()
            
            # Generate fake samples
            latent, fake_samples = model(data)
            # Ensure latent has the right shape for conditioning
            latent = latent.unsqueeze(1).repeat(1, data.shape[1], 1)
            
            # Calculate generator loss
            g_loss = wgan_generator_loss(discriminator, fake_samples, latent)
            g_loss.backward()
            optimizer.step()
            
            total_g_loss += g_loss.item()
            g_steps += 1
        
        # Calculate average losses for this epoch
        avg_d_loss = total_d_loss / d_steps if d_steps > 0 else 0
        avg_wasserstein_loss = total_wasserstein_loss / d_steps if d_steps > 0 else 0
        avg_gp_loss = total_gp_loss / d_steps if d_steps > 0 else 0
        avg_g_loss = total_g_loss / g_steps if g_steps > 0 else 0
        
        # Store losses for tracking
        losses['d_loss'].append(avg_d_loss)
        losses['wasserstein_loss'].append(avg_wasserstein_loss)
        losses['gp_loss'].append(avg_gp_loss)
        losses['g_loss'].append(avg_g_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}/{n_epochs}")
            print(f"  Discriminator Loss: {avg_d_loss:.4f}")
            print(f"    - Wasserstein Loss: {avg_wasserstein_loss:.4f}")
            print(f"    - Gradient Penalty: {avg_gp_loss:.4f}")
            print(f"  Generator Loss: {avg_g_loss:.4f}")

    return model, discriminator, losses