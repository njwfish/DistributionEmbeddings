import torch
import torch.nn as nn

from layers import MLP, MeanPooledFC, MedianPooledFC, SelfAttention

class DistributionEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size):
        super().__init__()
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.latent_dim = latent_dim
        # self.positional_embedding = nn.Embedding(set_size, latent_dim)

        self.latent_act = nn.SELU()
        self.encoder = None

    def forward(self, x):
        enc = self.encoder(x)
        # generate compressed latent by mean pooling
        enc_mean = torch.mean(enc, dim=1)
        # enc_mean = torch.median(enc, dim=1).values
        lat = self.latent_act(self.latent_proj(enc_mean))
        return lat

class DistributionEncoderTx(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, heads=4):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            *[SelfAttention(hidden_dim, heads) for _ in range(layers)]
        )

class DistributionEncoderGNN(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            MLP(in_dim, hidden_dim, hidden_dim, fc_layers),
            *[MeanPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)]
        )

class DistributionEncoderMedianGNN(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        self.encoder = nn.Sequential(
            MLP(in_dim, hidden_dim, hidden_dim, fc_layers),
            *[MedianPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) for _ in range(layers)]
        )

class DistributionEncoderResNet(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2, norm=True):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        
        # Initial projection to hidden dimension
        self.input_projection = MLP(in_dim, hidden_dim, hidden_dim, fc_layers)
        
        # Create MeanPooledFC layers
        self.pooled_layers = nn.ModuleList([
            MeanPooledFC(hidden_dim, hidden_dim, hidden_dim, fc_layers) 
            for _ in range(layers)
        ])
        
        # Create input projections for each layer (including initial)
        # These allow us to add properly scaled original inputs at each stage
        self.input_projections = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim) for _ in range(layers + 1)
        ])
        
        # Layer norms for normalization after skip connections
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])
            
        # Override the encoder in parent class
        self.encoder = None
        self.norm = norm
        
    def forward(self, x):
        # Store original input for later use
        original_input = x
        
        # Initial projection to hidden dimension
        x = self.input_projection(x)
        
        # Add first projection of original input
        x = x + self.input_projections[0](original_input)
        
        # Apply layer norm to the initial combined representation
        x = nn.functional.layer_norm(x, [x.size(-1)])
        
        # Store the processed output to use in skip connections
        identity = x
        
        # Apply MeanPooledFC layers with skip connections
        for i, (layer, norm, input_proj) in enumerate(zip(
                self.pooled_layers, 
                self.layer_norms, 
                self.input_projections[1:])):
            # Apply MeanPooledFC
            layer_output = layer(x)
            
            # Add skip connection from previous layer
            # Add projection of original input for this layer
            # This ensures the original signal is preserved throughout the network
            x = layer_output + identity + input_proj(original_input)
            
            # Apply normalization
            if self.norm:
                x = norm(x)
            
            # Update identity for next layer
            identity = x
        
        # Mean pooling for latent representation
        enc_mean = torch.mean(x, dim=1)
        
        # Final projection to latent space
        lat = self.latent_act(self.latent_proj(enc_mean))
        return lat
