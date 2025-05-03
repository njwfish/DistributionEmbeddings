import torch
import torch.nn as nn

from .perturbseq_encoders import DistributionEncoderResNet, DistributionEncoderResNetTx

import torch
import torch.nn as nn

class PertPredictor(nn.Module):
    def __init__(self, latent_dim, pert_embedding_dim, hidden_dim=32, dropout=0.1, activation='selu', layers=1):
        super(PertPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.pert_embedding_dim = pert_embedding_dim
        self.layers = layers
        
        if activation == 'selu':
            activation = nn.SELU()
        elif activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'leaky_relu':
            activation = nn.LeakyReLU()
            
        # add layers 
        layers = [
            nn.Linear(latent_dim + pert_embedding_dim, hidden_dim),
            nn.Dropout(dropout),
            activation,
        ]
        for _ in range(self.layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(activation)
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.pert_predictor = nn.Sequential(*layers)

    def forward(self, ctrl_latent, pert_embedding):
        return self.pert_predictor(torch.cat([ctrl_latent, pert_embedding], dim=1))
    
class FullyInteractedLinearPertPredictor(nn.Module):
    def __init__(self, latent_dim, pert_embedding_dim, dropout=0.1):
        super(FullyInteractedLinearPertPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.pert_embedding_dim = pert_embedding_dim
        
        # Create tensor for fully interactive weights
        # Shape: [latent_dim, pert_embedding_dim] - one weight for each latent-pert pair
        self.weights = nn.Parameter(torch.randn(latent_dim, pert_embedding_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, ctrl_latent, pert_embedding):
        # Assuming:
        # ctrl_latent shape: [batch_size, latent_dim]
        # pert_embedding shape: [batch_size, pert_embedding_dim]
        
        # Compute fully interacted model:
        # For each dimension in latent space, compute a weighted combination
        # of all perturbation embedding dimensions
        
        # Reshape ctrl_latent to [batch_size, latent_dim, 1]
        ctrl_latent_expanded = ctrl_latent.unsqueeze(-1)
        
        # Reshape pert_embedding to [batch_size, 1, pert_embedding_dim]
        pert_embedding_expanded = pert_embedding.unsqueeze(1)
        
        # Element-wise multiplication of the expanded tensors
        # Result: [batch_size, latent_dim, pert_embedding_dim]
        interaction = ctrl_latent_expanded * pert_embedding_expanded
        interaction = self.dropout(interaction)
        
        # Apply weights to the interaction
        # [batch_size, latent_dim, pert_embedding_dim] * [latent_dim, pert_embedding_dim]
        weighted_interaction = interaction * self.weights
        
        # Sum over the pert_embedding dimension
        # Result: [batch_size, latent_dim]
        pred = weighted_interaction.sum(dim=2) + self.bias
        
        return pred


class FullyInteractedPolynomialPertPredictor(nn.Module):
    def __init__(self, latent_dim, pert_embedding_dim, dropout=0.1):
        super(FullyInteractedPolynomialPertPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.pert_embedding_dim = pert_embedding_dim
        
        # Create tensor for fully interactive weights
        # Shape: [latent_dim, pert_embedding_dim] - one weight for each latent-pert pair
        self.weights = nn.Parameter(torch.zeros(latent_dim, pert_embedding_dim ** 2))
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, ctrl_latent, pert_embedding):
        # Assuming:
        # ctrl_latent shape: [batch_size, latent_dim]
        # pert_embedding shape: [batch_size, pert_embedding_dim]
        
        # Compute fully interacted model:
        # For each dimension in latent space, compute a weighted combination
        # of all perturbation embedding dimensions

        batch_size = ctrl_latent.shape[0]
        
        # Reshape ctrl_latent to [batch_size, latent_dim, 1]
        ctrl_latent_expanded = ctrl_latent.unsqueeze(-1)

        # Get 2nd degree interaction for pert_embedding
        interactions = torch.einsum('bi,bj->bij', pert_embedding, pert_embedding)
        interactions = interactions.reshape(batch_size, -1)  # Result will be (batch_size, d*d)
        
        # Reshape pert_embedding to [batch_size, 1, pert_embedding_dim]
        pert_embedding_expanded = interactions.unsqueeze(1)
        
        # Element-wise multiplication of the expanded tensors
        # Result: [batch_size, latent_dim, pert_embedding_dim]
        interaction = ctrl_latent_expanded * pert_embedding_expanded
        interaction = self.dropout(interaction)
        
        # Apply weights to the interaction
        # [batch_size, latent_dim, pert_embedding_dim] * [latent_dim, pert_embedding_dim]
        weighted_interaction = interaction * self.weights
        
        # Sum over the pert_embedding dimension
        # Result: [batch_size, latent_dim]
        pred = weighted_interaction.sum(dim=2) + self.bias
        
        return pred

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

class DistributionEncoderResNetTx(DistributionEncoder):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2, heads=4, norm=True):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        
        # Initial projection to hidden dimension
        self.input_projection = MLP(in_dim, hidden_dim, hidden_dim, fc_layers)
        
        # Create MeanPooledFC layers
        self.pooled_layers = nn.ModuleList([
            SelfAttention(hidden_dim, heads) 
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
        
        
        
        
        