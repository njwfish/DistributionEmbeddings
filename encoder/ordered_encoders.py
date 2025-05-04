import torch
import torch.nn as nn

from encoder.dna_conv_encoder import DNAConvEncoder

import torch
import torch.nn as nn


class LinearNextPredictor(nn.Module):
    def __init__(self, latent_dim, dropout=0.0):
        super(LinearNextPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.dropout = nn.Dropout(dropout)
        self.weights = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, ctrl_latent):
        return self.dropout(ctrl_latent @ self.weights + self.bias)

class FullyInteractedLinearNextPredictor(nn.Module):
    def __init__(self, latent_dim, dropout=0.1):
        super(FullyInteractedLinearNextPredictor, self).__init__()
        self.latent_dim = latent_dim
        
        # Create tensor for fully interactive weights
        # Shape: [latent_dim, pert_embedding_dim] - one weight for each latent-pert pair
        self.weights = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, ctrl_latent):
        # Assuming:
        # ctrl_latent shape: [batch_size, latent_dim]
        
        # Compute fully interacted model:
        # For each dimension in latent space, compute a weighted combination
        # of all perturbation embedding dimensions

        batch_size = ctrl_latent.shape[0]
        
        # Reshape ctrl_latent to [batch_size, latent_dim, 1]
        interactions = torch.einsum('bi,bj->bij', ctrl_latent, ctrl_latent)
        interactions = interactions.reshape(batch_size, -1)
        
        # Element-wise multiplication of the expanded tensors
        # Result: [batch_size, latent_dim, pert_embedding_dim]
        interaction = interactions 
        interaction = self.dropout(interaction)
        
        # Apply weights to the interaction
        # [batch_size, latent_dim, pert_embedding_dim] * [latent_dim, pert_embedding_dim]
        weighted_interaction = interaction * self.weights
        
        # Sum over the pert_embedding dimension
        # Result: [batch_size, latent_dim]
        pred = weighted_interaction.sum(dim=2) + self.bias
        
        return pred


class DNAConvEncoderNextPredictor(DNAConvEncoder):
    def __init__(
            self, 
            in_channels=5,  # ACGTN one-hot encoding
            hidden_channels=64, 
            out_channels=128, 
            hidden_dim=256,
            latent_dim=32, 
            num_layers=3, 
            kernel_size=7,
            seq_length=512,
            pool_type='mean',
            agg_type='mean',
            next_predictor_type='linear'
    ):
        super().__init__(
            in_channels, hidden_channels, out_channels, hidden_dim, latent_dim, num_layers, kernel_size, seq_length, pool_type, agg_type
        )
        
        if next_predictor_type == 'linear':
            self.next_predictor = LinearNextPredictor(latent_dim)
        elif next_predictor_type == 'interaction':
            self.next_predictor = FullyInteractedLinearNextPredictor(latent_dim)

        