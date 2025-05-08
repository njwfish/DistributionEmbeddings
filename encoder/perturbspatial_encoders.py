import torch
import torch.nn as nn

from encoder.conv_gnn import ConvDistributionEncoder

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
            nn.Linear(pert_embedding_dim, hidden_dim),
            nn.Dropout(dropout),
            activation,
        ]
        for _ in range(self.layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(activation)
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.pert_predictor = nn.Sequential(*layers)

    def forward(self, pert_embedding):
        return self.pert_predictor(pert_embedding)
    
class FullyInteractedLinearPertPredictor(nn.Module):
    def __init__(self, latent_dim, pert_embedding_dim, dropout=0.1):
        super(FullyInteractedLinearPertPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.pert_embedding_dim = pert_embedding_dim
        
        # Create tensor for fully interactive weights
        # Shape: [latent_dim, pert_embedding_dim] - one weight for each latent-pert pair
        self.weights = nn.Parameter(torch.randn(pert_embedding_dim ** 2, latent_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        
        # Add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, pert_embedding):
        # Assuming:
        # pert_embedding shape: [batch_size, pert_embedding_dim]
        
        # Compute fully interacted model:
        # For each dimension in latent space, compute a weighted combination
        # of all perturbation embedding dimensions

        batch_size = pert_embedding.shape[0]
        
        interactions = torch.einsum('bi,bj->bij', pert_embedding, pert_embedding)
        interactions = interactions.reshape(batch_size, -1)  # Result will be (batch_size, d*d)
        interactions = self.dropout(interactions)
        
        # Sum over the pert_embedding dimension
        # Result: [batch_size, latent_dim]
        pred = interactions @ self.weights + self.bias
        
        return pred


class ConvDistributionEncoderPertPredictor(ConvDistributionEncoder):
    def __init__(self, pert_embedding_dim=128, **kwargs):
        super().__init__(**kwargs)
        latent_dim = kwargs['latent_dim']
        self.pert_predictor = FullyInteractedLinearPertPredictor(latent_dim, pert_embedding_dim)
        
