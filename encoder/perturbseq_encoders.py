import torch
import torch.nn as nn

from encoder.encoders import DistributionEncoderResNet, DistributionEncoderResNetTx

import torch
import torch.nn as nn

class FullyInteractedPertPredictor(nn.Module):
    def __init__(self, latent_dim, pert_embedding_dim, dropout=0.1):
        super(FullyInteractedPertPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.pert_embedding_dim = pert_embedding_dim
        self.weights = nn.Parameter(torch.zeros(latent_dim * pert_embedding_dim, latent_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, ctrl_latent, pert_embedding):
        # compute interaction between latent and pert embedding flattened
        # then apply dropout
        # then apply a linear layer to get the final prediction

        batch_size = ctrl_latent.shape[0]

        interaction = torch.einsum('bi,bj->bij', ctrl_latent, pert_embedding)
        interaction = interaction.reshape(batch_size, -1)
        interaction = self.dropout(interaction)
        pred = interaction @ self.weights + self.bias
        
        return pred
        

class FullyInteractedPolynomialPertPredictor(nn.Module):
    def __init__(self, latent_dim, pert_embedding_dim, dropout=0.1):
        super(FullyInteractedPolynomialPertPredictor, self).__init__()
        self.latent_dim = latent_dim
        self.pert_embedding_dim = pert_embedding_dim
        
        # Create tensor for fully interactive weights
        # Shape: [latent_dim, pert_embedding_dim] - one weight for each latent-pert pair
        self.weights = nn.Parameter(torch.zeros(latent_dim * pert_embedding_dim ** 2, latent_dim))
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
        interaction = interaction.reshape(batch_size, -1)
        interaction = self.dropout(interaction)
        
        # Sum over the pert_embedding dimension
        # Result: [batch_size, latent_dim]
        pred = interaction @ self.weights + self.bias
        
        return pred

class DistributionEncoderResNetPertPredictor(DistributionEncoderResNet):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2, norm=True, pert_embedding_dim=128, dropout=0.1):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size, layers, fc_layers, norm)
        
        self.pert_predictor = FullyInteractedPertPredictor(latent_dim, pert_embedding_dim, dropout)
        self.mean_predictor = nn.Linear(latent_dim, in_dim)
        

class DistributionEncoderResNetTxPertPredictor(DistributionEncoderResNetTx):
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=2, fc_layers=2, heads=4, norm=True, pert_embedding_dim=128):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size, layers, fc_layers, heads, norm)
        
        self.pert_predictor = FullyInteractedPertPredictor(latent_dim, pert_embedding_dim, dropout)
        self.mean_predictor = nn.Linear(latent_dim, in_dim)
        
        
        
        