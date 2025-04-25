'''
Convolutional Neural Network for processing sets of DNA sequences.
This encoder processes sets of one-hot encoded DNA sequences and generates
a latent representation of the distribution.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DGNNLayer(nn.Module):
    """
    A single convolutional GNN layer that processes sets of DNA sequences.
    Each sequence is processed by a shared 1D CNN, and then message passing
    is performed by aggregating features across the set.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, agg_type='mean'):
        super(Conv1DGNNLayer, self).__init__()
        self.agg_type = agg_type  # Aggregation type: 'mean' or 'median'
        
        # Convolutional layers for processing individual sequences
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
        # Layer for processing aggregated features
        self.update = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, set_size, channels, seq_length]
              representing a batch of sets of sequences
        Returns:
            Updated tensor of same shape
        """
        batch_size, set_size, channels, seq_length = x.shape
        
        # Reshape to process all sequences with the shared CNN
        x_flat = x.view(batch_size * set_size, channels, seq_length)
        x_proc = self.conv(x_flat)
        
        # Reshape back
        _, out_channels, seq_out = x_proc.shape
        x_reshaped = x_proc.view(batch_size, set_size, out_channels, seq_out)
        
        # Aggregate features across the set dimension
        if self.agg_type == 'mean':
            agg = torch.mean(x_reshaped, dim=1, keepdim=True)
        elif self.agg_type == 'median':
            agg = torch.median(x_reshaped, dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown aggregation type: {self.agg_type}")
        
        # Broadcast aggregated features to all elements in the set
        agg = agg.expand(-1, set_size, -1, -1)
        
        # Concatenate original features with aggregated ones
        combined = torch.cat([x_reshaped, agg], dim=2)
        
        # Reshape for processing with self.update
        combined_flat = combined.view(batch_size * set_size, out_channels * 2, seq_out)
        updated = self.update(combined_flat)
        
        # Reshape back to the original format
        updated = updated.view(batch_size, set_size, out_channels, seq_out)
        
        return updated


class DNAConvEncoder(nn.Module):
    """
    Convolutional Neural Network for processing sets of DNA sequences.
    """
    def __init__(self, 
                 in_channels=5,  # ACGTN one-hot encoding
                 hidden_channels=64, 
                 out_channels=128, 
                 hidden_dim=256,
                 latent_dim=32, 
                 num_layers=3, 
                 kernel_size=7,
                 seq_length=512,
                 pool_type='mean',
                 agg_type='mean'):
        super(DNAConvEncoder, self).__init__()
        
        self.pool_type = pool_type
        self.out_channels = out_channels
        self.seq_length = seq_length
        
        # Initial convolutional layer
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU()
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(Conv1DGNNLayer(hidden_channels, hidden_channels, kernel_size, agg_type))
        
        # Final projection to output channels
        self.final_conv = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        
        # MLP for processing sequence representations before pooling
        self.pre_pool_mlp = nn.Sequential(
            nn.Linear(seq_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels)
        )
        
        # Final MLP for processing pooled representations
        self.final_mlp = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        """
        Args:
            x: DNA sequences tensor or dictionary containing encoder_inputs
            For sets: encoder_inputs shape is [batch_size, set_size, seq_length, channels]
            For non-sets: encoder_inputs shape is [batch_size, seq_length, channels]
        Returns:
            Latent representation of the distribution of shape [batch_size, latent_dim]
        """
        # Handle both dictionary input (with 'encoder_inputs' key) and direct tensor input
        if isinstance(x, dict):
            x = x['encoder_inputs']
            

        batch_size, set_size, seq_length, channels = x.shape

        
        # Reshape for 1D convolution (channels first)
        x = x.permute(0, 1, 3, 2)  # [batch_size, set_size, channels, seq_length]
        
        # Reshape for initial processing
        x_flat = x.reshape(batch_size * set_size, channels, seq_length)
        x_init = self.initial_conv(x_flat)
        
        # Reshape back
        _, hidden_channels, seq_out = x_init.shape
        x_reshaped = x_init.view(batch_size, set_size, hidden_channels, seq_out)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x_reshaped = gnn_layer(x_reshaped)
        
        # Apply final projection
        x_flat = x_reshaped.view(batch_size * set_size, hidden_channels, seq_out)
        x_final = self.final_conv(x_flat)
        x_final = x_final.view(batch_size, set_size, self.out_channels, seq_out)
        
        # Process each sequence with pre-pooling MLP
        x_final = x_final.view(batch_size, set_size, self.out_channels, seq_out)
        x_pre_pool = self.pre_pool_mlp(x_final)  # Collapse sequence dimension
        x_pre_pool = torch.mean(x_pre_pool, dim=2)  # Average over channel dimension
        
        # Pool across the set dimension to get a single representation per batch
        if self.pool_type == 'mean':
            pooled = torch.mean(x_pre_pool, dim=1)
        elif self.pool_type == 'median':
            pooled = torch.median(x_pre_pool, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")
        
        # Final transformation to latent space
        latent = self.final_mlp(pooled)
        
        return latent 