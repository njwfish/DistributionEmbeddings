'''
Convolutional Graph Neural Network (GNN) for processing sets of images.
This can be incorporated into the DDPM architecture for diffusion models
that operate on sets of images.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGNNLayer(nn.Module):
    """
    A single convolutional GNN layer that processes sets of images.
    Each image is processed by a shared CNN, and then message passing
    is performed by aggregating features across the set.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, agg_type='mean'):
        super(ConvGNNLayer, self).__init__()
        self.agg_type = agg_type  # Aggregation type: 'mean', 'max', 'median'
        
        # Convolutional layers for processing individual images
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        
        # Layer for processing aggregated features
        self.update = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, set_size, channels, height, width]
              representing a batch of sets of images
        Returns:
            Updated tensor of same shape
        """
        batch_size, set_size, channels, height, width = x.shape
        
        # Reshape to process all images with the shared CNN
        x_flat = x.view(batch_size * set_size, channels, height, width)
        x_proc = self.conv(x_flat)
        
        # Reshape back
        _, out_channels, h_out, w_out = x_proc.shape
        x_reshaped = x_proc.view(batch_size, set_size, out_channels, h_out, w_out)
        
        # Aggregate features across the set dimension
        if self.agg_type == 'mean':
            agg = torch.mean(x_reshaped, dim=1, keepdim=True)
        elif self.agg_type == 'median':
            agg = torch.median(x_reshaped, dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"Unknown aggregation type: {self.agg_type}")
        
        # Broadcast aggregated features to all elements in the set
        agg = agg.expand(-1, set_size, -1, -1, -1)
        
        # Concatenate original features with aggregated ones
        combined = torch.cat([x_reshaped, agg], dim=2)
        
        # Reshape for processing with self.update
        combined_flat = combined.view(batch_size * set_size, out_channels * 2, h_out, w_out)
        updated = self.update(combined_flat)
        
        # Reshape back to the original format
        updated = updated.view(batch_size, set_size, out_channels, h_out, w_out)
        
        return updated


class ImageSetGNN(nn.Module):
    """
    Graph Neural Network for processing sets of images.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_dim, latent_dim, 
                num_layers=2, kernel_size=3, height=28, width=28, pool_type='mean', agg_type='mean'):
        super(ImageSetGNN, self).__init__()
        
        self.pool_type = pool_type
        self.out_channels = out_channels  # Store out_channels as class variable
        
        # Initial convolutional layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU()
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(ConvGNNLayer(hidden_channels, hidden_channels, kernel_size, agg_type))
        
        # Final projection to output channels
        self.final_conv = nn.Conv2d(hidden_channels, self.out_channels, kernel_size=1)
        self.final_mlp = nn.Sequential(
            nn.Linear(self.out_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, set_size, channels, height, width]
              representing a batch of sets of images
        Returns:
            - Pooled representation of shape [batch_size, out_channels, height, width]
            - Updated set representations of shape [batch_size, set_size, out_channels, height, width]
        """
        batch_size, set_size, channels, height, width = x.shape
        
        # Reshape for initial processing
        x_flat = x.view(batch_size * set_size, channels, height, width)
        x_init = self.initial_conv(x_flat)
        
        # Reshape back
        _, hidden_channels, h_out, w_out = x_init.shape
        x_reshaped = x_init.view(batch_size, set_size, hidden_channels, h_out, w_out)
        
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x_reshaped = gnn_layer(x_reshaped)
        
        # Apply final projection
        x_flat = x_reshaped.view(batch_size * set_size, hidden_channels, h_out, w_out)
        x_final = self.final_conv(x_flat)
        x_final = x_final.view(batch_size, set_size, self.out_channels, h_out, w_out)

        pooled = torch.max(x_final, dim=-1).values
        pooled = torch.max(pooled, dim=-1).values
        
        # Pool across the set dimension to get a single representation per batch
        if self.pool_type == 'mean':
            pooled = torch.mean(pooled, dim=1)
        elif self.pool_type == 'median':
            pooled = torch.median(pooled, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")
        
        # pooled = pooled.view(batch_size, self.out_channels)
        pooled = self.final_mlp(pooled)
        
        return pooled
