import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels, is_res=True), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels, is_res=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class CNNVAE(nn.Module):
    def __init__(
        self, 
        in_channels=1,           # Number of input channels (1 for grayscale, 3 for RGB)
        n_feat=256,              # Number of features in hidden layers, similar to ContextUnet
        latent_dim=10,           # Dimension of the condition vector
        vae_latent_dim=100,      # Dimension of the VAE latent space
        image_size=28,           # Image dimensions (assuming square images)
    ):
        super(CNNVAE, self).__init__()
        
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.latent_dim = latent_dim
        self.vae_latent_dim = vae_latent_dim
        self.image_size = image_size
        
        # Calculate bottleneck size (similar to ContextUnet)
        self.bottleneck_size = image_size // 4
        
        # Context embedding
        self.context_embed = EmbedFC(latent_dim, n_feat)
        
        # Encoder
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = DownBlock(n_feat, n_feat)
        self.down2 = DownBlock(n_feat, 2 * n_feat)
        
        # Feature dimension at bottleneck
        self.bottleneck_dim = 2 * n_feat * (self.bottleneck_size ** 2)
        
        # Fully connected layers for encoder (mu and logvar)
        self.fc_mu = nn.Linear(self.bottleneck_dim + n_feat, vae_latent_dim)
        self.fc_logvar = nn.Linear(self.bottleneck_dim + n_feat, vae_latent_dim)
        
        # Decoder fully connected
        self.fc_decode = nn.Linear(vae_latent_dim + n_feat, self.bottleneck_dim)
        
        # Decoder network
        self.up0 = ResidualConvBlock(2 * n_feat, 2 * n_feat, is_res=True)
        self.up1 = UpBlock(2 * n_feat, n_feat)
        self.up2 = UpBlock(n_feat, n_feat)
        
        # Final output
        self.out = nn.Sequential(
            ResidualConvBlock(n_feat, n_feat, is_res=True),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),
            nn.Sigmoid()  # Output values between 0 and 1 for images
        )
    
    def encode(self, x, c):
        # Embed context
        c_embedded = self.context_embed(c)
        
        # Process through the CNN
        h = self.init_conv(x)
        h = self.down1(h)
        h = self.down2(h)
        
        # Flatten for fully connected layers
        h_flat = h.view(h.size(0), -1)
        
        # Concatenate with condition embedding
        h_concat = torch.cat([h_flat, c_embedded], dim=1)
        
        # Get mu and logvar
        mu = self.fc_mu(h_concat)
        logvar = self.fc_logvar(h_concat)
        
        return mu, logvar
    
    def decode(self, z, c):
        # Embed context
        c_embedded = self.context_embed(c)
        
        # Concatenate latent and context
        z_concat = torch.cat([z, c_embedded], dim=1)
        
        # Process through fully connected layer
        h = self.fc_decode(z_concat)
        
        # Reshape to spatial feature maps
        h = h.view(-1, 2 * self.n_feat, self.bottleneck_size, self.bottleneck_size)
        
        # Process through transpose convolutions
        h = self.up0(h)
        h = self.up1(h)
        h = self.up2(h)
        
        # Final output layer
        x_reconstructed = self.out(h)
        
        return x_reconstructed 