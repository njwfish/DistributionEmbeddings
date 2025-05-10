import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=True):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.is_res:
            if self.same_channels:
                return (x + x2) / 1.414
            else:
                return (x1 + x2) / 1.414
        return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return self.model(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUnetLarge(nn.Module):
    def __init__(self, in_channels=1, n_feat=384, latent_dim=10, image_size=64):
        super().__init__()
        self.n_feat = n_feat
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.init_conv = ResidualConvBlock(in_channels, n_feat)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)

        self.to_vec = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 4 * n_feat)
        self.timeembed2 = EmbedFC(1, 2 * n_feat)
        self.contextembed1 = EmbedFC(latent_dim, 4 * n_feat)
        self.contextembed2 = EmbedFC(latent_dim, 2 * n_feat)

        self.bottleneck_size = image_size // 8

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(4 * n_feat, 4 * n_feat, self.bottleneck_size, self.bottleneck_size),
            nn.GroupNorm(8, 4 * n_feat),
            nn.ReLU(),
        )

        self.upmid = UnetUp(8 * n_feat, 2 * n_feat)
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1),  # no sigmoid
        )

    def forward(self, x, c, t):
        x0 = self.init_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        hvec = self.to_vec(x3)

        cemb1 = self.contextembed1(c).view(-1, 4 * self.n_feat, 1, 1)
        temb1 = self.timeembed1(t).view(-1, 4 * self.n_feat, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, 2 * self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, 2 * self.n_feat, 1, 1)

        u0 = self.up0(hvec)
        u1 = self.upmid(cemb1 * temb1 + u0, x3)
        u2 = self.up1(cemb2 * temb2 + u1, x2)
        u3 = self.up2(u2, x1)
        out = self.out(torch.cat((u3, x0), dim=1))
        return out
