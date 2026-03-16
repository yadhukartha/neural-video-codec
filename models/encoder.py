"""CNN encoder with residual blocks for learned image/video compression.

Architecture follows Ballé et al. 2018 with strided convolutions for
downsampling and GDN (generalized divisive normalization) activations.
"""

import torch
import torch.nn as nn
from compressai.layers import GDN


class ResidualBlock(nn.Module):
    """Pre-activation residual block with two 3×3 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    """Downsampling encoder: 3→N channels over 4 strided conv stages (16× downsample).

    Args:
        in_channels: Input image channels (3 for RGB).
        latent_channels: Number of output latent channels.
        num_res_blocks: Residual blocks after each downsampling stage.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 128,
                 num_res_blocks: int = 3):
        super().__init__()
        ch = latent_channels

        layers = [
            nn.Conv2d(in_channels, ch, 5, stride=2, padding=2),
            GDN(ch),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(ch))

        layers += [
            nn.Conv2d(ch, ch, 5, stride=2, padding=2),
            GDN(ch),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(ch))

        layers += [
            nn.Conv2d(ch, ch, 5, stride=2, padding=2),
            GDN(ch),
        ]

        layers += [
            nn.Conv2d(ch, ch, 5, stride=2, padding=2),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HyperEncoder(nn.Module):
    """Hyperprior encoder: compresses the latent into side information.

    Two strided conv stages → 2× additional downsample of the latent.
    """

    def __init__(self, latent_channels: int = 128, hyper_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, 5, stride=2, padding=2),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)
