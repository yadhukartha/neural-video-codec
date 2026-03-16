"""Mirror decoder with transposed convolutions and inverse GDN."""

import torch
import torch.nn as nn
from compressai.layers import GDN

from .encoder import ResidualBlock


class Decoder(nn.Module):
    """Upsampling decoder: N→3 channels over 4 transposed conv stages (16× upsample).

    Mirrors the encoder architecture with ConvTranspose2d and inverse GDN.
    """

    def __init__(self, latent_channels: int = 128, out_channels: int = 3,
                 num_res_blocks: int = 3):
        super().__init__()
        ch = latent_channels

        layers = [
            nn.ConvTranspose2d(ch, ch, 5, stride=2, padding=2, output_padding=1),
            GDN(ch, inverse=True),
        ]

        layers += [
            nn.ConvTranspose2d(ch, ch, 5, stride=2, padding=2, output_padding=1),
            GDN(ch, inverse=True),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(ch))

        layers += [
            nn.ConvTranspose2d(ch, ch, 5, stride=2, padding=2, output_padding=1),
            GDN(ch, inverse=True),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(ch))

        layers += [
            nn.ConvTranspose2d(ch, out_channels, 5, stride=2, padding=2, output_padding=1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        return self.net(y_hat)


class HyperDecoder(nn.Module):
    """Hyperprior decoder: expands side information to predict latent distribution params.

    Outputs 2×latent_channels for (mean, scale) of the Gaussian entropy model.
    """

    def __init__(self, latent_channels: int = 128, hyper_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 5, stride=2,
                               padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 5, stride=2,
                               padding=2, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hyper_channels, latent_channels * 2, 3, stride=1, padding=1),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return self.net(z_hat)
