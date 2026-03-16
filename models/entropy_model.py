"""Hyperprior entropy model for learned compression.

Implements the scale hyperprior from Ballé et al. 2018: the latent y is
modelled as Gaussian with mean and scale predicted from hyperprior side
information z. During training, uniform noise replaces quantization;
during inference, actual rounding + arithmetic coding is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from .encoder import HyperEncoder
from .decoder import HyperDecoder


class HyperpriorEntropy(nn.Module):
    """Scale hyperprior entropy model.

    Encodes the latent y into side information z via a hyper-encoder,
    quantises z through an entropy bottleneck, decodes z to predict
    (mean, scale) of a Gaussian that models y.
    """

    def __init__(self, latent_channels: int = 128, hyper_channels: int = 128):
        super().__init__()
        self.hyper_encoder = HyperEncoder(latent_channels, hyper_channels)
        self.hyper_decoder = HyperDecoder(latent_channels, hyper_channels)
        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, y: torch.Tensor):
        """Forward pass: adds noise during training, quantises during eval.

        Returns:
            y_hat: Quantised (or noised) latent.
            likelihoods: Dict with 'y' and 'z' likelihoods for rate computation.
        """
        z = self.hyper_encoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.hyper_decoder(z_hat)
        means, scales = gaussian_params.chunk(2, dim=1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means=means)

        return y_hat, {"y": y_likelihoods, "z": z_likelihoods}

    def compress(self, y: torch.Tensor):
        """Compress latent y into byte strings."""
        z = self.hyper_encoder(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.hyper_decoder(z_hat)
        means, scales = gaussian_params.chunk(2, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means)
        return {"y_strings": y_strings, "z_strings": z_strings,
                "z_shape": z.size()[-2:]}

    def decompress(self, y_strings, z_strings, z_shape):
        """Decompress byte strings back to latent y_hat."""
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        gaussian_params = self.hyper_decoder(z_hat)
        means, scales = gaussian_params.chunk(2, dim=1)
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(
            y_strings, indexes, means=means
        )
        return y_hat
