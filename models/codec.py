"""Full end-to-end video codec tying together intra/inter-frame coding."""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .entropy_model import HyperpriorEntropy
from .motion import MotionCompensation


class IntraFrameCodec(nn.Module):
    """Image compression codec with hyperprior entropy model.

    Implements Ballé et al. 2018: encoder → quantise → entropy code → decoder,
    with a scale hyperprior for adaptive entropy modelling.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 128,
                 hyper_channels: int = 128, num_res_blocks: int = 3):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, num_res_blocks)
        self.decoder = Decoder(latent_channels, in_channels, num_res_blocks)
        self.entropy_model = HyperpriorEntropy(latent_channels, hyper_channels)

    def forward(self, x: torch.Tensor):
        """Encode, quantise, decode an image.

        Returns:
            x_hat: Reconstructed image.
            likelihoods: Dict with 'y' and 'z' likelihoods for rate loss.
        """
        y = self.encoder(x)
        y_hat, likelihoods = self.entropy_model(y)
        x_hat = self.decoder(y_hat)
        return x_hat, likelihoods

    def compress(self, x: torch.Tensor):
        """Compress an image to byte strings."""
        y = self.encoder(x)
        return self.entropy_model.compress(y)

    def decompress(self, strings: dict):
        """Decompress byte strings to an image."""
        y_hat = self.entropy_model.decompress(
            strings["y_strings"], strings["z_strings"], strings["z_shape"]
        )
        return self.decoder(y_hat)


class InterFrameCodec(nn.Module):
    """Inter-frame codec: motion compensation + residual coding.

    Given a reference frame, estimates motion, warps, then compresses
    the residual using the same hyperprior architecture.
    """

    def __init__(self, latent_channels: int = 64, hyper_channels: int = 64,
                 num_res_blocks: int = 2):
        super().__init__()
        self.motion = MotionCompensation(num_levels=4)
        # Residual codec (compresses prediction error)
        self.residual_encoder = Encoder(3, latent_channels, num_res_blocks)
        self.residual_decoder = Decoder(latent_channels, 3, num_res_blocks)
        self.residual_entropy = HyperpriorEntropy(latent_channels, hyper_channels)
        # Motion codec (compresses optical flow)
        self.motion_encoder = Encoder(2, latent_channels, num_res_blocks)
        self.motion_decoder = Decoder(latent_channels, 2, num_res_blocks)
        self.motion_entropy = HyperpriorEntropy(latent_channels, hyper_channels)

    def forward(self, ref: torch.Tensor, target: torch.Tensor):
        """Compress target frame given reference frame.

        Returns:
            x_hat: Reconstructed target.
            likelihoods: Dict with motion and residual likelihoods.
            extras: Dict with flow, prediction, residual for visualization.
        """
        # Motion estimation & compensation
        prediction, flow = self.motion(ref, target)

        # Compress flow
        flow_y = self.motion_encoder(flow)
        flow_y_hat, flow_likelihoods = self.motion_entropy(flow_y)
        flow_hat = self.motion_decoder(flow_y_hat)

        # Re-warp with decoded flow for consistency
        from .motion import warp
        prediction_hat = warp(ref, flow_hat)

        # Residual coding
        residual = target - prediction_hat
        res_y = self.residual_encoder(residual)
        res_y_hat, res_likelihoods = self.residual_entropy(res_y)
        residual_hat = self.residual_decoder(res_y_hat)

        x_hat = prediction_hat + residual_hat

        likelihoods = {
            "motion_y": flow_likelihoods["y"],
            "motion_z": flow_likelihoods["z"],
            "residual_y": res_likelihoods["y"],
            "residual_z": res_likelihoods["z"],
        }
        extras = {
            "flow": flow,
            "flow_hat": flow_hat,
            "prediction": prediction_hat,
            "residual": residual,
        }
        return x_hat, likelihoods, extras


class VideoCodec(nn.Module):
    """Full video codec: intra-frame for keyframes, inter-frame for P-frames."""

    def __init__(self, intra_cfg: dict, inter_cfg: dict):
        super().__init__()
        self.intra_codec = IntraFrameCodec(**intra_cfg)
        self.inter_codec = InterFrameCodec(**inter_cfg)

    def forward(self, frames: torch.Tensor, gop_size: int = 10):
        """Compress a batch of video frames.

        Args:
            frames: (B, T, C, H, W) tensor of T consecutive frames.
            gop_size: Group-of-pictures size (keyframe interval).

        Returns:
            reconstructions: List of reconstructed frames.
            all_likelihoods: List of likelihood dicts per frame.
        """
        B, T, C, H, W = frames.shape
        reconstructions = []
        all_likelihoods = []

        for t in range(T):
            frame = frames[:, t]
            if t % gop_size == 0:
                # I-frame
                x_hat, likelihoods = self.intra_codec(frame)
                reconstructions.append(x_hat)
                all_likelihoods.append(likelihoods)
            else:
                # P-frame
                ref = reconstructions[-1].detach()
                x_hat, likelihoods, _ = self.inter_codec(ref, frame)
                reconstructions.append(x_hat)
                all_likelihoods.append(likelihoods)

        return reconstructions, all_likelihoods
