"""Rate-distortion loss: L = R + λ·D

Rate (R) is estimated from learned likelihoods.
Distortion (D) is MSE by default, with optional MS-SSIM or LPIPS.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RateDistortionLoss(nn.Module):
    """Rate-distortion loss with configurable distortion metric.

    Args:
        lmbda: Rate-distortion tradeoff weight λ. Higher = better quality,
               higher bitrate.
        distortion: One of 'mse', 'ms-ssim', 'lpips'.
    """

    def __init__(self, lmbda: float = 0.013, distortion: str = "mse"):
        super().__init__()
        self.lmbda = lmbda
        self.distortion = distortion

        if distortion == "ms-ssim":
            from pytorch_msssim import ms_ssim
            self._ms_ssim = ms_ssim
        elif distortion == "lpips":
            import lpips
            self._lpips_net = lpips.LPIPS(net="vgg")
            self._lpips_net.eval()
            for p in self._lpips_net.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor,
                likelihoods: dict) -> dict:
        """Compute rate-distortion loss.

        Args:
            x: Original image (B, C, H, W), values in [0, 1].
            x_hat: Reconstructed image (B, C, H, W).
            likelihoods: Dict of likelihood tensors from the entropy model.

        Returns:
            Dict with 'loss', 'rate' (bpp), 'distortion', and per-component rates.
        """
        B, C, H, W = x.shape
        num_pixels = B * H * W

        # Rate: sum of -log2(likelihoods) over all components, normalized to bpp
        rate_components = {}
        total_rate = 0.0
        for key, likelihood in likelihoods.items():
            bits = -torch.log2(likelihood.clamp(min=1e-9)).sum()
            bpp = bits / num_pixels
            rate_components[f"rate_{key}"] = bpp
            total_rate = total_rate + bpp

        # Distortion
        if self.distortion == "mse":
            dist = F.mse_loss(x_hat, x)
            # Scale MSE to 255-range for consistency with literature:
            # MSE in [0,1] range → multiply by 255² to get MSE in [0,255] range
            dist_scaled = dist * (255.0 ** 2)
        elif self.distortion == "ms-ssim":
            dist = 1.0 - self._ms_ssim(x, x_hat, data_range=1.0, size_average=True)
            dist_scaled = dist
        elif self.distortion == "lpips":
            # LPIPS expects [-1, 1] range
            dist = self._lpips_net(x * 2 - 1, x_hat * 2 - 1).mean()
            dist_scaled = dist
        else:
            raise ValueError(f"Unknown distortion metric: {self.distortion}")

        loss = total_rate + self.lmbda * dist_scaled

        return {
            "loss": loss,
            "rate": total_rate,
            "distortion": dist,
            "psnr": -10.0 * math.log10(dist.item() + 1e-10) if self.distortion == "mse" else 0.0,
            **rate_components,
        }
