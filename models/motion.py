"""Optical flow estimation and motion compensation for inter-frame coding.

Uses a lightweight SpyNet-style architecture for optical flow, then
warps the reference frame and codes the residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpyNetUnit(nn.Module):
    """Single level of the SpyNet pyramid — estimates a flow residual."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 32, 7, padding=3),  # 2×RGB + 2-ch flow = 8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, padding=3),
        )

    def forward(self, ref: torch.Tensor, target: torch.Tensor,
                flow: torch.Tensor) -> torch.Tensor:
        warped = warp(ref, flow)
        return self.net(torch.cat([ref, target, warped, flow], dim=1)) + flow


class SpyNet(nn.Module):
    """Spatial pyramid network for optical flow (Ranjan & Black 2017).

    Coarse-to-fine flow estimation over `num_levels` pyramid levels.
    """

    def __init__(self, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.units = nn.ModuleList([SpyNetUnit() for _ in range(num_levels)])

    def forward(self, ref: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Estimate optical flow from ref to target.

        Args:
            ref: Reference frame (B, 3, H, W).
            target: Target frame (B, 3, H, W).

        Returns:
            flow: (B, 2, H, W) optical flow field.
        """
        B, C, H, W = ref.shape

        # Build image pyramids
        refs = [ref]
        targets = [target]
        for _ in range(self.num_levels - 1):
            refs.append(F.avg_pool2d(refs[-1], 2))
            targets.append(F.avg_pool2d(targets[-1], 2))

        # Coarse to fine
        flow = torch.zeros(B, 2, H // (2 ** (self.num_levels - 1)),
                           W // (2 ** (self.num_levels - 1)),
                           device=ref.device)

        for level in range(self.num_levels - 1, -1, -1):
            if flow.shape[2:] != refs[level].shape[2:]:
                flow = F.interpolate(flow, size=refs[level].shape[2:],
                                     mode="bilinear", align_corners=False) * 2.0
            flow = self.units[self.num_levels - 1 - level](
                refs[level], targets[level], flow
            )

        return flow


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp img according to flow using bilinear sampling.

    Args:
        img: (B, C, H, W) image to warp.
        flow: (B, 2, H, W) optical flow (dx, dy).

    Returns:
        Warped image (B, C, H, W).
    """
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype),
        indexing="ij",
    )
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1) + flow[:, 0]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1) + flow[:, 1]

    # Normalize to [-1, 1] for grid_sample
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border",
                         align_corners=True)


class MotionCompensation(nn.Module):
    """Motion estimation + compensation module.

    Estimates flow, warps the reference frame, then refines with a small CNN.
    """

    def __init__(self, num_levels: int = 4):
        super().__init__()
        self.flow_net = SpyNet(num_levels)
        self.refine = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),  # warped + target concat
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, ref: torch.Tensor, target: torch.Tensor):
        """Compute motion-compensated prediction.

        Returns:
            prediction: Motion-compensated prediction of target.
            flow: Estimated optical flow.
        """
        flow = self.flow_net(ref, target)
        warped = warp(ref, flow)
        refined = self.refine(torch.cat([warped, target], dim=1))
        prediction = warped + refined
        return prediction, flow
