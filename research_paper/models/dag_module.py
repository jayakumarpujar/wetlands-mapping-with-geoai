"""Depression-Aware Gating (DAG) module for wetland segmentation.

Core novelty: embeds geomorphological physics into the network architecture.
Wetlands only exist in topographic depressions — this module learns a spatial
gate from LiDAR depression depth maps that multiplicatively modulates decoder
feature maps, forcing the network to attend to depression regions.

Architecture:
    depression_depth → Conv2d → BN → ReLU → Conv2d → Sigmoid → gate
    features = features * gate + features * (1 - gate) * residual_weight

The gate is soft (sigmoid), not hard — the network can learn to override
the prior where depression depth alone is insufficient (e.g., riverine
wetlands, partially filled depressions).

References:
    - Wu et al. (2019) RSE: depression filtering as post-processing heuristic
    - Hu et al. (2018) CVPR: Squeeze-and-Excitation (spatial attention analog)
    - Woo et al. (2018) ECCV: CBAM spatial attention
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepressionAwareGating(nn.Module):
    """Learns a spatial gate from LiDAR depression depth to modulate features.

    The module takes a single-channel depression depth map and produces a
    spatial attention gate at the same resolution as the input feature map.
    Features are modulated via: out = features * gate + features * (1 - alpha)
    where alpha controls the strength of the gating.

    Args:
        in_channels: Number of channels in the input feature map.
        depression_channels: Number of intermediate channels for depression
            processing. Defaults to 32.
        residual_weight: Weight for the residual (ungated) connection.
            0.0 = pure gating, 1.0 = gate has no effect. Defaults to 0.1.
    """

    def __init__(
        self,
        in_channels: int,
        depression_channels: int = 32,
        residual_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.residual_weight = residual_weight

        # Depression depth encoder: 1-channel depth → spatial gate
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, depression_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depression_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depression_channels, depression_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depression_channels),
            nn.ReLU(inplace=True),
        )

        # Channel alignment: map depression features to match input channels
        self.channel_align = nn.Sequential(
            nn.Conv2d(depression_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        # Final gate: sigmoid produces [0, 1] spatial attention
        self.gate_activation = nn.Sigmoid()

    def forward(
        self,
        features: torch.Tensor,
        depression_depth: torch.Tensor,
    ) -> torch.Tensor:
        """Apply depression-aware gating to feature maps.

        Args:
            features: Decoder feature map, shape (B, C, H, W).
            depression_depth: LiDAR depression depth, shape (B, 1, H, W).
                Values >= 0 where 0 = not a depression, >0 = depth in meters.

        Returns:
            Gated feature map, same shape as input features.
        """
        # Resize depression map to match feature spatial dims if needed
        if depression_depth.shape[-2:] != features.shape[-2:]:
            depression_depth = F.interpolate(
                depression_depth,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Encode depression depth → spatial gate
        depth_features = self.depth_encoder(depression_depth)
        gate = self.gate_activation(self.channel_align(depth_features))

        # Gated features with residual connection
        return features * gate + features * self.residual_weight

    def extra_repr(self) -> str:
        return f"residual_weight={self.residual_weight}"
