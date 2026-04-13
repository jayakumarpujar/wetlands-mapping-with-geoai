"""Multi-Temporal SSM fusion for wetland phenology modeling.

Wetlands exhibit temporal dynamics: seasonal flooding cycles, vegetation
phenology, and long-term hydrological changes. Instead of naively
concatenating multi-epoch features along the channel dimension, this module
treats each epoch as a step in a temporal sequence and uses a Mamba SSM
to model state transitions.

Key insight: SSM state naturally captures "memory" of previous epochs,
making it ideal for distinguishing:
    - Permanent wetlands (consistently wet across epochs)
    - Seasonal wetlands (wet in spring, dry in summer)
    - Ephemeral wetlands (wet only after precipitation events)

Architecture:
    epoch_features: [(B, C, H, W), (B, C, H, W), ...] — one per epoch
    → stack along temporal dim → (B, T, C, H, W)
    → per-pixel temporal SSM: for each (h,w), process (T, C) sequence
    → output: (B, C, H, W) — temporally fused features

References:
    - ChangeMamba (Chen et al. 2024): SSM for change detection
    - VideoMamba (Li et al. 2024): SSM for temporal video modeling
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_paper.models.mamba_decoder import SSMBlock


class TemporalSSMFusion(nn.Module):
    """Fuse multi-epoch features using temporal state-space modeling.

    Each spatial position (h, w) is treated independently. For each pixel,
    the feature vectors from T epochs form a short temporal sequence that
    is processed by a lightweight SSM block.

    For efficiency, spatial dimensions are flattened into the batch dim:
    (B, T, C, H, W) → (B*H*W, T, C) → SSM → (B, C, H, W).

    Args:
        channels: Feature dimension per epoch.
        d_state: SSM state expansion. Defaults to 16.
        n_epochs_max: Maximum number of temporal epochs. Defaults to 6.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        n_epochs_max: int = 6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.n_epochs_max = n_epochs_max

        # Learnable temporal position embeddings
        self.temporal_pos = nn.Parameter(torch.randn(1, n_epochs_max, channels) * 0.02)

        # Temporal SSM block
        self.temporal_ssm = SSMBlock(dim=channels, d_state=d_state)

        # Output projection (fuse temporal info back to spatial)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels, bias=False),
        )

    def forward(self, epoch_features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multi-epoch features via temporal SSM.

        Args:
            epoch_features: List of T tensors, each (B, C, H, W).
                Must have same spatial dims. T <= n_epochs_max.

        Returns:
            Temporally fused features, shape (B, C, H, W).
        """
        n_epochs = len(epoch_features)
        assert n_epochs <= self.n_epochs_max, (
            f"Got {n_epochs} epochs, max is {self.n_epochs_max}"
        )

        B, C, H, W = epoch_features[0].shape

        # Stack epochs: (B, T, C, H, W)
        x = torch.stack(epoch_features, dim=1)

        # Reshape: treat each pixel as independent temporal sequence
        # (B, T, C, H, W) → (B*H*W, T, C)
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, n_epochs, C)

        # Add temporal position embeddings
        x = x + self.temporal_pos[:, :n_epochs, :]

        # Temporal SSM: model epoch-to-epoch state transitions
        x = self.temporal_ssm(x)

        # Take last temporal step (carries full temporal context via SSM state)
        x = x[:, -1, :]  # (B*H*W, C)

        # Project and reshape back to spatial
        x = self.out_proj(x)
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
