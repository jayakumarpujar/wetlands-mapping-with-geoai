"""Mamba SSM decoder blocks for semantic segmentation.

Implements a UNet-style decoder using Mamba (Selective State Space Model)
blocks instead of convolutional or transformer blocks. Key advantage:
O(n) sequence complexity vs O(n^2) for transformers, enabling processing
of high-resolution (1m) remote sensing imagery.

The decoder receives multi-scale features from the Prithvi encoder and
progressively upsamples them through Mamba blocks at each scale.

Scanning strategy: bi-directional raster scan (left→right + right→left)
following RS3Mamba convention. This captures spatial dependencies in
both directions along the flattened 2D sequence.

References:
    - Gu & Dao (2024): Mamba — selective state space model
    - Ma et al. (2024): RS3Mamba — Mamba for RS segmentation
    - Zhu et al. (2024): Vision Mamba (Vim)
    - Chen et al. (2024): UNetMamba — Mamba in UNet decoder
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMBlock(nn.Module):
    """Simplified Selective State Space Model block.

    This is a pure-PyTorch implementation that mirrors the Mamba SSM
    computation without requiring the CUDA-optimized mamba-ssm package.
    When mamba-ssm is available, WetMamba swaps this for the optimized
    version via the `use_native_mamba` flag.

    The block operates on flattened 2D feature maps treated as 1D sequences.

    Args:
        dim: Feature dimension (number of channels).
        d_state: SSM state expansion factor. Defaults to 16.
        d_conv: Local convolution width. Defaults to 4.
        expand: Channel expansion factor for inner projection. Defaults to 2.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        inner_dim = dim * expand

        # Input projection: split into two paths (gated architecture)
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)

        # Local convolution for position-aware features
        self.conv1d = nn.Conv1d(
            inner_dim, inner_dim,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=inner_dim, bias=True,
        )

        # SSM parameters (selective — input-dependent)
        self.dt_proj = nn.Linear(inner_dim, inner_dim, bias=True)
        self.A_log = nn.Parameter(torch.randn(inner_dim, d_state))
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.B_proj = nn.Linear(inner_dim, d_state, bias=False)
        self.C_proj = nn.Linear(inner_dim, d_state, bias=False)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

        self.norm = nn.LayerNorm(dim)

    def _ssm_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan (sequential, for correctness — optimized ver uses CUDA).

        Args:
            x: Input, shape (B, L, D_inner).
            dt: Time deltas, shape (B, L, D_inner).
            A: State matrix, shape (D_inner, N).
            B: Input matrix, shape (B, L, N).
            C: Output matrix, shape (B, L, N).
            D: Skip connection, shape (D_inner,).

        Returns:
            Output, shape (B, L, D_inner).
        """
        batch, seq_len, d_inner = x.shape
        n = A.shape[1]

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        dt = F.softplus(dt)  # (B, L, D_inner)
        A = -torch.exp(A.float())  # (D_inner, N) — negative for stability

        # Efficient parallel scan approximation for training
        # Full sequential scan for reference; in practice use associative scan
        outputs = []
        h = torch.zeros(batch, d_inner, n, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)  # (B, D_inner, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))  # (B, D_inner, N)
            B_bar = dt_t * B[:, t, :].unsqueeze(1)  # (B, D_inner, N)
            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)  # (B, D_inner, N)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, D_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D_inner)
        return y + x * D.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor, shape (B, L, D).

        Returns:
            Output tensor, shape (B, L, D).
        """
        residual = x
        x = self.norm(x)

        # Gated projection
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv1d on the x branch
        x_branch = x_branch.transpose(1, 2)  # (B, D_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]  # trim causal padding
        x_branch = x_branch.transpose(1, 2)  # (B, L, D_inner)
        x_branch = F.silu(x_branch)

        # SSM parameters (input-dependent / selective)
        dt = self.dt_proj(x_branch)
        B = self.B_proj(x_branch)
        C = self.C_proj(x_branch)

        # Selective scan
        y = self._ssm_scan(x_branch, dt, self.A_log, B, C, self.D)

        # Gate and project out
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y + residual


class MambaDecoderBlock(nn.Module):
    """Single decoder stage: upsample + fuse skip + bi-directional Mamba SSM.

    Processes features as bi-directional 1D sequences (flattened 2D spatial).
    Forward scan captures left→right dependencies; backward scan captures
    right→left. Outputs are summed for symmetric spatial context.

    Args:
        in_channels: Channels from lower (deeper) decoder stage.
        skip_channels: Channels from encoder skip connection.
        out_channels: Output channels for this stage.
        d_state: SSM state dimension. Defaults to 16.
        n_ssm_blocks: Number of stacked SSM blocks. Defaults to 2.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        d_state: int = 16,
        n_ssm_blocks: int = 2,
    ) -> None:
        super().__init__()

        # Fuse upsampled features with skip connection
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Bi-directional Mamba SSM blocks
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(dim=out_channels, d_state=d_state)
            for _ in range(n_ssm_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode one stage.

        Args:
            x: Features from previous (deeper) stage, shape (B, C_in, H, W).
            skip: Encoder skip features, shape (B, C_skip, H_skip, W_skip).
                If provided, x is upsampled to match skip's spatial dims.

        Returns:
            Decoded features, shape (B, C_out, H_out, W_out).
        """
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.fuse_conv(x)

        B, C, H, W = x.shape

        # Flatten 2D → 1D sequence for SSM
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Forward scan
        x_fwd = x_seq
        for blk in self.ssm_blocks:
            x_fwd = blk(x_fwd)

        # Backward scan (reverse sequence)
        x_bwd = x_seq.flip(dims=[1])
        for blk in self.ssm_blocks:
            x_bwd = blk(x_bwd)
        x_bwd = x_bwd.flip(dims=[1])

        # Merge bi-directional
        x_seq = x_fwd + x_bwd

        # Reshape back to 2D
        return x_seq.transpose(1, 2).reshape(B, C, H, W)


class MambaDecoder(nn.Module):
    """Full Mamba SSM decoder with multi-scale skip connections.

    Takes encoder features at 4 scales (1/4, 1/8, 1/16, 1/32 of input)
    and progressively decodes through MambaDecoderBlocks.

    Args:
        encoder_channels: List of encoder channel counts from finest to
            coarsest scale. E.g., [64, 128, 320, 512] for Prithvi-300M.
        decoder_channels: List of decoder channel counts at each stage.
            Defaults to [256, 128, 64, 32].
        d_state: SSM state dimension. Defaults to 16.
        n_ssm_blocks: SSM blocks per decoder stage. Defaults to 2.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: Optional[List[int]] = None,
        d_state: int = 16,
        n_ssm_blocks: int = 2,
    ) -> None:
        super().__init__()
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        assert len(encoder_channels) >= 2, "Need at least 2 encoder scales"

        self.stages = nn.ModuleList()

        # First stage: deepest encoder features (no skip from deeper stage)
        # Process the deepest features through a bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], decoder_channels[0], 1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Decoder stages: from deepest to shallowest
        n_stages = min(len(encoder_channels) - 1, len(decoder_channels) - 1)
        for i in range(n_stages):
            in_ch = decoder_channels[i]
            skip_ch = encoder_channels[-(i + 2)]  # encoder features, coarse→fine
            out_ch = decoder_channels[i + 1]
            self.stages.append(
                MambaDecoderBlock(in_ch, skip_ch, out_ch, d_state, n_ssm_blocks)
            )

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale encoder features.

        Args:
            encoder_features: List of feature maps from finest to coarsest,
                e.g., [feat_1/4, feat_1/8, feat_1/16, feat_1/32].

        Returns:
            Decoded features at the finest encoder scale.
        """
        x = self.bottleneck(encoder_features[-1])

        # Progressive decoding from coarse to fine
        for i, stage in enumerate(self.stages):
            skip_idx = -(i + 2)
            skip = encoder_features[skip_idx] if abs(skip_idx) <= len(encoder_features) else None
            x = stage(x, skip)

        return x
