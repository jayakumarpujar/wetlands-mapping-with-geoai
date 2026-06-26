"""Baseline model wrappers for wetland segmentation benchmarking.

All baselines share the same interface:
    forward(x, depression_depth=None) → (B, num_classes, H, W) logits

Models implemented:
    1. SegFormerBaseline — SegFormer-B2 (Xie et al. 2021)
    2. SwinUNetBaseline — Swin-UNet (Cao et al. 2022)
    3. UNetMambaBaseline — UNet with Mamba decoder (Chen et al. 2024)
    4. PrithviLinearBaseline — Prithvi encoder + linear segmentation head
    5. SMPBaseline — Any architecture from segmentation-models-pytorch

These wrap external implementations into a consistent API for fair comparison.
depression_depth is accepted but ignored by baselines (only WetMamba uses it).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SMPBaseline(nn.Module):
    """Wrapper for segmentation-models-pytorch architectures.

    Covers: U-Net++, DeepLabV3+, FPN, PSPNet, LinkNet, etc.
    Uses ResNet-50 encoder by default (matching current repo implementation).

    Args:
        arch: Architecture name (e.g., "unetplusplus", "deeplabv3plus").
        encoder_name: Backbone encoder. Defaults to "resnet50".
        input_channels: Number of input bands. Defaults to 10.
        num_classes: Output classes. Defaults to 3.
        encoder_weights: Pretrained weights. Defaults to "imagenet".
    """

    def __init__(
        self,
        arch: str = "unetplusplus",
        encoder_name: str = "resnet50",
        input_channels: int = 7,
        num_classes: int = 3,
        encoder_weights: str = "imagenet",
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        arch_map = {
            "unet": smp.Unet,
            "unetplusplus": smp.UnetPlusPlus,
            "deeplabv3": smp.DeepLabV3,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "pspnet": smp.PSPNet,
            "linknet": smp.Linknet,
            "manet": smp.MAnet,
            "pan": smp.PAN,
        }

        model_cls = arch_map.get(arch)
        if model_cls is None:
            raise ValueError(f"Unknown SMP arch '{arch}'. Choose from: {list(arch_map)}")

        self.model = model_cls(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=input_channels,
            classes=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        depression_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass. depression_depth ignored (API compatibility)."""
        return self.model(x)


class SegFormerBaseline(nn.Module):
    """SegFormer-B2 baseline for semantic segmentation.

    Uses timm for the Mix-ViT encoder and a lightweight MLP decoder.
    If timm is unavailable, falls back to a simple CNN.

    Args:
        input_channels: Number of input bands. Defaults to 10.
        num_classes: Output classes. Defaults to 3.
        variant: SegFormer variant (b0-b5). Defaults to "b2".
        allow_proxy: Fall back to CNN if timm unavailable. False = fail loud.
    """

    def __init__(
        self,
        input_channels: int = 7,
        num_classes: int = 3,
        variant: str = "b2",
        allow_proxy: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        try:
            self._init_segformer(variant)
        except (ImportError, OSError) as e:
            if not allow_proxy:
                raise RuntimeError(
                    f"SegFormer init failed ({e}) and allow_proxy=False"
                ) from e
            self._init_fallback()

    def _init_segformer(self, variant: str) -> None:
        """Initialize SegFormer from timm."""
        import timm

        encoder_name = f"mit_{variant}"
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=True,
            in_chans=self.input_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # MLP decoder head (following SegFormer paper)
        encoder_channels = self.encoder.feature_info.channels()
        embed_dim = 256

        self.linear_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for ch in encoder_channels
        ])

        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim * len(encoder_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, self.num_classes, 1),
        )
        self._use_timm = True

    def _init_fallback(self) -> None:
        """Simple CNN fallback when timm unavailable."""
        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, self.num_classes, 1),
        )
        self._use_timm = False

    def forward(
        self,
        x: torch.Tensor,
        depression_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[-2:]

        if not self._use_timm:
            return self.model(x)

        features = self.encoder(x)
        target_size = features[0].shape[-2:]

        fused = []
        for i, (feat, linear) in enumerate(zip(features, self.linear_fuse)):
            feat = linear(feat)
            feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            fused.append(feat)

        out = self.seg_head(torch.cat(fused, dim=1))
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


class SwinUNetBaseline(nn.Module):
    """Swin-UNet baseline using Swin Transformer encoder + CNN decoder.

    Uses timm Swin-T backbone with a simple UNet-style decoder.

    Args:
        input_channels: Number of input bands. Defaults to 10.
        num_classes: Output classes. Defaults to 3.
        allow_proxy: Fall back to CNN if timm unavailable. False = fail loud.
    """

    def __init__(
        self,
        input_channels: int = 7,
        num_classes: int = 3,
        allow_proxy: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        try:
            import timm
            self.encoder = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=True,
                in_chans=input_channels,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            enc_channels = self.encoder.feature_info.channels()
            self._use_timm = True
        except (ImportError, OSError) as e:
            if not allow_proxy:
                raise RuntimeError(
                    f"Swin-UNet init failed ({e}) and allow_proxy=False"
                ) from e
            enc_channels = [96, 192, 384, 768]
            self._build_fallback_encoder(input_channels, enc_channels)
            self._use_timm = False

        # UNet decoder (CNN-based, standard)
        self.decoder_blocks = nn.ModuleList()
        dec_channels = list(reversed(enc_channels))

        for i in range(len(dec_channels) - 1):
            in_ch = dec_channels[i] + dec_channels[i + 1]  # concat with skip
            out_ch = dec_channels[i + 1]
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

        self.seg_head = nn.Conv2d(dec_channels[-1], num_classes, 1)

    def _build_fallback_encoder(
        self, input_channels: int, channels: List[int]
    ) -> None:
        """CNN fallback encoder matching Swin output shapes."""
        stages = []
        in_ch = input_channels
        for out_ch in channels:
            stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.fallback_stages = nn.ModuleList(stages)

    def forward(
        self,
        x: torch.Tensor,
        depression_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[-2:]

        if self._use_timm:
            features = self.encoder(x)
        else:
            features = []
            h = x
            for stage in self.fallback_stages:
                h = stage(h)
                features.append(h)

        # Decode: coarse → fine
        features_rev = list(reversed(features))
        x_dec = features_rev[0]

        for i, block in enumerate(self.decoder_blocks):
            skip = features_rev[i + 1]
            x_dec = F.interpolate(x_dec, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x_dec = torch.cat([x_dec, skip], dim=1)
            x_dec = block(x_dec)

        out = self.seg_head(x_dec)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


class UNetMambaBaseline(nn.Module):
    """UNetMamba: CNN encoder + Mamba SSM decoder (Chen et al. 2024).

    Uses ResNet-50 CNN encoder (NOT a foundation model) with Mamba decoder.
    This isolates the contribution of the Mamba decoder from FM pretraining.

    Args:
        input_channels: Number of input bands. Defaults to 10.
        num_classes: Output classes. Defaults to 3.
        encoder_name: CNN backbone. Defaults to "resnet50".
        allow_proxy: Fall back to CNN if timm unavailable. False = fail loud.
    """

    def __init__(
        self,
        input_channels: int = 7,
        num_classes: int = 3,
        encoder_name: str = "resnet50",
        allow_proxy: bool = True,
    ) -> None:
        super().__init__()

        try:
            import timm
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=True,
                in_chans=input_channels,
                features_only=True,
                out_indices=(1, 2, 3, 4),
            )
            enc_channels = self.encoder.feature_info.channels()
            self._use_timm = True
        except (ImportError, OSError) as e:
            if not allow_proxy:
                raise RuntimeError(
                    f"UNetMamba init failed ({e}) and allow_proxy=False"
                ) from e
            enc_channels = [64, 128, 256, 512]
            self._build_fallback_encoder(input_channels, enc_channels)
            self._use_timm = False

        from research_paper.models.mamba_decoder import MambaDecoder

        self.decoder = MambaDecoder(
            encoder_channels=list(enc_channels),
            decoder_channels=[256, 128, 64, 32],
            d_state=16,
            n_ssm_blocks=2,
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def _build_fallback_encoder(
        self, input_channels: int, channels: List[int]
    ) -> None:
        """CNN fallback encoder."""
        stages = []
        in_ch = input_channels
        for out_ch in channels:
            stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.fallback_stages = nn.ModuleList(stages)
        self._use_timm = False

    def forward(
        self,
        x: torch.Tensor,
        depression_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[-2:]

        if self._use_timm:
            features = list(self.encoder(x))
        else:
            features = []
            h = x
            for stage in self.fallback_stages:
                h = stage(h)
                features.append(h)

        decoded = self.decoder(features)
        out = self.seg_head(decoded)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


class PrithviLinearBaseline(nn.Module):
    """Prithvi encoder + simple linear segmentation head.

    Tests the value of the Prithvi FM alone without Mamba decoder or DAG.
    Features are extracted at multiple scales, upsampled, concatenated,
    and classified with a single 1x1 conv.

    Args:
        input_channels: Number of input bands. Defaults to 10.
        num_classes: Output classes. Defaults to 3.
        encoder_name: Prithvi model ID.
        allow_proxy: Fall back to CNN if Prithvi unavailable. False = fail loud.
    """

    def __init__(
        self,
        input_channels: int = 7,
        num_classes: int = 3,
        encoder_name: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        allow_proxy: bool = True,
    ) -> None:
        super().__init__()
        from research_paper.models.wetmamba import PrithviEncoder

        self.encoder = PrithviEncoder(
            model_name=encoder_name,
            input_channels=input_channels,
            use_pretrained=True,
            use_lora=False,
            allow_proxy=allow_proxy,
        )

        total_channels = sum(self.encoder.feature_channels)
        self.seg_head = nn.Sequential(
            nn.Conv2d(total_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        depression_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        input_size = x.shape[-2:]
        features = self.encoder(x)

        # Upsample all scales to finest and concatenate
        target_size = features[0].shape[-2:]
        upsampled = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            upsampled.append(feat)

        out = self.seg_head(torch.cat(upsampled, dim=1))
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


def build_model(
    name: str,
    num_classes: int = 3,
    input_channels: int = 7,
    allow_proxy: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to build any benchmark model by name.

    Args:
        name: Model identifier. One of:
            "wetmamba", "unetplusplus", "deeplabv3plus", "segformer",
            "swin_unet", "unetmamba", "prithvi_linear", or any SMP arch.
        num_classes: Output classes.
        input_channels: Input bands per epoch.
        allow_proxy: Allow CNN proxy when FM weights unavailable.
            Set False for benchmark runs to fail loudly instead of
            silently degrading to a CNN.
        **kwargs: Additional model-specific arguments.

    Returns:
        Initialized model.
    """
    from research_paper.models.wetmamba import WetMamba

    name = name.lower().replace("-", "_")

    if name == "wetmamba":
        return WetMamba(
            num_classes=num_classes,
            input_channels=input_channels,
            allow_proxy=allow_proxy,
            **kwargs,
        )

    if name == "segformer":
        return SegFormerBaseline(
            input_channels=input_channels,
            num_classes=num_classes,
            allow_proxy=allow_proxy,
            **kwargs,
        )

    if name == "swin_unet":
        return SwinUNetBaseline(
            input_channels=input_channels,
            num_classes=num_classes,
            allow_proxy=allow_proxy,
            **kwargs,
        )

    if name == "unetmamba":
        return UNetMambaBaseline(
            input_channels=input_channels,
            num_classes=num_classes,
            allow_proxy=allow_proxy,
            **kwargs,
        )

    if name == "prithvi_linear":
        return PrithviLinearBaseline(
            input_channels=input_channels,
            num_classes=num_classes,
            allow_proxy=allow_proxy,
            **kwargs,
        )

    # Default: SMP baseline (no proxy concept — always real)
    return SMPBaseline(
        arch=name,
        input_channels=input_channels,
        num_classes=num_classes,
        **kwargs,
    )
