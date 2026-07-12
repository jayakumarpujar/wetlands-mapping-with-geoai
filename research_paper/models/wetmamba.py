"""WetMamba: Depression-Aware Multi-Temporal State Space Model for Wetland Segmentation.

Main architecture combining:
    1. Prithvi-EO-2.0 encoder (ViT, LoRA fine-tuned) — geospatial FM features
    2. Mamba SSM decoder — O(n) complexity for high-res processing
    3. Depression-Aware Gating — geomorphological physics prior
    4. Multi-Temporal SSM fusion — wetland phenology modeling

Input: Multi-epoch NAIP+LiDAR composites (10 bands per epoch × T epochs)
Output: 3-class Cowardin wetland segmentation map (Upland/Water/Emergent)

Architecture flow:
    For each epoch:
        composite (10-band) → Prithvi encoder → multi-scale features
    → TemporalSSMFusion at each scale → fused multi-scale features
    → MambaDecoder with skip connections → decoded features
    → DAG module (with depression depth) → gated features
    → segmentation head → 3-class prediction

References:
    - Jakubik et al. (2024): Prithvi-EO-2.0
    - Gu & Dao (2024): Mamba SSM
    - Wu et al. (2019): PPR wetland mapping
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from research_paper.models.dag_module import DepressionAwareGating
from research_paper.models.mamba_decoder import MambaDecoder
from research_paper.models.temporal_ssm import TemporalSSMFusion


class PrithviEncoder(nn.Module):
    """Wrapper for Prithvi-EO-2.0 encoder with multi-scale feature extraction.

    Prithvi is a ViT-based foundation model. We extract features at multiple
    depths to create a multi-scale feature pyramid for the decoder.

    When Prithvi weights are unavailable (e.g., testing), falls back to a
    simple CNN encoder with matching output shapes.

    Args:
        model_name: HuggingFace model ID. Defaults to Prithvi-300M.
        input_channels: Number of input bands. Defaults to 10 (NAIP 4 + indices 2 + LiDAR 4).
        use_pretrained: Load pretrained weights. Defaults to True.
        use_lora: Apply LoRA adapters. Defaults to True.
        lora_rank: LoRA rank. Defaults to 8.
        feature_channels: Output channels at each scale.
            Defaults to [64, 128, 320, 512] matching Prithvi-300M.
    """

    # Default multi-scale channels for Prithvi-300M ViT
    DEFAULT_CHANNELS = [64, 128, 320, 512]

    PRITHVI_IN_CHANS = 6
    PRITHVI_PATCH_SIZE = 16
    PRITHVI_HIDDEN_DIM = 768
    PRITHVI_NUM_LAYERS = 12

    def __init__(
        self,
        model_name: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        input_channels: int = 7,
        use_pretrained: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
        feature_channels: Optional[List[int]] = None,
        allow_proxy: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.feature_channels = feature_channels or self.DEFAULT_CHANNELS
        self.input_channels = input_channels
        self._use_pretrained = use_pretrained
        self._use_lora = use_lora
        self._lora_rank = lora_rank

        self._prithvi_loaded = False
        try:
            self._init_prithvi()
        except (ImportError, OSError, TypeError, ValueError) as e:
            # TypeError: Prithvi config fields are None (no internet / no cache)
            # ValueError: config validation failures
            import logging
            logging.getLogger(__name__).warning(
                "Prithvi unavailable (%s), using CNN proxy encoder", e
            )
            if not allow_proxy:
                raise RuntimeError(
                    f"Prithvi failed to load ({e}) and allow_proxy=False. "
                    "Ensure HF weights are cached or the node has internet."
                ) from e
            self._init_cnn_proxy()

    @staticmethod
    def _random_init_weights(m: nn.Module) -> None:
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        elif isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _init_prithvi(self) -> None:
        """Initialize Prithvi-EO-2.0 from HuggingFace cache.

        Strategy: Prithvi stores weights as Prithvi_EO_V2_300M.pt (non-standard).
        AutoModel.from_pretrained can't find them. We instead:
          1. snapshot_download → get local cache path
          2. import prithvi_mae.py → registers prithvi_eo_v2_300 with timm
          3. AutoModel.from_config → build arch (timm now knows it)
          4. torch.load Prithvi_EO_V2_300M.pt → inject weights manually
        """
        import importlib
        import importlib.util
        import sys
        from pathlib import Path

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModel

        # Pass cache_dir explicitly so path matches however the cache was populated.
        # HF_HOME=$X puts hub cache at $X/hub/, but snapshot_download cache_dir=$X
        # puts it at $X/ directly. We support both: try HF_HOME/hub first, fall
        # back to HF_HOME itself.
        import os
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        for _cache_dir in [hf_home, str(Path(hf_home) / "hub")]:
            try:
                snapshot_dir = snapshot_download(
                    repo_id=self.model_name,
                    cache_dir=_cache_dir,
                    local_files_only=True,
                )
                if (Path(snapshot_dir) / "prithvi_mae.py").exists():
                    break
            except Exception:
                continue
        else:
            raise FileNotFoundError(
                f"prithvi_mae.py not found in HF cache under {hf_home}. "
                "Re-run cache_prithvi.sh on the login node."
            )

        # Import prithvi_mae.py to register prithvi_eo_v2_300 with timm.
        # Must: (1) add snapshot_dir to sys.path so prithvi_mae.py's own relative
        # imports resolve, (2) register in sys.modules BEFORE exec_module so any
        # circular import or trust_remote_code lookup by name succeeds.
        snapshot_str = str(snapshot_dir)
        if snapshot_str not in sys.path:
            sys.path.insert(0, snapshot_str)

        prithvi_mae_file = Path(snapshot_dir) / "prithvi_mae.py"
        _spec = importlib.util.spec_from_file_location("prithvi_mae", prithvi_mae_file)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules.setdefault("prithvi_mae", _mod)
        _spec.loader.exec_module(_mod)

        config = AutoConfig.from_pretrained(
            snapshot_dir,
            trust_remote_code=True,
            num_frames=1,
            num_labels=3,
            local_files_only=True,
        )

        # Build architecture (timm arch now registered via prithvi_mae import)
        self.prithvi = AutoModel.from_config(config, trust_remote_code=True)

        if self._use_pretrained:
            weights_path = Path(snapshot_dir) / "Prithvi_EO_V2_300M.pt"
            state_dict = torch.load(weights_path, map_location="cpu")
            # Checkpoint may wrap weights under 'model' key
            if isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]
            missing, unexpected = self.prithvi.load_state_dict(
                state_dict, strict=False
            )
            import logging
            _log = logging.getLogger(__name__)
            if missing:
                _log.warning("Prithvi: %d missing keys (encoder-only is normal)", len(missing))
            if unexpected:
                _log.warning("Prithvi: %d unexpected keys", len(unexpected))
        else:
            # no_pretrained ablation: architecture with random weights
            self.prithvi.apply(self._random_init_weights)

        # Adapt input: our composite has input_channels bands → project to 6 HLS
        self.input_adapter = nn.Conv2d(
            self.input_channels, self.PRITHVI_IN_CHANS,
            kernel_size=1, bias=False,
        )

        # Multi-scale feature extraction via intermediate ViT layers
        # Prithvi-300M has 12 transformer blocks; extract at layers 3, 6, 9, 12
        self.extract_layers = [3, 6, 9, 12]

        vit_dim = getattr(
            self.prithvi.config, "hidden_size", self.PRITHVI_HIDDEN_DIM
        )

        # All ViT layers output at the same patch-grid spatial resolution
        # (H/patch_size, W/patch_size). We create synthetic multi-scale features
        # by progressively strided-conv-downsampling the patch features.
        self.scale_projections = nn.ModuleList()
        for i, ch in enumerate(self.feature_channels):
            if i == 0:
                proj = nn.Sequential(
                    nn.Conv2d(vit_dim, ch, 1, bias=False),
                    nn.BatchNorm2d(ch),
                )
            else:
                proj = nn.Sequential(
                    nn.Conv2d(vit_dim, ch, 3, stride=2 ** i, padding=1,
                              bias=False),
                    nn.BatchNorm2d(ch),
                )
            self.scale_projections.append(proj)

        if self._use_lora:
            self._apply_lora()

        if self._use_pretrained:
            for name, param in self.prithvi.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False

        self._prithvi_loaded = True

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to Prithvi attention layers."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self._lora_rank,
                lora_alpha=self._lora_rank * 2,
                target_modules=["qkv"],
                lora_dropout=0.05,
                bias="none",
            )
            self.prithvi = get_peft_model(self.prithvi, lora_config)
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "peft not installed, skipping LoRA. Install with: pip install peft"
            )

    def _init_cnn_proxy(self) -> None:
        """Lightweight CNN encoder matching Prithvi output shapes for testing."""
        layers = []
        in_ch = self.input_channels
        for out_ch in self.feature_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.proxy_stages = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input composite, shape (B, C_in, H, W).

        Returns:
            List of feature maps at 4 scales, finest to coarsest:
            [(B, 64, H/4, W/4), (B, 128, H/8, W/8),
             (B, 320, H/16, W/16), (B, 512, H/32, W/32)]
        """
        if self._prithvi_loaded:
            return self._forward_prithvi(x)
        return self._forward_proxy(x)

    def _forward_prithvi(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward through Prithvi with multi-scale extraction.

        Prithvi-EO-2.0 expects (B, C=6, T, H, W). All transformer layers
        output the same spatial size (H/patch, W/patch). We create multi-scale
        features by projecting different layers with progressive stride.
        """
        B, _, H, W = x.shape
        ps = self.PRITHVI_PATCH_SIZE
        h_p, w_p = H // ps, W // ps

        # Adapt input channels: (B, in_ch, H, W) → (B, 6, H, W)
        x = self.input_adapter(x)

        # Prithvi expects 5D: (B, C, T, H, W) with T=1
        x = x.unsqueeze(2)

        outputs = self.prithvi(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        features = []
        for i, layer_idx in enumerate(self.extract_layers):
            # ViT hidden states: (B, num_patches, hidden_dim)
            feat = hidden_states[layer_idx]

            # Some ViTs prepend a CLS token; remove if present
            expected_patches = h_p * w_p
            if feat.shape[1] > expected_patches:
                feat = feat[:, -expected_patches:, :]

            # Reshape to spatial grid: (B, h_p, w_p, D) → (B, D, h_p, w_p)
            feat = feat.reshape(B, h_p, w_p, -1).permute(0, 3, 1, 2)

            # Project to target channels (with progressive downsampling)
            feat = self.scale_projections[i](feat)
            features.append(feat)

        return features

    def _forward_proxy(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward through CNN proxy encoder."""
        features = []
        for stage in self.proxy_stages:
            x = stage(x)
            features.append(x)
        return features


class WetMamba(nn.Module):
    """WetMamba: Depression-Aware Multi-Temporal SSM for Wetland Segmentation.

    End-to-end architecture:
        multi-epoch composites + depression depth
        → per-epoch Prithvi encoding
        → temporal SSM fusion at each scale
        → Mamba decoder with skip connections
        → depression-aware gating
        → segmentation head

    Args:
        num_classes: Number of output classes. Defaults to 3 (Upland/Water/Emergent).
        input_channels: Bands per epoch. Defaults to 10.
        encoder_name: Prithvi model ID.
        encoder_channels: Feature channels at each encoder scale.
        decoder_channels: Feature channels at each decoder scale.
        use_pretrained: Use pretrained Prithvi. Defaults to True.
        use_lora: Apply LoRA to Prithvi. Defaults to True.
        lora_rank: LoRA rank. Defaults to 8.
        use_dag: Enable Depression-Aware Gating. Defaults to True.
        use_temporal: Enable temporal SSM fusion. Defaults to True.
        n_epochs_max: Maximum temporal epochs. Defaults to 6.
        d_state: SSM state dimension. Defaults to 16.
        allow_proxy: If True, fall back to CNN proxy when Prithvi
            weights unavailable. Set False for benchmark runs to fail loudly.
    """

    def __init__(
        self,
        num_classes: int = 3,
        input_channels: int = 7,
        encoder_name: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        encoder_channels: Optional[List[int]] = None,
        decoder_channels: Optional[List[int]] = None,
        use_pretrained: bool = True,
        use_lora: bool = True,
        lora_rank: int = 8,
        use_dag: bool = True,
        use_temporal: bool = True,
        n_epochs_max: int = 6,
        d_state: int = 16,
        allow_proxy: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_dag = use_dag
        self.use_temporal = use_temporal

        enc_ch = encoder_channels or PrithviEncoder.DEFAULT_CHANNELS
        dec_ch = decoder_channels or [256, 128, 64, 32]

        # Encoder: shared across epochs
        self.encoder = PrithviEncoder(
            model_name=encoder_name,
            input_channels=input_channels,
            use_pretrained=use_pretrained,
            use_lora=use_lora,
            lora_rank=lora_rank,
            feature_channels=enc_ch,
            allow_proxy=allow_proxy,
        )

        # Temporal SSM fusion at each encoder scale
        if use_temporal:
            self.temporal_fusions = nn.ModuleList([
                TemporalSSMFusion(channels=ch, d_state=d_state, n_epochs_max=n_epochs_max)
                for ch in enc_ch
            ])

        # Mamba decoder
        self.decoder = MambaDecoder(
            encoder_channels=enc_ch,
            decoder_channels=dec_ch,
            d_state=d_state,
        )

        # Depression-Aware Gating on decoder output
        if use_dag:
            self.dag = DepressionAwareGating(
                in_channels=dec_ch[-1],
                depression_channels=32,
            )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(dec_ch[-1], dec_ch[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch[-1], num_classes, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        depression_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Either:
                - Single epoch: (B, C, H, W) where C = input_channels
                - Multi-epoch: (B, T*C, H, W) or (B, T, C, H, W)
            depression_depth: LiDAR depression depth, shape (B, 1, H, W).
                Required if use_dag=True.

        Returns:
            Logits, shape (B, num_classes, H, W).
        """
        input_h, input_w = x.shape[-2:]

        # Parse multi-epoch input
        epoch_inputs = self._parse_epochs(x)

        # Encode each epoch independently (shared encoder)
        all_epoch_features: List[List[torch.Tensor]] = []
        for epoch_x in epoch_inputs:
            features = self.encoder(epoch_x)
            all_epoch_features.append(features)

        # Temporal fusion at each scale
        if self.use_temporal and len(epoch_inputs) > 1:
            n_scales = len(all_epoch_features[0])
            fused_features = []
            for s in range(n_scales):
                scale_epochs = [ef[s] for ef in all_epoch_features]
                fused = self.temporal_fusions[s](scale_epochs)
                fused_features.append(fused)
        else:
            # Single epoch: use features directly
            fused_features = all_epoch_features[0]

        # Decode
        decoded = self.decoder(fused_features)

        # Depression-Aware Gating
        if self.use_dag and depression_depth is not None:
            decoded = self.dag(decoded, depression_depth)

        # Segmentation head
        logits = self.seg_head(decoded)

        # Upsample to input resolution
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w),
                mode="bilinear", align_corners=False,
            )

        return logits

    def _parse_epochs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Parse input into per-epoch tensors.

        Handles three input formats:
            - 5D (B, T, C, H, W): explicit multi-epoch → split on dim 1
            - 4D (B, C, H, W) where C == input_channels: single epoch
            - 4D (B, C, H, W) where C > input_channels: split along channels

        For the PPR composite layout (e.g., 13 bands = 2×6 epoch bands + 1 DEM):
            Per-epoch bands (6 each) are split, and shared bands (DEM) are
            appended to each epoch so every epoch gets input_channels bands.
        """
        in_ch = self.encoder.input_channels

        if x.dim() == 5:
            # (B, T, C, H, W) → list of (B, C, H, W)
            return [x[:, t] for t in range(x.shape[1])]

        if x.dim() == 4:
            total_ch = x.shape[1]

            if total_ch == in_ch:
                return [x]

            if total_ch % in_ch == 0:
                n_epochs = total_ch // in_ch
                return [x[:, i * in_ch:(i + 1) * in_ch] for i in range(n_epochs)]

            # Handle shared bands (e.g., DEM): split epoch bands, append shared
            # Assume per-epoch bands come first, shared bands at end
            bands_per_epoch = 6  # NAIP(4) + NDVI + NDWI
            n_epoch_bands = total_ch - (total_ch % bands_per_epoch) if total_ch % bands_per_epoch != 0 else total_ch
            n_shared = total_ch - n_epoch_bands

            if n_epoch_bands > 0 and n_epoch_bands % bands_per_epoch == 0:
                n_epochs = n_epoch_bands // bands_per_epoch
                shared = x[:, n_epoch_bands:, :, :]  # (B, n_shared, H, W)
                epochs = []
                for i in range(n_epochs):
                    epoch_x = x[:, i * bands_per_epoch:(i + 1) * bands_per_epoch]
                    if n_shared > 0:
                        epoch_x = torch.cat([epoch_x, shared], dim=1)
                    epochs.append(epoch_x)
                return epochs

        return [x]

    def get_param_groups(self) -> List[Dict[str, Any]]:
        """Parameter groups for differential learning rates.

        Returns:
            List of param group dicts with 'params' and 'lr_scale' keys.
            Encoder (frozen/LoRA) gets lower lr, decoder/DAG get full lr.
        """
        encoder_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("encoder"):
                encoder_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": encoder_params, "lr_scale": 0.1},
            {"params": other_params, "lr_scale": 1.0},
        ]
