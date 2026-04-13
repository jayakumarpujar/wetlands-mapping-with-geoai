"""Tests for WetMamba model components.

Tests forward pass, output shapes, gradient flow, and ablation configurations
for all model modules: DAG, MambaDecoder, TemporalSSM, WetMamba, baselines.
"""

import pytest
import torch
import torch.nn as nn

# Test constants
BATCH = 2
CHANNELS = 32
HEIGHT = 64
WIDTH = 64
NUM_CLASSES = 6
INPUT_CHANNELS = 7


# ---------------------------------------------------------------------------
# Depression-Aware Gating (DAG) Module
# ---------------------------------------------------------------------------


class TestDepressionAwareGating:
    """Tests for the DAG module."""

    def _make_dag(self, in_channels: int = CHANNELS) -> nn.Module:
        from research_paper.models.dag_module import DepressionAwareGating
        return DepressionAwareGating(in_channels=in_channels)

    def test_output_shape(self) -> None:
        dag = self._make_dag()
        features = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
        depth = torch.rand(BATCH, 1, HEIGHT, WIDTH)
        out = dag(features, depth)
        assert out.shape == features.shape

    def test_output_shape_mismatched_spatial(self) -> None:
        """DAG should handle depth map at different resolution."""
        dag = self._make_dag()
        features = torch.randn(BATCH, CHANNELS, HEIGHT, WIDTH)
        depth = torch.rand(BATCH, 1, HEIGHT * 2, WIDTH * 2)
        out = dag(features, depth)
        assert out.shape == features.shape

    def test_gradient_flows(self) -> None:
        dag = self._make_dag()
        features = torch.randn(BATCH, CHANNELS, 16, 16, requires_grad=True)
        depth = torch.rand(BATCH, 1, 16, 16)
        out = dag(features, depth)
        loss = out.sum()
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_zero_depth_passthrough(self) -> None:
        """With zero depression depth, features should still pass through."""
        dag = self._make_dag()
        features = torch.randn(BATCH, CHANNELS, 16, 16)
        depth = torch.zeros(BATCH, 1, 16, 16)
        out = dag(features, depth)
        assert out.shape == features.shape
        # Output shouldn't be all zeros
        assert out.abs().sum() > 0

    def test_different_channel_sizes(self) -> None:
        for ch in [16, 64, 128, 256]:
            dag = self._make_dag(in_channels=ch)
            features = torch.randn(BATCH, ch, 16, 16)
            depth = torch.rand(BATCH, 1, 16, 16)
            out = dag(features, depth)
            assert out.shape == (BATCH, ch, 16, 16)


# ---------------------------------------------------------------------------
# Mamba Decoder
# ---------------------------------------------------------------------------


class TestSSMBlock:
    """Tests for the SSM block."""

    def test_output_shape(self) -> None:
        from research_paper.models.mamba_decoder import SSMBlock
        block = SSMBlock(dim=CHANNELS)
        x = torch.randn(BATCH, 16, CHANNELS)  # (B, L, D)
        out = block(x)
        assert out.shape == x.shape

    def test_gradient_flows(self) -> None:
        from research_paper.models.mamba_decoder import SSMBlock
        block = SSMBlock(dim=CHANNELS)
        x = torch.randn(BATCH, 8, CHANNELS, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


class TestMambaDecoderBlock:
    """Tests for a single decoder stage."""

    def test_with_skip(self) -> None:
        from research_paper.models.mamba_decoder import MambaDecoderBlock
        block = MambaDecoderBlock(in_channels=64, skip_channels=32, out_channels=32)
        x = torch.randn(BATCH, 64, 8, 8)
        skip = torch.randn(BATCH, 32, 16, 16)
        out = block(x, skip)
        assert out.shape == (BATCH, 32, 16, 16)

    def test_without_skip(self) -> None:
        from research_paper.models.mamba_decoder import MambaDecoderBlock
        block = MambaDecoderBlock(in_channels=64, skip_channels=0, out_channels=32)
        x = torch.randn(BATCH, 64, 8, 8)
        out = block(x, skip=None)
        assert out.shape == (BATCH, 32, 8, 8)


class TestMambaDecoder:
    """Tests for the full decoder."""

    def test_output_shape(self) -> None:
        from research_paper.models.mamba_decoder import MambaDecoder
        enc_ch = [64, 128, 320, 512]
        dec_ch = [256, 128, 64, 32]
        decoder = MambaDecoder(encoder_channels=enc_ch, decoder_channels=dec_ch)

        features = [
            torch.randn(BATCH, 64, 16, 16),
            torch.randn(BATCH, 128, 8, 8),
            torch.randn(BATCH, 320, 4, 4),
            torch.randn(BATCH, 512, 2, 2),
        ]
        out = decoder(features)
        assert out.shape == (BATCH, 32, 16, 16)

    def test_gradient_flows(self) -> None:
        from research_paper.models.mamba_decoder import MambaDecoder
        dec = MambaDecoder(encoder_channels=[32, 64], decoder_channels=[64, 32])
        f1 = torch.randn(BATCH, 32, 8, 8, requires_grad=True)
        f2 = torch.randn(BATCH, 64, 4, 4, requires_grad=True)
        out = dec([f1, f2])
        out.sum().backward()
        assert f1.grad is not None
        assert f2.grad is not None


# ---------------------------------------------------------------------------
# Temporal SSM Fusion
# ---------------------------------------------------------------------------


class TestTemporalSSMFusion:
    """Tests for multi-temporal fusion."""

    def test_two_epochs(self) -> None:
        from research_paper.models.temporal_ssm import TemporalSSMFusion
        fusion = TemporalSSMFusion(channels=CHANNELS)
        e1 = torch.randn(BATCH, CHANNELS, 8, 8)
        e2 = torch.randn(BATCH, CHANNELS, 8, 8)
        out = fusion([e1, e2])
        assert out.shape == (BATCH, CHANNELS, 8, 8)

    def test_single_epoch(self) -> None:
        from research_paper.models.temporal_ssm import TemporalSSMFusion
        fusion = TemporalSSMFusion(channels=CHANNELS)
        e1 = torch.randn(BATCH, CHANNELS, 8, 8)
        out = fusion([e1])
        assert out.shape == (BATCH, CHANNELS, 8, 8)

    def test_max_epochs(self) -> None:
        from research_paper.models.temporal_ssm import TemporalSSMFusion
        fusion = TemporalSSMFusion(channels=CHANNELS, n_epochs_max=6)
        epochs = [torch.randn(BATCH, CHANNELS, 4, 4) for _ in range(6)]
        out = fusion(epochs)
        assert out.shape == (BATCH, CHANNELS, 4, 4)

    def test_exceeds_max_epochs_raises(self) -> None:
        from research_paper.models.temporal_ssm import TemporalSSMFusion
        fusion = TemporalSSMFusion(channels=CHANNELS, n_epochs_max=2)
        epochs = [torch.randn(BATCH, CHANNELS, 4, 4) for _ in range(3)]
        with pytest.raises(AssertionError):
            fusion(epochs)

    def test_gradient_flows(self) -> None:
        from research_paper.models.temporal_ssm import TemporalSSMFusion
        fusion = TemporalSSMFusion(channels=CHANNELS)
        e1 = torch.randn(BATCH, CHANNELS, 4, 4, requires_grad=True)
        e2 = torch.randn(BATCH, CHANNELS, 4, 4, requires_grad=True)
        out = fusion([e1, e2])
        out.sum().backward()
        assert e1.grad is not None
        assert e2.grad is not None


# ---------------------------------------------------------------------------
# WetMamba (Full Architecture)
# ---------------------------------------------------------------------------


class TestWetMamba:
    """Tests for the full WetMamba model (uses CNN proxy encoder)."""

    def _make_model(self, **kwargs) -> nn.Module:
        from research_paper.models.wetmamba import WetMamba
        defaults = {
            "num_classes": NUM_CLASSES,
            "input_channels": INPUT_CHANNELS,
            "use_pretrained": False,  # Forces CNN proxy
        }
        defaults.update(kwargs)
        return WetMamba(**defaults)

    def test_single_epoch_forward(self) -> None:
        model = self._make_model()
        x = torch.randn(BATCH, INPUT_CHANNELS, HEIGHT, WIDTH)
        depth = torch.rand(BATCH, 1, HEIGHT, WIDTH)
        out = model(x, depression_depth=depth)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)

    def test_multi_epoch_forward_5d(self) -> None:
        model = self._make_model()
        x = torch.randn(BATCH, 2, INPUT_CHANNELS, HEIGHT, WIDTH)
        depth = torch.rand(BATCH, 1, HEIGHT, WIDTH)
        out = model(x, depression_depth=depth)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)

    def test_multi_epoch_forward_4d_concat(self) -> None:
        model = self._make_model()
        x = torch.randn(BATCH, INPUT_CHANNELS * 2, HEIGHT, WIDTH)
        depth = torch.rand(BATCH, 1, HEIGHT, WIDTH)
        out = model(x, depression_depth=depth)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)

    def test_ablation_no_dag(self) -> None:
        model = self._make_model(use_dag=False)
        x = torch.randn(BATCH, INPUT_CHANNELS, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)

    def test_ablation_no_temporal(self) -> None:
        model = self._make_model(use_temporal=False)
        x = torch.randn(BATCH, INPUT_CHANNELS, HEIGHT, WIDTH)
        depth = torch.rand(BATCH, 1, HEIGHT, WIDTH)
        out = model(x, depression_depth=depth)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)

    def test_ablation_no_dag_no_temporal(self) -> None:
        model = self._make_model(use_dag=False, use_temporal=False)
        x = torch.randn(BATCH, INPUT_CHANNELS, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)

    def test_gradient_flows(self) -> None:
        model = self._make_model()
        x = torch.randn(BATCH, INPUT_CHANNELS, 32, 32, requires_grad=True)
        depth = torch.rand(BATCH, 1, 32, 32)
        out = model(x, depression_depth=depth)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_param_groups(self) -> None:
        model = self._make_model()
        groups = model.get_param_groups()
        assert len(groups) == 2
        assert all("params" in g and "lr_scale" in g for g in groups)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class TestSMPBaseline:
    """Tests for SMP wrapper."""

    def test_unetplusplus_forward(self) -> None:
        from research_paper.models.baselines import SMPBaseline
        model = SMPBaseline(arch="unetplusplus", input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
        x = torch.randn(BATCH, INPUT_CHANNELS, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)

    def test_deeplabv3plus_forward(self) -> None:
        from research_paper.models.baselines import SMPBaseline
        model = SMPBaseline(arch="deeplabv3plus", input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
        x = torch.randn(BATCH, INPUT_CHANNELS, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)

    def test_depression_depth_ignored(self) -> None:
        from research_paper.models.baselines import SMPBaseline
        model = SMPBaseline(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
        x = torch.randn(BATCH, INPUT_CHANNELS, HEIGHT, WIDTH)
        depth = torch.rand(BATCH, 1, HEIGHT, WIDTH)
        out = model(x, depression_depth=depth)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)


class TestUNetMambaBaseline:
    """Tests for UNetMamba (CNN encoder + Mamba decoder)."""

    def test_forward(self) -> None:
        from research_paper.models.baselines import UNetMambaBaseline
        model = UNetMambaBaseline(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
        x = torch.randn(BATCH, INPUT_CHANNELS, HEIGHT, WIDTH)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES, HEIGHT, WIDTH)


class TestBuildModel:
    """Tests for the model factory function."""

    def test_build_wetmamba(self) -> None:
        from research_paper.models.baselines import build_model
        model = build_model("wetmamba", num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS, use_pretrained=False)
        assert model is not None

    def test_build_unetplusplus(self) -> None:
        from research_paper.models.baselines import build_model
        model = build_model("unetplusplus", num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS)
        assert model is not None

    def test_build_unetmamba(self) -> None:
        from research_paper.models.baselines import build_model
        model = build_model("unetmamba", num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS)
        assert model is not None

    def test_build_unknown_raises(self) -> None:
        from research_paper.models.baselines import build_model
        with pytest.raises(ValueError):
            build_model("nonexistent_arch", num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for segmentation metrics computation."""

    def test_perfect_prediction(self) -> None:
        import numpy as np
        from research_paper.train_benchmark import compute_metrics
        pred = np.array([0, 1, 2, 3, 4, 5])
        target = np.array([0, 1, 2, 3, 4, 5])
        metrics = compute_metrics(pred, target, 6)
        assert metrics["overall_accuracy"] == 1.0
        assert metrics["mean_iou"] == 1.0

    def test_wrong_prediction(self) -> None:
        import numpy as np
        from research_paper.train_benchmark import compute_metrics
        pred = np.array([1, 0, 3, 2, 5, 4])
        target = np.array([0, 1, 2, 3, 4, 5])
        metrics = compute_metrics(pred, target, 6)
        assert metrics["overall_accuracy"] == 0.0
        assert metrics["mean_iou"] == 0.0
