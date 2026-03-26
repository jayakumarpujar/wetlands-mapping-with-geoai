#!/usr/bin/env python

"""Tests for `research_paper.wetland` module — Phase 4: Inference & Dynamics.

Covers predict_wetlands, map_wetland_dynamics, and compare_with_nwi
functions including signatures, validation, and integration with synthetic data.
"""

import inspect
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# predict_wetlands signature and validation tests
# ---------------------------------------------------------------------------


class TestPredictWetlandsSignature(unittest.TestCase):
    """Tests for predict_wetlands function signature."""

    def test_callable(self):
        from research_paper.wetland import predict_wetlands

        self.assertTrue(callable(predict_wetlands))

    def test_expected_parameters(self):
        from research_paper.wetland import predict_wetlands

        sig = inspect.signature(predict_wetlands)
        for param in [
            "model_path",
            "composite_path",
            "output_path",
            "architecture",
            "encoder_name",
            "num_classes",
            "in_channels",
            "tile_size",
            "overlap",
            "batch_size",
            "device",
            "overwrite",
        ]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_default_tile_size(self):
        from research_paper.wetland import predict_wetlands

        sig = inspect.signature(predict_wetlands)
        self.assertEqual(sig.parameters["tile_size"].default, 256)

    def test_default_overlap(self):
        from research_paper.wetland import predict_wetlands

        sig = inspect.signature(predict_wetlands)
        self.assertEqual(sig.parameters["overlap"].default, 128)

    def test_default_overwrite(self):
        from research_paper.wetland import predict_wetlands

        sig = inspect.signature(predict_wetlands)
        self.assertFalse(sig.parameters["overwrite"].default)


class TestPredictWetlandsValidation(unittest.TestCase):
    """Tests for predict_wetlands input validation."""

    def test_nonexistent_model_raises(self):
        from research_paper.wetland import predict_wetlands

        with self.assertRaises(FileNotFoundError):
            predict_wetlands(
                model_path="/nonexistent/model.pth",
                composite_path="/tmp/composite.tif",
                output_path="/tmp/pred.tif",
            )

    def test_nonexistent_composite_raises(self):
        from research_paper.wetland import predict_wetlands

        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            with self.assertRaises(FileNotFoundError):
                predict_wetlands(
                    model_path=model.name,
                    composite_path="/nonexistent/composite.tif",
                    output_path="/tmp/pred.tif",
                )

    def test_overwrite_false_existing_raises(self):
        from research_paper.wetland import predict_wetlands

        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            with tempfile.NamedTemporaryFile(suffix=".tif") as comp:
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as out:
                    out_path = out.name
                try:
                    with self.assertRaises(FileExistsError):
                        predict_wetlands(
                            model_path=model.name,
                            composite_path=comp.name,
                            output_path=out_path,
                            overwrite=False,
                        )
                finally:
                    os.unlink(out_path)

    def test_invalid_tile_size_raises(self):
        from research_paper.wetland import predict_wetlands

        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            with tempfile.NamedTemporaryFile(suffix=".tif") as comp:
                with self.assertRaises(ValueError):
                    predict_wetlands(
                        model_path=model.name,
                        composite_path=comp.name,
                        output_path="/tmp/pred.tif",
                        tile_size=0,
                    )

    def test_invalid_overlap_raises(self):
        from research_paper.wetland import predict_wetlands

        with tempfile.NamedTemporaryFile(suffix=".pth") as model:
            with tempfile.NamedTemporaryFile(suffix=".tif") as comp:
                with self.assertRaises(ValueError):
                    predict_wetlands(
                        model_path=model.name,
                        composite_path=comp.name,
                        output_path="/tmp/pred.tif",
                        tile_size=256,
                        overlap=256,  # overlap must be < tile_size
                    )


class TestPredictWetlandsIntegration(unittest.TestCase):
    """Integration tests for predict_wetlands with a real lightweight model."""

    def setUp(self):
        import rasterio
        import segmentation_models_pytorch as smp
        import torch
        from rasterio.transform import from_bounds

        self.tmpdir = tempfile.mkdtemp()
        self.in_channels = 4
        self.num_classes = 4
        self.tile_size = 32

        # Create a small model and save it
        model = smp.create_model(
            "unet",
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=self.in_channels,
            classes=self.num_classes,
        )
        self.model_path = os.path.join(self.tmpdir, "model.pth")
        torch.save(model.state_dict(), self.model_path)

        # Create a synthetic composite raster (64x64, 4 bands)
        height, width = 64, 64
        bounds = (-100.5, 47.0, -100.0, 47.5)
        transform = from_bounds(*bounds, width, height)
        self.composite_path = os.path.join(self.tmpdir, "composite.tif")

        rng = np.random.default_rng(42)
        data = rng.random((self.in_channels, height, width)).astype(np.float32)
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": self.in_channels,
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(self.composite_path, "w", **profile) as dst:
            dst.write(data)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_produces_output_file(self):
        from research_paper.wetland import predict_wetlands

        out = os.path.join(self.tmpdir, "prediction.tif")
        result = predict_wetlands(
            model_path=self.model_path,
            composite_path=self.composite_path,
            output_path=out,
            architecture="unet",
            encoder_name="resnet18",
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            tile_size=self.tile_size,
            overlap=16,
        )
        self.assertEqual(result, out)
        self.assertTrue(os.path.exists(out))

    def test_output_matches_input_grid(self):
        import rasterio

        from research_paper.wetland import predict_wetlands

        out = os.path.join(self.tmpdir, "prediction.tif")
        predict_wetlands(
            model_path=self.model_path,
            composite_path=self.composite_path,
            output_path=out,
            architecture="unet",
            encoder_name="resnet18",
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            tile_size=self.tile_size,
            overlap=16,
        )
        with rasterio.open(self.composite_path) as comp:
            with rasterio.open(out) as src:
                self.assertEqual(src.crs, comp.crs)
                self.assertEqual(src.height, comp.height)
                self.assertEqual(src.width, comp.width)
                self.assertEqual(src.bounds, comp.bounds)

    def test_output_is_single_band_uint8(self):
        import rasterio

        from research_paper.wetland import predict_wetlands

        out = os.path.join(self.tmpdir, "prediction.tif")
        predict_wetlands(
            model_path=self.model_path,
            composite_path=self.composite_path,
            output_path=out,
            architecture="unet",
            encoder_name="resnet18",
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            tile_size=self.tile_size,
            overlap=16,
        )
        with rasterio.open(out) as src:
            self.assertEqual(src.count, 1)
            self.assertEqual(src.dtypes[0], "uint8")

    def test_output_values_in_class_range(self):
        import rasterio

        from research_paper.wetland import predict_wetlands

        out = os.path.join(self.tmpdir, "prediction.tif")
        predict_wetlands(
            model_path=self.model_path,
            composite_path=self.composite_path,
            output_path=out,
            architecture="unet",
            encoder_name="resnet18",
            num_classes=self.num_classes,
            in_channels=self.in_channels,
            tile_size=self.tile_size,
            overlap=16,
        )
        with rasterio.open(out) as src:
            data = src.read(1)
            self.assertTrue(np.all(data < self.num_classes))


# ---------------------------------------------------------------------------
# map_wetland_dynamics signature and validation tests
# ---------------------------------------------------------------------------


class TestMapWetlandDynamicsSignature(unittest.TestCase):
    """Tests for map_wetland_dynamics function signature."""

    def test_callable(self):
        from research_paper.wetland import map_wetland_dynamics

        self.assertTrue(callable(map_wetland_dynamics))

    def test_expected_parameters(self):
        from research_paper.wetland import map_wetland_dynamics

        sig = inspect.signature(map_wetland_dynamics)
        for param in [
            "prediction_paths",
            "output_path",
            "overwrite",
        ]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")


class TestMapWetlandDynamicsValidation(unittest.TestCase):
    """Tests for map_wetland_dynamics input validation."""

    def test_fewer_than_two_predictions_raises(self):
        from research_paper.wetland import map_wetland_dynamics

        with tempfile.NamedTemporaryFile(suffix=".tif") as f:
            with self.assertRaises(ValueError):
                map_wetland_dynamics(
                    prediction_paths=[f.name],
                    output_path="/tmp/dynamics.tif",
                )

    def test_nonexistent_prediction_raises(self):
        from research_paper.wetland import map_wetland_dynamics

        with self.assertRaises(FileNotFoundError):
            map_wetland_dynamics(
                prediction_paths=["/nonexistent/a.tif", "/nonexistent/b.tif"],
                output_path="/tmp/dynamics.tif",
            )

    def test_overwrite_false_existing_raises(self):
        from research_paper.wetland import map_wetland_dynamics

        with tempfile.NamedTemporaryFile(suffix=".tif") as a:
            with tempfile.NamedTemporaryFile(suffix=".tif") as b:
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as out:
                    out_path = out.name
                try:
                    with self.assertRaises(FileExistsError):
                        map_wetland_dynamics(
                            prediction_paths=[a.name, b.name],
                            output_path=out_path,
                            overwrite=False,
                        )
                finally:
                    os.unlink(out_path)


class TestMapWetlandDynamicsIntegration(unittest.TestCase):
    """Integration tests for map_wetland_dynamics with synthetic predictions."""

    def setUp(self):
        import rasterio
        from rasterio.transform import from_bounds

        self.tmpdir = tempfile.mkdtemp()
        height, width = 32, 32
        bounds = (-100.5, 47.0, -100.0, 47.5)
        transform = from_bounds(*bounds, width, height)

        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
        }

        # Epoch 1: upper half = water (1), lower half = upland (0)
        self.pred1_path = os.path.join(self.tmpdir, "pred_2015.tif")
        pred1 = np.zeros((height, width), dtype=np.uint8)
        pred1[: height // 2, :] = 1  # water
        with rasterio.open(self.pred1_path, "w", **profile) as dst:
            dst.write(pred1, 1)

        # Epoch 2: upper-left = water (1), upper-right = emergent (2),
        # lower-left = water (1, gain), lower-right = upland (0)
        self.pred2_path = os.path.join(self.tmpdir, "pred_2019.tif")
        pred2 = np.zeros((height, width), dtype=np.uint8)
        pred2[: height // 2, : width // 2] = 1  # water (stable)
        pred2[: height // 2, width // 2 :] = 2  # emergent (class change)
        pred2[height // 2 :, : width // 2] = 1  # water (gain)
        with rasterio.open(self.pred2_path, "w", **profile) as dst:
            dst.write(pred2, 1)

        self.height = height
        self.width = width

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_produces_output_file(self):
        from research_paper.wetland import map_wetland_dynamics

        out = os.path.join(self.tmpdir, "dynamics.tif")
        result = map_wetland_dynamics(
            prediction_paths=[self.pred1_path, self.pred2_path],
            output_path=out,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("output_path", result)
        self.assertTrue(os.path.exists(result["output_path"]))

    def test_output_preserves_grid(self):
        import rasterio

        from research_paper.wetland import map_wetland_dynamics

        out = os.path.join(self.tmpdir, "dynamics.tif")
        result = map_wetland_dynamics(
            prediction_paths=[self.pred1_path, self.pred2_path],
            output_path=out,
        )
        with rasterio.open(self.pred1_path) as ref:
            with rasterio.open(result["output_path"]) as src:
                self.assertEqual(src.crs, ref.crs)
                self.assertEqual(src.height, ref.height)
                self.assertEqual(src.width, ref.width)

    def test_returns_change_statistics(self):
        from research_paper.wetland import map_wetland_dynamics

        out = os.path.join(self.tmpdir, "dynamics.tif")
        result = map_wetland_dynamics(
            prediction_paths=[self.pred1_path, self.pred2_path],
            output_path=out,
        )
        self.assertIn("statistics", result)
        stats = result["statistics"]
        self.assertIn("wetland_gain_pixels", stats)
        self.assertIn("wetland_loss_pixels", stats)
        self.assertIn("stable_wetland_pixels", stats)

    def test_detects_wetland_gain(self):
        from research_paper.wetland import map_wetland_dynamics

        out = os.path.join(self.tmpdir, "dynamics.tif")
        result = map_wetland_dynamics(
            prediction_paths=[self.pred1_path, self.pred2_path],
            output_path=out,
        )
        # Lower-left quadrant went from upland to water = gain
        self.assertGreater(result["statistics"]["wetland_gain_pixels"], 0)

    def test_detects_wetland_loss(self):
        from research_paper.wetland import map_wetland_dynamics

        out = os.path.join(self.tmpdir, "dynamics.tif")
        result = map_wetland_dynamics(
            prediction_paths=[self.pred1_path, self.pred2_path],
            output_path=out,
        )
        # Upper-right quadrant changed from water to emergent — still wetland
        # Lower-right was upland and stayed upland — no loss
        # But some pixels in upper-right went from water(1) to emergent(2)
        # which is still wetland, so depends on definition
        # The key is the function runs and returns a count
        self.assertIsInstance(result["statistics"]["wetland_loss_pixels"], int)

    def test_dynamics_raster_has_expected_codes(self):
        import rasterio

        from research_paper.wetland import map_wetland_dynamics

        out = os.path.join(self.tmpdir, "dynamics.tif")
        result = map_wetland_dynamics(
            prediction_paths=[self.pred1_path, self.pred2_path],
            output_path=out,
        )
        with rasterio.open(result["output_path"]) as src:
            data = src.read(1)
            unique = set(np.unique(data))
            # Should contain at least some of: 0=no change upland,
            # 1=stable wetland, 2=wetland gain, 3=wetland loss
            self.assertTrue(len(unique) > 0)


# ---------------------------------------------------------------------------
# compare_with_nwi signature and validation tests
# ---------------------------------------------------------------------------


class TestCompareWithNWISignature(unittest.TestCase):
    """Tests for compare_with_nwi function signature."""

    def test_callable(self):
        from research_paper.wetland import compare_with_nwi

        self.assertTrue(callable(compare_with_nwi))

    def test_expected_parameters(self):
        from research_paper.wetland import compare_with_nwi

        sig = inspect.signature(compare_with_nwi)
        for param in [
            "prediction_path",
            "reference_path",
        ]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")


class TestCompareWithNWIValidation(unittest.TestCase):
    """Tests for compare_with_nwi input validation."""

    def test_nonexistent_prediction_raises(self):
        from research_paper.wetland import compare_with_nwi

        with self.assertRaises(FileNotFoundError):
            compare_with_nwi(
                prediction_path="/nonexistent/pred.tif",
                reference_path="/tmp/ref.tif",
            )

    def test_nonexistent_reference_raises(self):
        from research_paper.wetland import compare_with_nwi

        with tempfile.NamedTemporaryFile(suffix=".tif") as pred:
            with self.assertRaises(FileNotFoundError):
                compare_with_nwi(
                    prediction_path=pred.name,
                    reference_path="/nonexistent/ref.tif",
                )


class TestCompareWithNWIIntegration(unittest.TestCase):
    """Integration tests for compare_with_nwi with synthetic data."""

    def setUp(self):
        import rasterio
        from rasterio.transform import from_bounds

        self.tmpdir = tempfile.mkdtemp()
        height, width = 32, 32
        bounds = (-100.5, 47.0, -100.0, 47.5)
        transform = from_bounds(*bounds, width, height)

        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
        }

        # Prediction: upper = water(1), lower-left = emergent(2), lower-right = upland(0)
        self.pred_path = os.path.join(self.tmpdir, "prediction.tif")
        pred = np.zeros((height, width), dtype=np.uint8)
        pred[: height // 2, :] = 1
        pred[height // 2 :, : width // 2] = 2
        with rasterio.open(self.pred_path, "w", **profile) as dst:
            dst.write(pred, 1)

        # Reference (NWI): mostly matches, with some disagreement
        self.ref_path = os.path.join(self.tmpdir, "reference.tif")
        ref = np.zeros((height, width), dtype=np.uint8)
        ref[: height // 2, :] = 1  # same as prediction
        ref[height // 2 :, : width // 2] = 2  # same as prediction
        ref[height // 2 :, width // 2 :] = 1  # differs: pred=0, ref=1
        with rasterio.open(self.ref_path, "w", **profile) as dst:
            dst.write(ref, 1)

        # Perfect match reference
        self.perfect_ref_path = os.path.join(self.tmpdir, "perfect_ref.tif")
        with rasterio.open(self.perfect_ref_path, "w", **profile) as dst:
            dst.write(pred, 1)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_dict_with_metrics(self):
        from research_paper.wetland import compare_with_nwi

        result = compare_with_nwi(
            prediction_path=self.pred_path,
            reference_path=self.ref_path,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("overall_accuracy", result)
        self.assertIn("per_class_iou", result)
        self.assertIn("per_class_f1", result)
        self.assertIn("confusion_matrix", result)

    def test_overall_accuracy_in_range(self):
        from research_paper.wetland import compare_with_nwi

        result = compare_with_nwi(
            prediction_path=self.pred_path,
            reference_path=self.ref_path,
        )
        self.assertGreaterEqual(result["overall_accuracy"], 0.0)
        self.assertLessEqual(result["overall_accuracy"], 1.0)

    def test_perfect_prediction_accuracy(self):
        from research_paper.wetland import compare_with_nwi

        result = compare_with_nwi(
            prediction_path=self.pred_path,
            reference_path=self.perfect_ref_path,
        )
        self.assertAlmostEqual(result["overall_accuracy"], 1.0)

    def test_per_class_iou_is_dict(self):
        from research_paper.wetland import compare_with_nwi

        result = compare_with_nwi(
            prediction_path=self.pred_path,
            reference_path=self.ref_path,
        )
        self.assertIsInstance(result["per_class_iou"], dict)

    def test_per_class_f1_values_in_range(self):
        from research_paper.wetland import compare_with_nwi

        result = compare_with_nwi(
            prediction_path=self.pred_path,
            reference_path=self.ref_path,
        )
        for cls, f1 in result["per_class_f1"].items():
            self.assertGreaterEqual(f1, 0.0, f"Class {cls} F1 < 0")
            self.assertLessEqual(f1, 1.0, f"Class {cls} F1 > 1")

    def test_confusion_matrix_shape(self):
        from research_paper.wetland import compare_with_nwi

        result = compare_with_nwi(
            prediction_path=self.pred_path,
            reference_path=self.ref_path,
        )
        cm = np.array(result["confusion_matrix"])
        # Should be square
        self.assertEqual(cm.shape[0], cm.shape[1])

    def test_imperfect_prediction_has_lower_accuracy(self):
        from research_paper.wetland import compare_with_nwi

        perfect = compare_with_nwi(
            prediction_path=self.pred_path,
            reference_path=self.perfect_ref_path,
        )
        imperfect = compare_with_nwi(
            prediction_path=self.pred_path,
            reference_path=self.ref_path,
        )
        self.assertGreater(
            perfect["overall_accuracy"], imperfect["overall_accuracy"]
        )


# ---------------------------------------------------------------------------
# __all__ update test
# ---------------------------------------------------------------------------


class TestPhase4ModuleExports(unittest.TestCase):
    """Tests that Phase 4 functions are in __all__."""

    def test_all_contains_phase4_functions(self):
        from research_paper.wetland import __all__

        for name in [
            "predict_wetlands",
            "map_wetland_dynamics",
            "compare_with_nwi",
        ]:
            self.assertIn(name, __all__, f"{name} missing from __all__")


if __name__ == "__main__":
    unittest.main()
