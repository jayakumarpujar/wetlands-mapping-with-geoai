#!/usr/bin/env python

"""Tests for `research_paper.wetland` module — Phase 3: Model Training.

Covers train_wetland_model function including signature, validation,
default configuration, and integration with synthetic tile data.
"""

import inspect
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# train_wetland_model signature and default tests
# ---------------------------------------------------------------------------


class TestTrainWetlandModelSignature(unittest.TestCase):
    """Tests for train_wetland_model function signature."""

    def test_callable(self):
        from research_paper.wetland import train_wetland_model

        self.assertTrue(callable(train_wetland_model))

    def test_expected_parameters(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        for param in [
            "tiles_dir",
            "output_dir",
            "architecture",
            "encoder_name",
            "num_classes",
            "in_channels",
            "num_epochs",
            "batch_size",
            "learning_rate",
            "loss_function",
            "use_class_weights",
            "val_split",
            "seed",
            "device",
            "overwrite",
        ]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_default_architecture(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        self.assertEqual(sig.parameters["architecture"].default, "unetplusplus")

    def test_default_encoder(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        self.assertEqual(sig.parameters["encoder_name"].default, "resnet50")

    def test_default_num_classes(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        self.assertEqual(sig.parameters["num_classes"].default, 6)

    def test_default_in_channels(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        self.assertEqual(sig.parameters["in_channels"].default, 14)

    def test_default_loss_function(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        self.assertEqual(sig.parameters["loss_function"].default, "focal")

    def test_default_use_class_weights(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        self.assertTrue(sig.parameters["use_class_weights"].default)

    def test_default_overwrite(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        self.assertFalse(sig.parameters["overwrite"].default)

    def test_default_val_split(self):
        from research_paper.wetland import train_wetland_model

        sig = inspect.signature(train_wetland_model)
        self.assertAlmostEqual(sig.parameters["val_split"].default, 0.2)


class TestTrainWetlandModelValidation(unittest.TestCase):
    """Tests for train_wetland_model input validation."""

    def test_nonexistent_tiles_dir_raises(self):
        from research_paper.wetland import train_wetland_model

        with self.assertRaises(FileNotFoundError):
            train_wetland_model(
                tiles_dir="/nonexistent/tiles",
                output_dir="/tmp/output",
            )

    def test_missing_images_subdir_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_missing_labels_subdir_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            with self.assertRaises(FileNotFoundError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_num_classes_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(ValueError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                    num_classes=0,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_in_channels_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(ValueError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                    in_channels=0,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_architecture_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(ValueError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                    architecture="invalid_arch",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_val_split_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(ValueError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                    val_split=1.5,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_loss_function_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(ValueError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                    loss_function="invalid_loss",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_num_epochs_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(ValueError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                    num_epochs=0,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_batch_size_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(ValueError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                    batch_size=-1,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_invalid_learning_rate_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(ValueError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir="/tmp/output",
                    learning_rate=0,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_overwrite_false_existing_output_raises(self):
        from research_paper.wetland import train_wetland_model

        tmpdir = tempfile.mkdtemp()
        outdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(tmpdir, "images"))
            os.makedirs(os.path.join(tmpdir, "labels"))
            with self.assertRaises(FileExistsError):
                train_wetland_model(
                    tiles_dir=tmpdir,
                    output_dir=outdir,
                    overwrite=False,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            shutil.rmtree(outdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# SUPPORTED_ARCHITECTURES constant tests
# ---------------------------------------------------------------------------


class TestSupportedArchitectures(unittest.TestCase):
    """Tests for the SUPPORTED_ARCHITECTURES constant."""

    def test_is_frozenset(self):
        from research_paper.wetland import SUPPORTED_ARCHITECTURES

        self.assertIsInstance(SUPPORTED_ARCHITECTURES, frozenset)

    def test_contains_key_architectures(self):
        from research_paper.wetland import SUPPORTED_ARCHITECTURES

        for arch in ["unet", "unetplusplus", "deeplabv3", "deeplabv3plus", "fpn"]:
            self.assertIn(arch, SUPPORTED_ARCHITECTURES, f"Missing: {arch}")


# ---------------------------------------------------------------------------
# Integration test: train_wetland_model with synthetic tiles
# ---------------------------------------------------------------------------


class TestTrainWetlandModelIntegration(unittest.TestCase):
    """Integration tests for train_wetland_model with synthetic tile data.

    Uses small synthetic tiles and minimal training config to verify
    end-to-end functionality without requiring GPU or large datasets.
    """

    def setUp(self):
        import rasterio
        from rasterio.transform import from_bounds

        self.tmpdir = tempfile.mkdtemp()
        self.tiles_dir = os.path.join(self.tmpdir, "tiles")
        self.output_dir = os.path.join(self.tmpdir, "model_output")

        img_dir = os.path.join(self.tiles_dir, "images")
        lbl_dir = os.path.join(self.tiles_dir, "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        tile_size = 32
        in_channels = 4  # Use 4 channels for fast testing
        num_tiles = 8
        rng = np.random.default_rng(42)
        bounds = (-100.5, 47.0, -100.0, 47.5)
        transform = from_bounds(*bounds, tile_size, tile_size)

        img_profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": in_channels,
            "height": tile_size,
            "width": tile_size,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        lbl_profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": tile_size,
            "width": tile_size,
            "crs": "EPSG:4326",
            "transform": transform,
        }

        for i in range(num_tiles):
            name = f"tile_{i:06d}.tif"

            # Random image data
            img_data = rng.random((in_channels, tile_size, tile_size)).astype(
                np.float32
            )
            with rasterio.open(os.path.join(img_dir, name), "w", **img_profile) as dst:
                dst.write(img_data)

            # Random label data (classes 0-3)
            lbl_data = rng.integers(0, 4, size=(tile_size, tile_size)).astype(np.uint8)
            with rasterio.open(os.path.join(lbl_dir, name), "w", **lbl_profile) as dst:
                dst.write(lbl_data, 1)

        self.in_channels = in_channels

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_dict_with_expected_keys(self):
        from research_paper.wetland import train_wetland_model

        result = train_wetland_model(
            tiles_dir=self.tiles_dir,
            output_dir=self.output_dir,
            in_channels=self.in_channels,
            num_classes=4,
            num_epochs=1,
            batch_size=4,
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            use_class_weights=False,
            loss_function="crossentropy",
        )
        self.assertIsInstance(result, dict)
        self.assertIn("model_path", result)
        self.assertIn("architecture", result)
        self.assertIn("encoder_name", result)
        self.assertIn("num_classes", result)
        self.assertIn("in_channels", result)

    def test_creates_model_file(self):
        from research_paper.wetland import train_wetland_model

        result = train_wetland_model(
            tiles_dir=self.tiles_dir,
            output_dir=self.output_dir,
            in_channels=self.in_channels,
            num_classes=4,
            num_epochs=1,
            batch_size=4,
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            use_class_weights=False,
            loss_function="crossentropy",
        )
        self.assertTrue(
            os.path.exists(result["model_path"]),
            f"Model file not found at {result['model_path']}",
        )

    def test_output_dir_created(self):
        from research_paper.wetland import train_wetland_model

        train_wetland_model(
            tiles_dir=self.tiles_dir,
            output_dir=self.output_dir,
            in_channels=self.in_channels,
            num_classes=4,
            num_epochs=1,
            batch_size=4,
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            use_class_weights=False,
            loss_function="crossentropy",
        )
        self.assertTrue(os.path.isdir(self.output_dir))

    def test_result_contains_correct_config(self):
        from research_paper.wetland import train_wetland_model

        result = train_wetland_model(
            tiles_dir=self.tiles_dir,
            output_dir=self.output_dir,
            in_channels=self.in_channels,
            num_classes=4,
            num_epochs=1,
            batch_size=4,
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            use_class_weights=False,
            loss_function="crossentropy",
        )
        self.assertEqual(result["architecture"], "unet")
        self.assertEqual(result["encoder_name"], "resnet18")
        self.assertEqual(result["num_classes"], 4)
        self.assertEqual(result["in_channels"], self.in_channels)

    def test_overwrite_true_allows_retraining(self):
        from research_paper.wetland import train_wetland_model

        base_args = dict(
            tiles_dir=self.tiles_dir,
            output_dir=self.output_dir,
            in_channels=self.in_channels,
            num_classes=4,
            num_epochs=1,
            batch_size=4,
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            use_class_weights=False,
            loss_function="crossentropy",
        )
        # First training
        train_wetland_model(**base_args)
        # Second training with overwrite
        result = train_wetland_model(**base_args, overwrite=True)
        self.assertTrue(os.path.exists(result["model_path"]))


# ---------------------------------------------------------------------------
# __all__ update test
# ---------------------------------------------------------------------------


class TestPhase3ModuleExports(unittest.TestCase):
    """Tests that Phase 3 functions and constants are in __all__."""

    def test_all_contains_phase3_functions(self):
        from research_paper.wetland import __all__

        for name in [
            "train_wetland_model",
            "SUPPORTED_ARCHITECTURES",
        ]:
            self.assertIn(name, __all__, f"{name} missing from __all__")


if __name__ == "__main__":
    unittest.main()
