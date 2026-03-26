#!/usr/bin/env python

"""Tests for `research_paper.wetland` module — Phase 5: Paper Experiments.

Covers experiment configuration, result formatting, and the PPR experiment
orchestration helpers.
"""

import inspect
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# PPR_STUDY_AREA constant tests
# ---------------------------------------------------------------------------


class TestPPRStudyArea(unittest.TestCase):
    """Tests for the PPR_STUDY_AREA constant."""

    def test_exists(self):
        from research_paper.wetland import PPR_STUDY_AREA

        self.assertIsInstance(PPR_STUDY_AREA, dict)

    def test_has_bbox(self):
        from research_paper.wetland import PPR_STUDY_AREA

        self.assertIn("bbox", PPR_STUDY_AREA)
        bbox = PPR_STUDY_AREA["bbox"]
        self.assertEqual(len(bbox), 4)
        self.assertLess(bbox[0], bbox[2])  # min_lon < max_lon
        self.assertLess(bbox[1], bbox[3])  # min_lat < max_lat

    def test_has_naip_years(self):
        from research_paper.wetland import PPR_STUDY_AREA

        self.assertIn("naip_years", PPR_STUDY_AREA)
        years = PPR_STUDY_AREA["naip_years"]
        self.assertIsInstance(years, list)
        self.assertIn(2015, years)
        self.assertIn(2017, years)

    def test_has_name(self):
        from research_paper.wetland import PPR_STUDY_AREA

        self.assertIn("name", PPR_STUDY_AREA)
        self.assertIsInstance(PPR_STUDY_AREA["name"], str)

    def test_has_huc8_codes(self):
        from research_paper.wetland import PPR_STUDY_AREA

        self.assertIn("huc8_codes", PPR_STUDY_AREA)
        codes = PPR_STUDY_AREA["huc8_codes"]
        self.assertIsInstance(codes, list)
        self.assertGreater(len(codes), 0)


# ---------------------------------------------------------------------------
# EXPERIMENT_DEFAULTS constant tests
# ---------------------------------------------------------------------------


class TestExperimentDefaults(unittest.TestCase):
    """Tests for the EXPERIMENT_DEFAULTS constant."""

    def test_exists(self):
        from research_paper.wetland import EXPERIMENT_DEFAULTS

        self.assertIsInstance(EXPERIMENT_DEFAULTS, dict)

    def test_has_training_keys(self):
        from research_paper.wetland import EXPERIMENT_DEFAULTS

        for key in [
            "tile_size",
            "num_epochs",
            "batch_size",
            "learning_rate",
            "val_split",
            "num_classes",
        ]:
            self.assertIn(key, EXPERIMENT_DEFAULTS, f"Missing key: {key}")

    def test_tile_size(self):
        from research_paper.wetland import EXPERIMENT_DEFAULTS

        self.assertEqual(EXPERIMENT_DEFAULTS["tile_size"], 256)

    def test_num_classes(self):
        from research_paper.wetland import EXPERIMENT_DEFAULTS

        self.assertEqual(EXPERIMENT_DEFAULTS["num_classes"], 6)

    def test_has_architecture_configs(self):
        from research_paper.wetland import EXPERIMENT_DEFAULTS

        self.assertIn("architectures", EXPERIMENT_DEFAULTS)
        archs = EXPERIMENT_DEFAULTS["architectures"]
        self.assertIsInstance(archs, list)
        arch_names = [a["architecture"] for a in archs]
        self.assertIn("unetplusplus", arch_names)
        self.assertIn("deeplabv3plus", arch_names)


# ---------------------------------------------------------------------------
# build_experiment_config signature and tests
# ---------------------------------------------------------------------------


class TestBuildExperimentConfigSignature(unittest.TestCase):
    """Tests for build_experiment_config function signature."""

    def test_callable(self):
        from research_paper.wetland import build_experiment_config

        self.assertTrue(callable(build_experiment_config))

    def test_expected_parameters(self):
        from research_paper.wetland import build_experiment_config

        sig = inspect.signature(build_experiment_config)
        for param in ["study_area", "output_root", "architectures", "overrides"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_default_architectures_is_none(self):
        from research_paper.wetland import build_experiment_config

        sig = inspect.signature(build_experiment_config)
        self.assertIsNone(sig.parameters["architectures"].default)

    def test_default_overrides_is_none(self):
        from research_paper.wetland import build_experiment_config

        sig = inspect.signature(build_experiment_config)
        self.assertIsNone(sig.parameters["overrides"].default)


class TestBuildExperimentConfigValidation(unittest.TestCase):
    """Tests for build_experiment_config input validation."""

    def test_missing_bbox_raises(self):
        from research_paper.wetland import build_experiment_config

        with self.assertRaises(ValueError):
            build_experiment_config(
                study_area={"name": "test"},
                output_root="/tmp/test",
            )

    def test_missing_naip_years_raises(self):
        from research_paper.wetland import build_experiment_config

        with self.assertRaises(ValueError):
            build_experiment_config(
                study_area={"name": "test", "bbox": (-100, 46, -99, 47)},
                output_root="/tmp/test",
            )

    def test_invalid_architecture_raises(self):
        from research_paper.wetland import build_experiment_config

        with self.assertRaises(ValueError):
            build_experiment_config(
                study_area={
                    "name": "test",
                    "bbox": (-100, 46, -99, 47),
                    "naip_years": [2015],
                },
                output_root="/tmp/test",
                architectures=[{"architecture": "invalid_arch"}],
            )

    def test_unknown_override_key_raises(self):
        from research_paper.wetland import build_experiment_config

        with self.assertRaises(ValueError):
            build_experiment_config(
                study_area={
                    "name": "test",
                    "bbox": (-100, 46, -99, 47),
                    "naip_years": [2015],
                },
                output_root="/tmp/test",
                overrides={"num_epoch": 5},  # typo: should be num_epochs
            )


class TestBuildExperimentConfigIntegration(unittest.TestCase):
    """Integration tests for build_experiment_config."""

    def test_returns_dict(self):
        from research_paper.wetland import PPR_STUDY_AREA, build_experiment_config

        config = build_experiment_config(
            study_area=PPR_STUDY_AREA,
            output_root="/tmp/experiment_test",
        )
        self.assertIsInstance(config, dict)

    def test_has_expected_keys(self):
        from research_paper.wetland import PPR_STUDY_AREA, build_experiment_config

        config = build_experiment_config(
            study_area=PPR_STUDY_AREA,
            output_root="/tmp/experiment_test",
        )
        for key in [
            "study_area",
            "output_root",
            "paths",
            "training",
            "architectures",
        ]:
            self.assertIn(key, config, f"Missing key: {key}")

    def test_paths_has_subdirectories(self):
        from research_paper.wetland import PPR_STUDY_AREA, build_experiment_config

        config = build_experiment_config(
            study_area=PPR_STUDY_AREA,
            output_root="/tmp/experiment_test",
        )
        paths = config["paths"]
        for key in ["naip_dir", "dem_path", "nwi_path", "composites_dir",
                     "tiles_dir", "models_dir", "predictions_dir", "results_dir"]:
            self.assertIn(key, paths, f"Missing path: {key}")

    def test_custom_architectures(self):
        from research_paper.wetland import PPR_STUDY_AREA, build_experiment_config

        config = build_experiment_config(
            study_area=PPR_STUDY_AREA,
            output_root="/tmp/experiment_test",
            architectures=[
                {"architecture": "unet", "encoder_name": "resnet34"},
            ],
        )
        self.assertEqual(len(config["architectures"]), 1)
        self.assertEqual(config["architectures"][0]["architecture"], "unet")

    def test_overrides_applied(self):
        from research_paper.wetland import PPR_STUDY_AREA, build_experiment_config

        config = build_experiment_config(
            study_area=PPR_STUDY_AREA,
            output_root="/tmp/experiment_test",
            overrides={"num_epochs": 100, "batch_size": 16},
        )
        self.assertEqual(config["training"]["num_epochs"], 100)
        self.assertEqual(config["training"]["batch_size"], 16)

    def test_default_uses_two_architectures(self):
        from research_paper.wetland import PPR_STUDY_AREA, build_experiment_config

        config = build_experiment_config(
            study_area=PPR_STUDY_AREA,
            output_root="/tmp/experiment_test",
        )
        arch_names = [a["architecture"] for a in config["architectures"]]
        self.assertIn("unetplusplus", arch_names)
        self.assertIn("deeplabv3plus", arch_names)


# ---------------------------------------------------------------------------
# format_results_table signature and tests
# ---------------------------------------------------------------------------


class TestFormatResultsTableSignature(unittest.TestCase):
    """Tests for format_results_table function signature."""

    def test_callable(self):
        from research_paper.wetland import format_results_table

        self.assertTrue(callable(format_results_table))

    def test_expected_parameters(self):
        from research_paper.wetland import format_results_table

        sig = inspect.signature(format_results_table)
        for param in ["results", "class_names"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_default_class_names_is_none(self):
        from research_paper.wetland import format_results_table

        sig = inspect.signature(format_results_table)
        self.assertIsNone(sig.parameters["class_names"].default)


class TestFormatResultsTableValidation(unittest.TestCase):
    """Tests for format_results_table input validation."""

    def test_empty_results_raises(self):
        from research_paper.wetland import format_results_table

        with self.assertRaises(ValueError):
            format_results_table(results=[])

    def test_missing_overall_accuracy_raises(self):
        from research_paper.wetland import format_results_table

        with self.assertRaises(ValueError):
            format_results_table(results=[{"name": "test"}])


class TestFormatResultsTableIntegration(unittest.TestCase):
    """Integration tests for format_results_table."""

    def _make_result(self, name="UNet++"):
        return {
            "name": name,
            "overall_accuracy": 0.85,
            "mean_iou": 0.62,
            "per_class_iou": {0: 0.90, 1: 0.75, 2: 0.55, 3: 0.40, 4: 0.35, 5: 0.30},
            "per_class_f1": {0: 0.94, 1: 0.85, 2: 0.70, 3: 0.55, 4: 0.50, 5: 0.45},
            "per_class_precision": {0: 0.92, 1: 0.82, 2: 0.68, 3: 0.52, 4: 0.48, 5: 0.42},
            "per_class_recall": {0: 0.96, 1: 0.88, 2: 0.72, 3: 0.58, 4: 0.52, 5: 0.48},
        }

    def test_returns_string(self):
        from research_paper.wetland import format_results_table

        table = format_results_table(results=[self._make_result()])
        self.assertIsInstance(table, str)

    def test_contains_header(self):
        from research_paper.wetland import format_results_table

        table = format_results_table(results=[self._make_result()])
        self.assertIn("OA", table)
        self.assertIn("mIoU", table)

    def test_contains_model_name(self):
        from research_paper.wetland import format_results_table

        table = format_results_table(results=[self._make_result("DeepLabV3+")])
        self.assertIn("DeepLabV3+", table)

    def test_multiple_results(self):
        from research_paper.wetland import format_results_table

        table = format_results_table(
            results=[
                self._make_result("UNet++"),
                self._make_result("DeepLabV3+"),
            ]
        )
        self.assertIn("UNet++", table)
        self.assertIn("DeepLabV3+", table)

    def test_custom_class_names(self):
        from research_paper.wetland import COWARDIN_CLASSES, format_results_table

        table = format_results_table(
            results=[self._make_result()],
            class_names=COWARDIN_CLASSES,
        )
        self.assertIn("Water", table)
        self.assertIn("Emergent", table)

    def test_default_uses_cowardin_classes(self):
        from research_paper.wetland import format_results_table

        table = format_results_table(results=[self._make_result()])
        # Should use COWARDIN_CLASSES by default
        self.assertIn("Upland", table)


# ---------------------------------------------------------------------------
# save_experiment_results signature and tests
# ---------------------------------------------------------------------------


class TestSaveExperimentResultsSignature(unittest.TestCase):
    """Tests for save_experiment_results function signature."""

    def test_callable(self):
        from research_paper.wetland import save_experiment_results

        self.assertTrue(callable(save_experiment_results))

    def test_expected_parameters(self):
        from research_paper.wetland import save_experiment_results

        sig = inspect.signature(save_experiment_results)
        for param in ["results", "output_path", "config"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")


class TestSaveExperimentResultsValidation(unittest.TestCase):
    """Tests for save_experiment_results input validation."""

    def test_empty_results_raises(self):
        from research_paper.wetland import save_experiment_results

        with self.assertRaises(ValueError):
            save_experiment_results(
                results=[],
                output_path="/tmp/results.json",
            )


class TestSaveExperimentResultsIntegration(unittest.TestCase):
    """Integration tests for save_experiment_results."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_result(self, name="UNet++"):
        return {
            "name": name,
            "architecture": "unetplusplus",
            "encoder_name": "resnet50",
            "overall_accuracy": 0.85,
            "mean_iou": 0.62,
            "per_class_iou": {0: 0.90, 1: 0.75},
            "per_class_f1": {0: 0.94, 1: 0.85},
            "per_class_precision": {0: 0.92, 1: 0.82},
            "per_class_recall": {0: 0.96, 1: 0.88},
            "confusion_matrix": [[100, 10], [5, 85]],
        }

    def test_creates_json_file(self):
        from research_paper.wetland import save_experiment_results

        out = os.path.join(self.tmpdir, "results.json")
        save_experiment_results(
            results=[self._make_result()],
            output_path=out,
        )
        self.assertTrue(os.path.exists(out))

    def test_json_is_valid(self):
        from research_paper.wetland import save_experiment_results

        out = os.path.join(self.tmpdir, "results.json")
        save_experiment_results(
            results=[self._make_result()],
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 1)

    def test_includes_config(self):
        from research_paper.wetland import save_experiment_results

        out = os.path.join(self.tmpdir, "results.json")
        config = {"study_area": "PPR", "num_epochs": 50}
        save_experiment_results(
            results=[self._make_result()],
            output_path=out,
            config=config,
        )
        with open(out) as f:
            data = json.load(f)
        self.assertIn("config", data)
        self.assertEqual(data["config"]["study_area"], "PPR")

    def test_creates_parent_dirs(self):
        from research_paper.wetland import save_experiment_results

        out = os.path.join(self.tmpdir, "sub", "dir", "results.json")
        save_experiment_results(
            results=[self._make_result()],
            output_path=out,
        )
        self.assertTrue(os.path.exists(out))

    def test_includes_summary_table(self):
        from research_paper.wetland import save_experiment_results

        out = os.path.join(self.tmpdir, "results.json")
        save_experiment_results(
            results=[self._make_result()],
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        self.assertIn("summary_table", data)


# ---------------------------------------------------------------------------
# __all__ update test
# ---------------------------------------------------------------------------


class TestPhase5ModuleExports(unittest.TestCase):
    """Tests that Phase 5 functions and constants are in __all__."""

    def test_all_contains_phase5_exports(self):
        from research_paper.wetland import __all__

        for name in [
            "PPR_STUDY_AREA",
            "EXPERIMENT_DEFAULTS",
            "build_experiment_config",
            "format_results_table",
            "save_experiment_results",
        ]:
            self.assertIn(name, __all__, f"{name} missing from __all__")


if __name__ == "__main__":
    unittest.main()
