#!/usr/bin/env python

"""Tests for `research_paper.wetland` module — Phase 2: Weak Label Generation.

Covers reclassify_nwi, generate_weak_labels, and export_training_tiles
functions including constants, input validation, integration with synthetic data.
"""

import os
import inspect
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np


# ---------------------------------------------------------------------------
# reclassify_nwi signature and validation tests
# ---------------------------------------------------------------------------


class TestReclassifyNWISignature(unittest.TestCase):
    """Tests for reclassify_nwi function signature."""

    def test_callable(self):
        from research_paper.wetland import reclassify_nwi

        self.assertTrue(callable(reclassify_nwi))

    def test_expected_parameters(self):
        from research_paper.wetland import reclassify_nwi

        sig = inspect.signature(reclassify_nwi)
        for param in ["nwi_path", "raster_template", "output_path", "overwrite"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_overwrite_defaults_to_false(self):
        from research_paper.wetland import reclassify_nwi

        sig = inspect.signature(reclassify_nwi)
        self.assertFalse(sig.parameters["overwrite"].default)


class TestReclassifyNWIValidation(unittest.TestCase):
    """Tests for reclassify_nwi input validation."""

    def test_nonexistent_nwi_raises(self):
        from research_paper.wetland import reclassify_nwi

        with self.assertRaises(FileNotFoundError):
            reclassify_nwi(
                nwi_path="/nonexistent/nwi.gpkg",
                raster_template="/tmp/template.tif",
                output_path="/tmp/out.tif",
            )

    def test_nonexistent_template_raises(self):
        from research_paper.wetland import reclassify_nwi

        with tempfile.NamedTemporaryFile(suffix=".gpkg") as nwi:
            with self.assertRaises(FileNotFoundError):
                reclassify_nwi(
                    nwi_path=nwi.name,
                    raster_template="/nonexistent/template.tif",
                    output_path="/tmp/out.tif",
                )

    def test_overwrite_false_existing_raises(self):
        from research_paper.wetland import reclassify_nwi

        with tempfile.NamedTemporaryFile(suffix=".gpkg") as nwi:
            with tempfile.NamedTemporaryFile(suffix=".tif") as tmpl:
                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as out:
                    out_path = out.name
                try:
                    with self.assertRaises(FileExistsError):
                        reclassify_nwi(
                            nwi_path=nwi.name,
                            raster_template=tmpl.name,
                            output_path=out_path,
                            overwrite=False,
                        )
                finally:
                    os.unlink(out_path)


class TestReclassifyNWIIntegration(unittest.TestCase):
    """Integration tests for reclassify_nwi with synthetic data."""

    def setUp(self):
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_bounds
        from shapely.geometry import box

        self.tmpdir = tempfile.mkdtemp()

        # Create a template raster (16x16, EPSG:4326)
        self.template_path = os.path.join(self.tmpdir, "template.tif")
        height, width = 16, 16
        self.bounds = (-100.5, 47.0, -100.0, 47.5)
        transform = from_bounds(*self.bounds, width, height)
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(self.template_path, "w", **profile) as dst:
            dst.write(np.ones((height, width), dtype=np.uint8), 1)

        # Create synthetic NWI polygons with Cowardin-like codes
        mid_lon = (self.bounds[0] + self.bounds[2]) / 2
        mid_lat = (self.bounds[1] + self.bounds[3]) / 2
        polys = [
            # Emergent wetland in upper-left
            {
                "geometry": box(self.bounds[0], mid_lat, mid_lon, self.bounds[3]),
                "WETLAND_TYPE": "Freshwater Emergent Wetland",
                "ATTRIBUTE": "PEM1Ch",
            },
            # Forested wetland in upper-right
            {
                "geometry": box(mid_lon, mid_lat, self.bounds[2], self.bounds[3]),
                "WETLAND_TYPE": "Freshwater Forested/Shrub Wetland",
                "ATTRIBUTE": "PFO1A",
            },
            # Open water in lower-left
            {
                "geometry": box(self.bounds[0], self.bounds[1], mid_lon, mid_lat),
                "WETLAND_TYPE": "Lake",
                "ATTRIBUTE": "L1UBHh",
            },
        ]
        self.nwi_path = os.path.join(self.tmpdir, "nwi.gpkg")
        gdf = gpd.GeoDataFrame(polys, crs="EPSG:4326")
        gdf.to_file(self.nwi_path, driver="GPKG")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_produces_output_file(self):
        from research_paper.wetland import reclassify_nwi

        out = os.path.join(self.tmpdir, "nwi_reclassified.tif")
        result = reclassify_nwi(self.nwi_path, self.template_path, out)
        self.assertEqual(result, out)
        self.assertTrue(os.path.exists(out))

    def test_output_matches_template_grid(self):
        import rasterio

        from research_paper.wetland import reclassify_nwi

        out = os.path.join(self.tmpdir, "nwi_reclassified.tif")
        reclassify_nwi(self.nwi_path, self.template_path, out)
        with rasterio.open(self.template_path) as tmpl:
            with rasterio.open(out) as src:
                self.assertEqual(src.crs, tmpl.crs)
                self.assertEqual(src.height, tmpl.height)
                self.assertEqual(src.width, tmpl.width)
                self.assertEqual(src.bounds, tmpl.bounds)

    def test_output_contains_expected_classes(self):
        import rasterio

        from research_paper.wetland import reclassify_nwi

        out = os.path.join(self.tmpdir, "nwi_reclassified.tif")
        reclassify_nwi(self.nwi_path, self.template_path, out)
        with rasterio.open(out) as src:
            data = src.read(1)
            unique = set(np.unique(data))
            # Should contain water (1), emergent (2), forested (3)
            # and possibly 0 (upland/background) for lower-right
            self.assertTrue({1, 2, 3}.issubset(unique) or {1, 2, 3} == unique)

    def test_output_dtype_is_uint8(self):
        import rasterio

        from research_paper.wetland import reclassify_nwi

        out = os.path.join(self.tmpdir, "nwi_reclassified.tif")
        reclassify_nwi(self.nwi_path, self.template_path, out)
        with rasterio.open(out) as src:
            self.assertEqual(src.dtypes[0], "uint8")

    def test_class_values_in_valid_range(self):
        import rasterio

        from research_paper.wetland import COWARDIN_CLASSES, reclassify_nwi

        out = os.path.join(self.tmpdir, "nwi_reclassified.tif")
        reclassify_nwi(self.nwi_path, self.template_path, out)
        with rasterio.open(out) as src:
            data = src.read(1)
            valid_ids = set(COWARDIN_CLASSES.keys())
            for val in np.unique(data):
                self.assertIn(int(val), valid_ids, f"Class {val} not in COWARDIN_CLASSES")


# ---------------------------------------------------------------------------
# generate_weak_labels signature and validation tests
# ---------------------------------------------------------------------------


class TestGenerateWeakLabelsSignature(unittest.TestCase):
    """Tests for generate_weak_labels function signature."""

    def test_callable(self):
        from research_paper.wetland import generate_weak_labels

        self.assertTrue(callable(generate_weak_labels))

    def test_expected_parameters(self):
        from research_paper.wetland import generate_weak_labels

        sig = inspect.signature(generate_weak_labels)
        for param in [
            "nwi_raster_path",
            "depression_path",
            "ndvi_paths",
            "ndwi_paths",
            "output_path",
            "depression_threshold",
            "stability_threshold",
            "min_component_fraction",
            "overwrite",
        ]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_default_thresholds(self):
        from research_paper.wetland import generate_weak_labels

        sig = inspect.signature(generate_weak_labels)
        self.assertAlmostEqual(sig.parameters["depression_threshold"].default, 0.0)
        self.assertAlmostEqual(sig.parameters["stability_threshold"].default, 0.05)
        self.assertAlmostEqual(sig.parameters["min_component_fraction"].default, 0.5)
        self.assertFalse(sig.parameters["overwrite"].default)


class TestGenerateWeakLabelsValidation(unittest.TestCase):
    """Tests for generate_weak_labels input validation."""

    def test_nonexistent_nwi_raster_raises(self):
        from research_paper.wetland import generate_weak_labels

        with self.assertRaises(FileNotFoundError):
            generate_weak_labels(
                nwi_raster_path="/nonexistent/nwi.tif",
                depression_path="/tmp/dep.tif",
                ndvi_paths=["/tmp/ndvi1.tif"],
                ndwi_paths=["/tmp/ndwi1.tif"],
                output_path="/tmp/out.tif",
            )

    def test_nonexistent_depression_raises(self):
        from research_paper.wetland import generate_weak_labels

        with tempfile.NamedTemporaryFile(suffix=".tif") as nwi:
            with self.assertRaises(FileNotFoundError):
                generate_weak_labels(
                    nwi_raster_path=nwi.name,
                    depression_path="/nonexistent/dep.tif",
                    ndvi_paths=["/tmp/ndvi1.tif"],
                    ndwi_paths=["/tmp/ndwi1.tif"],
                    output_path="/tmp/out.tif",
                )

    def test_mismatched_epoch_counts_raises(self):
        from research_paper.wetland import generate_weak_labels

        with tempfile.NamedTemporaryFile(suffix=".tif") as nwi:
            with tempfile.NamedTemporaryFile(suffix=".tif") as dep:
                with tempfile.NamedTemporaryFile(suffix=".tif") as ndvi:
                    with self.assertRaises(ValueError):
                        generate_weak_labels(
                            nwi_raster_path=nwi.name,
                            depression_path=dep.name,
                            ndvi_paths=[ndvi.name],
                            ndwi_paths=[],  # mismatch
                            output_path="/tmp/out.tif",
                        )

    def test_negative_thresholds_raise(self):
        from research_paper.wetland import generate_weak_labels

        with tempfile.NamedTemporaryFile(suffix=".tif") as nwi:
            with tempfile.NamedTemporaryFile(suffix=".tif") as dep:
                with tempfile.NamedTemporaryFile(suffix=".tif") as idx:
                    with self.assertRaises(ValueError):
                        generate_weak_labels(
                            nwi_raster_path=nwi.name,
                            depression_path=dep.name,
                            ndvi_paths=[idx.name],
                            ndwi_paths=[idx.name],
                            output_path="/tmp/out.tif",
                            stability_threshold=-0.1,
                        )


class TestGenerateWeakLabelsIntegration(unittest.TestCase):
    """Integration tests for generate_weak_labels with synthetic rasters."""

    def setUp(self):
        import rasterio
        from rasterio.transform import from_bounds

        self.tmpdir = tempfile.mkdtemp()
        height, width = 32, 32
        bounds = (-100.5, 47.0, -100.0, 47.5)
        transform = from_bounds(*bounds, width, height)

        base_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
            "count": 1,
        }

        # NWI raster: upper half = emergent (2), lower half = water (1)
        self.nwi_path = os.path.join(self.tmpdir, "nwi.tif")
        nwi_data = np.zeros((height, width), dtype=np.uint8)
        nwi_data[:height // 2, :] = 2  # emergent
        nwi_data[height // 2:, :] = 1  # water
        with rasterio.open(self.nwi_path, "w", dtype="uint8", **base_profile) as dst:
            dst.write(nwi_data, 1)

        # Depression raster: center has depressions, edges don't
        self.dep_path = os.path.join(self.tmpdir, "depressions.tif")
        dep_data = np.zeros((height, width), dtype=np.float32)
        dep_data[8:24, 8:24] = 2.0  # depression in center
        with rasterio.open(
            self.dep_path, "w", dtype="float32", nodata=-9999.0, **base_profile
        ) as dst:
            dst.write(dep_data, 1)

        # NDVI epoch 1 and 2 — stable in center, changing at edges
        self.ndvi1_path = os.path.join(self.tmpdir, "ndvi_2015.tif")
        self.ndvi2_path = os.path.join(self.tmpdir, "ndvi_2017.tif")
        ndvi1 = np.full((height, width), 0.3, dtype=np.float32)
        ndvi2 = np.full((height, width), 0.32, dtype=np.float32)  # stable (diff=0.02)
        # Make edges unstable
        ndvi2[:4, :] = 0.8  # big change in top rows
        ndvi2[-4:, :] = 0.8  # big change in bottom rows
        with rasterio.open(self.ndvi1_path, "w", dtype="float32", **base_profile) as dst:
            dst.write(ndvi1, 1)
        with rasterio.open(self.ndvi2_path, "w", dtype="float32", **base_profile) as dst:
            dst.write(ndvi2, 1)

        # NDWI epoch 1 and 2 — stable everywhere for simplicity
        self.ndwi1_path = os.path.join(self.tmpdir, "ndwi_2015.tif")
        self.ndwi2_path = os.path.join(self.tmpdir, "ndwi_2017.tif")
        ndwi1 = np.full((height, width), 0.1, dtype=np.float32)
        ndwi2 = np.full((height, width), 0.12, dtype=np.float32)
        ndwi2[:4, :] = 0.7  # unstable at top
        ndwi2[-4:, :] = 0.7  # unstable at bottom
        with rasterio.open(self.ndwi1_path, "w", dtype="float32", **base_profile) as dst:
            dst.write(ndwi1, 1)
        with rasterio.open(self.ndwi2_path, "w", dtype="float32", **base_profile) as dst:
            dst.write(ndwi2, 1)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_produces_output_file(self):
        from research_paper.wetland import generate_weak_labels

        out = os.path.join(self.tmpdir, "labels.tif")
        result = generate_weak_labels(
            nwi_raster_path=self.nwi_path,
            depression_path=self.dep_path,
            ndvi_paths=[self.ndvi1_path, self.ndvi2_path],
            ndwi_paths=[self.ndwi1_path, self.ndwi2_path],
            output_path=out,
        )
        self.assertEqual(result, out)
        self.assertTrue(os.path.exists(out))

    def test_depression_filter_removes_non_depression_pixels(self):
        import rasterio

        from research_paper.wetland import generate_weak_labels

        out = os.path.join(self.tmpdir, "labels.tif")
        generate_weak_labels(
            nwi_raster_path=self.nwi_path,
            depression_path=self.dep_path,
            ndvi_paths=[self.ndvi1_path, self.ndvi2_path],
            ndwi_paths=[self.ndwi1_path, self.ndwi2_path],
            output_path=out,
            depression_threshold=0.5,
        )
        with rasterio.open(out) as src:
            data = src.read(1)
            # Corners (outside depression area 8:24,8:24) should be 0 (no label)
            self.assertEqual(data[0, 0], 0)
            self.assertEqual(data[-1, -1], 0)

    def test_stability_filter_removes_unstable_pixels(self):
        import rasterio

        from research_paper.wetland import generate_weak_labels

        out = os.path.join(self.tmpdir, "labels.tif")
        generate_weak_labels(
            nwi_raster_path=self.nwi_path,
            depression_path=self.dep_path,
            ndvi_paths=[self.ndvi1_path, self.ndvi2_path],
            ndwi_paths=[self.ndwi1_path, self.ndwi2_path],
            output_path=out,
            stability_threshold=0.05,
        )
        with rasterio.open(out) as src:
            data = src.read(1)
            # Top rows (0-3) should be filtered out (unstable)
            self.assertTrue(np.all(data[:4, :] == 0))

    def test_output_preserves_grid(self):
        import rasterio

        from research_paper.wetland import generate_weak_labels

        out = os.path.join(self.tmpdir, "labels.tif")
        generate_weak_labels(
            nwi_raster_path=self.nwi_path,
            depression_path=self.dep_path,
            ndvi_paths=[self.ndvi1_path, self.ndvi2_path],
            ndwi_paths=[self.ndwi1_path, self.ndwi2_path],
            output_path=out,
        )
        with rasterio.open(self.nwi_path) as nwi:
            with rasterio.open(out) as src:
                self.assertEqual(src.crs, nwi.crs)
                self.assertEqual(src.height, nwi.height)
                self.assertEqual(src.width, nwi.width)

    def test_output_dtype_is_uint8(self):
        import rasterio

        from research_paper.wetland import generate_weak_labels

        out = os.path.join(self.tmpdir, "labels.tif")
        generate_weak_labels(
            nwi_raster_path=self.nwi_path,
            depression_path=self.dep_path,
            ndvi_paths=[self.ndvi1_path, self.ndvi2_path],
            ndwi_paths=[self.ndwi1_path, self.ndwi2_path],
            output_path=out,
        )
        with rasterio.open(out) as src:
            self.assertEqual(src.dtypes[0], "uint8")

    def test_retained_labels_are_subset_of_original(self):
        import rasterio

        from research_paper.wetland import generate_weak_labels

        out = os.path.join(self.tmpdir, "labels.tif")
        generate_weak_labels(
            nwi_raster_path=self.nwi_path,
            depression_path=self.dep_path,
            ndvi_paths=[self.ndvi1_path, self.ndvi2_path],
            ndwi_paths=[self.ndwi1_path, self.ndwi2_path],
            output_path=out,
        )
        with rasterio.open(self.nwi_path) as nwi_src:
            nwi_data = nwi_src.read(1)
        with rasterio.open(out) as src:
            label_data = src.read(1)
            # Every non-zero label pixel must match original NWI
            mask = label_data > 0
            np.testing.assert_array_equal(label_data[mask], nwi_data[mask])


# ---------------------------------------------------------------------------
# export_training_tiles signature and validation tests
# ---------------------------------------------------------------------------


class TestExportTrainingTilesSignature(unittest.TestCase):
    """Tests for export_training_tiles function signature."""

    def test_callable(self):
        from research_paper.wetland import export_training_tiles

        self.assertTrue(callable(export_training_tiles))

    def test_expected_parameters(self):
        from research_paper.wetland import export_training_tiles

        sig = inspect.signature(export_training_tiles)
        for param in [
            "composite_path",
            "label_path",
            "output_dir",
            "tile_size",
            "stride",
            "min_valid_fraction",
            "overwrite",
        ]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_tile_size_default(self):
        from research_paper.wetland import export_training_tiles

        sig = inspect.signature(export_training_tiles)
        self.assertEqual(sig.parameters["tile_size"].default, 256)

    def test_overwrite_defaults_to_false(self):
        from research_paper.wetland import export_training_tiles

        sig = inspect.signature(export_training_tiles)
        self.assertFalse(sig.parameters["overwrite"].default)


class TestExportTrainingTilesValidation(unittest.TestCase):
    """Tests for export_training_tiles input validation."""

    def test_nonexistent_composite_raises(self):
        from research_paper.wetland import export_training_tiles

        with self.assertRaises(FileNotFoundError):
            export_training_tiles(
                composite_path="/nonexistent/composite.tif",
                label_path="/tmp/labels.tif",
                output_dir="/tmp/tiles",
            )

    def test_nonexistent_label_raises(self):
        from research_paper.wetland import export_training_tiles

        with tempfile.NamedTemporaryFile(suffix=".tif") as comp:
            with self.assertRaises(FileNotFoundError):
                export_training_tiles(
                    composite_path=comp.name,
                    label_path="/nonexistent/labels.tif",
                    output_dir="/tmp/tiles",
                )

    def test_invalid_tile_size_raises(self):
        from research_paper.wetland import export_training_tiles

        with tempfile.NamedTemporaryFile(suffix=".tif") as comp:
            with tempfile.NamedTemporaryFile(suffix=".tif") as lbl:
                with self.assertRaises(ValueError):
                    export_training_tiles(
                        composite_path=comp.name,
                        label_path=lbl.name,
                        output_dir="/tmp/tiles",
                        tile_size=0,
                    )

    def test_invalid_min_valid_fraction_raises(self):
        from research_paper.wetland import export_training_tiles

        with tempfile.NamedTemporaryFile(suffix=".tif") as comp:
            with tempfile.NamedTemporaryFile(suffix=".tif") as lbl:
                with self.assertRaises(ValueError):
                    export_training_tiles(
                        composite_path=comp.name,
                        label_path=lbl.name,
                        output_dir="/tmp/tiles",
                        min_valid_fraction=1.5,
                    )


class TestExportTrainingTilesIntegration(unittest.TestCase):
    """Integration tests for export_training_tiles with synthetic data."""

    def setUp(self):
        import rasterio
        from rasterio.transform import from_bounds

        self.tmpdir = tempfile.mkdtemp()

        height, width = 64, 64
        bounds = (-100.5, 47.0, -100.0, 47.5)
        transform = from_bounds(*bounds, width, height)

        # 4-band composite
        self.composite_path = os.path.join(self.tmpdir, "composite.tif")
        rng = np.random.default_rng(42)
        composite_data = rng.integers(10, 250, size=(4, height, width), dtype=np.uint8)
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 4,
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(self.composite_path, "w", **profile) as dst:
            dst.write(composite_data)

        # Label raster (single band, 3 classes + background)
        self.label_path = os.path.join(self.tmpdir, "labels.tif")
        label_data = np.zeros((height, width), dtype=np.uint8)
        label_data[:32, :32] = 1
        label_data[:32, 32:] = 2
        label_data[32:, :32] = 3
        # lower-right = 0 (background/no label)
        label_profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(self.label_path, "w", **label_profile) as dst:
            dst.write(label_data, 1)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_produces_output_directory(self):
        from research_paper.wetland import export_training_tiles

        out_dir = os.path.join(self.tmpdir, "tiles")
        result = export_training_tiles(
            self.composite_path, self.label_path, out_dir, tile_size=32
        )
        self.assertTrue(os.path.isdir(out_dir))
        self.assertIsInstance(result, dict)

    def test_creates_image_and_label_subdirs(self):
        from research_paper.wetland import export_training_tiles

        out_dir = os.path.join(self.tmpdir, "tiles")
        export_training_tiles(
            self.composite_path, self.label_path, out_dir, tile_size=32
        )
        self.assertTrue(os.path.isdir(os.path.join(out_dir, "images")))
        self.assertTrue(os.path.isdir(os.path.join(out_dir, "labels")))

    def test_tile_count_is_positive(self):
        from research_paper.wetland import export_training_tiles

        out_dir = os.path.join(self.tmpdir, "tiles")
        result = export_training_tiles(
            self.composite_path, self.label_path, out_dir, tile_size=32
        )
        self.assertGreater(result["num_tiles"], 0)

    def test_tile_files_exist(self):
        from research_paper.wetland import export_training_tiles

        out_dir = os.path.join(self.tmpdir, "tiles")
        result = export_training_tiles(
            self.composite_path, self.label_path, out_dir, tile_size=32
        )
        img_files = list(Path(out_dir, "images").glob("*.tif"))
        lbl_files = list(Path(out_dir, "labels").glob("*.tif"))
        self.assertEqual(len(img_files), result["num_tiles"])
        self.assertEqual(len(lbl_files), result["num_tiles"])

    def test_tile_dimensions(self):
        import rasterio

        from research_paper.wetland import export_training_tiles

        out_dir = os.path.join(self.tmpdir, "tiles")
        export_training_tiles(
            self.composite_path, self.label_path, out_dir, tile_size=32
        )
        img_files = list(Path(out_dir, "images").glob("*.tif"))
        self.assertGreater(len(img_files), 0)
        with rasterio.open(img_files[0]) as src:
            self.assertEqual(src.height, 32)
            self.assertEqual(src.width, 32)

    def test_stride_affects_tile_count(self):
        from research_paper.wetland import export_training_tiles

        out1 = os.path.join(self.tmpdir, "tiles_s32")
        out2 = os.path.join(self.tmpdir, "tiles_s16")
        r1 = export_training_tiles(
            self.composite_path, self.label_path, out1, tile_size=32, stride=32
        )
        r2 = export_training_tiles(
            self.composite_path, self.label_path, out2, tile_size=32, stride=16
        )
        # Smaller stride = more tiles (with overlap)
        self.assertGreaterEqual(r2["num_tiles"], r1["num_tiles"])

    def test_min_valid_fraction_filters_tiles(self):
        from research_paper.wetland import export_training_tiles

        # With high min_valid_fraction, tiles with mostly background are skipped
        out_low = os.path.join(self.tmpdir, "tiles_low")
        out_high = os.path.join(self.tmpdir, "tiles_high")
        r_low = export_training_tiles(
            self.composite_path,
            self.label_path,
            out_low,
            tile_size=32,
            min_valid_fraction=0.0,
        )
        r_high = export_training_tiles(
            self.composite_path,
            self.label_path,
            out_high,
            tile_size=32,
            min_valid_fraction=0.9,
        )
        self.assertGreaterEqual(r_low["num_tiles"], r_high["num_tiles"])

    def test_label_tiles_are_single_band(self):
        import rasterio

        from research_paper.wetland import export_training_tiles

        out_dir = os.path.join(self.tmpdir, "tiles")
        export_training_tiles(
            self.composite_path, self.label_path, out_dir, tile_size=32
        )
        lbl_files = list(Path(out_dir, "labels").glob("*.tif"))
        self.assertGreater(len(lbl_files), 0)
        with rasterio.open(lbl_files[0]) as src:
            self.assertEqual(src.count, 1)


# ---------------------------------------------------------------------------
# __all__ update test
# ---------------------------------------------------------------------------


class TestPhase2ModuleExports(unittest.TestCase):
    """Tests that Phase 2 functions are in __all__."""

    def test_all_contains_phase2_functions(self):
        from research_paper.wetland import __all__

        for name in [
            "reclassify_nwi",
            "generate_weak_labels",
            "export_training_tiles",
        ]:
            self.assertIn(name, __all__, f"{name} missing from __all__")


if __name__ == "__main__":
    unittest.main()
