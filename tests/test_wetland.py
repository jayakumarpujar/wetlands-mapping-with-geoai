#!/usr/bin/env python

"""Tests for `research_paper.wetland` module.

Covers constants, input validation, helper functions, and public API
signatures for the Phase 1 wetland data pipeline.
"""

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestCowardinClasses(unittest.TestCase):
    """Tests for the COWARDIN_CLASSES constant."""

    def test_is_dict(self):
        from research_paper.wetland import COWARDIN_CLASSES

        self.assertIsInstance(COWARDIN_CLASSES, dict)

    def test_keys_are_ints(self):
        from research_paper.wetland import COWARDIN_CLASSES

        for key in COWARDIN_CLASSES:
            self.assertIsInstance(key, int)

    def test_values_are_strings(self):
        from research_paper.wetland import COWARDIN_CLASSES

        for val in COWARDIN_CLASSES.values():
            self.assertIsInstance(val, str)

    def test_contains_expected_classes(self):
        from research_paper.wetland import COWARDIN_CLASSES

        expected = {0: "Upland", 1: "Water", 2: "Emergent", 3: "Forested", 4: "Scrub-Shrub"}
        for key, val in expected.items():
            self.assertIn(key, COWARDIN_CLASSES)
            self.assertEqual(COWARDIN_CLASSES[key], val)

    def test_class_ids_are_contiguous(self):
        from research_paper.wetland import COWARDIN_CLASSES

        ids = sorted(COWARDIN_CLASSES.keys())
        self.assertEqual(ids, list(range(len(ids))))


class TestNWICodeToClass(unittest.TestCase):
    """Tests for the NWI_CODE_TO_CLASS mapping."""

    def test_is_dict(self):
        from research_paper.wetland import NWI_CODE_TO_CLASS

        self.assertIsInstance(NWI_CODE_TO_CLASS, dict)

    def test_keys_are_strings(self):
        from research_paper.wetland import NWI_CODE_TO_CLASS

        for key in NWI_CODE_TO_CLASS:
            self.assertIsInstance(key, str)

    def test_values_are_valid_class_ids(self):
        from research_paper.wetland import COWARDIN_CLASSES, NWI_CODE_TO_CLASS

        valid_ids = set(COWARDIN_CLASSES.keys())
        for code, class_id in NWI_CODE_TO_CLASS.items():
            self.assertIn(
                class_id, valid_ids, f"NWI code {code!r} maps to invalid class {class_id}"
            )

    def test_water_codes(self):
        from research_paper.wetland import NWI_CODE_TO_CLASS

        for code in ["L", "R", "PAB", "PUB", "POW"]:
            self.assertEqual(NWI_CODE_TO_CLASS[code], 1, f"{code} should map to Water (1)")

    def test_emergent_code(self):
        from research_paper.wetland import NWI_CODE_TO_CLASS

        self.assertEqual(NWI_CODE_TO_CLASS["PEM"], 2)

    def test_forested_code(self):
        from research_paper.wetland import NWI_CODE_TO_CLASS

        self.assertEqual(NWI_CODE_TO_CLASS["PFO"], 3)

    def test_scrub_shrub_code(self):
        from research_paper.wetland import NWI_CODE_TO_CLASS

        self.assertEqual(NWI_CODE_TO_CLASS["PSS"], 4)


class TestNAIPBands(unittest.TestCase):
    """Tests for the NAIP_BANDS constant."""

    def test_has_four_bands(self):
        from research_paper.wetland import NAIP_BANDS

        self.assertEqual(len(NAIP_BANDS), 4)

    def test_keys_are_one_based(self):
        from research_paper.wetland import NAIP_BANDS

        self.assertEqual(sorted(NAIP_BANDS.keys()), [1, 2, 3, 4])

    def test_band_names(self):
        from research_paper.wetland import NAIP_BANDS

        self.assertEqual(NAIP_BANDS[1], "Red")
        self.assertEqual(NAIP_BANDS[4], "NIR")


class TestSpectralIndices(unittest.TestCase):
    """Tests for the SPECTRAL_INDICES constant."""

    def test_contains_ndvi(self):
        from research_paper.wetland import SPECTRAL_INDICES

        self.assertIn("ndvi", SPECTRAL_INDICES)

    def test_contains_ndwi(self):
        from research_paper.wetland import SPECTRAL_INDICES

        self.assertIn("ndwi", SPECTRAL_INDICES)

    def test_values_are_formula_strings(self):
        from research_paper.wetland import SPECTRAL_INDICES

        for name, formula in SPECTRAL_INDICES.items():
            self.assertIsInstance(formula, str, f"{name} formula should be a string")
            self.assertGreater(len(formula), 0)


# ---------------------------------------------------------------------------
# __all__ and lazy import tests
# ---------------------------------------------------------------------------


class TestModuleExports(unittest.TestCase):
    """Tests for module __all__ and importability."""

    def test_all_defined(self):
        from research_paper.wetland import __all__

        self.assertIsInstance(__all__, list)

    def test_all_contains_public_functions(self):
        from research_paper.wetland import __all__

        expected = [
            "COWARDIN_CLASSES",
            "NWI_CODE_TO_CLASS",
            "NAIP_BANDS",
            "SPECTRAL_INDICES",
            "download_naip_timeseries",
            "download_3dep_dem",
            "download_nwi",
            "compute_spectral_indices",
            "extract_surface_depressions",
            "create_wetland_composite",
        ]
        for name in expected:
            self.assertIn(name, __all__, f"{name} missing from __all__")

    def test_all_items_importable(self):
        import research_paper.wetland as mod

        for name in mod.__all__:
            self.assertTrue(
                hasattr(mod, name), f"{name} listed in __all__ but not defined"
            )

    @unittest.skip("Skipped: wetland module lives in research_paper/, not geoai/")
    def test_lazy_import_from_package(self):
        """Verify symbols are accessible via geoai.symbol_name."""
        import geoai

        for name in [
            "COWARDIN_CLASSES",
            "download_naip_timeseries",
            "download_3dep_dem",
            "download_nwi",
            "compute_spectral_indices",
            "extract_surface_depressions",
            "create_wetland_composite",
        ]:
            self.assertTrue(
                hasattr(geoai, name),
                f"geoai.{name} not accessible (check __init__.py registration)",
            )


# ---------------------------------------------------------------------------
# parse_nwi_code helper tests
# ---------------------------------------------------------------------------


class TestParseNWICode(unittest.TestCase):
    """Tests for the _parse_nwi_code helper function."""

    def test_lacustrine(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code("L1UBHh"), 1)

    def test_riverine(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code("R2UBH"), 1)

    def test_palustrine_emergent(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code("PEM1Ch"), 2)

    def test_palustrine_forested(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code("PFO1A"), 3)

    def test_palustrine_scrub_shrub(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code("PSS1A"), 4)

    def test_palustrine_open_water(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code("POWHh"), 1)

    def test_unknown_code_returns_other(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code("XYZABC"), 5)

    def test_empty_string_returns_other(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code(""), 5)

    def test_none_returns_other(self):
        from research_paper.wetland import _parse_nwi_code

        self.assertEqual(_parse_nwi_code(None), 5)

    def test_case_insensitive_prefix(self):
        from research_paper.wetland import _parse_nwi_code

        # NWI codes should work uppercase; lowercase not standard but handled
        self.assertEqual(_parse_nwi_code("pem1Ch"), 2)


# ---------------------------------------------------------------------------
# Function signature tests
# ---------------------------------------------------------------------------


class TestDownloadNAIPTimeseriesSignature(unittest.TestCase):
    """Tests for download_naip_timeseries function signature."""

    def test_callable(self):
        from research_paper.wetland import download_naip_timeseries

        self.assertTrue(callable(download_naip_timeseries))

    def test_expected_parameters(self):
        from research_paper.wetland import download_naip_timeseries

        sig = inspect.signature(download_naip_timeseries)
        for param in ["bbox", "output_dir", "years", "max_items_per_year", "overwrite"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_years_defaults_to_none(self):
        from research_paper.wetland import download_naip_timeseries

        sig = inspect.signature(download_naip_timeseries)
        self.assertEqual(sig.parameters["years"].default, None)

    def test_overwrite_defaults_to_false(self):
        from research_paper.wetland import download_naip_timeseries

        sig = inspect.signature(download_naip_timeseries)
        self.assertFalse(sig.parameters["overwrite"].default)


class TestDownload3DEPDEMSignature(unittest.TestCase):
    """Tests for download_3dep_dem function signature."""

    def test_callable(self):
        from research_paper.wetland import download_3dep_dem

        self.assertTrue(callable(download_3dep_dem))

    def test_expected_parameters(self):
        from research_paper.wetland import download_3dep_dem

        sig = inspect.signature(download_3dep_dem)
        for param in ["bbox", "output_path", "resolution", "overwrite"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_resolution_default(self):
        from research_paper.wetland import download_3dep_dem

        sig = inspect.signature(download_3dep_dem)
        self.assertEqual(sig.parameters["resolution"].default, 10)


class TestDownloadNWISignature(unittest.TestCase):
    """Tests for download_nwi function signature."""

    def test_callable(self):
        from research_paper.wetland import download_nwi

        self.assertTrue(callable(download_nwi))

    def test_expected_parameters(self):
        from research_paper.wetland import download_nwi

        sig = inspect.signature(download_nwi)
        for param in ["bbox", "output_path", "overwrite"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")


class TestComputeSpectralIndicesSignature(unittest.TestCase):
    """Tests for compute_spectral_indices function signature."""

    def test_callable(self):
        from research_paper.wetland import compute_spectral_indices

        self.assertTrue(callable(compute_spectral_indices))

    def test_expected_parameters(self):
        from research_paper.wetland import compute_spectral_indices

        sig = inspect.signature(compute_spectral_indices)
        for param in ["naip_path", "output_path", "indices", "overwrite"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")


class TestExtractSurfaceDepressionsSignature(unittest.TestCase):
    """Tests for extract_surface_depressions function signature."""

    def test_callable(self):
        from research_paper.wetland import extract_surface_depressions

        self.assertTrue(callable(extract_surface_depressions))

    def test_expected_parameters(self):
        from research_paper.wetland import extract_surface_depressions

        sig = inspect.signature(extract_surface_depressions)
        for param in ["dem_path", "output_path", "min_depth", "min_area", "overwrite"]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")

    def test_min_depth_default(self):
        from research_paper.wetland import extract_surface_depressions

        sig = inspect.signature(extract_surface_depressions)
        self.assertAlmostEqual(sig.parameters["min_depth"].default, 0.1)


class TestCreateWetlandCompositeSignature(unittest.TestCase):
    """Tests for create_wetland_composite function signature."""

    def test_callable(self):
        from research_paper.wetland import create_wetland_composite

        self.assertTrue(callable(create_wetland_composite))

    def test_expected_parameters(self):
        from research_paper.wetland import create_wetland_composite

        sig = inspect.signature(create_wetland_composite)
        for param in [
            "naip_paths",
            "dem_path",
            "output_path",
            "indices",
            "include_depressions",
            "overwrite",
        ]:
            self.assertIn(param, sig.parameters, f"Missing parameter: {param}")


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestBBoxValidation(unittest.TestCase):
    """Tests for bounding box validation across functions."""

    def test_invalid_bbox_length(self):
        from research_paper.wetland import _validate_bbox

        with self.assertRaises(ValueError) as ctx:
            _validate_bbox((1.0, 2.0, 3.0))
        self.assertIn("4 values", str(ctx.exception))

    def test_min_greater_than_max_lon(self):
        from research_paper.wetland import _validate_bbox

        with self.assertRaises(ValueError) as ctx:
            _validate_bbox((10.0, 0.0, 5.0, 1.0))
        self.assertIn("min_lon", str(ctx.exception).lower())

    def test_min_greater_than_max_lat(self):
        from research_paper.wetland import _validate_bbox

        with self.assertRaises(ValueError) as ctx:
            _validate_bbox((-100.0, 48.0, -99.0, 47.0))
        self.assertIn("min_lat", str(ctx.exception).lower())

    def test_valid_bbox_passes(self):
        from research_paper.wetland import _validate_bbox

        # Should not raise
        _validate_bbox((-100.5, 47.0, -100.0, 47.5))

    def test_bbox_with_non_numeric(self):
        from research_paper.wetland import _validate_bbox

        with self.assertRaises((ValueError, TypeError)):
            _validate_bbox(("a", 47.0, -100.0, 47.5))


class TestResolutionValidation(unittest.TestCase):
    """Tests for 3DEP resolution validation."""

    def test_invalid_resolution_raises(self):
        from research_paper.wetland import download_3dep_dem

        with self.assertRaises(ValueError) as ctx:
            download_3dep_dem(
                bbox=(-100.5, 47.0, -100.0, 47.5),
                output_path="/tmp/test_dem.tif",
                resolution=5,
            )
        self.assertIn("resolution", str(ctx.exception).lower())


class TestIndicesValidation(unittest.TestCase):
    """Tests for spectral index name validation."""

    def test_unknown_index_raises(self):
        from research_paper.wetland import _validate_index_names

        with self.assertRaises(ValueError) as ctx:
            _validate_index_names(["ndvi", "fake_index"])
        self.assertIn("fake_index", str(ctx.exception))

    def test_valid_indices_pass(self):
        from research_paper.wetland import _validate_index_names

        # Should not raise
        _validate_index_names(["ndvi", "ndwi"])

    def test_none_returns_all(self):
        from research_paper.wetland import SPECTRAL_INDICES, _validate_index_names

        result = _validate_index_names(None)
        self.assertEqual(result, list(SPECTRAL_INDICES.keys()))


# ---------------------------------------------------------------------------
# Depression filling algorithm tests
# ---------------------------------------------------------------------------


class TestFillDepressions(unittest.TestCase):
    """Tests for the _fill_depressions internal algorithm."""

    def test_flat_surface_unchanged(self):
        from research_paper.wetland import _fill_depressions

        dem = np.full((5, 5), 10.0, dtype=np.float64)
        filled = _fill_depressions(dem)
        np.testing.assert_array_equal(filled, dem)

    def test_single_depression_filled(self):
        from research_paper.wetland import _fill_depressions

        dem = np.array(
            [
                [10, 10, 10, 10, 10],
                [10, 8, 8, 8, 10],
                [10, 8, 5, 8, 10],
                [10, 8, 8, 8, 10],
                [10, 10, 10, 10, 10],
            ],
            dtype=np.float64,
        )
        filled = _fill_depressions(dem)
        # The interior depression should be filled to the pour point (8.0)
        # Actually the pour point is 8.0 (the rim around the center)
        # Wait - the border pixels are 10.0. The pixels at (1,1)...(3,3) form
        # a depression surrounded by 10.0 on the border. The lowest rim is 10.0.
        # So everything fills to 10.0
        self.assertGreaterEqual(filled.min(), 10.0)
        # The center pixel (was 5) should now be 10
        self.assertAlmostEqual(filled[2, 2], 10.0)

    def test_sloped_surface_unchanged(self):
        from research_paper.wetland import _fill_depressions

        # A uniformly sloped surface has no depressions
        dem = np.arange(25, dtype=np.float64).reshape(5, 5)
        filled = _fill_depressions(dem)
        np.testing.assert_array_almost_equal(filled, dem)

    def test_nodata_handling(self):
        from research_paper.wetland import _fill_depressions

        dem = np.full((5, 5), 10.0, dtype=np.float64)
        dem[2, 2] = -9999.0
        filled = _fill_depressions(dem, nodata=-9999.0)
        # Nodata pixels should remain NaN in output
        self.assertTrue(np.isnan(filled[2, 2]))
        # Valid pixels should be unchanged (flat surface)
        valid = ~np.isnan(filled)
        np.testing.assert_array_almost_equal(filled[valid], 10.0)

    def test_nan_treated_as_nodata(self):
        from research_paper.wetland import _fill_depressions

        dem = np.full((5, 5), 10.0, dtype=np.float64)
        dem[0, 2] = np.nan
        filled = _fill_depressions(dem)
        self.assertTrue(np.isnan(filled[0, 2]))

    def test_output_always_gte_input(self):
        from research_paper.wetland import _fill_depressions

        rng = np.random.default_rng(42)
        dem = rng.uniform(0, 100, size=(20, 20))
        filled = _fill_depressions(dem)
        diff = filled - dem
        self.assertTrue(np.all(diff >= -1e-10), "Filled DEM should be >= original everywhere")

    def test_depression_at_edge(self):
        from research_paper.wetland import _fill_depressions

        # Depression touching the border should drain out
        dem = np.full((5, 5), 10.0, dtype=np.float64)
        dem[0, 2] = 3.0  # Low point on border
        dem[1, 2] = 5.0  # Depression connected to border outlet
        filled = _fill_depressions(dem)
        # The border pixel stays at 3.0 (it IS the border)
        self.assertAlmostEqual(filled[0, 2], 3.0)
        # Interior pixel at (1,2) can drain to (0,2), so it stays at 5.0
        self.assertAlmostEqual(filled[1, 2], 5.0)

    def test_empty_array_raises(self):
        from research_paper.wetland import _fill_depressions

        with self.assertRaises(ValueError):
            _fill_depressions(np.array([]))

    def test_1d_array_raises(self):
        from research_paper.wetland import _fill_depressions

        with self.assertRaises(ValueError):
            _fill_depressions(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# Spectral index computation tests
# ---------------------------------------------------------------------------


class TestComputeIndex(unittest.TestCase):
    """Tests for the _compute_index helper function."""

    def test_ndvi_computation(self):
        from research_paper.wetland import _compute_index

        red = np.array([[100, 200]], dtype=np.float64)
        green = np.array([[150, 150]], dtype=np.float64)
        nir = np.array([[200, 100]], dtype=np.float64)
        bands = {"red": red, "green": green, "nir": nir}

        result = _compute_index("ndvi", bands)
        # NDVI = (NIR - Red) / (NIR + Red)
        expected = (nir - red) / (nir + red)
        np.testing.assert_array_almost_equal(result, expected)

    def test_ndwi_computation(self):
        from research_paper.wetland import _compute_index

        red = np.array([[100]], dtype=np.float64)
        green = np.array([[150]], dtype=np.float64)
        nir = np.array([[200]], dtype=np.float64)
        bands = {"red": red, "green": green, "nir": nir}

        result = _compute_index("ndwi", bands)
        # NDWI = (Green - NIR) / (Green + NIR)
        expected = (green - nir) / (green + nir)
        np.testing.assert_array_almost_equal(result, expected)

    def test_division_by_zero_returns_zero(self):
        from research_paper.wetland import _compute_index

        # Both bands zero -> denominator is zero
        red = np.array([[0, 100]], dtype=np.float64)
        green = np.array([[0, 50]], dtype=np.float64)
        nir = np.array([[0, 200]], dtype=np.float64)
        bands = {"red": red, "green": green, "nir": nir}

        result = _compute_index("ndvi", bands)
        self.assertAlmostEqual(result[0, 0], 0.0)
        # Non-zero pixel should compute normally
        self.assertAlmostEqual(result[0, 1], (200 - 100) / (200 + 100))

    def test_result_in_valid_range(self):
        from research_paper.wetland import _compute_index

        rng = np.random.default_rng(42)
        red = rng.uniform(0, 255, size=(10, 10))
        green = rng.uniform(0, 255, size=(10, 10))
        nir = rng.uniform(0, 255, size=(10, 10))
        bands = {"red": red, "green": green, "nir": nir}

        for index_name in ["ndvi", "ndwi"]:
            result = _compute_index(index_name, bands)
            self.assertTrue(np.all(result >= -1.0))
            self.assertTrue(np.all(result <= 1.0))

    def test_unknown_index_raises(self):
        from research_paper.wetland import _compute_index

        bands = {"red": np.zeros((2, 2)), "green": np.zeros((2, 2)), "nir": np.zeros((2, 2))}
        with self.assertRaises(ValueError):
            _compute_index("fake", bands)


# ---------------------------------------------------------------------------
# File I/O tests with rasterio mocks
# ---------------------------------------------------------------------------


class TestComputeSpectralIndicesValidation(unittest.TestCase):
    """Tests for compute_spectral_indices input validation."""

    def test_nonexistent_file_raises(self):
        from research_paper.wetland import compute_spectral_indices

        with self.assertRaises(FileNotFoundError):
            compute_spectral_indices("/nonexistent/naip.tif")

    def test_overwrite_false_existing_raises(self):
        from research_paper.wetland import compute_spectral_indices

        with tempfile.NamedTemporaryFile(suffix=".tif") as src:
            with tempfile.NamedTemporaryFile(suffix="_indices.tif", delete=False) as dst:
                dst_path = dst.name
            try:
                with self.assertRaises(FileExistsError):
                    compute_spectral_indices(
                        src.name, output_path=dst_path, overwrite=False
                    )
            finally:
                os.unlink(dst_path)


class TestExtractSurfaceDepressionsValidation(unittest.TestCase):
    """Tests for extract_surface_depressions input validation."""

    def test_nonexistent_dem_raises(self):
        from research_paper.wetland import extract_surface_depressions

        with self.assertRaises(FileNotFoundError):
            extract_surface_depressions("/nonexistent/dem.tif")

    def test_negative_min_depth_raises(self):
        from research_paper.wetland import extract_surface_depressions

        with tempfile.NamedTemporaryFile(suffix=".tif") as f:
            with self.assertRaises(ValueError):
                extract_surface_depressions(f.name, min_depth=-1.0)

    def test_negative_min_area_raises(self):
        from research_paper.wetland import extract_surface_depressions

        with tempfile.NamedTemporaryFile(suffix=".tif") as f:
            with self.assertRaises(ValueError):
                extract_surface_depressions(f.name, min_area=-10.0)


class TestCreateWetlandCompositeValidation(unittest.TestCase):
    """Tests for create_wetland_composite input validation."""

    def test_empty_naip_paths_raises(self):
        from research_paper.wetland import create_wetland_composite

        with self.assertRaises(ValueError):
            create_wetland_composite(
                naip_paths=[],
                dem_path="/tmp/dem.tif",
                output_path="/tmp/composite.tif",
            )

    def test_nonexistent_dem_raises(self):
        from research_paper.wetland import create_wetland_composite

        with tempfile.NamedTemporaryFile(suffix=".tif") as naip:
            with self.assertRaises(FileNotFoundError):
                create_wetland_composite(
                    naip_paths=naip.name,
                    dem_path="/nonexistent/dem.tif",
                    output_path="/tmp/composite.tif",
                )


# ---------------------------------------------------------------------------
# Integration test with real rasterio (small synthetic data)
# ---------------------------------------------------------------------------


class TestComputeSpectralIndicesIntegration(unittest.TestCase):
    """Integration tests for compute_spectral_indices with real GeoTIFFs."""

    def setUp(self):
        """Create a small synthetic 4-band NAIP GeoTIFF."""
        import rasterio
        from rasterio.transform import from_bounds

        self.tmpdir = tempfile.mkdtemp()
        self.naip_path = os.path.join(self.tmpdir, "naip_test.tif")

        # 4 bands: R, G, B, NIR
        height, width = 16, 16
        transform = from_bounds(-100.5, 47.0, -100.0, 47.5, width, height)
        rng = np.random.default_rng(42)
        data = rng.integers(10, 250, size=(4, height, width), dtype=np.uint8)

        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 4,
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(self.naip_path, "w", **profile) as dst:
            dst.write(data)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_produces_output_file(self):
        from research_paper.wetland import compute_spectral_indices

        out = os.path.join(self.tmpdir, "indices.tif")
        result = compute_spectral_indices(self.naip_path, output_path=out)
        self.assertEqual(result, out)
        self.assertTrue(os.path.exists(out))

    def test_output_has_correct_band_count(self):
        import rasterio

        from research_paper.wetland import compute_spectral_indices

        out = os.path.join(self.tmpdir, "indices.tif")
        compute_spectral_indices(self.naip_path, output_path=out)
        with rasterio.open(out) as src:
            self.assertEqual(src.count, 2)  # NDVI + NDWI

    def test_output_values_in_range(self):
        import rasterio

        from research_paper.wetland import compute_spectral_indices

        out = os.path.join(self.tmpdir, "indices.tif")
        compute_spectral_indices(self.naip_path, output_path=out)
        with rasterio.open(out) as src:
            data = src.read()
            self.assertTrue(np.all(data >= -1.0))
            self.assertTrue(np.all(data <= 1.0))

    def test_single_index(self):
        import rasterio

        from research_paper.wetland import compute_spectral_indices

        out = os.path.join(self.tmpdir, "ndvi_only.tif")
        compute_spectral_indices(self.naip_path, output_path=out, indices=["ndvi"])
        with rasterio.open(out) as src:
            self.assertEqual(src.count, 1)

    def test_preserves_crs(self):
        import rasterio

        from research_paper.wetland import compute_spectral_indices

        out = os.path.join(self.tmpdir, "indices.tif")
        compute_spectral_indices(self.naip_path, output_path=out)
        with rasterio.open(out) as src:
            self.assertEqual(src.crs.to_epsg(), 4326)

    def test_preserves_spatial_extent(self):
        import rasterio

        from research_paper.wetland import compute_spectral_indices

        out = os.path.join(self.tmpdir, "indices.tif")
        compute_spectral_indices(self.naip_path, output_path=out)
        with rasterio.open(self.naip_path) as orig:
            with rasterio.open(out) as result:
                self.assertEqual(orig.bounds, result.bounds)
                self.assertEqual(orig.height, result.height)
                self.assertEqual(orig.width, result.width)


class TestExtractSurfaceDepressionsIntegration(unittest.TestCase):
    """Integration tests for extract_surface_depressions with real GeoTIFFs."""

    def setUp(self):
        """Create a synthetic DEM with a known depression."""
        import rasterio
        from rasterio.transform import from_bounds

        self.tmpdir = tempfile.mkdtemp()
        self.dem_path = os.path.join(self.tmpdir, "dem_test.tif")

        height, width = 32, 32
        transform = from_bounds(-100.5, 47.0, -100.0, 47.5, width, height)

        # Create a bowl-shaped DEM with a depression in the center
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        cy, cx = height / 2, width / 2
        # Radial distance from center
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        # Bowl: high at edges, low at center, with a rim
        dem = 100.0 + 2.0 * r
        # Add a depression: lower the center below the rim
        mask = r < 5
        dem[mask] = 100.0 - (5 - r[mask])  # Depression 0-5m deep

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": height,
            "width": width,
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": -9999.0,
        }
        with rasterio.open(self.dem_path, "w", **profile) as dst:
            dst.write(dem, 1)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_produces_output_file(self):
        from research_paper.wetland import extract_surface_depressions

        out = os.path.join(self.tmpdir, "depressions.tif")
        result = extract_surface_depressions(self.dem_path, output_path=out)
        self.assertEqual(result, out)
        self.assertTrue(os.path.exists(out))

    def test_depression_detected(self):
        import rasterio

        from research_paper.wetland import extract_surface_depressions

        out = os.path.join(self.tmpdir, "depressions.tif")
        extract_surface_depressions(self.dem_path, output_path=out, min_depth=0.0)
        with rasterio.open(out) as src:
            depth = src.read(1)
            # Center of the DEM should have non-zero depression depth
            cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
            self.assertGreater(depth[cy, cx], 0.0)

    def test_output_non_negative(self):
        import rasterio

        from research_paper.wetland import extract_surface_depressions

        out = os.path.join(self.tmpdir, "depressions.tif")
        extract_surface_depressions(self.dem_path, output_path=out, min_depth=0.0)
        with rasterio.open(out) as src:
            depth = src.read(1)
            valid = depth != src.nodata
            self.assertTrue(np.all(depth[valid] >= 0.0))

    def test_min_depth_filters_shallow(self):
        import rasterio

        from research_paper.wetland import extract_surface_depressions

        out_no_filter = os.path.join(self.tmpdir, "no_filter.tif")
        out_filtered = os.path.join(self.tmpdir, "filtered.tif")

        extract_surface_depressions(self.dem_path, output_path=out_no_filter, min_depth=0.0)
        extract_surface_depressions(self.dem_path, output_path=out_filtered, min_depth=2.0)

        with rasterio.open(out_no_filter) as src1, rasterio.open(out_filtered) as src2:
            d1 = src1.read(1)
            d2 = src2.read(1)
            # Filtered version should have fewer non-zero pixels
            nonzero1 = np.count_nonzero(d1[d1 != src1.nodata])
            nonzero2 = np.count_nonzero(d2[d2 != src2.nodata])
            self.assertLessEqual(nonzero2, nonzero1)

    def test_preserves_crs_and_extent(self):
        import rasterio

        from research_paper.wetland import extract_surface_depressions

        out = os.path.join(self.tmpdir, "depressions.tif")
        extract_surface_depressions(self.dem_path, output_path=out)
        with rasterio.open(self.dem_path) as orig, rasterio.open(out) as result:
            self.assertEqual(orig.crs, result.crs)
            self.assertEqual(orig.bounds, result.bounds)

    def test_output_dtype_is_float32(self):
        import rasterio

        from research_paper.wetland import extract_surface_depressions

        out = os.path.join(self.tmpdir, "depressions.tif")
        extract_surface_depressions(self.dem_path, output_path=out)
        with rasterio.open(out) as src:
            self.assertEqual(src.dtypes[0], "float32")


# ---------------------------------------------------------------------------
# Download function tests (mocked network I/O)
# ---------------------------------------------------------------------------


class TestDownloadNAIPTimeseriesValidation(unittest.TestCase):
    """Tests for download_naip_timeseries validation."""

    def test_invalid_bbox_raises(self):
        from research_paper.wetland import download_naip_timeseries

        with self.assertRaises(ValueError):
            download_naip_timeseries(
                bbox=(10.0, 0.0, 5.0, 1.0),  # min_lon > max_lon
                output_dir="/tmp/naip",
            )

    @patch("research_paper.wetland.download_naip")
    def test_delegates_to_download_naip(self, mock_download):
        """Verify it calls download_naip for each year."""
        mock_download.return_value = ["/tmp/naip/file.tif"]

        from research_paper.wetland import download_naip_timeseries

        result = download_naip_timeseries(
            bbox=(-100.5, 47.0, -100.0, 47.5),
            output_dir="/tmp/naip",
            years=[2019, 2023],
        )
        self.assertEqual(mock_download.call_count, 2)
        self.assertIn(2019, result)
        self.assertIn(2023, result)

    @patch("research_paper.wetland.download_naip")
    def test_skips_years_with_no_data(self, mock_download):
        """Years with no NAIP data should be omitted from results."""
        mock_download.side_effect = [
            ["/tmp/naip/2019/file.tif"],  # 2019 has data
            [],  # 2023 has no data
        ]

        from research_paper.wetland import download_naip_timeseries

        result = download_naip_timeseries(
            bbox=(-100.5, 47.0, -100.0, 47.5),
            output_dir="/tmp/naip",
            years=[2019, 2023],
        )
        self.assertIn(2019, result)
        self.assertNotIn(2023, result)


class TestDownload3DEPDEMValidation(unittest.TestCase):
    """Tests for download_3dep_dem validation."""

    def test_invalid_resolution_raises(self):
        from research_paper.wetland import download_3dep_dem

        with self.assertRaises(ValueError):
            download_3dep_dem(
                bbox=(-100.5, 47.0, -100.0, 47.5),
                output_path="/tmp/dem.tif",
                resolution=5,
            )

    def test_valid_resolutions(self):
        """Verify accepted resolution values."""
        from research_paper.wetland import SUPPORTED_3DEP_RESOLUTIONS

        self.assertIn(1, SUPPORTED_3DEP_RESOLUTIONS)
        self.assertIn(3, SUPPORTED_3DEP_RESOLUTIONS)
        self.assertIn(10, SUPPORTED_3DEP_RESOLUTIONS)
        self.assertIn(30, SUPPORTED_3DEP_RESOLUTIONS)


class TestDownloadNWIValidation(unittest.TestCase):
    """Tests for download_nwi validation."""

    def test_invalid_bbox_raises(self):
        from research_paper.wetland import download_nwi

        with self.assertRaises(ValueError):
            download_nwi(
                bbox=(10.0, 0.0, 5.0, 1.0),
                output_path="/tmp/nwi.gpkg",
            )

    def test_overwrite_false_existing_raises(self):
        from research_paper.wetland import download_nwi

        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
            try:
                with self.assertRaises(FileExistsError):
                    download_nwi(
                        bbox=(-100.5, 47.0, -100.0, 47.5),
                        output_path=f.name,
                        overwrite=False,
                    )
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
