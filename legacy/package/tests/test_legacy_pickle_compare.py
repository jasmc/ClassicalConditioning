from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from classical_conditioning.preprocessing.legacy_pickle_compare import (
    classify_original_frame_mapping,
    compare_legacy_pickle_to_parquet,
)
from classical_conditioning.preprocessing.legacy_v1 import (
    interpolate_legacy,
    synchronize_legacy,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Original frame number": [1.0, 2.0],
            "Trial time (frame) [700 FPS]": pd.Series([0, 1], dtype="int64"),
            "CS beg": pd.Series([10, 10], dtype="category"),
            "CS end": pd.Series([20, 20], dtype="category"),
            "US beg": pd.Series([30, 30], dtype="category"),
            "US end": pd.Series([40, 40], dtype="category"),
            "Trial type": pd.Series(["CS", "CS"], dtype="category"),
            "Trial number": pd.Series([1, 1], dtype="category"),
            "Block name": pd.Series([pd.NA, "Acquisition 1"], dtype="category"),
            "Vigor (deg/ms)": pd.Series([0.1, 0.2], dtype="float32"),
            "Scaled vigor (AU)": pd.Series([1.0, 2.0], dtype="float32"),
            "Bout beg": [False, True],
            "Bout end": [False, False],
            "Bout": [False, True],
            "Angle of point 0 (deg)": pd.Series([0.0, 1.0], dtype="float32"),
            "Day": ["20221115", "20221115"],
            "Fish no.": ["04", "04"],
            "Exp.": ["delay", "delay"],
            "ProtocolRig": ["black-1", "black-1"],
            "Strain": ["mitfaminusminus", "mitfaminusminus"],
            "Age (dpf)": [6, 6],
        }
    )


class LegacyPickleCompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_equal_pickle_and_parquet_after_storage_normalization(self) -> None:
        pickle_frame = _sample_frame().set_index(
            ["Strain", "Age (dpf)", "Exp.", "ProtocolRig", "Day", "Fish no."]
        )
        pickle_path = self.root / "fish.pkl"
        pickle_frame.to_pickle(pickle_path, compression="gzip")

        parquet_frame = _sample_frame()
        parquet_frame["Trial time (frame) [700 FPS]"] = parquet_frame[
            "Trial time (frame) [700 FPS]"
        ].astype("int32")
        parquet_frame["CS beg"] = parquet_frame["CS beg"].astype("int32")
        parquet_frame["CS end"] = parquet_frame["CS end"].astype("int32")
        parquet_frame["US beg"] = parquet_frame["US beg"].astype("int32")
        parquet_frame["US end"] = parquet_frame["US end"].astype("int32")
        parquet_frame["Trial number"] = parquet_frame["Trial number"].astype("int32")
        parquet_frame["Block name"] = parquet_frame["Block name"].astype("string").fillna("")
        parquet_path = self.root / "samples_legacy-v1.parquet"
        parquet_frame.to_parquet(parquet_path, index=False)

        report = self.root / "compare.json"
        result = compare_legacy_pickle_to_parquet(
            pickle_path,
            parquet_path,
            report,
        )
        self.assertTrue(result.row_counts_equal)
        self.assertTrue(result.all_scientific_columns_equal)
        self.assertIsNone(result.first_divergence)
        self.assertTrue(report.is_file())

    def test_later_batches_keep_row_alignment(self) -> None:
        pickle_frame = pd.concat([_sample_frame(), _sample_frame().iloc[[0]]], ignore_index=True)
        pickle_frame = pickle_frame.set_index(
            ["Strain", "Age (dpf)", "Exp.", "ProtocolRig", "Day", "Fish no."]
        )
        pickle_path = self.root / "fish.pkl"
        pickle_frame.to_pickle(pickle_path, compression="gzip")
        parquet_frame = pd.concat(
            [_sample_frame(), _sample_frame().iloc[[0]]],
            ignore_index=True,
        )
        parquet_path = self.root / "samples_legacy-v1.parquet"
        parquet_frame.to_parquet(parquet_path, index=False)
        result = compare_legacy_pickle_to_parquet(
            pickle_path,
            parquet_path,
            self.root / "compare.json",
            batch_size=1,
        )
        self.assertTrue(result.all_scientific_columns_equal)
        self.assertIsNone(result.first_divergence)

    def test_reports_first_scientific_mismatch(self) -> None:
        pickle_frame = _sample_frame().set_index(
            ["Strain", "Age (dpf)", "Exp.", "ProtocolRig", "Day", "Fish no."]
        )
        pickle_path = self.root / "fish.pkl"
        pickle_frame.to_pickle(pickle_path, compression="gzip")

        parquet_frame = _sample_frame()
        parquet_frame.loc[1, "Vigor (deg/ms)"] = 9.9
        parquet_path = self.root / "samples_legacy-v1.parquet"
        parquet_frame.to_parquet(parquet_path, index=False)

        result = compare_legacy_pickle_to_parquet(
            pickle_path,
            parquet_path,
            self.root / "compare.json",
        )
        self.assertFalse(result.all_scientific_columns_equal)
        self.assertIn("Vigor (deg/ms)", result.first_divergence or "")

    def test_interpolate_original_frame_advances_at_predicted_over_expected(self) -> None:
        predicted = 702.56553238286
        expected = 700.0
        n = 200
        tracking = pd.DataFrame(
            {
                "FrameID": np.arange(1000, 1000 + n, dtype=np.int32),
                "Original frame number": np.arange(1000, 1000 + n, dtype=np.int32),
                "Angle of point 0 (deg)": np.linspace(0, 1, n, dtype=np.float32),
            }
        )
        camera = pd.DataFrame(
            {
                "FrameID": np.arange(1000, 1000 + n, dtype=np.int32),
                "ElapsedTime": np.arange(n, dtype=float) * (1000 / predicted),
                "AbsoluteTime": np.arange(n, dtype=float) * (1000 / predicted),
            }
        )
        merged = synchronize_legacy(tracking, camera)
        interpolated = interpolate_legacy(merged, expected, predicted)
        slope = float(
            np.polyfit(
                interpolated["Trial time (frame) [700 FPS]"].to_numpy(dtype=float)[:100],
                interpolated["Original frame number"].to_numpy(dtype=float)[:100],
                1,
            )[0]
        )
        self.assertAlmostEqual(slope, predicted / expected, places=10)

    def test_classifies_reciprocal_pickle_original_frame_slope(self) -> None:
        predicted = 702.56553238286
        expected = 700.0
        times = np.arange(-3000, 3001, dtype=np.int32)
        onset = 500_000.0
        # Shared physical vigor as a function of Original frame.
        parquet_orig = onset + times * (predicted / expected)
        pickle_orig = onset + times * (expected / predicted)
        vigor_from_orig = np.sin(parquet_orig / 500.0)
        # Parquet stores vigor on physics Original; pickle stores the same physical
        # vigor but indexed by reciprocal Original/trial-time warp.
        parquet = pd.DataFrame(
            {
                "Trial time (frame) [700 FPS]": times,
                "Original frame number": parquet_orig,
                "Trial type": ["CS"] * len(times),
                "Trial number": [1] * len(times),
                "Vigor (deg/ms)": vigor_from_orig,
            }
        )
        pickle_vigor = np.sin(pickle_orig / 500.0)
        pickle = pd.DataFrame(
            {
                "Trial time (frame) [700 FPS]": times,
                "Original frame number": pickle_orig,
                "Trial type": ["CS"] * len(times),
                "Trial number": [1] * len(times),
                "Vigor (deg/ms)": pickle_vigor,
            }
        )
        diagnosis = classify_original_frame_mapping(pickle, parquet)
        self.assertEqual(diagnosis["status"], "ok")
        self.assertIn("reciprocal", diagnosis["classification"])
        self.assertTrue(diagnosis["pickle_matches_reciprocal_rate"])
        self.assertTrue(diagnosis["parquet_matches_physics_rate"])
        self.assertAlmostEqual(diagnosis["onset_original_diff_at_trial_time_0"], 0.0, places=6)
        self.assertGreater(
            diagnosis["vigor_correlation_after_rate_warp"],
            diagnosis["full_trial_vigor_correlation"],
        )
        self.assertGreater(diagnosis["lag_versus_predicted_rate_warp_correlation"], 0.9)
