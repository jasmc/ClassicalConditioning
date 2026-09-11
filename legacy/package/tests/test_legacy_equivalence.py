from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import analysis_utils
import numpy as np
import pandas as pd

from classical_conditioning.preprocessing.legacy_equivalence import (
    compare_legacy_angles_reader_to_prepare,
    compare_legacy_tracking_preparation_paths,
    first_dataframe_divergence,
    operation_comparisons_to_dict,
    prepare_legacy_tracking_from_raw_txt,
    run_legacy_angle_operation_chain,
)
from classical_conditioning.preprocessing.legacy_v1 import (
    LegacyPreprocessingConfig,
    cumulative_angles_legacy,
)


class LegacyEquivalenceHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tracking = self.root / "fish_mp tail tracking.txt"
        self.config = LegacyPreprocessingConfig(
            angle_point_count=2,
            temporal_filter_frames=3,
            bout_max_window_frames=3,
            bout_min_window_frames=3,
            minimum_bout_duration_frames=0,
            minimum_interbout_frames=0,
        )
        self.tracking.write_text(
            "FrameID x0 y0 angle0 x1 y1 angle1\n"
            "10 0 0 0.0 1 0 0.1\n"
            "11 0 0 0.0 1 0.1 0.2\n"
            "12 0 0 0.0 1 0.2 0.3\n"
            "summary 0 0 0 0 0 0\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prepare_from_raw_keeps_single_summary_drop(self) -> None:
        prepared = prepare_legacy_tracking_from_raw_txt(self.tracking, self.config)
        self.assertEqual(prepared["FrameID"].tolist(), [10, 11, 12])
        self.assertAlmostEqual(
            float(prepared.loc[0, "Angle of point 1 (deg)"]),
            0.1 * 180.0 / math.pi,
            places=5,
        )
        comparison = compare_legacy_tracking_preparation_paths(
            self.tracking,
            self.config,
        )
        self.assertTrue(comparison.equal, comparison.detail)

    def test_legacy_angles_reader_matches_prepare(self) -> None:
        comparison = compare_legacy_angles_reader_to_prepare(
            self.tracking,
            self.config,
        )
        self.assertTrue(comparison.equal, comparison.detail)

    def test_first_divergence_localizes_column_and_row(self) -> None:
        left = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0, 0, 0]})
        right = pd.DataFrame({"a": [1.0, 2.5, 3.0], "b": [0, 0, 0]})
        detail = first_dataframe_divergence(left, right)
        self.assertIsNotNone(detail)
        assert detail is not None
        self.assertIn("column 'a'", detail)
        self.assertIn("row 1", detail)

    def test_operation_chain_reports_first_failure(self) -> None:
        rows = 9
        frame = pd.DataFrame(
            {
                "Trial time (frame) [700 FPS]": np.arange(rows),
                "AbsoluteTime": np.arange(rows, dtype=float),
                "Angle of point 0 (deg)": np.arange(rows, dtype=float),
                "Angle of point 1 (deg)": np.arange(rows, dtype=float) + 1,
            }
        )

        def broken_filter(data: pd.DataFrame) -> pd.DataFrame:
            result = analysis_utils.filter_data(
                data.copy(),
                space_window=3,
                time_window=3,
            )
            result["Angle of point 1 (deg)"] = (
                result["Angle of point 1 (deg)"].to_numpy() + 1.0
            )
            return result

        references = {
            "cumulative_angles_legacy": cumulative_angles_legacy,
            "filter_angles_legacy": broken_filter,
            "calculate_vigor_legacy": lambda data: data,
            "calculate_bout_metric_legacy": lambda data: data,
            "detect_bouts_legacy": lambda data: data,
        }
        comparisons = run_legacy_angle_operation_chain(
            frame,
            self.config,
            references,
        )
        report = operation_comparisons_to_dict(comparisons)
        self.assertFalse(report["all_equal"])
        self.assertEqual(
            report["first_divergence_operation"],
            "filter_angles_legacy",
        )
        self.assertTrue(comparisons[0].equal)


if __name__ == "__main__":
    unittest.main()
