from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_utils
import data_io
from general_configuration import config


class LegacyPreprocessingCharacterizationTests(unittest.TestCase):
    def test_tracking_reader_selects_angles_and_drops_final_row(self) -> None:
        columns = (
            ["FrameID"]
            + [value for index in range(16) for value in (f"x{index}", f"y{index}")]
            + [f"angle{index}" for index in range(16)]
        )
        rows = []
        for frame_id in (10, 11, 12):
            coordinates = [float(index) for index in range(32)]
            angles = [0.1 * index for index in range(16)]
            rows.append([frame_id, *coordinates, *angles])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fish_mp tail tracking.txt"
            frame = pd.DataFrame(rows, columns=columns)
            frame.to_csv(path, sep=" ", index=False)
            result = data_io.read_tail_tracking_data(path)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result), 2)
        self.assertEqual(result["FrameID"].tolist(), [10, 11])
        self.assertNotIn("x0", result.columns)
        self.assertNotIn("y0", result.columns)
        self.assertIn("Angle of point 15 (deg)", result.columns)
        self.assertIn("Original frame number", result.columns)
        self.assertAlmostEqual(
            result.loc[0, "Angle of point 1 (deg)"],
            0.1 * 180 / np.pi,
            places=5,
        )

    def test_spatial_filter_window_does_not_change_current_output(self) -> None:
        time_column = config.time_trial_frame_label
        source = pd.DataFrame(
            {
                time_column: np.arange(5),
                "Angle of point 0 (deg)": [0.0, 1.0, 2.0, 3.0, 4.0],
                "Angle of point 1 (deg)": [10.0, 11.0, 12.0, 13.0, 14.0],
                "Angle of point 2 (deg)": [20.0, 21.0, 22.0, 23.0, 24.0],
            }
        )

        window_one = analysis_utils.filter_data(
            source.copy(),
            space_window=1,
            time_window=1,
        )
        window_three = analysis_utils.filter_data(
            source.copy(),
            space_window=3,
            time_window=1,
        )
        pd.testing.assert_frame_equal(window_one, window_three)

    def test_current_vigor_is_absolute_single_signal_derivative(self) -> None:
        angles = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float64)
        vigor = analysis_utils.calculate_vigor_fast_pure_numpy(
            angles,
            framerate=1_000,
        )
        np.testing.assert_array_equal(
            vigor,
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        )

    def test_secondary_bout_threshold_does_not_change_current_output(self) -> None:
        metric = "Vigor for bout detection (deg/ms)"
        source = pd.DataFrame({metric: [0.0, 5.0, 5.0, 0.0, 0.0]})
        low_threshold = analysis_utils.find_beg_and_end_of_bouts(
            source.copy(),
            thr1=4.0,
            min_dur=0,
            min_gap=0,
            thr2=1.0,
        )
        impossible_threshold = analysis_utils.find_beg_and_end_of_bouts(
            source.copy(),
            thr1=4.0,
            min_dur=0,
            min_gap=0,
            thr2=1_000.0,
        )
        pd.testing.assert_series_equal(
            low_threshold["Bout"],
            impossible_threshold["Bout"],
        )

    def test_frame_loss_flag_does_not_inspect_frame_id_gaps(self) -> None:
        camera = pd.DataFrame(
            {
                "FrameID": [100, 101, 103, 104, 105, 106],
                "ElapsedTime": np.arange(6, dtype=float) * (1_000 / 700),
                "AbsoluteTime": np.arange(6, dtype=np.int64) + 1_000,
            }
        )
        _, _, has_lost_frames = analysis_utils.framerate_and_reference_frame(
            camera,
            "synthetic",
        )
        self.assertFalse(has_lost_frames)


if __name__ == "__main__":
    unittest.main()
