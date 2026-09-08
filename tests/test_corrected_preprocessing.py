from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.preprocessing.corrected_v1 import (
    CorrectedPreprocessConfig,
    calculate_corrected_frames,
    summarize_protocol_timing,
)


class CorrectedPreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CorrectedPreprocessConfig(point_count=4)
        self.frame_ids = np.array([10, 11, 12], dtype=np.int64)
        self.elapsed = np.array([0.0, 1.4, 2.8], dtype=np.float64)
        self.absolute = np.array([1000.0, 1001.4, 1002.8], dtype=np.float64)
        self.x = np.array(
            [
                [5.0, 6.0, 7.0, 8.0],
                [5.0, 6.0, 7.0, 8.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float64,
        )
        self.y = np.zeros_like(self.x)
        self.angles = np.zeros_like(self.x)

    def test_perfect_cadence_marks_first_row_invalid_derivative(self) -> None:
        result, _ = calculate_corrected_frames(
            self.frame_ids,
            self.elapsed,
            self.absolute,
            self.x,
            self.y,
            self.angles,
            config=self.config,
        )
        self.assertFalse(bool(result.loc[0, "derivative_valid"]))
        self.assertTrue(bool(result.loc[1, "derivative_valid"]))
        self.assertTrue(bool(result.loc[2, "derivative_valid"]))
        self.assertEqual(int(result.loc[0, "FrameStep"]), 0)
        self.assertEqual(int(result.loc[1, "FrameStep"]), 1)
        np.testing.assert_allclose(result["x0"], 0.0)
        np.testing.assert_allclose(result["base_x"], 5.0)

    def test_missing_frame_invalidates_cross_gap_derivative(self) -> None:
        frame_ids = np.array([10, 11, 13], dtype=np.int64)
        elapsed = np.array([0.0, 1.4, 4.2], dtype=np.float64)
        absolute = elapsed + 1000.0
        result, _ = calculate_corrected_frames(
            frame_ids,
            elapsed,
            absolute,
            self.x,
            self.y,
            self.angles,
            config=self.config,
        )
        self.assertTrue(bool(result.loc[1, "derivative_valid"]))
        self.assertFalse(bool(result.loc[2, "derivative_valid"]))
        self.assertTrue(bool(result.loc[2, "long_gap_invalid"]))
        self.assertEqual(int(result.loc[2, "FrameStep"]), 2)

    def test_long_measured_interval_invalidates_even_when_frame_step_is_one(
        self,
    ) -> None:
        elapsed = np.array([0.0, 1.4, 20.0], dtype=np.float64)
        absolute = elapsed + 1000.0
        result, _ = calculate_corrected_frames(
            self.frame_ids,
            elapsed,
            absolute,
            self.x,
            self.y,
            self.angles,
            config=self.config,
        )
        self.assertTrue(bool(result.loc[1, "derivative_valid"]))
        self.assertFalse(bool(result.loc[2, "derivative_valid"]))
        self.assertTrue(bool(result.loc[2, "long_gap_invalid"]))

    def test_duplicate_frame_ids_are_rejected(self) -> None:
        frame_ids = np.array([10, 10, 11], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            calculate_corrected_frames(
                frame_ids,
                self.elapsed,
                self.absolute,
                self.x,
                self.y,
                self.angles,
                config=self.config,
            )

    def test_nonmonotonic_elapsed_time_is_rejected(self) -> None:
        elapsed = np.array([0.0, 2.0, 1.0], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "ElapsedTime"):
            calculate_corrected_frames(
                self.frame_ids,
                elapsed,
                self.absolute,
                self.x,
                self.y,
                self.angles,
                config=self.config,
            )

    def test_chunk_carry_preserves_gap_across_batch_boundary(self) -> None:
        first, state = calculate_corrected_frames(
            np.array([10, 11], dtype=np.int64),
            np.array([0.0, 1.4], dtype=np.float64),
            np.array([1000.0, 1001.4], dtype=np.float64),
            self.x[:2],
            self.y[:2],
            self.angles[:2],
            config=self.config,
        )
        second, _ = calculate_corrected_frames(
            np.array([13], dtype=np.int64),
            np.array([4.2], dtype=np.float64),
            np.array([1004.2], dtype=np.float64),
            self.x[2:3],
            self.y[2:3],
            self.angles[2:3],
            config=self.config,
            previous=state,
        )
        self.assertTrue(bool(first.loc[1, "derivative_valid"]))
        self.assertFalse(bool(second.loc[0, "derivative_valid"]))
        self.assertTrue(bool(second.loc[0, "long_gap_invalid"]))

    def test_rigid_translation_is_removed_from_stored_coordinates(self) -> None:
        translated_x = self.x + np.arange(3)[:, None] * 10
        translated_y = self.y + np.arange(3)[:, None] * 4
        result, _ = calculate_corrected_frames(
            self.frame_ids,
            self.elapsed,
            self.absolute,
            translated_x,
            translated_y,
            self.angles,
            config=self.config,
        )
        np.testing.assert_allclose(result["x0"], 0.0)
        np.testing.assert_allclose(result["y0"], 0.0)
        np.testing.assert_allclose(result["base_x"], [5.0, 15.0, 25.0])

    def test_missing_points_reduce_valid_fraction(self) -> None:
        x = self.x.copy()
        y = self.y.copy()
        x[1, -1] = np.nan
        y[1, -1] = np.nan
        result, _ = calculate_corrected_frames(
            self.frame_ids,
            self.elapsed,
            self.absolute,
            x,
            y,
            self.angles,
            config=self.config,
        )
        self.assertEqual(int(result.loc[1, "valid_point_count"]), 3)
        self.assertAlmostEqual(float(result.loc[1, "valid_point_fraction"]), 0.75)
        self.assertFalse(bool(result.loc[1, "frame_valid"]))
        self.assertFalse(bool(result.loc[1, "derivative_valid"]))
        # Next derivative also invalid because previous frame was not frame_valid.
        self.assertFalse(bool(result.loc[2, "derivative_valid"]))

    def test_protocol_events_outside_and_between_frames(self) -> None:
        protocol = pd.DataFrame(
            {
                "Type": ["Cycle", "Cycle", "Cycle", "Cycle"],
                "Beg": [-1.0, 1000.0, 1000.7, 1005.0],
                "End": [0.0, 1001.0, 1001.7, 1006.0],
            }
        )
        camera_ids = np.array([1, 2, 3], dtype=np.int64)
        camera_absolute = np.array([1000.0, 1001.4, 1002.8], dtype=np.float64)
        summary = summarize_protocol_timing(protocol, camera_ids, camera_absolute)
        self.assertEqual(summary["event_count"], 4)
        self.assertEqual(summary["events_outside_acquisition"], 2)
        self.assertEqual(summary["events_on_observed_frames"], 1)
        self.assertEqual(summary["events_between_observed_frames"], 1)

    def test_frozen_config_rejects_enabled_interpolation(self) -> None:
        bad = CorrectedPreprocessConfig(
            point_count=4,
            interpolation_enabled=True,
        )
        with self.assertRaisesRegex(ValueError, "interpolation_enabled=False"):
            calculate_corrected_frames(
                self.frame_ids,
                self.elapsed,
                self.absolute,
                self.x,
                self.y,
                self.angles,
                config=bad,
            )


if __name__ == "__main__":
    unittest.main()
