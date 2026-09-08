from __future__ import annotations

import unittest

import numpy as np

from classical_conditioning.preprocessing.candidates_v1 import (
    CANDIDATE_COLUMNS,
    CandidateMetricConfig,
    _geometry_agreement,
    _validate_frame_order,
    calculate_candidate_metrics,
)


class CandidateMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CandidateMetricConfig(point_count=4)
        self.frame_ids = np.array([1, 2, 3])
        self.time = np.array([0.0, 1.0, 2.0])
        self.x = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        )
        self.y = np.zeros_like(self.x)
        self.angles = np.zeros_like(self.x)

    def test_stationary_tail_has_zero_activity_after_first_frame(self) -> None:
        result, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            self.x,
            self.y,
            self.angles,
            config=self.config,
        )
        self.assertFalse(result.loc[0, "valid_derivative"])
        for column in CANDIDATE_COLUMNS:
            self.assertTrue(np.isnan(result.loc[0, column]))
            np.testing.assert_allclose(result.loc[1:, column], 0.0)

    def test_rigid_translation_is_removed_by_tail_base_centering(self) -> None:
        translated_x = self.x + np.arange(3)[:, None] * 10
        translated_y = self.y + np.arange(3)[:, None] * 4
        result, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            translated_x,
            translated_y,
            self.angles,
            config=self.config,
        )
        np.testing.assert_allclose(
            result.loc[1:, "whole_tail_xy_rms_speed_px_per_ms"],
            0.0,
        )
        np.testing.assert_allclose(
            result.loc[1:, "whole_tail_xy_mean_speed_px_per_ms"],
            0.0,
        )

    def test_one_point_motion_produces_positive_xy_metrics(self) -> None:
        moved_y = self.y.copy()
        moved_y[1:, -1] = [1.0, 2.0]
        result, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            self.x,
            moved_y,
            self.angles,
            config=self.config,
        )
        self.assertTrue(
            result.loc[1:, "whole_tail_xy_rms_speed_px_per_ms"].gt(0).all()
        )
        self.assertTrue(
            result.loc[1:, "whole_tail_xy_mean_speed_px_per_ms"].gt(0).all()
        )

    def test_segment_motion_is_not_cancelled_in_manuscript_sum(self) -> None:
        moved_x = self.x.copy()
        moved_y = self.y.copy()
        orientations = np.array([0.0, 0.5, -0.5])
        moved_x[1] = np.concatenate(
            [[0.0], np.cumsum(np.cos(orientations))]
        )
        moved_y[1] = np.concatenate(
            [[0.0], np.cumsum(np.sin(orientations))]
        )
        result, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            moved_x,
            moved_y,
            self.angles,
            config=self.config,
        )
        self.assertGreater(
            result.loc[1, "segment_absolute_angular_speed_sum_rad_per_ms"],
            0,
        )

    def test_opposite_segment_rotations_cancel_in_legacy_distal_speed_but_not_manuscript_sum(self) -> None:
        # Mathematical identity: |sum dtheta| != sum |dphi|
        # When segments rotate in opposite directions, the distal cumulative angle
        # change cancels to zero, but segment angular speed sum retains the full motion.
        moved_x = self.x.copy()
        moved_y = self.y.copy()
        moved_angles = self.angles.copy()
        orientations = np.array([0.0, 0.5, -0.5])
        moved_x[1] = np.concatenate(
            [[0.0], np.cumsum(np.cos(orientations))]
        )
        moved_y[1] = np.concatenate(
            [[0.0], np.cumsum(np.sin(orientations))]
        )
        moved_angles[1] = np.array([0.0, 0.5, -0.5, 0.0])
        result, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            moved_x,
            moved_y,
            moved_angles,
            config=self.config,
        )
        self.assertAlmostEqual(
            float(result.loc[1, "legacy_distal_angular_speed_rad_per_ms"]),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(result.loc[1, "segment_absolute_angular_speed_sum_rad_per_ms"]),
            1.0,
            places=6,
        )

    def test_frame_gap_invalidates_only_cross_gap_derivative(self) -> None:
        result, _ = calculate_candidate_metrics(
            np.array([1, 3, 4]),
            self.time,
            self.x,
            self.y,
            self.angles,
            config=self.config,
        )
        self.assertEqual(
            result["valid_derivative"].tolist(),
            [False, False, True],
        )

    def test_missing_distal_point_reduces_valid_tail_fraction(self) -> None:
        x = self.x.copy()
        y = self.y.copy()
        x[1, -1] = np.nan
        y[1, -1] = np.nan
        result, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            x,
            y,
            self.angles,
            config=self.config,
        )
        self.assertLess(result.loc[1, "xy_valid_tail_fraction"], 1.0)
        self.assertLess(result.loc[1, "angular_valid_tail_fraction"], 1.0)

    def test_duplicate_or_reversed_frames_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            calculate_candidate_metrics(
                np.array([1, 1, 2]),
                self.time,
                self.x,
                self.y,
                self.angles,
                config=self.config,
            )

    def test_cross_chunk_frame_order_is_rejected(self) -> None:
        _, state = calculate_candidate_metrics(
            self.frame_ids[:2],
            self.time[:2],
            self.x[:2],
            self.y[:2],
            self.angles[:2],
            config=self.config,
        )
        with self.assertRaisesRegex(ValueError, "across chunk boundaries"):
            calculate_candidate_metrics(
                np.array([2]),
                np.array([2.0]),
                self.x[2:],
                self.y[2:],
                self.angles[2:],
                config=self.config,
                previous=state,
            )

    def test_raw_tracking_order_validation_includes_unmatched_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _validate_frame_order(np.array([1, 3, 2]), None)

    def test_degenerate_segments_are_excluded_from_geometry_validation(self) -> None:
        x = self.x.copy()
        y = self.y.copy()
        x[:, 2] = x[:, 1]
        y[:, 2] = y[:, 1]
        _, eligible, total = _geometry_agreement(x, y, self.angles)
        self.assertLess(eligible, total)

    def test_curvature_rate_includes_segment_length_change(self) -> None:
        angles = self.angles.copy()
        angles[:, 1] = 0.5
        stretched_x = self.x.copy()
        stretched_x[1:, 2:] += 1.0
        result, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            stretched_x,
            self.y,
            angles,
            config=self.config,
        )
        self.assertGreater(
            result.loc[1, "curvature_change_rms_rad_per_px_per_ms"],
            0,
        )

    def test_chunk_carry_matches_single_chunk(self) -> None:
        full, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            self.x,
            self.y,
            self.angles,
            config=self.config,
        )
        first, state = calculate_candidate_metrics(
            self.frame_ids[:2],
            self.time[:2],
            self.x[:2],
            self.y[:2],
            self.angles[:2],
            config=self.config,
        )
        second, _ = calculate_candidate_metrics(
            self.frame_ids[2:],
            self.time[2:],
            self.x[2:],
            self.y[2:],
            self.angles[2:],
            config=self.config,
            previous=state,
        )
        combined = np.concatenate(
            [
                first[list(CANDIDATE_COLUMNS)].to_numpy(),
                second[list(CANDIDATE_COLUMNS)].to_numpy(),
            ]
        )
        np.testing.assert_allclose(
            combined,
            full[list(CANDIDATE_COLUMNS)].to_numpy(),
            equal_nan=True,
        )


if __name__ == "__main__":
    unittest.main()
