from __future__ import annotations

import unittest

import numpy as np

from classical_conditioning.preprocessing.candidate_metric_kernel import (
    CANDIDATE_COLUMNS,
    METRIC_DEFINITIONS,
    SUPERSEDED_METRICS,
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

    def test_active_metric_shortlist_supersedes_redundant_candidates(self) -> None:
        self.assertEqual(
            CANDIDATE_COLUMNS,
            (
                "tail_length_weighted_angular_l1_rad_per_ms",
                "all_segment_angular_rms_rad_per_ms",
                "whole_tail_xy_mean_speed_tail_lengths_per_ms",
                "legacy_distal_angular_speed_rad_per_ms",
            ),
        )
        self.assertEqual(set(METRIC_DEFINITIONS), set(CANDIDATE_COLUMNS))
        self.assertIn("whole_tail_xy_rms_speed_px_per_ms", SUPERSEDED_METRICS)
        self.assertIn("curvature_change_rms_rad_per_px_per_ms", SUPERSEDED_METRICS)

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
            result.loc[1:, "whole_tail_xy_mean_speed_tail_lengths_per_ms"],
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
            result.loc[
                1:, "whole_tail_xy_mean_speed_tail_lengths_per_ms"
            ].gt(0).all()
        )

    def test_segment_motion_is_not_cancelled_in_angular_l1(self) -> None:
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
            result.loc[1, "tail_length_weighted_angular_l1_rad_per_ms"],
            0,
        )

    def test_opposite_segment_rotations_cancel_in_legacy_but_not_angular_l1(self) -> None:
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
            float(result.loc[1, "tail_length_weighted_angular_l1_rad_per_ms"]),
            1.0 / 3.0,
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

    def test_xy_mean_is_invariant_to_uniform_spatial_scale(self) -> None:
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
        scaled, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            self.x * 4.0,
            moved_y * 4.0,
            self.angles,
            config=self.config,
        )
        np.testing.assert_allclose(
            result.loc[1:, "whole_tail_xy_mean_speed_tail_lengths_per_ms"],
            scaled.loc[1:, "whole_tail_xy_mean_speed_tail_lengths_per_ms"],
        )

    def test_weighted_angular_l1_is_point_density_normalized(self) -> None:
        moved_x = self.x.copy()
        moved_y = self.y.copy()
        orientations = np.array([0.25, 0.25, 0.25])
        moved_x[1] = np.concatenate([[0.0], np.cumsum(np.cos(orientations))])
        moved_y[1] = np.concatenate([[0.0], np.cumsum(np.sin(orientations))])
        result, _ = calculate_candidate_metrics(
            self.frame_ids,
            self.time,
            moved_x,
            moved_y,
            self.angles,
            config=self.config,
        )
        self.assertAlmostEqual(
            float(result.loc[1, "tail_length_weighted_angular_l1_rad_per_ms"]),
            0.25,
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
