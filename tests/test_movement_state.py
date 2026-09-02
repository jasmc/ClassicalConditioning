from __future__ import annotations

import unittest

import numpy as np

from classical_conditioning.analysis.movement_state import (
    MovementCalibrationConfig,
    _positive_control,
    calibrate_quiet_window_thresholds,
    detect_hysteresis_bouts,
    evaluate_smoothing_sensitivity,
    resolve_candidate_metric_source,
    smooth_contiguous_median,
)


class MovementStateTests(unittest.TestCase):
    def test_metric_movement_recipe_pairing_is_frozen(self) -> None:
        development = resolve_candidate_metric_source(
            metric_recipe="tail-candidate-development-v1"
        )
        self.assertEqual(development.movement_recipe, "movement-candidate-v1")
        corrected = resolve_candidate_metric_source(
            movement_recipe="movement-candidate-corrected-v1"
        )
        self.assertEqual(
            corrected.metric_recipe,
            "tail-candidate-corrected-v1",
        )
        self.assertEqual(
            corrected.comparison_recipe,
            "candidate-metric-comparison-corrected-v1",
        )
        self.assertEqual(
            corrected.runner_recipe,
            "candidate-corrected-runner-v1",
        )
        with self.assertRaisesRegex(ValueError, "frozen pairing"):
            resolve_candidate_metric_source(
                metric_recipe="tail-candidate-development-v1",
                movement_recipe="movement-candidate-corrected-v1",
            )
    def test_smoothing_does_not_cross_frame_gap(self) -> None:
        values = np.array([0.0, 0.0, 100.0, 10.0, 10.0, 10.0])
        frame_steps = np.array([0, 1, 1, 2, 1, 1])
        smoothed = smooth_contiguous_median(
            values,
            frame_steps,
            window_samples=3,
        )
        self.assertTrue(np.isnan(smoothed[2]))
        self.assertTrue(np.isnan(smoothed[3]))
        self.assertEqual(smoothed[4], 10.0)

    def test_quiet_window_calibration_is_deterministic_and_ordered(self) -> None:
        rng = np.random.default_rng(10)
        quiet = rng.normal(1.0, 0.05, 1_000)
        active = rng.normal(5.0, 0.5, 1_000)
        values = np.concatenate([quiet, active])
        elapsed = np.arange(len(values), dtype=float)
        config = MovementCalibrationConfig(
            quiet_window_ms=100.0,
            quiet_window_fraction=0.5,
            low_threshold_quantile=0.90,
            high_threshold_quantile=0.99,
        )
        calibration = calibrate_quiet_window_thresholds(
            values,
            elapsed,
            np.ones(len(values), dtype=bool),
            config=config,
        )
        self.assertLess(
            calibration["low_threshold"],
            calibration["high_threshold"],
        )
        self.assertLess(calibration["high_threshold"], 2.0)
        self.assertEqual(
            calibration["quiet_window_count"],
            int(np.ceil(calibration["total_window_count"] * 0.5)),
        )

    def test_quiet_window_ties_do_not_expand_selection_or_connect_zeros(self) -> None:
        values = np.zeros(1_000)
        values[500] = 1.0
        elapsed = np.arange(len(values), dtype=float)
        config = MovementCalibrationConfig(
            quiet_window_ms=10.0,
            quiet_window_fraction=0.2,
            low_threshold_quantile=0.90,
            high_threshold_quantile=0.99,
        )
        with self.assertRaisesRegex(ValueError, "Invalid candidate thresholds"):
            calibrate_quiet_window_thresholds(
                values,
                elapsed,
                np.ones(len(values), dtype=bool),
                config=config,
            )

    def test_hysteresis_keeps_weak_support_connected_to_strong_seed(self) -> None:
        values = np.array([0.0, 2.0, 5.0, 2.0, 0.0])
        moving, bout_ids = detect_hysteresis_bouts(
            values,
            np.arange(5, dtype=float) * 10,
            np.full(5, 10.0),
            np.array([0, 1, 1, 1, 1]),
            np.ones(5, dtype=bool),
            low_threshold=1.0,
            high_threshold=4.0,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertEqual(moving.tolist(), [False, True, True, True, False])
        self.assertEqual(int(bout_ids.max()), 1)

    def test_short_bout_is_removed(self) -> None:
        moving, _ = detect_hysteresis_bouts(
            np.array([0.0, 5.0, 0.0]),
            np.array([0.0, 10.0, 20.0]),
            np.full(3, 10.0),
            np.array([0, 1, 1]),
            np.ones(3, dtype=bool),
            low_threshold=1.0,
            high_threshold=4.0,
            minimum_bout_duration_ms=20.0,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertFalse(moving.any())

    def test_short_valid_gap_is_merged_but_frame_gap_is_not(self) -> None:
        values = np.array([5.0, 5.0, 0.0, 5.0, 5.0])
        elapsed = np.arange(5, dtype=float) * 5
        valid_steps = np.array([0, 1, 1, 1, 1])
        merged, _ = detect_hysteresis_bouts(
            values,
            elapsed,
            np.full(5, 5.0),
            valid_steps,
            np.ones(5, dtype=bool),
            low_threshold=1.0,
            high_threshold=4.0,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=15.0,
        )
        self.assertTrue(merged.all())

        gap_steps = np.array([0, 1, 2, 1, 1])
        not_merged, _ = detect_hysteresis_bouts(
            values,
            elapsed,
            np.full(5, 5.0),
            gap_steps,
            np.ones(5, dtype=bool),
            low_threshold=1.0,
            high_threshold=4.0,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=15.0,
        )
        self.assertFalse(not_merged[2])

    def test_values_equal_to_low_threshold_are_not_weak_support(self) -> None:
        moving, _ = detect_hysteresis_bouts(
            np.array([0.0, 1.0, 5.0, 1.0, 0.0]),
            np.arange(5, dtype=float) * 10,
            np.full(5, 10.0),
            np.array([0, 1, 1, 1, 1]),
            np.ones(5, dtype=bool),
            low_threshold=1.0,
            high_threshold=4.0,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertEqual(moving.tolist(), [False, False, True, False, False])

    def test_exact_legacy_duration_boundaries_are_retained_and_merged(self) -> None:
        interval = 1_000 / 700
        moving_values = np.concatenate(
            [np.zeros(2), np.full(40, 5.0), np.zeros(2)]
        )
        moving, _ = detect_hysteresis_bouts(
            moving_values,
            np.arange(len(moving_values)) * interval,
            np.full(len(moving_values), interval),
            np.concatenate([[0], np.ones(len(moving_values) - 1)]),
            np.ones(len(moving_values), dtype=bool),
            low_threshold=1.0,
            high_threshold=4.0,
            minimum_bout_duration_ms=40 / 700 * 1_000,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertEqual(int(moving.sum()), 40)

        gap_values = np.concatenate(
            [np.full(20, 5.0), np.zeros(10), np.full(20, 5.0)]
        )
        merged, _ = detect_hysteresis_bouts(
            gap_values,
            np.arange(len(gap_values)) * interval,
            np.full(len(gap_values), interval),
            np.concatenate([[0], np.ones(len(gap_values) - 1)]),
            np.ones(len(gap_values), dtype=bool),
            low_threshold=1.0,
            high_threshold=4.0,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=10 / 700 * 1_000,
        )
        self.assertTrue(merged.all())

    def test_positive_control_coverage_uses_frame_span(self) -> None:
        result = _positive_control(
            np.array([0, 1, 3, 4]),
            np.array([0, 1, 3, 4]),
            np.ones(4, dtype=bool),
            np.ones(4, dtype=bool),
            __import__("pandas").DataFrame(
                {"Type": ["Reinforcer"], "Beg": [0], "End": [1]}
            ),
            MovementCalibrationConfig(positive_control_window_ms=4.0),
        )
        self.assertEqual(result["post_us_evaluated_event_count"], 1)
        self.assertEqual(result["post_us_valid_fraction"], 0.75)

    def test_smoothing_sensitivity_returns_only_compact_summaries(self) -> None:
        rows = 1_000
        frame = __import__("pandas").DataFrame(
            {
                "FrameID": np.arange(rows),
                "ElapsedTime": np.arange(rows, dtype=float),
                "AbsoluteTime": np.arange(rows, dtype=np.int64),
                "FrameStep": np.concatenate([[0], np.ones(rows - 1)]),
                "DeltaTimeMs": np.ones(rows),
                "valid_derivative": np.concatenate([[False], np.ones(rows - 1, dtype=bool)]),
                "xy_valid_tail_fraction": np.ones(rows),
                "angular_valid_tail_fraction": np.ones(rows),
                "curvature_valid_tail_fraction": np.ones(rows),
                **{
                    column: np.linspace(0.1, 1.0, rows)
                    + 0.01 * np.sin(np.arange(rows))
                    for column in __import__(
                        "classical_conditioning.preprocessing.candidates_v1",
                        fromlist=["CANDIDATE_COLUMNS"],
                    ).CANDIDATE_COLUMNS
                },
            }
        )
        protocol = __import__("pandas").DataFrame(
            {"Type": ["Reinforcer"], "Beg": [500], "End": [510]}
        )
        result = evaluate_smoothing_sensitivity(
            frame,
            protocol,
            smoothing_windows_ms=(0.0, 10.0),
            base_config=MovementCalibrationConfig(
                quiet_window_ms=100.0,
                low_threshold_quantile=0.90,
                high_threshold_quantile=0.99,
                minimum_bout_duration_ms=0.0,
            ),
        )
        self.assertEqual(set(result), {"0ms", "10ms"})
        self.assertIn("whole_tail_xy_rms_speed", result["10ms"])
        self.assertIn("positive_control", result["10ms"]["whole_tail_xy_rms_speed"])

    def test_sensitivity_rejects_unordered_timeline(self) -> None:
        rows = 200
        frame = __import__("pandas").DataFrame(
            {
                "FrameID": np.arange(rows),
                "ElapsedTime": np.concatenate(
                    [np.arange(rows - 1, dtype=float), [1.0]]
                ),
                "AbsoluteTime": np.arange(rows, dtype=np.int64),
                "FrameStep": np.concatenate([[0], np.ones(rows - 1)]),
                "DeltaTimeMs": np.ones(rows),
                "valid_derivative": np.ones(rows, dtype=bool),
                "xy_valid_tail_fraction": np.ones(rows),
                "angular_valid_tail_fraction": np.ones(rows),
                "curvature_valid_tail_fraction": np.ones(rows),
                **{
                    column: np.linspace(0.1, 1.0, rows)
                    for column in __import__(
                        "classical_conditioning.preprocessing.candidates_v1",
                        fromlist=["CANDIDATE_COLUMNS"],
                    ).CANDIDATE_COLUMNS
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "increasing elapsed time"):
            evaluate_smoothing_sensitivity(
                frame,
                __import__("pandas").DataFrame(
                    {"Type": ["Reinforcer"], "Beg": [100], "End": [110]}
                ),
            )


if __name__ == "__main__":
    unittest.main()
