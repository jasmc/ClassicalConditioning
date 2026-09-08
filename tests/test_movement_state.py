from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.analysis.movement_state import (
    MovementCalibrationConfig,
    _positive_control,
    detect_legacy_envelope_bouts,
    evaluate_smoothing_sensitivity,
    resolve_candidate_metric_source,
    rolling_extreme_envelope,
    smooth_contiguous_median,
)
from classical_conditioning.preprocessing.candidates_v1 import CANDIDATE_COLUMNS


def _detector_frame(rows: int, elapsed: np.ndarray | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FrameID": np.arange(rows),
            "ElapsedTime": (
                np.arange(rows, dtype=float) if elapsed is None else elapsed
            ),
            "AbsoluteTime": np.arange(rows, dtype=np.int64),
            "FrameStep": np.concatenate([[0], np.ones(rows - 1)]),
            "DeltaTimeMs": np.ones(rows),
            "valid_derivative": np.concatenate(
                [[False], np.ones(rows - 1, dtype=bool)]
            ),
            "xy_valid_tail_fraction": np.ones(rows),
            "angular_valid_tail_fraction": np.ones(rows),
            "curvature_valid_tail_fraction": np.ones(rows),
            **{
                column: np.linspace(0.1, 1.0, rows)
                + 0.01 * np.sin(np.arange(rows))
                for column in CANDIDATE_COLUMNS
            },
        }
    )


class MovementStateTests(unittest.TestCase):
    def test_metric_movement_recipe_pairing_is_frozen(self) -> None:
        development = resolve_candidate_metric_source(
            metric_recipe="tail-candidate-development-v1"
        )
        self.assertEqual(development.movement_recipe, "movement-candidate-v2")
        corrected = resolve_candidate_metric_source(
            movement_recipe="movement-candidate-corrected-v2"
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
                movement_recipe="movement-candidate-corrected-v2",
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

    def test_historical_thresholds_are_converted_from_degrees(self) -> None:
        config = MovementCalibrationConfig()
        self.assertAlmostEqual(
            config.envelope_threshold_rad_per_ms,
            4.0 * np.pi / 180.0,
        )
        self.assertAlmostEqual(
            config.bout_amplitude_threshold_rad_per_ms,
            1.0 * np.pi / 180.0,
        )
        # Historical frame counts at the interpolated 700 FPS rate.
        self.assertAlmostEqual(config.envelope_max_window_ms, 20 / 700 * 1_000)
        self.assertAlmostEqual(config.envelope_min_window_ms, 400 / 700 * 1_000)

    def test_envelope_is_rolling_max_minus_rolling_min(self) -> None:
        values = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        envelope = rolling_extreme_envelope(
            values,
            np.array([0, 1, 1, 1, 1]),
            max_window_samples=3,
            min_window_samples=3,
        )
        self.assertTrue(np.isnan(envelope[0]))
        self.assertTrue(np.isnan(envelope[4]))
        self.assertEqual(envelope[1:4].tolist(), [10.0, 10.0, 10.0])

    def test_envelope_never_spans_a_frame_discontinuity(self) -> None:
        values = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
        envelope = rolling_extreme_envelope(
            values,
            np.array([0, 1, 2, 1, 1]),
            max_window_samples=3,
            min_window_samples=3,
        )
        # The gap splits the trace into a 2-sample and a 3-sample segment, so
        # only the center of the second segment has a full window.
        self.assertTrue(np.all(np.isnan(envelope[[0, 1, 2, 4]])))
        self.assertEqual(envelope[3], 10.0)

    def test_envelope_above_threshold_becomes_a_bout(self) -> None:
        moving, bout_ids = detect_legacy_envelope_bouts(
            np.array([0.0, 5.0, 5.0, 5.0, 0.0]),
            np.ones(5),
            np.full(5, 10.0),
            np.array([0, 1, 1, 1, 1]),
            np.ones(5, dtype=bool),
            envelope_threshold=4.0,
            amplitude_threshold=0.5,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertEqual(moving.tolist(), [False, True, True, True, False])
        self.assertEqual(int(bout_ids.max()), 1)

    def test_short_bout_is_removed(self) -> None:
        moving, _ = detect_legacy_envelope_bouts(
            np.array([0.0, 5.0, 0.0]),
            np.ones(3),
            np.full(3, 10.0),
            np.array([0, 1, 1]),
            np.ones(3, dtype=bool),
            envelope_threshold=4.0,
            amplitude_threshold=0.5,
            minimum_bout_duration_ms=20.0,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertFalse(moving.any())

    def test_short_gap_is_merged_but_frame_gap_is_not(self) -> None:
        envelope = np.array([5.0, 5.0, 0.0, 5.0, 5.0])
        merged, _ = detect_legacy_envelope_bouts(
            envelope,
            np.ones(5),
            np.full(5, 5.0),
            np.array([0, 1, 1, 1, 1]),
            np.ones(5, dtype=bool),
            envelope_threshold=4.0,
            amplitude_threshold=0.5,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=15.0,
        )
        self.assertTrue(merged.all())

        not_merged, _ = detect_legacy_envelope_bouts(
            envelope,
            np.ones(5),
            np.full(5, 5.0),
            np.array([0, 1, 2, 1, 1]),
            np.ones(5, dtype=bool),
            envelope_threshold=4.0,
            amplitude_threshold=0.5,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=15.0,
        )
        self.assertFalse(not_merged[2])

    def test_weak_bout_below_amplitude_threshold_is_removed(self) -> None:
        envelope = np.array([0.0, 5.0, 5.0, 5.0, 0.0])
        weak, _ = detect_legacy_envelope_bouts(
            envelope,
            np.full(5, 0.1),
            np.full(5, 10.0),
            np.array([0, 1, 1, 1, 1]),
            np.ones(5, dtype=bool),
            envelope_threshold=4.0,
            amplitude_threshold=1.0,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertFalse(weak.any())

        # A single sample reaching the amplitude threshold keeps the bout.
        strong, _ = detect_legacy_envelope_bouts(
            envelope,
            np.array([0.1, 0.1, 1.5, 0.1, 0.1]),
            np.full(5, 10.0),
            np.array([0, 1, 1, 1, 1]),
            np.ones(5, dtype=bool),
            envelope_threshold=4.0,
            amplitude_threshold=1.0,
            minimum_bout_duration_ms=0.0,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertEqual(strong.tolist(), [False, True, True, True, False])

    def test_exact_legacy_duration_boundaries_are_retained_and_merged(self) -> None:
        interval = 1_000 / 700
        envelope = np.concatenate([np.zeros(2), np.full(40, 5.0), np.zeros(2)])
        moving, _ = detect_legacy_envelope_bouts(
            envelope,
            np.ones(len(envelope)),
            np.full(len(envelope), interval),
            np.concatenate([[0], np.ones(len(envelope) - 1)]),
            np.ones(len(envelope), dtype=bool),
            envelope_threshold=4.0,
            amplitude_threshold=0.5,
            minimum_bout_duration_ms=40 / 700 * 1_000,
            maximum_interbout_gap_ms=0.0,
        )
        self.assertEqual(int(moving.sum()), 40)

        gap_envelope = np.concatenate(
            [np.full(20, 5.0), np.zeros(10), np.full(20, 5.0)]
        )
        merged, _ = detect_legacy_envelope_bouts(
            gap_envelope,
            np.ones(len(gap_envelope)),
            np.full(len(gap_envelope), interval),
            np.concatenate([[0], np.ones(len(gap_envelope) - 1)]),
            np.ones(len(gap_envelope), dtype=bool),
            envelope_threshold=4.0,
            amplitude_threshold=0.5,
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
            pd.DataFrame({"Type": ["Reinforcer"], "Beg": [0], "End": [1]}),
            MovementCalibrationConfig(positive_control_window_ms=4.0),
        )
        self.assertEqual(result["post_us_evaluated_event_count"], 1)
        self.assertEqual(result["post_us_valid_fraction"], 0.75)

    def test_smoothing_sensitivity_reports_one_shared_detector(self) -> None:
        result = evaluate_smoothing_sensitivity(
            _detector_frame(1_000),
            pd.DataFrame({"Type": ["Reinforcer"], "Beg": [500], "End": [510]}),
            smoothing_windows_ms=(0.0, 10.0),
            base_config=MovementCalibrationConfig(
                minimum_bout_duration_ms=0.0,
            ),
        )
        self.assertEqual(set(result), {"0ms", "10ms"})
        # Sensitivity is reported per smoothing window, not per metric, because
        # one detector serves every metric.
        self.assertIn("positive_control", result["10ms"])
        self.assertIn("bout_count", result["10ms"])
        self.assertEqual(
            result["10ms"]["detector_source_column"],
            CANDIDATE_COLUMNS[5],
        )
        for metric_id in ("whole_tail_xy_rms_speed", "curvature_change_rms"):
            self.assertNotIn(metric_id, result["10ms"])

    def test_sensitivity_rejects_unordered_timeline(self) -> None:
        rows = 200
        elapsed = np.concatenate([np.arange(rows - 1, dtype=float), [1.0]])
        with self.assertRaisesRegex(ValueError, "increasing elapsed time"):
            evaluate_smoothing_sensitivity(
                _detector_frame(rows, elapsed=elapsed),
                pd.DataFrame(
                    {"Type": ["Reinforcer"], "Beg": [100], "End": [110]}
                ),
            )


if __name__ == "__main__":
    unittest.main()
