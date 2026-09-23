from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.analysis.temporal_profiles import (
    CANDIDATE_COLUMNS,
    METRIC_IDS,
    TemporalProfileConfig,
    _signed_bout_log_vigor,
    aggregate_event_profiles,
)


class TemporalProfileTests(unittest.TestCase):
    def test_signed_log_vigor_centres_bout_medians_on_pre_cs_baseline(self) -> None:
        values = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        result = _signed_bout_log_vigor(
            values,
            np.array([-20.0, -19.0, -18.0, 1.0, 2.0, 3.0]),
            np.array([0, 0, 0, 1, 1, 1]),
            np.ones(6, dtype=bool),
            np.ones(6, dtype=bool),
            np.array([1, 1, 1, 2, 2, 2]),
            bin_count=2,
            baseline_end_s=-15.0,
        )
        np.testing.assert_allclose(result, [0.0, np.log(2.0)])
        missing_baseline = _signed_bout_log_vigor(
            values,
            np.array([-10.0, -9.0, -8.0, 1.0, 2.0, 3.0]),
            np.array([0, 0, 0, 1, 1, 1]),
            np.ones(6, dtype=bool),
            np.ones(6, dtype=bool),
            np.array([1, 1, 1, 2, 2, 2]),
            bin_count=2,
            baseline_end_s=-15.0,
        )
        self.assertTrue(np.isnan(missing_baseline).all())

    def setUp(self) -> None:
        self.config = TemporalProfileConfig(
            window_start_s=-1.0,
            window_end_s=1.0,
            bin_width_s=0.5,
        )
        time_ms = np.arange(0, 2_001, 100)
        self.frames = pd.DataFrame(
            {
                "AbsoluteTime": time_ms,
                "FrameStep": np.concatenate([[0], np.ones(len(time_ms) - 1)]),
                "DeltaTimeMs": np.full(len(time_ms), 100.0),
                **{
                    column: np.full(len(time_ms), index + 1.0)
                    for index, column in enumerate(CANDIDATE_COLUMNS)
                },
            }
        )
        self.protocol = pd.DataFrame(
            {
                "Type": ["Cycle", "Reinforcer"],
                "Beg": [1_000, 1_000],
                "End": [1_100, 1_050],
            }
        )
        # One shared segmentation, not one per metric.
        self.movement = pd.DataFrame(
            {
                "AbsoluteTime": time_ms,
                "valid": np.ones(len(time_ms), dtype=bool),
                "moving": np.array([False] * 10 + [True] * 5 + [False] * 6),
                "bout_id": np.array(
                    [0] * 10 + [1] * 5 + [0] * 6,
                    dtype=np.int32,
                ),
            }
        )

    def test_builds_left_closed_time_bins_for_each_metric_and_alignment(self) -> None:
        result = aggregate_event_profiles(
            self.frames,
            self.protocol,
            config=self.config,
            block_lookup={("CS", 1): "Pre-train", ("US", 1): "Train 1"},
        )
        self.assertEqual(len(result), 2 * 4 * len(CANDIDATE_COLUMNS))
        self.assertEqual(set(result["Trial type"]), {"CS", "US"})
        self.assertEqual(
            sorted(result["Time bin start (s)"].unique()),
            [-1.0, -0.5, 0.0, 0.5],
        )
        self.assertEqual(set(result["Block name"]), {"Pre-train", "Train 1"})

    def test_total_activity_includes_zero_values(self) -> None:
        self.frames[CANDIDATE_COLUMNS[0]] = 0.0
        result = aggregate_event_profiles(
            self.frames,
            self.protocol.iloc[:1],
            config=self.config,
        )
        metric = result[result["Metric ID"] == "tail_length_weighted_angular_l1"]
        np.testing.assert_allclose(metric["Total activity mean"], 0.0)
        np.testing.assert_allclose(metric["Valid fraction"], 1.0)

    def test_invalid_values_reduce_valid_fraction_without_becoming_zero(self) -> None:
        self.frames.loc[
            self.frames["AbsoluteTime"].between(0, 400),
            CANDIDATE_COLUMNS[0],
        ] = np.nan
        result = aggregate_event_profiles(
            self.frames,
            self.protocol.iloc[:1],
            config=self.config,
        )
        row = result[
            (result["Metric ID"] == "tail_length_weighted_angular_l1")
            & (result["Time bin start (s)"] == -1.0)
        ].iloc[0]
        self.assertEqual(row["Valid fraction"], 0.0)
        self.assertTrue(np.isnan(row["Total activity mean"]))

    def test_window_end_is_excluded(self) -> None:
        result = aggregate_event_profiles(
            self.frames,
            self.protocol.iloc[:1],
            config=self.config,
        )
        sample_count = result[
            result["Metric ID"] == "tail_length_weighted_angular_l1"
        ]["Sample count"].sum()
        self.assertEqual(sample_count, 20)

    def test_trial_numbering_is_chronological_not_file_order(self) -> None:
        protocol = pd.DataFrame(
            {
                "Type": ["Cycle", "Cycle"],
                "Beg": [1_500, 1_000],
                "End": [1_600, 1_100],
            }
        )
        result = aggregate_event_profiles(
            self.frames,
            protocol,
            config=self.config,
        )
        trial_starts = (
            result[["Trial number", "Event start absolute time (ms)"]]
            .drop_duplicates()
            .sort_values("Trial number")
        )
        self.assertEqual(
            trial_starts["Event start absolute time (ms)"].tolist(),
            [1_000, 1_500],
        )

    def test_event_without_frames_retains_zero_coverage_bins(self) -> None:
        protocol = pd.DataFrame(
            {"Type": ["Cycle"], "Beg": [10_000], "End": [10_100]}
        )
        result = aggregate_event_profiles(
            self.frames,
            protocol,
            config=self.config,
        )
        self.assertEqual(len(result), 4 * len(CANDIDATE_COLUMNS))
        self.assertTrue(result["Sample count"].eq(0).all())
        self.assertTrue(result["Valid fraction"].eq(0).all())
        self.assertTrue(result["Valid expected fraction"].eq(0).all())
        self.assertTrue(result["Total activity mean"].isna().all())

    def test_movement_outcomes_use_detector_valid_samples(self) -> None:
        result = aggregate_event_profiles(
            self.frames,
            self.protocol.iloc[:1],
            config=self.config,
            movement_state=self.movement,
        )
        metric = result[
            result["Metric ID"] == "tail_length_weighted_angular_l1"
        ]
        self.assertEqual(
            metric["Movement probability"].tolist(),
            [0.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(metric["Bout count"].sum(), 1)
        self.assertAlmostEqual(
            metric["Mean bout duration (ms)"].dropna().iloc[0],
            500.0,
        )
        self.assertTrue(
            metric.loc[
                metric["Movement probability"] == 1.0,
                "Conditional intensity mean",
            ].notna().all()
        )

    def test_profile_contains_signed_log_vigor_from_distinct_bouts(self) -> None:
        time_ms = np.arange(0, 90_000, 100)
        relative_s = (time_ms - 45_000) / 1_000
        before = (relative_s >= -25) & (relative_s < -16)
        after = (relative_s >= 1) & (relative_s < 5)
        moving = before | after
        values = np.where(after, 2.0, 1.0)
        frames = pd.DataFrame(
            {
                "AbsoluteTime": time_ms,
                "FrameStep": np.ones(len(time_ms), dtype=int),
                "DeltaTimeMs": np.full(len(time_ms), 100.0),
                **{column: values for column in CANDIDATE_COLUMNS},
            }
        )
        movement = pd.DataFrame(
            {
                "AbsoluteTime": time_ms,
                "valid": np.ones(len(time_ms), dtype=bool),
                "moving": moving,
                "bout_id": np.where(before, 1, np.where(after, 2, 0)),
            }
        )
        protocol = pd.DataFrame(
            {"Type": ["Cycle"], "Beg": [45_000], "End": [45_100]}
        )
        result = aggregate_event_profiles(
            frames, protocol, config=TemporalProfileConfig(),
            movement_state=movement,
        )
        metric = result[result["Metric ID"] == "tail_length_weighted_angular_l1"]
        self.assertAlmostEqual(
            metric.loc[metric["Time bin center (s)"] == -20.25,
                       "Signed log vigor"].iloc[0],
            0.0,
        )
        self.assertAlmostEqual(
            metric.loc[metric["Time bin center (s)"] == 2.25,
                       "Signed log vigor"].iloc[0],
            np.log(2.0),
        )
        self.assertTrue(
            metric.loc[metric["Time bin center (s)"] == 10.25,
                       "Signed log vigor"].isna().all()
        )

    def test_invalid_detector_samples_are_not_counted_as_rest(self) -> None:
        movement = self.movement.copy()
        movement.loc[:4, "valid"] = False
        result = aggregate_event_profiles(
            self.frames,
            self.protocol.iloc[:1],
            config=self.config,
            movement_state=movement,
        )
        row = result[
            (result["Metric ID"] == "tail_length_weighted_angular_l1")
            & (result["Time bin start (s)"] == -1.0)
        ].iloc[0]
        self.assertTrue(np.isnan(row["Movement probability"]))
        self.assertEqual(row["Detector valid fraction"], 0.0)
        self.assertTrue(np.isnan(row["Bout count"]))

    def test_two_layer_scaling_clips_to_unit_interval_after_onset(self) -> None:
        config = TemporalProfileConfig()
        time_ms = np.arange(0, 90_001, 100)
        trial_seconds = (time_ms - 45_000) / 1_000
        rng = np.random.default_rng(0)
        baseline = 0.5 + 0.1 * rng.standard_normal(len(time_ms))
        values = np.where(trial_seconds < 0.0, baseline, 10.0)
        frames = pd.DataFrame(
            {
                "AbsoluteTime": time_ms,
                "FrameStep": np.concatenate([[0], np.ones(len(time_ms) - 1)]),
                "DeltaTimeMs": np.full(len(time_ms), 100.0),
                **{column: values for column in CANDIDATE_COLUMNS},
            }
        )
        protocol = pd.DataFrame(
            {"Type": ["Cycle"], "Beg": [45_000], "End": [45_100]}
        )
        result = aggregate_event_profiles(frames, protocol, config=config)
        scaled = result["Scaled total activity"]
        finite = scaled[scaled.notna()]
        self.assertTrue(finite.between(0.0, 1.0).all())

        after_onset = result.loc[
            result["Time bin center (s)"] > 1.0,
            "Scaled total activity",
        ]
        self.assertTrue((after_onset == 1.0).all())

        # Raw values are untouched by scaling.
        self.assertAlmostEqual(
            float(
                result.loc[
                    result["Time bin center (s)"] > 1.0,
                    "Total activity mean",
                ].iloc[0]
            ),
            10.0,
        )

    def test_scaling_is_nan_without_a_usable_pre_baseline_window(self) -> None:
        # The default layer-1 window starts earlier than -15 s; this trial has
        # no such samples, so scaling must refuse rather than rescale on itself.
        result = aggregate_event_profiles(
            self.frames,
            self.protocol,
            config=self.config,
        )
        self.assertTrue(result["Scaled total activity"].isna().all())

    def test_bout_outcomes_are_identical_across_every_metric(self) -> None:
        result = aggregate_event_profiles(
            self.frames,
            self.protocol,
            config=self.config,
            movement_state=self.movement,
        )
        bout_columns = [
            "Movement probability",
            "Fraction time moving",
            "Bout rate per minute",
            "Bout count",
            "Detector valid fraction",
        ]
        keys = ["Trial type", "Trial number", "Time bin start (s)"]
        reference = None
        for metric_id in METRIC_IDS.values():
            metric = (
                result[result["Metric ID"] == metric_id]
                .sort_values(keys)
                .reset_index(drop=True)
            )
            if reference is None:
                reference = metric
                continue
            pd.testing.assert_frame_equal(
                metric[bout_columns],
                reference[bout_columns],
                check_dtype=False,
            )


if __name__ == "__main__":
    unittest.main()
