from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import analysis_utils
from classical_conditioning.preprocessing.legacy_v1 import (
    BOUT_METRIC_COLUMN,
    LegacyPreprocessingConfig,
    _legacy_protocol,
    annotate_stimuli_legacy,
    calculate_bout_metric_legacy,
    calculate_vigor_legacy,
    cumulative_angles_legacy,
    detect_bouts_legacy,
    filter_angles_legacy,
    finalize_legacy_samples,
    prepare_legacy_tracking,
    synchronize_legacy,
)


class LegacyPreprocessingEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LegacyPreprocessingConfig(
            temporal_filter_frames=3,
            bout_max_window_frames=3,
            bout_min_window_frames=3,
            minimum_bout_duration_frames=0,
            minimum_interbout_frames=0,
        )

    def test_tracking_preparation_matches_current_reader_semantics(self) -> None:
        columns = ["FrameID"] + [
            name
            for index in range(16)
            for name in (f"x{index}", f"y{index}", f"angle{index}")
        ]
        rows = []
        for frame_id in (10, 11, 12):
            values = []
            for index in range(16):
                values.extend([index, index + 1, index / 10])
            rows.append([frame_id, *values])
        result = prepare_legacy_tracking(pd.DataFrame(rows, columns=columns), self.config)
        self.assertEqual(result["FrameID"].tolist(), [10, 11])
        self.assertNotIn("x0", result)
        self.assertAlmostEqual(
            result.loc[0, "Angle of point 1 (deg)"],
            0.1 * 180 / np.pi,
            places=5,
        )

    def test_synchronization_matches_current_function(self) -> None:
        tracking = pd.DataFrame({"FrameID": [1, 2, 4], "angle": [1.0, 2.0, 4.0]})
        camera = pd.DataFrame(
            {
                "FrameID": [2, 3, 4],
                "ElapsedTime": [2.0, 3.0, 4.0],
                "AbsoluteTime": [102, 103, 104],
            }
        )
        expected = analysis_utils.merge_camera_with_data(tracking, camera)
        actual = synchronize_legacy(tracking, camera)
        pd.testing.assert_frame_equal(actual, expected)

    def test_cumulative_filter_vigor_and_bout_metric_match_current_functions(self) -> None:
        rows = 9
        frame = pd.DataFrame(
            {
                "Trial time (frame) [700 FPS]": np.arange(rows),
                "AbsoluteTime": np.arange(rows, dtype=float),
                **{
                    f"Angle of point {index} (deg)": np.arange(rows, dtype=float)
                    + index
                    for index in range(16)
                },
            }
        )
        cumulative = cumulative_angles_legacy(frame)
        filtered = filter_angles_legacy(cumulative, self.config)

        expected_filtered = analysis_utils.filter_data(
            cumulative.copy(),
            space_window=3,
            time_window=3,
        )
        pd.testing.assert_frame_equal(filtered, expected_filtered)

        actual_vigor = calculate_vigor_legacy(filtered, self.config)
        expected_vigor = analysis_utils.calculate_vigor_fast_pure_numpy(
            filtered["Angle of point 15 (deg)"].to_numpy(),
            700,
        )
        np.testing.assert_array_equal(actual_vigor["Vigor (deg/ms)"], expected_vigor)

        actual_metric = calculate_bout_metric_legacy(actual_vigor, self.config)
        expected_metric = (
            actual_vigor["Vigor (deg/ms)"].rolling(3, center=True).max()
            - actual_vigor["Vigor (deg/ms)"].rolling(3, center=True).min()
        )
        np.testing.assert_allclose(
            actual_metric[BOUT_METRIC_COLUMN],
            expected_metric.dropna(),
        )

    def test_bout_detection_matches_current_function(self) -> None:
        source = pd.DataFrame(
            {BOUT_METRIC_COLUMN: [0.0, 5.0, 5.0, 0.0, 0.0]}
        )
        actual = detect_bouts_legacy(source, self.config)
        expected = analysis_utils.find_beg_and_end_of_bouts(
            source.copy(),
            thr1=self.config.bout_threshold_primary_deg_per_ms,
            min_dur=self.config.minimum_bout_duration_frames,
            min_gap=self.config.minimum_interbout_frames,
            thr2=self.config.bout_threshold_secondary_deg_per_ms,
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_stimulus_annotation_uses_strict_start_and_inclusive_end(self) -> None:
        data = pd.DataFrame(
            {
                "AbsoluteTime": [100.0, 101.0, 102.0, 103.0],
                "value": [0, 1, 2, 3],
            }
        )
        protocol = pd.DataFrame(
            {"beg (ms)": [101.0], "end (ms)": [102.0]},
            index=pd.Index(["Cycle"], name="Experiment type"),
        )
        result = annotate_stimuli_legacy(data, protocol)
        self.assertEqual(result["CS beg"].tolist(), [0, 0, 0, 0])
        self.assertEqual(result["CS end"].tolist(), [0, 0, 0, 0])

    def test_multiple_stimuli_match_legacy_start_and_end_semantics(self) -> None:
        data = pd.DataFrame(
            {
                "AbsoluteTime": [100.0, 101.0, 102.0, 103.0, 104.0],
                "value": [0, 1, 2, 3, 4],
            }
        )
        protocol = pd.DataFrame(
            {
                "beg (ms)": [100.0, 102.0],
                "end (ms)": [101.0, 103.0],
            },
            index=pd.Index(["Cycle", "Cycle"], name="Experiment type"),
        )
        actual = annotate_stimuli_legacy(data, protocol)
        expected = analysis_utils.stim_in_data(data, protocol)
        for column in ("CS beg", "CS end", "US beg", "US end"):
            np.testing.assert_array_equal(
                pd.to_numeric(actual[column]),
                pd.to_numeric(expected[column]),
            )

    def test_final_output_preserves_legacy_categorical_fields(self) -> None:
        rows = 2
        data = pd.DataFrame(
            {
                "Original frame number": [1.0, 2.0],
                "Trial time (frame) [700 FPS]": [-1, 0],
                "CS beg": [0, 1],
                "CS end": [0, 0],
                "US beg": [0, 0],
                "US end": [0, 0],
                "Trial type": pd.Categorical(["CS", "CS"]),
                "Trial number": [1, 1],
                "Block name": pd.Categorical(["Pre-train", "Pre-train"]),
                "Vigor (deg/ms)": [1.0, 2.0],
                "Scaled vigor (AU)": [0.0, 1.0],
                "Bout beg": [False, True],
                "Bout end": [False, False],
                "Bout": [False, True],
                BOUT_METRIC_COLUMN: [0.0, 5.0],
                **{
                    f"Angle of point {index} (deg)": np.zeros(rows)
                    for index in range(16)
                },
            }
        )
        result = finalize_legacy_samples(
            data,
            {
                "Day": "20260101",
                "Fish no.": "01",
                "Exp.": "delay",
                "ProtocolRig": "black-1",
                "Strain": "test",
                "Age (dpf)": "6",
            },
            self.config,
        )
        for column in (
            "Day",
            "Fish no.",
            "Exp.",
            "ProtocolRig",
            "Strain",
            "Age (dpf)",
            "CS beg",
            "CS end",
            "US beg",
            "US end",
            "Trial number",
        ):
            self.assertEqual(result[column].dtype.name, "category")

    def test_zero_start_protocol_is_rejected_like_current_reader(self) -> None:
        protocol = pd.DataFrame(
            {
                "Type": ["Cycle", "Cycle"],
                "Beg": [0, 100],
                "End": [10, 110],
            }
        )
        with self.assertRaisesRegex(ValueError, "zero-start"):
            _legacy_protocol(protocol)


if __name__ == "__main__":
    unittest.main()
