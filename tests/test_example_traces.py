from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cli import build_parser
from classical_conditioning.figures.example_traces import (
    build_example_trace_figure,
    prepare_example_trace_data,
    render_example_trace_figure,
)
from classical_conditioning.figures.export import FigureMode

METRIC = "legacy_distal_angular_speed"
METRIC_COLUMN = "legacy_distal_angular_speed_rad_per_ms"


def example_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = np.array([-2, -1, 0, 1, 10, 11], dtype=np.int64) * 1000
    absolute = np.concatenate((100_000 + times, 200_000 + times))
    frame_id = np.arange(len(absolute), dtype=np.int64)
    corrected = pd.DataFrame({
        "FrameID": frame_id,
        "AbsoluteTime": absolute,
        "angle0": [0.1, 0.1, 0.3, 0.5, 0.2, 0.1] * 2,
        "angle1": [0.1, 0.1, 0.2, 0.3, 0.1, 0.1] * 2,
    })
    metrics = pd.DataFrame({
        "FrameID": frame_id,
        "AbsoluteTime": absolute,
        METRIC_COLUMN: np.arange(1, len(absolute) + 1, dtype=float),
    })
    protocol = pd.DataFrame({
        "Type": ["Cycle", "Reinforcer", "Cycle", "Reinforcer"],
        "Beg": [100_000, 109_000, 200_000, 209_000],
        "End": [110_000, 109_100, 210_000, 209_100],
    })
    return corrected, metrics, protocol


class ExampleTraceTests(unittest.TestCase):
    def test_selected_trials_pair_angle_and_vigor_on_the_same_frames(self) -> None:
        corrected, metrics, protocol = example_inputs()
        frames, events = prepare_example_trace_data(
            corrected, metrics, protocol,
            trial_numbers=(2, 1), metric_id=METRIC, tail_point=1,
            window_s=(-3, 12),
        )
        self.assertEqual(frames["Trial number"].drop_duplicates().tolist(), [2, 1])
        second = frames.loc[frames["Trial number"].eq(2)]
        self.assertEqual(second["FrameID"].tolist(), list(range(6, 12)))
        self.assertEqual(second["Time relative to CS onset (s)"].tolist(),
                         [-2, -1, 0, 1, 10, 11])
        self.assertAlmostEqual(float(second["Tail angle (rad)"].iloc[0]), 0)
        self.assertAlmostEqual(float(second["Tail angle (rad)"].iloc[3]), 0.6)
        self.assertEqual(second["Vigor"].tolist(), [7, 8, 9, 10, 11, 12])
        actual_us = events.loc[events["Event"].eq("actual US onset")]
        self.assertEqual(actual_us["Time (s)"].tolist(), [9, 9])

    def test_rejects_mismatched_frames_and_unavailable_trials(self) -> None:
        corrected, metrics, protocol = example_inputs()
        metrics.loc[0, "FrameID"] = 99
        with self.assertRaisesRegex(Exception, "do not align"):
            prepare_example_trace_data(
                corrected, metrics, protocol,
                trial_numbers=(1,), metric_id=METRIC, tail_point=1,
            )
        metrics.loc[0, "FrameID"] = 0
        with self.assertRaisesRegex(Exception, "outside the recording"):
            prepare_example_trace_data(
                corrected, metrics, protocol,
                trial_numbers=(3,), metric_id=METRIC, tail_point=1,
            )

    def test_adjacent_frames_may_share_a_millisecond_timestamp(self) -> None:
        corrected, metrics, protocol = example_inputs()
        corrected.loc[1, "AbsoluteTime"] = corrected.loc[0, "AbsoluteTime"]
        metrics.loc[1, "AbsoluteTime"] = corrected.loc[1, "AbsoluteTime"]
        frames, _ = prepare_example_trace_data(
            corrected, metrics, protocol,
            trial_numbers=(1,), metric_id=METRIC, tail_point=1,
        )
        self.assertEqual(frames["FrameID"].iloc[:2].tolist(), [0, 1])

    def test_figure_and_cli_keep_user_selected_trial_order(self) -> None:
        corrected, metrics, protocol = example_inputs()
        frames, events = prepare_example_trace_data(
            corrected, metrics, protocol,
            trial_numbers=(2, 1), metric_id=METRIC, tail_point=1,
        )
        figure, panel_ids, mappings = render_example_trace_figure(
            frames, events, trial_numbers=(2, 1),
            metric_id=METRIC, window_s=(-20, 20),
        )
        try:
            self.assertEqual(panel_ids, ["C_trial_2", "D_trial_2", "C_trial_1", "D_trial_1"])
            self.assertEqual(len(figure.axes), 4)
            self.assertEqual(
                figure.axes[0].lines[0].get_xdata().tolist(),
                figure.axes[1].lines[0].get_xdata().tolist(),
            )
            self.assertEqual(mappings["trace__d_trial_2"]["metric_id"], METRIC)
        finally:
            plt.close(figure)
        args = build_parser().parse_args([
            "figure-example-traces", "--project-dir", "/tmp/project",
            "--recording-id", "fish-1", "--experiment", "allDelay",
            "--trial", "2", "--trial", "1", "--metric", METRIC,
        ])
        self.assertEqual(args.trial, [2, 1])

    def test_static_builder_exports_both_columns_with_provenance(self) -> None:
        corrected, metrics, protocol = example_inputs()
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = project / "Processed data" / "fish-1"
            root.mkdir(parents=True)
            corrected_path = root / "frame_preprocessed_corrected.parquet"
            metrics_path = root / "frame_activity_candidates-corrected.parquet"
            protocol_path = root / "stimulus_events.parquet"
            for path, frame in (
                (corrected_path, corrected), (metrics_path, metrics),
                (protocol_path, protocol),
            ):
                pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
            with (
                patch("classical_conditioning.figures.example_traces._verify_corrected_preprocess"),
                patch("classical_conditioning.figures.example_traces._verify_metrics"),
                patch("classical_conditioning.figures.example_traces.load_and_verify_source_manifest",
                      return_value=(None, {"protocol": {"sha256": sha256_file(protocol_path)}}, None)),
            ):
                result = build_example_trace_figure(
                    project, "fish-1", trial_numbers=(2, 1),
                    metric_id=METRIC, experiment="allDelay",
                    tail_point=1, mode=FigureMode.STATIC,
                )
            self.assertTrue(result.outputs[0].is_file())
            sidecar = json.loads(result.sidecar.read_text(encoding="utf-8"))
            self.assertIn("trace__c_trial_2", sidecar["artist_registry"])
            self.assertIn("trace__d_trial_1", sidecar["artist_registry"])
            self.assertEqual(len(sidecar["input_artifacts"]), 3)


if __name__ == "__main__":
    unittest.main()
