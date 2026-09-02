from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from classical_conditioning.analysis.movement_state import METRIC_IDS
from classical_conditioning.analysis.trace_review import (
    _static_figure,
    extract_review_traces,
    select_review_windows,
)
from classical_conditioning.preprocessing.candidates_v1 import CANDIDATE_COLUMNS


class TraceReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = 5_000
        self.frames = pd.DataFrame(
            {
                "FrameID": np.arange(rows),
                "ElapsedTime": np.arange(rows, dtype=float),
                "AbsoluteTime": np.arange(rows, dtype=np.int64),
                "FrameStep": np.concatenate([[0], np.ones(rows - 1)]),
                **{
                    column: np.linspace(0.1, 1.0, rows)
                    for column in CANDIDATE_COLUMNS
                },
            }
        )
        self.frames.loc[1_000:1_999, CANDIDATE_COLUMNS[2]] = 0.0
        self.frames.loc[3_000, CANDIDATE_COLUMNS[2]] = 10.0
        movement_data = {"FrameID": np.arange(rows)}
        for index, metric_id in enumerate(METRIC_IDS.values()):
            moving = np.zeros(rows, dtype=bool)
            moving[2_000 + index : 2_100 + index] = True
            movement_data[f"{metric_id}__valid"] = np.ones(rows, dtype=bool)
            movement_data[f"{metric_id}__moving"] = moving
            movement_data[f"{metric_id}__bout_id"] = moving.astype(np.int32)
        self.movement = pd.DataFrame(movement_data)
        self.protocol = pd.DataFrame(
            {
                "Type": ["Reinforcer", "Reinforcer", "Reinforcer"],
                "Beg": [500, 2_500, 4_500],
                "End": [510, 2_510, 4_510],
            }
        )

    def test_selects_balanced_unique_windows(self) -> None:
        windows = select_review_windows(
            self.frames,
            self.movement,
            self.protocol,
            half_window_ms=200,
        )
        self.assertIn("quiet", set(windows["Window ID"]))
        self.assertIn("strong", set(windows["Window ID"]))
        self.assertIn("disagreement", set(windows["Window ID"]))
        self.assertEqual(
            windows["Center absolute time (ms)"].nunique(),
            len(windows),
        )

    def test_extracts_all_metrics_with_threshold_context(self) -> None:
        windows = select_review_windows(
            self.frames,
            self.movement,
            self.protocol,
            half_window_ms=200,
        ).iloc[:1]
        calibration = {
            metric_id: {"low_threshold": 0.5, "high_threshold": 1.0}
            for metric_id in METRIC_IDS.values()
        }
        traces = extract_review_traces(
            self.frames,
            self.movement,
            windows,
            calibration,
            half_window_ms=100,
        )
        self.assertEqual(set(traces["Metric ID"]), set(METRIC_IDS.values()))
        self.assertIn("Detector input / high threshold", traces)
        self.assertIn("Moving", traces)
        self.assertIn("Bout ID", traces)

    def test_static_figure_uses_theme_size_and_spines(self) -> None:
        windows = select_review_windows(
            self.frames,
            self.movement,
            self.protocol,
            half_window_ms=200,
        )
        calibration = {
            metric_id: {"low_threshold": 0.5, "high_threshold": 1.0}
            for metric_id in METRIC_IDS.values()
        }
        traces = extract_review_traces(
            self.frames,
            self.movement,
            windows,
            calibration,
            half_window_ms=100,
        )
        figure = _static_figure(traces, windows)
        try:
            self.assertLess(figure.get_size_inches()[0], 8.0)
            for axis in figure.axes:
                self.assertFalse(axis.spines["top"].get_visible())
                self.assertFalse(axis.spines["right"].get_visible())
        finally:
            __import__("matplotlib.pyplot").pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
