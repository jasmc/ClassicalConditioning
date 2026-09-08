from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.cli import build_parser
from classical_conditioning.figures.metric_comparison import (
    _cohort_standardized_figure,
    build_metric_comparison_figure,
)
from classical_conditioning.figures.export import FigureMode
from classical_conditioning.figures.temporal_profiles import METRIC_LABELS


def recording_summary_fixture() -> pd.DataFrame:
    rows = []
    for recording_id, condition, offset in (
        ("20230315_05", "control", 0.1),
        ("20230316_11", "control", 0.2),
        ("20230315_06", "fixedtrace", 0.8),
        ("20230316_03", "fixedtrace", 1.1),
    ):
        for metric in METRIC_LABELS:
            rows.append(
                {
                    "Recording ID": recording_id,
                    "Condition ID": condition,
                    "Trial type": "CS",
                    "Metric ID": metric,
                    "Outcome ID": "movement-probability",
                    "Standardized difference": offset,
                }
            )
    return pd.DataFrame(rows)


class MetricComparisonFigureTests(unittest.TestCase):
    def test_plots_five_metrics_and_two_conditions(self) -> None:
        figure, panel_ids, mappings = _cohort_standardized_figure(
            recording_summary_fixture(),
            trial_type="CS",
            outcome_id="movement-probability",
            experiment_name="fixedVsIncreasingTrace",
        )
        try:
            self.assertEqual(panel_ids, ["A"])
            self.assertIn("bars__control", mappings)
            self.assertIn("bars__fixedtrace", mappings)
            self.assertEqual(len(figure.axes[0].collections), 2 * len(METRIC_LABELS))
        finally:
            __import__("matplotlib.pyplot").pyplot.close(figure)

    def test_delay_only_cohort_uses_delay_condition(self) -> None:
        rows = []
        for recording_id, offset in (("20221115_04", 0.4), ("20221116_12", 0.6)):
            for metric in METRIC_LABELS:
                rows.append(
                    {
                        "Recording ID": recording_id,
                        "Condition ID": "delay",
                        "Trial type": "CS",
                        "Metric ID": metric,
                        "Outcome ID": "movement-probability",
                        "Standardized difference": offset,
                    }
                )
        figure, panel_ids, mappings = _cohort_standardized_figure(
            pd.DataFrame(rows),
            trial_type="CS",
            outcome_id="movement-probability",
            experiment_name="allDelay",
        )
        try:
            self.assertEqual(panel_ids, ["A"])
            self.assertIn("bars__delay", mappings)
            self.assertNotIn("bars__control", mappings)
            self.assertEqual(len(figure.axes[0].collections), len(METRIC_LABELS))
        finally:
            __import__("matplotlib.pyplot").pyplot.close(figure)

    def test_builder_writes_static_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            analysis_id = "c-copy-4fish-cohort-v1"
            recipe = "candidate-metric-comparison-corrected-v1"
            output_dir = project / "Processed data" / "Analyses" / analysis_id
            output_dir.mkdir(parents=True)
            path = output_dir / f"{recipe}_recording_summary.parquet"
            pq.write_table(
                pa.Table.from_pandas(
                    recording_summary_fixture(),
                    preserve_index=False,
                ),
                path,
            )
            result = build_metric_comparison_figure(
                project,
                analysis_id,
                mode=FigureMode.STATIC,
                comparison_recipe="candidate-metric-comparison-corrected-v1",
                overwrite=True,
            )
            pngs = [path for path in result.outputs if path.suffix == ".png"]
            self.assertEqual(len(pngs), 1)
            self.assertTrue(pngs[0].is_file())
            self.assertIn("Analyses", str(pngs[0]))

    def test_cli_exposes_corrected_comparison_recipe(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "figure-metric-comparison",
                "--project-dir",
                "Paper data",
                "--analysis-id",
                "c-copy-4fish-cohort-v1",
                "--mode",
                "static",
            ]
        )
        self.assertEqual(
            args.recipe,
            "candidate-metric-comparison-corrected-v1",
        )
        profile = parser.parse_args(
            [
                "figure-candidate-profiles",
                "--project-dir",
                "Paper data",
                "--recording-id",
                "20230315_05",
                "--mode",
                "static",
                "--recipe",
                "candidate-temporal-outcomes-corrected-v3",
            ]
        )
        self.assertEqual(
            profile.recipe,
            "candidate-temporal-outcomes-corrected-v3",
        )


if __name__ == "__main__":
    unittest.main()
