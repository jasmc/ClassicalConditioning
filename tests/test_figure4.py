from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from classical_conditioning.analysis.figure4 import (
    EXPERIMENTS, STRATA, assign_plot_strata, load_classification_manifest,
    analyze_figure4, load_figure4_analysis, profile_groups, summarize_trial_bins,
    verify_expected_us,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.figure4 import _draw_rows
from classical_conditioning.figures.figure4 import render_figure4
from classical_conditioning.figures.export import FigureMode


class Figure4AnalysisTests(unittest.TestCase):
    def test_all_metrics_require_matching_frozen_manifest(self) -> None:
        for metric in (
            "tail_length_weighted_angular_l1", "whole_tail_xy_mean_speed_normalized",
            "legacy_distal_angular_speed",
        ):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "labels.csv"
                pd.DataFrame({
                    "experiment_id": ["allDelay"], "condition_id": ["delay"],
                    "fish_id": ["one"], "classifier_label": ["Learner"],
                    "classification_eligible": [True], "ineligible_reason": [""],
                    "input_metric_id": [metric],
                }).to_csv(path, index=False)
                metadata = {"table_sha256": sha256_file(path), "input_metric_id": metric,
                            "classifier_execution_id": "classifier-1", "validation_mode": "descriptive",
                            "cohort_hashes": {experiment: "hash" for experiment in EXPERIMENTS}}
                path.with_suffix(".manifest.json").write_text(json.dumps(metadata))
                table, loaded = load_classification_manifest(path, metric)
                self.assertEqual(len(table), 1)
                self.assertEqual(loaded["input_metric_id"], metric)
                with self.assertRaisesRegex(Exception, "metric"):
                    load_classification_manifest(path, "different-metric")

    def test_control_flag_is_preserved_as_reference(self) -> None:
        fish = pd.DataFrame({"cohort_role": ["reference", "reference", "conditioned", "conditioned", "reference"],
                             "classifier_label": ["Learner", "Non-learner", "Learner", "Non-learner", "Unclassified"]})
        assigned = assign_plot_strata(fish)
        self.assertEqual(assigned["plot_stratum"].tolist(), [*STRATA[2:], *STRATA[:2], "Unclassified"])
        self.assertEqual(assigned.loc[0, "cohort_role"], "reference")

    def test_fish_first_aggregation_and_missing_bouts(self) -> None:
        records = []
        for fish, signed, trials, label in (
            ("one", -2., (25, 39, 53, 59, 65), STRATA[0]),
            ("two", 0., (25,), STRATA[0]),
            ("three", np.nan, (25,), STRATA[1]),
        ):
            for trial in trials:
                records.append({"experiment_id": "allDelay", "recording_id": fish,
                                "fish_id": fish, "condition_id": "delay", "cohort_role": "conditioned",
                                "classifier_label": "Learner" if label == STRATA[0] else "Non-learner",
                                "plot_stratum": label, "trial_number": trial, "time_s": .25,
                                "signed_log_vigor": signed,
                                "movement_probability": 0. if fish == "three" else .5})
        fish_bins, group_bins = summarize_trial_bins(pd.DataFrame(records), "allDelay")
        pooled = group_bins.loc[group_bins["group_name"].eq("All catch trials")
                                  & group_bins["plot_stratum"].eq(STRATA[0])].iloc[0]
        self.assertEqual(pooled["signed_median"], -1.)
        self.assertEqual(pooled["signed_fish"], 2)
        self.assertEqual(pooled["signed_trials"], 6)
        no_bout = fish_bins.loc[fish_bins["fish_id"].eq("three")
                                & fish_bins["group_name"].eq("All catch trials")].iloc[0]
        self.assertTrue(np.isnan(no_bout["signed"]))
        self.assertEqual(no_bout["movement"], 0.)
        test1 = next(group for group in profile_groups("allDelay") if group["group_name"] == "Test 1")
        self.assertIn(65, test1["trials"])
        catches = [group["trials"][0] for group in profile_groups("allDelay")
                   if group["group_type"] == "individual_catch"]
        self.assertEqual(catches, [25, 39, 53, 59, 65])

    def test_expected_us_mismatch_is_explicit(self) -> None:
        cycles = [100_000 * trial for trial in range(94)]
        events = [{"Type": "Cycle", "Beg": start, "End": start + 10_000}
                  for start in cycles]
        catches = {25, 39, 53, 59}
        for trial in range(15, 65):
            if trial not in catches:
                events.append({"Type": "Reinforcer", "Beg": cycles[trial - 1] + 13_000,
                               "End": cycles[trial - 1] + 13_100})
        protocol = pd.DataFrame(events)
        with self.assertRaisesRegex(Exception, "disagrees with ExperimentSpec"):
            verify_expected_us(protocol, "all3sTrace")
        protocol.loc[protocol["Type"].eq("Reinforcer"), "Beg"] -= 4_000
        time, count = verify_expected_us(protocol, "all3sTrace")
        self.assertEqual((time, count), (9., 46))

    def test_renderer_has_expected_axes_and_no_single_fish_iqr(self) -> None:
        rows = []
        for group in profile_groups("allDelay"):
            for time in (-19.75, .25, 19.75):
                rows.append({"experiment_id": "allDelay", "plot_stratum": STRATA[0],
                             "group_type": group["group_type"], "group_name": group["group_name"],
                             "group_order": group["group_order"], "time_s": time,
                             "signed_median": -.2, "signed_q25": -.2, "signed_q75": -.2,
                             "signed_fish": 1, "movement_median": .5, "movement_q25": .5,
                             "movement_q75": .5, "movement_fish": 1})
        flow = pd.DataFrame({"experiment_id": ["allDelay"], "plot_stratum": [STRATA[0]]})
        figure, panel_ids, mappings = _draw_rows(pd.DataFrame(rows), flow,
            experiment_id="allDelay", expected_us_s=9., kind="main", value="signed", y_limit=.5)
        try:
            self.assertEqual(len(panel_ids), 10)
            self.assertEqual(tuple(figure.axes[0].get_xlim()), (-20., 20.))
            self.assertFalse(any(key.startswith("iqr__") for key in mappings))
            self.assertEqual(sum(key.startswith("us__") for key in mappings), 10)
        finally:
            import matplotlib.pyplot as plt
            plt.close(figure)

    def test_analysis_then_saved_data_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metric = "tail_length_weighted_angular_l1"
            rows = []
            fake_cohorts = {}
            for experiment in EXPERIMENTS:
                conditioned = "delay" if experiment == "allDelay" else "trace"
                recordings = (f"{experiment}-conditioned", f"{experiment}-control")
                fake_cohorts[experiment] = SimpleNamespace(
                    cohort_id=f"cohort-{experiment}", cohort_hash=f"hash-{experiment}",
                    experiment_id=experiment, metric_recipe="tail-candidate-corrected",
                    recording_ids=recordings,
                    fish_by_recording={item: item for item in recordings},
                    condition_by_recording={recordings[0]: conditioned, recordings[1]: "control"},
                )
                for condition, recording, label in ((conditioned, recordings[0], "Learner"),
                                                     ("control", recordings[1], "Non-learner")):
                    rows.append({"experiment_id": experiment, "condition_id": condition,
                                 "fish_id": recording, "classifier_label": label,
                                 "classification_eligible": True, "ineligible_reason": "",
                                 "input_metric_id": metric})
            path = root / "labels.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            path.with_suffix(".manifest.json").write_text(json.dumps({
                "table_sha256": sha256_file(path), "input_metric_id": metric,
                "classifier_execution_id": "run-1", "validation_mode": "descriptive",
                "cohort_hashes": {experiment: f"hash-{experiment}" for experiment in EXPERIMENTS},
            }))

            def fake_read(_root: Path, recording: str, _metric: str, _recipe: str):
                bins = pd.DataFrame({
                    "recording_id": recording,
                    "trial_number": np.repeat(np.arange(5, 95), 2),
                    "time_s": np.tile([-.25, .25], 90),
                    "signed_log_vigor": -.2 if "conditioned" in recording else .1,
                    "movement_probability": .3 if "conditioned" in recording else .6,
                    "coverage": 1.,
                })
                return bins, pd.DataFrame(), pd.DataFrame(), []

            with patch("classical_conditioning.analysis.figure4._load_primary_cohort",
                       side_effect=lambda _root, cohort_id: next(
                           item for item in fake_cohorts.values() if item.cohort_id == cohort_id)), \
                 patch("classical_conditioning.analysis.figure4._read_recording", side_effect=fake_read), \
                 patch("classical_conditioning.analysis.figure4.verify_expected_us",
                       side_effect=lambda _protocol, experiment: (
                           {"allDelay": 9., "all3sTrace": 13., "all10sTrace": 20.}[experiment], 46)):
                summary_path = analyze_figure4(root, analysis_id="figure4-test", metric_id=metric,
                    cohort_ids={item: f"cohort-{item}" for item in EXPERIMENTS}, learner_manifest=path)
            summary, tables = load_figure4_analysis(summary_path)
            self.assertEqual(len(tables["sample-flow"]), 6)
            self.assertEqual(summary["expected_us"]["all10sTrace"]["expected_us_s"], 20.)
            with patch("classical_conditioning.figures.export._git_dirty", return_value=False):
                results = render_figure4(summary_path, output_dir=root / "figures", mode=FigureMode.STATIC)
            self.assertEqual(len(results), 12)
            self.assertTrue(all(result.outputs[0].is_file() and result.sidecar.is_file()
                                for result in results))
            (root / "Processed data" / "Analyses" / "figure4-test" / "figure4" / "group-bins.parquet").write_bytes(b"changed")
            with self.assertRaisesRegex(Exception, "changed"):
                load_figure4_analysis(summary_path)


if __name__ == "__main__":
    unittest.main()
