from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from classical_conditioning.analysis.figure4 import load_classification_manifest
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cohort import load_cohort_manifest, logical_cohort_hash
from scripts.finalize_3strace_exploratory import METRIC, classify, freeze


class TraceExploratoryFinalizeTests(unittest.TestCase):
    def test_frozen_complete_cohort_and_wip_labels_authenticate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            metadata = project / "Metadata"
            assessment_dir = project / "Processed data" / "Discarding" / "all3sTrace-full"
            metadata.mkdir(parents=True)
            assessment_dir.mkdir(parents=True)
            selected = ("trace-fish", "control-fish")
            for name in selected:
                (metadata / f"{name}_source_manifest.json").write_text("{}")
            assessment_path = assessment_dir / "assessment-summary.json"
            artifacts = {}
            for name, filename in {
                "technical": "technical-assessment.parquet",
                "exploratory": "exploratory-assessment.parquet",
                "rules": "legacy-rule-results.parquet",
                "flow": "discarding-flow.parquet",
                "details": "rule-details.parquet",
            }.items():
                path = assessment_dir / filename
                path.write_bytes(name.encode())
                artifacts[name] = sha256_file(path)
            assessment_path.write_text(json.dumps({
                "assessment_hash": "assess-hash", "selected_metric": METRIC,
                "input_identity": {"experiment": "all3sTrace", "metric_id": METRIC,
                                   "metric_recipe": "tail-candidate-corrected",
                                   "selected_recording_ids": list(selected)},
                "artifacts": artifacts,
            }))
            (metadata / "all3sTrace-full_pipeline_run.json").write_text(json.dumps({
                "status": "complete", "recording_ids": list(selected),
                "active_recording_ids": list(selected), "stage_errors": [],
                "candidate_runner_status": "candidate-manifest.json",
                "selection_assessment": str(assessment_path),
            }))
            (metadata / "recording_inventory.json").write_text(json.dumps({
                "source_hashes_included": True,
                "records": [
                    {"recording_id": "trace-fish", "status": "COMPLETE", "condition_id": "trace"},
                    {"recording_id": "control-fish", "status": "COMPLETE", "condition_id": "control"},
                    {"recording_id": "incomplete", "status": "INCOMPLETE", "condition_id": "trace"},
                ],
            }))
            cohort_path = freeze(project)
            cohort = load_cohort_manifest(project, "all3sTrace-full-exploratory")
            self.assertEqual(set(cohort["recording_id"]), set(selected))
            self.assertTrue(cohort["primary_included"].all())
            comparison = project / "comparison"
            comparison.mkdir()
            pd.DataFrame({"Fish_ID": selected, "Condition": ["trace", "control"],
                          "learner_primary": [True, False]}).to_parquet(comparison / "legacy-wip.parquet")
            pd.DataFrame({
                "experiment_id": ["all3sTrace"] * 2,
                "condition_id": ["trace", "control"], "fish_id": list(selected),
                "legacy-wip_status": ["classified"] * 2,
                "legacy-wip_learner": [True, False],
            }).to_csv(comparison / "fish-comparison.csv", index=False)
            (comparison / "comparison.json").write_text(json.dumps({
                "experiment_id": "all3sTrace", "metric_id": METRIC,
                "cohort_hash": logical_cohort_hash(cohort),
                "cohort_sha256": sha256_file(cohort_path),
                "variants": [{"variant": "legacy-wip", "status": "completed",
                              "result_sha256": sha256_file(comparison / "legacy-wip.parquet"),
                              "algorithm_sha256": "algorithm", "config_sha256": "config"}],
            }))
            manifest_path = classify(
                project, comparison, analysis_id="figure4-3strace-window13",
                assessment_summary=assessment_path,
                classifier_execution_id="legacy-wip-3strace-tail-l1-window13-exploratory",
            )
            labels, identity = load_classification_manifest(manifest_path, METRIC, ("all3sTrace",))
            self.assertEqual(set(labels["classifier_label"]), {"Learner", "Non-learner"})
            self.assertEqual(identity["classifier_execution_id"], "legacy-wip-3strace-tail-l1-window13-exploratory")

            legacy_metric = "legacy_distal_angular_speed"
            legacy_assessment = project / "Processed data" / "Discarding" / "legacy-preview"
            legacy_assessment.mkdir()
            legacy_artifacts = {}
            for name, filename in {
                "technical": "technical-assessment.parquet",
                "exploratory": "exploratory-assessment.parquet",
                "rules": "legacy-rule-results.parquet",
                "flow": "discarding-flow.parquet",
                "details": "rule-details.parquet",
            }.items():
                path = legacy_assessment / filename
                path.write_bytes(name.encode())
                legacy_artifacts[name] = sha256_file(path)
            legacy_summary = legacy_assessment / "assessment-summary.json"
            legacy_summary.write_text(json.dumps({
                "assessment_hash": "legacy-assess-hash", "selected_metric": legacy_metric,
                "input_identity": {"experiment": "all3sTrace", "metric_id": legacy_metric,
                                   "metric_recipe": "tail-candidate-corrected",
                                   "selected_recording_ids": list(selected)},
                "artifacts": legacy_artifacts,
            }))
            comparison_report = json.loads((comparison / "comparison.json").read_text())
            comparison_report["metric_id"] = legacy_metric
            (comparison / "comparison.json").write_text(json.dumps(comparison_report))
            legacy_manifest = classify(
                project, comparison, analysis_id="figure4-3strace-legacy-preview",
                assessment_summary=legacy_summary,
                classifier_execution_id="legacy-wip-3strace-legacy-preview",
                metric_id=legacy_metric,
            )
            legacy_labels, legacy_identity = load_classification_manifest(
                legacy_manifest, legacy_metric, ("all3sTrace",)
            )
            self.assertEqual(set(legacy_labels["input_metric_id"]), {legacy_metric})
            self.assertEqual(legacy_identity["classifier_execution_id"],
                             "legacy-wip-3strace-legacy-preview")


if __name__ == "__main__":
    unittest.main()
