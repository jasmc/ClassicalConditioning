from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from classical_conditioning.analysis.discarding import assess_discarding
from classical_conditioning.analysis.observation import (
    observe_contributions, publish_observation, try_observation,
)
from classical_conditioning.analysis.selection_observation import project_later_checks
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import ArtifactIntegrityError, ConfigurationError


def projection_fixture():
    trials = [trial for trial in get_experiment_spec("allDelay").analysis_trials if trial.alignment.value == "CS"]
    return pd.DataFrame({
        "experiment_id": "allDelay", "alignment": "CS", "metric_id": "legacy_distal_angular_speed",
        "trial_number": [trial.trial_number for trial in trials],
        "block_10_name": [trial.block_10_name for trial in trials],
        "baseline_total_activity": 1., "response_total_activity": .5,
    })


class ObservationOnlyTests(unittest.TestCase):
    def test_variants_use_instantiated_feature_blocks_without_mutating_input(self):
        frame = projection_fixture()
        # Nominal uses Late Train/Early Test; later variants also require Train 6.
        frame.loc[frame.trial_number.between(40, 44), "response_total_activity"] = np.nan
        original = frame.copy(deep=True)
        evidence = project_later_checks(frame, "legacy_distal_angular_speed").set_index("rule_id")
        self.assertEqual(evidence.loc["learner_nominal_coverage", "status"], "pass")
        for variant in ("new", "improved", "wip"):
            self.assertEqual(evidence.loc[f"learner_{variant}_coverage", "status"], "fail")
            self.assertEqual(evidence.loc[f"learner_{variant}_features", "status"], "not_evaluable")
        pd.testing.assert_frame_equal(frame, original)
        self.assertFalse(evidence.affects_membership.any())

    def test_any_block_is_any_and_missing_original_metadata_is_explicit(self):
        frame = projection_fixture()
        frame.loc[~frame.trial_number.between(5, 10), "response_total_activity"] = np.nan
        evidence = project_later_checks(frame, "legacy_distal_angular_speed").set_index("rule_id")
        self.assertEqual(evidence.loc["normalized_any_block", "status"], "pass")
        self.assertEqual(evidence.loc["normalized_paired_blocks", "status"], "fail")
        self.assertEqual(evidence.loc["normalized_block_statistics", "status"], "not_evaluable")
        self.assertFalse(evidence.loc["normalized_window_nan_fraction", "enabled_in_legacy"])
        frame.loc[frame.trial_number.eq(10), "response_total_activity"] = np.nan
        evidence = project_later_checks(frame, "legacy_distal_angular_speed").set_index("rule_id")
        self.assertEqual(evidence.loc["normalized_any_block", "status"], "fail")

    def test_observations_preserve_snapshots_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = publish_observation(root, root / "audit", tables={"fish": pd.DataFrame({"id": ["a"]})}, identity={"run": 1})
            saved = Path(first["path"]).read_bytes()
            second = publish_observation(root, root / "audit", tables={"fish": pd.DataFrame({"id": ["a", "b"]})}, identity={"run": 2})
            self.assertNotEqual(first["path"], second["path"])
            self.assertEqual(Path(first["path"]).read_bytes(), saved)
            self.assertEqual(sha256_file(Path(first["path"])), first["sha256"])
            pd.testing.assert_frame_equal(pd.read_csv(Path(first["path"]).parent / "fish.csv"), pd.DataFrame({"id": ["a"]}))
            (Path(first["path"]).parent / "fish.parquet").write_bytes(b"broken")
            with self.assertRaises(ArtifactIntegrityError):
                publish_observation(root, root / "audit", tables={"fish": pd.DataFrame({"id": ["a"]})}, identity={"run": 1})

    def test_contribution_join_retains_zero_contributors_and_observer_failures_are_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = pd.DataFrame({"fish_id": ["a", "a", "b"], "trial": [1, 2, 1]})
            original = candidates.copy()
            result = observe_contributions(root, root / "audit", candidates=candidates,
                                           contributors=candidates.iloc[:1], keys=["fish_id", "trial"],
                                           consumer="test", identity={})
            counts = pd.read_parquet(Path(result["path"]).parent / "fish-counts.parquet").set_index("fish_id")
            self.assertEqual(counts.loc["a", "contributing_observations"], 1)
            self.assertEqual(counts.loc["b", "contributing_observations"], 0)
            pd.testing.assert_frame_equal(candidates, original)
            failed = try_observation(lambda: (_ for _ in ()).throw(OSError("offline")))
            self.assertEqual(failed["status"], "failed")
            self.assertIn("offline", failed["error"])

    def test_inventory_keeps_absent_expected_cases_and_never_proposes_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, save = root / "raw", root / "save"
            raw.mkdir(); save.mkdir()
            policy = root / "policy.json"
            policy.write_text(json.dumps({"approval_status": "approved", "approved_by": "fixture",
                                          "approved_at": "2026-10-30"}))
            inventory = {"records_sha256": "fixture", "records": [{
                "recording_id": "a", "recording_name": "a_control", "condition_id": "control",
                "status": "INCOMPLETE", "tracking_schema": {"ok": False},
            }]}
            result = assess_discarding(raw, save, analysis_id="audit", experiment="allDelay",
                                      metric_id="legacy_distal_angular_speed", inventory=inventory,
                                      expected_recording_ids=["missing"], technical_policy_path=policy,
                                      disabled_rules=["last_us"])
            self.assertEqual(result.observation["status"], "complete")
            evidence = Path(result.observation["path"]).parent
            records = pd.read_parquet(evidence / "inventory.parquet")
            self.assertEqual(set(records.recording_id), {"a", "missing"})
            self.assertEqual(set(records.decision_status), {"pending_review"})
            self.assertTrue(pd.read_parquet(result.technical_path).primary_candidate.isna().all())
            checks = pd.read_parquet(evidence / "historical-checks.parquet")
            self.assertTrue(set(checks.status) <= {"pass", "fail", "not_applicable", "not_evaluable"})
            self.assertFalse(checks.loc[checks.rule_id.eq("last_us"), "enabled"].any())
            later = pd.read_parquet(evidence / "later-checks.parquet")
            self.assertTrue(later.status.eq("not_evaluable").all())
            self.assertFalse((save / "Processed data" / "Cohorts").exists())
            with self.assertRaises(ConfigurationError):
                assess_discarding(raw, save, analysis_id="audit", experiment="allDelay",
                                  metric_id="legacy_distal_angular_speed", observation_only=False)

    def test_model_contributions_match_actual_transform(self):
        from tests.test_model_input import _synthetic_outcomes
        from classical_conditioning.analysis.inference.model_input import (
            build_candidate_model_input, observe_model_input, ModelInputConfig,
        )
        frame = _synthetic_outcomes()
        frame.loc[frame.index[0], "response_valid_sample_count"] = 0
        original = frame.copy(deep=True)
        model = build_candidate_model_input(frame)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = observe_model_input(root, root / "audit", frame, model, config=ModelInputConfig(), identity={})
            self.assertEqual(reference["status"], "complete", reference)
            rows = pd.read_parquet(Path(reference["path"]).parent / "contributions.parquet")
            self.assertEqual(int(rows.contributed.sum()), len(model))
            self.assertIn("missing_response_samples", ";".join(rows.reason))
        pd.testing.assert_frame_equal(frame, original)

    def test_saved_normalized_input_keeps_lme_and_test_populations_distinct(self):
        from classical_conditioning.analysis.selection_observation import observe_normalized_input
        rows = []
        for fish, blocks in (("a", ["PT"] * 6 + ["ET"]), ("b", ["PT"] * 2 + ["ET"] * 2 + ["LT"] * 2)):
            for i, block in enumerate(blocks):
                rows.append({"Fish": fish, "Exp.": "delay", "Trial number": i + 1, "Block name": block,
                             "Mean 15 s before": 1., "Mean CR": .5, "Normalized vigor": .5,
                             "extra_metadata": None if fish == "b" and block == "LT" else "present"})
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pooled.parquet"
            frame.to_parquet(source)
            digest = sha256_file(source)
            result = observe_normalized_input(root, root / "audit", source, ["PT", "ET", "LT"])
            trial = pd.read_parquet(Path(result["path"]).parent / "trial-contributions.parquet")
            # A passes ANY-block and keeps its sparse ET row; B passes no six-row block.
            self.assertEqual(int(trial.lme_contributed.sum()), 7)
            self.assertFalse(trial.wilcoxon_branch_contributed.any())
            self.assertEqual(int(trial.block_summary_contributed.sum()), 11)
            self.assertEqual(sha256_file(source), digest)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(source))

    def test_learner_observer_calls_original_once_and_does_not_change_result(self):
        from dataclasses import dataclass
        from types import SimpleNamespace
        from classical_conditioning.analysis.legacy_learner_worker import install_observers

        @dataclass
        class Config:
            minimum: int = 1

        calls = []
        def original(frame, config):
            calls.append("called")
            return frame.iloc[:1].copy()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = pd.DataFrame({"Fish_ID": ["a", "b"], "Condition": ["delay", "delay"], "Trial number": [1, 1]})
            source = root / "input.parquet"
            data.to_parquet(source)
            module = SimpleNamespace(__file__=__file__, analysis_cfg=Config(), _filter_fish_by_trials=original)
            finish = install_observers(module, input_path=source, output_path=root / "learner.parquet", variant="fixture")
            observed = module._filter_fish_by_trials(frame=data, config=Config())
            pd.testing.assert_frame_equal(observed, data.iloc[:1])
            self.assertEqual(calls, ["called"])
            finish(ValueError("fixture model failure"))
            reference = json.loads((root / "learner.observation.json").read_text())
            evidence = json.loads(Path(reference["path"]).read_text())
            self.assertEqual(evidence["identity"]["model_status"], "failed")
            stages = pd.read_parquet(Path(reference["path"]).parent / "stages.parquet")
            self.assertEqual(stages.status.tolist(), ["complete"])

    def test_routine_config_rejects_activation_and_records_observation_settings(self):
        from classical_conditioning.run_config import load_pipeline_run_config, pipeline_config_to_dict
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "config.json"
            settings = {"raw_dir": str(root / "raw"), "save_dir": str(root / "save"),
                        "experiment": "allDelay", "analysis_id": "audit", "expected_recording_ids": ["missing"]}
            path.write_text(json.dumps(settings))
            config = load_pipeline_run_config(path)
            self.assertEqual(config.expected_recording_ids, ("missing",))
            self.assertTrue(pipeline_config_to_dict(config)["observation_only"])
            path.write_text(json.dumps({**settings, "observation_only": False}))
            with self.assertRaises(ConfigurationError):
                load_pipeline_run_config(path)

    def test_pipeline_population_and_numbers_unchanged_on_off_and_after_failure(self):
        from tests.test_run_pipeline import PipelineRunConfigTests
        from classical_conditioning.pipeline import run_pipeline
        from classical_conditioning.run_config import PipelineRunConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, save = root / "raw", root / "save"
            raw.mkdir()
            PipelineRunConfigTests._write_two_trial_raw_fixture(raw)
            config = PipelineRunConfig(raw, save, "allDelay", "audit-test", show_progress=False,
                                       assessment_enabled=False, overwrite=True)
            first = run_pipeline(config)
            before = {path: pd.read_parquet(path) for path in (save / "Processed data").rglob("*.parquet")
                      if "observations" not in path.parts and "evidence" not in path.parts}
            enabled = run_pipeline(replace(config, assessment_enabled=True))
            status = json.loads(enabled.summary_path.read_text())["assessment_status"]
            self.assertEqual(status["status"], "complete", status)
            self.assertEqual(status["observation"]["status"], "complete", status)
            self.assertEqual(first.recording_ids, enabled.recording_ids)
            for path, frame in before.items():
                pd.testing.assert_frame_equal(frame, pd.read_parquet(path))
            with patch("classical_conditioning.pipeline.assess_discarding", side_effect=OSError("fixture assessment failure")):
                failed_audit = run_pipeline(replace(config, assessment_enabled=True))
            summary = json.loads(failed_audit.summary_path.read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["assessment_status"]["status"], "failed")
            self.assertFalse(summary["stage_errors"])
            for path, frame in before.items():
                pd.testing.assert_frame_equal(frame, pd.read_parquet(path))


if __name__ == "__main__":
    unittest.main()
