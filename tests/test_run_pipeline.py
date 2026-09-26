from __future__ import annotations

import json
import math
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from classical_conditioning.cli import build_parser
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cohort import freeze_cohort_manifest
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.pipeline import _paper_panel_statuses, run_pipeline
from classical_conditioning.intake import IntakeBatchResult
from classical_conditioning.run_config import load_pipeline_run_config


class PipelineRunConfigTests(unittest.TestCase):
    @staticmethod
    def _write_two_trial_raw_fixture(raw: Path) -> None:
        """Write a 100-Hz, 16-point triplet covering complete CS/US windows."""
        name = "20260101_01_delay_black-1_test_6dpf"
        camera = ["FrameID ElapsedTime AbsoluteTime"]
        fields = ["FrameID"] + [
            field for point in range(16)
            for field in (f"x{point}", f"y{point}", f"angle{point}")
        ]
        tracking = [" ".join(fields)]
        for frame in range(9001):
            time_ms = frame * 10
            camera.append(f"{frame + 1} {time_ms} {time_ms}")
            wave = math.sin(frame / 5)
            values = [str(frame + 1)]
            for point in range(16):
                values.extend((str(point), f"{point * wave * 0.01:.6f}", "0.000000"))
            tracking.append(" ".join(values))
        (raw / f"{name}_cam.txt").write_text("\n".join(camera) + "\n", encoding="utf-8")
        (raw / f"{name}_mp tail tracking.txt").write_text("\n".join(tracking) + "\n", encoding="utf-8")
        (raw / f"{name}_stim control.txt").write_text(
            "Type Beg End\nCycle 25000 35000\nReinforcer 34000 34100\n"
            "Cycle 65000 75000\nReinforcer 74000 74100\n",
            encoding="utf-8",
        )

    def test_loads_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            save = Path(temporary) / "save"
            raw.mkdir()
            save.mkdir()
            config_path = Path(temporary) / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "raw_dir": str(raw),
                        "save_dir": str(save),
                        "experiment": "allDelay",
                        "analysis_id": "delay",
                        "keep_conditions": ["control", "delay"],
                    }
                ),
                encoding="utf-8",
            )
            config = load_pipeline_run_config(config_path)
            self.assertEqual(config.experiment, "allDelay")
            self.assertEqual(
                config.resolved_candidate_analysis_id(),
                "delay",
            )

    def test_rejects_missing_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            save = Path(temporary) / "save"
            raw.mkdir()
            save.mkdir()
            config_path = Path(temporary) / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "raw_dir": str(raw),
                        "save_dir": str(save),
                        "analysis_id": "delay",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_pipeline_run_config(config_path)

    def test_defaults_to_the_single_corrected_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            save = Path(temporary) / "save"
            raw.mkdir()
            save.mkdir()
            config_path = Path(temporary) / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "raw_dir": str(raw),
                        "save_dir": str(save),
                        "experiment": "allDelay",
                        "analysis_id": "candidate-only",
                    }
                ),
                encoding="utf-8",
            )
            config = load_pipeline_run_config(config_path)
            self.assertEqual(config.resolved_candidate_analysis_id(), "candidate-only")

    def test_rejects_retired_legacy_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            save = Path(temporary) / "save"
            raw.mkdir()
            save.mkdir()
            config_path = Path(temporary) / "run.json"
            config_path.write_text(json.dumps({"raw_dir": str(raw), "save_dir": str(save), "experiment": "allDelay", "analysis_id": "retired", "routes": ["legacy"]}), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "Obsolete"):
                load_pipeline_run_config(config_path)

            config_path.write_text(
                json.dumps(
                    {
                        "raw_dir": str(raw),
                        "save_dir": str(save),
                        "experiment": "allDelay",
                        "analysis_id": "retired-field",
                        "legacy_alignment": "CS",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "Obsolete pipeline settings"):
                load_pipeline_run_config(config_path)

    def test_cli_exposes_run_pipeline(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-pipeline", "--config", "configs/example-run.json"]
        )
        self.assertEqual(args.command, "run-pipeline")
        self.assertEqual(args.config, Path("configs/example-run.json"))
        self.assertFalse(args.quiet)

    def test_cli_uses_friendly_preprocessing_mode_labels(self) -> None:
        parser = build_parser()
        preprocess = parser.parse_args(
            [
                "preprocess",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--recipe",
                "corrected",
            ]
        )
        metrics = parser.parse_args(
            [
                "activity-metrics",
                "--project-dir",
                "paper",
                "--recording-id",
                "recording-a",
                "--recipe",
                "development",
            ]
        )

        self.assertEqual(preprocess.recipe, "corrected-preprocess")
        self.assertEqual(metrics.recipe, "tail-candidate-development")

    def test_show_progress_defaults_true(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            save = Path(temporary) / "save"
            raw.mkdir()
            save.mkdir()
            config_path = Path(temporary) / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "raw_dir": str(raw),
                        "save_dir": str(save),
                        "experiment": "allDelay",
                        "analysis_id": "candidate-only",
                    }
                ),
                encoding="utf-8",
            )
            config = load_pipeline_run_config(config_path)
            self.assertTrue(config.show_progress)

    def test_rejects_empty_optional_identifier_and_string_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            save = Path(temporary) / "save"
            raw.mkdir()
            save.mkdir()
            config_path = Path(temporary) / "run.json"
            common = {
                "raw_dir": str(raw),
                "save_dir": str(save),
                "experiment": "allDelay",
                "analysis_id": "strict-config",
            }
            config_path.write_text(
                json.dumps({**common, "cohort_id": ""}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ConfigurationError, "cohort_id"):
                load_pipeline_run_config(config_path)

            config_path.write_text(
                json.dumps({**common, "overwrite": "false"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ConfigurationError, "overwrite"):
                load_pipeline_run_config(config_path)

    def test_strict_json_and_obsolete_stage_switches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, save = root / "raw", root / "save"
            raw.mkdir()
            save.mkdir()
            path = root / "run.json"
            common = {
                "raw_dir": str(raw), "save_dir": str(save),
                "experiment": "allDelay", "analysis_id": "stable",
            }
            path.write_text(json.dumps(common)[:-1] + ",}", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "Invalid JSON"):
                load_pipeline_run_config(path)
            for field in ("run_intake", "run_figures", "routes", "candidate_runner_recipe"):
                path.write_text(json.dumps({**common, field: True}), encoding="utf-8")
                with self.assertRaisesRegex(ConfigurationError, "Obsolete"):
                    load_pipeline_run_config(path)
            path.write_text(json.dumps({**common, "mystery": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "Unknown"):
                load_pipeline_run_config(path)
            path.write_text(json.dumps({**common, "analysis_id": "stable-v2"}), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "version suffix"):
                load_pipeline_run_config(path)
            path.write_text(json.dumps({**common, "cohort_id": "frozen", "metric": "unknown"}), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "metric must be one of"):
                load_pipeline_run_config(path)

    def test_paper_registry_has_all_panels_and_unapproved_mappings(self) -> None:
        panels = _paper_panel_statuses()
        registry = json.loads((Path(__file__).resolve().parents[1] / "configs" / "paper-figures" / "behavior-paper.json").read_text(encoding="utf-8"))
        self.assertEqual(len(panels), 27)
        self.assertEqual(set(panels), {
            *(f"fig-1{letter}" for letter in "ABCDEFGH"),
            *(f"fig-2{letter}" for letter in "ABCDEFGHI"),
            *(f"fig-3{letter}" for letter in "ABCDEFG"),
            *(f"fig-4{letter}" for letter in "ABC"),
        })
        self.assertIn("session protocol", panels["fig-1C"]["role"])
        self.assertIn("3sTrace", panels["fig-1G"]["role"])
        self.assertIn("inconclusive", panels["fig-2C"]["role"])
        self.assertIn("signed", panels["fig-4A"]["role"])
        self.assertIn("selected-block ratio", registry["artifact_templates"])
        self.assertTrue(registry["rendering"]["semantic_svg_ids"])
        self.assertTrue(all(item["status"] == "blocked" and item["reason"] for item in panels.values()))

    def test_failed_inventory_still_writes_summary_and_global_figure_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, save = root / "raw", root / "save"
            raw.mkdir()
            config_path = root / "run.json"
            config_path.write_text(json.dumps({
                "raw_dir": str(raw), "save_dir": str(save),
                "experiment": "allDelay", "analysis_id": "empty",
                "show_progress": False,
            }), encoding="utf-8")
            config = load_pipeline_run_config(config_path)
            with self.assertRaisesRegex(ConfigurationError, "Pipeline finished"):
                run_pipeline(config)
            summary = json.loads((save / "Metadata" / "empty_pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(len(summary["paper_panels"]), 27)
            self.assertEqual(summary["figures"]["metric-comparison:CS:total-activity"]["status"], "blocked")
            self.assertTrue(summary["stage_errors"])

    def test_end_to_end_twice_reuses_intake_and_refreshes_stable_derived_paths(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, save = root / "raw", root / "save"
            raw.mkdir()
            self._write_two_trial_raw_fixture(raw)
            config_path = root / "run.json"
            config_path.write_text(json.dumps({
                "raw_dir": str(raw), "save_dir": str(save),
                "experiment": "allDelay", "analysis_id": "fixture",
                "show_progress": False, "batch_size": 3000,
            }), encoding="utf-8")
            config = load_pipeline_run_config(config_path)
            try:
                first = run_pipeline(config)
            except ConfigurationError:
                failed_summary = json.loads(
                    (save / "Metadata" / "fixture_pipeline_run.json").read_text(encoding="utf-8")
                )
                self.fail(str(failed_summary["stage_errors"]))
            self.assertEqual(first.intake_completed, ("20260101_01",))
            first_summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(first_summary["status"], "complete")
            self.assertEqual(len(first_summary["paper_panels"]), 27)
            self.assertEqual(len(first_summary["figures"]), 32)
            self.assertEqual(
                sum(item["status"] == "completed" for item in first_summary["figures"].values()),
                23,
            )
            self.assertNotIn("20260101_01:profile:US:signed-log-vigor", first_summary["figures"])
            self.assertTrue(all(
                item["status"] in {"completed", "blocked"}
                for item in first_summary["figures"].values()
            ))

            cohort = freeze_cohort_manifest(
                save,
                pd.DataFrame([{
                    "experiment_id": "allDelay", "recording_id": "20260101_01",
                    "fish_id": "20260101_01", "condition_id": "delay",
                    "technical_valid": True, "technical_exclusion_reason": pd.NA,
                    "behavioral_engagement": True,
                    "behavioral_engagement_reason": "fixture only",
                    "primary_included": True, "sensitivity_population_ids": [],
                    "review_status": "approved", "reviewer": "fixture",
                    "reviewed_at": "2026-09-23T00:00:00Z",
                    "source_qc_artifact_id": "fixture-qc",
                }]),
                cohort_id="frozen-fixture", policy_id="fixture-policy",
            )
            cohort_hash = sha256_file(cohort.manifest_path)
            corrected = save / "Processed data" / "20260101_01" / "frame_preprocessed_corrected.parquet"
            stable_path = corrected.resolve()
            expected_corrected_hash = sha256_file(corrected)
            with corrected.open("ab") as stream:
                stream.write(b"tampered")
            second = run_pipeline(config)
            second_summary = json.loads(second.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(second.intake_skipped, ("20260101_01",))
            self.assertEqual(second_summary["status"], "complete")
            self.assertEqual(corrected.resolve(), stable_path)
            self.assertEqual(sha256_file(corrected), expected_corrected_hash)
            self.assertEqual(sha256_file(cohort.manifest_path), cohort_hash)
            self.assertEqual(
                first_summary["selection_assessment"],
                second_summary["selection_assessment"],
            )
            self.assertEqual(
                Path(second_summary["selection_assessment"]).parent.name,
                "fixture",
            )
            self.assertEqual(
                set(first_summary["figures"]), set(second_summary["figures"])
            )
            self.assertEqual(
                sum(item["status"] == "completed" for item in second_summary["figures"].values()),
                23,
            )
            self.assertTrue(all(
                item["status"] in {"completed", "blocked", "failed"}
                and (item["status"] == "completed" or item["reason"])
                for item in second_summary["figures"].values()
            ))

    def test_figure_failure_is_recorded_without_hiding_other_required_figures(self) -> None:
        class ImmediatePool:
            def __init__(self, *, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self) -> "ImmediatePool":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def submit(self, function: object, *_args: object, **kwargs: object) -> Future[Path]:
                future: Future[Path] = Future()
                if getattr(function, "__name__", "") == "build_candidate_profile_figure" and kwargs.get("figure_id") == "bout-outcomes":
                    future.set_exception(RuntimeError("renderer unavailable"))
                else:
                    future.set_result(Path("rendered.png"))
                return future

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, save = root / "raw", root / "save"
            raw.mkdir()
            ledger = save / "Metadata" / "intake_status.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_text(json.dumps({"recordings": {
                "20260101_01": {"status": "ready", "reason": None}
            }}), encoding="utf-8")
            inventory = {"records_sha256": "fixture-hash", "records": [{
                "recording_id": "20260101_01", "recording_name": "fixture-delay",
                "condition_id": "delay", "status": "COMPLETE",
            }]}

            def candidate(*_args: object, **kwargs: object) -> SimpleNamespace:
                callback = kwargs["on_stage_ready"]
                callback("20260101_01", "movement-state")
                callback("20260101_01", "temporal-profiles")
                return SimpleNamespace(
                    manifest_path=save / "Metadata" / "runner.json",
                    step_status={"cohort": {"three-metric-comparison": "completed"}},
                )

            config_path = root / "run.json"
            config_path.write_text(json.dumps({
                "raw_dir": str(raw), "save_dir": str(save),
                "experiment": "allDelay", "analysis_id": "failure-fixture",
                "show_progress": False,
            }), encoding="utf-8")
            config = load_pipeline_run_config(config_path)
            with (
                patch("classical_conditioning.pipeline.build_recording_inventory", return_value=inventory),
                patch("classical_conditioning.pipeline.intake_recordings", return_value=IntakeBatchResult(
                    recording_ids=("20260101_01",), completed=("20260101_01",),
                    skipped=(), failed=(), ledger_path=ledger,
                )),
                patch("classical_conditioning.pipeline.run_candidate_development_pipeline", side_effect=candidate),
                patch("classical_conditioning.pipeline.ProcessPoolExecutor", ImmediatePool),
            ):
                with self.assertRaisesRegex(ConfigurationError, "Pipeline finished"):
                    run_pipeline(config)
            summary = json.loads((save / "Metadata" / "failure-fixture_pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["figures"]["20260101_01:profile:CS:bout-outcomes"]["status"], "failed")
            self.assertIn("renderer unavailable", summary["figures"]["20260101_01:profile:CS:bout-outcomes"]["reason"])
            self.assertEqual(summary["figures"]["metric-comparison:CS:total-activity"]["status"], "completed")
            self.assertEqual(len(summary["figures"]), 32)

    def test_interrupted_run_settles_already_submitted_figures(self) -> None:
        class ImmediatePool:
            def __init__(self, *, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self) -> "ImmediatePool":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def submit(self, _function: object, *_args: object, **_kwargs: object) -> Future[Path]:
                future: Future[Path] = Future()
                future.set_result(Path("rendered.png"))
                return future

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, save = root / "raw", root / "save"
            raw.mkdir()
            ledger = save / "Metadata" / "intake_status.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_text(json.dumps({"recordings": {
                "20260101_01": {"status": "ready", "reason": None}
            }}), encoding="utf-8")
            config_path = root / "run.json"
            config_path.write_text(json.dumps({
                "raw_dir": str(raw), "save_dir": str(save),
                "experiment": "allDelay", "analysis_id": "interrupted",
                "cohort_id": "reviewed", "metric": "tail_length_weighted_angular_l1",
                "show_progress": False,
            }), encoding="utf-8")
            config = load_pipeline_run_config(config_path)

            def candidate(*_args: object, **kwargs: object) -> SimpleNamespace:
                kwargs["on_stage_ready"]("20260101_01", "movement-state")
                return SimpleNamespace(manifest_path=save / "runner.json", step_status={})

            with (
                patch("classical_conditioning.pipeline.build_recording_inventory", return_value={
                    "records_sha256": "fixture", "records": [{
                        "recording_id": "20260101_01", "condition_id": "delay",
                        "status": "COMPLETE",
                    }],
                }),
                patch("classical_conditioning.pipeline.intake_recordings", return_value=IntakeBatchResult(
                    recording_ids=("20260101_01",), completed=("20260101_01",),
                    skipped=(), failed=(), ledger_path=ledger,
                )),
                patch("classical_conditioning.pipeline.run_candidate_development_pipeline", side_effect=candidate),
                patch("classical_conditioning.pipeline.assess_discarding", return_value=SimpleNamespace(
                    summary_path=save / "assessment.json"
                )),
                patch("classical_conditioning.pipeline.build_cohort_trial_outcomes"),
                patch("classical_conditioning.pipeline.load_cohort_manifest", side_effect=RuntimeError("manifest unavailable")),
                patch("classical_conditioning.pipeline.ProcessPoolExecutor", ImmediatePool),
            ):
                with self.assertRaisesRegex(ConfigurationError, "Pipeline finished"):
                    run_pipeline(config)
            summary = json.loads((save / "Metadata" / "interrupted_pipeline_run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["figures"]["20260101_01:detector-review"]["status"], "completed")
            self.assertFalse(any(value["status"] == "pending" for value in summary["figures"].values()))
            self.assertIn("manifest unavailable", " ".join(summary["stage_errors"]))


if __name__ == "__main__":
    unittest.main()
