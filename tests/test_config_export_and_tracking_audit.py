from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from classical_conditioning.cli import main
from classical_conditioning.config import (
    config_hash,
    fish_key_from_recording_id,
    get_experiment_trial_map,
    get_legacy_paper_config,
    recording_id_from_fish_key,
)
from classical_conditioning.config.export import export_resolved_config
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.ingestion import (
    audit_tracking_file,
    classify_tracking_columns,
)


class IdentityAndTrialMapTests(unittest.TestCase):
    def test_fish_key_round_trip_from_recording_id(self) -> None:
        key = fish_key_from_recording_id(
            "20221115_04",
            experiment_id="allDelay",
        )
        self.assertEqual(key.experiment_id, "allDelay")
        self.assertEqual(key.day, "20221115")
        self.assertEqual(key.fish_number, "04")
        self.assertEqual(recording_id_from_fish_key(key), "20221115_04")

    def test_fish_key_rejects_condition_or_alignment_in_identity(self) -> None:
        with self.assertRaises(ConfigurationError):
            fish_key_from_recording_id(
                "20221115_04_delay",
                experiment_id="allDelay",
            )
        with self.assertRaises(ConfigurationError):
            fish_key_from_recording_id(
                "20221115_04",
                experiment_id="",
            )

    def test_all_delay_trial_map_covers_expected_blocks(self) -> None:
        trial_map = get_experiment_trial_map("allDelay")
        self.assertEqual(trial_map["artifact_kind"], "experiment-trial-map-v1")
        self.assertEqual(trial_map["row_count"], 90 + 46)
        self.assertEqual(trial_map["cs_trial_count"], 90)
        self.assertEqual(trial_map["us_trial_count"], 46)
        first_cs = next(
            row for row in trial_map["rows"] if row["trial_id"] == "CS-0005"
        )
        self.assertEqual(first_cs["block_10_name"], "Pre-train")
        self.assertEqual(first_cs["block_5_id"], 1)
        self.assertEqual(first_cs["phase"], "Pre")
        tenth_cs = next(
            row for row in trial_map["rows"] if row["trial_id"] == "CS-0014"
        )
        self.assertEqual(tenth_cs["block_5_id"], 2)
        self.assertEqual(tenth_cs["block_10_id"], 1)


class ResolvedConfigExportTests(unittest.TestCase):
    def test_export_writes_config_trial_map_and_source_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_dir = Path(temporary)
            result = export_resolved_config(project_dir, overwrite=True)
            self.assertTrue(result.config_path.is_file())
            self.assertTrue(result.trial_map_path.is_file())
            self.assertTrue(result.source_report_path.is_file())

            payload = json.loads(result.config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifact_kind"], "resolved-analysis-config-v1")
            self.assertEqual(
                payload["config_hash"],
                config_hash(get_legacy_paper_config()),
            )
            self.assertEqual(payload["resolved"]["recipe_id"], "legacy-paper-v1")
            trial_map = json.loads(result.trial_map_path.read_text(encoding="utf-8"))
            self.assertEqual(trial_map["experiment_id"], "allDelay")
            source = json.loads(result.source_report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {entry["section"] for entry in source["source_trace"]},
                {"experiment", "preprocessing", "outcomes"},
            )

            with self.assertRaises(FileExistsError):
                export_resolved_config(project_dir, overwrite=False)

    def test_resolve_config_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_dir = Path(temporary)
            main(
                [
                    "resolve-config",
                    "--project-dir",
                    str(project_dir),
                    "--experiment",
                    "allDelay",
                    "--overwrite",
                ]
            )
            metadata = project_dir / "Metadata"
            self.assertTrue(
                (
                    metadata
                    / "resolved_config_legacy-paper-v1_allDelay.json"
                ).is_file()
            )


class TrackingAuditTests(unittest.TestCase):
    def test_classify_tracking_columns_keeps_xy_and_unrecognized(self) -> None:
        classification = classify_tracking_columns(
            [
                "FrameID",
                "x0",
                "y0",
                "angle0",
                "x1",
                "y1",
                "angle1",
                "confidence0",
                "ExtraField",
            ]
        )
        self.assertEqual(classification.angle_indices, (0, 1))
        self.assertTrue(classification.has_xy)
        self.assertEqual(classification.confidence_columns, ("confidence0",))
        self.assertEqual(classification.unrecognized_columns, ("ExtraField",))

    def test_audit_tracking_file_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tracking = root / "fish_mp tail tracking.txt"
            tracking.write_text(
                "FrameID x0 y0 angle0 x1 y1 angle1\n"
                "1 0.0 0.0 0.1 1.0 0.0 0.2\n"
                "2 0.1 0.0 0.1 1.1 0.0 0.3\n",
                encoding="utf-8",
            )
            audit = audit_tracking_file(tracking, sample_rows=10)
            self.assertEqual(audit["artifact_kind"], "tracking-field-audit-v1")
            self.assertTrue(audit["classification"]["has_xy"])
            self.assertEqual(
                audit["gate_t0_observations"]["raw_angle_units"],
                "radian",
            )
            self.assertEqual(
                audit["gate_t0_observations"]["spatial_units"],
                "pixel",
            )
            self.assertIn(
                "local_intersegment_bend",
                audit["gate_t0_observations"]["angle_semantics"],
            )

            output = root / "tracking_audit.json"
            main(
                [
                    "audit-tracking",
                    "--input",
                    str(tracking),
                    "--output",
                    str(output),
                    "--overwrite",
                ]
            )
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
