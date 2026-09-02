from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from classical_conditioning.cli import main
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.inventory import (
    build_recording_inventory,
    write_recording_inventory,
)


class RecordingInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "Raw"
        self.raw.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_component(
        self,
        recording_name: str,
        suffix: str,
        content: str = "test\n",
        directory: Path | None = None,
    ) -> Path:
        target_dir = directory or self.raw
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{recording_name}{suffix}"
        path.write_text(content, encoding="utf-8")
        return path

    def _write_triplet(
        self,
        recording_name: str,
        directory: Path | None = None,
    ) -> None:
        self._write_component(recording_name, "_cam.txt", "camera\n", directory)
        self._write_component(
            recording_name,
            "_mp tail tracking.txt",
            "tracking\n",
            directory,
        )
        self._write_component(
            recording_name,
            "_stim control.txt",
            "protocol\n",
            directory,
        )

    def test_discovers_and_hashes_complete_nested_triplet(self) -> None:
        name = "20260101_01_delay_black-1_test_6dpf"
        self._write_triplet(name, self.raw / "nested")

        inventory = build_recording_inventory(self.raw)

        self.assertEqual(inventory["record_count"], 1)
        self.assertEqual(inventory["status_counts"]["COMPLETE"], 1)
        record = inventory["records"][0]
        self.assertEqual(record["recording_id"], "20260101_01")
        self.assertEqual(record["status"], "COMPLETE")
        for components in record["components"].values():
            self.assertEqual(len(components), 1)
            self.assertEqual(len(components[0]["sha256"]), 64)
            self.assertFalse(Path(components[0]["relative_path"]).is_absolute())

    def test_reports_missing_component_and_hashes_present_input(self) -> None:
        name = "20260101_01_delay_black-1_test_6dpf"
        self._write_component(name, "_cam.txt")

        record = build_recording_inventory(self.raw)["records"][0]

        self.assertEqual(record["status"], "INCOMPLETE")
        self.assertEqual(
            record["missing_components"],
            ["tracking", "protocol"],
        )
        self.assertIn("sha256", record["components"]["camera"][0])

    def test_reports_invalid_recording_name_without_aborting_inventory(self) -> None:
        self._write_component("malformed", "_cam.txt")
        self._write_triplet("20260101_01_delay_black-1_test_6dpf")

        inventory = build_recording_inventory(self.raw)

        self.assertEqual(inventory["record_count"], 2)
        invalid = next(
            record
            for record in inventory["records"]
            if record["status"] == "INVALID_RECORDING_NAME"
        )
        self.assertIsNone(invalid["recording_id"])
        self.assertEqual(inventory["status_counts"]["COMPLETE"], 1)

    def test_reports_duplicate_component_and_recording_id_collision(self) -> None:
        name = "20260101_01_delay_black-1_test_6dpf"
        self._write_triplet(name)
        self._write_component(name, "_cam.txt", directory=self.raw / "duplicate")
        duplicate_record = build_recording_inventory(self.raw)["records"][0]
        self.assertEqual(duplicate_record["status"], "AMBIGUOUS_COMPONENT")
        self.assertEqual(duplicate_record["duplicate_components"], ["camera"])
        self.assertTrue(
            all(
                "sha256" in component
                for components in duplicate_record["components"].values()
                for component in components
            )
        )

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "Raw"
        self.raw.mkdir()
        first = "20260101_01_delay_black-1_test_6dpf"
        second = "20260101_01_control_black-1_test_6dpf"
        self._write_triplet(first, self.raw / "first")
        self._write_triplet(second, self.raw / "second")
        collision_records = build_recording_inventory(self.raw)["records"]
        self.assertEqual(
            {record["status"] for record in collision_records},
            {"AMBIGUOUS_RECORDING_ID"},
        )

    def test_inventory_hash_is_stable_across_runs(self) -> None:
        name = "20260101_01_delay_black-1_test_6dpf"
        self._write_triplet(name)
        first = build_recording_inventory(self.raw)
        second = build_recording_inventory(self.raw)
        self.assertEqual(first["records_sha256"], second["records_sha256"])
        self._write_component(name, "_cam.txt", "changed camera\n")
        changed = build_recording_inventory(self.raw)
        self.assertNotEqual(first["records_sha256"], changed["records_sha256"])

    def test_hashing_can_be_skipped_for_fast_discovery(self) -> None:
        self._write_triplet("20260101_01_delay_black-1_test_6dpf")
        inventory = build_recording_inventory(self.raw, hash_files=False)
        self.assertFalse(inventory["source_hashes_included"])
        self.assertTrue(
            all(
                "sha256" not in component
                for record in inventory["records"]
                for components in record["components"].values()
                for component in components
            )
        )

    def test_write_refuses_to_modify_raw_tree(self) -> None:
        self._write_triplet("20260101_01_delay_black-1_test_6dpf")
        with self.assertRaises(ConfigurationError):
            write_recording_inventory(self.raw, self.raw / "inventory.json")

        output = self.root / "Metadata" / "recording_inventory.json"
        result = write_recording_inventory(self.raw, output)
        self.assertEqual(result, output.resolve())
        self.assertEqual(json.loads(output.read_text())["record_count"], 1)
        with self.assertRaises(FileExistsError):
            write_recording_inventory(self.raw, output)
        self.assertEqual(
            write_recording_inventory(self.raw, output, overwrite=True),
            output.resolve(),
        )

    def test_cli_prints_inventory_without_writing_output(self) -> None:
        self._write_triplet("20260101_01_delay_black-1_test_6dpf")
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            main(["inventory", "--input-dir", str(self.raw), "--skip-hashes"])
        payload = json.loads(capture.getvalue())
        self.assertEqual(payload["status_counts"]["COMPLETE"], 1)
        self.assertFalse(payload["source_hashes_included"])
        self.assertEqual(
            sorted(path.name for path in self.raw.iterdir()),
            sorted(
                [
                    "20260101_01_delay_black-1_test_6dpf_cam.txt",
                    "20260101_01_delay_black-1_test_6dpf_mp tail tracking.txt",
                    "20260101_01_delay_black-1_test_6dpf_stim control.txt",
                ]
            ),
        )

    def test_inspect_tracking_headers_summarizes_point_counts(self) -> None:
        name = "20260101_01_delay_black-1_test_6dpf"
        self._write_component(name, "_cam.txt", "FrameID ElapsedTime AbsoluteTime\n1 0 1\n")
        self._write_component(
            name,
            "_mp tail tracking.txt",
            "FrameID x0 y0 angle0 x1 y1 angle1\n1 0 0 0.1 1 0 0.2\n",
        )
        self._write_component(name, "_stim control.txt", "Type Beg End\nCycle 1 2\n")
        inventory = build_recording_inventory(
            self.raw,
            hash_files=False,
            inspect_tracking_headers=True,
        )
        self.assertTrue(inventory["tracking_headers_inspected"])
        self.assertTrue(inventory["tracking_schema_summary"]["stable_point_count"])
        self.assertEqual(
            inventory["tracking_schema_summary"]["point_count_histogram"],
            {"2": 1},
        )
        self.assertEqual(inventory["records"][0]["tracking_schema"]["point_count"], 2)

    def test_parses_condition_and_writes_under_nested_paper_data(self) -> None:
        name = "20230227_01_fixedTrace_orange-1_test_6dpf"
        self._write_triplet(name)
        paper = self.raw / "Paper data"
        (paper / "Processed data").mkdir(parents=True)
        (paper / "Processed data" / "decoy_cam.txt").write_text("nope\n", encoding="utf-8")
        inventory = build_recording_inventory(self.raw, hash_files=False)
        self.assertEqual(inventory["records"][0]["condition_id"], "fixedtrace")
        self.assertEqual(inventory["status_counts"]["COMPLETE"], 1)
        output = paper / "Metadata" / "recording_inventory.json"
        write_recording_inventory(self.raw, output, hash_files=False)
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
