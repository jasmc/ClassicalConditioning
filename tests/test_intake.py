from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.artifacts import (
    publish_transaction,
)
from classical_conditioning.intake import (
    discover_recording,
    inspect_table_structure,
    intake_recording,
    intake_recordings,
)


RECORDING = "20260101_01_delay_black-1_test_6dpf"


class IntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "Raw"
        self.project = self.root / "Paper data"
        self.raw.mkdir()

        (self.raw / f"{RECORDING}_cam.txt").write_text(
            "FrameID ElapsedTime AbsoluteTime\n"
            "10 0.0 1000\n"
            "11 1.5 1002\n"
            "12 3.0 1003\n",
            encoding="utf-8",
        )
        (self.raw / f"{RECORDING}_mp tail tracking.txt").write_text(
            "FrameID x0 y0 x1 y1 angle0 angle1\n"
            "9 1 2 3.5 4.5 0.1 0.2\n"
            "10 1 2 3.6 4.6 0.2 0.3\n"
            "11 1 2 3.7 4.7 0.3 0.4\n",
            encoding="utf-8",
        )
        (self.raw / f"{RECORDING}_stim control.txt").write_text(
            "Type Beg End\n"
            "Cycle 1000 1100\n"
            "Reinforcer 1200 1250\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_discovers_exact_triplet(self) -> None:
        sources = discover_recording(self.raw)
        self.assertEqual(sources.recording_id, "20260101_01")
        self.assertEqual(sources.recording_name, RECORDING)

    def test_preview_returns_structure_without_values(self) -> None:
        path = self.raw / f"{RECORDING}_cam.txt"
        preview = inspect_table_structure(path, rows=2)
        self.assertEqual(preview["preview_rows"], 2)
        self.assertEqual(
            preview["columns"],
            ["FrameID", "ElapsedTime", "AbsoluteTime"],
        )
        self.assertNotIn("values", preview)

    def test_rejects_incomplete_triplet(self) -> None:
        (self.raw / f"{RECORDING}_stim control.txt").unlink()
        with self.assertRaisesRegex(ValueError, "exactly one complete"):
            discover_recording(self.raw)

    def test_rejects_tracking_without_matching_tail_fields(self) -> None:
        tracking = self.raw / f"{RECORDING}_mp tail tracking.txt"
        tracking.write_text(
            "FrameID x0 y0\n"
            "9 1 2\n"
            "10 1 2\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "matching, contiguous xN, yN"):
            intake_recording(
                self.raw,
                self.project,
                chunk_rows=2,
                preview_rows=2,
            )

    def test_rejects_noncontiguous_or_unknown_tracking_fields(self) -> None:
        tracking = self.raw / f"{RECORDING}_mp tail tracking.txt"
        tracking.write_text(
            "FrameID x0 y0 angle0 x2 y2 angle2 confidence\n"
            "9 1 2 0.1 3 4 0.2 1\n"
            "10 1 2 0.1 3 4 0.2 1\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "matching, contiguous xN, yN"):
            intake_recording(
                self.raw,
                self.project,
                chunk_rows=2,
                preview_rows=2,
            )
        self.assertFalse((self.project / "Processed data").exists())

    def test_intake_writes_lossless_parquet_and_preserves_sources(self) -> None:
        source_state = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.raw.iterdir()
        }
        result = intake_recording(
            self.raw,
            self.project,
            chunk_rows=2,
            preview_rows=2,
        )

        processed = self.project / "Processed data" / "20260101_01"
        camera = pq.read_table(processed / "camera.parquet").to_pandas()
        tracking = pq.read_table(processed / "tracking.parquet").to_pandas()
        protocol = pq.read_table(processed / "stimulus_events.parquet").to_pandas()

        expected_camera = pd.read_csv(
            self.raw / f"{RECORDING}_cam.txt",
            sep=r"\s+",
            dtype={
                "FrameID": "int64",
                "ElapsedTime": "float64",
                "AbsoluteTime": "int64",
            },
        )
        expected_tracking = pd.read_csv(
            self.raw / f"{RECORDING}_mp tail tracking.txt",
            sep=r"\s+",
            dtype={
                "FrameID": "int64",
                "x0": "float64",
                "y0": "float64",
                "x1": "float64",
                "y1": "float64",
                "angle0": "float64",
                "angle1": "float64",
            },
        )
        expected_protocol = pd.read_csv(
            self.raw / f"{RECORDING}_stim control.txt",
            sep=r"\s+",
            dtype={"Beg": "int64", "End": "int64"},
        )
        pd.testing.assert_frame_equal(camera, expected_camera)
        pd.testing.assert_frame_equal(tracking, expected_tracking)
        pd.testing.assert_frame_equal(protocol, expected_protocol)

        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["camera"]["statistics"]["frames"]["row_count"], 3)
        self.assertEqual(summary["tracking"]["statistics"]["frames"]["row_count"], 3)
        self.assertTrue(
            summary["camera"]["artifact"]["compression_lossless"]
        )

        for path in self.raw.iterdir():
            content, modified = source_state[path.name]
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_flags_camera_timestamp_regression(self) -> None:
        camera = self.raw / f"{RECORDING}_cam.txt"
        camera.write_text(
            "FrameID ElapsedTime AbsoluteTime\n"
            "10 0.0 1000\n"
            "11 1.5 1002\n"
            "12 1.0 1001\n",
            encoding="utf-8",
        )
        result = intake_recording(
            self.raw,
            self.project,
            chunk_rows=2,
            preview_rows=2,
        )
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "REVIEW")
        self.assertEqual(
            summary["camera"]["statistics"]["ordered_sequences"]["ElapsedTime"][
                "reverse_steps"
            ],
            1,
        )
        self.assertEqual(
            summary["camera"]["statistics"]["ordered_sequences"]["AbsoluteTime"][
                "reverse_steps"
            ],
            1,
        )

    def test_disjoint_stream_ranges_report_actual_nonoverlap_span(self) -> None:
        tracking = self.raw / f"{RECORDING}_mp tail tracking.txt"
        tracking.write_text(
            "FrameID x0 y0 x1 y1 angle0 angle1\n"
            "20 1 2 3.5 4.5 0.1 0.2\n"
            "21 1 2 3.6 4.6 0.2 0.3\n"
            "22 1 2 3.7 4.7 0.3 0.4\n",
            encoding="utf-8",
        )
        result = intake_recording(
            self.raw,
            self.project,
            chunk_rows=2,
            preview_rows=2,
        )
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["alignment"]["overlap_frame_span"], 0)
        self.assertEqual(summary["alignment"]["stream_nonoverlap_frame_span"], 6)

    def test_reversed_frame_ids_never_produce_negative_range_counts(self) -> None:
        camera = self.raw / f"{RECORDING}_cam.txt"
        camera.write_text(
            "FrameID ElapsedTime AbsoluteTime\n"
            "10 0.0 1000\n"
            "11 1.5 1002\n"
            "5 3.0 1003\n",
            encoding="utf-8",
        )
        result = intake_recording(
            self.raw,
            self.project,
            chunk_rows=2,
            preview_rows=2,
        )
        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "REVIEW")
        self.assertGreaterEqual(
            summary["alignment"]["stream_nonoverlap_frame_span"],
            0,
        )
        self.assertEqual(
            summary["camera"]["statistics"]["frames"]["reverse_events"],
            1,
        )

    def test_refuses_to_overwrite_derived_artifacts(self) -> None:
        intake_recording(self.raw, self.project, chunk_rows=2, preview_rows=2)
        with self.assertRaises(FileExistsError):
            intake_recording(self.raw, self.project, chunk_rows=2, preview_rows=2)

    def test_refuses_to_overwrite_report_even_if_tables_are_absent(self) -> None:
        report = (
            self.project
            / "Quality checks"
            / "20260101_01"
            / "acquisition_report.html"
        )
        report.parent.mkdir(parents=True)
        report.write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "derived intake artifacts"):
            intake_recording(self.raw, self.project, chunk_rows=2, preview_rows=2)
        self.assertEqual(report.read_text(encoding="utf-8"), "existing")

    def test_rejects_raw_directory_as_project_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be the raw input"):
            intake_recording(self.raw, self.raw, chunk_rows=2, preview_rows=2)

    def test_publish_transaction_rolls_back_existing_artifacts(self) -> None:
        staging = self.root / "staging"
        backup_source = staging / "Processed data"
        backup_source.mkdir(parents=True)
        (backup_source / "new.txt").write_text("new", encoding="utf-8")
        staged_metadata = staging / "Metadata" / "manifest.json"
        staged_metadata.parent.mkdir()
        staged_metadata.write_text("new metadata", encoding="utf-8")

        final_data = self.root / "final" / "Processed data"
        final_data.mkdir(parents=True)
        (final_data / "old.txt").write_text("old", encoding="utf-8")
        final_metadata = self.root / "final" / "Metadata" / "manifest.json"
        final_metadata.parent.mkdir()
        final_metadata.write_text("old metadata", encoding="utf-8")

        real_replace = __import__("os").replace

        def fail_metadata_publish(source: Path, destination: Path) -> None:
            if Path(source) == staged_metadata:
                raise OSError("simulated publish failure")
            real_replace(source, destination)

        with patch(
            "classical_conditioning.artifacts.os.replace",
            side_effect=fail_metadata_publish,
        ):
            with self.assertRaisesRegex(OSError, "simulated publish failure"):
                publish_transaction(
                    (
                        (backup_source, final_data),
                        (staged_metadata, final_metadata),
                    ),
                    staging,
                    overwrite=True,
                )

        self.assertEqual((final_data / "old.txt").read_text(), "old")
        self.assertEqual(final_metadata.read_text(), "old metadata")
        self.assertFalse((final_data / "new.txt").exists())

    def test_allows_project_directory_containing_raw_directory(self) -> None:
        nested_root = self.root / "Contained Paper data"
        nested_raw = nested_root / "Raw single fish data"
        nested_raw.parent.mkdir()
        self.raw.rename(nested_raw)
        result = intake_recording(
            nested_raw,
            nested_root,
            chunk_rows=2,
            preview_rows=2,
        )
        self.assertTrue(result.processed_dir.is_dir())

    def test_selects_one_recording_from_a_shared_raw_folder(self) -> None:
        other = "20260101_02_control_black-1_test_6dpf"
        (self.raw / f"{other}_cam.txt").write_text(
            "FrameID ElapsedTime AbsoluteTime\n10 0.0 1000\n",
            encoding="utf-8",
        )
        (self.raw / f"{other}_mp tail tracking.txt").write_text(
            "FrameID x0 y0 x1 y1 angle0 angle1\n10 1 2 3.5 4.5 0.1 0.2\n",
            encoding="utf-8",
        )
        (self.raw / f"{other}_stim control.txt").write_text(
            "Type Beg End\nCycle 1000 1100\n",
            encoding="utf-8",
        )
        sources = discover_recording(self.raw, recording_id="20260101_02")
        self.assertEqual(sources.recording_id, "20260101_02")

    def test_intake_batch_keeps_named_conditions_and_nested_paper_data(self) -> None:
        other = "20260101_02_control_black-1_test_6dpf"
        (self.raw / f"{other}_cam.txt").write_text(
            "FrameID ElapsedTime AbsoluteTime\n10 0.0 1000\n11 1.5 1002\n",
            encoding="utf-8",
        )
        (self.raw / f"{other}_mp tail tracking.txt").write_text(
            "FrameID x0 y0 x1 y1 angle0 angle1\n"
            "9 1 2 3.5 4.5 0.1 0.2\n"
            "10 1 2 3.6 4.6 0.2 0.3\n",
            encoding="utf-8",
        )
        (self.raw / f"{other}_stim control.txt").write_text(
            "Type Beg End\nCycle 1000 1100\nReinforcer 1200 1250\n",
            encoding="utf-8",
        )
        project = self.raw / "Paper data"
        result = intake_recordings(
            self.raw,
            project,
            keep_conditions=("control",),
            chunk_rows=2,
            preview_rows=2,
        )
        self.assertEqual(result.recording_ids, ("20260101_02",))
        self.assertEqual(result.completed, ("20260101_02",))
        self.assertTrue(
            (project / "Metadata" / "20260101_02_source_manifest.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
