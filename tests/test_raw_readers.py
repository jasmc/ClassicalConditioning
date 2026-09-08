from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from classical_conditioning.cli import main
from classical_conditioning.exceptions import SchemaValidationError
from classical_conditioning.ingestion.readers import (
    read_camera,
    read_protocol,
    read_tracking,
)
from classical_conditioning.ingestion.frame_sequence import validate_frame_sequence
from classical_conditioning.ingestion.validate_raw import validate_raw_triplet


RECORDING = "20260101_01_delay_black-1_test_6dpf"


class RawReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "Raw"
        self.raw.mkdir()
        self.camera = self.raw / f"{RECORDING}_cam.txt"
        self.tracking = self.raw / f"{RECORDING}_mp tail tracking.txt"
        self.protocol = self.raw / f"{RECORDING}_stim control.txt"
        self._write_valid_triplet()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_valid_triplet(self) -> None:
        self.camera.write_text(
            "FrameID ElapsedTime AbsoluteTime\n"
            "10 0.0 1000\n"
            "11 1.5 1002\n"
            "12 3.0 1003\n"
            "13 4.5 1004\n",
            encoding="utf-8",
        )
        self.tracking.write_text(
            "FrameID x0 y0 angle0 x1 y1 angle1\n"
            "10 0 0 0.0 1 0 0.1\n"
            "11 0 0 0.0 1 0.1 0.2\n"
            "12 0 0 0.0 1 0.2 0.3\n"
            "summary 0 0 0 0 0 0\n",
            encoding="utf-8",
        )
        self.protocol.write_text(
            "Type Beg End\n"
            "Cycle 1000 1100\n"
            "Reinforcer 1200 1250\n",
            encoding="utf-8",
        )

    def test_read_camera_accepts_legacy_aliases(self) -> None:
        self.camera.write_text(
            "ID TotalTime AbsoluteTime\n"
            "1 0.0 100\n"
            "2 1.0 101\n",
            encoding="utf-8",
        )
        result = read_camera(self.camera)
        self.assertEqual(list(result.frame.columns), ["FrameID", "ElapsedTime", "AbsoluteTime"])
        self.assertEqual(int(result.frame.iloc[0]["FrameID"]), 1)

    def test_read_camera_rejects_missing_columns_and_duplicates(self) -> None:
        self.camera.write_text(
            "FrameID ElapsedTime\n"
            "1 0.0\n",
            encoding="utf-8",
        )
        with self.assertRaises(SchemaValidationError):
            read_camera(self.camera)

        self.camera.write_text(
            "FrameID ElapsedTime AbsoluteTime\n"
            "1 0.0 100\n"
            "1 1.0 101\n",
            encoding="utf-8",
        )
        with self.assertRaises(SchemaValidationError):
            read_camera(self.camera)

    def test_read_tracking_full_and_legacy_angle_modes(self) -> None:
        full = read_tracking(self.tracking, mode="full")
        self.assertEqual(full.schema.point_count, 2)
        self.assertTrue(full.dropped_trailing_summary_row)
        self.assertEqual(len(full.frame), 3)
        self.assertIn("x0", full.frame.columns)
        self.assertIn("angle1", full.frame.columns)

        legacy = read_tracking(
            self.tracking,
            mode="legacy_angles",
            convert_angles_to_degrees=True,
            legacy_angle_point_count=2,
        )
        self.assertTrue(legacy.angles_converted_to_degrees)
        self.assertEqual(
            list(legacy.frame.columns),
            ["FrameID", "Angle of point 0 (deg)", "Angle of point 1 (deg)"],
        )
        expected = 0.1 * 180.0 / math.pi
        self.assertAlmostEqual(
            float(legacy.frame.iloc[0]["Angle of point 1 (deg)"]),
            expected,
            places=6,
        )

    def test_read_tracking_rejects_empty_and_bad_schema(self) -> None:
        self.tracking.write_text("FrameID\n", encoding="utf-8")
        with self.assertRaises(SchemaValidationError):
            read_tracking(self.tracking)

        self.tracking.write_text(
            "FrameID x0 y0 angle0\n"
            "1 0 0 0.1\n"
            "2 0 0 0.2\n",
            encoding="utf-8",
        )
        with self.assertRaises(SchemaValidationError):
            read_tracking(self.tracking)

        self.tracking.write_text(
            "FrameID x0 y0 angle0 x1 y1 angle1 Extra\n"
            "1 0 0 0.1 1 0 0.2 9\n"
            "2 0 0 0.1 1 0 0.2 9\n",
            encoding="utf-8",
        )
        with self.assertRaises(SchemaValidationError):
            read_tracking(self.tracking)

    def test_read_protocol_and_frame_sequence_diagnostics(self) -> None:
        protocol = read_protocol(self.protocol)
        self.assertEqual(protocol.invalid_duration_count, 0)
        self.assertEqual(protocol.event_type_counts["Cycle"], 1)

        self.protocol.write_text(
            "Type Beg End\n"
            "Cycle 1100 1000\n",
            encoding="utf-8",
        )
        bad = read_protocol(self.protocol)
        self.assertEqual(bad.invalid_duration_count, 1)

        camera = read_camera(self.camera).frame
        report = validate_frame_sequence(camera)
        self.assertEqual(report.gap_event_count, 0)
        self.assertEqual(report.duplicate_frame_id_count, 0)

        gappy = camera.copy()
        gappy.loc[2, "FrameID"] = 20
        gap_report = validate_frame_sequence(gappy)
        self.assertGreater(gap_report.gap_event_count, 0)
        self.assertGreater(gap_report.missing_frame_count, 0)

    def test_validate_raw_triplet_cli(self) -> None:
        output = self.root / "raw_validation.json"
        result = validate_raw_triplet(self.raw, output=output, overwrite=True)
        self.assertEqual(result.summary["status"], "PASS")
        self.assertEqual(result.summary["tracking"]["point_count"], 2)
        self.assertTrue(output.is_file())

        main(
            [
                "validate-raw",
                "--input-dir",
                str(self.raw),
                "--output",
                str(output),
                "--overwrite",
            ]
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["artifact_kind"], "raw-acquisition-validation-v1")


if __name__ == "__main__":
    unittest.main()
