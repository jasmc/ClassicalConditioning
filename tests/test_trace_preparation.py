from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path


PREPARE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "prepare-trace-runs.py")
)["prepare"]


def write_triplet(root: Path, stem: str) -> None:
    (root / f"{stem}_cam.txt").write_text("camera\n")
    (root / f"{stem}_mp tail tracking.txt").write_text("tracking\n")
    events = []
    for index in range(94):
        start = 1_000_000 + index * 100_000
        events.append((start, f"Cycle {start} {start + 10_000}"))
        if (index < 79 and index not in {24, 38, 52, 58}) or index in {79, 80, 81, 82}:
            us = start + 13_000
            events.append((us, f"Reinforcer {us} {us + 50}"))
    lines = ["Type Beg End"] + [event for _, event in sorted(events)]
    (root / f"{stem}_stim control.txt").write_text("\n".join(lines) + "\n")


class TracePreparationTests(unittest.TestCase):
    def test_includes_all_complete_fish_then_blocks_incomplete_addition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, digested, configs = (root / name for name in ("raw", "digested", "configs"))
            raw.mkdir()
            digested.mkdir()
            write_triplet(raw, "20230301_01_trace_blue")
            write_triplet(raw, "20230301_02_control_black")

            ready = PREPARE(raw, digested, configs)
            self.assertEqual(ready["status"], "ready")
            config_path = configs / "all3sTrace-full-windows.json"
            config = json.loads(config_path.read_text())
            self.assertEqual(config["recording_ids"], ["20230301_01", "20230301_02"])
            self.assertFalse(config["continue_on_error"])
            self.assertEqual(config["analysis_id"], "all3sTrace-full")
            self.assertEqual(config["assessment_metric"], "tail_length_weighted_angular_l1")
            self.assertNotIn("run_inventory", config)

            (raw / "20230301_03_trace_green_cam.txt").write_text("camera\n")
            with self.assertRaisesRegex(RuntimeError, "20230301_03"):
                PREPARE(raw, digested, configs)
            self.assertFalse(config_path.exists())
            report = json.loads((configs / "trace-preflight.json").read_text())
            self.assertEqual(report["status"], "blocked")

            partial = PREPARE(raw, digested, configs, allow_incomplete=True)
            self.assertEqual(partial["status"], "ready_partial")
            self.assertEqual(partial["selected_recording_ids"], ["20230301_01", "20230301_02"])
            self.assertEqual(partial["incomplete"][0]["recording_id"], "20230301_03")
            self.assertEqual(json.loads(config_path.read_text())["recording_ids"],
                             ["20230301_01", "20230301_02"])


if __name__ == "__main__":
    unittest.main()
