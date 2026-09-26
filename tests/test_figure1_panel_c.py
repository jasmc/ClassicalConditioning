"""Check the Figure 1C SVG against the declared protocol counts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

SVG = {"s": "http://www.w3.org/2000/svg"}
REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_figure1_panel_c.py"


class Figure1PanelCTests(unittest.TestCase):
    def test_phase_counts_and_pulse_roles_survive_vector_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "C.svg"
            subprocess.run([sys.executable, str(BUILDER), "--output", str(target)],
                           check=True, capture_output=True)
            root = ET.parse(target).getroot()
            groups = {item.get("id"): item for item in root.findall("s:g", SVG)}
            self.assertEqual(groups["phase-priming"].get("data-cs-count"), "4")
            self.assertEqual(groups["phase-priming"].get("short-us-count"), "15")
            self.assertEqual(groups["phase-pretraining"].get("data-cs-count"), "10")
            self.assertEqual(groups["phase-pretraining"].get("short-us-count"), "3")
            self.assertEqual(groups["phase-training"].get("data-cs-count"), "50")
            self.assertEqual(groups["phase-training"].get("paired-long-us-count"), "46")
            self.assertEqual(groups["phase-training"].get("catch-cs-count"), "4")
            self.assertEqual(groups["phase-testing"].get("data-cs-count"), "30")
            self.assertEqual(groups["phase-testing"].get("short-us-count"), "14")
            self.assertNotIn("final-check", groups)
            self.assertNotIn("viability", " ".join(root.itertext()).lower())
            self.assertEqual(root.get("font-family"), "DejaVu Sans")
            self.assertFalse(root.findall(".//s:image", SVG))

    def test_historical_v1_retains_final_check_for_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "C-v1.svg"
            subprocess.run([sys.executable, str(BUILDER), "--variant", "v1",
                            "--output", str(target)], check=True, capture_output=True)
            root = ET.parse(target).getroot()
            final = root.find("s:g[@id='final-check']", SVG)
            self.assertEqual(final.get("data-us-duration-ms"), "500")

    def test_invalid_training_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = REPO_ROOT / "configs" / "paper-figures" / "figure1-session-protocol.json"
            data = json.loads(source.read_text(encoding="utf-8"))
            data["phases"][2]["catch_cs_count"] = 3
            spec = Path(directory) / "invalid.json"
            spec.write_text(json.dumps(data), encoding="utf-8")
            result = subprocess.run([sys.executable, str(BUILDER), "--spec", str(spec),
                                     "--output", str(Path(directory) / "C.svg")],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
