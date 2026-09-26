"""Scientific geometry checks for the exploratory vector condition inset."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


SVG = {"s": "http://www.w3.org/2000/svg"}
REPO_ROOT = Path(__file__).resolve().parents[1]


class Figure1PanelBTests(unittest.TestCase):
    def test_stimulus_geometry_matches_active_protocol_and_control_is_unpaired(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "B.svg"
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_figure1_panel_b.py"),
                            "--variant", "v1", "--output", str(target)],
                           check=True, capture_output=True)
            root = ET.parse(target).getroot()
            groups = {element.get("id"): element for element in root.findall("s:g", SVG)}
            cs_bars = []
            for key in ("control", "delay", "trace-3s", "trace-10s"):
                bar = groups[f"row-{key}"].find(f"s:rect[@id='cs-{key}']", SVG)
                self.assertIsNotNone(bar)
                cs_bars.append((bar.get("x"), bar.get("width"), bar.get("data-end-s")))
            self.assertEqual(len(set(cs_bars)), 1)
            self.assertEqual(cs_bars[0][2], "10")
            self.assertIsNone(groups["row-control"].find("s:line[@data-time-s]", SVG))
            scale = float(cs_bars[0][1]) / 10
            zero = float(cs_bars[0][0])
            for key, expected in (("delay", 9), ("trace-3s", 13), ("trace-10s", 20)):
                marker = groups[f"row-{key}"].find("s:line[@data-time-s]", SVG)
                self.assertIsNotNone(marker)
                self.assertEqual(float(marker.get("data-time-s")), expected)
                self.assertAlmostEqual(float(marker.get("x1")), zero + expected * scale)
            self.assertEqual(root.get("font-family"), "DejaVu Sans")
            self.assertFalse(root.findall(".//s:image", SVG))

    def test_v2_control_shows_alternative_onsets_across_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "B-v2.svg"
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_figure1_panel_b.py"),
                            "--variant", "v2", "--output", str(target)],
                           check=True, capture_output=True)
            root = ET.parse(target).getroot()
            groups = {element.get("id"): element for element in root.findall("s:g", SVG)}
            control = groups["row-control"]
            examples = control.findall("s:circle[@data-example-time-s]", SVG)
            self.assertEqual(sorted(float(item.get("data-example-time-s")) for item in examples),
                             [-3, 5, 16])
            self.assertTrue(all(item.get("fill") == "#78358c" for item in examples))
            self.assertEqual(sorted(int(item.get("data-example-trial")) for item in examples),
                             [1, 2, 3])
            self.assertEqual(len(control.findall("s:rect[@data-end-s]", SVG)), 3)
            self.assertIsNone(control.find("s:circle[@data-time-s]", SVG))
            for key, time_s in (("delay", 9), ("trace-3s", 13), ("trace-10s", 20)):
                marker = groups[f"row-{key}"].find("s:circle[@data-time-s]", SVG)
                self.assertEqual(float(marker.get("data-time-s")), time_s)
                self.assertEqual(marker.get("fill"), "#78358c")

    def test_v3_control_uses_one_full_size_cs_bar_and_alternative_us_onsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "B-v3.svg"
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_figure1_panel_b.py"),
                            "--variant", "v3", "--output", str(target)],
                           check=True, capture_output=True)
            root = ET.parse(target).getroot()
            groups = {element.get("id"): element for element in root.findall("s:g", SVG)}
            bars = [groups[f"row-{key}"].find(f"s:rect[@id='cs-{key}']", SVG)
                    for key in ("control", "delay", "trace-3s", "trace-10s")]
            self.assertTrue(all(bar is not None for bar in bars))
            self.assertEqual({(bar.get("width"), bar.get("height")) for bar in bars},
                             {("225.0", "16")})
            self.assertEqual(len(groups["row-control"].findall("s:rect[@data-end-s]", SVG)), 1)
            examples = groups["row-control"].findall("s:circle[@data-example-time-s]", SVG)
            self.assertEqual({float(item.get("data-example-time-s")) for item in examples},
                             {-3, 5, 16})
            self.assertEqual({item.get("data-example-trial") for item in examples},
                             {"1", "2", "3"})

    def test_v4_control_has_one_us_per_separate_full_size_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "B-v4.svg"
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_figure1_panel_b.py"),
                            "--variant", "v4", "--output", str(target)],
                           check=True, capture_output=True)
            root = ET.parse(target).getroot()
            control = root.find("s:g[@id='row-control']", SVG)
            trials = [control.find(f"s:g[@id='control-trial-{n}']", SVG)
                      for n in (1, 2, 3)]
            self.assertTrue(all(trial is not None for trial in trials))
            paired = root.find("s:g[@id='row-delay']", SVG)
            paired_bar = paired.find("s:rect[@id='cs-delay']", SVG)
            for trial in trials:
                bars = trial.findall("s:rect[@data-end-s]", SVG)
                dots = trial.findall("s:circle[@data-example-time-s]", SVG)
                self.assertEqual(len(bars), 1)
                self.assertEqual(len(dots), 1)
                self.assertEqual((bars[0].get("width"), bars[0].get("height")),
                                 (paired_bar.get("width"), paired_bar.get("height")))
                self.assertIn("?", " ".join(trial.itertext()))


if __name__ == "__main__":
    unittest.main()
