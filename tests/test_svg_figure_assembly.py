"""Checks for the reusable SVG composition boundary."""

from __future__ import annotations

import json
import base64
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.assemble_svg_figure import panel_layout, render


class SvgFigureAssemblyTests(unittest.TestCase):
    def test_one_panel_update_rebuilds_only_its_embedded_artwork(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / "a.svg"
            b = root / "b.svg"
            a.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><style>.mark{fill:red}</style><rect id="shape" class="mark" width="10" height="10"/></svg>')
            b.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><style>.mark{fill:blue}</style><rect id="shape" class="mark" width="10" height="10"/></svg>')
            layout = root / "layout.json"
            layout.write_text(json.dumps({
                "canvas": [30, 10],
                "panels": [
                    {"id": "A", "source": "a.svg", "box": [0, 0, 10, 10]},
                    {"id": "B", "source": "b.svg", "box": [20, 0, 10, 10]},
                ],
            }))
            output = root / "figure.svg"
            first = render(layout, output)
            original_b = first["panels"][1]["sha256"]
            svg = output.read_text()
            self.assertIn("src-A-mark", svg)
            self.assertIn("src-B-mark", svg)
            self.assertIn('id="src-A-shape"', svg)
            self.assertIn('id="src-B-shape"', svg)
            a.write_text(a.read_text().replace("fill:red", "fill:green"))
            second = render(layout, output)
            self.assertNotEqual(first["panels"][0]["sha256"], second["panels"][0]["sha256"])
            self.assertEqual(original_b, second["panels"][1]["sha256"])
            self.assertIn("fill:green", output.read_text())
            ET.parse(output)

    def test_missing_panel_is_visible_in_preview_and_rejected_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = root / "layout.json"
            layout.write_text(json.dumps({"canvas": [20, 20], "panels": [
                {"id": "D", "box": [0, 0, 20, 20]}
            ]}))
            output = root / "figure.svg"
            result = render(layout, output)
            self.assertEqual(result["panels"][0]["status"], "placeholder")
            with self.assertRaises(FileNotFoundError):
                render(layout, output, strict=True)

    def test_generated_png_is_embedded_without_external_file_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            png = root / "panel.png"
            png.write_bytes(base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lXcAAAAASUVORK5CYII="
            ))
            layout = root / "layout.json"
            layout.write_text(json.dumps({"canvas": [20, 20], "panels": [
                {"id": "D", "box": [0, 0, 20, 20], "source": "panel.png"}
            ]}))
            output = root / "figure.svg"
            result = render(layout, output)
            self.assertEqual(result["panels"][0]["source_kind"], "raster")
            self.assertIn("data:image/png;base64,", output.read_text())
            self.assertNotIn('href="panel.png"', output.read_text())

    def test_source_fonts_and_positioned_text_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "panel.svg"
            source.write_text('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20">
                <style>.label{font-family:Arial-BoldMT, Arial;font-weight:700}</style>
                <text class="label" x="0" y="10"><tspan x="0">Pre-</tspan><tspan x="20">Train</tspan></text>
            </svg>''')
            layout = root / "layout.json"
            layout.write_text(json.dumps({
                "canvas": [100, 20], "font_family": "DejaVu Sans",
                "panels": [{"id": "C", "source": "panel.svg", "box": [0, 0, 100, 20],
                            "flatten_text": ["Pre-Train"]}],
            }))
            output = root / "figure.svg"
            render(layout, output)
            svg = output.read_text()
            self.assertNotIn("Arial", svg)
            self.assertIn("DejaVu Sans", svg)
            self.assertIn("Pre-Train</text>", svg)
            self.assertNotIn("<tspan", svg)

    def test_standalone_panel_uses_ssd_source_and_local_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = root / "ssd"
            storage.mkdir()
            (storage / "scheme.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                '<circle cx="5" cy="5" r="4"/></svg>'
            )
            layout_data = {
                "canvas": [100, 100], "storage_root": str(storage),
                "panels": [{"id": "A", "source": "scheme.svg", "box": [20, 30, 40, 50],
                            "content_box": [25, 35, 30, 40],
                            "label": {"text": "A", "x": 22, "y": 44}}],
            }
            layout = root / "layout.json"
            layout.write_text(json.dumps(layout_data))
            standalone = panel_layout(layout_data, "A")
            self.assertEqual(standalone["canvas"], [40, 50])
            self.assertEqual(standalone["panels"][0]["content_box"], [5, 5, 30, 40])
            output = root / "panel.svg"
            result = render(layout, output, only_panel="A")
            self.assertEqual(result["panels"][0]["source"], str(storage / "scheme.svg"))
            self.assertEqual(result["panels"][0]["status"], "source")

    def test_alternate_source_preview_preserves_selected_layout_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "selected.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                '<circle id="selected" cx="5" cy="5" r="3"/></svg>'
            )
            (root / "exploratory.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                '<rect id="exploratory" width="10" height="10"/></svg>'
            )
            layout = root / "layout.json"
            layout.write_text(json.dumps({"canvas": [10, 10], "panels": [
                {"id": "B", "source": "selected.svg", "box": [0, 0, 10, 10],
                 "view_box": [0, 0, 5, 5]}
            ]}))
            output = root / "preview.svg"
            sidecar = render(layout, output,
                             source_overrides={"B": "exploratory.svg"})
            self.assertEqual(sidecar["panels"][0]["source"], str(root / "exploratory.svg"))
            self.assertIn('id="src-B-exploratory"', output.read_text())
            nested = ET.parse(output).getroot().find(
                ".//{http://www.w3.org/2000/svg}svg"
            )
            self.assertEqual(nested.get("viewBox"), "0 0 10 10")
            self.assertEqual(json.loads(layout.read_text())["panels"][0]["source"],
                             "selected.svg")


if __name__ == "__main__":
    unittest.main()
