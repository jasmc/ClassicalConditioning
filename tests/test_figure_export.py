from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.export import (
    PROVENANCE_NAMESPACE,
    FigureMode,
    FigureProvenance,
    _validate_svg_registry,
    export_matplotlib_figure,
)


class FigureExportTests(unittest.TestCase):
    def test_publication_export_has_semantic_ids_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figure, axis = plt.subplots()
            line = axis.plot([0, 1], [1, 2], label="condition")[0]
            line.set_gid("series__a__condition")
            axis.set_xlabel("Time")
            axis.set_ylabel("Activity")
            provenance = FigureProvenance(
                figure_id="test-figure",
                analysis_recipe="test-recipe",
                source_file=str(Path(__file__).resolve()),
                source_symbol="test_publication_export_has_semantic_ids_and_provenance",
                source_hash=sha256_file(Path(__file__).resolve()),
                reproduction_snippet="build_test_figure()",
                input_artifacts=({"path": "input.parquet", "sha256": "abc"},),
                artist_mappings={
                    "series__a__condition": {
                        "value_field": "activity_rad_per_ms",
                        "x_field": "time_s",
                        "units": "rad/ms",
                    }
                },
            )
            result = export_matplotlib_figure(
                figure,
                root / "figure",
                provenance,
                mode=FigureMode.PUBLICATION,
                panel_ids=["A"],
                allow_dirty_publication=True,
            )
            plt.close(figure)

            svg = result.outputs[0]
            tree = ET.parse(svg)
            ids = {
                element.attrib["id"]
                for element in tree.getroot().iter()
                if "id" in element.attrib
            }
            metadata = tree.getroot().find(
                f".//{{{PROVENANCE_NAMESPACE}}}analysis-provenance"
            )
            sidecar = json.loads(result.sidecar.read_text(encoding="utf-8"))
            sidecar_digest = sha256_file(result.sidecar)
            ET.SubElement(tree.getroot(), "{http://www.w3.org/2000/svg}script").text = "alert(1)"
            tree.write(svg, encoding="utf-8", xml_declaration=True)
            with self.assertRaisesRegex(ValueError, "active script"):
                _validate_svg_registry(svg, sidecar["artist_registry"])

        self.assertIn("axes__a__main", ids)
        self.assertIn("axis-title__a__x", ids)
        self.assertTrue(any(value.startswith("tick-label__a__x__") for value in ids))
        self.assertIn("series__a__condition", ids)
        self.assertIsNotNone(metadata)
        assert metadata is not None
        embedded = json.loads(metadata.text)
        self.assertEqual(embedded["reproduction_snippet"], "build_test_figure()")
        self.assertEqual(embedded["sidecar_sha256"], sidecar_digest)
        self.assertIn("axes__a__main", sidecar["artist_registry"])
        self.assertEqual(
            sidecar["artist_registry"]["series__a__condition"]["units"], "rad/ms"
        )

    def test_publication_export_rejects_dirty_worktree_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figure, _ = plt.subplots()
            provenance = FigureProvenance(
                figure_id="dirty-test",
                analysis_recipe="test",
                source_file=str(Path(__file__).resolve()),
                source_symbol="dirty",
                source_hash=sha256_file(Path(__file__).resolve()),
                reproduction_snippet="dirty()",
                input_artifacts=(),
            )
            with patch(
                "classical_conditioning.figures.export._git_dirty",
                return_value=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "clean Git worktree"):
                    export_matplotlib_figure(
                        figure,
                        Path(directory) / "figure",
                        provenance,
                        mode=FigureMode.PUBLICATION,
                        panel_ids=["A"],
                    )
            plt.close(figure)

    def test_static_export_writes_png_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            figure, _ = plt.subplots()
            provenance = FigureProvenance(
                figure_id="test-static",
                analysis_recipe="test",
                source_file=str(Path(__file__).resolve()),
                source_symbol="test_static_export_writes_png_and_sidecar",
                source_hash=sha256_file(Path(__file__).resolve()),
                reproduction_snippet="build_static()",
                input_artifacts=(),
            )
            result = export_matplotlib_figure(
                figure,
                root / "figure",
                provenance,
                mode=FigureMode.STATIC,
                panel_ids=["A"],
            )
            plt.close(figure)
            self.assertEqual(result.outputs[0].suffix, ".png")
            self.assertTrue(result.outputs[0].is_file())
            self.assertTrue(result.sidecar.is_file())


if __name__ == "__main__":
    unittest.main()
