"""Freeze gate integration tests run independently of analysis dependencies."""
import copy
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shutil
import unittest
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("figure_freeze", ROOT/"src/classical_conditioning/figure_freeze.py")
freeze = importlib.util.module_from_spec(loader)
loader.loader.exec_module(freeze)
SPEC = json.loads(freeze.DEFAULT_SPECIFICATION.read_text(encoding="utf-8"))


class FigureFreezeTests(unittest.TestCase):
    def setUp(self):
        # Inherit workspace ACLs; TemporaryDirectory's Windows sandbox ACL can
        # deny writes even to the process that created it.
        self.folder = ROOT/".cache"/"figure-freeze-tests"/uuid.uuid4().hex
        self.folder.mkdir(parents=True)
        self.addCleanup(self.cleanup_folder)
        self.svg = self.folder/"candidate.svg"
        self.original = self.folder/"original.svg"
        self.data = self.folder/"data.csv"; self.data.write_text("x,y\n0,1\n")
        self.candidate_path = self.folder/"candidate.json"
        self.svg.write_text('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 518.740157480315 100">
<g id="reference"><path d="M 0 50 L 100 50" style="fill:none;stroke:#000000;stroke-width:0.6"/></g>
<g id="trace"><path d="M 0 40 L 100 60" style="fill:none;stroke:#000000;stroke-width:0.6"/></g>
<g id="label"><text x="30" y="90" style="font: 8px 'DejaVu Sans';fill:#000000">Time (s)</text></g>
</svg>''', encoding="utf-8")
        self.original.write_bytes(self.svg.read_bytes())
        def record(element_id, role, artist_type, protection):
            style = SPEC["roles"][role]["style"]
            return {"element_id": element_id, "scientific_role": role, "artist_type": artist_type,
                    "figure_id": "fig1", "panel_id": "a", "subpanel_id": "main",
                    "scientific_context": {"units": "s" if role == "axis.label" else "dimensionless", "measure": "fixture"},
                    "coordinate_system": "data", "geometry": {"dimension": "x"},
                    "style_role": style, "resolved_style": copy.deepcopy(SPEC["styles"][style]),
                    "required_in_svg": True, "classification_confidence": "explicit",
                    "classification_evidence": ["Synthetic source fixture identity"], "protection": protection,
                    "style_verification": {key: {"status": "passed", "evidence": "Fixture source inspection"}
                                           for key in SPEC["styles"][style]}}
        self.candidate = {"figure_id": "fig1", "panel_ids": ["a"], "selection_record": "user-selected fixture",
            "assembly_scale": {"final_width_mm": 183., "source_to_final_transforms": {"root_user_unit_to_final_pt": 1., "svg_transforms": []}},
            "element_registry": {"reference": record("reference", "reference.response_ratio.equal_baseline", "Line2D", "data-geometry"),
                                 "trace": record("trace", "trace.raw_vigor", "Line2D", "data-geometry"),
                                 "label": record("label", "axis.label", "Text", "scientific-text")},
            "approved_exceptions": [], "source_artifacts": [{"path": "original.svg", "sha256": freeze.digest(self.original), "kind": "original_svg"}],
            "data_artifacts": [{"path": "data.csv", "sha256": freeze.digest(self.data)}],
            "exports": [{"path": "candidate.svg", "sha256": freeze.digest(self.svg)}],
            "verification": {name: {"status": "passed", "evidence": "Synthetic fixture review"}
                             for name in ("visual_review", "scientific_mapping_review", "structure_review")},
            "freeze_authorization": "Test user authorized freezing this fixture"}
        self.candidate["element_registry"]["reference"]["geometry"] = {"dimension": "y", "value": 1}

    def cleanup_folder(self):
        if not self.folder.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Test cleanup target must stay inside the workspace")
        shutil.rmtree(self.folder)

    def save(self):
        self.candidate["exports"][0]["sha256"] = freeze.digest(self.svg)
        self.candidate_path.write_text(json.dumps(self.candidate), encoding="utf-8")

    def check(self):
        self.save(); return freeze.check_candidate(self.candidate_path)

    def assert_invalid(self, phrase):
        report = self.check()
        self.assertFalse(report["valid"], report)
        self.assertIn(phrase, json.dumps(report))
        output = self.folder/"frozen.json"
        self.assertFalse(freeze.freeze_candidate(self.candidate_path, output)["valid"])
        self.assertFalse(output.exists())

    def test_valid_freeze_and_no_overwrite(self):
        self.assertTrue(self.check()["valid"])
        output = self.folder/"frozen.json"
        report = freeze.freeze_candidate(self.candidate_path, output)
        self.assertTrue(report["valid"], report)
        frozen = json.loads(output.read_text())
        self.assertEqual(frozen["specification_sha256"], freeze.digest(freeze.DEFAULT_SPECIFICATION))
        self.assertEqual(frozen["exports"][0]["path"], str(self.svg.resolve()))
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            freeze.freeze_candidate(self.candidate_path, output)
        self.assertEqual(output.read_bytes(), before)

    def test_actual_style_checked_not_just_declarations(self):
        self.svg.write_text(self.svg.read_text().replace("stroke-width:0.6", "stroke-width:2.4", 1))
        self.assert_invalid("stroke width")

    def test_approved_exception_only_in_its_scope(self):
        self.svg.write_text(self.svg.read_text().replace("stroke-width:0.6", "stroke-width:0.85", 1))
        self.candidate["approved_exceptions"] = [{"exception_id": "wide-reference", "scientific_role": "reference.response_ratio.equal_baseline", "property": "linewidth_pt", "value": .85,
             "reason": "Author selected thicker reference", "scope": {"figure_id": "fig1", "panel_ids": ["a"]}, "approval_evidence": "Explicit test user approval"}]
        self.candidate["element_registry"]["reference"]["resolved_style"]["linewidth_pt"] = .85
        self.assertTrue(self.check()["valid"])
        self.candidate["approved_exceptions"][0]["scope"]["panel_ids"] = ["b"]
        self.assert_invalid("scope")

    def test_unknown_identity_and_geometry_drift_block(self):
        self.candidate["element_registry"]["trace"]["classification_confidence"] = "unknown"
        self.assert_invalid("unknown")
        self.candidate["element_registry"]["trace"]["classification_confidence"] = "explicit"
        self.svg.write_text(self.svg.read_text().replace("M 0 40 L 100 60", "M 0 40 L 100 70"))
        self.assert_invalid("Protected geometry")

    def test_hash_drift_and_missing_visual_review_block(self):
        self.data.write_text("x,y\n0,2\n")
        self.assert_invalid("hash mismatch")
        self.candidate["data_artifacts"][0]["sha256"] = freeze.digest(self.data)
        self.candidate["verification"] = {"visual_review": {"status": "not-run"}}
        self.assert_invalid("visual review")

    def test_font_and_rgba_opacity_checked(self):
        self.svg.write_text(self.svg.read_text().replace("8px", "11px"))
        self.assert_invalid("font size")
        self.svg.write_text(self.svg.read_text().replace("11px", "8px").replace("stroke:#000000", "stroke:rgba(0,0,0,0.3)", 1))
        self.assert_invalid("effective opacity")

    def test_solid_stimulus_must_not_become_dashed_reference(self):
        record = self.candidate["element_registry"]["reference"]
        record.update(scientific_role="stimulus.cs.onset", style_role="stimulus_cs_onset",
                      resolved_style=copy.deepcopy(SPEC["styles"]["stimulus_cs_onset"]),
                      style_verification={"zorder": {"status": "passed", "evidence": "Fixture renderer"}})
        self.svg.write_text(self.svg.read_text().replace("stroke-width:0.6", "stroke-width:0.7;stroke-dasharray:2,1", 1).replace("stroke:#000000", "stroke:#0d8136", 1))
        self.assert_invalid("dash pattern")

    def test_nested_scaling_is_measured(self):
        self.svg.write_text(self.svg.read_text().replace('<g id="reference">', '<g id="reference" transform="scale(2)">').replace("stroke-width:0.6", "stroke-width:0.3", 1))
        self.original.write_bytes(self.svg.read_bytes())
        self.candidate["source_artifacts"][0]["sha256"] = freeze.digest(self.original)
        self.candidate["assembly_scale"]["source_to_final_transforms"]["svg_transforms"] = [{"element_id": "reference", "transform": "scale(2)"}]
        self.assertTrue(self.check()["valid"])
        self.svg.write_text(self.svg.read_text().replace("scale(2)", "scale(2,3)"))
        self.candidate["assembly_scale"]["source_to_final_transforms"]["svg_transforms"][0]["transform"] = "scale(2,3)"
        self.assert_invalid("Nonuniform")

    def test_unregistered_marks_and_container_coverage_cannot_bypass(self):
        self.svg.write_text(self.svg.read_text().replace("</svg>", '<circle cx="20" cy="20" r="2"/></svg>'))
        self.assert_invalid("Unregistered circle")
        self.candidate["element_registry"]["trace"]["scientific_role"] = "axes.container"
        self.candidate["element_registry"]["trace"]["artist_type"] = "Axes"
        self.candidate["element_registry"]["trace"]["style_role"] = "axes"
        self.candidate["element_registry"]["trace"]["resolved_style"] = SPEC["styles"]["axes"]
        self.assert_invalid("Unregistered path")

    def test_duplicate_ids_and_scientific_text_changes_block(self):
        self.svg.write_text(self.svg.read_text().replace('id="trace"', 'id="reference"'))
        self.assert_invalid("Duplicate SVG ID")
        self.svg.write_bytes(self.original.read_bytes())
        self.svg.write_text(self.svg.read_text().replace("Time (s)", "Response (AU)"))
        self.assert_invalid("Scientific text changed")

    def test_stale_inputs_between_check_and_publication_block(self):
        self.save(); report = freeze.check_candidate(self.candidate_path)
        self.data.write_text("changed during review")
        with patch.object(freeze, "check_candidate", return_value=report):
            result = freeze.freeze_candidate(self.candidate_path, self.folder/"frozen.json")
        self.assertFalse(result["valid"])
        self.assertEqual(result["issues"][0]["code"], "stale-review")

    def test_cli_check_only_and_failure_exit(self):
        self.save()
        args = [sys.executable, str(ROOT/"scripts/freeze_figure.py"), "--candidate", str(self.candidate_path), "--check-only"]
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertTrue(json.loads(result.stdout)["valid"])
        self.candidate["element_registry"] = {}; self.save()
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["valid"])
        self.assertFalse((self.folder/"frozen.json").exists())

    def test_malformed_input_reports_instead_of_crashing(self):
        self.candidate["approved_exceptions"] = ["not an exception record"]
        self.assert_invalid("input")

    def test_unmeasurable_properties_need_review_and_relationships_resolve(self):
        self.candidate["element_registry"]["trace"]["style_verification"] = {}
        self.assert_invalid("review evidence")
        self.candidate["element_registry"]["trace"]["style_verification"] = {"zorder": {"status": "passed", "evidence": "Fixture source"}}
        self.candidate["element_registry"]["trace"]["mappable_ids"] = ["missing"]
        self.assert_invalid("relationship")

    def test_ancestor_geometry_and_resource_changes_are_protected(self):
        self.svg.write_text(self.svg.read_text().replace('<g id="trace">', '<g transform="translate(0,10)"><g id="trace">').replace('</g>\n<g id="label">', '</g></g>\n<g id="label">'))
        self.candidate["assembly_scale"]["source_to_final_transforms"]["svg_transforms"] = [{"element_id": None, "transform": "translate(0,10)"}]
        self.assert_invalid("Protected geometry")
        self.svg.write_bytes(self.original.read_bytes())
        text = self.svg.read_text().replace('<g id="trace"><path d="M 0 40 L 100 60"', '<defs><path id="template" d="M 0 40 L 100 60"/></defs><g id="trace"><use href="#template"')
        self.svg.write_text(text); self.original.write_text(text)
        self.candidate["source_artifacts"][0]["sha256"] = freeze.digest(self.original)
        self.candidate["assembly_scale"]["source_to_final_transforms"]["svg_transforms"] = []
        self.assertTrue(self.check()["valid"])
        self.svg.write_text(text.replace('id="template" d="M 0 40 L 100 60"', 'id="template" d="M 0 40 L 100 70"'))
        self.assert_invalid("Protected geometry")

    def test_tick_size_direction_and_value_requirements(self):
        record = self.candidate["element_registry"].pop("reference")
        record.update(element_id="tick", scientific_role="axis.tick", protection="axis-definition",
                      style_role="tick", resolved_style=copy.deepcopy(SPEC["styles"]["tick"]),
                      geometry={"dimension": "x", "side": "bottom", "kind": "major", "value": 0},
                      style_verification={"pad_pt": {"status": "passed", "evidence": "Fixture renderer"}})
        self.candidate["element_registry"]["tick"] = record
        self.svg.write_text(self.svg.read_text().replace('id="reference"', 'id="tick"').replace('d="M 0 50 L 100 50"', 'd="M 0 0 L 0 2"').replace('stroke-width:0.6', 'stroke-width:0.5', 1))
        self.original.write_bytes(self.svg.read_bytes())
        self.candidate["source_artifacts"][0]["sha256"] = freeze.digest(self.original)
        self.assertTrue(self.check()["valid"])
        self.svg.write_text(self.svg.read_text().replace('M 0 0 L 0 2', 'M 0 0 L 0 -2'))
        self.assert_invalid("outward")
        self.svg.write_text(self.svg.read_text().replace('M 0 0 L 0 -2', 'M 0 0 L 0 3'))
        self.assert_invalid("tick length")
        self.svg.write_text(self.svg.read_text().replace('M 0 0 L 0 3', 'M 0 1 L 0 3'))
        self.assert_invalid("tick anchor")

    def test_reference_stacking_and_strict_json(self):
        text = self.svg.read_text()
        start, end = text.index('<g id="reference">'), text.index('</g>')+4
        self.svg.write_text(text[:start]+text[end:].replace('<g id="label">', text[start:end]+'\n<g id="label">'))
        self.assert_invalid("above scientific data")
        self.save()
        self.candidate_path.write_text(self.candidate_path.read_text().replace('"figure_id": "fig1"', '"figure_id": "fig1", "figure_id": "fig2"', 1))
        self.assertIn("Duplicate JSON key", json.dumps(freeze.check_candidate(self.candidate_path)))

    def test_main_cli_freeze_parser_and_dispatch(self):
        # Exercise the real parser without importing unrelated analysis libraries.
        source = ast.parse((ROOT/"src/classical_conditioning/cli.py").read_text())
        functions = [item for item in source.body if isinstance(item, ast.FunctionDef)]
        namespace = {"argparse": argparse, "Path": Path, "Sequence": list, "sys": sys,
                     "PAPER_METRIC_ID": "fixture", "ensure_supported_runtime": lambda: None}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "cli.py", "exec"), namespace)
        args = namespace["build_parser"]().parse_args(["freeze-figure", "--candidate", "candidate.json", "--check-only"])
        self.assertEqual(args.candidate, Path("candidate.json"))
        self.assertTrue(args.check_only)
        import types
        package = types.ModuleType("classical_conditioning")
        module = types.ModuleType("classical_conditioning.figure_freeze")
        with patch.dict(sys.modules, {"classical_conditioning": package, "classical_conditioning.figure_freeze": module}):
            with patch.object(module, "run_cli", create=True) as runner:
                namespace["main"](["freeze-figure", "--candidate", "candidate.json", "--output", "new.json"])
                runner.assert_called_once_with(["--candidate", "candidate.json", "--output", "new.json"])

    def test_hidden_ancestors_and_unresolved_svg_effects(self):
        self.svg.write_text(self.svg.read_text().replace('<g id="reference">', '<g display="none" id="reference">').replace('stroke-width:0.6', 'stroke-width:0.6;display:inline', 1))
        self.assert_invalid("hidden")
        self.svg.write_bytes(self.original.read_bytes())
        self.svg.write_text(self.svg.read_text().replace('stroke-width:0.6', 'stroke-width:0.6;vector-effect:non-scaling-stroke', 1))
        self.assert_invalid("vector-effect")

    def test_scientific_definitions_cannot_be_presentation_exceptions(self):
        self.candidate["element_registry"]["reference"]["geometry"]["value"] = 0
        self.assert_invalid("zero/one reference")
        self.candidate["element_registry"]["reference"]["geometry"]["value"] = 1
        self.candidate["approved_exceptions"] = [{"exception_id": "change-ratio", "scientific_role": "reference.response_ratio.equal_baseline",
            "property": "measure", "value": "different metric", "reason": "not a presentation choice",
            "scope": {"figure_id": "fig1", "panel_ids": ["a"]}, "approval_evidence": "fixture"}]
        self.assert_invalid("presentation default")

    def test_event_subroles_keep_distinct_patterns(self):
        original = self.svg.read_text()
        record = self.candidate["element_registry"]["reference"]
        for role, style, color, pattern in (
                ("stimulus.cs.onset", "stimulus_cs_onset", "#0d8136", "none"),
                ("stimulus.cs.offset", "stimulus_cs_offset", "#0d8136", "2.59,1.12"),
                ("stimulus.us.actual", "stimulus_us_actual", "#702e78", "none"),
                ("stimulus.us.expected", "stimulus_us_expected", "#702e78", "0.7,1.155")):
            with self.subTest(role=role):
                record.update(scientific_role=role, style_role=style, resolved_style=copy.deepcopy(SPEC["styles"][style]),
                              style_verification={"zorder": {"status": "passed", "evidence": "Fixture renderer"}})
                self.svg.write_text(original.replace("stroke-width:0.6", f"stroke-width:0.7;stroke-dasharray:{pattern}", 1).replace("stroke:#000000", f"stroke:{color}", 1))
                self.assertTrue(self.check()["valid"], self.check())

    def test_condition_color_is_measured_not_only_named(self):
        record = self.candidate["element_registry"]["trace"]
        record.update(scientific_role="trajectory.fish", style_role="fish_trajectory",
                      resolved_style=copy.deepcopy(SPEC["styles"]["fish_trajectory"]),
                      resolved_color="#123456", color_evidence="Fixture condition color mapping",
                      style_verification={key: {"status": "passed", "evidence": "Fixture renderer"}
                                          for key in ("zorder", "color_source")})
        text = self.svg.read_text().replace('<g id="trace">', '<g id="trace" stroke-opacity="0.3">')
        self.svg.write_text(text)
        self.assert_invalid("Rendered color differs")
        self.svg.write_text(text.replace('id="trace" stroke-opacity="0.3"><path d="M 0 40 L 100 60" style="fill:none;stroke:#000000', 'id="trace" stroke-opacity="0.3"><path d="M 0 40 L 100 60" style="fill:none;stroke:#123456'))
        self.assertTrue(self.check()["valid"], self.check())

    def test_generated_clip_ids_can_change_but_clip_geometry_cannot(self):
        text = self.svg.read_text().replace('<g id="trace">', '<defs><clipPath id="oldclip"><rect x="0" y="0" width="100" height="100"/></clipPath></defs><g clip-path="url(#oldclip)"><g id="trace">').replace('</g>\n<g id="label">', '</g></g>\n<g id="label">')
        self.svg.write_text(text); self.original.write_text(text)
        self.candidate["source_artifacts"][0]["sha256"] = freeze.digest(self.original)
        self.svg.write_text(text.replace('oldclip', 'newclip'))
        self.assertTrue(self.check()["valid"], self.check())
        self.svg.write_text(self.svg.read_text().replace('width="100"', 'width="50"'))
        self.assert_invalid("Protected geometry")


if __name__ == "__main__":
    unittest.main()
