from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib as mpl

from classical_conditioning.figures.presentation_profiles import ROOT, load_presentation_profile, main
from classical_conditioning.figures.theme import DEFAULT_THEME, apply_theme, get_theme


class PresentationProfileTests(unittest.TestCase):
    def test_single_role_lookup_includes_style_geometry_and_input_hashes(self) -> None:
        output = io.StringIO()
        with patch("sys.argv", ["presentation_profiles", "--role", "axis.tick"]), redirect_stdout(output):
            main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["section"], "role")
        self.assertEqual(result["value"]["definition"]["style"], "tick")
        self.assertEqual(result["value"]["style"]["length_pt"], 2)
        self.assertEqual(result["value"]["author_geometry"]["tick_end_to_label_edge_pt"], 3)
        self.assertTrue(result["render_verification_required"])
        self.assertEqual(len(result["inputs"]), 3)

    def test_overlay_preserves_semantics_colors_and_original_bytes(self) -> None:
        path = ROOT / "configs/paper-figures/figure-elements.json"
        raw = path.read_bytes()
        original = json.loads(raw)
        resolved = load_presentation_profile()
        for key in ("roles", "plot_families", "colors"):
            self.assertEqual(resolved.contract[key], original[key])
        self.assertEqual(resolved.contract["styles"]["summary_curve"]["linewidth_pt"], 1)
        self.assertEqual(original["styles"]["summary_curve"]["linewidth_pt"], 1.2)
        self.assertEqual(resolved.contract["styles"]["uncertainty_band"]["alpha"], .2)
        self.assertNotIn("color_source", resolved.contract["styles"]["condition_title"])
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(resolved.inputs["configs/paper-figures/figure-elements.json"], hashlib.sha256(raw).hexdigest())
        self.assertTrue(resolved.contract["presentation_profile"]["view_only"])
        self.assertFalse(resolved.contract["presentation_profile"]["freeze_gate_integration"])

    def test_resolutions_do_not_share_mutable_data(self) -> None:
        first = load_presentation_profile()
        first.contract["styles"]["trace"]["linewidth_pt"] = 99
        self.assertEqual(load_presentation_profile().contract["styles"]["trace"]["linewidth_pt"], .6)

    def test_theme_is_explicit_and_legacy_default_is_unchanged(self) -> None:
        self.assertIs(get_theme(), DEFAULT_THEME)
        theme = get_theme(profile="nature-main")
        self.assertEqual((theme.axes_labelsize, theme.tick_labelsize), (7, 6))
        self.assertEqual(theme.font_sans_serif, ("Arial",))
        with mpl.rc_context():
            apply_theme()
            self.assertEqual(mpl.rcParams["font.sans-serif"], ["DejaVu Sans"])
            self.assertEqual(mpl.rcParams["axes.labelsize"], 8)
            mpl.rcParams["font.style"] = "italic"
            mpl.rcParams["font.weight"] = "bold"
            mpl.rcParams["text.color"] = "red"
            mpl.rcParams["xtick.color"] = "red"
            mpl.rcParams["ytick.labelcolor"] = "red"
            with patch("classical_conditioning.figures.theme.fontManager.ttflist", [SimpleNamespace(name="Arial")]):
                apply_theme(theme)
            self.assertEqual(mpl.rcParams["font.sans-serif"], ["Arial"])
            self.assertEqual(mpl.rcParams["axes.labelsize"], 7)
            self.assertEqual(mpl.rcParams["xtick.labelsize"], 6)
            self.assertEqual(mpl.rcParams["font.style"], "normal")
            self.assertEqual(mpl.rcParams["font.weight"], "normal")
            self.assertEqual(mpl.rcParams["text.color"], "#000000")
            self.assertEqual(mpl.rcParams["xtick.color"], "#000000")
            self.assertEqual(mpl.rcParams["ytick.labelcolor"], "#000000")
            self.assertEqual(mpl.rcParams["pdf.fonttype"], 42)

    def test_missing_declared_font_fails_before_rcparams_mutation(self) -> None:
        with mpl.rc_context():
            before = dict(mpl.rcParams)
            with patch("classical_conditioning.figures.theme.fontManager.ttflist", [SimpleNamespace(name="DejaVu Sans")]):
                with self.assertRaisesRegex(RuntimeError, "Declared figure font"):
                    apply_theme(get_theme(profile="nature-main"))
            self.assertEqual(dict(mpl.rcParams), before)

    def test_invalid_profile_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_presentation_profile("../../figure-elements")

    def test_invalid_overrides_and_origins_fail_policy_preflight(self) -> None:
        policy = json.loads((ROOT / "configs/paper-figures/nature-main.json").read_text())
        changes = [
            ("stroke", lambda p: p["rules"]["project.style_overrides"]["value"]["trace"].update(linewidth_pt=2)),
            ("text", lambda p: p["rules"]["project.style_overrides"]["value"]["label"].update(font_size_pt=8)),
            ("style role", lambda p: p["rules"]["project.style_overrides"]["value"].update(unknown={"linewidth_pt": .5})),
            ("style property", lambda p: p["rules"]["project.style_overrides"]["value"]["label"].update(font_sze_pt=7)),
            ("rule origin", lambda p: p["rules"]["journal.text"].update(origin="invented")),
            ("approval reference", lambda p: p["rules"]["journal.text"].update(origin="scoped-exception")),
            ("Panel letter", lambda p: p["rules"]["project.style_overrides"]["value"]["panel_letter"].update(font_weight="normal")),
            ("fallback", lambda p: p["rules"]["project.typography"]["value"].update(allowed_families=["Arial", "Helvetica"])),
            ("opacity", lambda p: p["rules"]["project.style_overrides"]["value"]["trace"].update(alpha=2)),
            ("approval reference", lambda p: p["rules"]["journal.text"].update(origin="scoped-exception", approval_reference="invented-approval.json")),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "configs/paper-figures"
            folder.mkdir(parents=True)
            for name in ("figure-elements.json", "layout-frames.json"):
                (folder / name).write_bytes((ROOT / "configs/paper-figures" / name).read_bytes())
            for expected, change in changes:
                with self.subTest(expected=expected):
                    edited = copy.deepcopy(policy)
                    # Keep a separate stroke override for the rejection case.
                    edited["rules"]["project.style_overrides"]["value"]["trace"] = {"linewidth_pt": .6}
                    change(edited)
                    (folder / "nature-main.json").write_text(json.dumps(edited))
                    with self.assertRaisesRegex(ValueError, expected):
                        load_presentation_profile(root=root)

    def test_bound_policy_changes_are_visible_in_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "configs/paper-figures"
            folder.mkdir(parents=True)
            for name in ("figure-elements.json", "layout-frames.json", "nature-main.json"):
                (folder / name).write_bytes((ROOT / "configs/paper-figures" / name).read_bytes())
            first = load_presentation_profile(root=root)
            path = folder / "layout-frames.json"
            layout = json.loads(path.read_text())
            layout["selected_inset_mm"] = 1.5
            path.write_text(json.dumps(layout))
            second = load_presentation_profile(root=root)
            self.assertNotEqual(first.inputs, second.inputs)
            self.assertEqual(second.layout["selected_inset_mm"], 1.5)


if __name__ == "__main__":
    unittest.main()
