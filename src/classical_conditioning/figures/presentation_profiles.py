"""Read-only presentation overlays on the existing semantic/style contract.

No selection, freeze, publication or figure mutation occurs here. The theme
adapter is opt-in; rendered geometry and font embedding require separate checks.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class PresentationProfile:
    """Fresh owned policy data and the hashes of every resolved input."""

    policy: dict[str, Any]
    contract: dict[str, Any]
    layout: dict[str, Any]
    inputs: dict[str, str]

    def rule(self, name: str) -> Any:
        return copy.deepcopy(self.policy["rules"][name]["value"])

    def theme_parameters(self) -> dict[str, Any]:
        """Adapt supported fields to FigureTheme; leave role-specific artists explicit."""
        typography = self.contract["typography"]
        styles = self.contract["styles"]
        colors = self.contract["colors"]
        def rgb(value: str) -> tuple[float, ...]:
            return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
        return {
            "font_sans_serif": tuple(typography["allowed_families"]),
            "font_style": typography["font_style"],
            "font_weight": typography["font_weight"],
            "text_color": styles["label"]["color"],
            "tick_color": styles["tick"]["color"],
            "spine_color": styles["spine"]["color"],
            "tick_capstyle": self.rule("author.axis_geometry")["tick_cap"],
            "cs_linewidth": styles["stimulus_cs_onset"]["linewidth_pt"],
            "us_linewidth": styles["stimulus_us_actual"]["linewidth_pt"],
            "stimulus_span_zorder": styles["stimulus_window"]["zorder"],
            "font_size": typography["base_size_pt"],
            "axes_labelsize": styles["label"]["font_size_pt"],
            "axes_titlesize": styles["title"]["font_size_pt"],
            "tick_labelsize": styles["tick_label"]["font_size_pt"],
            "legend_fontsize": styles["legend"]["font_size_pt"],
            "figure_titlesize": styles["title"]["font_size_pt"],
            "axes_linewidth": styles["spine"]["linewidth_pt"],
            "lines_linewidth": styles["trace"]["linewidth_pt"],
            "tick_major_size": styles["tick"]["length_pt"],
            "tick_major_width": styles["tick"]["linewidth_pt"],
            "tick_major_pad": styles["tick"]["pad_pt"],
            "axes_labelpad": styles["label"]["pad_pt"],
            "axes_titlepad": styles["title"]["pad_pt"],
            "axes_titlelocation": self.rule("project.axes")["title_location"],
            "cs_color": rgb(colors["cs"]),
            "us_color": rgb(colors["us"]),
            "heatmap_bad_color": rgb(colors["heatmap_missing"]),
            "intensity_cmap": colors["heatmap_cmap"],
        }


def load_presentation_profile(
    profile: str = "nature-main", *, root: Path = ROOT,
) -> PresentationProfile:
    """Resolve a versioned overlay without modifying its legacy contract.

    Validate policy consistency only. A successful load is not rendered-figure
    evidence and cannot be substituted for the explicit freeze gate.
    """
    if not re.fullmatch(r"[a-z][a-z0-9-]*", profile):
        raise ValueError("Invalid presentation profile name")
    root = root.resolve()
    inputs: dict[str, str] = {}
    def read(relative: str) -> dict[str, Any]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Profile input escapes repository root")
        raw = path.read_bytes()
        inputs[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    policy = read(f"configs/paper-figures/{profile}.json")
    if policy.get("schema_version") != 1 or policy.get("profile_id") != profile:
        raise ValueError("Unsupported or mismatched presentation profile")
    rules = policy["rules"]
    for name, rule in rules.items():
        if rule["origin"] not in {"journal-requirement", "journal-recommendation", "author-rule", "project-default", "scoped-exception"}:
            raise ValueError(f"Unknown rule origin: {name}")
        if rule["origin"] == "scoped-exception":
            raise ValueError(f"Scoped exception needs a candidate-bound approval reference, not a global profile: {name}")
        for source in rule.get("sources", []):
            if source not in policy["sources"]:
                raise ValueError(f"Unresolved policy source: {source}")
    # Each JSON read already owns fresh data; no second full-contract copy is needed.
    contract = read(policy["base_contract"])
    layout = read(policy["layout_policy"])
    if layout.get("activation") != "active" or layout["selected_inset_mm"] <= 0:
        raise ValueError("Profile requires an active selected layout policy")
    contract["typography"].update(rules["project.typography"]["value"])
    known_properties = {key for style in contract["styles"].values() for key in style}
    for style, overrides in rules["project.style_overrides"]["value"].items():
        if style not in contract["styles"]:
            raise ValueError(f"Unknown existing style role: {style}")
        for key, value in overrides.items():
            if key not in known_properties:
                raise ValueError(f"Unknown style property: {style}.{key}")
            if value is None:
                contract["styles"][style].pop(key, None)
            else:
                contract["styles"][style][key] = value
    # Typography summary derives from the same role values used by renderers.
    for key, style in {
        "label_size_pt": "label", "title_size_pt": "title",
        "tick_size_pt": "tick_label", "legend_size_pt": "legend",
        "panel_letter_size_pt": "panel_letter",
    }.items():
        contract["typography"][key] = contract["styles"][style]["font_size_pt"]
    text = rules["journal.text"]["value"]
    stroke = rules["journal.stroke"]["value"]
    typography = contract["typography"]
    if not typography["allowed_families"] or typography["family"] != typography["allowed_families"][0]:
        raise ValueError("Profile must declare its preferred font first")
    if text.get("same_family_across_paper") and len(typography["allowed_families"]) != 1:
        raise ValueError("Paper-wide font choice cannot use machine-dependent fallback")
    if not text["ordinary_min_pt"] <= typography["base_size_pt"] <= text["ordinary_max_pt"]:
        raise ValueError("Profile base text size outside journal bounds")
    letters = contract["styles"]["panel_letter"]
    if letters["font_weight"] != text["panel_letter_weight"] or typography["font_style"] != text["panel_letter_style"]:
        raise ValueError("Panel letter typography disagrees with journal requirement")
    for name, style in contract["styles"].items():
        if "alpha" in style and not 0 <= style["alpha"] <= 1:
            raise ValueError(f"Profile opacity outside valid range: {name}")
        if "font_size_pt" in style:
            size = style["font_size_pt"]
            if name == "panel_letter":
                valid = size == text["panel_letter_pt"]
            else:
                valid = text["ordinary_min_pt"] <= size <= text["ordinary_max_pt"]
            if not valid:
                raise ValueError(f"Profile text size outside journal bounds: {name}")
        for key, value in style.items():
            if key.endswith("linewidth_pt") and not stroke["min_pt"] <= value <= stroke["max_pt"]:
                raise ValueError(f"Profile stroke outside journal bounds: {name}.{key}")
    geometry = rules["author.axis_geometry"]["value"]
    tick = contract["styles"]["tick"]
    if tick["length_pt"] != geometry["tick_length_pt"] or tick["linewidth_pt"] != geometry["tick_width_pt"]:
        raise ValueError("Profile tick style disagrees with author geometry")
    contract["presentation_profile"] = {
        "profile_id": profile, "version": policy["version"], "inputs": copy.deepcopy(inputs),
        "view_only": True, "freeze_gate_integration": False,
    }
    return PresentationProfile(policy, contract, layout, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="nature-main")
    parser.add_argument("--role", help="Look up one existing canonical element role")
    parser.add_argument("--section", choices=("summary", "rules", "typography", "styles", "colors", "roles", "theme"), default="summary")
    args = parser.parse_args()
    profile = load_presentation_profile(args.profile)
    if args.role:
        if args.role not in profile.contract["roles"]:
            parser.error(f"Unknown canonical element role: {args.role}")
        definition = profile.contract["roles"][args.role]
        value = {
            "role": args.role, "definition": definition,
            "style": profile.contract["styles"][definition["style"]],
            "overrides_rule": "project.style_overrides",
        }
        if args.role.startswith("axis."):
            value["author_geometry"] = profile.rule("author.axis_geometry")
        args.section = "role"
    elif args.section == "summary":
        value = {
            "scope": profile.policy["scope"],
            "typography": profile.contract["typography"],
            "dimensions": profile.rule("journal.dimensions"),
            "stroke_range": profile.rule("journal.stroke"),
            "selected_inset_mm": profile.layout["selected_inset_mm"],
            "unresolved": profile.rule("project.unresolved"),
        }
    elif args.section == "theme":
        value = profile.theme_parameters()
    elif args.section == "rules":
        value = profile.policy["rules"]
    else:
        value = profile.contract[args.section]
    print(json.dumps({
        "profile_id": profile.policy["profile_id"], "version": profile.policy["version"],
        "inputs": profile.inputs, "render_verification_required": True,
        "automatic_artwork_migration": False, "section": args.section, "value": value,
    }, indent=2))


if __name__ == "__main__":
    main()
