"""Normalize the supplied vector preparation scheme to the plot font."""

from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]


import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.paper.shared.assemble_svg import isolate_svg, remove_exact_text, repair_source_typography
from classical_conditioning.external_artifacts import resolve_artifact, relocated_external_directory

ROOT = Path('J:/ClassicalConditioning Outputs/figures/figure-01/panels-a-b-c/versions/source-schemes')
SOURCE = ROOT / "Fig1_PanelA_Setup_source.svg"
OUTPUT = ROOT / "Fig1_PanelA_Setup_DejaVuSans_v1.svg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.source=resolve_artifact(args.source)
    args.output=relocated_external_directory(args.output)
    root = ET.parse(args.source).getroot()
    remove_exact_text(root, ["A"])
    repair_source_typography(root, {
        "remove_groups_by_class": ["p"],
        "add_source_text": [
            {"text": "Basal", "x": 77, "y": 33.2, "size": 10, "anchor": "middle"},
            {"text": "illumination", "x": 77, "y": 45.2, "size": 10, "anchor": "middle"},
        ],
    })
    isolate_svg(root, "fig1-a", "DejaVu Sans")
    root.set("font-family", "DejaVu Sans")
    root.set("role", "img")
    root.set("aria-label", "Figure 1A larval preparation and CS/US illumination")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output)


if __name__ == "__main__":
    main()
