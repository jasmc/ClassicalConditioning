"""Normalize the supplied vector preparation scheme to the plot font."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from assemble_svg_figure import isolate_svg, remove_exact_text, repair_source_typography

ROOT = Path("J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/schemes")
SOURCE = ROOT / "Fig1_PanelA_Setup_source.svg"
OUTPUT = ROOT / "Fig1_PanelA_Setup_DejaVuSans_v1.svg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
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
