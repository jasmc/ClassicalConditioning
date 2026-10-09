"""Build the Figure 2 review layout, preserving SSD versions and source provenance.

This command composes supplied panels only; it never calculates scientific data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from assemble_svg_figure import digest, export_with_inkscape, render, storage_root

REPO = Path(__file__).resolve().parents[1]
LAYOUT = REPO / "configs/paper-figures/figure2-assembly.json"
DEFAULT_ROOTS = [
    REPO,
    REPO.parent / "ClassicalConditioningPaper",
    Path("J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs"),
    Path("J:/Digested Data"),
    Path("J:/Raw Data"),
    Path("F:/Digested Data"),
]


def inventory(roots: list[Path], destination: Path) -> dict:
    """Persist exhaustive figure candidates; record inaccessible roots as errors."""
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "roots": [], "candidates": []}
    pattern = re.compile(r"fig(?:ure)?[-_ ]?2|assembly|paper-panel-run|plot-versions", re.I)
    for root in roots:
        record = {"path": str(root), "status": "complete", "errors": [], "file_count": 0}
        if not root.is_dir():
            record.update(status="unavailable", errors=["Directory missing or inaccessible"])
            result["roots"].append(record)
            continue

        def on_error(error: OSError) -> None:
            record["status"] = "incomplete"
            record["errors"].append(str(error))

        # Path.walk reports permission failures rather than hiding them.
        for parent, directories, names in root.walk(on_error=on_error):
            directories[:] = [d for d in directories if d not in {".git", "node_modules", "__pycache__"}
                              and not d.startswith(".venv") and parent / d != destination]
            for name in names:
                path = parent / name
                record["file_count"] += 1
                if path.suffix.lower() not in {".svg", ".png", ".pdf", ".json", ".csv", ".parquet", ".py", ".md"}:
                    continue
                if path.suffix.lower() != ".svg" and not pattern.search(str(path.relative_to(root))):
                    continue
                try:
                    item = {"path": str(path), "bytes": path.stat().st_size,
                            "sha256": digest(path), "extension": path.suffix.lower()}
                    if path.suffix.lower() == ".svg":
                        tree = ET.parse(path).getroot()
                        item["text_excerpt"] = " | ".join(
                            "".join(e.itertext()).strip() for e in tree.iter()
                            if e.tag.endswith("}text") or e.tag.endswith("}title"))[:1200]
                    result["candidates"].append(item)
                except (OSError, ET.ParseError) as error:
                    on_error(error)
        result["roots"].append(record)
    # Assets at the SSD top level are Figure 1 apparatus/timing sources.
    for path in Path("J:/").glob("Asset *.svg"):
        result["candidates"].append({"path": str(path), "bytes": path.stat().st_size,
                                     "sha256": digest(path), "extension": ".svg",
                                     "selection": "Figure 1 scheme; excluded from Figure 2"})
    return result


def validate(layout: dict, base: Path) -> None:
    if [p["id"] for p in layout["panels"]] != list("ABCDEFGHI"):
        raise ValueError("Figure 2 must contain all planned panels A-I in order")
    if layout.get("baseline_s") != [-15, 0] or layout.get("baseline_interval") != "[-15,0)":
        raise ValueError("Figure 2 baseline must be [-15,0) s relative to CS onset")
    for panel in layout["panels"]:
        source = panel.get("source")
        if not source:
            continue
        path = (base / source).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Assigned panel {panel['id']} is missing: {path}")
        if path.suffix.lower() != ".svg":
            raise ValueError("Figure 2 populated panels require editable SVG sources")
        provenance = panel.get("source_provenance", {})
        if provenance.get("baseline_s") != [-15, 0] or provenance.get("baseline_interval") != "[-15,0)":
            raise ValueError(f"Panel {panel['id']} lacks verified frozen-baseline provenance")
        if provenance.get("metric_id") != layout["metric_id"]:
            raise ValueError(f"Panel {panel['id']} metric differs from the paper metric")
        if provenance.get("svg_sha256") != digest(path):
            raise ValueError(f"Panel {panel['id']} SVG provenance hash differs")
        if not provenance.get("cohort_hash") or not provenance.get("panel_data_sha256"):
            raise ValueError(f"Panel {panel['id']} requires cohort and panel-data identity")
        data_path = (base / provenance["panel_data"]).resolve()
        if provenance["panel_data_sha256"] != digest(data_path):
            raise ValueError(f"Panel {panel['id']} plotted-data hash differs")
        if provenance.get("significance_marks") != "none":
            raise ValueError("This descriptive review layout requires sources without significance marks")
        if panel["id"] in "DEF" and provenance.get("block_trials") != [[10, 14], [65, 69], [90, 94]]:
            raise ValueError(f"Panel {panel['id']} selected blocks differ from the current plan")
        sidecar_path = (base / provenance["sidecar"]).resolve()
        if provenance.get("sidecar_sha256") != digest(sidecar_path):
            raise ValueError(f"Panel {panel['id']} scientific sidecar hash differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=LAYOUT)
    parser.add_argument("--inventory", action="store_true", help="Repeat repo, SSD and raw-project inventory")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    base = storage_root(layout, args.layout)
    if base.is_relative_to(REPO):
        raise ValueError("Figure artifacts must be stored outside the repo")
    validate(layout, base)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    version = base / "versions" / stamp
    version.mkdir(parents=True)
    if args.inventory:
        payload = inventory(DEFAULT_ROOTS, base)
        (version / "inventory.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(version / "inventory.json", base / "inventory.json")
    # Font files are reusable SSD assets, sourced from the active plotting runtime.
    from matplotlib import font_manager
    font_dir = base / layout["font_directory"]
    font_dir.mkdir(exist_ok=True)
    for weight, filename in [("normal", "DejaVuSans.ttf"), ("bold", "DejaVuSans-Bold.ttf")]:
        source = Path(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight),
                                            fallback_to_default=False))
        shutil.copy2(source, font_dir / filename)
    output = version / layout["output"]
    sidecar = render(args.layout, output)
    if not args.no_preview:
        export_with_inkscape(output, ["png"], font_directory=font_dir)
    for panel in layout["panels"]:
        render(args.layout, version / "panels" / f"Fig2_Panel{panel['id']}.svg", only_panel=panel["id"])
    shutil.copy2(args.layout, version / "figure2-assembly.layout.json")
    if (base / "inventory.json").is_file():
        shutil.copy2(base / "inventory.json", version / "inventory.json")
    verification = {"version": str(version), "baseline_s": layout["baseline_s"],
                    "populated": [p["id"] for p in sidecar["panels"] if p["status"] == "source"],
                    "placeholders": [p["id"] for p in sidecar["panels"] if p["status"] == "placeholder"],
                    "files": {str(p.relative_to(version)): digest(p) for p in version.rglob("*") if p.is_file()},
                    "fonts": {p.name: digest(p) for p in font_dir.glob("*.ttf")},
                    "builder_sha256": digest(Path(__file__)),
                    "assembler_sha256": digest(REPO / "scripts/assemble_svg_figure.py")}
    tree = ET.parse(output).getroot()
    ids = [e.get("id") for e in tree.iter() if e.get("id")]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate SVG IDs")
    if any(e.tag.endswith("}image") for e in tree.iter()):
        raise ValueError("Figure 2 contains raster artwork")
    for panel in "ABCDEFGHI":
        if f"panel-{panel}" not in ids or f"label-{panel}" not in ids:
            raise ValueError(f"Panel {panel} missing from SVG")
    (version / "build.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    # Preserve any pre-existing stable entry point before publishing the new one.
    previous = version / "previous-main"
    for name in (layout["output"], layout["output"] + ".json", Path(layout["output"]).with_suffix(".png").name, "build.json"):
        old = base / name
        if old.is_file():
            previous.mkdir(exist_ok=True)
            shutil.copy2(old, previous / name)
        new = version / name
        if new.is_file():
            shutil.copy2(new, old)
    print(f"Built {base / layout['output']}")
    print(f"Preserved version: {version}")
    print(f"Populated: {verification['populated']}; placeholders: {verification['placeholders']}")


if __name__ == "__main__":
    main()
