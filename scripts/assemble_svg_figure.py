"""Compose replaceable SVG panels from a declarative paper-figure layout.

The assembly is presentation only: source SVGs are embedded as vector elements,
and scientific values are never inferred from artwork. Run with --help for use.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)


def q(name: str) -> str:
    return f"{{{SVG}}}{name}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def storage_root(layout: dict, layout_path: Path) -> Path:
    configured = layout.get("storage_root")
    if configured:
        return Path(configured).expanduser().resolve()
    return layout_path.resolve().parent


def panel_layout(layout: dict, panel_id: str) -> dict:
    """Make a standalone canvas for visual review of one named panel."""
    matches = [panel for panel in layout["panels"] if panel["id"] == panel_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate panel ID: {panel_id}")
    panel = copy.deepcopy(matches[0])
    x, y, width, height = panel["box"]
    panel["box"] = [0, 0, width, height]
    if "content_box" in panel:
        cx, cy, cw, ch = panel["content_box"]
        panel["content_box"] = [cx - x, cy - y, cw, ch]
    if "label" in panel:
        panel["label"]["x"] = panel["label"].get("x", x + 6) - x
        panel["label"]["y"] = panel["label"].get("y", y + 34) - y
    result = {key: value for key, value in layout.items() if key != "panels"}
    result["title"] = f"{layout.get('title', 'Figure')} — panel {panel_id} preview"
    result["canvas"] = [width, height]
    result["panels"] = [panel]
    return result


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    if not name:
        raise ValueError("Panel IDs must contain a letter or digit")
    return name


def normalize_css(css: str) -> str:
    # Illustrator's rgba() fill renders black in some Inkscape versions.
    # Equivalent RGB plus fill-opacity works in both SVG renderers.
    def rgba(match: re.Match[str]) -> str:
        kind, r, g, b, alpha = match.groups()
        return f"{kind}:rgb({r},{g},{b});{kind}-opacity:{alpha}"

    return re.sub(
        r"\b(fill|stroke):rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([.\d]+)\s*\)",
        rgba,
        css,
    )


def isolate_svg(source: ET.Element, prefix: str, font_family: str) -> None:
    """Prefix IDs and CSS classes so independently authored panels cannot clash."""
    ids: dict[str, str] = {}
    for element in source.iter():
        if element.tag == q("script"):
            raise ValueError("SVG scripts are not supported in figure panels")
        old_id = element.get("id")
        if old_id:
            ids[old_id] = f"{prefix}-{old_id}"
            element.set("id", ids[old_id])
        classes = element.get("class")
        if classes:
            element.set("class", " ".join(f"{prefix}-{part}" for part in classes.split()))
        if element.get("font-family"):
            element.set("font-family", font_family)

    for element in source.iter():
        for key, value in list(element.attrib.items()):
            if key in {"id", "class"}:
                continue
            if key == "font-family":
                continue
            value = re.sub(r"url\(#([^)]*)\)", lambda m: f"url(#{ids.get(m[1], m[1])})", value)
            if key == "style":
                value = re.sub(r"font-family\s*:[^;}]+", f"font-family:'{font_family}'", value)
            if key.rsplit("}", 1)[-1] == "href":
                if value.startswith("#"):
                    value = "#" + ids.get(value[1:], value[1:])
                elif not value.startswith("data:"):
                    raise ValueError(f"External SVG resource is not self-contained: {value}")
            element.set(key, value)
        if element.tag == q("style") and element.text:
            css = normalize_css(element.text)
            css = re.sub(r"font-family\s*:[^;}]+", f"font-family:'{font_family}'", css)
            if "@import" in css:
                raise ValueError("External CSS imports are not supported")
            css = re.sub(r"url\(#([^)]*)\)", lambda m: f"url(#{ids.get(m[1], m[1])})", css)
            css = re.sub(
                r"([^{}]+)(\{)",
                lambda m: re.sub(
                    r"([.#])([A-Za-z_][\w-]*)",
                    lambda token: token[1] + prefix + "-" + token[2],
                    m[1],
                ) + m[2],
                css,
            )
            element.text = css


def remove_exact_text(source: ET.Element, strings: list[str]) -> None:
    wanted = set(strings)
    for parent in source.iter():
        for child in list(parent):
            if child.tag == q("text") and "".join(child.itertext()).strip() in wanted:
                parent.remove(child)


def remove_lines_crossing_crop(source: ET.Element, crop_y: float) -> None:
    """Remove connector lines cut in half by a top-edge crop."""
    for parent in source.iter():
        for child in list(parent):
            if child.tag != q("line"):
                continue
            try:
                y1, y2 = float(child.get("y1", "nan")), float(child.get("y2", "nan"))
            except ValueError:
                continue
            if min(y1, y2) < crop_y < max(y1, y2):
                parent.remove(child)


def repair_source_typography(source: ET.Element, panel: dict) -> None:
    """Replace known outlined labels and Illustrator-positioned text runs."""
    removed_classes = set(panel.get("remove_groups_by_class", []))
    flattened = set(panel.get("flatten_text", []))
    adjustments = panel.get("text_adjustments", {})
    for parent in source.iter():
        for child in list(parent):
            if child.tag == q("g") and child.get("class") in removed_classes:
                parent.remove(child)
    for element in source.iter():
        if element.tag != q("text"):
            continue
        content = "".join(element.itertext()).strip()
        if content in flattened:
            for child in list(element):
                element.remove(child)
            element.text = content
            element.set("style", (element.get("style", "") + ";font-weight:bold").strip(";"))
        if content in adjustments:
            change = adjustments[content]
            if "transform" in change:
                element.set("transform", change["transform"])
            if "font_size" in change:
                element.set("style", (element.get("style", "") + f";font-size:{change['font_size']}px").strip(";"))
    for addition in panel.get("add_source_text", []):
        attributes = {
            "x": str(addition["x"]), "y": str(addition["y"]),
            "font-size": str(addition["size"]),
            "text-anchor": addition.get("anchor", "start"),
            "fill": addition.get("fill", "#231f20"),
        }
        if addition.get("weight"):
            attributes["font-weight"] = addition["weight"]
        ET.SubElement(source, q("text"), attributes).text = addition["text"]


def render(layout_path: Path, output_path: Path, *, strict: bool = False,
           only_panel: str | None = None,
           source_overrides: dict[str, str] | None = None) -> dict:
    layout_path = layout_path.resolve()
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if source_overrides:
        known = {panel["id"] for panel in layout["panels"]}
        unknown = set(source_overrides) - known
        if unknown:
            raise ValueError(f"Unknown panel source override: {', '.join(sorted(unknown))}")
        for panel in layout["panels"]:
            if panel["id"] in source_overrides:
                panel["source"] = source_overrides[panel["id"]]
                # Illustrator-specific crops and text repairs belong to the
                # selected source, not to a newly authored replacement SVG.
                for key in ("view_box", "remove_lines_crossing_crop",
                            "flatten_text", "text_adjustments",
                            "remove_groups_by_class", "omit_text_exact",
                            "add_source_text"):
                    panel.pop(key, None)
    if only_panel:
        layout = panel_layout(layout, only_panel)
    asset_base = storage_root(layout, layout_path)
    width, height = layout["canvas"]
    font_family = layout.get("font_family", "DejaVu Sans")
    if width <= 0 or height <= 0:
        raise ValueError("Canvas dimensions must be positive")
    root = ET.Element(q("svg"), {
        "width": str(width), "height": str(height),
        "viewBox": f"0 0 {width} {height}",
        "role": "img", "aria-label": layout.get("title", layout_path.stem),
        "font-family": font_family,
    })
    ET.SubElement(root, q("title")).text = layout.get("title", layout_path.stem)
    definitions = ET.SubElement(root, q("defs"))
    ET.SubElement(root, q("rect"), {
        "x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "white",
    })
    seen: set[str] = set()
    records = []
    for panel in layout["panels"]:
        panel_id = safe_name(panel["id"])
        if panel_id in seen:
            raise ValueError(f"Duplicate panel ID: {panel_id}")
        seen.add(panel_id)
        x, y, w, h = panel["box"]
        if any(not isinstance(n, (int, float)) for n in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid box for {panel_id}")
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f"Panel {panel_id} extends beyond the canvas")
        content_x, content_y, content_w, content_h = panel.get("content_box", [x, y, w, h])
        if (content_w <= 0 or content_h <= 0 or content_x < x or content_y < y
                or content_x + content_w > x + w or content_y + content_h > y + h):
            raise ValueError(f"Invalid content_box for {panel_id}")
        group = ET.SubElement(root, q("g"), {"id": f"panel-{panel_id}", "data-role": panel.get("role", "")})
        clip = ET.SubElement(definitions, q("clipPath"), {"id": f"clip-{panel_id}", "clipPathUnits": "userSpaceOnUse"})
        ET.SubElement(clip, q("rect"), {"x": str(x), "y": str(y), "width": str(w), "height": str(h)})
        if panel.get("frame"):
            ET.SubElement(group, q("rect"), {
                "x": str(x), "y": str(y), "width": str(w), "height": str(h),
                "rx": str(panel["frame"].get("radius", 0)),
                "fill": "none", "stroke": panel["frame"].get("stroke", "#a9adb2"),
                "stroke-width": str(panel["frame"].get("width", 1.5)),
            })
        source_name = panel.get("source")
        source_path = (asset_base / source_name).resolve() if source_name else None
        if source_path and source_path.is_file():
            kind = source_path.suffix.lower()
            if kind == ".svg":
                source = copy.deepcopy(ET.parse(source_path).getroot())
                if source.tag != q("svg"):
                    raise ValueError(f"Not an SVG root: {source_path}")
                remove_exact_text(source, panel.get("omit_text_exact", []))
                repair_source_typography(source, panel)
                if panel.get("remove_lines_crossing_crop") and "view_box" in panel:
                    remove_lines_crossing_crop(source, float(panel["view_box"][1]))
                isolate_svg(source, f"src-{panel_id}", font_family)
                source.set("x", str(content_x))
                source.set("y", str(content_y))
                source.set("width", str(content_w))
                source.set("height", str(content_h))
                source.set("preserveAspectRatio", panel.get("fit", "xMidYMid meet"))
                source.set("overflow", "hidden")
                if "view_box" in panel:
                    source.set("viewBox", " ".join(map(str, panel["view_box"])))
                source_kind = "vector"
            elif kind in {".png", ".jpg", ".jpeg"}:
                if "view_box" in panel:
                    raise ValueError("view_box cropping requires an SVG source")
                mime = "image/png" if kind == ".png" else "image/jpeg"
                source = ET.Element(q("image"), {
                    "x": str(content_x), "y": str(content_y),
                    "width": str(content_w), "height": str(content_h),
                    "preserveAspectRatio": panel.get("fit", "xMidYMid meet"),
                    "href": f"data:{mime};base64,{base64.b64encode(source_path.read_bytes()).decode('ascii')}",
                })
                source_kind = "raster"
            else:
                raise ValueError(f"Unsupported panel source type: {source_path.suffix}")
            clipped = ET.SubElement(group, q("g"), {"clip-path": f"url(#clip-{panel_id})"})
            clipped.append(source)
            state = "source"
            source_hash = digest(source_path)
        else:
            if strict:
                raise FileNotFoundError(f"Required panel {panel_id}: {source_name or 'source unset'}")
            ET.SubElement(group, q("rect"), {
                "x": str(x), "y": str(y), "width": str(w), "height": str(h),
                "fill": "#fafafa", "stroke": "#b8b8b8", "stroke-width": "1.5",
                "stroke-dasharray": "7 5",
            })
            ET.SubElement(group, q("text"), {
                "x": str(x + w / 2), "y": str(y + h / 2),
                "text-anchor": "middle", "font-family": font_family,
                "font-size": "22", "fill": "#777",
            }).text = panel.get("placeholder", f"{panel_id}: awaiting selected panel")
            state = "placeholder"
            source_hash = None
            source_kind = None
        if panel.get("label"):
            label = panel["label"]
            ET.SubElement(group, q("text"), {
                "id": f"label-{panel_id}",
                "x": str(label.get("x", x + 6)), "y": str(label.get("y", y + 34)),
                "font-family": font_family, "font-size": str(label.get("size", 30)),
                "font-weight": "bold", "fill": "#222",
            }).text = label["text"]
        records.append({
            "id": panel_id, "role": panel.get("role"), "status": state,
            "source": str(source_path) if source_path else None,
            "sha256": source_hash, "source_kind": source_kind,
            "box": [x, y, w, h], "content_box": [content_x, content_y, content_w, content_h],
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    temp.write_bytes(payload)
    temp.replace(output_path)
    sidecar = {
        "title": layout.get("title"), "layout": str(layout_path),
        "storage_root": str(asset_base),
        "layout_sha256": digest(layout_path), "output": str(output_path.resolve()),
        "font_family": font_family,
        "source_overrides": source_overrides or {},
        "output_sha256": digest(output_path), "panels": records,
        "note": "Composition is a layout preview. Panel scientific approval is separate.",
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    return sidecar


def export_with_inkscape(svg: Path, formats: list[str], *, font_directory: Path | None = None) -> None:
    exe = shutil.which("inkscape.com") or shutil.which("inkscape")
    if not exe and sys.platform == "win32":
        candidate = Path(r"C:\Program Files\Inkscape\bin\inkscape.com")
        exe = str(candidate) if candidate.is_file() else None
    if not exe:
        raise RuntimeError("Inkscape is required for PNG/PDF export")
    environment = os.environ.copy()
    if font_directory:
        if not all((font_directory / name).is_file() for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf")):
            raise FileNotFoundError(f"Required figure font files are missing from {font_directory}")
        config_dir = svg.parent / ".fontconfig"
        config_dir.mkdir(parents=True, exist_ok=True)
        font_path = font_directory.resolve().as_posix()
        cache_path = (config_dir / "cache").resolve().as_posix()
        windows_fonts = "<dir>C:/Windows/Fonts</dir>" if sys.platform == "win32" else ""
        config_path = config_dir / "fonts.conf"
        config_path.write_text(
            '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd">\n'
            f'<fontconfig><dir>{escape(font_path)}</dir>{windows_fonts}'
            f'<cachedir>{escape(cache_path)}</cachedir></fontconfig>\n',
            encoding="utf-8",
        )
        environment["FONTCONFIG_FILE"] = str(config_path.resolve())
    for fmt in formats:
        target = svg.with_suffix("." + fmt)
        subprocess.run([exe, str(svg), f"--export-type={fmt}", f"--export-filename={target}"],
                       check=True, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Override the layout's SSD output destination")
    parser.add_argument("--panel", help="Render one panel on its own canvas")
    parser.add_argument("--replace-panel-source", action="append", default=[], metavar="ID=PATH",
                        help="Preview one alternate SSD source without changing the layout JSON")
    parser.add_argument("--strict", action="store_true", help="Require every panel source")
    parser.add_argument("--export", nargs="*", choices=["png", "pdf"], default=[])
    parser.add_argument("--watch", action="store_true", help="Rebuild when the layout or a source changes")
    args = parser.parse_args()
    source_overrides = {}
    for item in args.replace_panel_source:
        panel_id, separator, source = item.partition("=")
        if not separator or not panel_id or not source:
            parser.error("--replace-panel-source requires ID=PATH")
        source_overrides[panel_id] = source
    while True:
        layout = json.loads(args.layout.read_text(encoding="utf-8"))
        root = storage_root(layout, args.layout)
        if args.output:
            output_path = args.output
        elif args.panel:
            figure_prefix = safe_name(layout["figure_id"]) + "_Panel" if layout.get("figure_id") else ""
            output_path = root / "panels" / f"{figure_prefix}{safe_name(args.panel)}.svg"
        else:
            output_path = root / layout.get("output", "figure.svg")
        sidecar = render(args.layout, output_path, strict=args.strict,
                         only_panel=args.panel, source_overrides=source_overrides)
        font_directory = (
            (root / layout["font_directory"]).resolve()
            if layout.get("font_directory") else None
        )
        export_with_inkscape(output_path, args.export, font_directory=font_directory)
        print(f"Built {output_path}: " + ", ".join(f"{p['id']}={p['status']}" for p in sidecar["panels"]), flush=True)
        if not args.watch:
            break
        baseline = [(p["source"], p["sha256"]) for p in sidecar["panels"]]
        baseline_layout = sidecar["layout_sha256"]
        while True:
            time.sleep(1)
            try:
                changed = digest(args.layout) != baseline_layout or any(
                    path and ((digest(Path(path)) if Path(path).is_file() else None) != old)
                    for path, old in baseline
                )
            except OSError:
                changed = True
            if changed:
                break


if __name__ == "__main__":
    main()
