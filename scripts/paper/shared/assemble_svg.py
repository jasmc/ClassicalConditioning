"""Compose replaceable SVG panels from a declarative paper-figure layout.

The assembly is presentation only: source SVGs are embedded as vector elements,
and scientific values are never inferred from artwork. Run with --help for use.
"""

from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]


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

REPO = _COMMAND_REPOSITORY
sys.path.insert(0, str(REPO / "src"))
from classical_conditioning.artifacts import artifact_staging, publish_transaction
from classical_conditioning.external_artifacts import resolve_artifact, relocated_external_directory, joaquim_root
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, StaleFigureState, state_lock
from classical_conditioning.figures.assembly_sources import preflight, prepare_layout, object_hash
from classical_conditioning.figures.layout_frames import assembly_layout, embed_spec, assess, POLICY, COLLECTOR

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
        return relocated_external_directory(Path(configured).expanduser()).resolve()
    return layout_path.resolve().parent


def panel_layout(layout: dict, panel_id: str) -> dict:
    """Use the same standalone geometry in preflight and composition."""
    return prepare_layout(layout, only_panel=panel_id)[1]


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


def _panel_payload_hash(group):
    canonical = ET.canonicalize(ET.tostring(group,encoding='unicode'),rewrite_prefixes=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _clip_payload_hash(root, panel_id):
    clip = root.find(f"./{q('defs')}/{q('clipPath')}[@id='clip-{panel_id}']")
    return _panel_payload_hash(clip) if clip is not None else None


def _shared_payload_hash(root):
    shared = copy.deepcopy(root)
    for node in list(shared):
        if node.tag==q('g') and node.get('id','').startswith('panel-'):
            shared.remove(node)
        elif node.tag==q('defs'):
            for child in list(node):
                if child.tag==q('clipPath') and child.get('id','').startswith('clip-'):
                    node.remove(child)
            if not len(node): shared.remove(node)
    return _panel_payload_hash(shared)


def _compose(layout_path, layout, inspection):
    """Build in memory from inspected source identities; publication is separate."""
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
         **({"id": "figure-background"} if layout.get("layout_frames") else {}), "x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "white",
    })
    seen: set[str] = set()
    records = []
    inspected = {item["id"]: item for item in inspection["panels"]}
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
        source_info = inspected[panel["id"]]
        source_path = Path(source_info["source"]["path"]) if source_info["source"]["path"] else None
        if source_path and source_path.is_file():
            if panel.get("frozen_sha256") and digest(source_path) != panel["frozen_sha256"]:
                raise ValueError(f"Frozen panel {panel_id} differs from its approved SVG")
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
            if source_info['source']['status'] == 'ready':
                raise StaleFigureState(f"Verified panel source disappeared: {panel_id}")
            ET.SubElement(group, q("rect"), {
                "x": str(x), "y": str(y), "width": str(w), "height": str(h),
                "fill": "#fafafa", "stroke": "#b8b8b8", "stroke-width": "1.5",
                "stroke-dasharray": "7 5",
            })
            lines = panel.get("placeholder_lines", [panel.get("placeholder", f"{panel_id}: awaiting selected panel")])
            line_height = panel.get("placeholder_line_height", 32)
            placeholder_text = ET.SubElement(group, q("text"), {
                "x": str(x + w / 2), "y": str(y + h / 2 - (len(lines) - 1) * line_height / 2),
                "text-anchor": "middle", "font-family": font_family,
                "font-size": str(panel.get("placeholder_font_size", 22)), "fill": "#666",
            })
            for index, line in enumerate(lines):
                ET.SubElement(placeholder_text, q("tspan"), {
                    "x": str(x + w / 2), "dy": "0" if index == 0 else str(line_height),
                }).text = line
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
            "panel_payload_sha256": _panel_payload_hash(group),
            "clip_payload_sha256": _clip_payload_hash(root,panel_id),
            "original_source": source_info["original_source"],
            "replacement": source_info["replacement"],
            **{key: panel[key] for key in ("selection_status", "pending_inputs", "scientific_definition", "source_provenance") if key in panel},
        })
    for index, annotation in enumerate(layout.get("annotations", [])):
        ET.SubElement(root, q("text"), {
            "id": f"annotation-{index}", "x": str(annotation["x"]), "y": str(annotation["y"]),
            "font-family": font_family, "font-size": str(annotation.get("size", 22)),
            "font-weight": annotation.get("weight", "normal"),
            "text-anchor": annotation.get("anchor", "start"), "fill": annotation.get("fill", "#333"),
        }).text = annotation["text"]
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    sidecar = {
        "schema_version": 2, "title": layout.get("title"), "layout": str(layout_path),
        "storage_root": str(asset_base), "layout_sha256": inspection["layout_sha256"],
        "font_family": font_family, "panels": records,
        "rendering_context_sha256": _shared_payload_hash(root),
        "rendering_context_scope": "shared-svg-v1",
        "effective_layout_sha256": object_hash(layout),
        "note": "Composition is an unfrozen layout preview; overrides are candidates, not approved scientific selections.",
        **{key: layout[key] for key in ("baseline_s", "baseline_interval", "metric_id", "scientific_status", "interpretation", "inventory_manifest") if key in layout},
    }
    return payload, sidecar


def _destination(output_path, inputs, *, suffix=".svg", protect_existing=True):
    logical = str(Path(output_path).expanduser()).replace('\\', '/')
    if logical.startswith('/Volumes/') and not logical.startswith('/Volumes/JOAQUIM/'):
        mount = Path('/Volumes') / logical.split('/')[2]
        if not os.path.ismount(mount):
            raise FileNotFoundError(f'External destination unavailable: {mount}; no mount will be fabricated')
    output = relocated_external_directory(Path(output_path).expanduser()).resolve()
    volume = None
    def external_scope_applies(value):
        nonlocal volume
        text = str(value).replace('\\', '/')
        if text.startswith('/Volumes/JOAQUIM/') or text.upper().startswith('J:/'):
            if volume is None:
                try:
                    volume = joaquim_root().resolve()
                except FileNotFoundError:
                    return False
            return output.is_relative_to(volume)
        return True
    for manifest in (REPO/'records/repository-archive').glob('**/transfer-inventory.json'):
        archive = json.loads(manifest.read_text(encoding='utf-8')).get('archive_root')
        if archive and not external_scope_applies(archive):
            continue
        try:
            archive_path = relocated_external_directory(archive).resolve() if archive else None
        except FileNotFoundError:
            archive_path = None
        if archive_path in (output, *output.parents):
            raise ValueError('Published archives are immutable; choose a candidate destination')
    if output.suffix.lower() != suffix:
        raise ValueError("Assembly output must be an editable .svg file")
    if output == REPO or REPO in output.parents:
        raise ValueError("Permanent assembly outputs must be outside the repository")
    if protect_existing and output.is_file():
        current_hash = digest(output)
        definitions = [*(REPO/'configs/paper-figures/selections').glob('*.json'),
                       *(REPO/'records/frozen-analyses').glob('**/manifests/freeze.json')]
        def hashes(value):
            if isinstance(value, dict):
                for key,item in value.items():
                    if 'sha256' in key.lower() and isinstance(item,str): yield item.lower()
                    yield from hashes(item)
            elif isinstance(value,list):
                for item in value: yield from hashes(item)
        if any(current_hash in set(hashes(json.loads(p.read_text(encoding='utf-8')))) for p in definitions):
            raise ValueError('Existing output matches accepted frozen bytes; create a separate candidate')
    if output in {Path(p).resolve() for p in inputs} or output.with_suffix(output.suffix+'.json') in {Path(p).resolve() for p in inputs}:
        raise ValueError("Assembly output would overwrite a bound input")
    if str(output).startswith('/Volumes/'):
        mount = Path('/Volumes') / output.parts[2]
        if not os.path.ismount(mount):
            raise FileNotFoundError(f"External destination unavailable: {mount}; no mount will be fabricated")
    # Current figure-state artifacts have a separate guarded publication protocol.
    for state in (REPO/'docs/paper/figures').glob('figure-*/current-state.json'):
        def paths(value):
            if isinstance(value, dict):
                if isinstance(value.get('path'), str):
                    if not external_scope_applies(value['path']):
                        return
                    try:
                        yield relocated_external_directory(Path(value['path']).expanduser()).resolve()
                    except FileNotFoundError:
                        yield Path(value['path']).expanduser().resolve()
                for v in value.values(): yield from paths(v)
            elif isinstance(value, list):
                for v in value: yield from paths(v)
        if output in set(paths(json.loads(state.read_text(encoding='utf-8')))):
            raise ValueError("Output is managed by a current figure state; use the guarded review/working-copy workflow or a new candidate destination")
    return output


def _baseline(path):
    metadata = resolve_artifact(path)
    result = json.loads(metadata.read_text(encoding='utf-8'))
    if not isinstance(result.get('panels'), list) or not result.get('output_sha256') or not result.get('output'):
        raise ValueError("Baseline needs a recorded assembly and panel provenance")
    image = resolve_artifact(result['output'], expected_sha256=result['output_sha256'])
    root = ET.parse(image).getroot()
    if root.tag != q('svg'):
        raise ValueError("Baseline output is not SVG")
    groups = {node.get('id'):node for node in root if node.tag==q('g')}
    for panel in result['panels']:
        if 'panel_payload_sha256' in panel:
            group = groups.get('panel-'+panel['id'])
            if group is None or _panel_payload_hash(group)!=panel['panel_payload_sha256']:
                raise ValueError('Baseline panel provenance does not match saved artwork: '+panel['id'])
        if 'clip_payload_sha256' in panel and _clip_payload_hash(root,panel['id'])!=panel['clip_payload_sha256']:
            raise ValueError('Baseline clipping provenance does not match artwork: '+panel['id'])
    if result.get('rendering_context_scope')=='shared-svg-v1' and _shared_payload_hash(root)!=result.get('rendering_context_sha256'):
        raise ValueError('Baseline shared rendering context does not match artwork')
    ids = [p['id'] for p in result['panels']]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate baseline panel ID")
    return result, metadata, image


def _comparison(baseline, candidate, replaced, *, enforce):
    before = {p['id']: p for p in baseline['panels']}; after = {p['id']: p for p in candidate['panels']}
    records = []
    context_same = (baseline.get('rendering_context_sha256') == candidate['rendering_context_sha256']
                    if baseline.get('rendering_context_sha256') and baseline.get('rendering_context_scope')==candidate.get('rendering_context_scope') else None)
    for pid in sorted(set(before) | set(after)):
        old, new = before.get(pid), after.get(pid)
        source_same = bool(old and new and old.get('sha256') == new.get('sha256'))
        geometry_same = bool(old and new and old.get('box') == new.get('box') and old.get('content_box') == new.get('content_box'))
        payload_same = (old['panel_payload_sha256'] == new.get('panel_payload_sha256')
                        if old and 'panel_payload_sha256' in old and new else None)
        clip_same = (old['clip_payload_sha256']==new.get('clip_payload_sha256')
                     if old and 'clip_payload_sha256' in old and new and 'clip_payload_sha256' in new else None)
        unchanged = source_same and geometry_same and payload_same is not False and clip_same is not False and context_same is not False
        if enforce and pid not in replaced and not unchanged:
            raise ValueError(f"Unrequested panel changed relative to baseline: {pid}")
        records.append(dict(id=pid, requested_replacement=pid in replaced, source_unchanged=source_same,
                            geometry_unchanged=geometry_same, embedded_payload_unchanged=payload_same, clipping_unchanged=clip_same,
                            unchanged=unchanged, before_sha256=old.get('sha256') if old else None,
                            after_sha256=new.get('sha256') if new else None))
    return dict(panels=records, rendering_context_unchanged=context_same,
                payload_check='compared where baseline records panel payloads; older sidecars support source/geometry checks only')


def render(layout_path: Path, output_path: Path, *, strict: bool = False,
           only_panel: str | None = None, source_overrides: dict[str, str] | None = None,
           baseline_manifest: Path | None = None, overwrite: bool = True,
           export_formats: tuple[str, ...] = (), font_directory: Path | None = None) -> dict:
    if len(export_formats) != len(set(export_formats)) or set(export_formats)-{'png','pdf'}:
        raise ValueError('Explicit exports support distinct png/pdf formats only')
    inspection = preflight(layout_path, strict=strict, only_panel=only_panel, source_overrides=source_overrides)
    if not inspection['can_render']:
        message = '; '.join(inspection['errors'])
        if any(p['status']=='blocked' and any(s['status'] in {'unavailable', 'unset'} for s in (p['source'],p['original_source'])) for p in inspection['panels']):
            raise FileNotFoundError(message)
        raise ValueError(message)
    layout_path = Path(layout_path).resolve()
    _, layout = prepare_layout(json.loads(layout_path.read_text(encoding='utf-8')), source_overrides, only_panel)
    inputs = [layout_path, Path(__file__), *[REPO/'src/classical_conditioning'/name for name in ('figures/assembly_sources.py','external_artifacts.py','artifacts.py','figure_workspace.py')]]
    for panel in inspection['panels']:
        inputs.extend(Path(s['path']) for s in (panel['source'], panel['original_source']) if s.get('path'))
    if font_directory and export_formats:
        inputs.extend(Path(font_directory)/name for name in ('DejaVuSans.ttf','DejaVuSans-Bold.ttf'))
    if layout.get("layout_frames") is not None:
        inputs.extend([POLICY, COLLECTOR, REPO/"src/classical_conditioning/figures/layout_frames.py"])
    output = _destination(output_path, inputs)
    sidecar_path = _destination(output.with_suffix(output.suffix+'.json'), inputs, suffix='.json')
    extra_paths = {fmt:_destination(output.with_suffix('.'+fmt),inputs,suffix='.'+fmt) for fmt in export_formats}
    previous = None; baseline_paths = []
    if output.exists() != sidecar_path.exists():
        raise ValueError("Existing assembly/provenance pair is incomplete; preserve it and use a new destination")
    if output.exists():
        if not overwrite: raise FileExistsError(output)
        previous, metadata, image = _baseline(sidecar_path)
        if image.resolve() != output: raise ValueError("Existing sidecar identifies a different assembly output")
        baseline_paths.extend((metadata, image))
    if baseline_manifest is not None:
        previous, metadata, image = _baseline(baseline_manifest); baseline_paths.extend((metadata, image))
    for fmt,path in extra_paths.items():
        if path.exists():
            recorded = next((e for e in (previous or {}).get('extra_exports',[]) if Path(e['path']).resolve()==path), None)
            if not overwrite or recorded is None or digest(path)!=recorded['sha256']:
                raise ValueError('Existing extra export is unverified or changed: '+str(path))
    snapshot = Snapshot([*command_sources(__file__), *([*inputs, *baseline_paths, output, sidecar_path, *extra_paths.values()])])
    if snapshot.hashes[str(layout_path)] != inspection['layout_sha256']:
        raise StaleFigureState('Layout changed after preflight')
    for panel in inspection['panels']:
        for record in (panel['source'], panel['original_source']):
            if record.get('path') and snapshot.hashes[str(Path(record['path']).resolve())] != record['sha256']:
                raise StaleFigureState('Source changed after preflight: '+record['path'])
    payload, result = _compose(layout_path, layout, inspection)
    if layout.get("layout_frames") is not None:
        _, frame_spec = assembly_layout(layout)
        payload = embed_spec(payload, frame_spec)
        result["layout_frames"] = assess(payload)
        result["rendering_context_sha256"] = _shared_payload_hash(ET.fromstring(payload))
    result.update(output=str(output), output_sha256=hashlib.sha256(payload).hexdigest(),
                  source_overrides=source_overrides or {}, preflight=inspection,
                  renderer={'path':str(Path(__file__).resolve()),'sha256':digest(Path(__file__))})
    command = [sys.executable, str(Path(__file__).resolve()), str(layout_path), '--output', str(output)]
    if strict: command += ['--strict']
    if only_panel: command += ['--panel', only_panel]
    for pid, reference in (source_overrides or {}).items(): command += ['--replace-panel-source', pid+'='+str(reference)]
    if overwrite: command += ['--overwrite']
    if baseline_manifest is not None: command += ['--baseline-manifest', str(baseline_manifest)]
    if export_formats: command += ['--export',*export_formats]
    result['reproduction_command'] = command
    result['bound_inputs'] = {str(Path(path).resolve()): snapshot.hashes[str(Path(path).resolve())] for path in inputs}
    if previous is not None:
        result['comparison'] = _comparison(previous, result, {safe_name(pid) for pid in (source_overrides or {})},
                                            enforce=bool(source_overrides) or baseline_manifest is not None)
        result['comparison']['baseline_sha256'] = digest(metadata)
        result['comparison']['baseline_manifest'] = str(metadata)
        result['comparison']['baseline_output_sha256'] = previous['output_sha256']
        result['comparison']['baseline_reference_retained'] = metadata.resolve() != sidecar_path
        result['comparison']['baseline_panel_identities'] = [{key: p.get(key) for key in ('id','sha256','panel_payload_sha256','clip_payload_sha256','box','content_box')} for p in previous['panels']]
    else:
        result['comparison'] = {'status':'no prior assembly baseline; original source provenance retained'}
    snapshot.check()
    # No competing current-state pointer: publish a verified artifact pair only.
    with artifact_staging(output.parent, prefix='.assembly-') as stage:
        staged_svg = stage/output.name; staged_json = stage/sidecar_path.name
        staged_svg.write_bytes(payload)
        staged_extras = []
        result['extra_exports'] = []
        if export_formats:
            export_with_inkscape(staged_svg,list(export_formats),font_directory=font_directory)
            for fmt,path in extra_paths.items():
                staged = staged_svg.with_suffix('.'+fmt)
                result['extra_exports'].append({'format':fmt,'path':str(path),'sha256':digest(staged)})
                staged_extras.append((staged,path))
        result['unmodified_prior_exports'] = [e for e in (previous or {}).get('extra_exports',[]) if e['format'] not in export_formats]
        staged_json.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        units = ((staged_svg,output), *staged_extras, (staged_json,sidecar_path))
        expected = {destination:digest(staged) for staged,destination in units}
        def verify_saved():
            for path, value in expected.items():
                _destination(path, inputs, suffix=path.suffix, protect_existing=False)
                if digest(path) != value: raise ValueError('Saved assembly hash mismatch: '+str(path))
            snapshot.check(ignore=expected)
        with state_lock(sidecar_path):
            snapshot.check()
            for destination in expected:
                _destination(destination, inputs, suffix=destination.suffix)
            publish_transaction(units,stage,
                                overwrite=overwrite,verify_published=verify_saved)
    return result


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
    parser.add_argument("--plan", action="store_true", help="Check sources/hashes/geometry without rendering or writing")
    parser.add_argument("--baseline-manifest", type=Path, help="Verified assembly sidecar for replacement protection")
    parser.add_argument("--overwrite", action="store_true", help="Replace a verified existing candidate pair under snapshot/lock protection")
    parser.add_argument("--export", nargs="*", choices=["png", "pdf"], default=[])
    parser.add_argument("--watch", action="store_true", help="Rebuild when the layout or a source changes")
    args = parser.parse_args()
    if args.plan and (args.watch or args.export): parser.error("--plan cannot export or watch")
    if not args.plan and not args.output: parser.error("Provide an explicit external --output destination")
    source_overrides = {}
    for item in args.replace_panel_source:
        panel_id, separator, source = item.partition("=")
        if not separator or not panel_id or not source:
            parser.error("--replace-panel-source requires ID=PATH")
        if panel_id in source_overrides: parser.error('Duplicate panel source override: '+panel_id)
        source_overrides[panel_id] = source
    if args.plan:
        report = preflight(args.layout, strict=args.strict, only_panel=args.panel, source_overrides=source_overrides)
        try:
            if args.output:
                sources = [s['path'] for p in report['panels'] for s in (p['source'],p['original_source']) if s.get('path')]
                report['output'] = str(_destination(args.output,sources))
            if args.baseline_manifest and report['can_render']:
                baseline, metadata, _ = _baseline(args.baseline_manifest)
                _, prepared = prepare_layout(json.loads(args.layout.read_text(encoding='utf-8')),source_overrides,args.panel)
                candidate = {'panels':[{'id':safe_name(p['id']),'sha256':p['source']['sha256'],'box':p['box'],
                                       'content_box':next(x for x in prepared['panels'] if x['id']==p['id']).get('content_box',p['box'])}
                                      for p in report['panels']],
                             'rendering_context_sha256':object_hash({'canvas':prepared['canvas'],
                               'font_family':prepared.get('font_family','DejaVu Sans'),
                               'annotations':prepared.get('annotations',[])})}
                report['comparison'] = _comparison(baseline,candidate,{safe_name(pid) for pid in source_overrides},enforce=True)
                report['comparison']['check_scope'] = 'source identities and geometry only; embedded payload compared when rendering'
                report['comparison']['baseline_sha256'] = digest(metadata)
        except (ValueError,FileNotFoundError) as error:
            report['can_render']=False; report['status']='blocked';report['errors'].append(str(error))
        print(json.dumps(report, indent=2, allow_nan=False))
        raise SystemExit(0 if report['can_render'] else 1)
    built = False
    while True:
        layout = json.loads(args.layout.read_text(encoding="utf-8"))
        root = storage_root(layout, args.layout)
        output_path = args.output
        font_directory = None
        if args.export and layout.get('font_directory'):
            from classical_conditioning.figures.assembly_sources import _source_reference
            font_directory = resolve_artifact(_source_reference(layout,args.layout,layout['font_directory']+'/DejaVuSans.ttf')).parent
        sidecar = render(args.layout, output_path, strict=args.strict,
                         only_panel=args.panel, source_overrides=source_overrides,
                         baseline_manifest=args.baseline_manifest, overwrite=args.overwrite or (args.watch and built),
                         export_formats=tuple(args.export),font_directory=font_directory)
        built = True
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
