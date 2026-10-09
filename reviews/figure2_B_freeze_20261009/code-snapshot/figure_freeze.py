"""Explicit freeze-time checks; ordinary rendering never invokes this module.

The input is a proposed freeze manifest, not an assertion of compliance. SVG
IDs, effective presentation, protected geometry and artifact bytes are checked
before an immutable manifest is published. No scientific roles are inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
import xml.etree.ElementTree as ET


DEFAULT_SPECIFICATION = Path(__file__).resolve().parents[2] / "configs/paper-figures/figure-elements.json"
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
DRAWABLE = {"path", "line", "polyline", "polygon", "rect", "circle", "ellipse", "text", "image", "use"}
PROTECTED_ATTRIBUTES = {"d", "points", "transform", "x", "y", "cx", "cy", "r", "rx", "ry", "width", "height", "x1", "x2", "y1", "y2", "href"}


def _read_json(value: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = item
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(value, object_pairs_hook=unique, parse_constant=nonfinite)


def _passed_review(value) -> bool:
    return isinstance(value, dict) and value.get("status") == "passed" and bool(value.get("evidence"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _tag(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _multiply(a: tuple, b: tuple) -> tuple:
    return (a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1],
            a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3],
            a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5])


def _transform(value: str) -> tuple:
    matrix = (1., 0., 0., 1., 0., 0.)
    remaining = re.sub(r"([a-zA-Z]+)\s*\(([^)]*)\)", "", value).strip(" ,\t\n")
    if remaining:
        raise ValueError(f"Unrecognized SVG transform: {value}")
    for kind, arguments in re.findall(r"([a-zA-Z]+)\s*\(([^)]*)\)", value):
        numbers = [float(v) for v in re.findall(NUMBER, arguments)]
        if kind == "matrix" and len(numbers) == 6:
            change = tuple(numbers)
        elif kind == "translate" and len(numbers) in (1, 2):
            change = (1, 0, 0, 1, numbers[0], numbers[1] if len(numbers) == 2 else 0)
        elif kind == "scale" and len(numbers) in (1, 2):
            change = (numbers[0], 0, 0, numbers[-1], 0, 0)
        elif kind == "rotate" and len(numbers) in (1, 3):
            angle = math.radians(numbers[0]); c, s = math.cos(angle), math.sin(angle)
            change = (c, s, -s, c, 0, 0)
            if len(numbers) == 3:
                x, y = numbers[1:]
                change = _multiply(_multiply((1, 0, 0, 1, x, y), change), (1, 0, 0, 1, -x, -y))
        else:
            raise ValueError(f"Unsupported SVG transform: {kind}")
        matrix = _multiply(matrix, change)
    return matrix


def _length(value: str) -> float:
    match = re.fullmatch(rf"\s*({NUMBER})(px|pt|mm|cm|in)?\s*", value)
    if not match:
        raise ValueError(f"Unresolved SVG length: {value}")
    factor = {None: 1., "px": 1., "pt": 96/72, "mm": 96/25.4, "cm": 96/2.54, "in": 96.}
    return float(match[1]) * factor[match[2]]


def _properties(node: ET.Element) -> dict:
    result = dict(node.attrib)
    for item in node.get("style", "").split(";"):
        if ":" in item:
            key, value = item.split(":", 1); result[key.strip()] = value.strip()
    # Matplotlib serializes text through a CSS font shorthand.
    font = result.get("font", "")
    match = re.search(rf"({NUMBER}(?:px|pt|mm))\s+(.+)$", font)
    if match:
        result.setdefault("font-size", match[1]); result.setdefault("font-family", match[2].strip())
        result.setdefault("font-weight", "bold" if re.search(r"\bbold\b|\b700\b", font[:match.start()]) else "normal")
        result.setdefault("font-style", "italic" if "italic" in font[:match.start()] else "normal")
    return result


def _paint(value: str) -> tuple[str, float]:
    value = value.strip().lower()
    if value in {"black", "white"}:
        return {"black": "#000000", "white": "#ffffff"}[value], 1.
    if re.fullmatch(r"#[0-9a-f]{3}", value):
        return "#" + "".join(c*2 for c in value[1:]), 1.
    if re.fullmatch(r"#[0-9a-f]{6}", value):
        return value, 1.
    if re.fullmatch(r"#[0-9a-f]{8}", value):
        return value[:7], int(value[7:], 16)/255
    match = re.fullmatch(r"rgba?\(([^)]+)\)", value)
    if match:
        parts = [float(v.strip()) for v in match[1].split(",")]
        return "#" + "".join(f"{round(v):02x}" for v in parts[:3]), parts[3] if len(parts) == 4 else 1.
    raise ValueError(f"Unresolved SVG color: {value}")


class SvgEvidence:
    def __init__(self, path: Path, final_width_mm: float):
        self.root = ET.parse(path).getroot()
        self.nodes = list(self.root.iter()); self.parent = {c: p for p in self.nodes for c in p}
        self.ids = {}
        for node in self.nodes:
            if node.get("id"):
                if node.get("id") in self.ids:
                    raise ValueError(f"Duplicate SVG ID: {node.get('id')}")
                self.ids[node.get("id")] = node
            if _tag(node) in {"script", "foreignObject"}:
                raise ValueError("SVG contains active content")
            for key, value in node.attrib.items():
                name = key.rsplit("}", 1)[-1]
                if name.lower().startswith("on") or (name in {"href", "src"} and not value.startswith(("#", "data:image/png;base64,", "data:image/jpeg;base64,"))):
                    raise ValueError("SVG contains active or external references")
            if _tag(node) == "style" and node.text and not re.fullmatch(r"\s*\*\s*\{\s*stroke-linejoin:\s*round;\s*stroke-linecap:\s*butt\s*\}\s*", node.text):
                raise ValueError("Stylesheet rules need resolution to inline presentation before freeze")
        box = [float(v) for v in re.split(r"[ ,]+", self.root.get("viewBox", "").strip())]
        if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
            raise ValueError("SVG requires a positive viewBox for final-size verification")
        self.root_scale = final_width_mm*72/25.4/box[2]

    def chain(self, node):
        result = [node]
        while node in self.parent:
            node = self.parent[node]; result.append(node)
        return result[::-1]

    def presentation(self, node):
        properties = {"fill": "black", "stroke": "none", "stroke-width": "1", "font-weight": "normal", "font-style": "normal"}
        opacity = 1.; matrix = (1., 0., 0., 1., 0., 0.); display_none = False
        for ancestor in self.chain(node):
            local = _properties(ancestor)
            if local.get("display") == "none":
                display_none = True
            if "transform" in local and local["transform"] != ancestor.get("transform"):
                raise ValueError("CSS transforms must be resolved into SVG transform attributes")
            for unsupported in ("mask", "filter", "mix-blend-mode", "vector-effect"):
                if local.get(unsupported, "none") not in {"none", "normal"}:
                    raise ValueError(f"Unsupported SVG presentation requires resolution: {unsupported}")
            if _tag(ancestor) == "use":
                target = self.ids.get(ancestor.get("href", ancestor.get("{http://www.w3.org/1999/xlink}href", "")).removeprefix("#"))
                if target is None:
                    raise ValueError("Broken SVG use reference")
                target_props = _properties(target)
                if _tag(target) not in DRAWABLE or target.get("transform"):
                    raise ValueError("SVG use needs a simple untransformed drawable template")
                for key in ("fill", "stroke", "stroke-width", "fill-opacity", "stroke-opacity", "opacity"):
                    if key in target_props and key in local and target_props[key] != local[key]:
                        raise ValueError("Conflicting SVG use/template presentation needs resolution")
                local = {**target_props, **local}
            opacity *= float(local.get("opacity", 1))
            properties.update(local)
            matrix = _multiply(matrix, _transform(ancestor.get("transform", "")))
            if ancestor is not self.root and _tag(ancestor) == "svg":
                vb = [float(v) for v in re.split(r"[ ,]+", ancestor.get("viewBox", "").strip())]
                if len(vb) != 4 or min(vb[2:]) <= 0:
                    raise ValueError("Nested SVG needs a positive viewBox")
                sx = _length(ancestor.get("width", ""))/vb[2]; sy = _length(ancestor.get("height", ""))/vb[3]
                if ancestor.get("preserveAspectRatio", "xMidYMid meet") != "none":
                    sx = sy = max(sx, sy) if "slice" in ancestor.get("preserveAspectRatio", "") else min(sx, sy)
                matrix = _multiply(matrix, (sx, 0, 0, sy, 0, 0))
        sx, sy = math.hypot(matrix[0], matrix[1]), math.hypot(matrix[2], matrix[3])
        if not math.isclose(sx, sy, rel_tol=1e-6) or not math.isclose(matrix[0]*matrix[2]+matrix[1]*matrix[3], 0, abs_tol=1e-6):
            raise ValueError("Nonuniform/sheared assembly scale requires resolution")
        if display_none:
            properties["display"] = "none"
        return properties, opacity, self.root_scale*sx

    def leaves(self, node):
        return [n for n in node.iter() if _tag(n) in DRAWABLE and not any(_tag(p) in {"defs", "clipPath", "mask"} for p in self.chain(n))]

    def geometry(self, node):
        def attributes(n):
            result = []
            for key, value in n.attrib.items():
                name = key.rsplit("}", 1)[-1]
                if name not in PROTECTED_ATTRIBUTES and name not in {"viewBox", "preserveAspectRatio", "clip-path"}:
                    continue
                if (name == "href" and value.startswith("#")) or (name == "clip-path" and value.startswith("url(#")):
                    value = "internal-resource"  # Compare referenced geometry, not generated resource IDs.
                result.append((name, value))
            return sorted(result)

        def clip_reference(n):
            value = _properties(n).get("clip-path")
            return "internal-resource" if value and value.startswith("url(#") else value

        # Ancestor transforms/clipping can move scientific marks without changing
        # any attribute on the registered artist itself.
        ancestors = [(_tag(n), attributes(n), clip_reference(n))
                     for n in self.chain(node)[:-1] if attributes(n) or _properties(n).get("clip-path")]
        descendants = [(_tag(n), attributes(n), clip_reference(n))
                       for n in node.iter() if _tag(n) in DRAWABLE or n.get("transform")]
        resources = []
        for n in self.chain(node)[:-1] + list(node.iter()):
            href = n.get("href", n.get("{http://www.w3.org/1999/xlink}href", ""))
            if href.startswith("#"):
                target = self.ids.get(href[1:])
                if target is None:
                    raise ValueError("Protected geometry contains a broken resource reference")
                resources.append([(_tag(part), attributes(part)) for part in target.iter()])
            clip = _properties(n).get("clip-path", "")
            if clip.startswith("url(#"):
                target = self.ids.get(clip[5:-1])
                if target is None:
                    raise ValueError("Protected geometry contains a broken clip reference")
                resources.append([(_tag(part), attributes(part)) for part in target.iter()])
        return ancestors, descendants, resources

    def tick_anchors(self, node):
        result = []
        for leaf in self.leaves(node):
            target = leaf
            if _tag(leaf) == "use":
                href = leaf.get("href", leaf.get("{http://www.w3.org/1999/xlink}href", ""))
                target = self.ids.get(href.removeprefix("#"))
            if target is None:
                raise ValueError("Tick anchor has no source template")
            coordinates = [float(v) for v in re.findall(NUMBER, target.get("d", ""))]
            if len(coordinates) != 4:
                raise ValueError("Tick anchor needs a simple source line template")
            chain = [(_tag(part), sorted((key, value) for key, value in part.attrib.items()
                      if key in (PROTECTED_ATTRIBUTES-{"d", "href"}) | {"viewBox", "preserveAspectRatio"}))
                     for part in self.chain(leaf)]
            result.append((chain, coordinates[:2]))
        return result


def _check_candidate(candidate_path: Path, specification_path: Path) -> dict:
    """Return all unresolved checks without editing the candidate or artifacts."""
    issues = []; candidate_path = candidate_path.resolve(); specification_path = specification_path.resolve()
    def issue(code, message, element_id=None):
        issues.append({"code": code, "element_id": element_id, "message": message})
    try:
        candidate_bytes = candidate_path.read_bytes(); candidate = _read_json(candidate_bytes)
        specification = _read_json(specification_path.read_bytes())
        if not isinstance(candidate, dict):
            raise ValueError("Candidate must be a JSON object")
    except (OSError, ValueError) as error:
        return {"valid": False, "issues": [{"code": "input", "message": str(error)}]}
    for field in ["figure_id", "panel_ids", "selection_record", "assembly_scale", "element_registry", "approved_exceptions", "source_artifacts", "data_artifacts", "exports", "verification", "freeze_authorization"]:
        if not candidate.get(field) and field != "approved_exceptions":
            issue("missing-field", f"Missing/nonempty required candidate field: {field}")
        elif field == "approved_exceptions" and field not in candidate:
            issue("missing-field", "Declare approved_exceptions, including an empty list when none")
    visual_review = candidate.get("verification", {}).get("visual_review", {})
    if not _passed_review(visual_review):
        issue("visual-review", "Record a completed visual review and its evidence; code does not infer visual approval")
    for name in ("scientific_mapping_review", "structure_review"):
        if not _passed_review(candidate.get("verification", {}).get(name)):
            issue("review", f"Record completed {name} with evidence; SVG alone cannot validate scientific definitions/plot-family structure")
    def artifact_path(record):
        path = Path(record["path"]); return (candidate_path.parent/path).resolve() if not path.is_absolute() else path.resolve()
    artifacts = []; svg = None; original = None
    for group in ["source_artifacts", "data_artifacts", "exports"]:
        for record in candidate.get(group, []):
            try:
                path = artifact_path(record)
                if not re.fullmatch(r"[a-fA-F0-9]{64}", record.get("sha256", "")) or digest(path).lower() != record["sha256"].lower():
                    raise ValueError(f"Artifact hash mismatch: {path}")
                artifacts.append({"path": str(path), "sha256": record["sha256"].lower()})
                if group == "exports" and path.suffix.lower() == ".svg":
                    if svg is not None:
                        raise ValueError("Freeze one assembled SVG per candidate; freeze panels separately or assemble first")
                    svg = path
                if group == "source_artifacts" and record.get("kind") == "original_svg":
                    if original is not None:
                        raise ValueError("Declare one original_svg for protected-geometry comparison")
                    original = path
            except (OSError, KeyError, TypeError, ValueError) as error:
                issue("artifact", str(error))
    evidence = baseline = None
    try:
        width = float(candidate.get("assembly_scale", {}).get("final_width_mm", 0))
        if width <= 0 or not math.isfinite(width):
            raise ValueError("Positive final_width_mm required")
        if "source_to_final_transforms" not in candidate.get("assembly_scale", {}):
            raise ValueError("Record source_to_final_transforms, including identity if unscaled")
        if svg is None or original is None:
            raise ValueError("A hashed SVG export and an original_svg source are required")
        evidence = SvgEvidence(svg, width); baseline = SvgEvidence(original, width)
        transforms = candidate["assembly_scale"]["source_to_final_transforms"]
        measured_transforms = [{"element_id": node.get("id"), "transform": node.get("transform")} for node in evidence.nodes if node.get("transform")]
        if not isinstance(transforms, dict) or transforms.get("svg_transforms") != measured_transforms or not math.isclose(float(transforms.get("root_user_unit_to_final_pt", 0)), evidence.root_scale, rel_tol=1e-6):
            raise ValueError("Recorded assembly transforms do not match the exported SVG")
    except (OSError, ValueError, ET.ParseError) as error:
        issue("svg-evidence", str(error))
    exceptions = candidate.get("approved_exceptions", [])
    exception_ids = set()
    for exception in exceptions:
        missing = [k for k in specification["exception_record_contract"]["required_fields"] if k not in exception or exception[k] in (None, "", {})]
        if missing or exception.get("exception_id") in exception_ids:
            issue("exception", f"Incomplete/duplicate exception: {missing}")
        exception_ids.add(exception.get("exception_id"))
        role = specification["roles"].get(exception.get("scientific_role"), {})
        if exception.get("property") not in specification["styles"].get(role.get("style"), {}):
            issue("exception", "Exception property is not a presentation default for its role")
        scope = exception.get("scope", {})
        if scope.get("figure_id") != candidate.get("figure_id") or not scope.get("panel_ids") or not set(scope.get("panel_ids", [])) <= set(candidate.get("panel_ids", [])):
            issue("exception", "Exception must have explicit figure/panel scope within this candidate")
    records = candidate.get("element_registry", {})
    if not isinstance(records, dict):
        issue("registry", "element_registry must be keyed by unique element_id"); records = {}
    covered = set(); data_order = []; reference_order = []
    for element_id, record in records.items():
        try:
            contract = specification["element_record_contract"]
            missing = (set(contract["required_fields"]) | set(contract.get("freeze_required_fields", [])))-set(record)
            if missing or record.get("element_id") != element_id:
                raise ValueError(f"Incomplete element record: {sorted(missing)}")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", element_id):
                raise ValueError("Element ID must be XML-safe")
            role = record["scientific_role"]; definition = specification["roles"].get(role)
            if definition is None or record["artist_type"] not in definition["artist_types"]:
                raise ValueError("Unknown role or incompatible artist_type")
            if record["figure_id"] != candidate.get("figure_id") or record["panel_id"] not in candidate.get("panel_ids", []):
                raise ValueError("Element scope does not match candidate")
            if record.get("classification_confidence") not in {"explicit", "high"} or not record.get("classification_evidence"):
                raise ValueError("Explicit/high classification with evidence is required; unknowns remain unresolved")
            if record.get("protection") not in {"presentation", "annotation", "scientific-text", "axis-definition", "data-geometry", "resource"}:
                raise ValueError("Declare an explicit protection class")
            scientific_geometry = role.startswith(("trace.", "trajectory.", "observation.", "summary.", "uncertainty.", "heatmap.", "stimulus.", "reference.")) or role in {"baseline.window", "phase.separator.heatmap", "phase.boundary", "annotation.example_trial"}
            if scientific_geometry and record["protection"] != "data-geometry":
                raise ValueError("Scientific data/event geometry cannot be assigned weaker presentation protection")
            protected = scientific_geometry or record["protection"] == "data-geometry"
            scientific_text = role in {"phase.label", "label.fish_id", "title.condition", "axis.label", "axis.tick_label", "annotation.statistical_label"}
            if scientific_text and record["protection"] not in {"scientific-text", "axis-definition"}:
                raise ValueError("Scientific labels require scientific-text/axis-definition protection")
            if not isinstance(record["scientific_context"], dict) or not record["scientific_context"]:
                raise ValueError("Scientific context must be explicit, including not-applicable where appropriate")
            fixed_reference = {"reference.time.zero": ("x", 0), "reference.signal.zero": ("y", 0),
                               "reference.response_ratio.equal_baseline": ("y", 1)}.get(role)
            if fixed_reference and (record["geometry"].get("dimension"), record["geometry"].get("value")) != fixed_reference:
                raise ValueError("Scientific zero/one reference dimension/value does not match its role")
            if record["coordinate_system"] not in specification["element_record_contract"]["coordinate_systems"]:
                raise ValueError("Unknown coordinate system")
            for additional_role in record.get("scientific_roles", []):
                if additional_role not in specification["roles"]:
                    raise ValueError("Unknown additional scientific role")
            if any(mapped not in records for mapped in record.get("mappable_ids", [])):
                raise ValueError("Composite/mappable relationship refers to an unknown element")
            if record.get("composite_id") and not record.get("composite_part"):
                raise ValueError("Composite membership requires a named composite_part")
            if role in {"axis.tick", "axis.tick_label"}:
                if not set(specification["element_record_contract"]["tick_required_fields"]) <= set(record["geometry"]):
                    raise ValueError("Tick dimension/side/kind/value must be recorded")
                if record["geometry"]["dimension"] not in {"x", "y"} or record["geometry"]["kind"] not in {"major", "minor"}:
                    raise ValueError("Tick dimension/kind mapping is invalid")
            expected = dict(specification["styles"][definition["style"]])
            if record["style_role"] != definition["style"]:
                raise ValueError("Style role does not match scientific role")
            for exception in exceptions:
                scope = exception.get("scope", {})
                if exception.get("scientific_role") == role and record["panel_id"] in scope.get("panel_ids", []) and (not scope.get("element_ids") or element_id in scope["element_ids"]) and (not scope.get("subpanel_ids") or record["subpanel_id"] in scope["subpanel_ids"]):
                    expected[exception["property"]] = exception["value"]
            applied = {}
            for exception in exceptions:
                scope = exception.get("scope", {})
                if exception.get("scientific_role") == role and record["panel_id"] in scope.get("panel_ids", []) and (not scope.get("element_ids") or element_id in scope["element_ids"]) and (not scope.get("subpanel_ids") or record["subpanel_id"] in scope["subpanel_ids"]):
                    key = exception["property"]
                    if key in applied and applied[key] != exception["value"]:
                        raise ValueError("Conflicting approved exceptions for the same element/property")
                    applied[key] = exception["value"]
            for key, value in expected.items():
                if record["resolved_style"].get(key) != value:
                    raise ValueError(f"Style deviation needs approved exception: {key} expected {value!r}")
            # Some renderer properties (e.g. cmap, padding, shared-axis layout)
            # cannot be recovered reliably from SVG. Require explicit evidence
            # instead of treating declared resolved_style values as proof.
            measured_keys = {"color", "linewidth_pt", "alpha", "linestyle", "font_size_pt", "font_weight", "length_pt", "direction", "edge_visible", "behind_all_data"}
            for key in set(expected)-measured_keys:
                if not _passed_review(record.get("style_verification", {}).get(key)):
                    raise ValueError(f"Unmeasured style property needs completed review evidence: {key}")
            resolved_color = expected.get("color")
            if "color_source" in expected:
                resolved_color = record.get("resolved_color")
                if not resolved_color or not record.get("color_evidence"):
                    raise ValueError("Semantic color source requires resolved_color and color_evidence")
            required = record["required_in_svg"] in (True, "true")
            if not evidence:
                continue
            node = evidence.ids.get(element_id)
            if node is None:
                if required:
                    raise ValueError("Required semantic ID absent from SVG")
                continue
            container = role in {"axes.container", "axis.component", "colorbar.scale", "legend.key"}
            if not container:
                covered.add(node)
            leaves = [] if container else evidence.leaves(node)
            leaves = [leaf for leaf in leaves if not any(p is not node and p.get("id") in records for p in evidence.chain(leaf)[evidence.chain(leaf).index(node)+1:])]
            if required and not container and not leaves:
                raise ValueError("Required artist has no independently mapped visible geometry/text")
            if not container and "font_size_pt" in expected and not any(_tag(leaf) == "text" for leaf in leaves):
                raise ValueError("Typography verification requires editable SVG text; outlined glyphs need a text export")
            for leaf in leaves:
                props, opacity, scale = evidence.presentation(leaf)
                if props.get("display") == "none" or props.get("visibility") == "hidden":
                    if required:
                        raise ValueError("Required artist is hidden")
                    continue
                if _tag(leaf) == "image" and not _passed_review(record.get("style_verification", {}).get("raster_content")):
                    raise ValueError("Embedded image content/opacity needs completed raster_content review evidence")
                if _tag(leaf) == "text":
                    if "font_size_pt" in expected and not math.isclose(_length(props.get("font-size", ""))*scale, expected["font_size_pt"], abs_tol=.02):
                        raise ValueError("Rendered font size differs at final assembly scale")
                    if props.get("font-family", "").strip("\"'") != specification["typography"]["family"]:
                        raise ValueError("Rendered font family differs from specification")
                    if props.get("font-weight", "normal") not in {expected.get("font_weight", "normal"), "400" if expected.get("font_weight", "normal") == "normal" else "700"}:
                        raise ValueError("Rendered font weight differs")
                    if props.get("font-style", "normal") != specification["typography"]["font_style"]:
                        raise ValueError("Rendered font style differs")
                if props.get("stroke", "none") != "none" and "linewidth_pt" in expected:
                    if not math.isclose(_length(props["stroke-width"])*scale, expected["linewidth_pt"], abs_tol=.02):
                        raise ValueError("Rendered stroke width differs at final assembly scale")
                    pattern = props.get("stroke-dasharray", "none")
                    if expected.get("linestyle") == "solid" and pattern not in {"none", ""}:
                        raise ValueError("Solid scientific line was rendered with a dash pattern")
                    if expected.get("linestyle") in {"dashed", "dotted"}:
                        lengths = [_length(v)*scale for v in re.split(r"[ ,]+", pattern)]
                        if len(lengths) < 2 or min(lengths) <= 0:
                            raise ValueError("Scientific line requires a valid dash pattern")
                        is_dotted = lengths[0] <= expected["linewidth_pt"]*1.1
                        if is_dotted != (expected["linestyle"] == "dotted"):
                            raise ValueError("Scientific line dash pattern changes dotted/dashed identity")
                elif "linewidth_pt" in expected:
                    raise ValueError("Line/tick artist has no visible stroke")
                if expected.get("edge_visible") is False and props.get("stroke", "none") != "none":
                    raise ValueError("Artist unexpectedly has a visible edge")
                if role == "axis.tick" and "length_pt" in expected:
                    target = leaf
                    if _tag(leaf) == "use":
                        target = evidence.ids.get(leaf.get("href", leaf.get("{http://www.w3.org/1999/xlink}href", "")).removeprefix("#"))
                    coordinates = [float(v) for v in re.findall(NUMBER, target.get("d", ""))] if target is not None else []
                    if len(coordinates) != 4 or not re.fullmatch(r"\s*M\s*"+NUMBER+r"[ ,]+"+NUMBER+r"\s*L\s*"+NUMBER+r"[ ,]+"+NUMBER+r"\s*", target.get("d", "")):
                        raise ValueError("Tick length needs an explicit simple line template")
                    if not math.isclose(math.hypot(coordinates[2]-coordinates[0], coordinates[3]-coordinates[1])*scale, expected["length_pt"], abs_tol=.02):
                        raise ValueError("Rendered tick length differs at final assembly scale")
                    side = record["geometry"]["side"]
                    dx, dy = coordinates[2]-coordinates[0], coordinates[3]-coordinates[1]
                    if expected.get("direction") != "out" or side not in {"left", "right", "top", "bottom"}:
                        raise ValueError("Tick direction requires an outward side mapping")
                    if not {"left": dx < 0 and dy == 0, "right": dx > 0 and dy == 0,
                            "top": dy < 0 and dx == 0, "bottom": dy > 0 and dx == 0}[side]:
                        raise ValueError("Rendered tick is not outward for its declared side")
                for channel in ["stroke", "fill"]:
                    if props.get(channel, "none") == "none":
                        continue
                    color, paint_alpha = _paint(props[channel])
                    actual_alpha = opacity*float(props.get(channel+"-opacity", 1))*paint_alpha
                    if "alpha" in expected and not math.isclose(actual_alpha, expected["alpha"], abs_tol=.002):
                        raise ValueError("Rendered effective opacity differs")
                    if resolved_color and color != _paint(resolved_color)[0]:
                        raise ValueError("Rendered color differs")
            if protected and baseline:
                source_id = record.get("source_element_id", element_id)
                source_node = baseline.ids.get(source_id)
                if source_node is None or evidence.geometry(node) != baseline.geometry(source_node):
                    raise ValueError("Protected geometry changed or has no original source mapping")
            if (record["protection"] == "scientific-text" or scientific_text) and baseline:
                source_node = baseline.ids.get(record.get("source_element_id", element_id))
                if source_node is None or "".join(source_node.itertext()).strip() != "".join(node.itertext()).strip():
                    raise ValueError("Scientific text changed or has no original source mapping")
            if role.startswith("axis.") and baseline and role != "axis.tick":
                source_node = baseline.ids.get(record.get("source_element_id", element_id))
                if source_node is None:
                    raise ValueError("Axis element has no original source mapping")
                if role == "axis.tick_label" and "".join(source_node.itertext()).strip() != "".join(node.itertext()).strip():
                    raise ValueError("Axis tick text changed")
            if role == "axis.tick" and baseline:
                source_node = baseline.ids.get(record.get("source_element_id", element_id))
                if source_node is None or evidence.tick_anchors(node) != baseline.tick_anchors(source_node):
                    raise ValueError("Axis tick anchor changed or has no original source mapping")
            scope = (record["panel_id"], record["subpanel_id"])
            order = evidence.nodes.index(node)
            if role.startswith(("trace.", "trajectory.", "observation.", "summary.", "uncertainty.", "heatmap.")):
                data_order.append((scope, order))
            if expected.get("behind_all_data"):
                reference_order.append((scope, order, element_id))
        except (KeyError, TypeError, ValueError) as error:
            issue("element", str(error), element_id)
    if evidence:
        for node in evidence.leaves(evidence.root):
            try:
                props, opacity, _ = evidence.presentation(node)
                if props.get("display") == "none" or props.get("visibility") == "hidden" or opacity == 0:
                    continue
            except ValueError as error:
                issue("svg-evidence", str(error))
            if not any(ancestor in covered for ancestor in evidence.chain(node)):
                issue("unregistered-visible-element", f"Unregistered {_tag(node)}: {node.get('id', '(no ID)')}")
        for scope, order, element_id in reference_order:
            if any(data_scope == scope and data_index < order for data_scope, data_index in data_order):
                issue("stacking", "Reference is drawn above scientific data", element_id)
    return {"valid": not issues, "issues": issues, "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "specification_sha256": digest(specification_path), "specification_version": specification["specification_version"],
            "specification_id": specification["specification_id"], "checked_artifacts": artifacts, "checked_elements": len(records)}


def check_candidate(candidate_path: Path, specification_path: Path = DEFAULT_SPECIFICATION) -> dict:
    """Malformed/unsupported input always produces an unresolved report, not approval."""
    try:
        return _check_candidate(candidate_path, specification_path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ET.ParseError) as error:
        return {"valid": False, "issues": [{"code": "input", "message": str(error)}]}


def freeze_candidate(candidate_path: Path, output_path: Path, specification_path: Path = DEFAULT_SPECIFICATION) -> dict:
    """Validate and write a new freeze record exclusively; never overwrite a freeze."""
    report = check_candidate(candidate_path, specification_path)
    if not report["valid"]:
        return report
    candidate = _read_json(candidate_path.read_bytes())
    # Reject any source/candidate/specification revision between review and publication.
    for item in report["checked_artifacts"] + [
        {"path": str(candidate_path), "sha256": report["candidate_sha256"]},
        {"path": str(specification_path), "sha256": report["specification_sha256"]},
    ]:
        if digest(Path(item["path"])) != item["sha256"]:
            return {**report, "valid": False, "issues": [{"code": "stale-review", "message": "Input changed after validation"}]}
    output_path = output_path.resolve()
    # Normalize referenced paths before relocating the candidate manifest.
    for group in ("source_artifacts", "data_artifacts", "exports"):
        for item in candidate[group]:
            item["path"] = str((candidate_path.resolve().parent/Path(item["path"])).resolve())
    candidate.update(specification_id=report["specification_id"], specification_version=report["specification_version"],
                     specification_path=str(specification_path.resolve()), specification_sha256=report["specification_sha256"],
                     resolved_styles={key: value["resolved_style"] for key, value in candidate["element_registry"].items()},
                     freeze_check=report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent/f".figure-freeze-{uuid.uuid4().hex}"; staging.mkdir()
    try:
        staged = staging/"freeze.json"; staged.write_text(json.dumps(candidate, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        if os.name == "nt":
            os.rename(staged, output_path)  # Windows rename refuses existing destinations.
        else:
            os.link(staged, output_path)  # Exclusive, atomic publication on POSIX.
        return {**report, "freeze_manifest": str(output_path)}
    finally:
        if not staging.resolve().is_relative_to(output_path.parent.resolve()):
            raise ValueError("Freeze staging cleanup must stay inside its output directory")
        shutil.rmtree(staging)


def run_cli(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only and args.output is None:
        parser.error("--output is required when finalizing a freeze")
    try:
        report = check_candidate(args.candidate, args.specification) if args.check_only else freeze_candidate(args.candidate, args.output, args.specification)
    except (OSError, ValueError) as error:
        report = {"valid": False, "issues": [{"code": "freeze", "message": str(error)}]}
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)
