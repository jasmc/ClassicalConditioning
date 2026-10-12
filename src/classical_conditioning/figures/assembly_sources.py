"""Read-only assembly source/geometry inspection; never selects scientific versions."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from classical_conditioning.external_artifacts import resolve_artifact

SVG = '{http://www.w3.org/2000/svg}svg'
SOURCE_REPAIRS = ('view_box', 'remove_lines_crossing_crop', 'flatten_text',
                  'text_adjustments', 'remove_groups_by_class', 'omit_text_exact', 'add_source_text')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _source_reference(layout, layout_path, name):
    name = str(name).replace('\\', '/')
    if re.match(r'^[A-Za-z]:/', name) or Path(name).is_absolute():
        return name
    base = str(layout.get('storage_root') or layout_path.parent).replace('\\', '/')
    if re.match(r'^[A-Za-z]:/', base):
        return base.rstrip('/') + '/' + name
    return str((Path(base).expanduser() / name).resolve())


def inspect_source(layout, layout_path, name, expected=None):
    result = {'reference': name, 'expected_sha256': expected, 'path': None, 'sha256': None}
    if not name:
        return {**result, 'status': 'unset'}
    reference = _source_reference(layout, layout_path, name)
    try:
        source = resolve_artifact(reference, expected_sha256=expected)
        result.update(path=str(source.resolve()), sha256=file_hash(source))
        if expected and result['sha256'] != expected:
            raise ValueError(f'SHA-256 mismatch: {source}')
        if source.suffix.lower() == '.svg':
            root = ET.parse(source).getroot()
            if root.tag != SVG:
                raise ValueError(f'Not an SVG root: {source}')
            if root.get('viewBox'):
                box = [float(v) for v in re.split(r'[\s,]+',root.get('viewBox').strip())]
                if len(box)!=4 or not all(math.isfinite(v) for v in box) or min(box[2:])<=0:
                    raise ValueError(f'Invalid SVG viewBox: {source}')
        elif source.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            raise ValueError(f'Unsupported panel source type: {source.suffix}')
        return {**result, 'status': 'ready'}
    except OSError as error:
        return {**result, 'status': 'unavailable', 'error': str(error)}
    except (ValueError, ET.ParseError) as error:
        return {**result, 'status': 'invalid', 'error': str(error)}


def prepare_layout(layout, overrides=None, only_panel=None):
    original = copy.deepcopy(layout)
    layout = copy.deepcopy(layout)
    overrides = overrides or {}
    ids = [p['id'] for p in layout['panels']]
    if set(overrides) - set(ids):
        raise ValueError('Unknown panel source override: ' + ', '.join(sorted(set(overrides)-set(ids))))
    for panel in layout['panels']:
        if panel['id'] in overrides:
            panel['source'] = overrides[panel['id']]
            # Candidate sources do not inherit the selected file's freeze claim or repairs.
            for key in (*SOURCE_REPAIRS, 'frozen_sha256', 'source_provenance'):
                panel.pop(key, None)
            panel['selection_status'] = 'candidate source override; not approved or frozen'
    if only_panel:
        matches = [p for p in layout['panels'] if p['id'] == only_panel]
        if len(matches) != 1:
            raise ValueError('Unknown or duplicate panel ID: ' + only_panel)
        panel = matches[0]
        x, y, w, h = panel['box']
        panel['box'] = [0, 0, w, h]
        if 'content_box' in panel:
            cx, cy, cw, ch = panel['content_box']; panel['content_box'] = [cx-x, cy-y, cw, ch]
        if panel.get('label'):
            panel['label']['x'] = panel['label'].get('x', x+6)-x
            panel['label']['y'] = panel['label'].get('y', y+34)-y
        layout['panels'] = [panel]; layout['canvas'] = [w, h]; layout.pop('annotations', None)
        layout['title'] = f"{layout.get('title', 'Figure')} — panel {only_panel} preview"
    return original, layout


def validate_geometry(layout):
    def numbers(values, count):
        return (len(values) == count and all(isinstance(n, (int, float)) and not isinstance(n, bool)
                                             and math.isfinite(n) for n in values))
    canvas = layout['canvas']
    if not numbers(canvas, 2) or min(canvas) <= 0:
        raise ValueError('Canvas dimensions must be finite and positive')
    if 'font_family' in layout and (not isinstance(layout['font_family'],str) or not layout['font_family'].strip()):
        raise ValueError('Font family must be a nonempty string')
    for annotation in layout.get('annotations',[]):
        if not numbers([annotation['x'],annotation['y']],2) or not numbers([annotation.get('size',22)],1) or annotation.get('size',22)<=0:
            raise ValueError('Invalid annotation position/font size')
    width, height = canvas; seen = set()
    for panel in layout['panels']:
        pid = re.sub(r'[^A-Za-z0-9_-]+', '-', panel['id']).strip('-')
        if not pid or pid in seen:
            raise ValueError('Unknown/duplicate normalized panel ID: ' + pid)
        seen.add(pid)
        label = panel.get('label',{})
        if any(not numbers([label[k]],1) for k in ('x','y','size') if k in label) or label.get('size',30)<=0:
            raise ValueError('Invalid label position/font size for '+pid)
        box = panel['box']
        if not numbers(box, 4):
            raise ValueError('Invalid box for ' + pid)
        x, y, w, h = box
        if min(w, h) <= 0 or min(x, y) < 0 or x+w > width or y+h > height:
            raise ValueError('Panel outside canvas or invalid box: ' + pid)
        content = panel.get('content_box', box)
        if not numbers(content, 4):
            raise ValueError('Invalid content_box for ' + pid)
        cx, cy, cw, ch = content
        if min(cw, ch) <= 0 or cx < x or cy < y or cx+cw > x+w or cy+ch > y+h:
            raise ValueError('Invalid content_box for ' + pid)
        if 'view_box' in panel and (not numbers(panel['view_box'], 4) or min(panel['view_box'][2:]) <= 0):
            raise ValueError('Invalid view_box for ' + pid)


def preflight(layout_path, *, strict=False, only_panel=None, source_overrides=None):
    """Inspect explicit sources without writing, rendering or changing selections."""
    path = Path(layout_path).resolve()
    raw = path.read_bytes(); stored = json.loads(raw)
    report = {'layout': str(path), 'layout_sha256': hashlib.sha256(raw).hexdigest(),
              'status': 'ready', 'can_render': True, 'errors': [], 'panels': [],
              'scientific_status': 'assembly review; no new approval or freeze'}
    try:
        original, layout = prepare_layout(stored, source_overrides, only_panel)
        validate_geometry(layout)
    except (ValueError, TypeError, KeyError) as error:
        return {**report, 'status': 'blocked', 'can_render': False, 'errors': [str(error)]}
    report['canvas'] = layout['canvas']
    original_by_id = {p['id']: p for p in original['panels']}
    for panel in layout['panels']:
        selected = original_by_id[panel['id']]
        replaced = panel['id'] in (source_overrides or {})
        expected = selected.get('frozen_sha256') or selected.get('source_provenance', {}).get('svg_sha256')
        current = inspect_source(original, path, selected.get('source'), expected)
        actual = inspect_source(layout, path, panel.get('source')) if replaced else current
        required = strict or bool(expected) or replaced
        errors = []
        if actual['status']=='ready' and 'view_box' in panel and Path(actual['path']).suffix.lower()!='.svg':
            errors.append('view_box cropping requires an SVG source')
        if actual['status'] == 'invalid' or (required and actual['status'] != 'ready'):
            errors.append(actual.get('error', 'Required source is unset'))
        if replaced and expected and current['status'] != 'ready':
            errors.append('Original frozen source could not be authenticated: ' + current.get('error', current['status']))
        item = {'id': panel['id'], 'source': actual, 'original_source': current,
                'replacement': replaced, 'status': 'blocked' if errors else 'ready' if actual['status']=='ready' else 'placeholder',
                'selection_status': panel.get('selection_status'), 'box': panel['box'],
                'errors': errors}
        report['panels'].append(item)
        report['errors'].extend(f"{panel['id']}: {error}" for error in errors)
    report['can_render'] = not report['errors']
    report['status'] = 'blocked' if report['errors'] else 'ready_with_placeholders' if any(p['status']=='placeholder' for p in report['panels']) else 'ready'
    return report
