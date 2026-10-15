"""Physical invisible layout frames; no scientific selection or state publication.

The browser collector measures real glyphs/strokes and also unclipped bounds.
Evidence is bound to the exact SVG, frame metadata, policy and collectors.
"""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / 'configs/paper-figures/layout-frames.json'
COLLECTOR = Path(__file__).with_name('layout_frames_browser.js')
SVG = '{http://www.w3.org/2000/svg}'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def object_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def load_policy(path=POLICY):
    raw = Path(path).read_bytes()
    policy = json.loads(raw)
    if policy.get('units') != 'mm' or policy.get('outside_edges') != 'shared-not-additive':
        raise ValueError('Unsupported frame policy')
    if not finite(policy.get('tolerance_mm')) or policy['tolerance_mm'] <= 0:
        raise ValueError('Invalid frame tolerance')
    selected = policy.get('selected_inset_mm')
    if policy.get('activation') == 'active':
        if not finite(selected) or selected <= 0 or not policy.get('author_selection'):
            raise ValueError('Active inset requires a positive value and author selection evidence')
    return policy, sha(raw)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def rectangle(value):
    return (isinstance(value, list) and len(value) == 4 and all(finite(v) for v in value)
            and min(value[2:]) > 0)


def inside(inner, outer, tolerance=0):
    return (inner[0] >= outer[0]-tolerance and inner[1] >= outer[1]-tolerance
            and inner[0]+inner[2] <= outer[0]+outer[2]+tolerance
            and inner[1]+inner[3] <= outer[1]+outer[3]+tolerance)


def inset(box, margin):
    result = [box[0]+margin, box[1]+margin, box[2]-2*margin, box[3]-2*margin]
    if not rectangle(result):
        raise ValueError('Inset leaves no usable area')
    return result


def validate_spec(spec, policy=None):
    policy = policy or load_policy()[0]
    size = spec.get('final_size_mm', [])
    if len(size) != 2 or not all(finite(v) and v > 0 for v in size):
        raise ValueError('Declare positive final_size_mm')
    margin = spec.get('inset_mm')
    if not finite(margin) or margin <= 0:
        raise ValueError('Declare positive inset_mm')
    if spec.get('calibration'):
        if margin not in policy['calibration_presets_mm']:
            raise ValueError('Unknown calibration preset')
    elif policy['activation'] != 'active' or margin != policy['selected_inset_mm']:
        raise ValueError('Inset pending author calibration, or differs from selected paper inset')
    if spec.get('scope') not in {'figure', 'panel'}:
        raise ValueError('Declare figure or panel scope')
    figure = [0, 0, *size]
    inset(figure, margin)
    allocations = spec.get('allocations', [])
    if not allocations:
        raise ValueError('Declare owned panel/shared allocations')
    ids, groups = set(), set()
    for item in allocations:
        identity, box = item['id'], item['box_mm']
        if identity in ids or not rectangle(box) or not inside(box, figure):
            raise ValueError('Invalid/duplicate allocation: '+identity)
        ids.add(identity)
        if item.get('kind') not in {'panel', 'shared'}:
            raise ValueError('Declare allocation kind panel/shared')
        inset(box, margin)
        if not item.get('group_ids') or groups.intersection(item['group_ids']):
            raise ValueError('Each content group needs one owner')
        groups.update(item['group_ids'])
    if spec['scope'] == 'panel' and (len(allocations) != 1 or allocations[0]['kind'] != 'panel'
                                   or allocations[0]['box_mm'] != figure):
        raise ValueError('Panel scope needs one full-canvas panel allocation')
    # Tiles may touch; positive gaps require an explicit named grouping gap.
    for i, a in enumerate(allocations):
        for b in allocations[i+1:]:
            x, y, w, h = a['box_mm']; xx, yy, ww, hh = b['box_mm']
            overlap_x, overlap_y = min(x+w, xx+ww)-max(x, xx), min(y+h, yy+hh)-max(y, yy)
            if overlap_x > 1e-8 and overlap_y > 1e-8:
                raise ValueError('Overlapping allocations: '+a['id']+'/'+b['id'])
            gap = -overlap_x if overlap_y > 0 else -overlap_y if overlap_x > 0 else 0
            intervening = False
            for c in allocations:
                if c is a or c is b:
                    continue
                cx, cy, cw, ch = c['box_mm']
                if overlap_y > 0 and min(x+w, xx+ww) <= cx and cx+cw <= max(x, xx) and min(cy+ch, y+h, yy+hh) > max(cy, y, yy):
                    intervening = True
                if overlap_x > 0 and min(y+h, yy+hh) <= cy and cy+ch <= max(y, yy) and min(cx+cw, x+w, xx+ww) > max(cx, x, xx):
                    intervening = True
            if gap > 1e-8 and not intervening and not any(set(g.get('between', [])) == {a['id'], b['id']}
                                     and g.get('reason') and finite(g.get('gap_mm'))
                                     and abs(g['gap_mm']-gap) < 1e-8 for g in spec.get('grouping_gaps', [])):
                raise ValueError('Record named grouping gap: '+a['id']+'/'+b['id'])
    # Exterior rows/columns must actually share the figure boundary, not add another inset.
    tol = 1e-8
    for a in allocations:
        x, y, w, h = a['box_mm']
        for edge, coordinate, boundary in [('left', x, 0), ('top', y, 0),
                                            ('right', x+w, size[0]), ('bottom', y+h, size[1])]:
            neighbour = False
            for other in allocations:
                if other is a: continue
                ox, oy, ow, oh = other['box_mm']
                vertical = min(y+h, oy+oh) > max(y, oy)
                horizontal = min(x+w, ox+ow) > max(x, ox)
                if ((edge == 'left' and vertical and ox+ow <= x+tol)
                    or (edge == 'right' and vertical and ox >= x+w-tol)
                    or (edge == 'top' and horizontal and oy+oh <= y+tol)
                    or (edge == 'bottom' and horizontal and oy >= y+h-tol)):
                    neighbour = True
            if (edge in a.get('outside_edges', []) or not neighbour) and abs(coordinate-boundary) > tol:
                raise ValueError('Outside panel edge must share canvas boundary: '+a['id']+'/'+edge)
    if spec['scope'] == 'figure':
        for edge, index, end, boundary in [('left', 0, False, 0), ('top', 1, False, 0),
                                            ('right', 0, True, size[0]), ('bottom', 1, True, size[1])]:
            coordinates = [a['box_mm'][index]+(a['box_mm'][index+2] if end else 0) for a in allocations]
            if abs((max(coordinates) if end else min(coordinates))-boundary) > tol:
                raise ValueError('Exterior allocations must reach figure '+edge+' boundary')
    for clip in spec.get('scientific_clips', []):
        if not clip.get('id') or not clip.get('evidence'):
            raise ValueError('Scientific clipping requires explicit ID/source evidence')
    for alignment in spec.get('alignments', []):
        if alignment.get('property') not in {'left', 'right', 'top', 'bottom', 'width', 'height'}:
            raise ValueError('Unsupported alignment property')
        if alignment.get('target') not in {'content', 'data_axes'} or len(alignment.get('ids', [])) < 2:
            raise ValueError('Declare named content/data_axes alignment targets')
    spacing_ids = set()
    for group in spec.get('spacing_groups', []):
        if not group.get('id') or group['id'] in spacing_ids:
            raise ValueError('Spacing groups need unique IDs')
        spacing_ids.add(group['id'])
        available = ids if group.get('target') == 'content' else {a['id'] for a in spec.get('data_axes', [])}
        members = group.get('ids', [])
        if (group.get('target') not in {'content', 'data_axes'} or group.get('axis') not in {'x', 'y'}
                or len(members) < 3 or len(set(members)) != len(members) or not set(members) <= available):
            raise ValueError('Declare ordered, distinct spacing targets')
        if not finite(group.get('minimum_gap_mm', 0)) or group.get('minimum_gap_mm', 0) < 0:
            raise ValueError('Invalid minimum spacing')
    return spec


def embed_spec(data, spec):
    validate_spec(spec)
    root = ET.fromstring(data)
    previous = root.find(SVG+'metadata[@id="layout-frames"]')
    if previous is not None:
        root.remove(previous)
    node = ET.SubElement(root, SVG+'metadata', {'id': 'layout-frames'})
    node.text = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False)
    root.set('width', f'{spec["final_size_mm"][0]}mm')
    root.set('height', f'{spec["final_size_mm"][1]}mm')
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def read_spec(data):
    node = ET.fromstring(data).find(SVG+'metadata[@id="layout-frames"]')
    return json.loads(node.text) if node is not None else None


def assembly_layout(layout):
    """Opt-in geometry; legacy layouts are unchanged. All coordinates stay explicit."""
    import copy
    result = copy.deepcopy(layout)
    settings = result.get('layout_frames')
    if settings is None:
        return result, None
    width = settings['final_width_mm']
    canvas_w, canvas_h = result['canvas']
    scale = width/canvas_w
    spec = dict(settings)
    spec.pop('final_width_mm')
    spec['final_size_mm'] = [width, canvas_h*scale]
    spec.setdefault('scope', 'figure')
    spec['allocations'] = []
    spec.setdefault('background_ids', ['figure-background'])
    for p in result['panels']:
        if p.get('frame') or p.get('view_box') or p.get('remove_lines_crossing_crop'):
            raise ValueError('Frame-aware assembly rejects visible layout borders and crop repairs')
        if p.get('fit', 'xMidYMid meet') != 'xMidYMid meet':
            raise ValueError('Preserve natural aspect ratio in frame-aware assembly')
        box = [v*scale for v in p['box']]
        inner = inset(box, settings['inset_mm'])
        p['content_box'] = [v/scale for v in inner]
        spec['allocations'].append(dict(id=p['id'], kind=p.get('layout_kind', 'panel'), box_mm=box,
                                       group_ids=['panel-'+p['id']]))
    validate_spec(spec)
    return result, spec


def assess(data, browser=None, policy_path=POLICY):
    """Evaluate saved rendering evidence; absent/unsupported evidence never passes."""
    policy, policy_hash = load_policy(policy_path)
    issues = []
    try:
        spec = read_spec(data)
    except (ET.ParseError, ValueError) as error:
        return {'status': 'unresolved', 'issues': ['Invalid frame SVG/metadata: '+str(error)]}
    if spec is None:
        return {'status': 'unresolved', 'issues': ['Missing layout-frames metadata']}
    try:
        validate_spec(spec, policy)
        root = ET.fromstring(data)
        for attr, expected in zip(('width', 'height'), spec['final_size_mm']):
            value = root.get(attr, '')
            if not value.endswith('mm') or abs(float(value[:-2])-expected)>1e-7:
                raise ValueError('SVG physical dimensions must match frame final_size_mm')
    except (ValueError, KeyError, TypeError) as error:
        return {'status': 'unresolved', 'issues': [str(error)]}
    identity = dict(artifact_sha256=sha(data), spec_sha256=object_sha(spec), policy_sha256=policy_hash,
                    collector_sha256=sha(COLLECTOR.read_bytes()), helper_sha256=sha(Path(__file__).read_bytes()),
                    final_size_mm=spec['final_size_mm'], scope=spec['scope'])
    if browser is None:
        issues.append('Browser rendering evidence required; geometry alone cannot establish blank margins')
    else:
        for key in ('artifact_sha256', 'spec_sha256', 'collector_sha256', 'final_size_mm', 'scope'):
            if browser.get(key) != identity[key]:
                issues.append('Stale/mismatched browser '+key)
        if browser.get('raster_dpi', 0) < 600:
            issues.append('600 dpi browser paint measurement required')
        if not browser.get('user_agent') or not browser.get('fonts_ready') or not browser.get('images_ready'):
            issues.append('Browser/fonts/images completion evidence required')
        issues.extend(browser.get('issues', []))
        measured = browser.get('allocations', {})
        tolerance = policy['tolerance_mm']
        for item in spec['allocations']:
            record = measured.get(item['id'], {})
            allowed = inset(item['box_mm'], spec['inset_mm'])
            for name in ('painted_bounds_mm', 'unclipped_bounds_mm'):
                box = record.get(name)
                if not rectangle(box) or not inside(box, allowed, tolerance):
                    issues.append(item['id']+': missing/outside '+name)
        for a in spec.get('alignments', []):
            bounds = browser.get('data_axes', {}) if a['target'] == 'data_axes' else {
                k: v.get('painted_bounds_mm') for k, v in measured.items()}
            values = []
            for target in a['ids']:
                b = bounds.get(target)
                if not rectangle(b):
                    issues.append('Missing alignment measurement: '+target)
                    continue
                prop = a['property']; index = {'left': 0, 'top': 1, 'width': 2, 'height': 3, 'right': 0, 'bottom': 1}[prop]
                values.append(b[index]+(b[index+2] if prop in {'right', 'bottom'} else 0))
            if len(values) >= 2 and max(values)-min(values) > tolerance:
                issues.append('Alignment failed: '+a['id'])
        for group in spec.get('spacing_groups', []):
            variants = ('painted_bounds_mm', 'unclipped_bounds_mm') if group['target'] == 'content' else ('data_axes',)
            for variant in variants:
                bounds = browser.get('data_axes', {}) if variant == 'data_axes' else {k:v.get(variant) for k,v in measured.items()}
                boxes = [bounds.get(k) for k in group['ids']]
                if not all(rectangle(b) for b in boxes):
                    issues.append('Missing spacing measurement: '+group['id']);continue
                index = 0 if group['axis'] == 'x' else 1
                gaps = [b[index]-a[index]-a[index+2] for a,b in zip(boxes,boxes[1:])]
                if max(gaps)-min(gaps) > tolerance or min(gaps) < group.get('minimum_gap_mm',0)-tolerance:
                    issues.append('Spacing failed: '+group['id']+'/'+variant)
    return {**identity, 'status': 'unresolved' if issues else 'passed', 'issues': issues,
            'calibration_only': bool(spec.get('calibration')), 'browser': browser,
            'policy_activation': policy['activation']}


def freeze_issues(data, record, final_width_mm, panel_ids, policy_path=POLICY):
    """Authenticate evidence and re-evaluate, independent of its claimed status."""
    if not isinstance(record, dict):
        return ['New freezes require hash-bound layout_frames evidence']
    try:
        path = Path(record['path'])
        raw = path.read_bytes()
        if sha(raw) != record['sha256']:
            return ['Frame evidence hash mismatch']
        report = json.loads(raw)
        actual = assess(data, report.get('browser'), policy_path)
        if actual['status'] != 'passed':
            return actual['issues']
        for key in ('artifact_sha256', 'spec_sha256', 'policy_sha256', 'collector_sha256', 'helper_sha256', 'final_size_mm', 'scope'):
            if report.get(key) != actual.get(key):
                return ['Stale frame evidence: '+key]
        spec = read_spec(data)
        if spec.get('calibration'):
            return ['Calibration previews cannot be frozen']
        if abs(spec['final_size_mm'][0]-final_width_mm) > 1e-8:
            return ['Frame evidence final width differs from freeze assembly scale']
        owned = {a['id'] for a in spec['allocations'] if a['kind'] == 'panel'}
        if owned != set(panel_ids):
            return ['Frame evidence panel scope differs from freeze scope']
        return []
    except (OSError, ValueError, KeyError, TypeError) as error:
        return ['Invalid frame evidence: '+str(error)]
