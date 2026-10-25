"""Remove explicitly selected opaque SVG canvases without reserializing artwork.

This is a presentation repair, never a search for arbitrary white artwork.
Only a named group's sole, viewport-sized white path is eligible. All bytes
outside its inline fill value remain unchanged, including data and resources.
"""
import math
import re
import xml.etree.ElementTree as ET
from xml.parsers import expat

from classical_conditioning.figure_workspace import sha, tag


def transparent_canvases(raw: bytes, group_ids: tuple[str, ...]):
    if not group_ids or len(set(group_ids)) != len(group_ids):
        raise ValueError('Select distinct explicit canvas group IDs')
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('SVG document declarations are unsupported')
    root = ET.fromstring(raw)
    parents = {child: parent for parent in root.iter() for child in parent}
    number = r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'
    rectangle = re.compile(r'\s*M\s*(' + number + r')\s+(' + number + r')\s*'
        + ''.join(r'L\s*(' + number + r')\s+(' + number + r')\s*' for _ in range(3)) + r'[zZ]\s*')
    selected = {}
    for identity in group_ids:
        matches = [n for n in root.iter() if n.get('id') == identity]
        if len(matches) != 1:
            raise ValueError('Canvas group must be unique: ' + identity)
        group = matches[0]
        if group.tag != tag('g') or set(group.attrib) != {'id'} or len(group) != 1:
            raise ValueError('Canvas group must contain only one untransformed path: ' + identity)
        path = group[0]
        if path.tag != tag('path') or set(path.attrib) != {'d', 'style'} or len(path):
            raise ValueError('Canvas must be an untransformed plain path: ' + identity)
        if path.get('style') != 'fill: #ffffff':
            raise ValueError('Canvas must have the expected opaque white fill: ' + identity)
        ancestor = parents[group]
        while ancestor.tag != tag('svg'):
            ancestor = parents[ancestor]
        view = [float(v) for v in ancestor.get('viewBox', '').split()]
        shape = rectangle.fullmatch(path.get('d', ''))
        if len(view) != 4 or view[:2] != [0, 0] or not shape:
            raise ValueError('Canvas requires a zero-origin rectangular viewport: ' + identity)
        width, height = view[2:]
        actual = [float(v) for v in shape.groups()]
        expected = [0, height, width, height, width, 0, 0, 0]
        if min(width, height) <= 0 or not all(math.isfinite(v) for v in view + actual) or not all(
                math.isclose(a, b, rel_tol=0, abs_tol=1e-9) for a, b in zip(actual, expected)):
            raise ValueError('White path does not match the full viewport: ' + identity)
        selected[identity] = {'viewport': view, 'path_sha256': sha(path.get('d').encode())}

    # Expat byte offsets let us replace only attribute values, preserving XML
    # prefixes, comments, whitespace, numeric spelling and embedded resources.
    parser = expat.ParserCreate(namespace_separator='}')
    stack, replacements = [], []

    def start(name, attrs):
        if stack and stack[-1] in selected and name == 'http://www.w3.org/2000/svg}path':
            offset = parser.CurrentByteIndex
            end = raw.index(b'>', offset)
            token = raw[offset:end]
            match = re.search(rb'\bstyle\s*=\s*([\'"])(.*?)\1', token)
            if not match or match[2] != b'fill: #ffffff':
                raise ValueError('Cannot locate exact canvas fill bytes')
            replacements.append((offset + match.start(2), offset + match.end(2), stack[-1]))
        stack.append(attrs.get('id'))

    parser.StartElementHandler = start
    parser.EndElementHandler = lambda name: stack.pop()
    parser.Parse(raw, True)
    if len(replacements) != len(selected):
        raise ValueError('Canvas byte mapping is incomplete')
    candidate = raw
    for start, end, _ in sorted(replacements, reverse=True):
        candidate = candidate[:start] + b'fill: none' + candidate[end:]
    restored = ET.fromstring(candidate)
    for group in restored.iter():
        if group.get('id') in selected:
            group[0].set('style', 'fill: #ffffff')
    if ET.tostring(restored) != ET.tostring(root):
        raise ValueError('Unexpected artwork change outside canvas fill')
    return candidate, {'input_sha256': sha(raw), 'output_sha256': sha(candidate),
                       'changes': [{'group_id': identity, **selected[identity],
                                    'old_fill': '#ffffff', 'new_fill': 'none'} for identity in group_ids],
                       'all_other_bytes_preserved': True}
