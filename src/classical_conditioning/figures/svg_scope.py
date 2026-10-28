"""Scoped SVG extraction and collision-free comparison identifiers."""
import copy
import re
import xml.etree.ElementTree as ET
from classical_conditioning.figure_workspace import tag, sha

def scoped_working_svg(raw):
    root = ET.fromstring(raw)
    group = next(n for n in root if n.get('id') == 'panel-FGH')
    nested = next(n for n in group if n.tag == tag('svg'))
    result = ET.Element(tag('svg'), {k: v for k, v in nested.attrib.items()
                                   if k not in ('id', 'x', 'y', 'overflow')})
    for node in root:
        if node.tag == tag('defs'):
            result.append(copy.deepcopy(node))
    for node in nested:
        if node.tag in (tag('defs'), tag('g')):
            result.append(copy.deepcopy(node))
    return ET.tostring(result, encoding='utf-8'), sha(ET.tostring(nested))

def prefix_svg(raw, prefix):
    root = ET.fromstring(raw)
    names = {n.get('id'): prefix + '--' + n.get('id') for n in root.iter() if n.get('id')}
    for node in root.iter():
        if node.get('id'):
            node.set('data-comparison-source-id', node.get('id'))
            node.set('id', names[node.get('id')])
        for key, value in list(node.attrib.items()):
            if key.endswith('href') and value.startswith('#') and value[1:] in names:
                node.set(key, '#' + names[value[1:]])
            elif 'url(#' in value:
                node.set(key, re.sub(r'url\(#([^)]*)\)', lambda m: 'url(#' + names.get(m[1], m[1]) + ')', value))
    return ET.tostring(root, encoding='unicode')
