#!/usr/bin/env python3
"""Read-only static guide/resource/hash checks; not scientific or browser QA."""
from __future__ import annotations
import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, unquote_to_bytes
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GuideParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids=[];self.fragments=[];self.resources=[];self.svg_images=[]
        self.styles=[];self.style_depth=0;self.json_depth=0;self.json_current=[];self.json_blocks=[]

    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs)
        if 'id' in attrs:self.ids.append(attrs['id'])
        if tag=='a' and attrs.get('href','').startswith('#') and len(attrs['href'])>1:
            self.fragments.append(unquote(attrs['href'][1:]))
        for name in (['src','poster'] if tag in {'img','script','iframe','video','audio','source','track','embed','input'} else ['data'] if tag=='object' else []):
            if attrs.get(name):self.resources.append(attrs[name])
        if tag=='link' and any(x in attrs.get('rel','').split() for x in ['stylesheet','preload','modulepreload']):
            if attrs.get('href'):self.resources.append(attrs['href'])
        if 'style' in attrs:self.styles.append(attrs['style'])
        if tag=='style':self.style_depth+=1
        if tag=='img' and attrs.get('src','').startswith('data:image/svg+xml'):
            self.svg_images.append(attrs['src'])
        if tag=='script' and attrs.get('type') in {'application/json','application/ld+json'}:
            self.json_depth=1;self.json_current=[]
        if tag=='iframe' and attrs.get('srcdoc'):
            child=GuideParser();child.feed(attrs['srcdoc']);self.resources+=child.resources;self.styles+=child.styles;self.svg_images+=child.svg_images

    def handle_data(self, data):
        if self.style_depth:self.styles.append(data)
        if self.json_depth:self.json_current.append(data)

    def handle_endtag(self, tag):
        if tag=='style':self.style_depth=max(0,self.style_depth-1)
        if tag=='script' and self.json_depth:
            self.json_blocks.append(''.join(self.json_current));self.json_depth=0


def verify(path, offline=False, manifest=None):
    path=Path(path).resolve();parser=GuideParser();parser.feed(path.read_text(encoding='utf-8'))
    issues=[]
    duplicates=sorted({x for x in parser.ids if parser.ids.count(x)>1})
    if duplicates:issues.append('Duplicate static IDs: '+', '.join(duplicates))
    missing=sorted(set(parser.fragments)-set(parser.ids))
    if missing:issues.append('Missing static anchor targets: '+', '.join(missing))
    for text in parser.styles:
        parser.resources += [m[1].strip() for m in re.findall(r'url\(\s*([\'\"]?)(.*?)\1\s*\)',text,re.I)]
        parser.resources += re.findall(r'@import\s+[\'\"]([^\'\"]+)',text,re.I)
    external=[x for x in parser.resources if not x.startswith(('data:','#'))]
    if offline and external:issues.append('Non-embedded runtime resources: '+', '.join(sorted(set(external))))
    for i,text in enumerate(parser.json_blocks):
        try:json.loads(text)
        except ValueError as error:issues.append(f'Invalid embedded JSON block {i+1}: {error}')
    for i,value in enumerate(parser.svg_images):
        try:
            header,payload=value.split(',',1)
            raw=base64.b64decode(payload,validate=True) if ';base64' in header else unquote_to_bytes(payload)
            root=ET.fromstring(raw)
            if root.tag.rsplit('}',1)[-1]!='svg':raise ValueError('Not an SVG root')
            if offline:
                for element in root.iter():
                    if element.tag.rsplit('}',1)[-1] in {'image','use','script'}:
                        href=element.get('href',element.get('{http://www.w3.org/1999/xlink}href',''))
                        if href and not href.startswith(('data:','#')):issues.append(f'SVG image {i+1} has external resource: {href}')
        except (ValueError,ET.ParseError) as error:issues.append(f'Invalid embedded SVG image {i+1}: {error}')
    verified=0
    if manifest is not None:
        manifest=Path(manifest).resolve();record=json.loads(manifest.read_text());guide=record.get('guide')
        if not isinstance(guide,dict):issues.append('Manifest must contain guide: {path, sha256}')
        def resolve(value):
            p=Path(value);return p.resolve() if p.is_absolute() else (manifest.parent/p).resolve()
        if isinstance(guide,dict) and guide.get('path') and resolve(guide['path'])!=path:issues.append('Manifest selects a different HTML file')
        groups=[('guide',[guide] if isinstance(guide,dict) else []),('sources',record.get('sources',[])),('protected',record.get('protected',[]))]
        for group,items in groups:
            if not isinstance(items,list):issues.append(f'Manifest {group} must be a list');continue
            for item in items:
                if not isinstance(item,dict) or not item.get('path') or not re.fullmatch(r'[0-9a-fA-F]{64}',str(item.get('sha256',''))):
                    issues.append(f'Invalid {group} hash record');continue
                p=resolve(item['path'])
                if not p.is_file():issues.append('Missing bound file: '+str(p))
                elif digest(p)!=item['sha256'].lower():issues.append('Changed bound file: '+str(p))
                else:verified+=1
    return {'status':'passed' if not issues else 'failed','html':str(path),'sha256':digest(path),
        'static_ids':len(parser.ids),'embedded_svg_images':len(parser.svg_images),'embedded_json_blocks':len(parser.json_blocks),
        'nonembedded_runtime_resource_count':len(external),'verified_hash_records':verified,'issues':issues,
        'limits':'Static checks only; JavaScript behavior, full offline execution, scientific claims and visual layout need separate review.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('html',type=Path);p.add_argument('--offline',action='store_true');p.add_argument('--manifest',type=Path);a=p.parse_args()
    try:result=verify(a.html,a.offline,a.manifest)
    except (OSError,ValueError) as error:result={'status':'failed','issues':[str(error)]}
    print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
