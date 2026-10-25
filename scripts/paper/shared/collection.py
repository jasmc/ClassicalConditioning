"""Build or resolve the authenticated, self-contained main-paper figure collection."""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

import argparse
from collections import Counter
from contextlib import ExitStack
import html
import json
import math
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET
ROOT=_COMMAND_REPOSITORY
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, file_hash, sha, state_lock, publish_state, StaleFigureState
from classical_conditioning.figures.layout_frames import assess, POLICY, COLLECTOR
STATE=ROOT/'docs/paper/figures/main-paper-current-state.json'
OUTPUT=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/main-paper-figures.html')
DEFAULT_FIGURES=[{'figure_id':f'fig{n}','state':f'docs/paper/figures/figure-{n:02}/current-state.json'} for n in (1,2)]


def encode(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def reference(path,payload=None):return {'path':str(path),'sha256':sha(path.read_bytes() if payload is None else payload)}

def authenticate(ref,snapshot):
    p=Path(ref['path']);raw=p.read_bytes()
    if sha(raw)!=ref['sha256']:raise StaleFigureState('Changed bound artifact: '+str(p))
    snapshot.hashes[str(p.resolve())]=sha(raw)
    return raw


def resolve(state_path=STATE):
    snapshot=Snapshot([*command_sources(__file__), *([state_path,Path(__file__),POLICY,COLLECTOR,COLLECTOR.with_name('layout_frames.py')])])
    state=json.loads(state_path.read_text()) if state_path.exists() else {'schema_version':1,'figures':DEFAULT_FIGURES}
    figures=[]
    for selected in state['figures']:
        p=ROOT/selected['state'];raw=p.read_bytes();snapshot.hashes[str(p.resolve())]=sha(raw);s=json.loads(raw)
        if s['figure_id']!=selected['figure_id']:raise ValueError('Figure identity mismatch')
        d=s[s.get('default_display','draft')]
        svg=authenticate(d['working_svg'],snapshot);legend=authenticate(d['legend'],snapshot).decode()
        authenticate(d['review'],snapshot)
        for ref in s.get('scientific_selection',{}).get('records',[]):authenticate(ref,snapshot)
        frame=d.get('layout_frames')
        if not frame:raise ValueError(s['figure_id']+': verified layout-frame evidence required')
        evidence=json.loads(authenticate(frame,snapshot));assessment=assess(svg,evidence.get('browser'))
        if assessment['status']!='passed':raise ValueError(s['figure_id']+': '+str(assessment['issues']))
        for key in ('artifact_sha256','spec_sha256','policy_sha256','collector_sha256','helper_sha256'):
            if evidence.get(key)!=assessment[key]:raise ValueError('Stale layout evidence: '+key)
        figures.append({'id':s['figure_id'],'state':reference(p,raw),'svg':d['working_svg'],'legend':d['legend'],'layout_frames':frame,'whole_figure_frozen':s.get('whole_figure_frozen',s.get('scientific_selection',{}).get('whole_figure_frozen',False)),'svg_bytes':svg,'legend_text':legend})
    snapshot.check()
    return state,snapshot,figures


def embedded_svg(data,prefix):
    text=data.decode()
    families=set()
    for face in re.findall(r'@font-face\s*\{[^}]*\}',text):
        match=re.search(r'font-family\s*:\s*[\"\']?([^;\"\'}]+)',face)
        if match:families.add(match[1])
    for family in sorted(families,key=len,reverse=True):text=text.replace(family,prefix+family)
    root=ET.fromstring(text);ids={n.get('id'):prefix+n.get('id') for n in root.iter() if n.get('id')}
    counts=Counter(n.get('id') for n in root.iter() if n.get('id'));seen=Counter()
    # Legacy nested Matplotlib exports repeat unreferenced tick-group handles.
    # Give each occurrence a unique embedded address. Referenced duplicates
    # remain ambiguous and must fail rather than redirect scientific resources.
    referenced=set()
    for node in root.iter():
        for key,value in node.attrib.items():
            if key.rsplit('}',1)[-1]=='href' and value.startswith('#'):referenced.add(value[1:])
            referenced.update(re.findall(r'url\([\"\']?#([^\s)\"\']+)',value))
        if node.tag.rsplit('}',1)[-1]=='style':referenced.update(re.findall(r'#([A-Za-z_][\w.-]*)',node.text or ''))
    if any(counts[i]>1 for i in referenced):raise ValueError('Ambiguous referenced SVG ID')
    def refs(value):
        for target in re.findall(r'url\(([^)]+)\)',value):
            if not target.strip().strip('\"\'').startswith(('#','data:')):raise ValueError('External SVG/CSS resource')
        value=re.sub(r'url\(([\"\']?)#([^\s)\"\']+)\1\)',lambda m:'url(#'+ids.get(m[2],m[2])+')',value)
        return value
    for node in root.iter():
        name=node.tag.rsplit('}',1)[-1]
        if name in {'script','foreignObject','animate','animateTransform','set'}:raise ValueError('Unsupported active SVG content')
        for key,value in list(node.attrib.items()):
            if key.lower().startswith('on'):raise ValueError('SVG event handler')
            local=key.rsplit('}',1)[-1]
            if local=='id':
                seen[value]+=1
                node.set(key,ids[value]+(f'--instance-{seen[value]}' if counts[value]>1 else ''));continue
            if local=='href':
                if not value.startswith(('#','data:')):raise ValueError('External SVG resource')
                if value.startswith('#'):value='#'+ids[value[1:]]
            if local in {'aria-labelledby','aria-describedby'}:value=' '.join(ids.get(v,v) for v in value.split())
            node.set(key,refs(value))
        if name=='style':
            css=node.text or ''
            if '@import' in css:raise ValueError('External CSS import')
            for target in re.findall(r'url\(([^)]+)\)',css):
                if not target.strip().strip('\"\'').startswith(('#','data:')):raise ValueError('External CSS resource')
            css=refs(css)
            css=re.sub(r'#([A-Za-z_][\w.-]*)',lambda m:'#'+ids.get(m[1],m[1]),css)
            node.text=css
    # The enclosing shadow root isolates selectors and embedded font faces.
    return ET.tostring(root,encoding='unicode')


def legend_html(text):
    paragraphs=[p for p in text.strip().split('\n\n') if not p.lstrip().startswith('#')]
    if not paragraphs:raise ValueError('Empty legend')
    return '<p><strong>'+html.escape(paragraphs[0])+'</strong></p>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs[1:])


def publication_size(data):
    root=ET.fromstring(data);vb=[float(v)for v in root.get('viewBox').split()]
    native=root.get('width','183mm')
    if not re.fullmatch(r'[0-9]+(?:[.][0-9]+)?mm',native):raise ValueError('Publication width must use mm')
    width=float(native[:-2]);height=width*vb[3]/vb[2]
    if not all(math.isfinite(v)and v>0 for v in [width,height]):raise ValueError('Invalid publication size')
    return width,height


def render(figures):
    sections=[];provenance=[];fonts=[];widths=[]
    for f in figures:
        svg=embedded_svg(f['svg_bytes'],f['id']+'--')
        fonts.extend(re.findall(r'@font-face\s*\{[^}]*\}',svg))
        width,height=publication_size(f['svg_bytes']);widths.append(width)
        sections.append('<figure id="'+f['id']+'" style="--native-width:'+str(width)+'mm"><div class="artwork" data-width-mm="'+str(width)+'" style="aspect-ratio:'+str(width)+'/'+str(height)+'"><template shadowrootmode="open"><style>:host{display:block;width:100%;height:100%}.sheet{width:'+str(width)+'mm;height:'+str(height)+'mm;transform-origin:top left;transform:scale(var(--paper-scale,1))}svg{display:block;width:100%;height:100%}</style><div class="sheet">'+svg+'</div></template></div><figcaption>'+legend_html(f['legend_text'])+'</figcaption></figure>')
        provenance.append({**{k:v for k,v in f.items() if k not in ('svg_bytes','legend_text')},'embedded_svg_sha256':sha(svg.encode()),'publication_size_mm':[width,height]})
    metadata={'figures':provenance,'builder':reference(Path(__file__)),'layout_policy':reference(POLICY),'scientific_status':'Presentation collection; scope-specific freezes remain authoritative'}
    css='''html{background:white;color:#111}body{margin:0}main{width:COLLECTION_WIDTHmm;max-width:calc(100% - 24px);margin:24px auto}figure{width:var(--native-width);max-width:100%;margin:0 auto 12mm}figcaption{font:9pt/1.35 "Times New Roman",Times,serif;margin-top:3mm}figcaption p{margin:0 0 1.8mm}figcaption p:first-child{break-after:avoid}.artwork{width:100%;break-inside:avoid;page-break-inside:avoid}@page{size:A4;margin:10mm 13.5mm}@media print{main{width:COLLECTION_WIDTHmm;max-width:none;margin:0}figure{margin-bottom:8mm}.artwork{width:var(--native-width)}figcaption{orphans:3;widows:3}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}'''
    css=css.replace('COLLECTION_WIDTH',format(max(widths),'g'))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Main paper figures</title><style>'+css+'\n'+'\n'.join(fonts)+'</style></head><body><main>'+''.join(sections)+'</main><script type="application/json" id="paper-figure-provenance">'+json.dumps(metadata).replace('<','\\u003c')+'</script><script>const fit=()=>{for(const el of document.querySelectorAll(".artwork"))el.style.setProperty("--paper-scale",el.getBoundingClientRect().width/(Number(el.dataset.widthMm)*96/25.4));};const observer=new ResizeObserver(fit);document.querySelectorAll(".artwork").forEach(el=>observer.observe(el));addEventListener("beforeprint",fit);fit();</script></body></html>').encode(),metadata


def display_target(state_path=STATE):
    state,snapshot,figures=resolve(state_path)
    if state.get('inputs')!=snapshot_inputs(snapshot,state_path):raise StaleFigureState('Collection is stale; run build')
    authenticate(state['html'],snapshot)
    if state.get('verification'):
        verification=json.loads(authenticate(state['verification'],snapshot))
        if verification.get('html_sha256')!=state['html']['sha256']:raise StaleFigureState('Stale collection verification')
    snapshot.check()
    return {'html':state['html'],'figures':[f['id'] for f in figures],'open_target':'built-in-browser','whole_figure_freeze_created':False}


def snapshot_inputs(snapshot,state_path):
    return {p:h for p,h in snapshot.hashes.items() if p not in (str(state_path.resolve()),str(OUTPUT.resolve()))}


def build(state_path=STATE,output=OUTPUT):
    state,snapshot,figures=resolve(state_path)
    if output.is_relative_to('/Volumes') and not Path('/Volumes/JOAQUIM').is_mount():raise ValueError('JOAQUIM is not mounted')
    if not output.parent.is_dir():raise FileNotFoundError(output.parent)
    inputs=snapshot_inputs(snapshot,state_path)
    if state.get('html'):
        if Path(state['html']['path'])!=output:raise ValueError('Recorded output differs')
        authenticate(state['html'],snapshot)
        if state.get('inputs')==inputs:return display_target(state_path)
    elif output.exists():raise FileExistsError('Unrecorded output must be preserved: '+str(output))
    snapshot.hashes[str(output.resolve())]=file_hash(output)
    data,metadata=render(figures)
    state.pop('verification',None)  # A changed collection needs a new rendering review.
    state.update({'html':reference(output,data),'inputs':inputs,'provenance':metadata,'refresh_policy':'Follow selected figure states on plural paper-figure requests','scope':'main-paper-figures','layout_verified':True})
    # Same locks/order as figure publication. The collection writer never edits
    # individual state. publish_state owns the collection lock and rollback.
    with ExitStack() as stack:
        for f in sorted(figures,key=lambda f:f['state']['path']):stack.enter_context(state_lock(Path(f['state']['path'])))
        publish_state(state_path,state,snapshot,evidence_updates=[(output,data)])
    return display_target(state_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['build','display-target']);args=parser.parse_args()
    print(json.dumps(build() if args.command=='build' else display_target(),indent=2))
if __name__=='__main__':main()
