"""Build or resolve the authenticated, self-contained main-paper figure collection."""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

import argparse
import ast
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
from classical_conditioning.external_artifacts import resolve_artifact, relocated_external_directory, normalized_reference
from classical_conditioning.figure_workspace import Snapshot, file_hash, sha, state_lock, publish_state, StaleFigureState
from classical_conditioning.figures.layout_frames import assess, POLICY, COLLECTOR
STATE=ROOT/'docs/paper/figures/main-paper-current-state.json'
OUTPUT=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/shared/cross-figure-reviews/main-paper-figures/main-paper-figures.html')
DEFAULT_FIGURES=[{'figure_id':f'fig{n}','state':f'docs/paper/figures/figure-{n:02}/current-state.json'} for n in (1,2)]


def encode(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def reference(path,payload=None):return {'path':str(path),'sha256':sha(path.read_bytes() if payload is None else payload)}

def authenticate(ref,snapshot,*,allow_historical=True):
    try:
        p=resolve_artifact(ref['path'],expected_sha256=ref['sha256'],repo=ROOT,allow_historical=allow_historical)
        raw=p.read_bytes()
    except (ValueError,FileNotFoundError) as error:
        raise StaleFigureState('Changed bound artifact: '+str(ref['path'])) from error
    if sha(raw)!=ref['sha256']:raise StaleFigureState('Changed bound artifact: '+str(p))
    snapshot.hashes[str(p.resolve())]=sha(raw)
    return raw


def resolve(state_path=STATE,*,include_recovery=True):
    snapshot=Snapshot([*command_sources(__file__), *([state_path,Path(__file__),POLICY,COLLECTOR,COLLECTOR.with_name('layout_frames.py')])])
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'schema_version':1,'figures':DEFAULT_FIGURES}
    if include_recovery and state.get('recovery'):
        recovery=json.loads(authenticate(state['recovery'],snapshot))
        for ref in recovery['sources']:
            authenticate(ref,snapshot)
    figures=[]
    for selected in state['figures']:
        p=ROOT/selected['state'];raw=p.read_bytes();snapshot.hashes[str(p.resolve())]=sha(raw);s=json.loads(raw)
        if s['figure_id']!=selected['figure_id']:raise ValueError('Figure identity mismatch')
        d=s[s.get('default_display','draft')]
        historical=s.get('default_display','draft')!='draft'
        svg=authenticate(d['working_svg'],snapshot,allow_historical=historical);legend=authenticate(d['legend'],snapshot,allow_historical=historical).decode()
        authenticate(d['review'],snapshot,allow_historical=historical)
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
    if state.get('inputs')!=snapshot_inputs(snapshot,state_path,state['html']['path']):raise StaleFigureState('Collection is stale; run build or recover its authenticated saved wrapper')
    authenticate(state['html'],snapshot)
    if state.get('verification'):
        verification=json.loads(authenticate(state['verification'],snapshot))
        if verification.get('html_sha256')!=state['html']['sha256']:raise StaleFigureState('Stale collection verification')
    snapshot.check()
    return {'html':{**state['html'],'path':str(resolve_artifact(state['html']['path'],expected_sha256=state['html']['sha256'],repo=ROOT))},'figures':[f['id'] for f in figures],'open_target':'built-in-browser','whole_figure_freeze_created':False}


def snapshot_inputs(snapshot,state_path,output_path=OUTPUT):
    excluded={normalized_reference(p,ROOT) for p in (state_path,output_path)}
    result={}
    for path,digest in snapshot.hashes.items():
        key=normalized_reference(path,ROOT)
        if key in excluded:continue
        if key in result and result[key]!=digest:raise ValueError('Ambiguous collection input alias: '+key)
        result[key]=digest
    return result


def validate_saved_wrapper(raw,figures,receipt,authored_builder,current_builder):
    """Authenticate an existing wrapper against its saved and current sources.

    This compares existing SVG/legend bytes and renderer code; it never exports
    HTML or recomputes a figure. A historical builder is parsed, never executed.
    """
    if sha(raw)!=receipt['collection']['html']['sha256']:
        raise ValueError('Saved collection differs from completion receipt')
    if receipt.get('scientific_change') is not False or receipt.get('artwork_regenerated') is not False:
        raise ValueError('A storage-only collection completion receipt is required')
    text=raw.decode('utf-8')
    matches=re.findall(r'<script type="application/json" id="paper-figure-provenance">(.*?)</script>',text,re.S)
    if len(matches)!=1:raise ValueError('Saved collection provenance is missing or ambiguous')
    metadata=json.loads(matches[0])
    if metadata['builder']['sha256']!=sha(authored_builder):raise ValueError('Authored collection builder changed')
    names=('encode','reference','embedded_svg','legend_html','publication_size','render')
    def functions(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    old,new=functions(authored_builder),functions(current_builder)
    if any(old.get(n)!=new.get(n) or n not in old for n in names):
        raise ValueError('Collection rendering logic changed; saved wrapper cannot be rebound')
    svgs=re.findall(r'<div class="sheet">(.*?)</div></template>',text,re.S)
    captions=re.findall(r'<figcaption>(.*?)</figcaption>',text,re.S)
    if not len(figures)==len(metadata['figures'])==len(svgs)==len(captions):
        raise ValueError('Saved collection scope differs from selected figures')
    for index,(saved,current) in enumerate(zip(metadata['figures'],figures)):
        if saved['id']!=current['id']:raise ValueError('Saved collection figure order changed')
        for key in ('state','svg','legend','layout_frames'):
            if saved[key]['sha256']!=current[key]['sha256']:
                raise ValueError('Saved collection source changed: '+current['id']+' '+key)
        if saved['whole_figure_frozen']!=current['whole_figure_frozen']:
            raise ValueError('Saved collection freeze status changed')
        source=receipt['source_identities'][current['id']]
        if source!={'svg':current['svg']['sha256'],'legend':current['legend']['sha256']}:
            raise ValueError('Collection completion source identities changed')
        if (sha(svgs[index].encode())!=saved['embedded_svg_sha256'] or
                sha(embedded_svg(current['svg_bytes'],current['id']+'--').encode())!=saved['embedded_svg_sha256']):
            raise ValueError('Saved embedded artwork differs from selected SVG')
        if captions[index]!=legend_html(current['legend_text']):
            raise ValueError('Saved collection legend differs from selected legend')
    return metadata,list(names)


def recover(*,receipt_path,receipt_sha256,builder_source,evidence_path,state_path=STATE):
    """Publish only recovered repository evidence/state; preserve all SSD bytes."""
    state_path=Path(state_path);evidence_path=Path(evidence_path)
    if not evidence_path.resolve().is_relative_to(ROOT/'records') or not evidence_path.parent.is_dir():
        raise ValueError('Use an existing repository evidence directory')
    if evidence_path.exists():raise FileExistsError('Preserve existing recovery evidence: '+str(evidence_path))
    state,snapshot,figures=resolve(state_path)
    before=state_path.read_bytes()
    receipt_file=resolve_artifact(receipt_path,expected_sha256=receipt_sha256,repo=ROOT)
    receipt_ref=reference(receipt_file);receipt=json.loads(authenticate(receipt_ref,snapshot))
    output_ref=receipt['collection']['html'];raw=authenticate(output_ref,snapshot,allow_historical=False)
    matches=re.findall(r'<script type="application/json" id="paper-figure-provenance">(.*?)</script>',raw.decode('utf-8'),re.S)
    if len(matches)!=1:raise ValueError('Saved collection provenance is missing or ambiguous')
    metadata=json.loads(matches[0])
    builder_file=resolve_artifact(builder_source,expected_sha256=metadata['builder']['sha256'],repo=ROOT)
    builder_ref=reference(builder_file);authored=authenticate(builder_ref,snapshot)
    metadata,functions=validate_saved_wrapper(raw,figures,receipt,authored,Path(__file__).read_bytes())
    sources=[receipt_ref,builder_ref,output_ref]
    if state.get('recovery'):
        previous=json.loads(authenticate(state['recovery'],snapshot))
        sources.extend([*previous['sources'],state['recovery']])
    sources=list({(normalized_reference(ref['path'],ROOT),ref['sha256']):
                  {**ref,'path':normalized_reference(ref['path'],ROOT)} for ref in sources}.values())
    evidence={'schema_version':1,'operation':'recover-authenticated-saved-collection',
              'prior_state_sha256':sha(before),'html':output_ref,
              'sources':sources,
              'authored_provenance':metadata,'unchanged_renderer_functions':functions,
              'current_builder':{**reference(Path(__file__)),'path':normalized_reference(__file__,ROOT)},
              'embedded_svg_and_legend_bytes_match':True,'ssd_bytes_written':False,
              'scientific_selection_changed':False,'freeze_created':False}
    data=encode(evidence)
    state.update({'html':output_ref,'provenance':metadata,
                  'recovery':{'path':normalized_reference(evidence_path,ROOT),'sha256':sha(data)}})
    inputs=snapshot_inputs(snapshot,state_path,output_ref['path'])
    inputs[normalized_reference(evidence_path,ROOT)]=sha(data)
    state['inputs']=inputs
    with ExitStack() as locks:
        for figure in sorted(figures,key=lambda f:f['state']['path']):locks.enter_context(state_lock(Path(figure['state']['path'])))
        publish_state(state_path,state,snapshot,artifacts=[(evidence_path,data)])
    return display_target(state_path)


def build(state_path=STATE,output=None):
    output=relocated_external_directory(OUTPUT) if output is None else Path(output)
    state,snapshot,figures=resolve(state_path)
    if output.is_relative_to('/Volumes') and not Path('/Volumes/JOAQUIM').is_mount():raise ValueError('JOAQUIM is not mounted')
    if not output.parent.is_dir():raise FileNotFoundError(output.parent)
    inputs=snapshot_inputs(snapshot,state_path,output)
    if state.get('html'):
        if resolve_artifact(state['html']['path'],expected_sha256=state['html']['sha256'],repo=ROOT)!=output:raise ValueError('Recorded output differs')
        authenticate(state['html'],snapshot)
        if state.get('inputs')==inputs:return display_target(state_path)
    elif output.exists():raise FileExistsError('Unrecorded output must be preserved: '+str(output))
    if state.get('recovery'):
        # A new build is bound to current selected sources. The prior recovery
        # record remains immutable, but cannot keep binding overwritten HTML.
        state,snapshot,figures=resolve(state_path,include_recovery=False)
        state.pop('recovery')
        inputs=snapshot_inputs(snapshot,state_path,output)
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['build','display-target','recover'])
    parser.add_argument('--receipt',type=Path);parser.add_argument('--receipt-sha256')
    parser.add_argument('--builder-source',type=Path);parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    if args.command=='recover':
        if not all((args.receipt,args.receipt_sha256,args.builder_source,args.evidence)):parser.error('recover requires --receipt, --receipt-sha256, --builder-source and --evidence')
        result=recover(receipt_path=args.receipt,receipt_sha256=args.receipt_sha256,builder_source=args.builder_source,evidence_path=args.evidence)
    else:result=build() if args.command=='build' else display_target()
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
