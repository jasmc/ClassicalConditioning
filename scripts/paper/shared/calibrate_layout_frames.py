"""Separate preset comparison; never rewrites selected figure artifacts or state."""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

import argparse
import base64
import copy
import json
import re
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=_COMMAND_REPOSITORY
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from classical_conditioning.figures.layout_frames import SVG, COLLECTOR, POLICY, embed_spec, sha
from scripts.paper.shared.verify_layout_frames import external_destination, review_page
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, state_lock
from classical_conditioning.artifacts import artifact_staging, publish_transaction


def synthetic(width, margin):
    height=width*.34
    def tag(name):return SVG+name
    root=ET.Element(tag('svg'),{'viewBox':f'0 0 {width} {height}','font-family':'DejaVu Sans','font-size':'2.4'})
    allocations=[]
    for index, name in enumerate(('Annotated trace','Heatmap and colourbar','Protocol diagram')):
        x=index*width/3;w=width/3;inner=[x+margin,margin,w-2*margin,height-2*margin]
        group=ET.SubElement(root,tag('g'),{'id':f'example-{index}'})
        allocations.append(dict(id=chr(65+index),kind='panel',box_mm=[x,0,w,height],group_ids=[f'example-{index}']))
        # Marks retain their own dimensions; recomposition changes available space.
        ix,iy,iw,ih=inner
        ET.SubElement(group,tag('text'),{'x':str(ix+.2),'y':str(iy+2.5),'font-size':'2.4'}).text=chr(65+index)
        left=ix+4;right=ix+iw-2;top=iy+5;bottom=iy+ih-5
        if index==0:
            ET.SubElement(group,tag('path'),{'d':f'M {left} {top} V {bottom} H {right}','fill':'none','stroke':'black','stroke-width':'.18'})
            points=' '.join(f'{left+(right-left)*i/29},{(top+bottom)/2+((i*7)%11-5)*.35}' for i in range(30))
            ET.SubElement(group,tag('polyline'),{'points':points,'fill':'none','stroke':'#0072B2','stroke-width':'.25'})
            ET.SubElement(group,tag('text'),{'x':str(left),'y':str(bottom+3),'font-size':'2.3'}).text='Time (s)'
        elif index==1:
            for row in range(8):
                for col in range(12):
                    value=(row*3+col*7)%12
                    ET.SubElement(group,tag('rect'),{'x':str(left+col*(right-left-3)/12),'y':str(top+row*(bottom-top)/8),'width':str((right-left-3)/12+.005),'height':str((bottom-top)/8+.005),'fill':f'rgb({30+value*17},{60+value*10},{170-value*8})'})
            for row in range(12):
                ET.SubElement(group,tag('rect'),{'x':str(right-1),'y':str(top+row*(bottom-top)/12),'width':'1','height':str((bottom-top)/12+.005),'fill':f'rgb({30+row*17},{60+row*10},{170-row*8})'})
            ET.SubElement(group,tag('text'),{'x':str(left),'y':str(bottom+3),'font-size':'2.3'}).text='Trials / time'
        else:
            for row,color in enumerate(('#56B4E9','#E69F00','#009E73')):
                yy=top+row*(bottom-top)/3
                ET.SubElement(group,tag('rect'),{'x':str(left),'y':str(yy),'width':str(right-left),'height':str((bottom-top)/4),'rx':'.4','fill':color})
                ET.SubElement(group,tag('text'),{'x':str(left+.6),'y':str(yy+2.8),'font-size':'2.1'}).text=('Pre-train','Train','Test')[row]
    spec=dict(scope='figure',calibration=True,inset_mm=margin,final_size_mm=[width,height],allocations=allocations,background_ids=[],alignments=[],scientific_clips=[])
    return embed_spec(ET.tostring(root),spec)


def authenticated_excerpts():
    records=[];cases=[]
    for number,panel in ((1,'A'),(2,'ABC')):
        state_path=ROOT/f'docs/paper/figures/figure-{number:02d}/current-state.json'
        state_bytes=state_path.read_bytes();state=json.loads(state_bytes);record=state['draft']['working_svg'];path=Path(record['path'])
        if not path.is_file():
            records.append(dict(figure=number,status='unavailable',path=str(path)));continue
        raw=path.read_bytes()
        if sha(raw)!=record['sha256']:raise ValueError('Manual/current source hash conflict: '+str(path))
        root=ET.fromstring(raw);groups=[n for n in root if n.get('id')=='panel-'+panel]
        if len(groups)!=1:raise ValueError('Selected excerpt group missing')
        # Keep authenticated resources and coordinate system. This is a source
        # excerpt for reference, not a claimed framing-compliant candidate.
        for n in list(root):
            if n not in groups and n.tag not in {SVG+'defs',SVG+'style',SVG+'metadata'}:root.remove(n)
        # Explicit authenticated coordinate regions; never scientific clipping.
        if number==1:root.set('viewBox','0 0 400 620');root.set('height','138mm')
        else:root.set('viewBox','0 0 1800 650');root.set('height','66.0833333333mm')
        root.set('width','89mm' if number==1 else '183mm')
        # show the actual associated legend, supplied separately by caller below
        legend=state['draft'].get('legend')
        if not legend:raise ValueError('Missing associated legend')
        legend_bytes=Path(legend['path']).read_bytes()
        if sha(legend_bytes)!=legend['sha256']:raise ValueError('Legend hash conflict')
        records.append(dict(figure=number,panel=panel,status='authenticated',path=str(path),sha256=sha(raw),state_sha256=sha(state_bytes),legend=legend))
        scoped_legend = re.search(r'\(A\).*?(?=\s*\(B\)|$)', legend_bytes.decode(), re.S).group(0) if number==1 else legend_bytes.decode().split('(D, E)')[0].split('(A)',1)[1]
        cases.append(dict(name=f'Authenticated Figure {number} {panel} reference excerpt — no compliance claim',svg_base64=base64.b64encode(ET.tostring(root)).decode(),reference_only=True,legend=scoped_legend))
    return cases,records


def build(output, refresh=False):
    output=Path(output).resolve()
    manifest_path=output.with_suffix('.json')
    prior=Snapshot([*command_sources(__file__), *([output,manifest_path])])
    if output.is_relative_to('/Volumes') and not Path('/Volumes/JOAQUIM').is_mount():
        raise ValueError('External storage is not mounted')
    if refresh:
        if not output.exists() or not manifest_path.exists():
            raise ValueError('Refresh requires an existing complete calibration pair')
        previous=json.loads(manifest_path.read_text())
        if previous['review_path'] != str(output) or previous['review_sha256'] != sha(output.read_bytes()):
            raise ValueError('Calibration manually changed; preserve before reconciliation')
        if output.is_relative_to(ROOT) or not output.parent.is_dir():
            raise ValueError('Calibration must remain external')
    else:
        output=external_destination(output)
        if manifest_path.exists():raise FileExistsError(manifest_path)
    initial_states = Snapshot([*command_sources(__file__), *([ROOT/f'docs/paper/figures/figure-{n:02d}/current-state.json' for n in (1,2)])])
    references,records=authenticated_excerpts()
    initial_states.check()
    cases=[]
    for width in (89.,183.):
        for margin in (1.,1.5,2.):
            name=f'{width:g} mm wide — {margin:g} mm inset'+(' — provisional recommendation' if margin==1.5 else '')
            cases.append(dict(name=name,svg_base64=base64.b64encode(synthetic(width,margin)).decode()))
    source_paths=[POLICY,COLLECTOR,Path(__file__),ROOT/'scripts/paper/shared/verify_layout_frames.py',
                  ROOT/'src/classical_conditioning/figures/layout_frames.py']
    for record in records:
        source_paths.append(ROOT/f'docs/paper/figures/figure-{record["figure"]:02d}/current-state.json')
        if record['status']=='authenticated':source_paths.extend([Path(record['path']),Path(record['legend']['path'])])
    snapshot=Snapshot([*command_sources(__file__), *(source_paths)])
    for record in records:
        state_path=ROOT/f'docs/paper/figures/figure-{record["figure"]:02d}/current-state.json'
        if record['status']=='authenticated':
            if snapshot.hashes[str(state_path)] != record['state_sha256'] or snapshot.hashes[record['path']] != record['sha256'] or snapshot.hashes[record['legend']['path']] != record['legend']['sha256']:
                raise ValueError('Source changed during calibration capture; preserve and rebuild')
    document=review_page(cases+references,'Figure and panel spacing calibration')
    # Reference panels are excluded from the frame measurement loop.
    document=document.replace('for(const item of cases){document.querySelector', 'for(const item of cases.filter(c=>!c.reference_only)){document.querySelector')
    document=document.replace("section.append(holder);document.querySelector", "section.append(holder);if(item.legend){const legend=document.createElement('pre');legend.textContent=item.legend;section.append(legend);}document.querySelector")
    manifest=dict(review_path=str(output),review_sha256=sha(document.encode()),sources=records,
                  policy_sha256=sha(POLICY.read_bytes()),collector_sha256=sha(COLLECTOR.read_bytes()),bound_inputs=snapshot.hashes,
                  status='calibration pending author selection; no current artifact or pointer changed',cases=cases)
    snapshot.check();prior.check()
    with artifact_staging(output.parent,prefix='.frame-calibration-') as staging:
        staged_html=staging/output.name;staged_manifest=staging/manifest_path.name
        staged_html.write_text(document);staged_manifest.write_text(json.dumps(manifest,indent=2)+'\n')
        def verify():
            snapshot.check()
            if sha(output.read_bytes()) != manifest['review_sha256']:
                raise ValueError('Saved calibration hash mismatch')
        with state_lock(manifest_path):
            snapshot.check();prior.check()
            publish_transaction(((staged_html,output),(staged_manifest,manifest_path)),staging,overwrite=refresh,verify_published=verify)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--refresh',action='store_true',help='Guarded update of the authenticated calibration pair');args=p.parse_args()
    manifest=build(args.output,args.refresh);print(json.dumps({k:v for k,v in manifest.items() if k!='cases'},indent=2))
