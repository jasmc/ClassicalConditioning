"""Correct typed panel relationships and named axis-title formatting in drafts.

Use prepare with a pre-edit ticket, collect real browser reports, refine if
needed, collect again and publish through layout_main_paper_figures.py.
"""
from __future__ import annotations
import argparse
import base64
import copy
import json
import math
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot, file_hash, sha, tag, measured_presentation, _viewport
from classical_conditioning.figure_freeze import _multiply, _transform
from classical_conditioning.figures.layout_frames import read_spec, embed_spec, COLLECTOR, POLICY
from layout_main_paper_figures import find, encode
from check_layout_frames import review_page


def matrices(root):
    result={}
    def visit(node,matrix):
        matrix=_multiply(matrix,_transform(node.get('transform','')))
        if node is not root and node.tag==tag('svg'):matrix=_multiply(matrix,_viewport(node))
        result[node]=matrix
        for child in node:visit(child,matrix)
    visit(root,(1.,0.,0.,1.,0.,0.));return result


def move(root,identity,dx_mm):
    node=find(root,identity);factor=matrices(root)[node][0]*183/float(root.get('viewBox').split()[2])
    if not factor>0:raise ValueError('Unsupported translated owner transform')
    node.set('transform',(f'translate({dx_mm/factor:.12g} 0) '+node.get('transform','')).strip())


def axis_titles(root,number):
    # Explicit roles from renderer-owned axis-title groups and known named
    # schematic/trace artists. Tick labels and condition titles are excluded.
    if number==1:
        ids=['edit-B-column-time-label-left','edit-B-column-time-label-right','edit-D-time-label','edit-D-value-label','edit-E-time-label','edit-E-value-label',
             'edit-E-H-fig1__h__colorbar__label-y','edit-FGH-fig1__h__colorbar__label-y',*[f'edit-FGH-fig1__{p}__heatmap__label-x' for p in ('f','g','h')]]
    else:
        ids=[*[f'fig2__{p}0__heatmap__label-x' for p in ('a','b','c')],'fig2__shared__colorbar__label-y','edit-D-axis-title__d-delay__y','fig2__g__native__label-x','fig2__g__native__label-y','edit-H-node-55','edit-H-node-90']
    targets=[]
    for identity in ids:
        try:node=find(root,identity)
        except ValueError:
            if identity=='edit-D-axis-title__d-delay__y':
                node=next(n for n in root.iter() if n.get('data-source-id')=='axis-title__d-delay__y')
            else:raise
        texts=[node] if node.tag==tag('text') else list(node.iter(tag('text')))
        if not texts:raise ValueError('Missing axis-title text: '+identity)
        for i,text in enumerate(texts):targets.append((identity,i,text))
    return targets


def style_titles(root,number):
    unit=183/float(root.get('viewBox').split()[2])*72/25.4;maps=matrices(root);roles=[]
    nodes=list(root.iter())
    for group,index,node in axis_titles(root,number):
        source_index=nodes.index(node);original=dict(node.attrib)
        if not node.get('id'):node.set('id',group+f'--axis-title-{index}')
        m=maps[node];scale=math.hypot(m[2],m[3])*unit
        size=8/scale
        style=re.sub(r'(?:^|;)\s*font-(?:size|family|weight|style)\s*:[^;]*(?=;|$)','',node.get('style',''))
        node.set('style',style+f';font-family:"DejaVu Sans",sans-serif !important;font-size:{size:.12g}px !important;font-weight:normal !important;font-style:normal !important')
        for key,value in {'font-family':'DejaVu Sans','font-weight':'normal','font-style':'normal','font-size':f'{size:.12g}'}.items():node.set(key,value)
        roles.append({'element_id':node.get('id'),'source_index':source_index,'source_attributes':original,'role':'axis.title','renderer_artist':group,'text':''.join(node.itertext()),'expected':{'family':'DejaVu Sans','size_pt':8.0,'weight':'normal','style':'normal'}})
    return roles


def union(boxes):
    x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
    return [x,y,max(b[0]+b[2] for b in boxes)-x,max(b[1]+b[3] for b in boxes)-y]


def equal_gaps(boxes):
    if len(boxes)<3:raise ValueError('Need at least three related panels')
    gap=(boxes[-1][0]+boxes[-1][2]-boxes[0][0]-sum(b[2] for b in boxes))/(len(boxes)-1)
    if gap<2:raise ValueError('Insufficient space for equal gaps')
    result=[];x=boxes[0][0]
    for b in boxes:result.append(x-b[0]);x+=b[2]+gap
    return result


def retile(spec,painted,row):
    allocations={a['id']:a for a in spec['allocations']};start=allocations[row[0]]['box_mm'][0]
    end=allocations[row[-1]]['box_mm'][0]+allocations[row[-1]]['box_mm'][2]
    borders=[start,*[(painted[a][0]+painted[a][2]+painted[b][0])/2 for a,b in zip(row,row[1:])],end]
    for p,left,right in zip(row,borders,borders[1:]):
        box=allocations[p]['box_mm'];box[0]=left;box[2]=right-left


def write_candidate(directory,number,root,spec):
    data=embed_spec(ET.tostring(root),spec);path=directory/f'figure-{number:02}-layout.svg';path.write_bytes(data)
    (directory/f'figure-{number:02}-check.html').write_text(review_page([{'name':path.name,'svg_base64':base64.b64encode(data).decode()}]))


def prepare(ticket,directory):
    if not directory.is_dir() or not Path('/Volumes/JOAQUIM').is_mount():raise ValueError('Existing external storage required')
    initial=json.loads(ticket.read_text());planned=[ROOT/'src/classical_conditioning/figures/layout_frames.py',COLLECTOR,ROOT/'scripts/layout_main_paper_figures.py']
    snapshot=Snapshot([]);snapshot.hashes={p:h for p,h in initial.items() if p not in {str(p) for p in planned}};snapshot.check()
    snapshot.hashes.update(Snapshot([*planned,Path(__file__),ROOT/'scripts/check_layout_frames.py']).hashes)
    correction={'author_direction':'Equal spacing within named related panel groups; keep layout and scientific associations distinct; consistent axis-title formatting across Figures 1 and 2','original_ticket':initial,'intentional_implementation_changes':{str(p):{'before':initial.get(str(p)),'after':file_hash(p)} for p in planned},'figures':{}}
    for number in (1,2):
        statepath=ROOT/f'docs/paper/figures/figure-{number:02}/current-state.json';state=json.loads(statepath.read_text());draft=state['draft'];raw=Path(draft['working_svg']['path']).read_bytes();assert sha(raw)==draft['working_svg']['sha256']
        root=ET.fromstring(raw);spec=read_spec(raw);evidence=json.loads(Path(draft['layout_frames']['path']).read_text());browser=evidence['browser'];assert browser['artifact_sha256']==sha(raw)
        roles=style_titles(root,number);allocations={a['id']:a for a in spec['allocations']};painted={k:union(list(v.values())) for k,v in browser['allocations'].items()}
        if number==1:
            members=['F','G','H'];boxes=[browser['data_axes'][p] for p in members];owners=[allocations[p]['group_ids'][0] for p in members];target='data_axes';ids=members
        else:
            members=['A','B','C'];boxes=[union([browser['data_axes'][p.lower()+'0'],browser['data_axes'][p.lower()+'1']]) for p in members];owners=[allocations[p]['group_ids'][0] for p in members];target='data_axes';ids=[p+'-pair' for p in members]
            spec['data_axes'] += [{'id':p+'-pair','element_ids':[f'fig2__{p.lower()}{i}__heatmap__background' for i in (0,1)]} for p in members]
        translations={}
        for p,owner,dx in zip(members,owners,equal_gaps(boxes)):
            move(root,owner,dx);painted[p][0]+=dx;translations[p]=dx
        retile(spec,painted,members)
        group={'id':f'fig{number}-'+''.join(members)+'-equal-spacing','target':target,'axis':'x','ids':ids,'minimum_gap_mm':2.0}
        spec['spacing_groups']=[group]
        relationships=[{'id':group['id'],'type':'layout','panels':members,'properties':['equal plot-body gaps','aligned data-axis tops/bottoms'],'spacing_group':group,'shared_axis_information':{'axis':'y','quantity':'displayed trial number','mapping':'acquisition trial minus 4','sharing':'common row coordinate and first-panel labels'},'source':'Author identification; authenticated selected SVG and existing data-axis mappings','status':'required'}]
        if number==2:
            for row in (['D','E','F'],['G','H','I']):
                spacing={'id':'fig2-'+''.join(row)+'-equal-spacing','target':'content','axis':'x','ids':row,'minimum_gap_mm':2.0};spec['spacing_groups'].append(spacing)
                relationships.append({'id':spacing['id'],'type':'layout','panels':row,'properties':['equal outer panel-content gaps'],'spacing_group':spacing,'source':'Author identification; explicit owned panel contents','scientific_equivalence':'Not inferred; H ratio differs from G log difference, and placeholders carry no data','status':'required'})
        else:
            semantic=draft.get('semantic_registry',{})
            relationships.append({'id':'fig1-DEF-example-data-link','type':'scientific','panels':['D','E','F'],'relations':['D and E use the same fish, acquisition trials and measured frame times','E black traces use D stored vigor with the documented raw baseline/scaling','E colour bands copy the exact corresponding selected F heatmap rows'],'source_mapping':copy.deepcopy(semantic.get('relationships')),'verification':copy.deepcopy(semantic.get('verification')),'selected_freezes':state['scientific_selection']['records'],'caution':'E black trace and F-matched bands retain distinct normalization and timing definitions; no common layout group is implied','status':'inherited verified mappings retained'})
        spec['panel_relationships']=relationships
        correction['figures'][str(number)]={'axis_titles':roles,'panel_relationships':relationships,'translations_mm':translations,'source_svg':draft['working_svg']}
        (directory/f'figure-{number:02}-placements.json').write_bytes(encode({'source_svg':draft['working_svg'],'translations_mm':translations}))
        write_candidate(directory,number,root,spec)
    (directory/'correction.json').write_bytes(encode(correction));(directory/'original-ticket.json').write_bytes(encode(initial));(directory/'starting-ticket.json').write_bytes(encode(snapshot.hashes))
    (directory/'measurement-input.json').write_bytes(encode({'kind':'typed-panel-relationship-correction','source':'correction.json'}))
    print('Prepared typed relationships, spacing candidates and 8 pt normal DejaVu Sans axis titles')


def refine(directory):
    correction=json.loads((directory/'correction.json').read_text())
    for number in (1,2):
        path=directory/f'figure-{number:02}-layout.svg';raw=path.read_bytes();root=ET.fromstring(raw);spec=read_spec(raw)
        browser=next(iter(json.loads((directory/f'figure-{number:02}-browser.json').read_text()).values()))
        if browser['artifact_sha256']!=sha(raw) or browser['issues']:raise ValueError('Missing/stale browser evidence')
        painted={k:union(list(v.values())) for k,v in browser['allocations'].items()}
        allocations={a['id']:a for a in spec['allocations']}
        if number==2:
            for row in (['D','E','F'],['G','H','I']):
                x=painted[row[0]][0]
                for p in row:
                    dx=x-painted[p][0];move(root,allocations[p]['group_ids'][0],dx);painted[p][0]+=dx
                    old=correction['figures'][str(number)]['translations_mm'].get(p,0);correction['figures'][str(number)]['translations_mm'][p]=old+dx
                    x+=painted[p][2]+2.1
                if x-2.1>182:raise ValueError('Row no longer fits without changing plot scale')
                retile(spec,painted,row)
        retile(spec,painted,['F','G','H'] if number==1 else ['A','B','C'])
        (directory/f'figure-{number:02}-placements.json').write_bytes(encode({'source_svg':correction['figures'][str(number)]['source_svg'],'translations_mm':correction['figures'][str(number)]['translations_mm']}))
        write_candidate(directory,number,root,spec)
    (directory/'correction.json').write_bytes(encode(correction))


def verify_correction(source,candidate,details):
    a=list(ET.fromstring(source).iter());b=list(ET.fromstring(candidate).iter())
    if len(a)!=len(b):raise ValueError('Unexpected element structure change')
    title_indices={r['source_index'] for r in details['axis_titles']};owners={x['group_ids'][0] for x in read_spec(source)['allocations']}
    def nonfont(style):return re.sub(r'(?:^|;)\s*font-(?:size|weight|family|style)\s*:[^;]*(?=;|$)','',style).strip(';')
    for index,(old,new) in enumerate(zip(a,b)):
        if old.tag!=new.tag:raise ValueError('Changed element type')
        if old.get('id')=='layout-frames':continue
        attrs1=dict(old.attrib);attrs2=dict(new.attrib)
        if old.get('id') in owners:attrs1.pop('transform',None);attrs2.pop('transform',None)
        if index in title_indices:
            if nonfont(attrs1.get('style',''))!=nonfont(attrs2.get('style','')):raise ValueError('Non-font style changed')
            for key in ('id','style','font-size','font-weight','font-style','font-family'):attrs1.pop(key,None);attrs2.pop(key,None)
        if attrs1!=attrs2 or old.text!=new.text:raise ValueError('Protected content changed: '+str(old.get('id')))
    report=measured_presentation(candidate);textmap={t['element_id']:t for t in report['texts'] if t['element_id']}
    for role in details['axis_titles']:
        t=textmap[role['element_id']]
        if abs(t['font_height_pt']-8)>.01 or t['weight'] not in ('normal','400') or 'DejaVu Sans' not in t['font_family']:raise ValueError('Axis-title style verification failed')
    return {'scientific_geometry_text_resources_preserved':True,'allowed_changes':['explicit owner translations','registered axis-title font properties and IDs','typed frame relationship metadata'],'axis_titles':len(title_indices),'numerical_processing_changed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','refine']);parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--ticket',type=Path);args=parser.parse_args()
    if args.command=='prepare':prepare(args.ticket,args.directory)
    else:refine(args.directory)
if __name__=='__main__':main()
