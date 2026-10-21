"""Scoped Figure 2 pair-title centring and author-requested row-two view limits."""
from pathlib import Path
import argparse, copy, json, re, sys
import xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot, publish_state, tag, sha
from classical_conditioning.figure_freeze import _transform, _multiply
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme
from classical_conditioning.figures.export import FigureProvenance, FigureMode, export_matplotlib_figure
from main_paper_figures import render as render_page
from assemble_figure2_trace3_matched import block_panel, namespace, D_SOURCE, OUT as INPUT

OUT=INPUT/'axis-refinement'
STATE=REPO/'docs/paper/figures/figure-02/current-state.json'
IDENTITY=(1.,0.,0.,1.,0.,0.)

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def enc(o):return (json.dumps(o,indent=2,allow_nan=False)+'\n').encode()
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
def save(p,data):
    with Path(p).open('xb') as f:f.write(data)

def baseline():
    snapshot=Snapshot([]);snapshot.hashes=load(OUT/'starting-ticket.json')
    snapshot.check(ignore=[REPO/'scripts/assemble_figure2_trace3_matched.py'])
    snapshot.hashes.pop(str((REPO/'scripts/assemble_figure2_trace3_matched.py').resolve()))
    return snapshot

def relative(node,stop,parents):
    chain=[]
    while node is not stop:
        chain.append(node);node=parents[node]
    matrix=IDENTITY
    for n in reversed(chain):matrix=_multiply(matrix,_transform(n.get('transform','')))
    return matrix

def mapped(matrix,x,y):
    a,b,c,d,e,f=matrix;return a*x+c*y+e,b*x+d*y+f

def centre_titles(root):
    lookup={n.get('id'):n for n in root.iter() if n.get('id')};parents={c:p for p in root.iter() for c in p};evidence=[]
    for panel in 'abc':
        owner=lookup['layout-'+panel.upper()];group=lookup[f'fig2__{panel}0__heatmap__label-x']
        text=next(group.iter(tag('text')));extents=[]
        for i in [0,1]:
            background=lookup[f'fig2__{panel}{i}__heatmap__background']
            path=next(background.iter(tag('path')))
            if re.search(r'[ACHQSTVachqstv]',path.get('d')):raise ValueError('Unsupported heatmap rectangle geometry')
            values=list(map(float,re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',path.get('d'))))
            matrix=relative(path,owner,parents)
            extents.extend(mapped(matrix,x,y)[0] for x,y in zip(values[::2],values[1::2]))
        centre=(min(extents)+max(extents))/2
        old_text=relative(text,owner,parents);old_centre=mapped(old_text,float(text.get('x')),float(text.get('y')))[0]
        group_matrix=list(relative(group,owner,parents));group_matrix[4]+=centre-old_centre
        parents[group].remove(group);owner.append(group)
        group.set('transform','matrix('+','.join(f'{v:.15g}' for v in group_matrix)+')')
        group.set('data-scientific-role','axis.label.shared-pair');group.set('data-axis-members',f'{panel}0 {panel}1')
        evidence.append({'panel':panel.upper(),'element_id':group.get('id'),'text_id':text.get('id'),'old_anchor_x_local':old_centre,'pair_centre_x_local':centre,'translation_x_local':centre-old_centre,'owner':owner.get('id'),'label_count':1})
    return evidence

def build():
    baseline();state=load(STATE);draft=state['draft'];source=Path(draft['working_svg']['path'])
    root=ET.fromstring(source.read_bytes());lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    protected={p:sha(ET.tostring(lookup[p])) for p in ['panel-F','panel-G','panel-H','panel-I']}
    rasters={n.get('id'):sha(ET.tostring(n)) for n in root.iter() if n.tag==tag('image')}
    title_evidence=centre_titles(root)
    registry=load(Path(draft['semantic_registry']['path']));alloc={a['id']:a for a in registry['presentation_layout']['allocations']}
    profile=load_presentation_profile('nature-main');tests=pd.read_csv(INPUT/'DE-statistics-Holm36.csv');outside={}
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        for p in ['D','E']:
            if not (OUT/('panel-'+p+'.svg')).exists():
                fig,ids,mappings,extra,size=block_panel(p,alloc[p]['box_mm'][2],profile,theme,tests,y_limits=(-.25,.25),y_ticks=[-.2,-.1,0,.1,.2])
                spec={'scope':'panel','final_size_mm':size,'inset_mm':1.,'allocations':[{'id':p,'kind':'panel','box_mm':[0,0,*size],'group_ids':['axes__'+x.lower()+'__main' for x in ids]+extra}],'background_ids':['canvas-'+p],'data_axes':[],'alignments':[]}
                data_path=D_SOURCE if p=='D' else INPUT/'E-fish-blocks.parquet'
                provenance=FigureProvenance(figure_id='Figure 2 '+p,analysis_recipe='view-limits-only-no-statistical-recomputation',source_file=str(Path(__file__)),source_symbol='build',source_hash=sha256_file(Path(__file__)),reproduction_snippet='.venv-trace/Scripts/python.exe scripts/refine_figure2_axes.py',input_artifacts=(ref(data_path),ref(INPUT/'DE-statistics-Holm36.csv')),artist_mappings=mappings,analysis_identity={'y_limits':[-.25,.25],'data_and_statistics':'unchanged; outside-view values remain in analysis','author_request':'19 October 2026 row-two range -0.25 to 0.25','presentation_profile_inputs':profile.inputs})
                export_matplotlib_figure(fig,OUT/('panel-'+p),provenance,mode=FigureMode.PUBLICATION,panel_ids=ids,formats=('svg',),allow_dirty_publication=True,layout_frames=spec);plt.close(fig)
            data=pd.read_parquet(D_SOURCE if p=='D' else INPUT/'E-fish-blocks.parquet');values=data.loc[data.Eligible,'Fish median response / baseline']
            outside[p]={'below_view':int((values<-.25).sum()),'above_view':int((values>.25).sum()),'data':ref(D_SOURCE if p=='D' else INPUT/'E-fish-blocks.parquet')}
    meta=next(n for n in root.iter() if n.get('id')=='layout-frames');frames=json.loads(meta.text);new_clips=[]
    for p in ['D','E']:
        container=lookup['panel-'+p];oldsvg=next(n for n in container if n.tag==tag('svg'))
        attrs={k:oldsvg.get(k) for k in ['x','y','width','height','overflow']}
        for child in list(container):container.remove(child)
        svg=ET.fromstring((OUT/('panel-'+p+'.svg')).read_bytes());namespace(svg,'matched-'+p+'__')
        for n in list(svg):
            if n.get('id')=='matched-'+p+'__layout-frames':svg.remove(n)
        svg.attrib.update({k:v for k,v in attrs.items() if v is not None});container.append(svg)
        for n in svg.iter():
            if n.tag==tag('clipPath'):new_clips.append({'id':n.get('id'),'evidence':{'purpose':'Author-requested display range [-0.25,0.25]; source values and statistical tests unchanged, outside-view counts recorded','definition_sha256':sha(ET.tostring(n)),'source_svg':ref(OUT/('panel-'+p+'.svg'))}})
    ids={n.get('id') for n in root.iter() if n.get('id')}
    frames['scientific_clips']=[c for c in frames.get('scientific_clips',[]) if c['id'] in ids]+new_clips
    # Existing semantic data rectangle IDs remain stable, and allocations do not move.
    meta.text=json.dumps(frames,sort_keys=True,separators=(',',':'),allow_nan=False)
    for p,h in protected.items():
        if sha(ET.tostring(lookup[p]))!=h:raise ValueError('Unrequested panel changed: '+p)
    for n in root.iter():
        if n.tag==tag('image') and sha(ET.tostring(n))!=rasters[n.get('id')]:raise ValueError('Heatmap pixels changed')
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    legend=Path(draft['legend']['path']).read_text(encoding='utf-8').replace('Thin paired fish; black median/IQR.','Thin paired fish; black median/IQR. D/E display range: −0.25 to 0.25; observations outside this view remain in the analysis.')
    save(OUT/'figure2-axis-refined.svg',svg);save(OUT/'legend.md',legend.encode())
    page=render_page([{'id':'fig2','svg_bytes':svg,'legend_text':legend,'svg':ref(OUT/'figure2-axis-refined.svg'),'legend':ref(OUT/'legend.md'),'whole_figure_frozen':False}])[0]
    save(OUT/'figure2-axis-refined.html',page)
    evidence={'source':ref(source),'svg':ref(OUT/'figure2-axis-refined.svg'),'html':ref(OUT/'figure2-axis-refined.html'),'legend':ref(OUT/'legend.md'),'generator':ref(Path(__file__)),'shared_pair_titles':title_evidence,'row_two_y_limits':[-.25,.25],'row_two_ticks':[-.2,-.1,0,.1,.2],'outside_view_observations':outside,'protected_panel_subtrees':protected,'protected_raster_hashes':rasters,'data_and_statistical_results_unchanged':True,'profile_inputs':profile.inputs}
    save(OUT/'edit-evidence.json',enc(evidence));print(OUT/'figure2-axis-refined.html')

def publish(note):
    snapshot=baseline();evidence=load(OUT/'edit-evidence.json');frames=load(OUT/'layout-evidence.json');centres=load(OUT/'title-centres.json')
    if frames['status']!='passed' or frames['artifact_sha256']!=evidence['svg']['sha256']:raise ValueError('Passing exact frame evidence required')
    if centres['svg_sha256']!=evidence['svg']['sha256'] or any(abs(r['centre_delta_mm'])>.02 for r in centres['titles']):raise ValueError('Exact pair-title centring evidence required')
    state=load(STATE);draft=state['draft'];registry=load(Path(draft['semantic_registry']['path']))
    for k in ['svg','html','legend','generator']:
        a=evidence[k]
        path=Path(a['path'])
        if k=='generator' and sha256_file(path)!=a['sha256']:path=OUT/'refine-build-source.py'
        if sha256_file(path)!=a['sha256']:raise ValueError('Edited candidate: '+a['path'])
    registry['axis_refinement_20261019']={**evidence,'layout_frames':ref(OUT/'layout-evidence.json'),'pair_title_measurements':ref(OUT/'title-centres.json'),'visual_review':note}
    for a in registry['axis_inventory']['axes']:
        if a.get('panel') in ['D','E'] and a.get('orientation')=='y':a['limits']=[-.25,.25];a['tick_values']=[-.2,-.1,0,.1,.2]
        if a.get('panel') in ['A','B','C'] and a.get('orientation')=='x':
            p=a['panel'].lower();a['title_ids']=[f'fig2__{p}0__heatmap__label-x--axis-title-0'];a['shared_axis_title_group']='fig2-'+p+'-heatmap-pair-title';a['shared_title_members']=[p+'0',p+'1']
    for p in ['D','E']:
        registry['matched_panel_registries'][p]=ref(OUT/('panel-'+p+'.figure.json'))
        side=load(OUT/('panel-'+p+'.figure.json'))
        for key,row in side['artist_registry'].items():registry['element_registry']['matched-'+p+'__'+key]={**row,'element_id':'matched-'+p+'__'+key,'panel_id':p.lower(),'source_panel_registry':ref(OUT/('panel-'+p+'.figure.json'))}
    for row in evidence['shared_pair_titles']:
        registry['element_registry'][row['element_id']]={'element_id':row['element_id'],'scientific_role':'axis.label','panel_id':row['panel'].lower(),'quantity':'time from CS onset','units':'s','shared_members':row['element_id'].split('__')[1][0]+'0 '+row['element_id'].split('__')[1][0]+'1','reference':'combined data-axis extent of the named pair','owner':row['owner']}
    registry['axis_inventory']['source_svg_sha256']=evidence['svg']['sha256']
    save(OUT/'semantic-registry.json',enc(registry))
    paths={k:Path(draft[k]['path']) for k in ['working_svg','review','legend']}
    draft['working_svg']={'path':str(paths['working_svg']),'sha256':evidence['svg']['sha256']};draft['review']={'path':str(paths['review']),'sha256':evidence['html']['sha256']};draft['legend']={**draft['legend'],'sha256':evidence['legend']['sha256']};draft['semantic_registry']=ref(OUT/'semantic-registry.json')
    draft['axis_refinement']={'evidence':ref(OUT/'edit-evidence.json'),'layout_frames':ref(OUT/'layout-evidence.json'),'pair_title_measurements':ref(OUT/'title-centres.json'),'visual_review':note}
    draft['reproduction']['last_scoped_edit']={'generator':ref(Path(__file__)),'build_snapshot':ref(OUT/'refine-build-source.py'),'command':'.venv-trace/Scripts/python.exe scripts/refine_figure2_axes.py','view_policy':{'ABC_x_titles':'one shared, centred over each pair','row_two_y_limits':[-.25,.25]}}
    draft['verification']={**draft.get('verification',{}),'layout_frames':ref(OUT/'layout-evidence.json'),'visual_review':{'status':'performed','note':note}}
    draft.setdefault('consistency_requests',[]).append({'id':'fig2-pair-x-titles-and-row2-range-20261019','scope':['A','B','C','D','E','F'],'author_direction':'One shared x-axis title centered on each heatmap pair; second-row range -0.25 to 0.25','properties':{'title_count_per_pair':1,'title_centre_tolerance_mm':.02,'row_two_y_limits':[-.25,.25],'F':'pending; range applies when populated'},'status':'verified for populated panels; F pending','evidence':ref(OUT/'edit-evidence.json'),'render_evidence':ref(OUT/'layout-evidence.json')})
    snapshot.hashes.update(Snapshot([OUT/'edit-evidence.json',OUT/'layout-evidence.json',OUT/'title-centres.json',OUT/'semantic-registry.json',Path(__file__),REPO/'scripts/assemble_figure2_trace3_matched.py',OUT/'refine-build-source.py']).hashes)
    publish_state(STATE,state,snapshot,evidence_updates=[(paths['working_svg'],Path(evidence['svg']['path']).read_bytes()),(paths['review'],Path(evidence['html']['path']).read_bytes()),(paths['legend'],Path(evidence['legend']['path']).read_bytes())])
    print(paths['review'])

if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');parser.add_argument('--visual-review-note');args=parser.parse_args()
    if args.publish:
        if not args.visual_review_note:raise ValueError('Actual visual review required')
        publish(args.visual_review_note)
    else:build()
