"""Scoped translation-only Figure 2 E/B and F/C column alignment."""
from pathlib import Path
import argparse, json, sys, xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from classical_conditioning.figure_workspace import Snapshot, publish_state, sha, tag
from classical_conditioning.artifacts import sha256_file
from main_paper_figures import render
OUT=Path('J:/ClassicalConditioning Outputs/current/reviews/figure-02-3strace/matched-delay-statistics/column-alignment')
STATE=REPO/'docs/paper/figures/figure-02/current-state.json'
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def enc(d):return (json.dumps(d,indent=2,allow_nan=False)+'\n').encode()
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
def save(p,b):
    with Path(p).open('xb') as f:f.write(b)
def snapshot():
    s=Snapshot([]);s.hashes=load(OUT/'starting-ticket.json');s.check();return s
def build():
    snapshot();state=load(STATE);d=state['draft'];root=ET.fromstring(Path(d['working_svg']['path']).read_bytes())
    ids={n.get('id'):n for n in root.iter() if n.get('id')};scale=float(root.get('viewBox').split()[2])/136
    # Browser-measured physical coordinates of named data rectangles and label groups.
    # Pair centres use the union of their two data rectangles, excluding labels.
    geometry={'B_pair':[50.60625486800983,67.14070531402852+12.451286325845787],
        'C_pair':[90.7268606127332,107.26131105875189+12.451279688759705],
        'E_pair':[59.83828674684527,75.2350344445247+13.048093106605268],
        'F_placeholder':[95.33899828151526,95.33899828151526+39.52789183371627],
        'B_letter_left':47.04874421119577,'C_letter_left':87.16934996437332,
        'E_letter_left':47.49120431711656,'F_letter_left':95.3390248298596}
    shifts={p:(sum(geometry[q+'_pair'])-sum(geometry[p+'_pair' if p=='E' else 'F_placeholder']))/2 for p,q in [('E','B'),('F','C')]}
    protected={p:sha(ET.tostring(ids[p])) for p in ['layout-A','layout-B','layout-C','panel-D','panel-G','panel-H','panel-I','fig2__shared__colorbar__axes']}
    nested=next(n for n in ids['panel-E'] if n.tag==tag('svg'))
    nested.set('x',str(float(nested.get('x'))+shifts['E']*scale))
    # Keep panel letters on their column's heading line instead of shifting into D.
    e_ratio=float(nested.get('width'))/float(nested.get('viewBox').split()[2])
    e_letter=(geometry['B_letter_left']-geometry['E_letter_left']-shifts['E'])*scale/e_ratio
    ids['matched-E__E-letter'].set('transform',f'translate({e_letter:.15g} 0) '+ids['matched-E__E-letter'].get('transform',''))
    f=ids['panel-F'];f.set('transform',f'translate({shifts["F"]*scale:.15g} 0) '+f.get('transform',''))
    f_letter=(geometry['C_letter_left']-geometry['F_letter_left']-shifts['F'])*scale/.72
    letter=ids['fig2__f__placeholder__panel-letter'];letter.set('transform',f'translate({f_letter:.15g} 0) '+letter.get('transform',''))
    metadata=ids['layout-frames'];frames=json.loads(metadata.text)
    alloc={x['id']:x for x in frames['allocations']}
    x1=alloc['B']['box_mm'][0];x2=alloc['C']['box_mm'][0]
    for p,x,w in [('D',0,x1),('E',x1,x2-x1),('F',x2,136-x2)]:
        alloc[p]['box_mm'][0]=x;alloc[p]['box_mm'][2]=w
    # Author's new column alignment supersedes outer-content equal spacing in D/E/F.
    old=next(x for x in frames['spacing_groups'] if x['id']=='fig2-DEF-equal-spacing')
    frames['spacing_groups']=[x for x in frames['spacing_groups'] if x['id']!='fig2-DEF-equal-spacing']
    for r in frames['panel_relationships']:
        if r['id']=='fig2-DEF-equal-spacing':
            r['status']='superseded within D/E/F by author-requested E/B and F/C column alignment';r['previous_spacing_group']=r.pop('spacing_group')
    frames['data_axes'].extend([{'id':'E-pair','element_ids':['matched-E__E-control-data-rectangle','matched-E__E-trace-data-rectangle']},
        {'id':'F-placeholder','element_id':'fig2__f__placeholder__frame'}])
    relation={'id':'fig2-E-B-F-C-column-centres-20261019','type':'layout','panels':['B','C','E','F'],
        'properties':['E data-pair centre equals B heatmap-pair centre','F placeholder centre equals C heatmap-pair centre','E/F panel-letter left edges match B/C'],
        'tolerance_mm':.02,'author_direction':'Align E with B and F with C','status':'requires exact saved browser verification'}
    frames['panel_relationships'].append(relation)
    metadata.text=json.dumps(frames,sort_keys=True,separators=(',',':'),allow_nan=False)
    for p,h in protected.items():
        if sha(ET.tostring(ids[p]))!=h:raise ValueError('Unrequested panel changed: '+p)
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True);save(OUT/'figure2-columns.svg',svg)
    legend=Path(d['legend']['path']).read_text(encoding='utf-8')
    page=render([{'id':'fig2','svg_bytes':svg,'legend_text':legend,'svg':ref(OUT/'figure2-columns.svg'),'legend':d['legend'],'whole_figure_frozen':False}])[0]
    save(OUT/'figure2-columns.html',page)
    evidence={'source':d['working_svg'],'svg':ref(OUT/'figure2-columns.svg'),'html':ref(OUT/'figure2-columns.html'),
        'generator':ref(Path(__file__)),'initial_browser_geometry_mm':geometry,'translations_mm':shifts,
        'protected_subtrees':protected,'superseded_spacing':old,'relationship':relation,
        'data_statistics_and_panel_sizes_unchanged':True,'legend_unchanged':d['legend']}
    save(OUT/'edit-evidence.json',enc(evidence));print(json.dumps({'candidate':str(OUT/'figure2-columns.html'),'translations_mm':shifts}))
def publish(note):
    s=snapshot();e=load(OUT/'edit-evidence.json');frame=load(OUT/'layout-evidence.json');alignment=load(OUT/'alignment-evidence.json')
    if frame['status']!='passed' or frame['artifact_sha256']!=e['svg']['sha256']:raise ValueError('Passing frame evidence required')
    if alignment['svg_sha256']!=e['svg']['sha256'] or alignment['status']!='passed':raise ValueError('Passing column evidence required')
    for k in ['svg','html','generator']:
        path=Path(e[k]['path'])
        if k=='generator' and sha256_file(path)!=e[k]['sha256']:path=OUT/'build-source.py'
        if sha256_file(path)!=e[k]['sha256']:raise ValueError('Edited candidate '+k)
    state=load(STATE);d=state['draft'];reg=load(d['semantic_registry']['path'])
    evidence={**e,'layout_frames':ref(OUT/'layout-evidence.json'),'measurements':ref(OUT/'alignment-evidence.json'),'visual_review':note}
    reg['column_alignment_20261019']=evidence
    for relationship in reg.get('panel_relationships',[]):
        if relationship.get('id')=='fig2-DEF-equal-spacing':
            relationship['status']='superseded by author-requested E/B and F/C column alignment'
    reg.setdefault('panel_relationships',[]).append({**e['relationship'],'status':'verified','evidence':ref(OUT/'alignment-evidence.json')})
    # Existing older layout records remain historical, with explicit current precedence.
    reg['column_alignment_20261019']['precedence']='E/B and F/C centres supersede D/E/F equal outer-content gaps; all other prior relationships retained.'
    reg['axis_inventory']['source_svg_sha256']=e['svg']['sha256']
    save(OUT/'semantic-registry.json',enc(reg))
    d['working_svg']={**d['working_svg'],'sha256':e['svg']['sha256']};d['review']={**d['review'],'sha256':e['html']['sha256']};d['semantic_registry']=ref(OUT/'semantic-registry.json')
    d['column_alignment']=evidence;d['verification']['layout_frames']=ref(OUT/'layout-evidence.json')
    d['verification']['visual_review']={'status':'performed','note':note}
    d.setdefault('consistency_requests',[]).append({**e['relationship'],'status':'verified','evidence':ref(OUT/'alignment-evidence.json')})
    d['reproduction']['last_scoped_edit']={'generator':ref(Path(__file__)),'command':'.venv-trace/Scripts/python.exe scripts/align_figure2_columns.py','source_snapshot':str(OUT/'before-working_svg.svg')}
    s.hashes.update(Snapshot([OUT/'edit-evidence.json',OUT/'layout-evidence.json',OUT/'alignment-evidence.json',OUT/'semantic-registry.json',OUT/'build-source.py',Path(__file__)]).hashes)
    publish_state(STATE,state,s,evidence_updates=[(Path(d['working_svg']['path']),Path(e['svg']['path']).read_bytes()),(Path(d['review']['path']),Path(e['html']['path']).read_bytes())])
    print(d['review']['path'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');p.add_argument('--visual-review-note');a=p.parse_args()
    if a.publish:
        if not a.visual_review_note:raise ValueError('Actual visual review required')
        publish(a.visual_review_note)
    else:build()
