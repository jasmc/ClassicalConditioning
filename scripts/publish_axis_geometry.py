"""Guarded publication of measured axis presentation corrections; no science edits."""
from __future__ import annotations
import argparse,base64,copy,json,sys,re
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,file_hash,sha,tag,publish_state,measured_presentation
from classical_conditioning.figure_freeze import _transform
from classical_conditioning.figures.layout_frames import read_spec,embed_spec,assess
from main_paper_figures import reference,encode,authenticate
from register_figure_axes import build_inventory
from check_layout_frames import review_page
from build_figure_legend_review import svg_editing_page
from correct_panel_relationships import matrices,find,union,retile

EXPECTED={'tick_length_pt':2.,'tick_stroke_width_pt':.5,'tick_color':'#000000','tick_linecap':'butt','tick_label_size_pt':7.,'axis_title_size_pt':8.,'tick_label_edge_gap_pt':3.,'title_label_edge_gap_pt':3.,'measurement_tolerance_mm':.01,'font_size_tolerance_pt':.01}

def at(root,path):
    for i in path:root=list(root)[i]
    return root

def verify_protection(source,candidate,manifest,annotations=None,alignment=None):
    a=ET.fromstring(source);b=ET.fromstring(candidate);old=list(a.iter());new=list(b.iter())
    annotations=annotations or {};alignment=alignment or {}
    for identity in annotations.get('added_axis_ids',[]):
        node=find(b,identity);digest=sha(encode([(x.tag,sorted(x.attrib.items()),x.text)for x in node.iter()]))
        if digest!=annotations['added_axis_payloads'][identity]:raise ValueError('Restored E axis payload changed')
        parent=next(x for x in b.iter()if node in list(x));parent.remove(node)
    new=list(b.iter())
    if len(old)!=len(new):raise ValueError('Unexpected element structure change')
    edits={id(at(a,c['xml_index_path'])):c for c in manifest['changes']}
    annotation_edits={id(at(a,c['xml_index_path'])):c for c in annotations.get('changes',[])}
    aligned={id(at(a,c['xml_index_path'])):c for c in alignment.get('changes',[])}
    owners={i for r in read_spec(source)['allocations'] for i in r['group_ids']}
    fonts={'font-size','font-family','font-weight','font-style'}
    tick_styles={'stroke-width','stroke','stroke-linecap','stroke-linejoin','stroke-dasharray','fill','opacity'}
    def css(s):return dict(tuple(x.strip()for x in p.split(':',1)) for p in s.split(';') if ':'in p)
    for x,y in zip(old,new):
        if x.tag!=y.tag:raise ValueError('Changed node type')
        if x.get('id')=='layout-frames':continue
        aa=dict(x.attrib);bb=dict(y.attrib)
        if x is a:
            aa.pop('viewBox',None);bb.pop('viewBox',None)
            aa.pop('height',None);bb.pop('height',None)
        if x.get('id')=='assembly-node-1':aa.pop('height',None);bb.pop('height',None)
        if x.get('id')=='fig2__i__placeholder__frame':aa.pop('width',None);bb.pop('width',None)
        if x.get('id')=='fig2__i__placeholder__pending':aa.pop('x',None);bb.pop('x',None)
        if x.get('id')in owners:
            m=_transform(aa.pop('transform',''));n=_transform(bb.pop('transform',''))
            if any(abs(v-w)>1e-9 for v,w in zip(m[:4],n[:4])):raise ValueError('Owner stretch forbidden')
        if id(x)in edits:
            roles=edits[id(x)]['roles'];allowed=tick_styles if 'axis.tick'in roles else fonts
            ca,cb=css(aa.pop('style','')),css(bb.pop('style',''))
            if {k:v for k,v in ca.items()if k not in allowed}!={k:v for k,v in cb.items()if k not in allowed}:raise ValueError('Protected style changed: '+str(x.get('id')))
            for k in ('id','transform'):aa.pop(k,None);bb.pop(k,None)
        if id(x)in annotation_edits or id(x)in aligned:
            if id(x)in aligned:
                ca,cb=css(aa.pop('style','')),css(bb.pop('style',''))
                if {k:v for k,v in ca.items()if k not in fonts}!={k:v for k,v in cb.items()if k not in fonts}:raise ValueError('Non-font legend style changed')
                for k in ('id','transform'):aa.pop(k,None);bb.pop(k,None)
            for k in ('display','data-semantic-tag','data-scientific-role','data-align'):aa.pop(k,None);bb.pop(k,None)
        expected_text=x.text
        if id(x)in annotation_edits and x.tag==tag('text'):
            from refine_paper_annotations import wording
            expected_text=wording(expected_text or '')
            if 'sample-count-to-caption'in annotation_edits[id(x)]['roles']:expected_text=re.sub(r'\s*\(n=\d+\)','',expected_text)
        if aa!=bb or expected_text!=y.text:raise ValueError('Protected scientific content changed: '+str(x.get('id')))
    return {'scientific_geometry_text_rasters_preserved':True,'allowed':'registered axis leaf style/normal offsets; example-label fonts; whole allocation translations; frame metadata/canvas height; empty I placeholder width and Pending centering','changed_registered_leaves':len(edits)}

def verify_metrics(report,svg):
    if report['artifact_sha256']!=sha(svg):raise ValueError('Stale axis measurements')
    for axis in report['axes']:
        for tick in axis['ticks']:
            if abs(tick['length_pt']-2)>.01 or abs(tick['stroke_width_pt']-.5)>.01 or tick['stroke']!='rgb(0, 0, 0)' or tick['linecap']!='butt':raise ValueError('Inconsistent tick: '+tick['element_id'])
        for label in axis['tick_labels']:
            if abs(label['size_pt']-7)>.01 or 'DejaVu Sans'not in label.get('font_family','') or label.get('font_weight')not in ('normal','400'):raise ValueError('Inconsistent tick font')
        for title in axis.get('titles',[]):
            if abs(title['size_pt']-8)>.01 or 'DejaVu Sans'not in title['font_family'] or title['font_weight']not in ('normal','400'):raise ValueError('Inconsistent axis title font')
        for gap in axis['tick_label_gaps_mm']:
            if abs(gap-3*25.4/72)>.01:raise ValueError(('Inconsistent tick gap',axis['id'],gap))
        if axis['title_gap_mm']is not None and abs(axis['title_gap_mm']-3*25.4/72)>.01:raise ValueError(('Inconsistent title gap',axis['id'],axis['title_gap_mm']))
    for label in report.get('example_labels',[]):
        expected=8 if label['element_id'].endswith('-block')else 7
        if abs(label['size_pt']-expected)>.01 or abs(label['gap_to_y_title_mm']-3*25.4/72)>.01:raise ValueError('Inconsistent example-row label: '+label['element_id'])

def pages(directory):
    for n in (1,2):
        p=directory/f'figure-{n:02}-layout.svg';(directory/f'figure-{n:02}-check.html').write_text(review_page([{'name':p.name,'svg_base64':base64.b64encode(p.read_bytes()).decode()}]))

def repack(directory):
    for n in (1,2):
        path=directory/f'figure-{n:02}-layout.svg';raw=path.read_bytes();root=ET.fromstring(raw);spec=read_spec(raw)
        browser=next(iter(json.loads((directory/f'figure-{n:02}-browser.json').read_text()).values()))
        if browser['artifact_sha256']!=sha(raw):raise ValueError('Stale layout browser measurements')
        bounds={k:union(list(v.values())) for k,v in browser['allocations'].items()};alloc={a['id']:a for a in spec['allocations']}
        # Translate complete rows vertically, retaining named data-axis alignment.
        rows=[['A','B'],['C'],['D','E','E-colorbar'],['F','G','H','FGH-colorbar']] if n==1 else [['A','B','C','ABC-colorbar'],['D','E','F'],['G','H','I']]
        unit=183/float(root.get('viewBox').split()[2]);y=0
        def dxmove(p,dx):
            node=find(root,alloc[p]['group_ids'][0]);m=matrices(root)[node];node.set('transform',f'translate({dx/unit/m[0]:.12g} 0) '+node.get('transform',''));bounds[p][0]+=dx
        if n==1:
            offset=max(0,1.06-bounds['D'][0])
            for p in ['D','E']:dxmove(p,offset)
            dxmove('E-colorbar',min(0,181.94-bounds['E-colorbar'][0]-bounds['E-colorbar'][2]))
        else:
            # I has no data/axes: narrow only its empty presentation frame,
            # retaining the Pending text, to accommodate full-size G/H labels.
            frame=find(root,'fig2__i__placeholder__frame');pending=find(root,'fig2__i__placeholder__pending');scale=unit*matrices(root)[frame][0]
            width=(49.-.5*25.4/72)/scale;frame.set('width',str(width));pending.set('x',str(float(frame.get('x'))+width/2));bounds['I'][2]=49.
        group=['F','G','H']if n==1 else ['A','B','C'];color='FGH-colorbar'if n==1 else 'ABC-colorbar'
        overflow=max(0,bounds[color][0]+bounds[color][2]-181.94)
        for i,p in enumerate(group):dxmove(p,-overflow*i/2)
        dxmove(color,-overflow)
        for row in rows:
            top=min(bounds[p][1]for p in row);bottom=max(bounds[p][1]+bounds[p][3]for p in row);dy=y+1.06-top
            if n==2 and row in (['D','E','F'],['G','H','I']):
                x=1.06
                for p in row:
                    dxmove(p,x-bounds[p][0]);x+=bounds[p][2]+2.12
                if x-2.12>181.94:raise ValueError('Row cannot fit without stretching data')
                retile(spec,bounds,row)
            height=bottom-top+2.12
            for p in row:
                node=find(root,alloc[p]['group_ids'][0]);m=matrices(root)[node];node.set('transform',f'translate(0 {dy/unit/m[3]:.12g}) '+node.get('transform',''));bounds[p][1]+=dy
                alloc[p]['box_mm'][1]=y;alloc[p]['box_mm'][3]=height
            # Retile horizontal borders from exact content bounds, retaining all
            # existing plot-body positions and therefore equal data-body gaps.
            retile(spec,bounds,row)
            y+=height
        root.set('viewBox',f'0 0 {183/unit:.12g} {y/unit:.12g}');find(root,'assembly-node-1').set('height',f'{y/unit:.12g}');spec['final_size_mm']=[183,y]
        path.write_bytes(embed_spec(ET.tostring(root),spec))
    pages(directory)

def publish(directory,note):
    if not Path('/Volumes/JOAQUIM').is_mount():raise ValueError('JOAQUIM unavailable; publication stopped')
    baseline=json.loads((directory/'starting-ticket.json').read_text());snapshot=Snapshot([]);snapshot.hashes=baseline;snapshot.check()
    snapshot.hashes.update(Snapshot([Path(__file__),ROOT/'scripts/style_axis_geometry.cjs',ROOT/'scripts/register_figure_axes.py',ROOT/'src/classical_conditioning/figure_workspace.py']).hashes)
    prepared=[]
    for n in (1,2):
        sp=ROOT/f'docs/paper/figures/figure-{n:02}/current-state.json';raw=sp.read_bytes();state=json.loads(raw);draft=state['draft'];source=authenticate(draft['working_svg'],snapshot)
        cp=directory/f'figure-{n:02}-layout.svg';candidate=cp.read_bytes();snapshot.hashes.update(Snapshot([cp,directory/f'figure-{n:02}-axis-geometry.json',directory/f'figure-{n:02}-axis-geometry-verification.json',directory/f'figure-{n:02}-browser.json']).hashes)
        ap=directory/f'figure-{n:02}-annotation-changes.json';lp=directory/f'figure-{n:02}-annotation-alignment.json';vp=directory/f'figure-{n:02}-annotation-verification.json';annotations=json.loads(ap.read_text())if ap.exists()else {};alignment=json.loads(lp.read_text())if lp.exists()else {}
        if annotations:
            snapshot.hashes.update(Snapshot([ap,lp,vp,ROOT/'scripts/refine_paper_annotations.py',ROOT/'scripts/align_paper_annotations.cjs']).hashes)
            av=json.loads(vp.read_text())
            if av['artifact_sha256']!=sha(candidate)or any(abs(g['right_edge_delta_mm'])>.01 for g in av['legend_groups'])or any(abs(h['center_delta_mm'])>.01 for h in av['condition_headings']):raise ValueError('Legend/heading alignment is stale or failed')
            naming=av['naming']
            if naming['obsolete_trace_labels']or naming['visible_sample_counts']or naming.get('ambiguous_G_heading_visible')or abs(naming.get('E_local_spine_offset_mm',0))>.01:raise ValueError('Heading, nomenclature, counts or local E-axis check failed')
            for g in av['legend_groups']:
                for e in g['entries']:
                    if abs(e['size_pt']-7)>.01 or 'DejaVu Sans'not in e['family']or e['weight']not in ('normal','400'):raise ValueError('Legend text style mismatch')
            baselines={}
            for h in av['condition_headings']:
                for t in h['text_metrics']:
                    if abs(t['size_pt']-8)>.01:raise ValueError('Condition heading font mismatch')
                    baselines.setdefault(h['panel'],[]).append(t['baseline_mm'])
            if any(max(v)-min(v)>.01 for v in baselines.values()):raise ValueError('Condition heading baselines differ')
        metrics=json.loads((directory/f'figure-{n:02}-axis-geometry-verification.json').read_text());verify_metrics(metrics,candidate)
        protection=verify_protection(source,candidate,json.loads((directory/f'figure-{n:02}-axis-geometry.json').read_text()),annotations,alignment)
        browser=next(iter(json.loads((directory/f'figure-{n:02}-browser.json').read_text()).values()));frame=assess(candidate,browser)
        if frame['status']!='passed':raise ValueError(frame['issues'])
        fp=directory/f'figure-{n:02}-layout-evidence.json';fb=encode({**frame,'protection':protection,'visual_review':{'status':'reviewed','note':note}});fr=reference(fp,fb)
        report={'figure_id':state['figure_id'],'artifact_sha256':sha(candidate),'expected':EXPECTED,'measurement_method':'Exact saved SVG at 183 mm in Chromium; geometric glyph bounding edges and tick endpoints in mm; no baseline-pad equivalence assumed','axis_measurements':metrics,'annotation_alignment':av if annotations else None,'scientific_coverage':'Inherited partial; no data, coordinate, estimator, cohort or freeze change','layout_frames':fr,'protection':protection,'visual_review':note,'generator':reference(Path(__file__)),'browser_helper':reference(ROOT/'scripts/style_axis_geometry.cjs')}
        ep=directory/f'figure-{n:02}-axis-evidence.json';eb=encode(report);er=reference(ep,eb)
        reg=draft['semantic_registry'];registry=json.loads(authenticate(reg,snapshot))if 'path'in reg else copy.deepcopy(reg)
        oldlegend=authenticate(draft['legend'],snapshot);legendpath=directory/f'figure-{n:02}-legend.md';legend=legendpath.read_bytes()if legendpath.exists()else oldlegend
        if legendpath.exists():snapshot.hashes.update(Snapshot([legendpath]).hashes)
        draft['legend']=reference(Path(draft['legend']['path']),legend)
        registry['axis_inventory']=build_inventory(candidate,n,draft['legend'],state['scientific_selection']['records'],registry)
        registry['axis_inventory']['registration_evidence']=er
        registry['presentation_styles']['axis_titles']['evidence']=er
        if registry.get('condition_order_mapping'):
            registry['condition_order_mapping']['inherited_evidence']=registry['condition_order_mapping']['evidence']
            registry['condition_order_mapping'].update(artifact_sha256=sha(candidate),evidence=er)
        registry['presentation_styles']['axis_geometry']={'group_id':'main-paper-axis-geometry','scope':['Figure 1','Figure 2'],'expected':EXPECTED,'axes':[a['id']for a in metrics['axes']],'tickless_axes':[a['id']for a in metrics['axes']if a['tickless']],'tickless_policy':'Retain intentionally absent categorical/statistical ticks; no invented scientific marks','evidence':er,'recheck_after_edits':True}
        if n==1:registry['presentation_styles']['example_trial_labels']={'scope':['D','E'],'block_size_pt':8.,'trial_size_pt':7.,'family':'DejaVu Sans','weight':'normal','evidence':er}
        if annotations:
            registry['plot_legends']={'semantic_role':'legend.key','semantic_tags':['plot.legend.condition-key','plot.legend.stimulus-key'],'alignment':'Group right edge coincides with its named plot/reference right edge','font_size_pt':7.,'tolerance_mm':.01,'groups':av['legend_groups'],'evidence':er,'recheck_after_edits':True}
            registry['heading_semantics']={'condition_headings':av['condition_headings'],'experiment_vs_condition':annotations['scientific_scope'],'nomenclature':{'trace_3s':'3sTrace','trace_10s':'10sTrace'},'sample_counts':'Caption rather than color-key or experiment-heading text','evidence':er}
        registry['presentation_layout']['allocations']=read_spec(candidate)['allocations'];registry['presentation_layout']['evidence']=fr
        for relationship in registry.get('panel_relationships',[]):
            if relationship['type']=='layout':relationship.update(artifact_sha256=sha(candidate),verification_evidence=fr,status='verified')
        registry['presentation_layout']['panel_relationships']=registry.get('panel_relationships',[])
        spacing={}
        for group in read_spec(candidate).get('spacing_groups',[]):
            boxes=browser['data_axes']if group['target']=='data_axes'else{k:v['painted_bounds_mm']for k,v in browser['allocations'].items()}
            i=0 if group['axis']=='x'else 1;selected=[boxes[k]for k in group['ids']]
            spacing[group['id']]=[b[i]-a[i]-a[i+2]for a,b in zip(selected,selected[1:])]
        registry['presentation_layout']['spacing_measurements_mm']=spacing
        rp=directory/f'figure-{n:02}-semantic-registry.json';rb=encode(registry)
        draft['semantic_registry']=reference(rp,rb)if 'path'in reg else registry
        draft['working_svg']={**draft['working_svg'],'sha256':sha(candidate)}
        page=svg_editing_page(candidate,scope_label=f'Figure {n}',legend=legend.decode()).encode();draft['review']={**draft['review'],'sha256':sha(page)}
        pp=directory/f'figure-{n:02}-effective-presentation.json';pb=encode({**measured_presentation(candidate),'axis_geometry':er,'layout_frames':fr});draft['effective_presentation']=reference(pp,pb);draft['layout_frames']=fr
        draft['assembly_scale']={**draft.get('assembly_scale',{}),'final_width_mm':183,'final_height_mm':frame['final_size_mm'][1]}
        draft['reproduction']={'generator':reference(Path(__file__)),'browser_helper':reference(ROOT/'scripts/style_axis_geometry.cjs'),'inputs':reference(directory/f'figure-{n:02}-input.json'),'command':'style_axis_geometry.cjs; collect bounds and repack; refine_paper_annotations.py; align_paper_annotations.cjs; regenerate pages; collect fresh layout/axis/annotation verification; publish with actual visual review note'}
        draft.setdefault('consistency_requests',[]).append({'id':f'fig{n}-axis-physical-geometry-20261015','author_direction':'Exact physical axis-title/tick-label/tick gaps; identical horizontal/vertical ticks and D/E example-label sizes','scope':[f'Figure {n}'],'status':'verified-for-registered-axis-artists','expected':EXPECTED,'affected_artifact_hashes':{'svg':sha(candidate)},'evidence':er})
        if annotations:draft['consistency_requests'].append({'id':f'fig{n}-legend-alignment-heading-semantics-20261015','scope':[f'Figure {n}'],'author_direction':'Right-align all plot color keys, tag their semantic role, separate experiment from condition headings, use 3sTrace/10sTrace, move sample counts to captions, remove G Delay experiment heading, retain E local Y boundary/ticks','property':'legend right edges; centered condition headings; experiment/condition meaning; trace naming; sample-count placement','status':'verified-for-registered-artists','evidence':er,'affected_artifact_hashes':{'svg':sha(candidate),'legend':sha(legend)}})
        artifacts=[(fp,fb),(ep,eb),(rp,rb),(pp,pb),(directory/f'figure-{n:02}-before-state.json',raw),(directory/f'figure-{n:02}-before.svg',source),(directory/f'figure-{n:02}-before-legend.md',oldlegend),(directory/f'figure-{n:02}-before.html',authenticate(json.loads(raw)['draft']['review'],snapshot))]
        prepared.append((sp,state,artifacts,[(Path(draft['working_svg']['path']),candidate),(Path(draft['review']['path']),page),(Path(draft['legend']['path']),legend)]))
    snapshot.check()
    for sp,state,artifacts,updates in prepared:
        publish_state(sp,state,snapshot,artifacts=artifacts,evidence_updates=updates)
        for p in [sp,*[p for p,_ in updates]]:snapshot.hashes[str(p.resolve())]=file_hash(p)
    print('Published measured axis geometry for both figures')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['pages','repack','publish']);p.add_argument('--directory',type=Path,required=True);p.add_argument('--visual-review-note');a=p.parse_args()
    if a.command=='publish':
        if not a.visual_review_note:raise ValueError('Actual visual review note required')
        publish(a.directory,a.visual_review_note)
    elif a.command=='repack':repack(a.directory)
    else:pages(a.directory)
if __name__=='__main__':main()
