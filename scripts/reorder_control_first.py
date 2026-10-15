"""Guarded presentation-only control-first ordering for Figure 1 F–H and Figure 2 D–F.

prepare requires an existing pre-edit starting-ticket.json; check via the standard
browser collector, then publish. Scientific frozen sources and data stay immutable.
"""
from __future__ import annotations
import argparse, copy, json, sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,sha,file_hash,tag,publish_state,measured_presentation
from classical_conditioning.figures.layout_frames import read_spec,embed_spec,assess
from correct_panel_relationships import matrices,move,retile,union,write_candidate
from layout_main_paper_figures import find,encode
from build_figure_legend_review import svg_editing_page


def shift(node,dx,changes):
    node.set('transform',(f'translate({dx:.12g} 0) '+node.get('transform','')).strip())
    changes.append(node.get('id'))


def protected(source,candidate,details):
    allowed=set(details['translated_elements']);letters=details.get('panel_letters',{})
    def inventory(raw,new=False):
        items=[]
        root=ET.fromstring(raw);parents={c:p for p in root.iter() for c in p}
        for node in root.iter():
            if node.get('id')=='layout-frames':continue
            attrs=dict(node.attrib);identity=node.get('id')
            if identity in allowed:attrs.pop('transform',None)
            text=node.text
            letter_key=parents[node].get('id') if node in parents and node.tag==tag('text') else None
            if letter_key in letters:
                if text!=letters[letter_key]['after' if new else 'before']:raise ValueError('Unexpected panel letter')
                text=letters[letter_key]['before']
            items.append((node.tag,json.dumps(attrs,sort_keys=True),text))
        return Counter(items)
    if inventory(source)!=inventory(candidate,True):raise ValueError('Protected SVG payload changed')
    return {'all_elements_preserved':True,'all_raster_resources_and_scientific_coordinates_preserved':True,'allowed_changes':['named group translations and reparenting','three panel-letter text substitutions','explicit frame and presentation mappings'],'numerical_processing_changed':False}


def prepare(directory):
    ticket=directory/'starting-ticket.json';snapshot=Snapshot([]);snapshot.hashes=json.loads(ticket.read_text());snapshot.hashes.pop(str(Path(__file__)),None);snapshot.check()
    snapshot.hashes.update(Snapshot([Path(__file__)]).hashes)
    details={'author_direction':'2026-10-15: Control first in Figure 2 D/E/F and Figure 1 last row Control, Delay, 3 s Trace','figures':{}}
    for number in (1,2):
        statepath=ROOT/f'docs/paper/figures/figure-{number:02}/current-state.json';state=json.loads(statepath.read_text());draft=state['draft'];raw=Path(draft['working_svg']['path']).read_bytes()
        if sha(raw)!=draft['working_svg']['sha256']:raise ValueError('Unrecorded working edits')
        root=ET.fromstring(raw);spec=read_spec(raw);browser=json.loads(Path(draft['layout_frames']['path']).read_text())['browser']
        if browser['artifact_sha256']!=sha(raw):raise ValueError('Stale source layout evidence')
        changes=[];letters={};mapping=[];legend=Path(draft['legend']['path']).read_text()
        if number==1:
            # IDs continue to identify the original immutable source panels.
            # Only displayed panel letters and current presentation mappings change.
            axes={p:find(root,f'edit-FGH-fig1__{p.lower()}__heatmap__axes') for p in 'FGH'}
            a,b=axes['F'],axes['H'];maps=matrices(root)
            dx=321.618898-46.686614
            # Shared Y ticks and phase labels belong at the leftmost heatmap.
            for node in list(a):
                if node.get('id','').endswith(('__axis-y','__phase-label-pre','__phase-label-train','__phase-label-test')):
                    a.remove(node);b.append(node);shift(node,dx,changes)
            # Old H had unlabelled ticks, so move that axis to the now-middle Delay.
            node=find(root,'edit-FGH-fig1__h__heatmap__axis-y');b.remove(node);a.append(node);shift(node,-dx,changes)
            for old,new,condition,fish in [('H','F','Control','20221115_09'),('F','G','Delay','20221115_07'),('G','H','3 s Trace','20230307_12')]:
                owner='layout-'+old;offset=browser['data_axes'][new][0]-browser['data_axes'][old][0];move(root,owner,offset);changes.append(owner)
                heading=find(root,f'edit-FGH-fig1__{old.lower()}__heading__panel-letter');text=next(heading.iter(tag('text')));identity=heading.get('id')
                letters[identity]={'before':old,'after':new};text.text=new
                mapping.append({'display_panel':new,'source_panel':old,'condition':condition,'fish_id':fish,'owner_group':owner,'axis_element':f'edit-FGH-fig1__{old.lower()}__heatmap__background','source_svg':draft['working_svg']})
            allocations={a['id']:a for a in spec['allocations']}
            for item in mapping:
                allocations[item['display_panel']]['group_ids']=[item['owner_group']]
                next(a for a in spec['data_axes'] if a['id']==item['display_panel'])['element_id']=item['axis_element']
            for relationship in spec.get('panel_relationships',[]):
                if relationship.get('type')=='scientific' and relationship['id']=='fig1-DEF-example-data-link':
                    relationship.update(id='fig1-DEG-example-data-link',panels=['D','E','G'],relations=['D and E retain their shared fish, trials and frame times','E colour bands match the Delay heatmap now displayed as G (frozen source panel F)'],presentation_source_panel_mapping={'G':'F'},caution='Source mappings and frozen panel identifiers retain original F; displayed Delay panel is G. Distinct normalizations are unchanged.')
            for old,new in [('rows from F as','rows from G as'),('copied from H and','copied from the shared F–H colourbar and'),('F-matched','G-matched'),('new frozen F’s','frozen Delay source’s'),('retaining F’s','retaining the Delay source’s'),('for Delay (20221115_07), 3 s Trace (20230307_12) and Control (20221115_09)','for Control (F; 20221115_09), Delay (G; 20221115_07) and 3 s Trace (H; 20230307_12)'),('Arrowheads in F','Arrowheads in G'),('marked F rows','marked G rows')]:legend=legend.replace(old,new)
        else:
            for panel,condition in [('D','delay'),('E','trace')]:
                low=panel.lower();exp=find(root,f'edit-{panel}-axes__{low}-{condition}__main');ctrl=find(root,f'edit-{panel}-axes__{low}-control__main');parents={c:p for p in root.iter() for c in p}
                title=find(root,f'edit-{panel}-fig2__{low}__{condition}__condition-title');ctitle=find(root,f'edit-{panel}-fig2__{low}__control__condition-title')
                dx=float(next(ctitle.iter(tag('text'))).get('x'))-float(next(title.iter(tag('text'))).get('x'))
                # Keep shared Y ticks/title and the sole left spine at their
                # current physical position, outside the moved condition body.
                for node in list(exp):
                    if node.get('id','').startswith('shared-DEF-') or node.get('data-source-id','').startswith(('axis-title__','spine__')):
                        exp.remove(node);parents[exp].append(node)
                for node,offset in [(exp,dx),(ctrl,-dx),(title,dx),(ctitle,-dx)]:shift(node,offset,changes)
                next(a for a in spec['data_axes'] if a['id']==panel)['element_id']=f'edit-{panel}-fig2__{low}__control__background'
                mapping.append({'display_panel':panel,'condition_order':['control',condition],'source_svg':draft['working_svg'],'condition_axis_groups':[ctrl.get('id'),exp.get('id')],'between_condition_annotations':'Symmetric ET–ET comparison endpoints remain fixed; within-condition and zero annotations move with their condition'})
            mapping.append({'display_panel':'F','condition_order':['control','10sTrace'],'status':'pending authenticated data; placeholder preserved'})
            legend=legend.replace('(D, E) Block comparisons for Delay and 3 s Trace.','(D, E) Block comparisons for Delay and 3 s Trace, respectively, with Control on the left and the experimental condition on the right.').replace('(F) 10 s Trace block comparison: pending authenticated inputs; no result is shown.','(F) 10 s Trace block comparison: pending authenticated inputs; no result is shown. When populated, Control precedes 10 s Trace.')
        spec['condition_order_mapping']=mapping
        write_candidate(directory,number,root,spec)
        candidate=(directory/f'figure-{number:02}-layout.svg').read_bytes()
        detail={'source_svg':draft['working_svg'],'translated_elements':changes,'panel_letters':letters,'presentation_mapping':mapping}
        detail['protection']=protected(raw,candidate,detail)
        details['figures'][str(number)]=detail
        (directory/f'figure-{number:02}-legend.md').write_text(legend)
    (directory/'ordering.json').write_bytes(encode(details));ticket.write_bytes(encode(snapshot.hashes))
    print('Prepared control-first candidates; original sources authenticated')


def refine(directory):
    details=json.loads((directory/'ordering.json').read_text())
    for number in (1,2):
        path=directory/f'figure-{number:02}-layout.svg';raw=path.read_bytes();root=ET.fromstring(raw);spec=read_spec(raw)
        report=next(iter(json.loads((directory/f'figure-{number:02}-browser.json').read_text()).values()))
        if report['artifact_sha256']!=sha(raw) or report['issues']:raise ValueError('Stale/unresolved browser report')
        painted={k:union(list(v.values())) for k,v in report['allocations'].items()}
        if number==1:retile(spec,painted,['F','G','H'])
        # Conditions move inside unchanged panel bodies; outer grouping retained.
        write_candidate(directory,number,root,spec)
    print('Updated frame allocations from exact browser bounds; recollect evidence')


def publish(directory,note):
    snapshot=Snapshot([]);snapshot.hashes=json.loads((directory/'starting-ticket.json').read_text());snapshot.check()
    details=json.loads((directory/'ordering.json').read_text());snapshot.hashes.update(Snapshot([directory/'ordering.json',Path(__file__)]).hashes)
    for number in (1,2):
        sp=ROOT/f'docs/paper/figures/figure-{number:02}/current-state.json';state=json.loads(sp.read_text());draft=state['draft'];svg=Path(draft['working_svg']['path']);html=Path(draft['review']['path']);lp=Path(draft['legend']['path'])
        cp=directory/f'figure-{number:02}-layout.svg';bp=directory/f'figure-{number:02}-browser.json';lg=directory/f'figure-{number:02}-legend.md';snapshot.hashes.update(Snapshot([cp,bp,lg]).hashes)
        data=cp.read_bytes();browser=next(iter(json.loads(bp.read_text()).values()));evidence=assess(data,browser)
        if evidence['status']!='passed':raise ValueError(evidence['issues'])
        detail=details['figures'][str(number)];detail['protection']=protected(svg.read_bytes(),data,detail)
        evidence.update({'visual_review':{'status':'reviewed','note':note},'source_svg':draft['working_svg'],'protection':detail['protection'],'condition_order':detail['presentation_mapping']})
        ep=directory/f'figure-{number:02}-layout-evidence.json';eb=encode(evidence);ref={'path':str(ep),'sha256':sha(eb)}
        spec=read_spec(data);registry=draft.get('semantic_registry',{});registry_external='path' in registry and 'sha256' in registry
        if registry_external:registry=json.loads(Path(registry['path']).read_text())
        registry['condition_order_mapping']={'artifact_sha256':sha(data),'mapping':detail['presentation_mapping'],'evidence':ref,'source_ids':'Preserved original scientific source identifiers; displayed letters resolved by this mapping'}
        registry['presentation_layout']={**registry.get('presentation_layout',{}),'allocations':spec['allocations'],'panel_relationships':spec.get('panel_relationships',[]),'evidence':ref}
        if number==1:registry['panel_relationships']=spec.get('panel_relationships',[])
        artifacts=[]
        if registry_external:
            rp=directory/f'figure-{number:02}-draft-registry.json';rb=encode(registry);artifacts.append((rp,rb));draft['semantic_registry']={'path':str(rp),'sha256':sha(rb)}
        else:draft['semantic_registry']=registry
        draft.setdefault('consistency_requests',[]).append({'id':f'fig{number}-control-first-20261015','scope':['F','G','H'] if number==1 else ['D','E','F'],'author_direction':details['author_direction'],'status':'verified' if number==1 else 'D/E verified; F order registered, data pending','presentation_mapping':detail['presentation_mapping'],'affected_artifact_hashes':{'svg':sha(data)},'evidence':ref})
        draft['layout_frames']=ref;draft['condition_order']=detail['presentation_mapping']
        presentation=measured_presentation(data);presentation['layout_frames']=ref;pp=directory/f'figure-{number:02}-effective-presentation.json';pb=encode(presentation)
        draft['effective_presentation']={'path':str(pp),'sha256':sha(pb)}
        legend=lg.read_bytes();page=svg_editing_page(data,scope_label=f'Figure {number}',legend=legend.decode()).encode()
        draft['working_svg']={**draft['working_svg'],'sha256':sha(data)};draft['review']={**draft['review'],'sha256':sha(page)};draft['legend']={**draft['legend'],'sha256':sha(legend)}
        draft['reproduction']={'generator':{'path':str(Path(__file__)),'sha256':file_hash(__file__)},'inputs':{'path':str(directory/'ordering.json'),'sha256':file_hash(directory/'ordering.json')},'command':'prepare; standard browser collector; refine and recollect if needed; publish --visual-review-note'}
        history=directory/f'figure-{number:02}-before';history.mkdir(exist_ok=True)
        artifacts += [(history/'selected.svg',svg.read_bytes()),(history/'selected.html',html.read_bytes()),(history/'legend.md',lp.read_bytes()),(history/'current-state.json',sp.read_bytes()),(ep,eb),(pp,pb)]
        publish_state(sp,state,snapshot,artifacts=artifacts,evidence_updates=[(svg,data),(html,page),(lp,legend)])
        for p in (sp,svg,html,lp):snapshot.hashes[str(p.resolve())]=file_hash(p)
    print('Published control-first presentation drafts and matching legends; scientific selections preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','refine','publish']);p.add_argument('--directory',type=Path,required=True);p.add_argument('--visual-review-note');a=p.parse_args()
    if a.command=='publish':
        if not a.visual_review_note:raise ValueError('Actual visual review required')
        publish(a.directory,a.visual_review_note)
    else:globals()[a.command](a.directory)
if __name__=='__main__':main()
