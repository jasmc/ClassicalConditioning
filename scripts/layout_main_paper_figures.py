"""Apply explicit main-paper layout ownership to authenticated selected drafts.

prepare writes external candidates; publish requires passing exact browser evidence.
No scientific freezes or numerical transformations are performed.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot, sha, file_hash, tag, publish_state, state_lock
from classical_conditioning.figures.layout_frames import embed_spec, assess, POLICY, COLLECTOR
from check_layout_frames import review_page
import base64


def encode(value): return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def find(root, identity):
    matches=[n for n in root.iter() if n.get('id')==identity]
    if len(matches)!=1: raise ValueError(f'Expected one element: {identity}')
    return matches[0]

def own(parent, identity, children):
    group=ET.Element(tag('g'),{'id':identity})
    position=min(list(parent).index(c) for c in children)
    for child in children: parent.remove(child);group.append(child)
    parent.insert(position,group)
    return group


def owned_source(data,number):
    root=ET.fromstring(data)
    backgrounds=['assembly-node-1']; groups={}
    for p in ('A','B','C','D') if number==1 else ('D','E','F','G','H','I'):
        groups[p]='panel-'+p
    if number==1:
        e=find(root,'panel-E');groups['E']='layout-E'
        own(e,'layout-E',list(e)[:2]);groups['E-colorbar']='edit-E-independent-colourbar'
        parent=find(root,'edit-FGH-figure_1');backgrounds.append('edit-FGH-fig1__f__canvas__background')
        for p in ('F','G','H'):
            identity='layout-'+p;groups[p]=identity
            own(parent,identity,[c for c in parent if (c.get('id') or '').startswith('edit-FGH-fig1__'+p.lower()+'__') and '__colorbar__' not in c.get('id','') and '__canvas__' not in c.get('id','')])
        groups['FGH-colorbar']='edit-FGH-fig1__h__colorbar__axes'
    else:
        parent=next(c for c in next(iter(find(root,'panel-ABC'))) if c.tag==tag('g'))
        backgrounds.append('fig2__a0__canvas__background')
        for p in ('A','B','C'):
            identity='layout-'+p;groups[p]=identity
            own(parent,identity,[c for c in parent if (c.get('id') or '').startswith(('fig2__'+p.lower()+'0__','fig2__'+p.lower()+'1__')) and '__canvas__' not in c.get('id','')])
        groups['ABC-colorbar']='fig2__shared__colorbar__axes'
    return root,groups,backgrounds


def initialize(ticket, directory):
    if not directory.is_dir() or not Path('/Volumes/JOAQUIM').is_mount(): raise ValueError('Existing mounted external destination required')
    snap=Snapshot([]);snap.hashes=json.loads(ticket.read_text());snap.check()
    snap.hashes[str(Path(__file__).resolve())]=file_hash(__file__)
    snap.hashes[str(COLLECTOR)]=file_hash(COLLECTOR)
    result={}
    for number in (1,2):
        state_path=ROOT/f'docs/paper/figures/figure-{number:02}/current-state.json'
        state=json.loads(state_path.read_text());ref=state['draft']['working_svg']
        root,groups,backgrounds=owned_source(Path(ref['path']).read_bytes(),number)
        dest=directory/f'figure-{number:02}-owned.svg'
        with dest.open('xb') as stream:stream.write(ET.tostring(root))
        result[str(number)]={'path':str(dest),'groups':list(groups.values()),'backgrounds':backgrounds,'owners':groups}
    (directory/'measurement-input.json').write_bytes(encode(result))
    (directory/'starting-ticket.json').write_bytes(encode(snap.hashes))
    print(directory/'measurement-input.json')


def arrange(directory):
    inputs=json.loads((directory/'measurement-input.json').read_text());measure=json.loads((directory/'bounds.json').read_text())
    # Author-selected composition: schematic A/B, session C, paired traces D/E,
    # example heatmaps F/G/H; Figure 2 retains its three-row structure.
    for number in (1,2):
        item=inputs[str(number)];root=ET.fromstring(Path(item['path']).read_bytes());g=item['owners'];bounds=measure[str(number)]
        rows=[['A','B'],['C'],['D','E','E-colorbar'],['F','G','H','FGH-colorbar']] if number==1 else [['A','B','C','ABC-colorbar'],['D','E','F'],['G','H','I']]
        # Tight measured content packing, translations only. Width is never
        # reduced by scaling artwork; an overwide row fails for explicit repair.
        placements={};y=0.;allocations=[]
        for row in rows:
            widths=[bounds[g[p]]['bounds'][2]-bounds[g[p]]['bounds'][0]+2.1 for p in row]
            heights=[bounds[g[p]]['bounds'][3]-bounds[g[p]]['bounds'][1]+2.1 for p in row]
            if sum(widths)>183+1e-6: raise ValueError(f'Figure {number} row {row} needs {sum(widths):.3f} mm; translations cannot fit 183 mm')
            # Preserve existing vertical alignment within a row, and reserve
            # equal top/bottom space from the union of the original row bounds.
            top=min(bounds[g[p]]['bounds'][1] for p in row)
            bottom=max(bounds[g[p]]['bounds'][3] for p in row)
            height=bottom-top+2.1;x=0.
            # Allocate remaining whitespace to the widest plot in the row.
            widths[widths.index(max(widths))]+=183-sum(widths)
            for p,width in zip(row,widths):
                b=bounds[g[p]]['bounds'];factor=bounds[g[p]]['mm_per_unit']
                dx=x+1.05-b[0];dy=y+1.05-top
                node=find(root,g[p]);old=node.get('transform','')
                node.set('transform',(f'translate({dx/factor:.12g} {dy/factor:.12g}) '+old).strip())
                placements[p]={'translation_mm':[dx,dy],'source_bounds_mm':b,'owner_group':g[p]}
                allocations.append({'id':p,'kind':'shared' if 'colorbar' in p else 'panel','box_mm':[x,y,width,height],'group_ids':[g[p]]})
                x+=width
            y+=height
        root.set('viewBox',f'0 0 1800 {y*1800/183:.12g}')
        if number==2:
            # Its explicitly identified canvas background was painted last in
            # the legacy assembly; it must sit behind the newly adjacent rows.
            abc=find(root,'panel-ABC');root.remove(abc)
            root.insert(list(root).index(find(root,'panel-D')),abc)
        # Moving content must not leave it behind an old nested viewport crop.
        # Scientific data-axis clipPaths remain unchanged and explicit below.
        for viewport in root.iter(tag('svg')):
            if viewport is not root:
                viewport.set('overflow','visible')
                viewport.set('style',viewport.get('style','')+';overflow:visible')
        background=find(root,'assembly-node-1');background.set('height',str(y*1800/183))
        # Existing scientific viewport clips are preserved verbatim. Evidence
        # explicitly binds each ID and its selected source rather than assuming
        # clipping is a presentation fix.
        state=json.loads((ROOT/f'docs/paper/figures/figure-{number:02}/current-state.json').read_text())
        source=state['draft']['working_svg']
        clips=[{'id':c.get('id'),'evidence':{'source_svg':source,'definition_sha256':sha(ET.tostring(c)),'purpose':'Existing selected data-axis viewport; preserved unchanged during translation-only layout'}} for c in root.iter(tag('clipPath'))]
        axes=[];alignments=[]
        if number==1:
            for p in ('D','E'):
                for trial,index in ((9,3),(17,5),(63,7),(66,9),(93,11)):
                    axes.append({'id':f'{p}-{trial}','element_id':f'edit-{p}-node-{index}'})
            for trial in (9,17,63,66,93):
                for prop in ('top','width','height'):
                    alignments.append({'id':f'DE-{trial}-{prop}','target':'data_axes','property':prop,'ids':[f'D-{trial}',f'E-{trial}']})
            for p in ('F','G','H'):
                axes.append({'id':p,'element_id':f'edit-FGH-fig1__{p.lower()}__heatmap__background'})
            for prop in ('top','bottom','width','height'):
                alignments.append({'id':'FGH-'+prop,'target':'data_axes','property':prop,'ids':['F','G','H']})
        else:
            for p in ('a0','a1','b0','b1','c0','c1'):
                axes.append({'id':p,'element_id':f'fig2__{p}__heatmap__background'})
            for prop in ('top','bottom','width','height'):
                alignments.append({'id':'ABC-'+prop,'target':'data_axes','property':prop,'ids':['a0','a1','b0','b1','c0','c1']})
            # The plot-body clip rectangles provide exact D/E axis boundaries.
            for p in ('D','E'):
                clip=find(root,f'edit-{p}-p46edbc9d57')
                rect=next(iter(clip))
                if not rect.get('id'):rect.set('id',f'layout-{p}-data-axis')
                axes.append({'id':p,'element_id':rect.get('id')})
            for prop in ('top','bottom','width','height'):
                alignments.append({'id':'DE-'+prop,'target':'data_axes','property':prop,'ids':['D','E']})
        for record in clips:
            record['evidence']['definition_sha256']=sha(ET.tostring(find(root,record['id'])))
        spec={'scope':'figure','final_size_mm':[183,y],'inset_mm':1.0,'allocations':allocations,'background_ids':item['backgrounds'],'scientific_clips':clips,'alignments':alignments,'data_axes':axes}
        data=embed_spec(ET.tostring(root),spec)
        dest=directory/f'figure-{number:02}-layout.svg';dest.write_bytes(data)
        (directory/f'figure-{number:02}-placements.json').write_bytes(encode(placements))
        review=review_page([{'name':dest.name,'svg_base64':base64.b64encode(data).decode()}])
        (directory/f'figure-{number:02}-check.html').write_text(review)
        print(dest)


def refine(directory):
    """Repack from actual painted/unclipped extents, retaining source scale."""
    inputs=json.loads((directory/'measurement-input.json').read_text())
    measurements=json.loads((directory/'bounds.json').read_text())
    for number in (1,2):
        report=next(iter(json.loads((directory/f'figure-{number:02}-browser.json').read_text()).values()))
        if report['issues']:raise ValueError(report['issues'])
        placements=json.loads((directory/f'figure-{number:02}-placements.json').read_text())
        for owner,record in report['allocations'].items():
            boxes=[record['painted_bounds_mm'],record['unclipped_bounds_mm']]
            if any(b is None for b in boxes):raise ValueError('Missing painted content: '+owner)
            dx,dy=placements[owner]['translation_mm']
            measurements[str(number)][inputs[str(number)]['owners'][owner]]['bounds']=[min(b[0] for b in boxes)-dx,min(b[1] for b in boxes)-dy,max(b[0]+b[2] for b in boxes)-dx,max(b[1]+b[3] for b in boxes)-dy]
    (directory/'bounds.json').write_bytes(encode(measurements))


def preserved_content(source,candidate):
    """Compare every scientific drawable's attributes/text, regardless of order."""
    from collections import Counter
    drawable={'path','rect','line','polyline','polygon','circle','ellipse','text','image','use'}
    def inventory(data):
        result=[]
        for n in ET.fromstring(data).iter():
            if n.tag.rsplit('}',1)[-1] not in drawable or n.get('id')=='assembly-node-1':continue
            attrs=dict(n.attrib)
            # Added addresses identify existing clip rectangles; no geometry changes.
            if attrs.get('id','').startswith('layout-') and attrs['id'].endswith('-data-axis'):attrs.pop('id')
            result.append((n.tag,json.dumps(attrs,sort_keys=True),''.join(n.itertext())))
        return Counter(result)
    if inventory(source)!=inventory(candidate):raise ValueError('Scientific drawable payload changed')
    return {'all_drawable_attributes_and_text_preserved':True,'root_background_height_changed':True,'translation_only':True,'scientific_selection_changed':False}


def publish(directory,visual_note):
    from classical_conditioning.figure_workspace import measured_presentation
    from build_figure_legend_review import svg_editing_page
    baseline=json.loads((directory/'starting-ticket.json').read_text())
    # Source tickets predate implementation. Generator/collector revisions are
    # intentional task edits; bind their final bytes as candidate dependencies.
    baseline.pop(str(Path(__file__).resolve()),None);baseline.pop(str(COLLECTOR),None)
    snapshot=Snapshot([]);snapshot.hashes=baseline;snapshot.check()
    dependencies=[Path(__file__),POLICY,COLLECTOR,COLLECTOR.with_name('layout_frames.py'),ROOT/'scripts/build_figure_legend_review.py',ROOT/'src/classical_conditioning/figure_workspace.py']
    correction=None
    if (directory/'correction.json').exists():
        dependencies += [directory/'correction.json',ROOT/'scripts/correct_panel_relationships.py']
        correction=json.loads((directory/'correction.json').read_text())
    snapshots=Snapshot(dependencies);snapshot.hashes.update(snapshots.hashes)
    prepared=[]
    for number in (1,2):
        state_path=ROOT/f'docs/paper/figures/figure-{number:02}/current-state.json'
        state=json.loads(state_path.read_text());draft=state['draft'];svgpath=Path(draft['working_svg']['path']);reviewpath=Path(draft['review']['path'])
        candidate=directory/f'figure-{number:02}-layout.svg';browserpath=directory/f'figure-{number:02}-browser.json'
        snapshot.hashes.update(Snapshot([candidate,browserpath,directory/f'figure-{number:02}-placements.json']).hashes)
        data=candidate.read_bytes();browser=next(iter(json.loads(browserpath.read_text()).values()));evidence=assess(data,browser)
        if evidence['status']!='passed':raise ValueError(evidence['issues'])
        details=correction['figures'][str(number)] if correction else None
        if details:
            from correct_panel_relationships import verify_correction
            if details['source_svg']['sha256']!=file_hash(svgpath):raise ValueError('Correction source changed')
            protection=verify_correction(svgpath.read_bytes(),data,details)
        else:protection=preserved_content(svgpath.read_bytes(),data)
        evidence.update({'visual_review':{'status':'reviewed','note':visual_note},'source_svg':draft['working_svg'],'source_state':{'path':str(state_path),'sha256':file_hash(state_path)},'protection':protection,'placements':json.loads((directory/f'figure-{number:02}-placements.json').read_text())})
        evidencepath=directory/f'figure-{number:02}-layout-evidence.json';eb=encode(evidence)
        layoutref={'path':str(evidencepath),'sha256':sha(eb)}
        if details:
            for relationship in details['panel_relationships']:
                relationship['artifact_sha256']=sha(data)
                relationship['verification_evidence']=layoutref
                if relationship['type']=='layout':relationship['status']='verified'
            spec=json.loads(find(ET.fromstring(data),'layout-frames').text)
            spacing_results={}
            for group in spec['spacing_groups']:
                boxes=browser['data_axes'] if group['target']=='data_axes' else {k:v['painted_bounds_mm'] for k,v in browser['allocations'].items()}
                selected=[boxes[k] for k in group['ids']];axis=0 if group['axis']=='x' else 1
                spacing_results[group['id']]=[b[axis]-a[axis]-a[axis+2] for a,b in zip(selected,selected[1:])]
        presentation=measured_presentation(data);presentation['layout_frames']=layoutref
        presentationpath=directory/f'figure-{number:02}-effective-presentation.json';pb=encode(presentation)
        registry=draft.get('semantic_registry',{})
        layout_mapping={'allocations':json.loads(find(ET.fromstring(data),'layout-frames').text)['allocations'],'evidence':layoutref,'scientific_registry_coverage':'Existing coverage retained; layout ownership does not establish scientific roles'}
        if details:
            layout_mapping.update({'spacing_groups':spec['spacing_groups'],'spacing_measurements_mm':spacing_results,'panel_relationships':details['panel_relationships'],'axis_title_roles':details['axis_titles']})
            title_style={'group_id':'main-paper-axis-titles','scope':['Figure 1','Figure 2'],'expected':{'family':'DejaVu Sans','size_pt':8.0,'weight':'normal','style':'normal'},'roles':details['axis_titles'],'evidence':layoutref}
        registry_artifacts=[]
        if 'path' in registry and 'sha256' in registry:
            rp=Path(registry['path']);snapshot.hashes[str(rp.resolve())]=registry['sha256'];snapshot.check()
            existing=json.loads(rp.read_text());existing['presentation_layout']=layout_mapping
            if details:
                existing['panel_relationships']=details['panel_relationships']
                existing.setdefault('presentation_styles',{})['axis_titles']=title_style
            registry_path=directory/f'figure-{number:02}-draft-registry.json';rb=encode(existing)
            registry_artifacts=[(registry_path,rb)];draft['semantic_registry']={'path':str(registry_path),'sha256':sha(rb)}
        else:
            draft.setdefault('semantic_registry',{})['presentation_layout']=layout_mapping
            if details:
                draft['semantic_registry']['panel_relationships']=details['panel_relationships']
                draft['semantic_registry'].setdefault('presentation_styles',{})['axis_titles']=title_style
        request={'id':f'fig{number}-invisible-frame-layout-20261015','scope':[f'Figure {number}'],'author_direction':'Apply the active 1 mm invisible-frame policy to both main-paper assemblies before collection inclusion','property':'owned allocations, outside margins, adjacent-panel spacing and existing axis alignments','expected':{'inset_mm':1.0,'normal_minimum_content_spacing_mm':2.0,'width_mm':183},'status':'verified','affected_artifact_hashes':{'svg':sha(data)},'evidence':layoutref,'scientific_coverage':'Partial inherited scientific mappings; this verifies presentation layout only'}
        if details:
            request.update({'id':f'fig{number}-related-panel-spacing-axis-titles-20261015','author_direction':correction['author_direction'],'property':'typed panel relationships, equal registered group gaps and axis-title formatting','expected':{'inset_mm':1.0,'spacing_tolerance_mm':0.1,'axis_title_font':'DejaVu Sans','axis_title_size_pt':8.0,'axis_title_weight':'normal'},'relationships':details['panel_relationships'],'axis_title_roles':details['axis_titles'],'style_group':'main-paper-axis-titles'})
        draft.setdefault('consistency_requests',[]).append(request)
        draft['layout_frames']=layoutref;draft['effective_presentation']={'path':str(presentationpath),'sha256':sha(pb)}
        draft['working_svg']={**draft['working_svg'],'sha256':sha(data)}
        legend=Path(draft['legend']['path']).read_text();page=svg_editing_page(data,scope_label=f'Figure {number}',legend=legend).encode()
        draft['review']={**draft['review'],'sha256':sha(page)}
        draft['assembly_scale']={**draft.get('assembly_scale',{}),'final_width_mm':183,'final_height_mm':evidence['final_size_mm'][1]}
        draft['reproduction']={'generator':{'path':str(Path(__file__).resolve()),'sha256':file_hash(__file__)},'layout_sources':{'path':str(directory/'measurement-input.json'),'sha256':file_hash(directory/'measurement-input.json')},'command':'initialize with pre-edit ticket; measure geometry; arrange; collect browser evidence; refine if needed; arrange; collect and visually inspect; publish'}
        draft['status']='Presentation draft with verified 1 mm invisible frames; inherited unfinished scientific scopes retained'
        history=directory/f'figure-{number:02}-before-layout';history.mkdir(exist_ok=True)
        artifacts=[(history/'selected.svg',svgpath.read_bytes()),(history/'selected.html',reviewpath.read_bytes()),(history/'current-state.json',state_path.read_bytes()),(evidencepath,eb),(presentationpath,pb),*registry_artifacts]
        prepared.append((state_path,state,artifacts,[(svgpath,data),(reviewpath,page)]))
    snapshot.check()
    for state_path,state,artifacts,updates in prepared:
        publish_state(state_path,state,snapshot,artifacts=artifacts,evidence_updates=updates)
        # Only this transaction's known writes advance the shared starting ticket.
        for p in [state_path,*[p for p,_ in updates]]:snapshot.hashes[str(p.resolve())]=file_hash(p)
    print('Published both layout drafts; scoped scientific selections unchanged')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('initialize');p.add_argument('--ticket',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p=sub.add_parser('arrange');p.add_argument('--directory',type=Path,required=True)
    p=sub.add_parser('refine');p.add_argument('--directory',type=Path,required=True)
    p=sub.add_parser('publish');p.add_argument('--directory',type=Path,required=True);p.add_argument('--visual-review-note',required=True)
    args=parser.parse_args()
    if args.command=='initialize':initialize(args.ticket,args.directory)
    elif args.command=='arrange':arrange(args.directory)
    elif args.command=='refine':refine(args.directory)
    else:publish(args.directory,args.visual_review_note)
if __name__=='__main__':main()
