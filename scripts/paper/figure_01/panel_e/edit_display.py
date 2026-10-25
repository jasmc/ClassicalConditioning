"""Guarded E-only display edits using the current trace and selected frozen F rows."""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

from pathlib import Path
import argparse, copy, json, re, sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=_COMMAND_REPOSITORY
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot, bind_loaded_state, file_hash, measured_presentation, publish_state, sha, tag
from classical_conditioning.figures.review_html import svg_editing_page
from classical_conditioning.figures.figure_01.selected_fgh import apply_selected_fgh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--y-limit',type=float,default=3)
    parser.add_argument('--y-min',type=float,help='Lower bound; defaults to the negative upper limit')
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--preview-svg',type=Path)
    parser.add_argument('--visual-review-note')
    parser.add_argument('--remove-baseline-line',action='store_true')
    parser.add_argument('--legend-from-h',action='store_true')
    parser.add_argument('--continuous-cs',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.y_limit) or args.y_limit<=0:raise ValueError('Positive finite y limit required')
    ymin=-args.y_limit if args.y_min is None else args.y_min
    ymax=args.y_limit
    if not np.isfinite(ymin) or ymin>=ymax or not ymin<=0<=ymax:raise ValueError('Range must contain zero and increase')
    limits=[ymin,ymax]
    zero_position='midpoint' if ymin==-ymax else ('bottom' if ymin==0 else 'inside the axis')
    ticket=json.loads(args.baseline.read_text());starting=Snapshot([*command_sources(__file__), *([])]);starting.hashes=ticket['starting_hashes'];starting.check()
    state_path=ROOT/'docs/paper/figures/figure-01/current-state.json';state_bytes=state_path.read_bytes();state=json.loads(state_bytes)
    svg_path=Path(state['draft']['working_svg']['path']);html_path=Path(state['draft']['review']['path']);legend_path=Path(state['draft']['legend']['path'])
    measurement_path=Path(state['draft']['effective_presentation']['path']);panel_path=svg_path.parent/'figure-01-panel-e.html'
    evidence_path=ROOT/'records/figure-reviews/figure-01/panel-e-current-display.json'
    pointer=state['scoped_fgh_freeze']
    paths=[state_path,svg_path,html_path,legend_path,measurement_path,panel_path,evidence_path,Path(__file__),ROOT/'scripts/paper/figure_01/panels_f_g_h/reproduction/publish_one_second.py',ROOT/'scripts/paper/shared/build_legend_review.py',ROOT/'src/classical_conditioning/figure_workspace.py']
    paths.extend(Path(pointer[k]['path']) for k in ('svg','data','freeze_manifest'))
    if state['draft'].get('layout_spacing_override'):paths.append(ROOT/'src/classical_conditioning/figures/figure_01/legacy_layout.py')
    snapshot=Snapshot([*command_sources(__file__), *(paths)]);bind_loaded_state(snapshot,state,state_bytes,state_path)
    raw=svg_path.read_bytes();original=ET.fromstring(raw);root=ET.fromstring(raw)
    selected,band_report,_=apply_selected_fgh(raw,pointer);selected=ET.fromstring(selected)
    # Use the supported F row extractor, while retaining every other current panel.
    old_e=next(n for n in root if n.get('id')=='panel-E');new_e=copy.deepcopy(next(n for n in selected if n.get('id')=='panel-E'))
    at=list(root).index(old_e);root.remove(old_e);root.insert(at,new_e)
    meta=next(n for n in root if n.get('id')=='editable-copy-provenance');provenance=json.loads(meta.text)
    updated_provenance=json.loads(next(n for n in selected if n.get('id')=='editable-copy-provenance').text)
    registry=provenance['scientific_registry']
    for key in list(registry):
        if key.startswith('edit-E-'):del registry[key]
    registry.update({k:v for k,v in updated_provenance['scientific_registry'].items() if k.startswith('edit-E-')})
    nodes={n.get('id'):n for n in root.iter() if n.get('id')};counts={};values_hashes={}
    for trial in (9,17,63,66,93):
        key=f'edit-E-trial-{trial}-trace';record=registry[key];values=np.asarray(record['data_mapping']['scaled_frame_values'],float);finite=np.isfinite(values)
        path=nodes[key];points=re.findall(r'([ML])([-+\d.eE]+),([-+\d.eE]+)',path.get('d'))
        if len(points)!=int(finite.sum()):raise ValueError('Existing frame trace mapping does not match its saved path')
        clip=next(iter(nodes[f'edit-E-trial-{trial}-clip']));y=float(clip.get('y'));height=float(clip.get('height'))
        yy=y+height*(1-(values[finite]-ymin)/(ymax-ymin))
        path.set('d',' '.join(f'{command}{float(x):.5f},{v:.5f}' for (command,x,_),v in zip(points,yy)))
        record['geometry']['d']=path.get('d');record['display_y_limits']=limits
        record['visual_clipping']={'upper_frames':int((values>args.y_limit).sum()),'lower_frames':int((values<ymin).sum()),'numerical_values_changed':False}
        counts[str(trial)]={'finite_frames':int(finite.sum()),**record['visual_clipping']};values_hashes[str(trial)]=sha(values.astype('<f8').tobytes())
        for index,value in enumerate((ymin,(ymin+ymax)/2,ymax)):
            label=nodes[f'edit-E-trial-{trial}__ytick_{index}-label'];label.text=f'{value:g}'.replace('-','−')
            registry[label.get('id')]['geometry'].update(text=label.text)
            tick=registry[f'edit-E-trial-{trial}__ytick_{index}'];tick['geometry']['value']=value
        zero=nodes.get(f'edit-E-trial-{trial}__computational-zero');zy=y+height*(1-(0-ymin)/(ymax-ymin))
        if zero is not None:
            zero.set('y1',str(zy));zero.set('y2',str(zy))
            registry[zero.get('id')]['geometry'].update(y1=str(zy),y2=str(zy))
            if args.remove_baseline_line:
                next(p for p in new_e.iter() if zero in list(p)).remove(zero)
                registry.pop(zero.get('id'))
    if args.legend_from_h:
        scale=nodes['edit-E-colour-scale'];parent=next(p for p in new_e.iter() if scale in list(p))
        parent.remove(scale)
        for child in list(parent):
            if child.get('id','').startswith('edit-E-colour'):parent.remove(child)
        for key in list(registry):
            if key.startswith('edit-E-colour') or key.startswith('edit-E-node-') and key in {n.get('id') for n in scale.iter()}:registry.pop(key)
        h=next(n for n in original.iter() if n.get('id')=='edit-FGH-fig1__h__colorbar__axes')
        cloned=copy.deepcopy(h)
        # Same uniform scale and typography as H at the 183-mm assembly width.
        wrapper=ET.SubElement(parent,tag('g'),{'id':'edit-E-colour-scale','data-reference':'H','transform':'translate(754 30) scale(3.469945355) translate(-466.866142 -44.220472)'})
        wrapper.append(cloned)
        label=next(n for n in cloned.iter() if n.get('id')=='edit-FGH-fig1__h__colorbar__label-y')
        for text in label.iter(tag('text')):
            text.set('x','499');text.set('transform','rotate(-90 499 109.445669)')
        # The colorbar polygons are already bounded by the original rectangle;
        # clipping is redundant and not portable inside a relocated nested SVG.
        for n in cloned.iter():n.attrib.pop('clip-path',None)
        referenced=set()
        for n in cloned.iter():
            for value in n.attrib.values():
                referenced.update(re.findall(r'url\(#([^)]*)\)',value))
                if value.startswith('#') and value.startswith('#edit-FGH-'):referenced.add(value[1:])
        definitions=ET.SubElement(wrapper,tag('defs'))
        for key in referenced:definitions.append(copy.deepcopy(next(n for n in original.iter() if n.get('id')==key)))
        for n in wrapper.iter():
            for key,value in list(n.attrib.items()):n.set(key,value.replace('edit-FGH-','edit-E-H-'))
        for key,record in list(provenance['scientific_registry'].items()):
            if key.startswith('edit-FGH-fig1__h__colorbar__'):
                rec=copy.deepcopy(record);eid=key.replace('edit-FGH-','edit-E-H-');rec.update(element_id=eid,panel_id='e',source_H_element_id=key)
                rec['mappable_ids']=[f'edit-E-trial-{t}-heatmap' for t in (9,17,63,66,93)] if 'mappable_ids' in rec else rec.get('mappable_ids',[])
                registry[eid]=rec
        band_report['E_color_legend_reference']='Exact current H colorbar subtree, uniform scale identical to H; translated only'
    if args.continuous_cs:
        for role,token in [('stimulus.cs.onset','onset'),('stimulus.cs.offset','offset')]:
            originals=[(key,record) for key,record in registry.items() if key.startswith('edit-E-') and record.get('scientific_role')==role]
            first=nodes[originals[0][0]];attrs=dict(first.attrib);attrs.update(id=f'edit-E-continuous-cs-{token}',y1='30',y2='445')
            attrs.pop('data-source-id',None)
            for key,_ in originals:
                old=nodes[key];next(p for p in new_e.iter() if old in list(p)).remove(old);registry.pop(key)
            parent=next(n for n in new_e.iter() if n.tag==tag('svg'))
            line=ET.SubElement(parent,tag('line'),attrs)
            record=copy.deepcopy(originals[0][1]);record.update(element_id=line.get('id'),geometry=attrs,source_element_id=line.get('id'),source_events=[r['event'] for _,r in originals],subpanel_id='shared-cs')
            record['shared_display_policy']='One CS-alignment boundary across all rows and strips; source event times unchanged'
            registry[line.get('id')]=record
    provenance['E_display_override']={'trace_y_limits':limits,'visual_clipping_only':True,'current_F_selection':pointer,'starting_svg_sha256':sha(raw)}
    meta.text=json.dumps(provenance,separators=(',',':'))
    preserved={}
    for old in original:
        if old.get('id','').startswith('panel-') and old.get('id')!='panel-E':
            new=next(n for n in root if n.get('id')==old.get('id'));assert ET.tostring(old)==ET.tostring(new)
            preserved[old.get('id')]=sha(ET.tostring(old))
    upper=sum(c['upper_frames'] for c in counts.values());lower=sum(c['lower_frames'] for c in counts.values())
    legend=legend_path.read_text()
    old='The shared −30 to +30 axis contains all eligible scaled frames, with raw vigor equal to B at the zero midpoint. Numerical values are not clipped.'
    new=f'The shared {ymin:g} to {ymax:g} display axis places raw vigor equal to B at zero at the {zero_position}. The upper plot boundary visually clips {upper} high-valued frames; {lower} frames fall below the lower boundary. All numerical values remain unchanged.'
    if args.remove_baseline_line:new+=' The dashed baseline reference is omitted.'
    if args.legend_from_h:new+=' The color legend is copied from H and applies only to the F-matched color bands.'
    if args.continuous_cs:new+=' Each CS boundary is drawn once continuously across all traces and bands.'
    if old in legend:legend=legend.replace(old,new)
    else:legend=re.sub(r'The shared [^\n.]+ display axis.*?All numerical values remain unchanged\.',new,legend,count=1)
    updated=ET.tostring(root,encoding='utf-8')
    if state['draft'].get('layout_spacing_override'):
        from classical_conditioning.figures.figure_01.legacy_layout import apply_layout_spacing
        updated,_=apply_layout_spacing(updated);root=ET.fromstring(updated)
    page=svg_editing_page(updated,legend=legend).encode()
    scoped=copy.deepcopy(root)
    for node in list(scoped):
        if node.get('id')!='panel-E' and node.tag not in (tag('defs'),tag('rect')):scoped.remove(node)
    bounds=[900,810,900,635];scoped.attrib.update(viewBox=' '.join(map(str,bounds)),width='900',height='635',id='panel-E-scoped-view',**{'aria-label':'Figure 1 panel E: current unfrozen draft'})
    start=legend.index('(E)');end=legend.index('\n\n(F–H)',start);panel_legend=legend[start:end]
    scope_meta=ET.SubElement(scoped,tag('metadata'),{'id':'panel-view-provenance'});scope_meta.text=json.dumps({'source_svg_path':str(svg_path),'source_svg_sha256':sha(updated),'scope':['E'],'bounds':bounds})
    panel_page=svg_editing_page(ET.tostring(scoped,encoding='utf-8'),scope_label='Figure 1 · Panel E',legend=panel_legend).encode()
    report={'svg_sha256':sha(updated),'starting_svg_sha256':sha(raw),'trace_quantity':'raw vigor minus raw-frame median baseline, scaled by unbinned raw baseline-frame percentile spread; no log','trace_y_limits':limits,'frame_counts':counts,'frame_value_hashes':values_hashes,'numerical_values_changed':False,'F_band_match':band_report,'preserved_panel_hashes':preserved,'visual_review':args.visual_review_note or 'pending'}
    if args.preview_svg:args.preview_svg.write_bytes(updated)
    if args.preview_only:print(json.dumps({'upper_clipped_frames':upper,'lower_clipped_frames':lower,'candidate_svg_sha256':sha(updated)}));return
    measurement=json.loads(measurement_path.read_text());measurement.update(measured_presentation(updated));measurement['panel_e_display_update']=report
    consistency=measurement.setdefault('consistency_evidence',{})
    consistency.setdefault('panel_e_trace',{}).update(display_limits=limits,frames_outside_display=upper+lower,visual_clipping=True,numeric_clipping=False)
    consistency['panel_e_scale']={'range':[-.25,.25],'midpoint':0,'tick_labels':['0.25','0.00','−0.25'],'title':'Log vigor relative to baseline'}
    consistency['F_band_match']=band_report
    consistency['artifact_hashes']={'svg':sha(updated),'review':sha(page)}
    request={'id':'fig1-e-zero-positive-axis-20261013','author_direction':'Remove the negative y axis; make it from 0 to positive (6).','scope':['E','F'],'property':'selected F bands and black-trace display range','expected':{'F_selection':pointer['svg'],'trace_y_limits':limits,'zero_midpoint':zero_position=='midpoint','zero_position':zero_position,'visual_clipping_only':True},'status':'verified','evidence_reference':str(evidence_path),'affected_artifact_hashes':{'svg':sha(updated),'review':sha(page)}}
    requests=[r for r in state['draft'].get('consistency_requests',[]) if r['id']!=request['id']]+[request];state['draft']['consistency_requests']=requests;measurement['consistency_requests']=requests
    measurement_bytes=(json.dumps(measurement,indent=2)+'\n').encode();evidence_bytes=(json.dumps(report,indent=2)+'\n').encode()
    state['draft']['working_svg'].update(sha256=sha(updated),source_svg_sha256=sha(raw));state['draft']['review']['sha256']=sha(page);state['draft']['legend']['sha256']=sha(legend.encode());state['draft']['effective_presentation']['sha256']=sha(measurement_bytes)
    state['draft']['semantic_registry']['relationships']['E_trace_y_limits']=limits
    state['draft']['semantic_registry']['E_display_update']={'path':str(evidence_path),'sha256':sha(evidence_bytes)}
    state['draft']['scientific_changes']['E_display']=f'Black trace display range {ymin:g} to {ymax:g}, baseline zero at {zero_position}; upper {upper}/lower {lower} frames visually clipped; saved frame values and normalization unchanged. Bands verified against current frozen F.'
    state['draft']['scientific_changes']['E']=re.sub(r'Shared (?:display )?y limits \[[^]]+\][^.]*\.',f'Shared display y limits [{ymin:g},{ymax:g}]; baseline zero at {zero_position}; {upper} upper and {lower} lower frame values visually clipped, numerical values unchanged.',state['draft']['scientific_changes']['E'])
    state['draft'].setdefault('panel_views',{})['E']={'path':str(panel_path),'sha256':sha(panel_page),'source_svg_sha256':sha(updated),'source_svg_path':str(svg_path),'legend_source_sha256':sha(legend.encode()),'legend_scope_sha256':sha(panel_legend.encode()),'scope':['E'],'source_bounds':bounds}
    state['verification']['visual_review']={'svg_sha256':sha(updated),'observations':report['visual_review']}
    outputs=[(svg_path,updated),(html_path,page),(legend_path,legend.encode()),(measurement_path,measurement_bytes),(panel_path,panel_page),(evidence_path,evidence_bytes)]
    publish_state(state_path,state,snapshot,artifacts=[(p,b) for p,b in outputs if not p.exists()],evidence_updates=[(p,b) for p,b in outputs if p.exists()])
    print(json.dumps({'panel_view':str(panel_path),'upper_clipped_frames':upper,'lower_clipped_frames':lower,'preserved_panels':list(preserved)}))


if __name__=='__main__':main()
