"""Selected legacy F/G/H integration, distinct from a publication transaction."""
from pathlib import Path
import base64
import io
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
from classical_conditioning.figure_workspace import tag, file_hash, sha
from classical_conditioning.figures.svg_scope import prefix_svg

def left_align_f_example_arrows(raw):
    """Working-only author override: right-pointing arrows in F's number column."""
    root=ET.fromstring(raw);nodes={n.get('id'):n for n in root.iter() if n.get('id')}
    prefix='edit-FGH-fig1__f__heatmap__'
    label=next(nodes[prefix+'tickmym10-label'].iter(tag('text')))
    number_x=float(label.get('x'))
    bg=next(nodes[prefix+'background'].iter(tag('path')))
    xy=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',bg.get('d'))]
    left=min(xy[::2]);top,bottom=min(xy[1::2]),max(xy[1::2]);records=[]
    tip_x=left-.5;arrow_width=left-number_x-1.5;assert arrow_width>0
    for trial in (9,17,63,66,93):
        node=nodes[prefix+f'example-trial-{trial}'];path=next(node.iter(tag('path')))
        coords=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',path.get('d'))]
        xs,ys=coords[::2],coords[1::2];width=arrow_width;half=width*.32
        centre=top+(trial-4.5)/90*(bottom-top)
        assert abs(ys[0]-centre)<2e-6,'Trial row alignment changed'
        path.set('d',f'M {tip_x:.6f} {centre:.6f} L {tip_x-width:.6f} {centre-half:.6f} L {tip_x-width:.6f} {centre+half:.6f} z')
        assert not node.get('transform'),'Unexpected arrow transform'
        node.set('data-arrow-side','left');node.set('data-arrow-direction','right');node.set('data-display-row',str(trial-4))
        records.append(dict(element_id=node.get('id'),acquisition_trial=trial,display_trial=trial-4,tip_x=tip_x,centre_y=centre,number_anchor_x=number_x,path=path.get('d')))
    for meta in root.iter(tag('metadata')):
        if meta.get('id')=='editable-copy-provenance':
            provenance=json.loads(meta.text)
            for record in records:
                mapped=provenance['scientific_registry'][record['element_id']]
                mapped['geometry'].update(side='left',direction='right',coordinate='trial row centre; arrow column between trial numbers and left heatmap spine',tip_x=tip_x,centre_y=record['centre_y'],path=record['path'])
                mapped['working_presentation_override']=dict(author_direction='Author 2026-10-15: move F arrowheads left, flip, align with trial numbers',frozen_original_preserved=True)
            meta.text=json.dumps(provenance)
    return ET.tostring(root),records

def apply_selected_fgh(raw,pointer):
    """Preserve the existing assembly; replace FGH and the five exact F bands."""
    for key in ['svg','data','freeze_manifest']:
        assert file_hash(pointer[key]['path'])==pointer[key]['sha256'],key
    frozen=Path(pointer['svg']['path']).read_bytes();table=pd.read_csv(pointer['data']['path'])
    root=ET.fromstring(raw);parents={c:p for p in root.iter() for c in p}
    original=next(n for n in root.iter() if n.get('id')=='edit-FGH-node-0')
    replacement=ET.fromstring(prefix_svg(frozen,'edit-FGH'))
    vb=list(map(float,replacement.get('viewBox').split()));width=float(original.get('width'))
    replacement.attrib.update(id='edit-FGH-node-0',x=original.get('x'),y=original.get('y'),width=str(width),height=str(width*vb[3]/vb[2]),preserveAspectRatio='xMidYMid meet',overflow='hidden')
    parent=parents[original];index=list(parent).index(original);parent.remove(original);parent.insert(index,replacement)
    fnode=next(n for n in replacement.iter() if n.get('id')=='edit-FGH--fig1__f__heatmap__vigor-cells')
    # Preserve the established assembly element IDs used by scoped tools.
    for n in replacement.iter():
        for k,v in list(n.attrib.items()):
            if k=='id' or 'url(#' in v or k.endswith('href') and v.startswith('#'):n.set(k,v.replace('edit-FGH--','edit-FGH-'))
    next(n for n in replacement.iter() if n.get('id')=='edit-FGH-fig1__h__colorbar__colour-ramp').set('shape-rendering','crispEdges')
    # Author requested matching the existing Figure 1 bold axis typography.
    # This applies to the working display; immutable frozen SVGs stay intact.
    axis_label_ids=[f'edit-FGH-fig1__{panel}__heatmap__label-x' for panel in 'fgh']+['edit-FGH-fig1__h__colorbar__label-y']
    for element in axis_label_ids:
        group=next(n for n in replacement.iter() if n.get('id')==element)
        for text in group.iter(tag('text')):
            text.set('font-weight','bold');text.set('font-family','DejaVu Sans, sans-serif');text.set('font-style','normal')
            text.set('style',text.get('style','')+';font-family:"DejaVu Sans",sans-serif;font-weight:bold;font-style:normal')
    png=base64.b64decode((fnode.get('href') or fnode.get('{http://www.w3.org/1999/xlink}href')).split(',')[1])
    pixels=np.asarray(Image.open(io.BytesIO(png)).convert('RGBA'))
    ys=((np.arange(90)+.5)*pixels.shape[0]/90).astype(int);xs=((np.arange(40)+.5)*pixels.shape[1]/40).astype(int)
    sampled=pixels[ys[:,None],xs][::-1]
    ftable=table[table.panel.eq('F')];matrix=ftable.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').to_numpy()
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black')
    expected=np.asarray(cmap(np.ma.masked_invalid(np.clip((matrix+.25)/.5,0,1)))*255,dtype=np.uint8)
    assert np.max(np.abs(expected.astype(int)-sampled.astype(int)))<=1
    assert np.array_equal(np.isnan(matrix),np.all(sampled[:,:,:3]==0,axis=2))
    nodes={n.get('id'):n for n in root.iter() if n.get('id')};registry={};bands=[]
    for trial in [9,17,63,66,93]:
        group=nodes[f'edit-E-trial-{trial}-heatmap'];old=list(group);assert len(old) in (40,80)
        x=float(old[0].get('x'));y=float(old[0].get('y'));height=float(old[0].get('height'));width=sum(float(n.get('width')) for n in old)
        for n in old:group.remove(n)
        cells=ftable[ftable.acquisition_trial.eq(trial)].sort_values('bin_index')
        for j,row in enumerate(cells.itertuples()):
            rgb=sampled[trial-5,j,:3];color='#'+''.join(f'{int(c):02x}' for c in rgb)
            element=f'edit-E-trial-{trial}-bin-{j}'
            attrs=dict(id=element,x=str(x+j*width/40),y=str(y),width=str(width/40),height=str(height),fill=color,stroke='none')
            ET.SubElement(group,tag('rect'),attrs)
            registry[element]={'element_id':element,'scientific_role':'heatmap.vigor','figure_id':'fig1','panel_id':'e','svg_type':'rect','geometry':attrs,'data_mapping':dict(fish='20221115_07',trial=trial,display_trial=trial-4,bin_index=j,start_s=row.start_s,end_s=row.end_s,delta_log_vigor=float(row.delta_log_vigor) if np.isfinite(row.delta_log_vigor) else None,source=pointer['data'],color_source='exact frozen F cell-centre RGB'),'normalization':pointer['scientific_definition']}
        bands.append({'acquisition_trial':trial,'display_trial':trial-4,'cells':40,'rgba_sha256':sha(sampled[trial-5].tobytes()),'values_sha256':sha(matrix[trial-5].astype('<f8').tobytes())})
    scale=nodes['edit-E-colour-scale'];rects=[n for n in scale if n.tag==tag('rect')]
    if scale.get('data-reference')!='H':
        assert len(rects)==256
        for j,n in enumerate(rects):
            color=cmap(1-j/255);n.set('fill','#'+''.join(f'{int(c*255):02x}' for c in color[:3]))
        for j in [1,3]:
            for suffix in ['', '-label']:
                n=nodes.get(f'edit-E-colour__ytick_{j}'+suffix)
                if n is not None:parents[n].remove(n)
        for j,label in [(0,'0.25'),(2,'0.00'),(4,'−0.25')]:nodes[f'edit-E-colour__ytick_{j}-label'].text=label
        nodes['edit-E-colour-label'].text='Log vigor relative to baseline'
    current_ids={n.get('id') for n in root.iter() if n.get('id')}
    frozen_registry=json.loads(Path(pointer['freeze_manifest']['path']).read_text())['element_registry']
    for meta in root.iter(tag('metadata')):
        if meta.get('id')=='editable-copy-provenance':
            provenance=json.loads(meta.text)
            mapped={k:v for k,v in provenance['scientific_registry'].items() if k in current_ids and not k.startswith('edit-FGH-') and not (k.startswith('edit-E-') and '-bin-' in k)}
            mapped.update(registry)
            for key,value in frozen_registry.items():
                record=copy.deepcopy(value);record['element_id']='edit-FGH-'+key
                record['source_frozen_element_id']=key;record['source_frozen_svg']=pointer['svg']
                if record['element_id'] in axis_label_ids:
                    record['resolved_style']['font_weight']='bold'
                    record['working_presentation_override']={'font_weight':'bold','font_family':'DejaVu Sans','effective_font_size_pt':8,'authority':'Author 2026-10-13: axes labels must match the other Figure 1 panels in font/bold; frozen scientific originals preserved'}
                mapped[record['element_id']]=record
            for trial in [9,17,63,66,93]:
                key=f'edit-E-trial-{trial}-heatmap'
                if key in mapped:mapped[key].update(normalization=pointer['scientific_definition'],mappable_ids=[f'edit-E-trial-{trial}-bin-{j}' for j in range(40)],source=pointer['data'])
            provenance.update(scientific_registry=mapped,current_scoped_FGH_selection=pointer,structural_inventory_status='Historical source inventory; current scientific registry and measured presentation supersede it')
            meta.text=json.dumps(provenance)
    report={'F_raster_png_sha256':sha(png),'frozen_svg_sha256':pointer['svg']['sha256'],'bands':bands,'range':[-.25,.25],'ticks':[-.25,0,.25],'palette':'managua_r','zero':0,'E_black_trace_changed':False,'D_changed':False,'FGH_display_trials':[1,90],'source':pointer}
    for n in list(root):
        if n.get('id')=='current-fgh-one-second-selection':root.remove(n)
    meta=ET.SubElement(root,tag('metadata'));meta.set('id','current-fgh-one-second-selection');meta.text=json.dumps({'scientific_selection':pointer,'E_linked_band_registry':registry,'verification':report})
    working,arrow_records=left_align_f_example_arrows(ET.tostring(root))
    report['working_example_arrows']=arrow_records
    if any(n.get('id')=='editable-copy-provenance' and 'EH_FGH_alignment_override' in json.loads(n.text) for n in root.iter(tag('metadata'))):
        from classical_conditioning.figures.figure_01.legacy_layout import align_colourbars_and_space_fgh
        working,report['working_row_alignment']=align_colourbars_and_space_fgh(working)
    return working,report,registry
