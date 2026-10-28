"""Exact historical Figure 1 named-element layout overrides; reproduction only."""
import json
import xml.etree.ElementTree as ET
from classical_conditioning.figure_workspace import tag

def apply_layout_spacing(raw):
    root=ET.fromstring(raw);nodes={n.get('id'):n for n in root.iter() if n.get('id')};changes=[]
    phases=['priming','pretraining','training','testing']
    for phase,x in zip(phases,[95,475,855,1235]):
        group=nodes['edit-C-phase-'+phase];box=next(n for n in group if n.tag==tag('rect'))
        old_x=float(box.get('x'));width=350
        dx=x-old_x
        # Translate actual coordinates to keep existing row wrappers and IDs intact.
        for n in group.iter():
            for key in ['x','cx','x1','x2']:
                if key in n.attrib:n.set(key,f'{float(n.get(key))+dx:g}')
        box.set('width',str(width));box.set('height','210')
        if phase=='priming':
            titles=[n for n in group if n.tag==tag('text')][:2]
            titles[0].set('y','136');titles[1].set('y','172')
        changes.append(dict(element_id=group.get('id'),box_x=x,box_width=width,box_height=210))
        # Reserve clear vertical gaps between title, CS and US rows.
        direct=[n for n in group if n.tag==tag('text')]
        direct[-1].set('y','216')
        direct_rects=[n for n in group if n.tag==tag('rect')]
        direct_rects[1].set('y','206')
        if phase!='training':nodes['edit-C-phase-'+phase+'-short-US'].set('transform','translate(0 -11)')
        else:
            nodes['edit-C-phase-training-catch-CS'].set('transform','translate(0 -34)')
            nodes['edit-C-phase-training-long-US'].set('transform','translate(0 42)')
    nodes['edit-C-node-0'].set('viewBox','0 0 1690 320');nodes['edit-C-node-0'].set('height','320')
    arrows=nodes['edit-C-phase-arrows']
    for i,(line,head) in enumerate(zip(arrows.iter(tag('line')),arrows.iter(tag('polygon')))):
        end=95+i*380+350;line.set('x1',str(end+3));line.set('x2',str(end+22));head.set('points',f'{end+19},200 {end+26},205 {end+19},210')
    scale=nodes['edit-E-colour-scale'];assert scale.get('data-reference')=='H'
    scale.set('transform','translate(730 30) scale(3.469945355) translate(-466.866142 -44.220472)')
    viewport=nodes['edit-E-independent-colourbar'];viewport.set('x','1640');viewport.set('width','160');viewport.set('viewBox','725 0 160 550')
    e_label=nodes['edit-E-H-fig1__h__colorbar__label-y'];h_label=nodes['edit-FGH-fig1__h__colorbar__label-y']
    e_text=next(e_label.iter(tag('text')));h_text=next(h_label.iter(tag('text')))
    # Preserve H's tick-to-title spacing, typography and scale at assembly size.
    for key in ['x','y','transform']:e_text.set(key,h_text.get(key))
    for meta in root.iter(tag('metadata')):
        if meta.get('id')=='editable-copy-provenance':
            provenance=json.loads(meta.text);registry=provenance['scientific_registry']
            for key in [scale.get('id'),e_label.get('id')]:
                if key in registry:
                    n=nodes[key];registry[key]['working_presentation_override']=dict(author_direction='2026-10-15: avoid E colourbar tick/title overlap, with H as reference',attributes=dict(n.attrib),text_attributes=dict(e_text.attrib) if key==e_label.get('id') else None)
            provenance['CE_spacing_override']=dict(author_direction='2026-10-15: professional C box sizing, no overlaps or cropping; E colourbar label spacing as H',cards=changes,E_scale_transform=scale.get('transform'),E_title_x=e_text.get('x'),F_arrows_unchanged=True)
            meta.text=json.dumps(provenance)
    updated,_=align_colourbars_and_space_fgh(ET.tostring(root))
    return updated,dict(cards=changes,E_scale_transform=scale.get('transform'),E_title_x=e_text.get('x'),H_title_x=h_text.get('x'))

def align_colourbars_and_space_fgh(raw):
    """Rigid translations: E/H bars share an x, with three equal FGH gutters."""
    import re
    root=ET.fromstring(raw);nodes={n.get('id'):n for n in root.iter() if n.get('id')}
    nested=nodes['edit-FGH-node-0'];view=list(map(float,nested.get('viewBox').split()));scale=float(nested.get('width'))/view[2];offset=float(nested.get('x'))
    # Independent E viewport is x=1640 with native origin 725: origin at 915.
    viewport=nodes['edit-E-independent-colourbar'];native_origin=float(viewport.get('x'))-float(viewport.get('viewBox').split()[0])
    e_bar_x=native_origin+730
    bar_native=(e_bar_x-offset)/scale
    def bounds(panel):
        path=next(nodes[f'edit-FGH-fig1__{panel}__heatmap__background'].iter(tag('path')))
        nums=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',path.get('d'))]
        return min(nums[::2]),max(nums[::2])
    f,g,h=[bounds(p) for p in 'fgh'];width=f[1]-f[0]
    assert abs((g[1]-g[0])-width)<1e-5 and abs((h[1]-h[0])-width)<1e-5
    gutter=(bar_native-f[0]-3*width)/3;assert gutter>0
    shifts={'f':0.,'g':f[0]+width+gutter-g[0],'h':f[0]+2*(width+gutter)-h[0]}
    transforms={}
    for panel in 'gh':
        prefix=f'edit-FGH-fig1__{panel}__'
        for suffix in ['heatmap__axes','heading__panel-letter','heading__condition','heading__fish-id']:
            n=nodes[prefix+suffix];n.set('transform',f'translate({shifts[panel]:.9f} 0)');transforms[n.get('id')]=n.get('transform')
    ramp=nodes['edit-FGH-fig1__h__colorbar__colour-ramp']
    # Shared colourbar left edge is specified by its named axes patch, not pixels.
    bar_bg=nodes.get('edit-FGH-fig1__h__colorbar__background')
    if bar_bg is None:
        old_bar_left=466.866142
    else:
        path=next(bar_bg.iter(tag('path')));nums=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',path.get('d'))];old_bar_left=min(nums[::2])
    colourbar=nodes['edit-FGH-fig1__h__colorbar__axes'];colourbar.set('transform',f'translate({bar_native-old_bar_left:.9f} 0)');transforms[colourbar.get('id')]=colourbar.get('transform')
    for meta in root.iter(tag('metadata')):
        if meta.get('id')=='editable-copy-provenance':
            provenance=json.loads(meta.text)
            provenance['EH_FGH_alignment_override']=dict(author_direction='2026-10-15: align E/H colour scales at right; space FGH better',same_bar_left_in_assembly=e_bar_x,FGH_equal_gutter_native=gutter,translations=transforms,heatmap_scale_unchanged=True)
            for key,record in provenance['scientific_registry'].items():
                if key.startswith('edit-FGH-fig1__g__'):dx=shifts['g']
                elif key.startswith('edit-FGH-fig1__h__colorbar__'):dx=bar_native-old_bar_left
                elif key.startswith('edit-FGH-fig1__h__'):dx=shifts['h']
                else:continue
                record['working_layout_mapping']=dict(ancestor_translation_x=dx,units='FGH native SVG user units',scale_unchanged=True,scientific_geometry_unchanged=True)
            meta.text=json.dumps(provenance)
    return ET.tostring(root),dict(shared_colourbar_left_assembly=e_bar_x,heatmap_width_native=width,equal_gutter_native=gutter,translations=transforms,heatmap_shifts=shifts)
