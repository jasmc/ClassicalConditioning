"""Stage and guard the selected main-paper SVG presentation migration.

Uses the existing axis, annotation, frame collectors and scoped state publisher.
Never rebuilds science from historical plotting scripts or cached panels.
"""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

import argparse,base64,copy,json,math,re,sys
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=_COMMAND_REPOSITORY;sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot,sha,tag,publish_state,measured_presentation,file_hash
from classical_conditioning.figure_freeze import _transform
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.layout_frames import read_spec,embed_spec,assess
from scripts.paper.shared.collection import resolve, reference, encode, authenticate, render
from scripts.paper.shared.register_axes import build_inventory
from scripts.paper.shared.correct_relationships import matrices, find, union, retile
from scripts.paper.shared.publish_axis_geometry import pages, verify_metrics, EXPECTED


def axis_style(profile):
    styles=profile.contract['styles'];family=profile.contract['typography']['family']
    geometry=profile.rule('author.axis_geometry')
    return {**EXPECTED,'tick_length_pt':geometry['tick_length_pt'],'tick_stroke_width_pt':geometry['tick_width_pt'],'tick_color':geometry['tick_color'],'tick_linecap':geometry['tick_cap'],'tick_label_edge_gap_pt':geometry['tick_end_to_label_edge_pt'],'title_label_edge_gap_pt':geometry['label_envelope_to_title_edge_pt'],'family':family,'tick_label_size_pt':styles['tick_label']['font_size_pt'],'axis_title_size_pt':styles['label']['font_size_pt'],'block_size_pt':styles['title']['font_size_pt'],'trial_size_pt':styles['annotation']['font_size_pt']}


def prepare(directory):
    profile=load_presentation_profile();state,snapshot,figures=resolve()
    ticket=directory/'starting-ticket.json'
    if ticket.exists():
        old=Snapshot([*command_sources(__file__), *([])]);old.hashes=json.loads(ticket.read_text());old.check()
    snapshot.hashes.update(Snapshot([*command_sources(__file__), *([ROOT/p for p in profile.inputs])]).hashes)
    snapshot.hashes.update(Snapshot([*command_sources(__file__), *([Path(__file__),ROOT/'scripts/paper/shared/verify_axis_geometry.cjs',ROOT/'scripts/paper/shared/verify_annotations.cjs',ROOT/'scripts/paper/shared/publish_axis_geometry.py',ROOT/'scripts/paper/shared/register_axes.py'])]).hashes)
    regular=Path('/System/Library/Fonts/Supplemental/Arial.ttf');bold=regular.with_name('Arial Bold.ttf')
    snapshot.hashes.update(Snapshot([*command_sources(__file__), *([regular,bold])]).hashes)
    font_css=''.join('@font-face{font-family:"Arial";font-weight:'+weight+';font-style:normal;src:url("data:font/ttf;base64,'+base64.b64encode(p.read_bytes()).decode()+'") format("truetype");}' for p,weight in [(regular,'400'),(bold,'700')])
    for f in figures:
        n=int(f['id'][-1]);prefix=f'figure-{n:02}';source=f['svg_bytes'];root=ET.fromstring(source);spec=read_spec(source)
        draft=json.loads(Path(f['state']['path']).read_text())['draft'];ref=draft['semantic_registry'];registry=json.loads(authenticate(ref,snapshot))if'path'in ref else copy.deepcopy(ref)
        if registry['axis_inventory']['unresolved']:raise ValueError('Incomplete axis mapping')
        for name,data in [('source.svg',source),('source-registry.json',encode(registry))]:
            dest=directory/f'{prefix}-{name}'
            if dest.exists()and dest.read_bytes()!=data:raise ValueError('Preserve existing candidate: '+str(dest))
            dest.write_bytes(data)
        # Uniform allocation scaling only: data geometry, native raster cells and
        # plot aspect ratios are untouched. Physical text/ticks are compensated.
        for alloc in spec['allocations']:
            for identity in alloc['group_ids']:
                node=find(root,identity);node.set('transform','scale(0.72) '+node.get('transform',''))
        styles=list(root.iter(tag('style')))
        if not styles:raise ValueError('Expected existing SVG stylesheet for embedded fonts')
        styles[0].text=(styles[0].text or '')+'\n'+font_css
        unit=spec['final_size_mm'][0]/float(root.get('viewBox').split()[2])*72/25.4;maps=matrices(root)
        paths={}
        def walk(node,p):
            paths[node]=p
            for i,c in enumerate(node):walk(c,p+[i])
        walk(root,[])
        roles={}
        for a in registry['axis_inventory']['axes']:
            for kind,key in [('axis.tick_label','tick_labels'),('axis.label','titles')]:
                for item in a[key]:roles[tuple(item['xml_index_path'])]=kind
        for group in registry['plot_legends']['groups']:
            for node in find(root,group['element_id']).iter(tag('text')):roles[tuple(paths[node])]='legend.key'
        entries=[]
        # Structural addresses identify all visible and hidden text. Generic
        # annotations remain explicitly presentation-only, not scientific IDs.
        for node in root.iter(tag('text')):
            text=''.join(node.itertext());path=paths[node];identity=node.get('id','');role=roles.get(tuple(path),'annotation')
            if re.fullmatch('[A-I]',text):role='panel.letter'
            elif text in ['Control','Delay','3sTrace','10sTrace']:role=roles.get(tuple(path),'title.condition')
            elif identity.endswith('-block')or text in ['Pre','Train','Test','Priming /','habituation','Pre-training','Training','Testing']:role='title'
            style_name={'panel.letter':'panel_letter','axis.label':'label','axis.tick_label':'tick_label','title.condition':'condition_title','title':'title','legend.key':'legend'}.get(role,'annotation')
            size=profile.contract['styles'][style_name]['font_size_pt']
            m=maps[node];scale=math.hypot(m[2],m[3])*unit
            # Keep XML IDs and acquisition mappings. Only the displayed panel
            # letters change case, with an explicit source/display alias.
            if role=='panel.letter':node.text=text.lower()
            node.set('style',node.get('style','')+f';font-family:"Arial" !important;font-size:{size/scale:.14g}px !important;font-weight:{"bold"if role=="panel.letter"else"normal"} !important;font-style:normal !important;fill:#000000 !important')
            entries.append({'xml_index_path':path,'element_id':identity or None,'role':role,'size_pt':size,'source_text':text,'display_text':''.join(node.itertext()),'scientific_identity':'inherited explicit axis/legend mapping'if role in ('axis.label','axis.tick_label','legend.key')else'presentation classification only'})
        stat_aliases=[{'element_id':node.get('id'),'xml_index_path':paths[node],'source_text':node.text,'display_text':node.text[1:],'role':'annotation.statistical_label','meaning':'Same star count and contrast; hypothesis identified by unchanged labeled row; original text retained'}for node in find(root,'edit-G-axes__g-stats__main').iter(tag('text'))if re.fullmatch(r'[DMR]\*+',node.text or'')]if n==2 else []
        for entry in entries:
            for alias in stat_aliases:
                if entry['element_id']==alias['element_id']:entry.update(role=alias['role'],display_text=alias['display_text'],scientific_identity=alias['meaning'])
        candidate=embed_spec(ET.tostring(root),spec);(directory/f'{prefix}-styled.svg').write_bytes(candidate)
        inv=build_inventory(candidate,n,draft['legend'],json.loads(Path(f['state']['path']).read_text())['scientific_selection']['records'],registry)
        expected=axis_style(profile)
        (directory/f'{prefix}-input.json').write_bytes(encode({'svg':reference(directory/f'{prefix}-styled.svg'),'axes':inv['axes'],'axis_style':expected}))
        annotations={'stat_text_aliases':stat_aliases,'legend_groups':registry['plot_legends']['groups'],'condition_headings':registry['heading_semantics']['condition_headings'],'stat_row_layout_done':True,'stat_phase_headers':[{'element_id':f'edit-G-node-{53+2*i}','label':label,'role':'statistical.phase.heading'}for i,label in enumerate(['Pre','Tr1','Tr2','Tr3','Tr4','Tr5','Te1','Te2','Te3'])]if n==2 else [],'annotation_style':{'family':expected['family'],'legend_size_pt':profile.contract['styles']['legend']['font_size_pt'],'heading_size_pt':profile.contract['styles']['condition_title']['font_size_pt']}}
        (directory/f'{prefix}-annotation-changes.json').write_bytes(encode(annotations))
        (directory/f'{prefix}-presentation-map.json').write_bytes(encode({'source_sha256':sha(source),'profile_inputs':profile.inputs,'text':entries,'stat_text_aliases':stat_aliases,'panel_aliases':{c:c.lower()for c in 'ABCDEFGH'+('I'if n==2 else'')},'uniform_allocation_scale':.72,'stroke_styles':{role:profile.contract['styles'][style]['linewidth_pt']for role,style in [('spine','spine'),('trace','trace'),('summary','median_iqr'),('stimulus','stimulus_cs_onset'),('separator','heatmap_phase_separator')]},'unresolved':['Scientific coverage remains inherited and partial; presentation annotation tags do not establish experiment/cohort identity.','Per-role arrows, markers, dash spacing and color accessibility require visual review; no whole-figure compliance or freeze.']}))
    ticket.write_bytes(encode(snapshot.hashes))


def compact(directory):
    """Stage the author-selected 136-mm composition, preserving physical type.

    Keep the existing source ticket and preserve the previous candidates. Only
    the top related heatmap row and the empty I placeholder need smaller uniform
    allocations; dense G retains its legible statistical inset.
    """
    prepared=[]
    for n in (1,2):
        base=f'figure-{n:02}';p=directory/f'{base}-layout.svg';raw=p.read_bytes()
        mp=directory/f'{base}-presentation-map.json';mapping=json.loads(mp.read_text())
        if mapping.get('compact_composition'):raise ValueError('Compact composition already staged; preserve it')
        root=ET.fromstring(raw);spec=read_spec(raw);unit=spec['final_size_mm'][0]/float(root.get('viewBox').split()[2])*72/25.4
        factors={'A':.9,'B':.9,'C':.9,'I':.75}if n==2 else {}
        scales=mapping.setdefault('allocation_scales',{})
        for alloc in spec['allocations']:
            factor=factors.get(alloc['id'],1.)
            for identity in alloc['group_ids']:
                node=find(root,identity)
                if factor!=1:node.set('transform',f'scale({factor}) '+node.get('transform',''))
            scales[alloc['id']]=scales.get(alloc['id'],.72)*factor
        transforms=matrices(root)
        for entry in mapping['text']:
            node=root
            for i in entry['xml_index_path']:node=list(node)[i]
            m=transforms[node];size=entry['size_pt']/(math.hypot(m[2],m[3])*unit)
            node.set('style',node.get('style','')+f';font-size:{size:.14g}px !important')
        mapping['compact_composition']={'width_mm':136,'author_selection':'2026-10-17: Compact 136 mm where feasible','allocation_factors':factors,'reason':'Minimize inter-panel whitespace; preserve native data ratios and physical typography. G statistical inset remains enlarged; I is an empty Pending placeholder.'}
        backup=directory/f'{base}-before-compact.svg'
        if backup.exists():raise ValueError('Preserve existing compact baseline')
        prepared.append((p,raw,ET.tostring(root),mp,encode(mapping),backup))
    for p,raw,candidate,mp,mapping,backup in prepared:
        backup.write_bytes(raw);mp.with_name(mp.stem+'-before-compact.json').write_bytes(mp.read_bytes());mp.write_bytes(mapping);p.write_bytes(candidate)
        staged=p.with_name(p.stem+'-compact-source.svg');staged.write_bytes(candidate)
        inp=directory/(p.name[:9]+'-input.json');data=json.loads(inp.read_text());data['svg']=reference(staged);inp.write_bytes(encode(data))
    pages(directory)


def separate_paired_ticks(directory):
    """Transfer 1 mm from inter-panel whitespace to each paired-axis gap.

    The packing estimate is explicitly derived from the preceding exact bounds;
    it is not frame evidence. Collect fresh painted/unclipped bounds afterwards.
    """
    p=directory/'figure-02-layout.svg';raw=p.read_bytes();root=ET.fromstring(raw)
    mp=directory/'figure-02-presentation-map.json';mapping=json.loads(mp.read_text())
    if mapping.get('intra_panel_layout'):raise ValueError('Paired-axis spacing already staged')
    browser=next(iter(json.loads((directory/'figure-02-browser.json').read_text()).values()))
    if browser['artifact_sha256']!=sha(raw):raise ValueError('Stale paired-axis source bounds')
    spec=read_spec(raw);unit=spec['final_size_mm'][0]/float(root.get('viewBox').split()[2]);mats=matrices(root);moves=[]
    for panel in 'ABC':
        prefix='fig2__'+panel.lower()+'1__'
        for suffix in ['heatmap__axes','heading__condition','heading__fish-id']:
            node=find(root,prefix+suffix);parent=next(p for p in root.iter()if node in list(p));m=mats[parent]
            node.set('transform',f'translate({1/unit/m[0]:.14g} 0) '+node.get('transform',''))
            moves.append({'element_id':node.get('id'),'role':'layout.condition.axis'if suffix=='heatmap__axes'else'heading.condition','dx_mm':1.,'dy_mm':0.,'reason':'Separate adjacent endpoint tick-label glyphs at unchanged 6-pt size; preserve paired-axis ratios and values'})
        for bounds in browser['allocations'][panel].values():bounds[2]+=1
        browser['data_axes'][panel.lower()+'1'][0]+=1;browser['data_axes'][panel+'-pair'][2]+=1
    candidate=ET.tostring(root);mapping['intra_panel_layout']=moves
    p.with_name('figure-02-before-pair-spacing.svg').write_bytes(raw);p.write_bytes(candidate);mp.write_bytes(encode(mapping))
    browser['artifact_sha256']=sha(candidate);browser['measurement_use']='Provisional packing estimate: preceding exact painted bounds expanded by the known 1-mm translation. Not publication evidence.'
    (directory/'figure-02-packing.json').write_bytes(encode(browser))
    # Figure 1 is unchanged; use its current exact bounds for this packing pass.
    other=json.loads((directory/'figure-01-browser.json').read_text());estimate=next(iter(other.values()));estimate['measurement_use']='Unchanged exact bounds reused only for provisional packing'
    (directory/'figure-01-packing.json').write_bytes(encode(estimate))
    repack(directory,width=136)


def repack(directory,exact=False,width=None):
    prepared=[]
    for n in (1,2):
        p=directory/f'figure-{n:02}-layout.svg';raw=p.read_bytes();root=ET.fromstring(raw);spec=read_spec(raw)
        packing=directory/f'figure-{n:02}-packing.json'
        browser=json.loads(packing.read_text())if packing.exists()and not exact else next(iter(json.loads((directory/f'figure-{n:02}-browser.json').read_text()).values()))
        if browser['artifact_sha256']!=sha(raw):raise ValueError('Stale packing measurements')
        bounds={k:union(list(v.values()))for k,v in browser['allocations'].items()};alloc={a['id']:a for a in spec['allocations']}
        native_width=spec['final_size_mm'][0];target_width=width or native_width
        unit=native_width/float(root.get('viewBox').split()[2]);y=0;reserve=spec['inset_mm']+.06
        def move(k,dx,dy):
            for identity in alloc[k]['group_ids']:
                node=find(root,identity);parent=next(p for p in root.iter()if node in list(p));m=matrices(root)[parent]
                node.set('transform',f'translate({dx/unit/m[0]:.14g} {dy/unit/m[3]:.14g}) '+node.get('transform',''))
            bounds[k][0]+=dx;bounds[k][1]+=dy
        rows=[['A','B'],['C'],['D','E','E-colorbar'],['F','G','H','FGH-colorbar']]if n==1 else [['A','B','C','ABC-colorbar'],['D','E','F'],['G','H','I']]
        if n==2:
            for k in ['G','I']:move(k,0,bounds['H'][1]-bounds[k][1])
        for row in rows:
            # Equal data-body gaps for heatmap groups; outer content gaps for
            # other named relationships. All retain the same vertical offset.
            group=['F','G','H']if n==1 and'F'in row else ['A','B','C']if n==2 and'A'in row else None
            if group:
                db=[browser['data_axes'][k if n==1 else k+'-pair']for k in group]
                # Translate first content to the outside inset, then distribute
                # residual width while preserving each body's aspect ratio.
                cb=row[-1];cbwidth=bounds[cb][2];right=target_width-reserve-cbwidth-2*reserve
                start=reserve+db[0][0]-bounds[group[0]][0]
                offsets=[(db[i][0]-bounds[group[i]][0],bounds[group[i]][0]+bounds[group[i]][2]-db[i][0]-db[i][2])for i in range(3)]
                gap=(right-start-sum(b[2]for b in db)-offsets[-1][1])/2
                minimum=max(offsets[i][1]+offsets[i+1][0]+2*reserve for i in (0,1))
                if gap<minimum:raise ValueError('Cannot fit heatmap group without collisions')
                x=start
                for k,b in zip(group,db):move(k,x-b[0],0);x+=b[2]+gap
                move(cb,target_width-reserve-cbwidth-bounds[cb][0],0)
            else:
                widths=sum(bounds[k][2]for k in row);gap=(target_width-2*reserve-widths)/(len(row)-1)if len(row)>1 else 0
                if len(row)>1 and gap<2*reserve:raise ValueError('Cannot fit row '+str(row))
                x=reserve
                for k in row:move(k,x-bounds[k][0],0);x+=bounds[k][2]+gap
            top=min(bounds[k][1]for k in row);bottom=max(bounds[k][1]+bounds[k][3]for k in row)
            dy=y+reserve-top;height=bottom-top+2*reserve
            for k in row:
                move(k,0,dy);alloc[k]['box_mm']=[0,y,target_width,height]
            retile(spec,bounds,row);y+=height
        if y>170:raise ValueError(f'Figure {n}: {y:.3f} mm exceeds 170 mm; recompose before publication')
        root.set('viewBox',f'0 0 {target_width/unit:.14g} {y/unit:.14g}');root.set('height',f'{y}mm');find(root,'assembly-node-1').set('height',f'{y/unit:.14g}');find(root,'assembly-node-1').set('width',f'{target_width/unit:.14g}');spec['final_size_mm']=[target_width,y]
        prepared.append((p,embed_spec(ET.tostring(root),spec)))
    for p,data in prepared:p.write_bytes(data)
    pages(directory)


def protection(source,candidate,registry=None,scales=None,text_aliases=None,layout_moves=None):
    registry=registry or {};scales=scales or {}
    a=ET.fromstring(source);b=ET.fromstring(candidate)
    units=[read_spec(raw)['final_size_mm'][0]/float(root.get('viewBox').split()[2])for raw,root in [(source,a),(candidate,b)]]
    if abs(units[0]-units[1])>1e-10:raise ValueError('Canvas resizing changed physical data/font units')
    old=list(a.iter());new=list(b.iter())
    if len(old)!=len(new):raise ValueError('Unexpected artwork structure change')
    owners={i:r['id']for r in read_spec(source)['allocations']for i in r['group_ids']}
    offsets={g['element_id']for g in registry.get('plot_legends',{}).get('groups',[])}|{g['element_id']for g in registry.get('heading_semantics',{}).get('condition_headings',[])}
    offsets.update(entry['element_id']for g in registry.get('plot_legends',{}).get('groups',[])if g.get('panel')=='H'for entry in g.get('entries',[])if entry.get('element_id'))
    for g in registry.get('plot_legends',{}).get('groups',[]):
        if g.get('panel')=='H':
            parentmap={c:p for p in a.iter()for c in p}
            legend=find(a,g['element_id'])
            for text in legend.iter(tag('text')):
                artist=text
                while parentmap[artist]is not legend:artist=parentmap[artist]
                siblings=list(legend);index=siblings.index(artist);offsets.add(artist.get('id'))
                if index:offsets.add(siblings[index-1].get('id'))
    offsets.update(m['element_id']for m in layout_moves or [])
    offsets.discard(None)
    axis_paths={tuple(r['xml_index_path'])for a in registry.get('axis_inventory',{}).get('axes',[])for r in a['axis_marks']if r['scientific_role']=='axis.tick'or(a['panel']=='B'and r['svg_type']=='line')}
    # B's named protocol-axis vertical line artists are the registered ticks;
    # their endpoints remain protected, as in the existing axis helper.
    paths={}
    def walk(node,p):
        paths[node]=tuple(p)
        for i,c in enumerate(node):walk(c,p+[i])
    walk(a,[])
    aliases={tuple(t['xml_index_path']):t for t in text_aliases or []};parentmap={c:p for p in a.iter()for c in p}
    allowed={'font-size','font-family','font-style','font-weight','fill','stroke-width','stroke-linecap','stroke-linejoin','stroke-dasharray','vector-effect'}
    def css(s):
        def canonical(v):
            v=v.strip().replace(' !important','')
            if v.startswith(('rgb(','rgba(')):v=re.sub(r',\s*',', ',v)
            if re.fullmatch(r'[+-]?(?:[0-9]*[.])?[0-9]+',v):return str(float(v))
            if re.fullmatch('#[0-9a-fA-F]{6}',v):return 'rgb('+', '.join(str(int(v[i:i+2],16))for i in (1,3,5))+')'
            return v
        return {k.strip():canonical(v)for p in s.split(';')if':'in p for k,v in[p.split(':',1)]}
    for x,z in zip(old,new):
        if x.tag!=z.tag:raise ValueError('Changed element type')
        if x.get('id')=='layout-frames':continue
        aa=dict(x.attrib);bb=dict(z.attrib)
        if x is a:
            for k in ('width','height','viewBox'):aa.pop(k,None);bb.pop(k,None)
        if x.get('id')=='assembly-node-1':
            for k in ('width','height'):aa.pop(k,None);bb.pop(k,None)
        if x.get('id')in owners:
            m=_transform(aa.pop('transform',''));v=_transform(bb.pop('transform',''))
            if any(abs(v[i]-scales.get(owners[x.get('id')],.72)*m[i])>1e-8 for i in range(4)):raise ValueError('Nonuniform scientific allocation change')
        # Leaf text/axis shifts and whole named annotation groups may move;
        # raster/data paths retain coordinates and all data transforms.
        if x.get('id','').startswith('edit-G-fig2__g__stats__')and any(x.get('id','').startswith('edit-G-fig2__g__stats__'+prefix)for prefix in ['D-','mean_p_fdr-','slope_p_raw-','slope_p_fdr-','trial-']):
            om=_transform(aa.pop('transform',''));nm=_transform(bb.pop('transform',''))
            if any(abs(v-w)>1e-9 for v,w in zip(om[:4],nm[:4])):raise ValueError('Statistical row geometry changed')
        if x.get('id')in offsets:
            om=_transform(aa.pop('transform',''));nm=_transform(bb.pop('transform',''))
            if any(abs(v-w)>1e-9 for v,w in zip(om[:4],nm[:4])):raise ValueError('Legend/heading shape changed')
        if paths[x]in axis_paths or x.tag==tag('text')or x.get('id','').startswith(('fig1-axis-registered','fig2-axis-registered'))or any(t in x.get('id','')for t in ('__xtick_','__ytick_')):
            aa.pop('transform',None);bb.pop('transform',None);aa.pop('id',None);bb.pop('id',None)
        for key in ('style',):
            ca,cb=css(aa.pop(key,'')),css(bb.pop(key,''))
            if x.tag!=tag('text')and ca.get('fill')!=cb.get('fill'):raise ValueError('Scientific fill changed: '+str(x.get('id')))
            if {k:v for k,v in ca.items()if k not in allowed}!={k:v for k,v in cb.items()if k not in allowed}:raise ValueError('Protected style changed: '+str(x.get('id')))
        if aa!=bb:raise ValueError('Protected attributes changed: '+str(x.get('id')))
        if x.tag==tag('style'):
            if not(z.text or'').startswith(x.text or''):raise ValueError('Existing font/style removed')
        elif x.text!=z.text:
            panel_alias=x.tag==tag('text')and re.fullmatch('[A-I]',x.text or'')and z.text==x.text.lower()
            alias=aliases.get(paths[x]);stat_alias=False
            if alias and x.tag==tag('text')and alias.get('role')=='annotation.statistical_label'and alias.get('source_text')==x.text and re.fullmatch(r'[DMR]\*+',x.text or'')and z.text==x.text[1:]==alias.get('display_text'):
                parent=x;ids=[]
                while parent in parentmap:parent=parentmap[parent];ids.append(parent.get('id',''))
                expected_prefix={'D':'D-','M':'mean_p_fdr-','R':'slope_p_'}[x.text[0]]
                stat_alias='edit-G-axes__g-stats__main'in ids and any(i.startswith('edit-G-fig2__g__stats__'+expected_prefix)for i in ids)
            if not(panel_alias or stat_alias):raise ValueError('Scientific text changed: '+str(x.get('id')))
    return {'scientific_values_geometry_and_raster_payloads':'preserved','scientific_text':'Unchanged except explicit panel-case and redundant hypothesis-prefix presentation aliases; star counts and row hypotheses preserved','native_aspect_ratios':'preserved by uniform per-allocation scaling','scope':'Presentation only; no scientific freeze'}



def publish(directory,note):
    if not note:raise ValueError('Actual final saved-artifact visual review note required')
    baseline=json.loads((directory/'starting-ticket.json').read_text());snapshot=Snapshot([*command_sources(__file__), *([])])
    # Implementation evolves during this authorized migration; bind both the
    # original hashes and exact final implementation, never relax source locks.
    code=[ROOT/'src/classical_conditioning/figure_workspace.py',ROOT/'src/classical_conditioning/figures/layout_frames_browser.js',ROOT/'scripts/paper/shared/collection.py',ROOT/'scripts/paper/shared/refine_composition.cjs',ROOT/'scripts/paper/shared/measure_packing.cjs',Path(__file__),ROOT/'scripts/paper/shared/verify_axis_geometry.cjs',ROOT/'scripts/paper/shared/verify_annotations.cjs',ROOT/'scripts/paper/shared/publish_axis_geometry.py',ROOT/'scripts/paper/shared/register_axes.py',ROOT/'scripts/paper/shared/verify_nature_presentation.cjs']
    snapshot.hashes={p:h for p,h in baseline.items()if Path(p)not in code};snapshot.check();snapshot.hashes.update(Snapshot([*command_sources(__file__), *(code)]).hashes)
    profile=load_presentation_profile();expected=axis_style(profile);prepared=[]
    methods=ROOT/'docs/paper/figures/shared/figure-specific-methods.md';snapshot.hashes.update(Snapshot([*command_sources(__file__), *([methods])]).hashes)
    for n in (1,2):
        base=f'figure-{n:02}';sp=ROOT/f'docs/paper/figures/figure-{n:02}/current-state.json';oldstate=sp.read_bytes();state=json.loads(oldstate);draft=state['draft'];source=authenticate(draft['working_svg'],snapshot)
        ref=draft['semantic_registry'];registry=json.loads(authenticate(ref,snapshot))if'path'in ref else copy.deepcopy(ref)
        cp=directory/f'{base}-layout.svg';candidate=cp.read_bytes();mapping=json.loads((directory/f'{base}-presentation-map.json').read_text())
        if mapping['profile_inputs']!=profile.inputs:raise ValueError('Changed Nature profile inputs')
        if mapping['source_sha256']!=sha(source):raise ValueError('Candidate source changed')
        protect=protection(source,candidate,registry,mapping.get('allocation_scales'),mapping.get('stat_text_aliases'),mapping.get('intra_panel_layout'))
        ap=directory/f'{base}-axis-geometry-verification.json';axis=json.loads(ap.read_text());verify_metrics(axis,candidate,expected)
        vp=directory/f'{base}-nature-verification.json';visual=json.loads(vp.read_text())
        if visual['artifact_sha256']!=sha(candidate)or visual['issues']:raise ValueError('Stale/failed presentation measurements')
        avp=directory/f'{base}-annotation-verification.json';annotation=json.loads(avp.read_text())
        if annotation['artifact_sha256']!=sha(candidate)or any(abs(g['right_edge_delta_mm'])>.01 for g in annotation['legend_groups'])or any(abs(h['center_delta_mm'])>.01 for h in annotation['condition_headings']):raise ValueError('Stale/failed legend/heading alignment')
        if n==2:
            dense=annotation['dense_layout'];headers=dense['phase_headers'];marks=dense['first_row_marks'];labels=dense['stat_row_labels'];key=dense['H_key']
            if len(headers)!=9 or len(labels)!=5 or len(key)!=2:raise ValueError('Incomplete dense annotation mapping')
            if min(t['box_mm'][1]for t in [*marks,labels[0]])-max(t['box_mm'][3]for t in headers)<.8:raise ValueError('Statistical heading/row collision')
            if key[1]['box_mm'][1]-key[0]['box_mm'][3]<.5:raise ValueError('Vertical H key collision')
            for series in (headers,marks):
                ordered=sorted(series,key=lambda t:t['box_mm'][0])
                if any(b['box_mm'][0]-a['box_mm'][2]<.2 for a,b in zip(ordered,ordered[1:])):raise ValueError('Dense annotation horizontal collision')
        if annotation['naming']['obsolete_trace_labels']or annotation['naming']['visible_sample_counts']or annotation['naming'].get('ambiguous_G_heading_visible'):raise ValueError('Obsolete naming/counts/headings')
        bp=directory/f'{base}-browser.json';browser=next(iter(json.loads(bp.read_text()).values()));frame=assess(candidate,browser)
        if frame['status']!='passed' or frame['final_size_mm'][0]not in [89,120,136,183] or frame['final_size_mm'][1]>profile.rule('journal.dimensions')['max_height_mm']:raise ValueError(frame['issues']or'Nature height exceeded')
        lp=directory/f'{base}-legend.md';legend=lp.read_bytes()
        if len(legend.decode().split())>=profile.rule('journal.caption')['word_limit_exclusive']:raise ValueError('Caption exceeds Nature limit')
        original_copy=directory/f'{base}-source.svg'
        if original_copy.read_bytes()!=source:raise ValueError('Original SVG snapshot changed')
        snapshot.hashes.update(Snapshot([*command_sources(__file__), *([original_copy])]).hashes)
        inputs=[cp,ap,vp,avp,bp,lp,directory/f'{base}-presentation-map.json'];snapshot.hashes.update(Snapshot([*command_sources(__file__), *(inputs)]).hashes)
        fp=directory/f'{base}-layout-evidence.json';fb=encode({**frame,'protection':protect,'visual_review':{'status':'reviewed','note':note}});fr=reference(fp,fb)
        ep=directory/f'{base}-nature-evidence.json';evidence={'figure_id':state['figure_id'],'artifact_sha256':sha(candidate),'profile_inputs':profile.inputs,'source_svg':reference(original_copy),'protection':protect,'axes':reference(ap),'presentation':reference(vp),'annotations':reference(avp),'layout_frames':fr,'mapping':reference(inputs[-1]),'caption_words':len(legend.decode().split()),'full_definitions':reference(methods),'visual_review':note,'coverage':'Presentation draft; scientific coverage remains scoped and partial. C/F/I remain Pending; B/H retain distinct unfrozen estimators. Per-role marker/dash/color accessibility and PDF production-font validation remain open; no full Nature compliance or scientific freeze.','implementation':{str(p):{'before':baseline.get(str(p)),'after':file_hash(p)}for p in code}};eb=encode(evidence);er=reference(ep,eb)
        oldlegend=authenticate(draft['legend'],snapshot);oldhtml=authenticate(draft['review'],snapshot)
        draft['legend']=reference(Path(draft['legend']['path']),legend)
        registry['axis_inventory']=build_inventory(candidate,n,draft['legend'],state['scientific_selection']['records'],registry)
        if registry['axis_inventory']['unresolved']:raise ValueError('Unresolved axis ownership')
        registry['axis_inventory']['registration_evidence']=er
        registry['presentation_profile']={'id':'nature-main','version':profile.policy['version'],'inputs':profile.inputs,'evidence':er,'status':'verified measured presentation properties; remaining coverage explicit in evidence'}
        registry['text_presentation_inventory']=mapping
        if mapping.get('intra_panel_layout'):registry['presentation_layout']['intra_panel_layout']=mapping['intra_panel_layout']
        registry['presentation_styles']['axis_geometry'].update(expected=expected,evidence=er,recheck_after_edits=True)
        for role in [*registry['presentation_layout'].get('axis_title_roles',[]),*registry['presentation_styles']['axis_titles'].get('roles',[])]:
            role['expected']={'family':expected['family'],'size_pt':expected['axis_title_size_pt'],'weight':'normal','style':'normal'};role['evidence']=er
        if n==1:registry['presentation_styles']['example_trial_labels']={'scope':['D','E'],'block_size_pt':expected['block_size_pt'],'trial_size_pt':expected['trial_size_pt'],'family':expected['family'],'weight':'normal','evidence':er}
        registry['presentation_styles']['axis_titles'].update(family=expected['family'],size_pt=expected['axis_title_size_pt'],expected={'family':expected['family'],'size_pt':expected['axis_title_size_pt'],'weight':'normal','style':'normal'},evidence=er)
        registry['plot_legends'].update(font_size_pt=profile.contract['styles']['legend']['font_size_pt'],groups=annotation['legend_groups'],evidence=er)
        registry['heading_semantics'].update(condition_headings=annotation['condition_headings'],evidence=er)
        if n==2:registry['heading_semantics']['statistical_phase_headers']={'role':'statistical.phase.heading','members':annotation['dense_layout']['phase_headers'],'hypothesis_rows':'D interaction; M block mean; R raw/FDR slope; trial change FDR; inherited selected scientific definitions','evidence':er}
        registry['presentation_layout'].update(allocations=read_spec(candidate)['allocations'],evidence=fr)
        spacing={}
        for group in read_spec(candidate).get('spacing_groups',[]):
            boxes=browser['data_axes']if group['target']=='data_axes'else{k:v['painted_bounds_mm']for k,v in browser['allocations'].items()}
            i=0 if group['axis']=='x'else 1;selected=[boxes[k]for k in group['ids']]
            spacing[group['id']]=[b[i]-a[i]-a[i+2]for a,b in zip(selected,selected[1:])]
        registry['presentation_layout']['spacing_measurements_mm']=spacing
        registry['presentation_layout']['larger_gap_disposition']={'measurements_mm':spacing,'reason':'Author-selected compact 136-mm composition; equal data-body gaps also accommodate the wider first-panel shared Y-axis labels. Remaining row width is distributed equally within named related groups; no fixed 183-mm endpoint constraint.','status':'measured draft composition; not final aesthetic approval','evidence':er}

        for relationship in registry.get('panel_relationships',[]):
            if relationship['type']=='layout':relationship.update(artifact_sha256=sha(candidate),verification_evidence=fr,status='verified')
        if registry.get('condition_order_mapping'):registry['condition_order_mapping'].update(artifact_sha256=sha(candidate),evidence=er)
        rp=directory/f'{base}-semantic-registry.json';rb=encode(registry);draft['semantic_registry']=reference(rp,rb)if'path'in ref else registry
        draft['working_svg']={**draft['working_svg'],'sha256':sha(candidate)}
        page=render([{'id':f'fig{n}','svg_bytes':candidate,'legend_text':legend.decode(),'svg':draft['working_svg'],'legend':draft['legend'],'whole_figure_frozen':False}])[0];draft['review']={**draft['review'],'sha256':sha(page)}
        draft['layout_frames']=fr;draft['presentation_profile']=registry['presentation_profile'];draft['assembly_scale'].update(final_width_mm=frame['final_size_mm'][0],final_height_mm=frame['final_size_mm'][1])
        pp=directory/f'{base}-effective-presentation.json';pb=encode({**measured_presentation(candidate,width_mm=frame['final_size_mm'][0]),'nature_profile':er,'layout_frames':fr});draft['effective_presentation']=reference(pp,pb)
        draft.setdefault('consistency_requests',[]).append({'id':f'fig{n}-nature-presentation-20261016','scope':[f'Figure {n}'],'status':'verified measured typography/axis geometry/frames/relationships; partial overall coverage','evidence':er,'affected_artifact_hashes':{'svg':sha(candidate),'legend':sha(legend)}})
        draft['reproduction']={'generator':reference(Path(__file__)),'directory':str(directory),'profile_inputs':profile.inputs,'evidence':er}
        artifacts=[(fp,fb),(ep,eb),(rp,rb),(pp,pb),(directory/f'{base}-before-state.json',oldstate),(directory/f'{base}-before.html',oldhtml),(directory/f'{base}-before-legend.md',oldlegend)]
        prepared.append((sp,state,artifacts,[(Path(draft['working_svg']['path']),candidate),(Path(draft['review']['path']),page),(Path(draft['legend']['path']),legend)]))
    snapshot.check()
    for sp,state,artifacts,updates in prepared:
        publish_state(sp,state,snapshot,artifacts=artifacts,evidence_updates=updates)
        for p in [sp,*[p for p,_ in updates]]:snapshot.hashes[str(p.resolve())]=file_hash(p)
    print('Published guarded Nature presentation drafts; scientific freezes unchanged')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','compact','pair-spacing','repack','publish']);p.add_argument('--directory',type=Path,required=True);p.add_argument('--visual-review-note');p.add_argument('--exact',action='store_true');p.add_argument('--width-mm',type=float,choices=[89,120,136,183]);a=p.parse_args()
    if not Path('/Volumes/JOAQUIM').is_mount()or not a.directory.is_dir():raise ValueError('Mounted existing external directory required')
    publish(a.directory,a.visual_review_note)if a.command=='publish'else repack(a.directory,a.exact,a.width_mm)if a.command=='repack'else compact(a.directory)if a.command=='compact'else separate_paired_ticks(a.directory)if a.command=='pair-spacing'else prepare(a.directory)
if __name__=='__main__':main()
