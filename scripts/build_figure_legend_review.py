"""Build one text/layout-only Figure 1–2 review from authenticated selected SVGs."""
from __future__ import annotations
import argparse, base64, copy, hashlib, html, json, re, sys
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from classical_conditioning.external_artifacts import resolve_artifact
from classical_conditioning.figures.aligned_example_pair import build_pair, dependencies as pair_dependencies
from classical_conditioning.figure_workspace import Snapshot, editable_svg, measured_presentation, publish_state, file_hash, draft_axis_findings, bind_loaded_state
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
def tag(name): return '{'+NS+'}'+name
def digest(data): return hashlib.sha256(data).hexdigest()
def svg_bytes(root): return ET.tostring(root, encoding='utf-8')
def label(node): return ''.join(node.itertext())

# Exact strings protect ticks, scientific marks and similarly named axes.
REMOVE = {
 (1,'B'): {'Training trial','Relationship','Alternative US onsets','US at 9 s','US at 13 s','US at 20 s','Timing varies between trials','During CS','3 s after CS offset','10 s after CS offset'},
 (1,'C'): {'Session sequence','Schematic · not to scale','includes left-side light','Trial timing in B','Short US pulses are interspersed; they are not conditioning pairings.','Control:','Same long-US times; training CS onsets randomized per fish'},
 (1,'D'): {'Distal cumulative angle; -15 to 0 s median centred'},
 (1,'E'): {'Raw vigor + frozen heatmap rows','Delay 20221115_07 · 0.5 s bins · baseline [−15,0) s','Each strip is the matching frozen V12 heatmap row · all 80 bins shown','Raw display capped at 1 rad/ms: 12 frame values exceed the range.','Frozen colours saturate at ±1 (12 bins); missing bins are black. Signed data are not clipped.'},
 (1,'FGH'): {'20221115_07','20230310_08','20221115_09'},
 (2,'D'): {'Fish median/IQR · Holm36','† versus 0 · sign / BM'},
 (2,'E'): {'Fish median/IQR · Holm36','† versus 0 · sign / BM'},
 (2,'G'): {'Delay | Historical LogMedian · bout frames only','D: Holm8 · M/R: BH9 · trial stars: phase-aware LMM, change from Pre, BH90','95% CI: 5,000 whole-fish resamples, seed 10; missing trials retained. Test3 local M/R: n/a.','Exploratory mixed-effects inference; heavy residual tails; stars do not establish learning onset.'},
}
REPLACE = {(1,'D'): {'Tail angle (°) · Delay fish 20221115_07':'Tail angle (°)'}}
# Whitespace trims and the expanded stimulus-aligned protocol-card viewport.
CROP = {(1,'C'):(0,0,1690,365),(1,'E'):(0,64,648,334),(2,'D'):(0,0,155.622047,137),(2,'E'):(0,0,155.622047,137),(2,'G'):(0,0,518.740157,241)}

def transform_panel(number, panel, data):
    original = ET.fromstring(data); candidate = copy.deepcopy(original)
    notes = next((e for e in original.iter(tag('g')) if number==1 and panel=='C' and e.get('id')=='protocol-notes'), None)
    explanatory = set(notes.iter()) if notes is not None else set()
    before = Counter(digest(ET.tostring(e)) for e in original.iter() if e not in explanatory and e.tag in {tag(t) for t in ('path','rect','line','polyline','polygon','circle','ellipse','image','use')})
    parents = {child: parent for parent in candidate.iter() for child in parent}
    seen=set(); edits=[]
    for node in list(candidate.iter(tag('text'))):
        text=label(node); parent=parents[node]
        if text in REMOVE.get((number,panel),set()):
            edits.append(dict(text=text,element_id=parent.get('id') or node.get('id'),action='move-to-legend-or-review-notes'))
            seen.add(text); parent.remove(node)
        elif text in REPLACE.get((number,panel),{}):
            replacement=REPLACE[number,panel][text]
            assert not list(node), 'Do not flatten composite scientific text'
            node.text=replacement;edits.append(dict(text=text,replacement=replacement,element_id=parent.get('id'),action='retain-unit; move-fish-id-to-legend'))
    assert seen==REMOVE.get((number,panel),set()), (number,panel,'missing removal target',REMOVE.get((number,panel),set())-seen)
    if notes is not None:
        group = next(e for e in candidate.iter(tag('g')) if e.get('id')=='protocol-notes')
        parents[group].remove(group)
        edits.append(dict(element_id='protocol-notes', action='move-explanatory-legend-key-to-caption', text='Short-US and control key markers accompanying the removed protocol notes'))
    after = Counter(digest(ET.tostring(e)) for e in candidate.iter() if e.tag in {tag(t) for t in ('path','rect','line','polyline','polygon','circle','ellipse','image','use')})
    assert before==after, (number,panel,'Scientific graphics changed')
    if (number,panel) in CROP:
        x,y,w,h=CROP[number,panel];candidate.set('viewBox',f'{x} {y} {w} {h}');candidate.set('width',str(w));candidate.set('height',str(h))
    if number==1 and panel=='B':
        candidate=grid_timing(candidate)
        edits.append(dict(action='two-column-layout', text='Control | Delay; Trace (3 s) | Trace (10 s); experiment names use FGH colours; vertical colour markers removed; event coordinates unchanged'))
    if number==1 and panel=='FGH':
        move_delay_example_arrows(candidate)
        edits.append(dict(action='move-example-arrows-G-to-F',text='Delay fish 20221115_07: trials 9, 17, 63, 66, 93, matching D/E; heatmap data unchanged'))
    if number==1 and panel=='C':
        align_stimulus_rows(candidate)
        edits.append(dict(action='align-stimulus-types', text='CS presentations, CS-only catches, paired long US and separate short US occupy common rows across phases'))
    return svg_bytes(candidate), edits

def grid_timing(source):
    """Reposition exact source stimulus groups; duplicate the same axis per cell."""
    result=ET.Element(tag('svg'),dict(source.attrib))
    result.set('viewBox','0 0 1400 430');result.set('width','1400');result.set('height','430')
    for kind in ('title','desc'):
        node=source.find(tag(kind))
        if node is not None:result.append(copy.deepcopy(node))
    heading=copy.deepcopy(next(g for g in source if g.get('id')=='heading'))
    for node in list(heading):
        if node.tag==tag('text') and label(node)=='Time relative to CS onset (s)':
            heading.remove(node)
        elif 'x' in node.attrib:node.set('x',str(float(node.get('x'))+370))
        elif 'cx' in node.attrib:node.set('cx',str(float(node.get('cx'))+370))
    result.append(heading)
    axis=next(g for g in source if g.get('id')=='shared-time-axis')
    for index,(name,cy) in enumerate([('control',82),('delay',151),('trace-3s',220),('trace-10s',289)]):
        x=20+(index%2)*710;y=100+(index//2)*185
        cell=ET.SubElement(result,tag('g'),{'id':'cell-'+name,'transform':f'translate({x-200} {y-cy})'})
        row=copy.deepcopy(next(g for g in source if g.get('id')=='row-'+name))
        # Full-width phase separators are decoration belonging to the old rows.
        for node in list(row):
            if node.tag==tag('line') and node.get('x1')=='20' and node.get('x2')=='1090':row.remove(node)
        condition_colour=next(n.get('fill') for n in row if n.tag==tag('rect') and n.get('x')=='24')
        title=ET.Element(tag('g'),{'transform':'translate(190 -40)'})
        for node in list(row):
            if (node.tag==tag('text') and label(node) in {'Control','Delay','Trace (3 s)','Trace (10 s)'}) or (node.tag==tag('rect') and node.get('x')=='24'):
                row.remove(node)
                if node.tag==tag('text'):
                    color=condition_colour
                    node.set('fill',color);node.set('style',node.get('style','')+';fill:'+color)
                    node.set('data-condition',name);title.append(node)
                # Author requested removal of the former coloured vertical marker.
        row.append(title);cell.append(row)
        ticks=copy.deepcopy(axis);ticks.set('id','time-axis-'+name);ticks.set('transform',f'translate(0 {cy+40-324})')
        for node in list(ticks):
            if node.tag==tag('polygon') and node.get('points','').startswith('227.5,'):
                ticks.remove(node)
        ET.SubElement(ticks,tag('text'),{'x':'531.25','y':'386','text-anchor':'middle','data-role':'axis-label'}).text='Time relative to CS onset (s)'
        cell.append(ticks)
    # Recomposition cannot alter any event or its scientific identity.
    for original in source.iter():
        if original.get('data-start-s') is not None or original.get('data-time-s') is not None or original.get('data-example-time-s') is not None:
            match=next(e for e in result.iter() if e.get('id')==original.get('id'))
            assert ET.tostring(match)==ET.tostring(original)
    return result

def move_delay_example_arrows(panel):
    """Move only named example arrows, preserving trial y mappings and cell pixels."""
    axes={pid:next(n for n in panel.iter() if n.get('id')==f'fig1__{pid}__heatmap__axes') for pid in ('f','g')}
    def bounds(pid):
        bg=next(n for n in panel.iter() if n.get('id')==f'fig1__{pid}__heatmap__background')
        nums=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',next(bg.iter(tag('path'))).get('d'))]
        return min(nums[::2]),max(nums[::2]),min(nums[1::2]),max(nums[1::2])
    f,g=bounds('f'),bounds('g')
    assert f[2:]==g[2:] and abs((f[1]-f[0])-(g[1]-g[0]))<1e-5
    for trial in (9,17,63,66,93):
        node=next(n for n in axes['g'] if n.get('id')==f'fig1__g__heatmap__example-trial-{trial}')
        path=next(node.iter(tag('path')));nums=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',path.get('d'))]
        expected=g[2]+(trial-4.5)/90*(g[3]-g[2])
        assert abs(nums[1]-expected)<2e-6,'Arrow trial centre does not match heatmap coordinate'
        axes['g'].remove(node);node.set('id',f'fig1__f__heatmap__example-trial-{trial}')
        node.set('transform',f'translate({f[1]-g[1]} 0)')
        node.set('data-fish-id','20221115_07');node.set('data-trial',str(trial))
        node.set('data-linked-panels','D E');axes['f'].append(node)


def align_stimulus_rows(source):
    for phase in [g for g in source.iter(tag('g')) if g.get('id','').startswith('phase-') and g.get('id')!='phase-arrows']:
        box=phase.find(tag('rect'));box.set('height','250')
        # Shift symbol and its count label together, preserving their relative geometry.
        types=[('short-US','separate short US',34),('long-US','paired long US',27),('catch-CS','CS-only catch trials',-49)]
        for name,phrase,delta in types:
            text=next((e for e in phase.findall(tag('text')) if phrase in label(e)),None)
            if text is None:continue
            index=list(phase).index(text);symbol=list(phase)[index-1]
            assert symbol.tag in {tag('rect'),tag('circle')}
            wrapper=ET.SubElement(phase,tag('g'),{'id':phase.get('id')+'-'+name,'transform':f'translate(0 {delta})'})
            phase.remove(symbol);phase.remove(text);wrapper.extend([symbol,text])


def load_sources():
    path=resolve_artifact('reviews/latest_figure_assemblies_20261010.html')
    text=path.read_text();meta=json.loads(re.search(r'<script[^>]*id="assembly-provenance"[^>]*>(.*?)</script>',text,re.S)[1])
    blobs=re.findall(r'<img class="figure"[^>]*src="data:image/svg\+xml;base64,([^"]+)"',text)
    assert len(blobs)==2
    result=[]
    for n,b in enumerate(blobs,1):
        data=base64.b64decode(b); info=meta['figures'][n-1]
        assert digest(data)==info['svg_sha256']
        original=ET.fromstring(data); panels={};recorded={p['id']:p for p in info['panels']}
        for group in original:
            pid=group.get('id','').removeprefix('panel-')
            if group.get('id','').startswith('panel-'):
                images=list(group.iter(tag('image')))
                if images:
                    assert len(images)==1
                    im=images[0];raw=base64.b64decode(im.get('href').split(',',1)[1]);assert digest(raw)==recorded[pid]['source']['sha256']
                    panels[pid]=dict(bytes=raw,placement=dict(im.attrib),group=group,provenance=recorded[pid])
                else:panels[pid]=dict(bytes=None,group=group,provenance=recorded[pid])
        result.append(dict(number=n,bytes=data,root=original,panels=panels,metadata=info))
    # Current pointers must still select the exact original review's panel bytes.
    selections=ROOT/'configs/paper-figures/selections'
    e=json.loads((selections/'figure1-panel-e-heatmap-rows-freeze-20261009.json').read_text())
    f=json.loads((selections/'figure1-fgh-version12-freeze-20261009.json').read_text())
    de=json.loads((selections/'figure2-DE-B-freeze-20261009.json').read_text())
    g=json.loads((selections/'figure2-G-logmedian-freeze-20261009.json').read_text())
    assert digest(result[0]['panels']['E']['bytes'])==e['export']['sha256']
    assert digest(result[0]['panels']['FGH']['bytes'])==f['exports'][0]['sha256']
    for pid in 'DE':assert digest(result[1]['panels'][pid]['bytes'])==de['current_panels'][pid]['svg']['sha256']
    assert digest(result[1]['panels']['G']['bytes'])==g['svg']['sha256']
    return result,dict(path=str(path),sha256=digest(path.read_bytes()))

def image_extent(attrs,old,new):
    # Preserve original xMidYMid-meet scale, including unused viewport margins.
    bx,by,bw,bh=map(float,ET.fromstring(old).get('viewBox').split())
    nx,ny,nw,nh=map(float,ET.fromstring(new).get('viewBox').split())
    x,y,w,h=(float(attrs[k]) for k in ('x','y','width','height'));scale=min(w/bw,h/bh)
    return dict(x=x+(w-bw*scale)/2+(nx-bx)*scale,y=y+(h-bh*scale)/2+(ny-by)*scale,width=nw*scale,height=nh*scale,scale=scale)

def build_candidate(source, aligned_panels=None):
    n=source['number'];root=copy.deepcopy(source['root']);edits=[];layout=[]
    # Reuse selected source graphics; apply only requested presentation/layout changes.
    shifts={1:{'FGH':-1170},2:{'A':-145,'B':-145,'C':-145,'D':-150,'E':-150,'F':-150,'G':-205,'H':-325,'I':-325}}
    for child in list(root):
        if child.tag==tag('text'):
            edits.append(dict(figure=n,panel='assembly',text=label(child),action='move-to-review-notes'));root.remove(child)
        pid=child.get('id','').removeprefix('panel-')
        if not child.get('id','').startswith('panel-'):continue
        original=source['panels'][pid];shift=shifts[n].get(pid,0)
        if shift:child.set('transform',f'translate(0 {shift})')
        if original['bytes']:
            candidate,panel_edits=transform_panel(n,pid,original['bytes'])
            if n==1 and aligned_panels and pid in aligned_panels:
                candidate=aligned_panels[pid];panel_edits=[dict(action='shared-D-E-renderer',text='Author-requested D measured-frame input and common layout; E colours recomputed')]
            edits.extend(dict(figure=n,panel=pid,**e) for e in panel_edits)
            im=next(child.iter(tag('image')));extent=image_extent(original['placement'],original['bytes'],candidate)
            if n==1 and pid=='B':
                extent=dict(x=405,y=35,width=1340,height=1340*430/1400,scale=1340/1400)
            if n==1 and pid in {'D','E'}:
                # Identical allocated frames; xMidYMid meet prevents stretching.
                vx,vy,vw,vh=map(float,ET.fromstring(candidate).get('viewBox').split())
                extent=dict(x=45 if pid=='D' else 915,y=930,width=840,height=550,scale=min(840/vw,550/vh))
                for node in child.findall(tag('text')):
                    node.set('x',str(extent['x']));node.set('y','910')
            for key in ('x','y','width','height'):im.set(key,str(extent[key]))
            im.set('href','data:image/svg+xml;base64,'+base64.b64encode(candidate).decode())
            original_extent=image_extent(original['placement'],original['bytes'],original['bytes'])
            layout.append(dict(panel=pid,translate_y=shift,source_scale=original_extent['scale'],candidate_scale=extent['scale'],candidate_sha256=digest(candidate),extent=extent))
        else:
            # Keep missing-data placeholders visibly explicit, not silent blanks.
            nodes=list(child.iter(tag('text')))
            for node in nodes:
                if label(node) in {pid}:continue
                edits.append(dict(figure=n,panel=pid,text=label(node),action='replace-placeholder-explanation',replacement='10 s Trace / Inputs unavailable / Panel pending'))
                child.remove(node)
            box=next(child.iter(tag('rect')))
            for offset,line in enumerate(['10 s Trace','Inputs unavailable','Panel pending']):
                ET.SubElement(child,tag('text'),{'x':str(float(box.get('x'))+float(box.get('width'))/2),'y':str(float(box.get('y'))+float(box.get('height'))/2+(offset-1)*32),'font-size':'20','text-anchor':'middle'}).text=line
    if n==1:
        by_id={item['panel']:item for item in layout}
        c=by_id['C'];d=by_id['D'];e=by_id['E'];f=by_id['FGH']
        assert c['extent']['y']+c['extent']['height'] <= d['extent']['y']-40
        assert d['extent']['y']==e['extent']['y']
        assert d['extent']['width']==e['extent']['width'] and d['extent']['height']==e['extent']['height']
        assert d['extent']['y']+d['extent']['height'] < f['extent']['y']+f['translate_y']
    height=2330 if n==1 else 2600
    root.set('viewBox',f'0 0 1800 {height}');root.set('height',str(height))
    root.set('aria-label',f'Figure {n}: text and layout review draft; not frozen')
    background=root.find(tag('rect'));background.set('height',str(height))
    return svg_bytes(root),edits,layout

def harmonize_typography(data,layout,number):
    """Normalize live candidate typography at 183 mm, leaving selected bytes intact."""
    root=ET.fromstring(data);pt_per_unit=183/25.4*72/1800
    report=[]
    for group in root:
        pid=group.get('id','').removeprefix('panel-')
        if not group.get('id','').startswith('panel-'):continue
        for text in group.findall(tag('text')):
            size=10 if label(text)==pid else 7
            set_font(text,size/pt_per_unit,'bold' if size==10 else 'normal')
            report.append(dict(figure=number,panel=pid,text=label(text),role='panel-letter' if size==10 else 'placeholder',final_size_pt=size,font_family='DejaVu Sans',weight='bold' if size==10 else 'normal'))
        image=next(group.iter(tag('image')),None)
        if image is None:continue
        panel=ET.fromstring(base64.b64decode(image.get('href').split(',',1)[1]))
        if panel.get('data-renderer')=='aligned-example-pair':
            # Already rendered at final assembly scale from explicit scientific roles.
            # Do not apply the old renderer-ID/affine D/E adjustments.
            report.append(dict(figure=number,panel=pid,role='shared-renderer',final_axis_labels_pt=8,final_ticks_pt=7,source='named D/E semantic registry'))
            continue
        protected_before=graphic_signature(panel)
        # Enlarged labels require real gutters, not merely a font substitution.
        if number==1 and pid=='D':
            # Match the plotted data width to E, not just the allocated frame.
            other_group=next(g for g in root if g.get('id')=='panel-E')
            other_svg=ET.fromstring(base64.b64decode(next(other_group.iter(tag('image'))).get('href').split(',',1)[1]))
            own_clip=next(e for g in panel.iter(tag('clipPath')) for e in g if e.tag==tag('rect'))
            other_clip=next(e for g in other_svg.iter(tag('clipPath')) for e in g if e.tag==tag('rect'))
            matched_scale=float(other_clip.get('width'))/float(own_clip.get('width'))
            panel.set('viewBox',f'-140 -15 760 {550/matched_scale}')
            for axis in panel.iter(tag('g')):
                if re.fullmatch(r'axes_[1-5]',axis.get('id','')):
                    axis.set('transform',f"translate(0 {(int(axis.get('id').split('_')[1])-1)*30})")
        if number==1 and pid=='E':panel.set('viewBox','-165 0 840 550')
        if number==2 and pid in {'A','B','H'}:
            ox,oy,ow,oh=map(float,panel.get('viewBox').split())
            old_scale=min(float(image.get('width'))/ow,float(image.get('height'))/oh)
            panel.set('viewBox',f'-35 -15 {ow+70} {oh+55}')
            if pid=='H':panel.set('viewBox',f'-70 -15 {ow+105} {oh+55}')
            image.set('x',str(float(image.get('x'))-(70 if pid=='H' else 35)*old_scale));image.set('y',str(float(image.get('y'))-15*old_scale))
            image.set('width',str((ow+(105 if pid=='H' else 70))*old_scale));image.set('height',str((oh+55)*old_scale))
        if number==2 and pid=='H':
            for legend in panel.iter(tag('g')):
                if legend.get('id')=='legend_1':
                    # Reposition only explicit legend artists, preserving their identities.
                    for i,child in enumerate(legend):
                        if child.get('id','').startswith('text_') or child.get('id','').startswith('line2d_'):
                            value=''.join(child.itertext())
                            delta=20 if '3sTrace' in value or child.get('id')=='line2d_16' else 0
                            child.set('transform',f'translate(-140 {delta})')
        _,_,vw,vh=map(float,panel.get('viewBox').split())
        scale=min(float(image.get('width'))/vw,float(image.get('height'))/vh)
        parents={child:parent for parent in panel.iter() for child in parent}
        removed=[]
        for text in list(panel.iter(tag('text'))):
            ancestors=[];parent=text
            while parent is not None:
                ancestors.append(parent.get('id',''));parent=parents.get(parent)
            context=' '.join(ancestors);value=label(text)
            if number==2 and pid in {'A','B'} and value=='Time from CS onset (s)':
                if removed:
                    parents[text].remove(text);continue
                text.set('x','194');text.set('y','350');text.set('transform','rotate(0 194 350)');removed.append(value)
            if number==1 and pid=='D' and value=='Time relative to CS onset (s)':
                text.set('y','425');text.set('transform',f"rotate(0 {text.get('x')} 425)")
            if number==1 and pid=='E':
                if '__phase-trial' in context:
                    old_y=float(re.search(r'translate\([^ ]+ ([^)]+)\)',text.get('transform'))[1])
                    new_y=old_y+(15 if value.startswith('Trial ') else -8)
                    text.set('transform',f'translate(45 {new_y})');text.set('text-anchor','end');text.set('style',text.get('style','')+';text-anchor:end')
                if value=='Raw vigor (rad/ms)':text.set('transform','translate(-145 257) rotate(-90)');text.set('style',text.get('style','')+';text-anchor:middle')
                if value=='Time from measured CS onset (s)':
                    text.set('y','435');text.set('transform','rotate(0 314.28 435)')
            if number==1 and pid in {'D','E'} and re.search(r'xtick',context):
                if text.get('y') is not None:
                    y=float(text.get('y'))+(10 if pid=='D' else 20);text.set('y',str(y));text.set('transform',f"rotate(0 {text.get('x')} {y})")
            if number==1 and pid=='E' and value=='Scaled deviation from baseline':
                text.set('x','640');text.set('transform','rotate(-90 640 220.32)')
            if number==2 and pid in {'A','B'} and value=='Global CS trial':
                text.set('x','-15');text.set('transform',f"rotate(-90 -15 {text.get('y')})")
            if number==2 and pid in {'A','B'} and value=='Mean signed log vigor':
                text.set('x','430');text.set('transform',f"rotate(-90 430 {text.get('y')})")
            if number==2 and pid=='H' and value=='Response / pre-CS baseline':
                text.set('x','-20');text.set('transform','rotate(-90 -20 115.389688)');text.text=None
                ET.SubElement(text,tag('tspan'),{'x':'-20','y':'115.389688'}).text='Response / pre-CS'
                ET.SubElement(text,tag('tspan'),{'x':'-20','y':'90.389688'}).text=' baseline'
            if number==2 and pid=='H' and value=='Global CS trial':
                text.set('y','285');text.set('transform',f"rotate(0 {text.get('x')} 285)")
            role='label';size=8;weight='normal' 
            if re.search(r'xtick|ytick|tick.label|tick-label',context):role='tick';size=7
            elif re.search(r'legend|sample.count|sample-count|stimulus-key',context):role='legend';size=7
            elif (value in {'F','G','H'} and number==1 and pid=='FGH') or (value=='G' and number==2 and pid=='G'):role='panel-letter';size=10;weight='bold'
            elif number==1 and pid=='B' and (('time-axis' in context and text.get('data-role')!='axis-label') or value=='?'):role='tick';size=7
            elif number==1 and pid=='C' and any(word in value for word in ['CS','US']):role='annotation';size=7
            elif re.search(r'legend_',context):role='legend';size=7
            elif number==2 and pid=='G' and re.search(r'annot|stat',context) and '__phase-' not in context:role='annotation';size=7
            elif value.startswith('Trial '):role='annotation';size=7
            if number==1 and (text.get('data-role')=='axis-label' or re.search(r'__label-[xy]',context) or value in {'Tail angle (°)','Raw vigor (rad/ms)','Time relative to CS onset (s)','Time from measured CS onset (s)','Scaled deviation from baseline'}):
                role='axis-label';weight='bold'
                text.set('fill','#000000');text.set('style',text.get('style','')+';fill:#000000')
            set_font(text,size/pt_per_unit/scale,weight)
            for span in text.iter(tag('tspan')):set_font(span,size/pt_per_unit/scale,weight)
            report.append(dict(figure=number,panel=pid,text=value,role=role,final_size_pt=size,font_family='DejaVu Sans',weight=weight))
            if number==1 and pid=='D' and value in {'Pre-Train','Early Train','Late Train','Early Test','Late Test'}:
                text.set('x','45');text.set('y',str(float(text.get('y'))+4));text.set('transform',f"rotate(0 45 {text.get('y')})")
        stroke_report=harmonize_strokes(panel,number,pid,scale*pt_per_unit);report.extend(stroke_report)
        if number==1:
            legacy_axes_style(panel,pid,scale*pt_per_unit)
            if pid in {'D','E'}:expand_trace_rows(panel,pid)
            if pid=='FGH':shorten_heatmaps(panel,scale*pt_per_unit)
        assert graphic_signature(panel)==protected_before, 'Scientific coordinates or embedded payload changed during styling'
        image.set('href','data:image/svg+xml;base64,'+base64.b64encode(svg_bytes(panel)).decode())
        entry=next(e for e in layout if e['panel']==pid);entry['candidate_sha256']=digest(svg_bytes(panel));entry['candidate_scale']=scale;entry['extent']={k:float(image.get(k)) for k in ('x','y','width','height')};entry['candidate_viewbox']=panel.get('viewBox')
    return svg_bytes(root),report

def legacy_axes_style(panel,pid,pt_per_source):
    """Candidate-only axis formatting from legacy/helpers/plotting_style.py.

    Keep all scientific paths/resources intact; use group translations for the
    outward bottom spine and ticks. Tick lengths and label pads are physical.
    """
    parents={c:p for p in panel.iter() for c in p}
    def context(node):
        ids=[]
        while node is not None:ids.append(node.get('id',''));node=parents.get(node)
        return ' '.join(ids)
    offset=6/pt_per_source
    # Legacy has bold labels, 0.5-pt black axes, 2-pt outward ticks and 2-pt pads.
    if pid=='B':
        for axis in [g for g in panel.iter(tag('g')) if g.get('id','').startswith('time-axis-')]:
            for node in axis:
                node.set('fill','#000000') if node.tag in {tag('text'),tag('polygon')} else None
                if node.tag==tag('line'):
                    node.set('stroke','#000000');node.set('stroke-width',str(.5/pt_per_source))
                    if node.get('x1')==node.get('x2'):
                        node.set('y1','324');node.set('y2',str(324+2/pt_per_source))
                elif node.tag==tag('text') and node.get('data-role')!='axis-label':
                    node.set('y',str(324+4/pt_per_source+float(node.get('font-size'))*.76))
            for node in axis.iter(tag('text')):
                if node.get('data-role')=='axis-label':
                    node.set('y',str(324+4/pt_per_source+7/pt_per_source+4/pt_per_source+8/pt_per_source*.76))
        return
    if pid not in {'D','E','FGH'}:return
    # Normalize referenced tick markers without editing their original paths.
    marker_shapes={n.get('id'):n for n in panel.iter(tag('path')) if n.get('id')}
    for node in list(panel.iter(tag('use'))):
        ctx=context(node)
        if not re.search(r'xtick|ytick',ctx):continue
        ref=node.get('{http://www.w3.org/1999/xlink}href',node.get('href','')).lstrip('#')
        shape=marker_shapes.get(ref)
        if shape is None:continue
        nums=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',shape.get('d',''))]
        if len(nums)!=4:continue
        dx,dy=nums[2]-nums[0],nums[3]-nums[1]
        length=abs(dx or dy)
        if not length:continue
        # A template's explicit stroke width overrides the use element's width.
        # Inherit the compensated width from each use; retain marker geometry.
        shape.attrib.pop('stroke-width',None)
        shape.set('style',';'.join(part for part in shape.get('style','').split(';') if not re.match(r'\s*stroke-width\s*:',part)))
        ratio=2/pt_per_source/length
        x=float(node.get('x','0'));y=float(node.get('y','0'))
        node.set('transform',f'translate({x} {y}) scale({ratio if dx else 1} {ratio if dy else 1}) translate({-x} {-y})')
        # Compensate marker stroke after its length scaling.
        node.set('style',node.get('style','')+f';stroke:#000000;stroke-width:{.5/pt_per_source}')
    bottom_patches={'patch_4','patch_7','patch_10','patch_13','patch_16'}
    for group in panel.iter(tag('g')):
        gid=group.get('id','')
        if pid=='D' and gid in bottom_patches:
            group.set('transform',f'translate(0 {offset})')
        elif pid=='E' and '__spine-bottom' in gid and '-trace__' in gid:
            group.set('transform',f'translate(0 {offset})')
        elif pid=='D' and gid.startswith('xtick_'):
            group.set('transform',f'translate(0 {offset})')
    # Labels sit next to the actual outward ticks, not the original source fonts.
    for tick in panel.iter(tag('g')):
        if not re.fullmatch(r'[xy]tick_\d+',tick.get('id','')):continue
        marks=list(tick.iter(tag('use')));texts=list(tick.iter(tag('text')))
        if not marks or not texts:continue
        mark=marks[0];text=texts[0];is_x=tick.get('id').startswith('x')
        if is_x:
            y=float(mark.get('y'))+4/pt_per_source+7/pt_per_source*.76
            text.set('y',str(y));text.set('transform',f"rotate(0 {text.get('x')} {y})")
        else:
            is_right='colorbar' in context(tick)
            x=float(mark.get('x'))+(1 if is_right else -1)*4/pt_per_source
            y=float(mark.get('y'))+7/pt_per_source*.32
            text.set('x',str(x));text.set('y',str(y));text.set('transform',f'rotate(0 {x} {y})')
        text.set('style',text.get('style','')+';fill:#000000')

def expand_trace_rows(panel,pid):
    """Enlarge y display space within each row, preserving x widths and values."""
    factor=1.5 if pid=='D' else 1.8
    if pid=='E':
        trials=['9','17','63','66','93']
        for axis in panel.iter(tag('g')):
            match=re.fullmatch(r'fig1__e__trial-(\d+)-(?:trace|heatmap)__axes',axis.get('id',''))
            if match:axis.set('transform',f'translate(0 {trials.index(match[1])*(80-61.38)})')
    for axis in list(panel.iter(tag('g'))):
        gid=axis.get('id','')
        if not ((pid=='D' and re.fullmatch(r'axes_[1-5]',gid)) or (pid=='E' and gid.endswith('-trace__axes'))):continue
        # The trace-row background declares the exact retained data viewport.
        bg=next(node for node in axis.iter(tag('path')) if 'fill: #ffffff' in node.get('style',''))
        coords=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',bg.get('d'))]
        ys=coords[1::2];center=(min(ys)+max(ys))/2 if pid=='D' else max(ys)
        wrapper=ET.Element(tag('g'),{'id':gid+'-vertical-space-review','transform':f'translate(0 {(1-factor)*center}) scale(1 {factor})'})
        children=list(axis);attrs=dict(axis.attrib);axis.clear();axis.attrib.update(attrs)
        if pid=='D':axis.set('transform',f"translate(0 {(int(gid.split('_')[1])-1)*30})")
        wrapper.extend(children);axis.append(wrapper)
        parents={c:p for p in wrapper.iter() for c in p}
        for node in wrapper.iter(tag('path')):
            ids=[];ancestor=node
            while ancestor is not None:ids.append(ancestor.get('id',''));ancestor=parents.get(ancestor)
            is_spine=any('__spine-bottom' in i for i in ids) or (pid=='D' and any(i in {'patch_4','patch_7','patch_10','patch_13','patch_16'} for i in ids))
            coords=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',node.get('d',''))]
            if is_spine and len(coords)==4 and coords[1]==coords[3]:
                widths=re.findall(r'stroke-width:\s*([\d.]+)',node.get('style',''))
                if widths:node.set('style',node.get('style','')+f';stroke-width:{float(widths[-1])/factor}')
        for node in list(wrapper.iter()):
            if node.tag not in {tag('text'),tag('use')}:continue
            y=float(node.get('y','0'));transform=node.get('transform','')
            match=re.search(r'translate\(([^)]+)\)',transform) if node.tag==tag('text') else None
            if match:
                vals=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',match[1])]
                if len(vals)>1:y+=vals[1]
            context=[];ancestor=node
            while ancestor is not None:context.append(ancestor.get('id',''));ancestor=parents.get(ancestor)
            fixed=node.tag==tag('text') and ('Time relative' in label(node) or '__phase-trial' in ' '.join(context))
            shift=(1-factor)*(y-center)/factor if fixed else 0
            node.set('transform',f'translate(0 {shift+(1-1/factor)*y}) scale(1 {1/factor}) '+transform)

def shorten_heatmaps(panel,pt_per_source,factor=.65):
    """Scale heatmap/trial geometry vertically; preserve glyphs and tick lengths.

    All primitive coordinates, cell payloads and tick values remain unchanged.
    The affine display transform includes phase boundaries and example arrows.
    """
    top=35.274331
    figure=next(g for g in panel.iter(tag('g')) if g.get('id')=='figure_1')
    wrapper=ET.Element(tag('g'),{'id':'figure1-heatmap-height-review','transform':f'translate(0 {(1-factor)*top}) scale(1 {factor})','data-height-factor':str(factor)})
    for child in list(figure):
        if '__heading__' not in child.get('id','') and '__canvas__' not in child.get('id',''):
            figure.remove(child);wrapper.append(child)
    figure.append(wrapper)
    # Cancel anisotropic scaling for text and marker glyphs around their anchors.
    for node in list(wrapper.iter()):
        if node.tag not in {tag('text'),tag('use')}:continue
        y=float(node.get('y','0'))
        transform=node.get('transform','')
        match=re.search(r'translate\(([^)]+)\)',transform) if node.tag==tag('text') else None
        if match:
            values=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',match[1])]
            if len(values)>1:y+=values[1]
        node.set('transform',f'translate(0 {(1-1/factor)*y}) scale(1 {1/factor}) '+transform)
    # Horizontal strokes need compensation for the y-only reduction.
    for node in wrapper.iter():
        if node.tag in {tag('path'),tag('line')} and 'stroke:' in node.get('style',''):
            nums=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',node.get('d',''))]
            if len(nums)==4 and nums[1]==nums[3]:
                widths=re.findall(r'stroke-width:\s*([\d.]+)',node.get('style',''))
                if widths:node.set('style',node.get('style','')+f';stroke-width:{float(widths[-1])/factor}')

def graphic_signature(panel):
    names={tag(t) for t in ('path','rect','line','polygon','polyline','circle','ellipse','image','use')}
    # B's explicitly identified time axes are presentation geometry. Event
    # glyphs and their scientific metadata remain included in the guard.
    axis_artists={node for group in panel.iter(tag('g')) if group.get('id','').startswith('time-axis-') for node in group.iter()}
    return Counter((node.tag,tuple(sorted((k,v) for k,v in node.attrib.items() if k not in {'style','stroke-width','transform'}))) for node in panel.iter() if node.tag in names and node not in axis_artists)


def harmonize_strokes(panel,number,pid,pt_per_source):
    """Roles verified against selected renderers/semantic SVG IDs; width only."""
    mapping={}
    if (number,pid)==(1,'D'):
        for i in [9,20,32,44,55]:mapping['line2d_'+str(i)]=('trace',.6)
        for i in [10,11,21,22,23,33,34,35,45,46,56,57]:mapping['line2d_'+str(i)]=('stimulus',.7)
        for i in [3,4,6,7,9,10,12,13,15,16]:mapping['patch_'+str(i)]=('spine',.5)
    if number==2 and pid in {'A','B'}:
        for i in [13,14,29,30]:mapping['line2d_'+str(i)]=('stimulus',.7)
        for i in [15,16,31,32]:mapping['line2d_'+str(i)]=('heatmap-phase-separator',.8)
        for i in [3,4,5,6,8,9,10,11,13]:mapping['patch_'+str(i)]=('spine',.5)
    if (number,pid)==(2,'H'):
        for i in [10,11,15,16]:mapping['line2d_'+str(i)]=('summary-curve',1.2)
        for i in [12,13,14]:mapping['line2d_'+str(i)]=('reference-or-phase-boundary',.6)
        for i in [3,4]:mapping['patch_'+str(i)]=('spine',.5)
    parents={c:p for p in panel.iter() for c in p};report=[];tick_refs=set()
    for node in panel.iter():
        if node.tag not in {tag('path'),tag('line'),tag('use')}:continue
        context=[];p=node
        while p is not None:context.append(p.get('id',''));p=parents.get(p)
        role=next((mapping[i] for i in context if i in mapping),None)
        context=' '.join(context)
        if re.search(r'xtick|ytick',context):
            role=('tick',.5)
            if node.tag==tag('use'):tick_refs.add(node.get('{http://www.w3.org/1999/xlink}href',node.get('href','')).removeprefix('#'))
        if role is None and '__' in context:
            if re.search(r'__spine-|__colorbar.*__outline',context):role=('spine',.5)
            elif re.search(r'__.*(?:cs-onset|cs-offset|us-actual|us-expected|actual-us-onset)',context):role=('stimulus',.7)
            elif re.search(r'__.*phase-separator',context):role=('heatmap-phase-separator',.8)
            elif re.search(r'__.*raw-trace|__.*raw-vigor|__.*tail-angle',context):role=('trace',.6)
        if role:
            style=node.get('style','');node.set('style',style+f';stroke-width:{role[1]/pt_per_source:.6f}')
            report.append(dict(figure=number,panel=pid,element_id=context,role=role[0],final_stroke_pt=role[1]))
    return report


def set_font(node,size,weight):
    # Inline important declarations override source class rules and shorthand fonts.
    style=node.get('style','')
    node.set('style',style+f';font-family:DejaVu Sans,sans-serif !important;font-size:{size:.6f}px !important;font-weight:{weight} !important;font-style:normal !important')
    node.set('font-family','DejaVu Sans');node.set('font-size',f'{size:.6f}');node.set('font-weight',weight)


def legend_html(text):
    return ''.join('<p>'+html.escape(p).replace('\n','<br>')+'</p>' for p in text.strip().split('\n\n') if not p.startswith('#'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--baseline',type=Path,help='Starting ticket from figure_working_copy.py begin; required after state bootstrap');parser.add_argument('--preserve-figure2-from',type=Path,help='Authenticated version-bound review to preserve Figure 2 byte-for-byte');args=parser.parse_args()
    target=args.output.resolve();assert ROOT not in target.parents, 'Permanent visual reviews must be external'
    review_day=datetime.now(ZoneInfo('Europe/Madrid'))
    assert target.parent.is_dir(),'Destination parent must already exist';assert not target.exists(),'Do not overwrite a review'
    state_path=ROOT/'docs/paper/figures/figure-01/current-state.json'
    mapping_path=ROOT/'records/figure-reviews/legend-text-relocation.json'
    state_bytes=state_path.read_bytes() if state_path.exists() else None
    state=json.loads(state_bytes) if state_bytes is not None else None
    starting=None
    if state:
        if args.baseline is None:parser.error('--baseline is required; capture it with figure_working_copy.py begin before editing')
        ticket=json.loads(args.baseline.read_text())
        if ticket.get('state_path')!=str(state_path) or str(state_path.resolve()) not in ticket.get('starting_hashes',{}):raise ValueError('Baseline ticket does not bind the current figure state')
        starting=Snapshot([]);starting.hashes=ticket['starting_hashes'];starting.check()
    dependencies=[state_path,mapping_path,Path(__file__),ROOT/'src/classical_conditioning/figure_workspace.py',
                  ROOT/'docs/paper/figures/shared/review.md',
                  *(ROOT/f'docs/paper/figures/figure-{n:02d}/legend.md' for n in (1,2)),
                  *(ROOT/'configs/paper-figures/selections'/p for p in ('figure1-freeze.json','figure1-panel-e-heatmap-rows-freeze-20261009.json','figure1-fgh-version12-freeze-20261009.json','figure2-DE-B-freeze-20261009.json','figure2-G-logmedian-freeze-20261009.json')),
                  resolve_artifact('reviews/latest_figure_assemblies_20261010.html'), *pair_dependencies()]
    if state:
        for key in ('frozen_display','draft'):
            item=state[key].get('working_svg')
            if item:
                if file_hash(item['path'])!=item['sha256']:raise ValueError('Working SVG changed: reconcile before generating a replacement')
                dependencies.append(Path(item['path']))
            evidence=state[key].get('effective_presentation')
            if evidence:
                if file_hash(evidence['path'])!=evidence['sha256']:raise ValueError('Measurement evidence changed/unavailable')
                dependencies.append(Path(evidence['path']))
        old_review=state['draft']['review'];dependencies.append(Path(old_review['path']))
        if file_hash(old_review['path'])!=old_review['sha256']:raise ValueError('Bound draft review changed/unavailable')
        for selection in state['scientific_selection']['records']:
            if file_hash(selection['path'])!=selection['sha256']:raise ValueError('Frozen selection changed; update scoped state explicitly')
    previous_review=None;authenticated_bytes=None
    if state:
        if args.preserve_figure2_from:
            authenticated_bytes=(ROOT/'records/figure-reviews/legend-review-verification.json').read_bytes()
            authenticated=json.loads(authenticated_bytes)
            if args.preserve_figure2_from.resolve()!=Path(authenticated['review_path']).resolve() or file_hash(args.preserve_figure2_from)!=authenticated['review_sha256']:
                raise ValueError('Figure 2 preservation input is not the authenticated version-bound review')
            previous_review=dict(path=str(args.preserve_figure2_from.resolve()),sha256=authenticated['review_sha256'])
            dependencies.append(ROOT/'records/figure-reviews/legend-review-verification.json')
        else:previous_review=state.get('preserved_figures',{}).get('figure2',state['draft']['review'])
        dependencies.append(Path(previous_review['path']))
        if file_hash(previous_review['path'])!=previous_review['sha256']:raise ValueError('Figure 2 preservation source changed/unavailable')
    snapshot=Snapshot(dependencies)
    bind_loaded_state(snapshot,state,state_bytes,state_path)
    if previous_review:snapshot.hashes[str(Path(previous_review['path']).resolve())]=previous_review['sha256']
    if authenticated_bytes is not None:snapshot.hashes[str((ROOT/'records/figure-reviews/legend-review-verification.json').resolve())]=digest(authenticated_bytes)
    if starting:snapshot.hashes.update(starting.hashes)
    snapshot.check()
    sources,container=load_sources();outputs=[];edits=[];layouts=[];typography=[]
    aligned_panels,aligned_registry,pair_report=build_pair()
    aligned_registry['FGH']={f'fig1__f__heatmap__example-trial-{t}':dict(scientific_role='annotation.example_trial',figure_id='fig1',panel_id='f',svg_type='g',scientific_context=dict(fish_id='20221115_07',condition='delay',trial=t,linked_panels=['D','E'],status='author-selected illustrative trial'),geometry=dict(trial=t,coordinate='heatmap trial centre; x outside right edge'),historical_source_element_id=f'fig1__g__heatmap__example-trial-{t}',role_evidence='Author requested moving G arrows to F for matching D/E Delay fish/trials on 2026-10-13; verified named trial y positions before relocation',review_status='draft; historical freeze preserved') for t in pair_report['shared_contract']['trials']}
    for source in sources:
        if state and source['number']==2:
            previous=Path(previous_review['path']).read_text()
            embedded=re.findall(r'<img class="figure"[^>]*src="data:image/svg\+xml;base64,([^"]+)"',previous)
            if len(embedded)!=4:raise ValueError('Figure 2 preservation review has an unexpected structure')
            original=base64.b64decode(embedded[3]);candidate=base64.b64decode(embedded[2])
            if original!=source['bytes']:raise ValueError('Preserved Figure 2 original does not match current scoped sources')
            outputs.append((original,candidate));layouts.append([])
            edits.append(dict(figure=2,panel='assembly',action='preserve-authenticated-review-bytes',text='Figure 2 unchanged; no regeneration',source=previous_review))
            continue
        candidate,changes,layout=build_candidate(source,aligned_panels);candidate,text_styles=harmonize_typography(candidate,layout,source['number']);typography+=text_styles;outputs.append((source['bytes'],candidate));edits+=changes;layouts.append(layout)
    for edit in edits:
        edit['destination']='docs/paper/figures/shared/review.md' if edit['panel']=='assembly' else f"docs/paper/figures/figure-{edit['figure']:02d}/legend.md"
    legend_paths=[ROOT/f'docs/paper/figures/figure-{n:02d}/legend.md' for n in (1,2)]
    legends=[p.read_text() for p in legend_paths]
    notes=(ROOT/'docs/paper/figures/shared/review.md').read_text().split('\n\n[Open the consolidated original/candidate review]')[0]
    legacy=ROOT/'legacy/helpers/plotting_style.py'
    presentation=dict(scope='Figure 1 candidate only',approval_evidence='Author requested legacy axis/tick/label style across Figure 1, right-only B arrowheads and below-axis units, and shorter heatmaps in this chat on 2026-10-11',style_source=dict(path=str(legacy),sha256=digest(legacy.read_bytes())),axis_labels=dict(size_pt=8,weight='bold',color='#000000'),ticks=dict(size_pt=7,length_pt=2,pad_pt=2,direction='out',width_pt=.5),trace_bottom_spine_offset_pt=6,heatmap_spines='retain boxed flush axes',display_transforms=dict(DE='shared native axis rectangles; no affine row stretching',FGH_height_factor=.65),B_arrowheads='right only',B_unit_label_position='below each time axis',scientific_definitions='D preserved; E recomputed from D input on author request; FGH unchanged')
    edits.append(dict(figure=1,panel='presentation',action='author-requested-legacy-axis-style-and-display-transforms',details=presentation))
    provenance=dict(date=review_day.date().isoformat(),status='review draft; no new freeze',source_container=container,original_assemblies=[s['metadata'] for s in sources],edits=edits,layouts=layouts,typography=typography,presentation=presentation,aligned_pair=pair_report,legend_sources=[dict(path=str(p),sha256=digest(p.read_bytes())) for p in legend_paths],generator=dict(path=str(Path(__file__).resolve()),sha256=digest(Path(__file__).read_bytes()),source_base64=base64.b64encode(Path(__file__).read_bytes()).decode()),checks=dict(selected_source_hashes=True,selected_frozen_bytes_preserved=True,unmodified_panels_coordinates_and_payloads_preserved=True,DE_mapping_verified=pair_report['verification'],uniform_panel_fitting=True,author_requested_resizing=True,display_y_transforms_recorded=True,figure1_panel_d_vector_svg=True),width_mm=183)
    css='body{font:16px/1.55 system-ui;color:#20272c;background:#eceff1;margin:0}main{max-width:1100px;margin:auto;padding:24px}header,section,details{background:white;padding:24px;margin:20px 0;border-radius:8px}h1{font-size:28px}nav{display:flex;gap:20px}button{padding:8px 16px;cursor:pointer;border:1px solid #789;background:white;border-radius:4px;margin:0 6px 12px 0}button.active{background:#20272c;color:white}.figure{display:block;width:100%;height:auto;background:white}.version[hidden]{display:none}.legend{max-width:85ch;border-top:1px solid #ddd;padding-top:20px}pre{white-space:pre-wrap;font-size:12px}.status{color:#59656e}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}@media print{body{background:white}button,nav{display:none}.version.original{display:none}header,section{padding:0;border-radius:0}.figure{width:183mm;max-width:100%}}'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Figures 1–2: legend and layout drafts</title><style>'+css+'</style><main><header><h1>Figures 1–2 · legend and layout drafts</h1><p>11 October 2026. Candidate shown first. Switch to the authenticated original to compare. Frozen sources are preserved. Draft D/E share measured frames, event guides and aligned axes; E raw and scaled vigor are recomputed from D’s stored vigor. Other panels retain their selected data. Figure 1 uses legacy axis styling and shorter F–H heatmaps. Figure 2 is unchanged. These drafts are not frozen.</p><nav><a href="#figure1">Figure 1</a><a href="#figure2">Figure 2</a><a href="#concerns">Review notes</a></nav></header>'
    page=page.replace('11 October 2026.',review_day.strftime('%d %B %Y')+'.',1)
    for n,(old,new) in enumerate(outputs,1):
        page+=f'<section id="figure{n}"><h2>Figure {n}</h2><button class="active" onclick="showVersion({n},\'candidate\',this)">Decluttered draft</button><button onclick="showVersion({n},\'original\',this)">Original selected assembly</button>'
        for kind,data in [('candidate',new),('original',old)]:
            page+=f'<div class="version {kind}"'+(' hidden' if kind=='original' else '')+'><img class="figure" alt="Figure '+str(n)+' '+kind+'" src="data:image/svg+xml;base64,'+base64.b64encode(data).decode()+'"></div>'
        page+='<div class="legend"><h3>Draft legend</h3>'+legend_html(legends[n-1])+'</div></section>'
    page+='<section id="concerns"><h2>Review notes</h2>'+legend_html(notes)+'</section><details><summary>Text-relocation record and exact provenance</summary><pre>'+html.escape(json.dumps(provenance,indent=2))+'</pre></details><script type="application/json" id="review-provenance">'+json.dumps(provenance).replace('<','\\u003c')+'</script><script>function showVersion(n,kind,button){const section=document.getElementById("figure"+n);section.querySelectorAll(".version").forEach(el=>el.hidden=!el.classList.contains(kind));section.querySelectorAll("button").forEach(el=>el.classList.toggle("active",el===button));}</script></main></html>'
    assert not re.search(r'(?:src|href)="(?:https?://|file:)',page)
    page_bytes=page.encode()
    assert digest(resolve_artifact('reviews/latest_figure_assemblies_20261010.html').read_bytes())==container['sha256']
    mapping=dict(review_path=str(target),review_sha256=digest(page_bytes),edits=edits,layouts=layouts,checks=provenance['checks'],typography=typography)
    if state:
        draft_svg,meta=editable_svg(outputs[0][1],kind='Figure 1: current unfrozen presentation draft',source_records=sources[0]['metadata']['panels'],registries=aligned_registry)
        svg_path=target.with_suffix('.svg')
        measurements=measured_presentation(draft_svg)
        measurements['axis_findings']=draft_axis_findings(measurements)
        measurement_path=ROOT/'records/figure-reviews/figure-01'/f'{target.stem}-effective-presentation.json'
        measurement_bytes=(json.dumps(measurements,indent=2)+'\n').encode()
        snapshot.hashes[str(measurement_path.resolve())]=file_hash(measurement_path)
        state['draft'].update(review=dict(path=str(target),sha256=digest(page_bytes)),
                              working_svg=dict(path=str(svg_path),sha256=digest(draft_svg),source_svg_sha256=digest(outputs[0][1])),
                              source_assembly_sha256=digest(outputs[0][1]),
                              effective_presentation=dict(path=str(measurement_path),sha256=digest(measurement_bytes)))
        state.pop('effective_presentation',None)
        state['draft']['semantic_registry']=dict(scope='D/E explicit artists and cross-panel relationships; F selected-trial annotations',scientific_records=len(meta['scientific_registry']),whole_figure_complete=False,relationships=pair_report['shared_contract'],verification=pair_report['verification'])
        state['draft']['scientific_changes']=pair_report['scientific_changes']
        state['outstanding_questions']=[q for q in state.get('outstanding_questions',[]) if 'trial 9' not in str(q).lower() and 'arrowheads' not in str(q).lower()]
        state['draft']['example_arrows']=dict(panel='F',fish='20221115_07',trials=[9,17,63,66,93],linked_panels=['D','E'],heatmap_values_unchanged=True)
        state['verification']=dict(current_sources_authenticated=True,editable_source_structure=True,
                                   visual_review='pending review of this new candidate',
                                   measured_axis_findings=len(measurements['axis_findings']),
                                   concurrency='Starting ticket and captured inputs checked under exclusive publication lock')
        state['preserved_figures']={'figure2':previous_review}
        # Current state is the sole pointer. Legacy mapping is a version-bound
        # evidence record; write it only after guarded artifact publication.
        mapping['state_authority']=str(state_path)
        publish_state(state_path,state,snapshot,[(target,page_bytes),(svg_path,draft_svg),(measurement_path,measurement_bytes)],
                      [(mapping_path,(json.dumps(mapping,indent=2)+'\n').encode())])
    else:
        # Bootstrap requires explicit working-copy preparation before subsequent
        # guarded figure edits. Existing workflows retain their behavior here.
        snapshot.check()
        with target.open('xb') as stream:stream.write(page_bytes)
        from classical_conditioning.artifacts import write_json_atomic
        write_json_atomic(mapping_path,mapping)
    print(target);print(json.dumps(dict(text_edits=len(edits),single_self_contained_review=True)))
if __name__=='__main__':main()
