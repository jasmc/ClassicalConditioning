"""Five H-only candidate views, using selected artwork and verified fish caches.

This comparison never edits Figure 1 or its selection. Shared trial labels are
copied from the same assembly and translated with the data-axis origin.
"""
from __future__ import annotations
import base64
import copy
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image
from classical_conditioning.figure_workspace import Snapshot,sha,tag,publish_state
from classical_conditioning.figure_freeze import _transform,_multiply
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme
from replace_figure1_trace3_example import bound,load,ref,existing_sampling,matrix,svg_editing_page
from build_figure1_fgh_version_comparison import prefix_svg
from analyze_figure1_trace3_learner_candidates import summarize

REVIEWS=Path(r'J:\ClassicalConditioning Outputs\current\reviews')
PRIOR=REVIEWS/'figure-01-trace3-learner-selection-20261019'
DATA=REVIEWS/'figure-02-3strace/fish'
OUT=REVIEWS/'figure-01-panel-H-five-options-20261020'
STATE=ROOT/'docs/paper/figures/figure-01/current-state.json'
TICKET=ROOT/'records/figure-reviews/figure-01/panel-h-five-options-start-20261020.json'
IDENTITY=(1.,0.,0.,1.,0.,0.)
HREF='{http://www.w3.org/1999/xlink}href'
IMAGE='edit-FGH-fig1__g__heatmap__vigor-cells'


def relative(node,stop,parents):
    chain=[]
    while node is not stop:
        chain.append(node);node=parents[node]
    result=IDENTITY
    for n in reversed(chain):result=_multiply(result,_transform(n.get('transform','')))
    return result


def inverse(m):
    a,b,c,d,e,f=m;det=a*d-b*c
    if abs(det)<1e-12:raise ValueError('Singular panel transform')
    return d/det,-b/det,-c/det,a/det,(c*f-d*e)/det,(b*e-a*f)/det


def point(m,x,y):
    a,b,c,d,e,f=m;return a*x+c*y+e,b*x+d*y+f


def image_origin(node,stop,parents):
    m=relative(node,stop,parents)
    x,y,w,h=[float(node.get(k)) for k in ('x','y','width','height')]
    corners=[point(m,a,b) for a,b in [(x,y),(x+w,y),(x,y+h),(x+w,y+h)]]
    return min(p[0] for p in corners),min(p[1] for p in corners)


def scope(source):
    original=ET.fromstring(source)
    lookup={n.get('id'):n for n in original.iter() if n.get('id')}
    parents={c:p for p in original.iter() for c in p}
    targets={lookup['layout-G'],lookup['edit-FGH-fig1__h__colorbar__axes']}
    ancestors=set()
    for node in targets:
        while node in parents:
            node=parents[node];ancestors.add(node)
    def prune(node):
        if node in targets or node.tag in (tag('defs'),tag('style')):return copy.deepcopy(node)
        if node not in ancestors:return None
        result=ET.Element(node.tag,dict(node.attrib));result.text=node.text;result.tail=node.tail
        for child in node:
            kept=prune(child)
            if kept is not None:result.append(kept)
        return result
    scoped=prune(original)
    # The source has shared trial tick/phase labels in displayed F. Move only
    # these labels by the measured data-axis displacement into the H-only view.
    body=lookup['edit-FGH-figure_1'];howner=lookup['layout-G']
    fxy=image_origin(lookup['edit-FGH-fig1__h__heatmap__vigor-cells'],body,parents)
    hxy=image_origin(lookup[IMAGE],body,parents)
    shift=(1.,0.,0.,1.,hxy[0]-fxy[0],hxy[1]-fxy[1])
    hmatrix=relative(howner,body,parents)
    destination=next(n for n in scoped.iter() if n.get('id')=='layout-G')
    shared_ids=['edit-FGH-fig1__f__heatmap__axis-y',*[f'edit-FGH-fig1__f__heatmap__phase-label-{p}' for p in ('pre','train','test')]]
    for identifier in shared_ids:
        node=lookup[identifier]
        clone=copy.deepcopy(node)
        transform=_multiply(inverse(hmatrix),_multiply(shift,relative(node,body,parents)))
        clone.set('transform','matrix('+','.join(f'{v:.15g}' for v in transform)+')')
        destination.append(clone)
    for parent in list(scoped.iter()):
        for node in list(parent):
            if node.get('id')=='edit-FGH-fig1__g__heatmap__axis-y':parent.remove(node)
    # Copy font resources once to the comparison root, rather than per option.
    fonts=[copy.deepcopy(n) for n in scoped if n.tag==tag('defs')]
    for n in list(scoped):
        if n.tag==tag('defs'):scoped.remove(n)
    return scoped,fonts,dict(source_groups=['layout-G','edit-FGH-fig1__h__colorbar__axes'],
        shared_trial_label_sources=shared_ids,shared_label_translation_native=list(shift),
        other_panels_removed=True,original_coordinate_transforms_preserved=True)


def main():
    ticket=load(TICKET)
    for path,expected in ticket['starting_hashes'].items():bound(path,expected)
    state=load(STATE);draft=state['draft'];current=next(i['fish_id'] for i in draft['condition_order'] if i['display_panel']=='H')
    source_path=bound(draft['working_svg']['path'],draft['working_svg']['sha256'])
    prior=load(PRIOR/'selection-report.json')
    windows_path=REVIEWS/'figure-02-3strace/matched-delay-statistics/windows.parquet'
    plan_path=windows_path.with_name('plan.json')
    for path in (windows_path,plan_path):bound(path,prior['sources_sha256'][str(path)])
    paths=[*map(Path,ticket['starting_hashes']),TICKET,Path(__file__),
        ROOT/'scripts/replace_figure1_trace3_example.py',ROOT/'scripts/build_figure1_fgh_version_comparison.py',
        ROOT/'scripts/analyze_figure1_trace3_learner_candidates.py',ROOT/'scripts/build_figure_legend_review.py',
        ROOT/'configs/paper-figures/nature-main.json',ROOT/'configs/paper-figures/figure-elements.json',ROOT/'configs/paper-figures/layout-frames.json']
    existing=[]
    if OUT.is_dir():
        previous=load(OUT/'review.json')
        existing=[OUT/'review.json',*[OUT/name for name in previous['outputs']]]
        for name,expected in previous['outputs'].items():bound(OUT/name,expected)
    snapshot=Snapshot([*paths,*existing])
    scores=summarize(pd.read_parquet(windows_path))
    eligible=scores.loc[scores.condition.eq('trace')&scores.pre_n.eq(10)&scores.late_n.eq(25)&
        ~scores.fish_id.isin([current,'20230802_08','20230309_09'])].sort_values('rank_trace_primary').head(5)
    fish=eligible.fish_id.tolist()
    if fish!=ticket['fish']:raise ValueError('Candidate ordering changed')
    roster={v['recording_id']:v['condition'] for v in load(plan_path)['selected']}
    raw=source_path.read_bytes();source=ET.fromstring(raw)
    source_image=next(n for n in source.iter() if n.get('id')==IMAGE)
    old_png=np.asarray(Image.open(io.BytesIO(base64.b64decode(source_image.get(HREF).split(',',1)[1]))).convert('RGBA'))
    template,fonts,isolation=scope(raw)
    frames=json.loads(next(n.text for n in source if n.get('id')=='layout-frames'))
    hframe=next(a['box_mm'] for a in frames['allocations'] if a['id']=='H')
    # Extend the isolated H frame to include its required trial labels and
    # source shared colour scale. All nested source transforms remain intact.
    left=hframe[0]-8.;top=hframe[1];right=136.;height=hframe[3];width=right-left
    units=float(source.get('viewBox').split()[2])/136.
    view=[left*units,top*units,width*units,height*units]
    policy=load(ROOT/'configs/paper-figures/layout-frames.json');inset=policy['selected_inset_mm']
    margin=inset;gap=2*inset;cols=3;rows=2
    total_width=2*margin+cols*width+(cols-1)*gap;total_height=2*margin+rows*height+(rows-1)*gap
    result=ET.Element(tag('svg'),dict(viewBox=f'0 0 {total_width} {total_height}',width=f'{total_width}mm',
        height=f'{total_height}mm',role='img',**{'aria-label':'Five alternative fish for Figure 1 panel H'}))
    ET.SubElement(result,tag('rect'),id='comparison-background',x='0',y='0',width=str(total_width),height=str(total_height),fill='white')
    for node in fonts:result.append(node)
    options=[]
    profile=load_presentation_profile('nature-main')
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        cmap=mpl.colormaps[theme.intensity_cmap].copy();cmap.set_bad(theme.heatmap_bad_color)
        pixel_rows,pixel_columns=existing_sampling(old_png,matrix(DATA/(current+'-bins.parquet')),cmap)
        for index,f in enumerate(fish):
            record_path=DATA/(f+'.json');bins_path=DATA/(f+'-bins.parquet');record=load(record_path)
            if roster.get(f)!='trace' or record['recording_id']!=f or record['heatmap']!='complete':raise ValueError('Candidate source/roster mismatch')
            expected=next(o['sha256'] for o in record['outputs'] if o['path']==str(bins_path));bound(bins_path,expected)
            cadence=record['heatmap_provenance']['cadence']
            if cadence['buffer_capacity_exceeded'] or cadence['missing_frame_ids']:raise ValueError('Candidate timing gate failed')
            values=matrix(bins_path)
            rgba=cmap(mpl.colors.Normalize(-.25,.25)(values[::-1]),bytes=True)[pixel_rows[:,None],pixel_columns[None,:]]
            image=Image.fromarray(rgba);buffer=io.BytesIO();image.save(buffer,format='PNG')
            panel=copy.deepcopy(template)
            panel.attrib.update(viewBox=' '.join(map(str,view)),x=str(margin+(index%cols)*(width+gap)),
                y=str(margin+(index//cols)*(height+gap)),width=str(width),height=str(height),overflow='hidden')
            panel.attrib.pop('id',None)
            next(n for n in panel.iter() if n.get('id')==IMAGE).set(HREF,'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
            labels=[n for n in panel.iter() if n.tag==tag('text') and n.text==current]
            if len(labels)!=1:raise ValueError('Ambiguous panel fish label')
            labels[0].text=f
            panel=ET.fromstring(prefix_svg(ET.tostring(panel,encoding='utf-8'),'option-'+f))
            panel.set('id','option-'+f)
            result.append(panel)
            row=eligible.loc[eligible.fish_id.eq(f)].iloc[0]
            options.append(dict(fish_id=f,score_rank=int(row.rank_trace_primary),score=float(row.primary_pre5_14_minus_late40_64),
                pre_outcomes=10,late_outcomes=25,fish_record=ref(record_path),bins=ref(bins_path),
                finite_cells=int(np.isfinite(values).sum()),nominal_us_s=13,
                measured_expected_us_s=record['heatmap_provenance']['expected_us_s'],raster_sha256=sha(rgba.tobytes())))
    legend=('Five unfrozen alternatives to the current panel H fish, ordered by the existing descriptive acquisition score among fish with complete Pre/late-training outcome coverage. '
        'The current fish and previously rejected abrupt example are excluded. Each heatmap shows direct 1-s median log positive eligible bout-frame vigor, '
        'centred within trial by the median finite baseline-bin value over [−15,0) s; no amplitude scaling. '
        'Shared colour limits ±0.25 log units; black indicates missing data. Rows 1–90 correspond to acquisition trials 5–94. '
        'Green guides: CS onset/offset; violet dotted guide: expected US at 13 s. Boundaries separate Pre, Train and Test. '
        'These are descriptive options, without an inferred learning-onset trial.')
    metadata=dict(scope='Five H-only alternatives; Figure 1 unchanged',current_fish=current,selection=ticket['selection'],
        source_svg=ref(source_path),source_state=ref(STATE),options=options,isolation=isolation,
        crop_viewBox=view,source_crop_physical_mm=[left,top,width,height],
        layout=dict(inset_mm=inset,shared_outside_margin_mm=margin,between_options_mm=gap,final_size_mm=[total_width,total_height]),
        source_coordinate_geometry_preserved=True,profile_inputs=profile.inputs,
        inputs={p:h for p,h in snapshot.hashes.items() if Path(p) not in existing})
    ET.SubElement(result,tag('metadata'),id='panel-H-options-provenance').text=json.dumps(metadata)
    svg=ET.tostring(result,encoding='utf-8')
    page=svg_editing_page(svg,scope_label='Figure 1 panel H alternatives',legend=legend)
    page=page.replace('width:183mm;',f'width:{total_width}mm;')
    snapshot.check()
    payloads=[('panel-H-options.svg',svg),('panel-H-options.html',page.encode()),('legend.md',(legend+'\n').encode())]
    metadata['outputs']={name:sha(data) for name,data in payloads}
    if existing:
        publish_state(OUT/'review.json',metadata,snapshot,evidence_updates=[(OUT/name,data) for name,data in payloads])
    else:
        OUT.mkdir(exist_ok=False)
        for name,data in payloads:
            with (OUT/name).open('xb') as stream:stream.write(data)
            if sha((OUT/name).read_bytes())!=sha(data):raise ValueError('Saved output mismatch')
        snapshot.check()
        with (OUT/'review.json').open('x',encoding='utf-8') as stream:json.dump(metadata,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(OUT),fish=fish,source_crop_mm=[left,top,width,height],svg_sha256=sha(svg))))


if __name__=='__main__':main()
