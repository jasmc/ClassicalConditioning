"""Build scoped Figure 2 A with the selected Figure 1 F–H pipeline and styles.

Prepare authenticated numerical data first, inspect the saved render, and publish
with --publish. Sources and the pre-edit baseline are rechecked under the scoped
state lock; evidence and artifacts precede the atomic state pointer.
"""
from pathlib import Path
import argparse, base64, copy, io, json, sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
from matplotlib.colors import Normalize
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot, sha, tag, measured_presentation, publish_state
from classical_conditioning.figures.pooled_log_heatmap import analyze_recording, pool_equal_fish, StreamingSnapshot
from render_legacy_ssd_figure2_delay import _verify_cohort
from build_figure_legend_review import svg_editing_page, figure_font_paths

STATE=ROOT/'docs/paper/figures/figure-02/panel-a/current-state.json'
SELECTION=ROOT/'configs/paper-figures/selections/figure1-fgh-historical-one-second-managua-freeze-20261013.json'
OUT=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-02-a-pooled')
EVIDENCE=ROOT/'records/figure-reviews/figure-02/pooled-heatmap-evidence.json'
LEGEND=ROOT/'docs/paper/figures/figure-02/legend.md'
README=ROOT/'docs/paper/figures/figure-02/README.md'
SOURCE_MODULE=ROOT/'src/classical_conditioning/figures/pooled_log_heatmap.py'

def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def artifact(path):return dict(path=str(path.resolve()),sha256=sha256_file(path))

def prepare(project):
    if not OUT.parent.is_dir():raise FileNotFoundError(OUT.parent)
    OUT.mkdir(exist_ok=True)
    cohort,cohort_hash,cohort_inputs=_verify_cohort(project)
    if (OUT/'calculation.json').exists():
        saved=json.loads((OUT/'calculation.json').read_text())
        if saved['cohort_hash']!=cohort_hash:raise ValueError('Preserve the previous cohort calculation')
        for key in ('calculation','fish_bins','pooled_bins'):
            a=saved[key]
            if sha256_file(Path(a['path']))!=a['sha256']:raise ValueError('A bound calculation artifact was edited: '+a['path'])
        for record in saved['records']:StreamingSnapshot(record['sources']).check()
        return
    module_hash=sha256_file(SOURCE_MODULE);parts=[];records=[]
    for i,row in enumerate(cohort.itertuples(index=False),1):
        rid=str(row.recording_id);data_path=OUT/(rid+'.parquet');record_path=data_path.with_suffix('.json')
        if data_path.exists() and record_path.exists():
            record=json.loads(record_path.read_text())
            if record['calculation_sha256']!=module_hash:raise ValueError('Preserve incompatible cache: '+str(record_path))
            if sha256_file(data_path)!=record['data_sha256']:raise ValueError('Edited numerical cache')
            StreamingSnapshot(record['sources']).check()
            table=pd.read_parquet(data_path)
        else:
            print(f'Analyzing {i}/{len(cohort)} {row.condition_id} {rid}',flush=True)
            matrix,table,record=analyze_recording(project,rid,'-v1','allDelay',str(row.condition_id))
            table.to_parquet(data_path,index=False)
            record.update(calculation_sha256=module_hash,data_sha256=sha256_file(data_path))
            write(record_path,record)
        parts.append(table);records.append(record)
        print(f'Ready {i}/{len(cohort)} {rid}',flush=True)
    bins=pd.concat(parts,ignore_index=True)
    pooled=pool_equal_fish(bins,cohort.set_index('recording_id').condition_id.to_dict())
    bins.to_parquet(OUT/'fish-bins.parquet',index=False);pooled.to_parquet(OUT/'pooled-bins.parquet',index=False)
    write(OUT/'calculation.json',dict(cohort_hash=cohort_hash,cohort_inputs=cohort_inputs,cohort_counts=cohort.groupby('condition_id').size().to_dict(),records=records,calculation=artifact(SOURCE_MODULE),fish_bins=artifact(OUT/'fish-bins.parquet'),pooled_bins=artifact(OUT/'pooled-bins.parquet')))

def compare_presentation(svg, template, width_mm):
    """Measure every surviving named text, stroke and tick at the native scale."""
    reference=measured_presentation(template,width_mm=183)
    current=measured_presentation(svg,width_mm=width_mm)
    def address(row,prefix=''):
        ids=[x.removeprefix(prefix) for x in row['context'] if x.startswith(prefix+'fig1__')]
        return ids[-1] if ids else None
    checks=[]
    properties={'texts':['font_family','weight','font_height_pt','font_width_scale_pt'],
                'strokes':['color','effective_opacity','width_range_pt','normal_width_pt'],
                'ticks':['length_pt','normal_width_pt','color','effective_opacity']}
    for kind,props in properties.items():
        index={address(r):r for r in reference[kind] if address(r)}
        for row in current[kind]:
            key=address(row,'fig2-a-')
            if not key:raise ValueError('Unmapped measured element')
            old=index[key]
            for prop in props:
                if prop not in row and prop not in old:continue
                a,b=row.get(prop),old.get(prop)
                if isinstance(a,(float,int,list)) and isinstance(b,(float,int,list)):
                    np.testing.assert_allclose(a,b,rtol=0,atol=1e-7)
                elif a!=b:raise ValueError('Style differs: '+key+' / '+prop)
            checks.append({'source_element_id':key,'properties':props,'status':'matched'})
    if current['unresolved']:raise ValueError('Unresolved presentation measurements')
    return current,dict(status='verified',checks=checks,source_svg_sha256=sha(template),candidate_svg_sha256=sha(svg),tolerance_pt=1e-7)

def verify_examples(selected,metadata):
    """Reproduce all cells and eligible counts of the frozen F/H example fish."""
    source=selected['data'];path=Path(source['path'])
    if sha256_file(path)!=source['sha256']:raise ValueError('Frozen numerical reference changed')
    fish_path=Path(metadata['fish_bins']['path'])
    if sha256_file(fish_path)!=metadata['fish_bins']['sha256']:raise ValueError('Edited fish-level bins')
    reference=pd.read_csv(path);new=pd.read_parquet(fish_path);checks=[]
    for panel,fish in [('F','20221115_07'),('H','20221115_09')]:
        a=reference[reference.panel.eq(panel)].sort_values(['acquisition_trial','bin_index'])
        b=new[new.fish.eq(fish)].sort_values(['acquisition_trial','bin_index'])
        np.testing.assert_array_equal(a[['acquisition_trial','bin_index']],b[['acquisition_trial','bin_index']])
        np.testing.assert_array_equal(a.eligible_frames,b.eligible_frames)
        for field in ('bin_log_vigor','baseline_log_vigor','delta_log_vigor'):
            np.testing.assert_allclose(a[field],b[field],atol=1e-12,rtol=0,equal_nan=True)
        checks.append(dict(fish=fish,reference_panel=panel,cells=len(a),same_eligible_frame_counts=True,same_missing_mask=True,bin_log_vigor_and_baseline_verified=True,maximum_absolute_difference=float(np.nanmax(np.abs(a.delta_log_vigor.to_numpy()-b.delta_log_vigor.to_numpy())))))
    return dict(reference_data=source,checks=checks,status='verified')

def render():
    selected=json.loads(SELECTION.read_text());template=Path(selected['svg']['path'])
    if sha256_file(template)!=selected['svg']['sha256']:raise ValueError('Selected F–H template changed')
    metadata=json.loads((OUT/'calculation.json').read_text());example_reproduction=verify_examples(selected,metadata);pool=pd.read_parquet(OUT/'pooled-bins.parquet')
    if sha256_file(OUT/'pooled-bins.parquet')!=metadata['pooled_bins']['sha256']:raise ValueError('Edited pooled bins')
    freeze=Path(selected['freeze_manifest']['path'])
    if sha256_file(freeze)!=selected['freeze_manifest']['sha256']:raise ValueError('Frozen reference registry changed')
    raw=template.read_bytes();root=ET.fromstring(raw);old_registry=json.loads(freeze.read_text())['element_registry']
    figure=next(n for n in root if n.get('id')=='figure_1');lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    shift=.265*518.740157;new_width=518.740157-shift
    root.set('width',f'{new_width}pt');root.set('viewBox',f'0 0 {new_width} 221.102362')
    # Retain the source axes dimensions. Only the unused third heatmap is removed.
    keep={'fig1__f__canvas__background','fig1__f__heatmap__axes','fig1__g__heatmap__axes','fig1__h__colorbar__axes',*[f'fig1__{p}__heading__{s}' for p in 'fg' for s in ['condition','fish-id']], 'fig1__f__heading__panel-letter'}
    for node in list(figure):
        if node.get('id') not in keep:figure.remove(node)
    lookup['fig1__h__colorbar__axes'].set('transform',f'translate({-shift} 0)')
    for node in root.iter():
        for child in list(node):
            if child.get('id','').startswith('fig1__f__heatmap__example-trial-'):node.remove(child)
    # Expected US occurs only in the paired Delay condition, at 9 s.
    f_axes=lookup['fig1__f__heatmap__axes'];g_axes=lookup['fig1__g__heatmap__axes']
    us=copy.deepcopy(lookup['fig1__f__heatmap__us-expected']);f_axes.remove(lookup['fig1__f__heatmap__us-expected'])
    us_index=list(g_axes).index(lookup['fig1__g__heatmap__us-expected'])
    g_axes.remove(lookup['fig1__g__heatmap__us-expected']);us.set('id','fig1__g__heatmap__us-expected')
    wrapper=ET.Element(tag('g'),{'transform':f'translate({shift} 0)'});wrapper.append(us);g_axes.insert(us_index,wrapper)
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black');norm=Normalize(-.25,.25,clip=True)
    href='{http://www.w3.org/1999/xlink}href'
    heading_styles={p:next(n for n in lookup[f'fig1__{p}__heading__condition'].iter() if n.tag==tag('text')).get('style') for p in 'fgh'}
    for panel,condition,name,style_source in [('f','control','Control','h'),('g','delay','Delay','f')]:
        matrix=pool[pool.condition_id.eq(condition)].pivot(index='acquisition_trial',columns='bin_index',values='mean_delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
        group=lookup[f'fig1__{panel}__heatmap__vigor-cells'];group.set('data-scientific-role','heatmap.vigor');group.set('data-condition',condition);group.set('data-cohort-hash',metadata['cohort_hash']);group.set('data-pooled-values-sha256',sha(np.asarray(matrix,dtype='<f8').tobytes()));image=next(n for n in group.iter() if n.tag==tag('image'))
        original=Image.open(io.BytesIO(base64.b64decode(image.get(href).split(',')[1])))
        rgba=np.asarray(cmap(norm(np.ma.masked_invalid(matrix)))*255,dtype=np.uint8)
        raster=Image.fromarray(rgba[::-1]).resize(original.size,resample=Image.Resampling.NEAREST)
        buffer=io.BytesIO();raster.save(buffer,format='PNG');image.set(href,'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
        heading=next(n for n in lookup[f'fig1__{panel}__heading__condition'].iter() if n.tag==tag('text'))
        heading.set('style',heading_styles[style_source]);heading.text=name
        next(n for n in lookup[f'fig1__{panel}__heading__fish-id'].iter() if n.tag==tag('text')).text=f'n = {metadata["cohort_counts"][condition]}'
    next(n for n in lookup['fig1__f__heading__panel-letter'].iter() if n.tag==tag('text')).text='A'
    for parent in root.iter():
        for child in list(parent):
            if child.tag==tag('metadata'):parent.remove(child)
    # Rebind copied style roles to their explicit population sources and addresses.
    registry={};names={n.get('id'):'fig2-a-'+n.get('id') for n in root.iter() if n.get('id')}
    for node in root.iter():
        old=node.get('id')
        if old:
            node.set('id',names[old]);node.set('data-style-source-id',old)
            if old in old_registry:
                record=copy.deepcopy(old_registry[old]);record.update(element_id=names[old],figure_id='fig2',panel_id='a',style_reference={'svg':selected['svg'],'element_id':old},status='unfrozen population draft')
                record.pop('style_verification',None)
                condition='control' if old.startswith('fig1__f__') else 'delay' if old.startswith('fig1__g__') else 'shared'
                record['scientific_context']=dict(selected['scientific_definition'],condition=condition,cohort_hash=metadata['cohort_hash'],pooling='equal-fish mean of individually baseline-centred 1 s bins',data=metadata['pooled_bins'])
                if old.endswith('fish-id'):
                    record.update(scientific_role='label.cohort_count',geometry={'cohort_fish':metadata['cohort_counts'][condition]})
                if old.endswith('us-expected'):record['geometry']={'time_s':9,'training_trial_extent':[10.5,60.5]}
                if record.get('mappable_ids'):record['mappable_ids']=[names['fig1__'+p+'__heatmap__vigor-cells'] for p in 'fg']
                if old.endswith('heading__condition'):
                    style_panel='h' if condition=='control' else 'f'
                    style_key=f'fig1__{style_panel}__heading__condition'
                    record['style_reference']['element_id']=style_key
                    record['resolved_color']=old_registry[style_key].get('resolved_color')
                if record.get('composite_id'):record['composite_id']='fig2-a-'+record['composite_id']
                record.pop('renderer_properties',None)
                registry[names[old]]=record
        for key,value in list(node.attrib.items()):
            if 'url(#' in value:
                for source,target in names.items():value=value.replace('url(#'+source+')','url(#'+target+')')
                node.set(key,value)
            if key.endswith('href') and value.startswith('#'):node.set(key,'#'+names[value[1:]])
    defs=ET.SubElement(root,tag('defs'));css=[]
    for weight,path in zip((400,700),figure_font_paths()):
        css.append('@font-face{font-family:"DejaVu Sans";font-weight:'+str(weight)+';src:url(data:font/ttf;base64,'+base64.b64encode(path.read_bytes()).decode()+') format("truetype")}')
    ET.SubElement(defs,tag('style')).text=''.join(css)
    ET.SubElement(root,tag('metadata'),{'id':'figure-02-a-draft-provenance'}).text=json.dumps({'scope':['A'],'pipeline_reference':artifact(SELECTION),'style_reference':selected['svg'],'pooled_data':metadata['pooled_bins'],'cohort_hash':metadata['cohort_hash'],'cohort_counts':metadata['cohort_counts'],'frozen':False,'native_width_mm':new_width*25.4/72})
    svg=ET.tostring(root)
    (OUT/'candidate.svg').write_bytes(svg)
    legend=f'(A) Delay and matched control population heatmaps ({metadata["cohort_counts"]["delay"]} Delay fish; {metadata["cohort_counts"]["control"]} controls). Each fish/trial uses the selected Figure 1 F–H pipeline: cadence-reconstructed timing, legacy distal angular vigor in rad/ms and finite positive valid adjacent detected-bout frames; natural log per original frame; direct medians within left-inclusive/right-exclusive 1 s bins; subtract the median of finite baseline-bin medians over [−15, 0) s, each finite bin one vote. One baseline bin is sufficient; an empty baseline makes the whole fish/trial undefined. Each population cell is the arithmetic mean of available fish-level centred values, with one vote per contributing fish. Missing cells are black; contributing-fish counts are retained in the associated data. No amplitude scaling or numerical clipping; native managua_r saturates visually at −0.25 and +0.25. Display trials 1–90 map to acquisition trials 5–94. Green guides show CS onset and nominal 10 s offset; the dotted violet guide marks the expected 9 s US during paired Delay training. White boundaries separate Pre, Train and Test. This descriptive scientific draft is unfrozen.'
    full_legend=LEGEND.read_text();start=full_legend.index('(A, B)' if '(A, B)' in full_legend else '(A)');end=full_legend.index('(D, E)')
    retained='(B) 3 s Trace (40 fish) versus its matched control cohort (19 fish) retains the earlier signed bout-log-vigor calculation in 0.5 s bins, referenced to [−15, 0) s; population values average fish equally. It has not been updated by this Delay-only request. (C) 10 s Trace population heatmaps: pending authenticated inputs; no result is shown.\n\n'
    (OUT/'legend.md').write_text(full_legend[:start]+legend+'\n\n'+retained+full_legend[end:])
    readme=README.read_text().replace('| A | Control versus Delay equal-fish pooled signed bout-log-vigor heatmaps, managua_r at −0.25 to +0.25 |','| A | Control versus Delay, direct 1 s log-frame medians centred within each fish/trial, pooled with equal fish weight; managua_r at −0.25 to +0.25 |')
    if 'Current A presentation draft:' not in readme:readme+='\nCurrent A presentation draft: [scoped state](panel-a/current-state.json). The author requested the selected Figure 1 F–H frame pipeline and exact element formatting for the Delay/control pooled heatmaps on 2026-10-13. This draft is unfrozen. Its own review and editable SVG are authoritative for A; the earlier whole-figure review predates this update.\n'
    (OUT/'README.md').write_text(readme)
    (OUT/'candidate.html').write_text(svg_editing_page(svg,scope_label='Figure 2 A',legend=legend).replace('width:183mm',f'width:{new_width*25.4/72:g}mm'))
    measurements,style_comparison=compare_presentation(svg,raw,new_width*25.4/72)
    report=dict(environment={'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},renderer_dependencies=[artifact(ROOT/'scripts/build_figure_legend_review.py'),artifact(ROOT/'src/classical_conditioning/figure_workspace.py')],example_reproduction=example_reproduction,reference_registry=selected['freeze_manifest'],generator=artifact(Path(__file__)),reproduction='.venv/bin/python scripts/build_figure2_pooled_heatmaps.py --prepare --render; inspect the saved candidate; publish with the captured baseline ticket',font_resources=[artifact(p) for p in figure_font_paths()],style_comparison=style_comparison,scope=['A'],pipeline_reference=artifact(SELECTION),style_reference=selected['svg'],physical_width_mm=new_width*25.4/72,template_axis_scale=1,style_policy='Cloned exact selected F/G/H elements at their original physical scale; headings/counts, panel identity, data and paired-US identity rebound; example-fish arrows omitted',element_registry=registry,calculation=metadata,measured_presentation=measurements,visual_review={'status':'pending'},whole_figure_frozen=False)
    write(OUT/'candidate-evidence.json',report)
    print(OUT/'candidate.html',flush=True)

def publish(baseline):
    report=json.loads((OUT/'candidate-evidence.json').read_text())
    if report['visual_review']['status']!='reviewed':raise ValueError('Inspect the exact saved render before publishing')
    svg_bytes=(OUT/'candidate.svg').read_bytes();review_bytes=(OUT/'candidate.html').read_bytes()
    if sha(svg_bytes)!=report['style_comparison']['candidate_svg_sha256'] or sha(svg_bytes)!=report['visual_review']['svg_sha256']:
        raise ValueError('Preserve edited candidate; its recorded verification is stale')
    if svg_bytes not in review_bytes:raise ValueError('Review does not contain the matching editable SVG')
    snapshot=Snapshot([]);snapshot.hashes=json.loads(baseline.read_text())
    for name in ('candidate.svg','candidate.html','candidate-evidence.json','legend.md','README.md'):
        path=OUT/name;snapshot.hashes[str(path.resolve())]=sha256_file(path)
    # Retain the pre-edit ticket; bind only this task's deliberate code changes.
    for path in (Path(__file__), SOURCE_MODULE):snapshot.hashes[str(path.resolve())]=sha256_file(path)
    snapshot.hashes[str(SELECTION.resolve())]=report['pipeline_reference']['sha256']
    snapshot.hashes[str(Path(report['style_reference']['path']).resolve())]=report['style_reference']['sha256']
    snapshot.hashes[str(Path(report['reference_registry']['path']).resolve())]=report['reference_registry']['sha256']
    if sha256_file(Path(__file__))!=report['generator']['sha256']:raise ValueError('Renderer code changed after verification')
    snapshot.check()
    source_hashes={}
    for record in report['calculation']['records']:source_hashes.update(record['sources'])
    for row in report['calculation']['cohort_inputs']:source_hashes[row['path']]=row['sha256']
    reference=report['example_reproduction']['reference_data'];source_hashes[reference['path']]=reference['sha256']
    for record in report['renderer_dependencies']:source_hashes[record['path']]=record['sha256']
    for record in report['font_resources']:source_hashes[record['path']]=record['sha256']
    for key in ('calculation','fish_bins','pooled_bins'):
        record=report['calculation'][key];source_hashes[record['path']]=record['sha256']
    # Preserve the guard's publication ordering while streaming large source checks.
    regular_snapshot=snapshot
    class PublicationSnapshot:
        hashes=regular_snapshot.hashes
        verified_stats=None
        def check(self, ignore=()):
            regular_snapshot.check(ignore=ignore)
            current={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns,Path(p).stat().st_ctime_ns) for p in source_hashes}
            if self.verified_stats is None:
                StreamingSnapshot(source_hashes).check()
                after={p:(Path(p).stat().st_size,Path(p).stat().st_mtime_ns,Path(p).stat().st_ctime_ns) for p in source_hashes}
                if current!=after:raise ValueError('A source changed during final authentication')
                self.verified_stats=current
            elif current!=self.verified_stats:raise ValueError('A source changed under the publication lock')
    guarded_snapshot=PublicationSnapshot()
    review=OUT.parent/'figure-02-a-pooled.html';svg=review.with_suffix('.svg')
    state=dict(schema_version=1,figure_id='fig2',panel_ids=['A'],default_display='draft',default_display_format='html-svg-and-legend',active_edit_scope=['A'],assembly_integration='Scoped A review and editable SVG; the earlier whole-figure review predates this draft',whole_figure_frozen=False,scientific_changes={'A':'Direct original-frame log medians in 1 s bins replace earlier 0.5 s signed bout means; baseline-bin median replaces the frame-level baseline; original cadence reconstruction and detector/metric pipeline are applied to each fish before equal-fish pooling.'},draft=dict(assembly_scale={'final_width_mm':report['physical_width_mm'],'final_height_mm':78,'reference_element_scale':1},status='unfrozen-author-requested-pooled-pipeline',working_svg={'path':str(svg),'sha256':sha((OUT/'candidate.svg').read_bytes())},review={'path':str(review),'sha256':sha((OUT/'candidate.html').read_bytes())},legend={'path':str(LEGEND),'sha256':sha((OUT/'legend.md').read_bytes()),'panel_ids':['A'],'extraction':'A paragraph and required shared definitions'},semantic_registry={'path':str(EVIDENCE),'sha256':sha((OUT/'candidate-evidence.json').read_bytes())},consistency_requests=[dict(id='fig2-a-fgh-pipeline-and-formatting',author_direction='Apply the Figure 1 F/G/H pipeline adapted to pooled data and exact formatting on every element; author subsequently scoped implementation to Delay now.',reference=artifact(SELECTION),scope=['A'],properties=['scientific pipeline','all named element styles'],status='verified',evidence=str(EVIDENCE))]))
    snapshot.hashes.setdefault(str(STATE.resolve()),None)
    snapshot.hashes.setdefault(str(EVIDENCE.resolve()),None)
    STATE.parent.mkdir(exist_ok=True);EVIDENCE.parent.mkdir(exist_ok=True)
    artifacts=[(svg,svg_bytes),(review,review_bytes)]
    updates=[(EVIDENCE,(OUT/'candidate-evidence.json').read_bytes()),(LEGEND,(OUT/'legend.md').read_bytes()),(README,(OUT/'README.md').read_bytes())]
    if STATE.is_file():
        prior=json.loads(STATE.read_text())
        for name,target in [('working_svg',svg),('review',review)]:
            bound=prior['draft'][name]
            if Path(bound['path'])!=target or sha256_file(target)!=bound['sha256']:
                raise ValueError('Preserve the manually changed working pair; reconcile before updating')
            if str(target.resolve()) not in snapshot.hashes:
                raise ValueError('Capture the active working pair before edits')
        updates=artifacts+updates;artifacts=[]
    publish_state(STATE,state,guarded_snapshot,artifacts=artifacts,evidence_updates=updates)
    print(review)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=Path('/Volumes/JOAQUIM/Digested Data/allDelay-full-v1'));parser.add_argument('--prepare',action='store_true');parser.add_argument('--render',action='store_true');parser.add_argument('--publish',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare(args.project)
    if args.render:render()
    if args.publish:publish(args.publish)
if __name__=='__main__':main()
