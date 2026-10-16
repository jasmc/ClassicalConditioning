"""Build/verify/publish a guarded exploratory Figure 3 atlas on external storage."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import numpy as np
import pandas as pd
from classical_conditioning.analysis.review.individual_atlas import prepare,classify,sha,save_json,VARIANTS
from classical_conditioning.figure_workspace import Snapshot,publish_state,state_lock
from classical_conditioning.figures.individual_atlas import render,results_for,decision,feature_value,score_value,COLOURS,HEATMAP_LIMIT

STATE=ROOT/'docs/paper/figures/figure-03/exploratory-state.json'
G_ROOT=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-G-independent-review/20261014-analysis-02')


def source_paths():
    return [Path(__file__),ROOT/'src/classical_conditioning/analysis/review/individual_atlas.py',
        ROOT/'src/classical_conditioning/figures/individual_atlas.py',
        ROOT/'src/classical_conditioning/analysis/legacy_learner_worker.py',
        ROOT/'src/classical_conditioning/figures/layout_frames.py',
        ROOT/'src/classical_conditioning/figures/layout_frames_browser.js',
        ROOT/'configs/paper-figures/layout-frames.json',*sorted((ROOT/'legacy/scripts').glob('6_LearnersQuantification*.py'))]


def snapshot_sources(out,stage):
    dest=out/'source-snapshots'/stage
    dest.mkdir(parents=True,exist_ok=False)
    records=[]
    for i,p in enumerate(source_paths()):
        copy=dest/f'{i:02d}-{p.name}';shutil.copyfile(p,copy)
        records.append({'path':str(p),'sha256':sha(p),'snapshot':str(copy.relative_to(out))})
    save_json(dest/'sources.json',records)
    return records


def classification_manifest(out):
    """Retain every fish for every method, with explicit pre-fit versus model reasons."""
    inventory=pd.read_csv(out/'fish-inventory.csv')
    trials=pd.read_parquet(out/'trial-data.parquet')
    results=results_for(out);rows=[]
    for fish in inventory.itertuples(index=False):
        data=trials.loc[trials.fish_id.eq(fish.fish_id)]
        for variant in VARIANTS:
            state,result=decision(results,variant,fish.condition_id,fish.fish_id)
            good=np.isfinite(data.baseline_bout_mean)&np.isfinite(data.response_bout_mean)&np.isfinite(data.classifier_ratio)
            if variant=='legacy-improved':good&=(data.baseline_bout_mean>0)&(data.response_bout_mean>0)
            if variant=='legacy-wip':good&=data.classifier_ratio>0
            middle_start=60 if variant=='legacy-nominal' else 40
            reason=[];counts={}
            for name,lo,hi in [('pre',5,14),('middle',middle_start,69),('late',85,94)]:
                counts[name]=int((good&data.trial_number.between(lo,hi)).sum())
                if counts[name]<6:reason.append(f'{name}: fewer than 6 usable trials')
                for start in range(lo,hi+1,5):
                    n=int((good&data.trial_number.between(start,start+4)).sum())
                    if n<3:reason.append(f'CS{start}-{start+4}: {n}/5 usable (<3)')
            if result is None and not reason:reason.append('method failed or model feature unavailable; see classifier log')
            row={'experiment_id':'allDelay','condition_id':fish.condition_id,'fish_id':fish.fish_id,
                'variant':variant,'decision':state,'eligibility_reason':'; '.join(reason) or 'classified',
                'pre_trials':counts['pre'],'middle_trials':counts['middle'],'late_trials':counts['late'],
                'middle_start':middle_start,'acquisition':feature_value(result,'acquisition'),
                'recovery':feature_value(result,'extinction'),'score':score_value(result),
                'threshold':float(result.decision_threshold) if result is not None else np.nan,
                'protocol_review_flag':fish.protocol_review_flag,
                'metric_id':'legacy_distal_angular_speed','vigor_sample_policy':'valid-bout-frames-only-v1',
                'scientific_status':'exploratory same-data legacy-rule characterization'}
            rows.append(row)
    table=pd.DataFrame(rows)
    if len(table)!=len(inventory)*4 or table.duplicated(['condition_id','fish_id','variant']).any():
        raise ValueError('Classification inventory cardinality failure')
    table.to_csv(out/'classification-manifest.csv',index=False)
    table.to_parquet(out/'classification-manifest.parquet',index=False)
    return table


def check_outputs(out):
    marker=out/'render-outputs.json'
    if marker.exists():
        for r in json.loads(marker.read_text()):
            if sha(out/r['path'])!=r['sha256']:
                raise ValueError('Manual output edit detected; preserve and reconcile '+r['path'])


def render_guarded(out):
    with state_lock(out/'render'):
        check_outputs(out)
        classification_manifest(out)
        inputs=[out/'trial-data.parquet',out/'heatmaps.npz',out/'fish-inventory.csv',
            *sorted((out/'classifiers').glob('*/*.parquet')),*source_paths()]
        if (out/'reference-style.json').exists():
            reference=json.loads((out/'reference-style.json').read_text())
            if reference['condition_colors']!=COLOURS or reference['heatmap_limits']!=[-HEATMAP_LIMIT,HEATMAP_LIMIT]:
                raise ValueError('Figure 3 style differs from authenticated Figure 2 reference')
            inputs.append(out/'reference-style.json')
            for record in reference['references']:
                if sha(record['path'])!=record['sha256']:
                    raise ValueError('Changed Figure 2 reference: '+record['path'])
                inputs.append(Path(record['path']))
        snapshot={str(p):sha(p) for p in inputs}
        with tempfile.TemporaryDirectory(dir=out,prefix='.render-') as temp:
            temp=Path(temp)
            for name in ['trial-data.parquet','heatmaps.npz','fish-inventory.csv','classifiers','classification-manifest.csv']:
                (temp/name).symlink_to(out/name)
            render(temp)
            for p,h in snapshot.items():
                if sha(p)!=h:raise ValueError('Source/input changed while rendering: '+p)
            check_outputs(out)
            names=['overview.svg','review.html','legend.md','selected-examples.json','card-svg-data.json','g-logmedian-outcomes.csv']
            # Keep a previous matching pair recoverable when refining this unpublished review.
            if (out/'review.html').exists():
                history=out/'history';history.mkdir(exist_ok=True)
                key=sha(out/'review.html')[:12];prior=history/key;prior.mkdir(exist_ok=True)
                for name in names:
                    if (out/name).exists() and not (prior/name).exists():shutil.copyfile(out/name,prior/name)
            for name in names:os.replace(temp/name,out/name)
        save_json(out/'render-outputs.json',[{'path':n,'sha256':sha(out/n)} for n in names])
        save_json(out/'render-inputs.json',[{'path':p,'sha256':h} for p,h in snapshot.items()])
        classification_manifest(out)


def verify(out,deep=False):
    check_outputs(out)
    checks=json.loads((out/'data-checks.json').read_text())
    manifest=pd.read_csv(out/'classification-manifest.csv')
    if checks['fish']!=57 or len(manifest)!=228:raise ValueError('Incomplete atlas inventory')
    if deep:
        for i,r in enumerate(json.loads((out/'input-lineage.json').read_text()),1):
            if sha(r['path'])!=r['sha256']:raise ValueError('Changed bound input '+r['path'])
            if i%30==0:print(f'Final source authentication {i}',flush=True)
    for r in json.loads((out/'render-inputs.json').read_text()):
        if sha(r['path'])!=r['sha256']:raise ValueError('Changed rendering dependency '+r['path'])
    return {'fish':57,'fish_method_rows':228,'scheduled_trial_rows':checks['scheduled_trials'],
            'deep_source_authentication':deep,'rendered_outputs_verified':True}


def publish(out):
    """Verified artifacts/evidence precede this separate exploratory pointer."""
    checked=verify(out,deep=True)
    evidence=out/'layout-evidence.json'
    if not evidence.is_file():raise ValueError('Exact-render layout evidence is required')
    layout=json.loads(evidence.read_text())
    if len(layout)!=58 or any(r['status']!='passed' for r in layout.values()):
        raise ValueError('Resolve all overview/card layout evidence before publishing')
    from classical_conditioning.figures.layout_frames import assess
    vectors={'overview':(out/'overview.svg').read_bytes(),**{fish:base64.b64decode(data)
        for fish,data in json.loads((out/'card-svg-data.json').read_text()).items()}}
    if set(vectors)!=set(layout):raise ValueError('Incomplete exact-SVG evidence inventory')
    for name,data in vectors.items():
        actual=assess(data,layout[name]['browser'])
        if actual['status']!='passed' or actual['artifact_sha256']!=layout[name]['artifact_sha256']:
            raise ValueError('Stale/invalid exact-render evidence: '+name)
    ticket=json.loads((out/'start-ticket.json').read_text())
    bound=[STATE,out/'start-ticket.json',out/'render-inputs.json',out/'render-outputs.json',evidence,
           *[Path(r['path']) for r in json.loads((out/'render-inputs.json').read_text())],
           *[out/r['path'] for r in json.loads((out/'render-outputs.json').read_text())]]
    for name in ['draft-registry.json','consistency-evidence.json','visual-review.json']:
        if (out/name).exists():bound.append(out/name)
    snapshot=Snapshot(bound)
    for r in json.loads((out/'render-inputs.json').read_text()):
        if snapshot.hashes[str(Path(r['path']).resolve())]!=r['sha256']:
            raise ValueError('Changed presentation input before publication: '+r['path'])
    for r in json.loads((out/'render-outputs.json').read_text()):
        if snapshot.hashes[str((out/r['path']).resolve())]!=r['sha256']:
            raise ValueError('Manual output edit before publication: '+r['path'])
    if snapshot.hashes[str(STATE.resolve())]!=ticket['state_before']:
        raise ValueError('Exploratory state changed since starting ticket')
    # Current pointers of unrelated work may advance; record, never overwrite them.
    protected={p:{'before':h,'after':sha(p),'changed_outside_this_task':sha(p)!=h}
               for p,h in ticket['protected'].items()}
    save_json(out/'publication-checks.json',{**checked,'protected_records':protected,
        'scope':'Figure 3 exploratory pointer only; no paper selection or freeze'})
    artifacts=[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*'))
        if p.is_file() and 'history' not in p.parts and not p.name.endswith('.lock')
        and p.name not in ['artifact-manifest.json','run.log'] and not p.name.startswith('._')]
    save_json(out/'artifact-manifest.json',{'schema':'figure3-exploratory-atlas/1.0','artifacts':artifacts,
        'scientific_status':'exploratory; no selected learner method'})
    def record(name):return {'path':str(out/name),'sha256':sha(out/name)}
    state={'scope':'figure-03 individual-fish exploratory review','status':'exploratory',
        'date':'2026-10-15','canonical_classifier':None,'metric_id':'legacy_distal_angular_speed',
        'default_display':'review','review':record('review.html'),'working_svg':record('overview.svg'),
        'legend':record('legend.md'),'manifest':record('artifact-manifest.json'),
        'layout_evidence':record('layout-evidence.json'),'fish':57,'conditions':{'control':28,'delay':29},
        'limitations':['Delay/control only: authenticated processed Trace inputs unavailable in checked mounted roots',
            'All legacy default labels use acquisition and recovery, including late Test; descriptive, not held-out validation',
            'Legacy fit warnings and numerical fallbacks retained; individual uncertainty and method remain unapproved']}
    if (out/'draft-registry.json').exists():
        state['semantic_registry']=record('draft-registry.json')
        state['consistency_requests']=[{'id':'fig3-figure2-heatmap-scale-and-G-colors-20261015',
            'scope':'All 57 full fish cards and four illustrative cards',
            'author_direction':'Use Figure 2 analysis and plotting scales/colors (0.25)',
            'properties':['native managua_r at -0.25 to +0.25; undefined intensity black',
                          'selected Figure 2 G condition colors',
                          'unchanged source heatmap values and original classifier decisions'],
            'evidence':record('consistency-evidence.json'),'status':'verified for listed properties'}]
    # Recheck the final manifest and all bound presentation data under publication lock.
    snapshot.hashes[str((out/'artifact-manifest.json').resolve())]=sha(out/'artifact-manifest.json')
    publish_state(STATE,state,snapshot)
    print(STATE,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','classify','render','verify','publish'])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--g-root',type=Path,default=G_ROOT)
    args=parser.parse_args();out=args.output_dir.resolve()
    volume=Path('/Volumes/JOAQUIM')
    if not os.path.ismount(volume) or not out.is_relative_to(volume) or not out.is_dir():
        raise ValueError('Use an existing explicit directory on mounted JOAQUIM')
    if args.stage=='prepare':prepare(args.g_root,out)
    elif args.stage=='classify':classify(out)
    elif args.stage=='render':render_guarded(out)
    elif args.stage=='verify':print(json.dumps(verify(out,deep=True)))
    elif args.stage=='publish':publish(out)


if __name__=='__main__':main()
