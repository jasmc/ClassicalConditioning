"""Assay-isolated scientific drafts; never publishes shared figure selections.

Reuses authenticated G windows and the supported Figure 4 recording reader.
Condition profiles are independent of the unresolved Gate L classifier.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from classical_conditioning.analysis.figure4 import _read_recording, profile_groups, verify_expected_us
from classical_conditioning.analysis.metric_comparison import _verify_temporal_profiles
from classical_conditioning.analysis.movement_state import resolve_candidate_metric_source
from classical_conditioning.figures.signed_bout_heatmap import calculate_fish_heatmaps
from classical_conditioning.figures.example_traces import METRIC_COLUMNS
from classical_conditioning.analysis.trial_outcomes import parse_recording_identity
from classical_conditioning.artifacts import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(r'F:\Digested Data\all3sTrace-full-v1')
BASE = Path(r'J:\ClassicalConditioning Outputs\current\reviews\figure-02-3strace')
OUT = Path(r'J:\ClassicalConditioning Outputs\current\reviews\figures-03-04-3strace-20261020')
METRIC = 'legacy_distal_angular_speed'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(obj, f, indent=2, default=str)
def bound(p): return {'path': str(p), 'sha256': sha256_file(Path(p))}
def verify(item):
    assert sha256_file(Path(item['path'])) == item['sha256'], item['path']

def read_processed(fish):
    """Authenticate consumed processed bytes; no tracking/raw byte re-audit.

    Uses the same supported calculator and coverage join as _read_recording.
    Numerical equivalence is checked against an unmodified reader result.
    """
    source=resolve_candidate_metric_source(metric_recipe='tail-candidate-corrected')
    verified=_verify_temporal_profiles(PROJECT,fish,source)
    directory=PROJECT/'Processed data'/fish
    metric=directory/source.metrics_name; movement=directory/source.movement_artifact_name
    protocol_path=directory/'stimulus_events.parquet'
    sm_path=PROJECT/'Metadata'/f'{fish}_source_manifest.json';sm=read(sm_path)
    sources=[bound(p) for p in [metric,movement,protocol_path,verified.path,verified.summary_path,verified.marker_path,sm_path]]
    digests={x['path']:x['sha256'] for x in sources}
    assert sm['recording_id']==fish
    assert sm['artifacts']['protocol']['sha256']==digests[str(protocol_path)]
    for summary_name,marker_suffix,artifact,hash_key,recipe in [
        (source.metric_summary_name,source.metric_marker_suffix,metric,'metrics_sha256',source.metric_recipe),
        (source.movement_summary_name,source.movement_marker_suffix,movement,'movement_sha256',source.movement_recipe)]:
        sp=PROJECT/'Quality checks'/fish/summary_name;mp=PROJECT/'Metadata'/f'{fish}_{marker_suffix}'
        summary=read(sp);marker=read(mp)
        assert marker['status']=='complete' and marker['recipe']==recipe
        assert marker[hash_key]==digests[str(artifact)]==summary['artifact']['sha256']
        assert marker['summary_sha256']==sha256_file(sp)
        sources.extend([bound(sp),bound(mp)])
    temporal=read(verified.summary_path)
    for name,path in [('candidate_frames',metric),('movement_state',movement),('protocol',protocol_path)]:
        assert temporal['inputs'][name]['sha256']==digests[str(path)]
    metrics=pd.read_parquet(metric,columns=['FrameID','AbsoluteTime',METRIC_COLUMNS[METRIC]])
    states=pd.read_parquet(movement,columns=['FrameID','AbsoluteTime','valid','moving','bout_id'])
    protocol=pd.read_parquet(protocol_path)
    cycles=protocol[protocol.Type.astype(str).eq('Cycle')].sort_values('Beg',kind='stable').reset_index(drop=True)
    profiles=pd.read_parquet(verified.path)
    profiles=profiles[profiles['Trial type'].eq('CS')&profiles['Metric ID'].eq(METRIC)&profiles['Time bin center (s)'].between(-20,20,inclusive='neither')]
    signal=calculate_fish_heatmaps(metrics,states,cycles,recording_id=fish,metric_ids=(METRIC,))
    keys=['Recording ID','Trial number','Time bin center (s)','Metric ID']
    joined=signal.merge(profiles[keys+['Valid expected fraction','Movement probability']],on=keys,validate='one_to_one',how='left',indicator=True)
    assert joined['_merge'].eq('both').all()
    covered=joined['Valid expected fraction']>=.9
    bins=pd.DataFrame({'recording_id':fish,'trial_number':joined['Trial number'].astype(int),
        'time_s':joined['Time bin center (s)'].astype(float),
        'signed_log_vigor':pd.to_numeric(joined['Signed log vigor']).where(covered),
        'movement_probability':pd.to_numeric(joined['Movement probability']).where(covered),
        'coverage':joined['Valid expected fraction'].astype(float)})
    return bins,protocol,profiles,sources

def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / 'analysis.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        plan_path = BASE/'matched-delay-statistics/plan.json'
        windows_path = BASE/'matched-delay-statistics/windows.parquet'
        plan = read(plan_path)
        selected = {x['recording_id']: x['condition'] for x in plan['selected']}
        assert len(selected) == len(plan['selected'])
        assert not set(selected).intersection(plan['excluded_recordings'])
        dependencies = [plan_path, windows_path, BASE/'matched-delay-statistics/completion.json',
            ROOT/'docs/paper/figures/figure-03/current-state.json',
            ROOT/'docs/paper/figures/figure-04/README.md',
            ROOT/'docs/running-analysis/figure3-main-paper-draft.md',
            ROOT/'Plans/04_FIGURE_3_LEARNER_ANALYSIS.md', ROOT/'Plans/05_FIGURE_4_CR_PROFILES.md',
            ROOT/'src/classical_conditioning/analysis/figure4.py',
            ROOT/'src/classical_conditioning/figures/signed_bout_heatmap.py',
            ROOT/'configs/paper-figures/nature-main.json', ROOT/'configs/paper-figures/layout-frames.json', Path(__file__)]
        if not (OUT/'start-ticket.json').exists():
            write(OUT/'start-ticket.json', {'inputs':[bound(p) for p in dependencies],
                'status':'scientific draft; no accepted Figure 3/4 selection',
                'cohort_status':'author-selected source roster; not newly approved primary technical cohort',
                'condition_counts':pd.Series(selected).value_counts().to_dict(),
                'excluded':plan['excluded_recordings'], 'pid':os.getpid()})
        for item in read(OUT/'start-ticket.json')['inputs']:
            if Path(item['path']) != Path(__file__): verify(item)
        execution_code=bound(Path(__file__))
        write(OUT/f'execution-code-{time.time_ns()}.json',execution_code)
        windows = pd.read_parquet(windows_path)
        assert set(windows.fish_id) == set(selected)
        assert not windows.duplicated(['fish_id','trial_number']).any()
        assert windows.condition_id.eq(windows.fish_id.map(selected)).all()
        records=[]; rows=[]; parts=[]
        (OUT/'fish').mkdir(exist_ok=True)
        first=next(iter(selected))
        if not (OUT/'reader-equivalence.json').exists() and (OUT/'fish'/f'{first}.parquet').exists():
            check,_,_,_=read_processed(first)
            original=pd.read_parquet(OUT/'fish'/f'{first}.parquet')
            pd.testing.assert_frame_equal(check,original[check.columns],check_dtype=True)
            write(OUT/'reader-equivalence.json',{'fish_id':first,'rows':len(check),
                'result':'exact dataframe equality to supported _read_recording result',
                'original':bound(OUT/'fish'/f'{first}.parquet'),'code':execution_code,
                'authentication_boundary':'consumed processed bytes authenticated; camera/tracking raw dependency bytes not rehashed'})
        for i, (fish, condition) in enumerate(selected.items()):
            rec_path=BASE/'fish'/f'{fish}.json'; rec=read(rec_path)
            assert rec['condition']==condition and rec['recording_id']==fish
            for item in rec['outputs']: verify(item)
            local=pd.read_parquet(BASE/'fish'/f'{fish}-windows.parquet')
            current=windows[windows.fish_id.eq(fish)].drop(columns=['fish_id','condition_id']).reset_index(drop=True)
            pd.testing.assert_frame_equal(current, local[current.columns].reset_index(drop=True), check_dtype=False)
            source_manifest=PROJECT/'Metadata'/f'{fish}_source_manifest.json'
            sm=read(source_manifest)
            identity=parse_recording_identity(sm['recording_name'],experiment_name='all3sTrace')
            assert sm['recording_id']==fish and identity['recording_id']==fish and identity['condition_id']==condition
            summary_path=Path(next(x['summary_path'] for x in plan['selected'] if x['recording_id']==fish))
            summary=read(summary_path)
            assert summary['identity']['condition_id']==condition
            assert summary['identity']['experiment_id']=='all3sTrace'
            assert summary['config']['baseline_window_s']==[-15.0,0.0]
            assert summary['config']['response_window_s']==[0.0,13.0]
            g=windows[windows.fish_id.eq(fish)]
            pre=g[g.trial_number.between(5,14)].legacy_log_median_difference
            late=g[g.trial_number.between(40,64)].legacy_log_median_difference
            rows.append({'fish_id':fish,'condition_id':condition,'pre_median':pre.median(),
                'late_median':late.median(),'acquisition_contrast':pre.median()-late.median(),
                'pre_n':int(pre.count()),'late_n':int(late.count()),'observed_trial_n':int(g.legacy_log_median_difference.count()),
                'eligibility_reason':'' if pre.count() and late.count() else 'no finite outcome in at least one epoch',
                'scientific_status':'retrospective descriptive; no binary label; minimum coverage unapproved'})
            cache=OUT/'fish'/f'{fish}.parquet'; evidence=cache.with_suffix('.json')
            if cache.exists():
                ev=read(evidence); verify(ev['output'])
                for item in ev['sources']: verify(item)
                bins=pd.read_parquet(cache)
            else:
                bins,protocol,profiles,sources=read_processed(fish)
                for src in rec['window_sources']:
                    match=next(x for x in sources if x['path']==src['path'])
                    assert match['sha256']==src['sha256']
                expected,count=verify_expected_us(protocol,'all3sTrace') if condition=='trace' else (None,0)
                control_overlaps=[]
                if condition=='control':
                    cycles=protocol[protocol.Type.astype(str).eq('Cycle')].sort_values('Beg')
                    us=protocol.loc[protocol.Type.astype(str).eq('Reinforcer'),'Beg'].to_numpy(float)
                    for trial_number,onset in enumerate(cycles.iloc[4:94].Beg,start=5):
                        latency=(us-float(onset))/1000
                        control_overlaps.extend({'trial_number':trial_number,'us_latency_s':float(t)}
                            for t in latency[(latency>=-20)&(latency<20)])
                # Existing temporal Signed log vigor has a different baseline; use
                # the supported reader's recomputation, never that saved column.
                saved=profiles
                scaled=saved[['Trial number','Time bin center (s)','Scaled total activity']].rename(columns={
                    'Trial number':'trial_number','Time bin center (s)':'time_s','Scaled total activity':'scaled_activity'})
                bins=bins.merge(scaled,on=['trial_number','time_s'],validate='one_to_one')
                bins['scaled_activity']=bins.scaled_activity.where(bins.coverage>=.9)
                bins['fish_id']=fish; bins['condition_id']=condition
                bins.to_parquet(cache,index=False)
                ev={'sources':sources+[bound(rec_path),bound(source_manifest),bound(summary_path)],
                    'output':bound(cache),'expected_us_s':expected,'paired_training_trials':count,
                    'control_unpaired_us_in_display_windows':control_overlaps,
                    'low_coverage_bins':int((bins.coverage<.9).sum()),
                    'undefined_signed_bins':int(bins.signed_log_vigor.isna().sum()),
                    'rows':len(bins)}
                write(evidence,ev)
            records.append({'fish_id':fish,'condition_id':condition,**ev})
            parts.append(bins)
            if (i+1)%5==0: print(f'Authenticated profiles {i+1}/{len(selected)}',flush=True)
        scores=pd.DataFrame(rows)
        scores.to_csv(OUT/'acquisition-scores.csv',index=False)
        windows.to_parquet(OUT/'observed-g-trial-outcomes.parquet',index=False)
        trial=pd.concat(parts,ignore_index=True)
        assert not trial.duplicated(['fish_id','trial_number','time_s']).any()
        trial.to_parquet(OUT/'trial-bins.parquet',index=False)
        assignments=pd.DataFrame([{'trial_number':t,**{k:g[k] for k in ['group_type','group_name','group_order']}}
            for g in profile_groups('all3sTrace') for t in g['trials']])
        expanded=trial.merge(assignments,on='trial_number',validate='many_to_many')
        keys=['fish_id','condition_id','group_type','group_name','group_order','time_s']
        signals=['signed_log_vigor','movement_probability','scaled_activity']
        agg={f'{s}_{kind}':(s,kind) for s in signals for kind in ['median','count']}
        fishbins=expanded.groupby(keys,sort=False).agg(**agg).reset_index()
        fishbins.to_parquet(OUT/'fish-bins.parquet',index=False)
        groups=[]
        for key,g in fishbins.groupby(keys[1:],sort=False):
            r=dict(zip(keys[1:],key));r['roster_fish']=g.fish_id.nunique()
            for s in signals:
                x=g[f'{s}_median'].dropna()
                for q,n in [(.25,'q25'),(.5,'median'),(.75,'q75')]:r[f'{s}_{n}']=x.quantile(q)
                r[f'{s}_fish']=len(x);r[f'{s}_trials']=int(g[f'{s}_count'].sum())
            groups.append(r)
        pd.DataFrame(groups).to_parquet(OUT/'group-bins.parquet',index=False)
        scores.groupby('condition_id').agg(fish=('fish_id','count'),estimable=('acquisition_contrast','count'),
            median=('acquisition_contrast','median'),q25=('acquisition_contrast',lambda x:x.quantile(.25)),
            q75=('acquisition_contrast',lambda x:x.quantile(.75)),pre_min=('pre_n','min'),late_min=('late_n','min')).to_csv(OUT/'condition-score-summary.csv')
        write(OUT/'source-authentication.json',records)
        write(OUT/'analysis.json',{'status':'provisional scientific draft; not accepted/frozen',
            'metric':METRIC,'roster_sha256':hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
            'condition_counts':pd.Series(selected).value_counts().to_dict(),
            'groups':profile_groups('all3sTrace'), 'signed':'supported Figure 4 reader: 0.5 s bins; baseline [-20,0); positive moving-frame/bout signal',
            'scaled_activity':'separate supported saved Scaled total activity; not signed signal',
            'aggregation':'trial medians within fish, equal-fish condition median and IQR',
            'minimum_bin_coverage':.9,'gate_L':'unresolved; no categorical strata or group inference',
            'coverage_rule':'one finite outcome per epoch is mathematical estimability only; no scientific minimum approved',
            'validation_mode':'retrospective descriptive; no independent held-out claim',
            'excluded_timing_flagged':plan['excluded_recordings'],
            'remaining':['technical primary cohort approval','Gate L representation and validation','categorical Figure 4 integration','full Nature exact-render production checks'],
            'reused_illustration_review':str(BASE.parent/'figure-01-trace3-learner-selection-20261019/selection-report.json')})
        for item in read(OUT/'start-ticket.json')['inputs']:
            if Path(item['path']) != Path(__file__):verify(item)
        verify(execution_code)
        write(OUT/'complete.json',{'analysis':bound(OUT/'analysis.json'),
            'outputs':[bound(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['analysis.lock','complete.json']],
            'fish':len(records),'trial_bins':len(trial),'score_estimable':int(scores.acquisition_contrast.count())})
        print(pd.read_csv(OUT/'condition-score-summary.csv').to_string(index=False),flush=True)
    finally: lock.unlink()

if __name__=='__main__':
    analyze()
