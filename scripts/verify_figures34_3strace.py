"""Inventory and numerical verification for the assay-scoped draft."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_figures34_3strace import ROOT,PROJECT,BASE,OUT,METRIC,read,write,bound,verify
from classical_conditioning.analysis.trial_outcomes import parse_recording_identity

def inventory():
    plan=read(BASE/'matched-delay-statistics/plan.json')
    selected={x['recording_id']:x['condition'] for x in plan['selected']}
    summaries=sorted((PROJECT/'Quality checks').glob('*/candidate-trial-outcomes-corrected_summary.json'))
    rows=[]
    for p in summaries:
        s=read(p);fish=p.parent.name
        manifest=PROJECT/'Metadata'/f'{fish}_source_manifest.json';m=read(manifest)
        identity=parse_recording_identity(m['recording_name'],experiment_name='all3sTrace')
        assert s['identity']['condition_id']==identity['condition_id']
        assert s['identity']['recording_id']==m['recording_id']==fish
        assert s['identity']['experiment_id']=='all3sTrace'
        rows.append({'fish_id':fish,'condition_id':identity['condition_id'],'raw_source_condition':m['condition_id'],'selected':fish in selected,
            'reason':'timing flagged, author excluded from common source roster' if fish in plan['excluded_recordings'] else '',
            'trial_summary_path':str(p),'trial_summary_sha256':bound(p)['sha256'],
            'source_manifest_path':str(manifest),'source_manifest_sha256':bound(manifest)['sha256']})
    frame=pd.DataFrame(rows)
    assert set(frame.fish_id)-set(selected)==set(plan['excluded_recordings'])
    assert set(selected).issubset(frame.fish_id)
    frame.to_csv(OUT/'source-inventory.csv',index=False)
    frame[~frame.selected].to_csv(OUT/'excluded-recordings.csv',index=False)
    c=read(BASE/'matched-delay-statistics/completion.json')
    item=next(x for x in c['outputs'] if Path(x['path']).name=='windows.parquet');verify(item)
    write(OUT/'inventory-verification.json',{'processed_inventory':len(frame),'selected':len(selected),
        'condition_counts':frame[frame.selected].condition_id.value_counts().to_dict(),
        'excluded':frame[~frame.selected][['fish_id','condition_id','reason']].to_dict('records'),
        'cached_outcome_completion_authentication':item,'inventory':bound(OUT/'source-inventory.csv'),
        'status':'source inventory authenticated; not C1 scientific primary-cohort approval'})

def numerical():
    completion=read(OUT/'complete.json')
    for item in completion['outputs']:verify(item)
    trial=pd.read_parquet(OUT/'trial-bins.parquet');fish=pd.read_parquet(OUT/'fish-bins.parquet')
    groups=pd.read_parquet(OUT/'group-bins.parquet');score=pd.read_csv(OUT/'acquisition-scores.csv')
    assert len(trial)==len(score)*90*80
    assert not trial.duplicated(['fish_id','trial_number','time_s']).any()
    assert trial.groupby('fish_id').size().eq(7200).all()
    signals=['signed_log_vigor','movement_probability','scaled_activity']
    assert trial.loc[trial.coverage.lt(.9),signals].isna().all().all()
    assert len(score)==score.fish_id.nunique()==trial.fish_id.nunique()
    # Directly reconstruct selected within-fish and population bins independently.
    source=trial[trial.trial_number.isin([25,39,53,59,65])]
    direct=source.groupby(['fish_id','condition_id','time_s'])[signals].median()
    target=fish[fish.group_type.eq('pooled_catch')].set_index(['fish_id','condition_id','time_s'])
    for s in signals:np.testing.assert_allclose(direct[s],target.loc[direct.index,s+'_median'],equal_nan=True)
    for (condition,time),part in direct.reset_index().groupby(['condition_id','time_s']):
        row=groups[groups.group_type.eq('pooled_catch')&groups.condition_id.eq(condition)&groups.time_s.eq(time)].iloc[0]
        for s in signals:
            np.testing.assert_allclose(part[s].median(),row[s+'_median'],equal_nan=True)
            assert part[s].count()==row[s+'_fish']
    auth=read(OUT/'source-authentication.json')
    profile_configs=[]
    for recording in auth:
        summary=read(PROJECT/'Quality checks'/recording['fish_id']/'candidate-corrected_temporal_outcomes_summary.json')
        assert summary['experiment']=='all3sTrace' and METRIC in summary['metric_ids']
        config=summary['config']
        assert config['bin_width_s']==.5 and config['scaling_baseline_end_s']==-15
        assert config['window_start_s']==-45 and config['window_end_s']==45
        assert config['scaling_clip']==[0.,1.] and config['scaling_lower_quantile']==.1 and config['scaling_upper_quantile']==.9
        profile_configs.append(config)
    overlaps=[{'fish_id':r['fish_id'],**event} for r in auth for event in r.get('control_unpaired_us_in_display_windows',[])]
    pd.DataFrame(overlaps,columns=['fish_id','trial_number','us_latency_s']).to_csv(OUT/'control-unpaired-us-overlaps.csv',index=False)
    coverage=trial.groupby('condition_id').agg(fish=('fish_id','nunique'),bins=('coverage','size'),
        low_coverage_bins=('coverage',lambda x:int(x.lt(.9).sum())),
        signed_finite=('signed_log_vigor','count'),movement_finite=('movement_probability','count'),scaled_finite=('scaled_activity','count'))
    coverage.to_csv(OUT/'coverage-summary.csv')
    write(OUT/'numerical-verification.json',{'result':'passed','fish':len(score),'trial_bins':len(trial),
        'checks':['unique and complete fish/trial/bin grid','source hashes','coverage mask never zero-filled',
            'complete score roster','direct pooled-catch trial-within-fish medians','equal-fish group median and counts'],
        'coverage':coverage.reset_index().to_dict('records'),
        'control_unpaired_us_events_in_display':len(overlaps),
        'controls_with_overlap':len(set(x['fish_id'] for x in overlaps)),
        'scaled_profile_config':profile_configs[0],
        'source_output':bound(OUT/'complete.json'),'verification_code':bound(Path(__file__))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','numerical']);args=p.parse_args()
    inventory() if args.stage=='inventory' else numerical()
