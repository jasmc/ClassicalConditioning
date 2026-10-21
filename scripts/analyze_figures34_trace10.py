"""Isolated, descriptive 10sTrace Figures 3/4 analysis; never assigns learner labels."""
from pathlib import Path
import gc, html, json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot, publish_state
from classical_conditioning.analysis.figure4 import _read_recording, profile_groups, verify_expected_us
from classical_conditioning.analysis.metric_comparison import _verify_temporal_profiles
from classical_conditioning.analysis.movement_state import resolve_candidate_metric_source

OUT=Path('J:/ClassicalConditioning Outputs/current/reviews/figures-03-04-10strace-20261020')
DATA=OUT.parent/'figure-02-10strace-20261020'
OLD=OUT.parent/'figure-01-trace10-learner-selection-20261020'
PROJECT=Path('F:/Digested Data/all10sTrace-full-v1')
METRIC='legacy_distal_angular_speed'
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def ref(p): return {'path':str(Path(p).resolve()),'sha256':sha256_file(p)}
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,allow_nan=False)
def check(r):
    assert sha256_file(Path(r['path']))==r['sha256'], r['path']

def prepare():
    OUT.mkdir(exist_ok=False)
    (OUT/'fish').mkdir()
    roster=load(DATA/'common-roster.json'); plan=load(DATA/'plan.json')
    assert roster['status']=='author-approved'
    assert len(roster['selected'])==61 and len(plan['cohort'])==63
    assert pd.Series([x['condition'] for x in roster['selected']]).value_counts().to_dict()==roster['counts']
    assert len({x['recording_id'] for x in plan['cohort']})==63
    assert {x['recording_id'] for x in plan['cohort']}=={x['recording_id'] for x in roster['selected']+roster['excluded']}
    for r in roster['selected']:
        assert r['shared_control_identity']==(r['condition']=='control')
    calc=load(DATA/'calculation.json')
    for r in calc['outputs']+calc['fish_records']: check(r)
    for name in ('plan.json','calculation.json','common-roster.json','final-spacing/scientific-evidence.json'):
        assert (DATA/name).is_file()
    completion=load(OLD/'completion-final.json'); check(completion['full_table'])
    state=load(ROOT/'docs/paper/figures/figure-01/trace10-learner-selection/current-state.json')
    for name in ('score_table','working_svg','review','legend','method'): check(state['draft'][name])
    bindings=[Path(__file__),DATA/'common-roster.json',DATA/'plan.json',DATA/'calculation.json',DATA/'windows-observed.parquet',DATA/'final-spacing/scientific-evidence.json',OLD/'completion-final.json',Path(completion['full_table']['path']),ROOT/'docs/paper/figures/figure-03/current-state.json',ROOT/'docs/paper/figures/figure-03/exploratory-state.json',ROOT/'docs/paper/figures/figure-04/README.md',ROOT/'Plans/04_FIGURE_3_LEARNER_ANALYSIS.md',ROOT/'Plans/05_FIGURE_4_CR_PROFILES.md',ROOT/'configs/paper-figures/layout-frames.json']
    save(OUT/'starting-ticket.json',Snapshot(bindings).hashes)
    save(OUT/'source-roster.json',roster)
    windows=pd.read_parquet(DATA/'windows-observed.parquet')
    assert not windows.duplicated(['fish_id','trial_number']).any()
    assert windows.fish_id.nunique()==63
    np.testing.assert_allclose(windows.legacy_log_median_difference,windows.response_log_median-windows.baseline_log_median,equal_nan=True)
    selected={x['recording_id']:x for x in roster['selected']}
    excluded={x['recording_id']:x['reason'] for x in roster['excluded']}
    old=pd.read_csv(completion['full_table']['path']).set_index('fish_id')
    rows=[]
    for r in plan['cohort']:
        rid=r['recording_id']; g=windows.loc[windows.fish_id.eq(rid)]
        assert set(g.condition_id)=={r['condition']}
        record=load(DATA/'fish'/f'{rid}.json') if (DATA/'fish'/f'{rid}.json').exists() else next(x for x in calc['issues'] if x['recording_id']==rid)
        for x in record['outputs']: check(x)
        y=g.legacy_log_median_difference
        pre=y[g.trial_number.between(5,14)].dropna(); late=y[g.trial_number.between(40,64)].dropna()
        score=float(pre.median()-late.median()) if len(pre) and len(late) else np.nan
        np.testing.assert_allclose(score,old.loc[rid,'primary_score'],equal_nan=True)
        rows.append(dict(experiment_id='all10sTrace',fish_id=rid,condition_id=r['condition'],selected=rid in selected,exclusion_reason=excluded.get(rid,''),shared_control_identity=r['shared_control_identity'],acquisition_identity=('all3sTrace:'+rid if r['shared_control_identity'] else 'all10sTrace:'+rid),acquisition_score=score,pre_median=pre.median(),late_median=late.median(),pre_n=len(pre),late_n=len(late),finite_outcomes=int(y.notna().sum()),observed_trials=len(g),estimable=bool(np.isfinite(score)),minimum_coverage_approved=False,validation_mode='descriptive same-data; Gate L unresolved'))
    scores=pd.DataFrame(rows)
    scores.to_csv(OUT/'acquisition-scores-all63.csv',index=False)
    windows.to_parquet(OUT/'observed-trial-outcomes-all63.parquet',index=False)
    save(OUT/'analysis-spec.json',{'metric':METRIC,'roster':ref(DATA/'common-roster.json'),'scientific_status':'assay-specific scientific draft; no accepted Figures 3/4 selection','outcome':'observed response [0,20) minus baseline [-15,0) log-median eligible positive moving-frame vigor; cached Figure 2 I outcome, no median-pooling display migration','score':'median trial outcome Pre 5–14 minus late Train 40–64; at least one finite value each is mathematical estimability only','profile':'supported Figure 4 signed bout-log-vigor [-20,0) baseline and 0.5 s bins; coverage >=0.9; scaled activity separately retained from authenticated temporal artifact','grouping':'condition only; no classifier manifest or fabricated labels','aggregation':'trial medians within fish then equal-fish condition median and IQR','source_candidate_selection':'recomputed scores compared to all 63 existing scores; provisional flags not consumed','shared_controls':'32 controls are historical all3sTrace acquisition identities; do not sum as independent across assays','Gate_L':'unresolved: categories, precision, coverage rule, threshold, calibration and validation; full classifier-dependent integrated run cannot run','sources':[ref(p) for p in bindings]})
    print('Prepared',scores.groupby(['selected','condition_id']).size().to_dict(),flush=True)

def fish():
    roster=load(OUT/'source-roster.json'); source=resolve_candidate_metric_source(metric_recipe='tail-candidate-corrected')
    for r in roster['selected']:
        rid=r['recording_id']; dest=OUT/'fish'/f'{rid}.parquet'; report=OUT/'fish'/f'{rid}.json'
        if report.exists():
            check(load(report)['artifact']); continue
        assert not dest.exists(), 'Preserve uncommitted partial output '+rid
        bins,protocol,_,inputs=_read_recording(PROJECT,rid,METRIC,'tail-candidate-corrected')
        verified=_verify_temporal_profiles(PROJECT,rid,source)
        raw=pd.read_parquet(verified.path,columns=['Trial type','Trial number','Time bin center (s)','Metric ID','Scaled total activity'])
        raw=raw.loc[raw['Trial type'].eq('CS')&raw['Metric ID'].eq(METRIC)]
        raw=raw.rename(columns={'Trial number':'trial_number','Time bin center (s)':'time_s','Scaled total activity':'scaled_activity'})
        bins=bins.merge(raw[['trial_number','time_s','scaled_activity']],on=['trial_number','time_s'],validate='one_to_one')
        bins['scaled_activity']=bins.scaled_activity.where(bins.coverage.ge(.9))
        bins['fish_id']=rid; bins['condition_id']=r['condition']; bins['shared_control_identity']=r['shared_control_identity']
        assert len(bins)==7200 and not bins.duplicated(['trial_number','time_s']).any()
        timing=verify_expected_us(protocol,'all10sTrace') if r['condition']=='trace' else None
        bins.to_parquet(dest,index=False)
        metadata=[verified.summary_path,verified.marker_path]
        save(report,{'artifact':ref(dest),'inputs':inputs+[ref(x) for x in metadata],'expected_us_and_pairs':timing,'coverage_bins':int(bins.coverage.ge(.9).sum()),'signed_bins':int(bins.signed_log_vigor.notna().sum()),'movement_bins':int(bins.movement_probability.notna().sum()),'scaled_bins':int(bins.scaled_activity.notna().sum())})
        del bins,raw,protocol
        gc.collect()
        print('Completed',rid,flush=True)

def summarize():
    roster=load(OUT/'source-roster.json'); parts=[]; reports=[]
    for r in roster['selected']:
        report=load(OUT/'fish'/f"{r['recording_id']}.json"); check(report['artifact']); reports.append(report)
        parts.append(pd.read_parquet(report['artifact']['path']))
    trial=pd.concat(parts,ignore_index=True)
    assignments=pd.DataFrame({'trial_number':t,'group_name':g['group_name'],'group_type':g['group_type'],'group_order':g['group_order']} for g in profile_groups('all10sTrace') for t in g['trials'])
    assert set(assignments.loc[assignments.group_type.eq('pooled_catch'),'trial_number'])=={25,39,53,59,65}
    expanded=trial.merge(assignments,on='trial_number',validate='many_to_many')
    signals=['signed_log_vigor','movement_probability','scaled_activity']
    keys=['fish_id','condition_id','group_name','group_type','group_order','time_s']
    agg={}
    for s in signals: agg[s]=(s,'median'); agg[s+'_trials']=(s,'count')
    fishbins=expanded.groupby(keys,observed=True).agg(**agg).reset_index()
    rows=[]
    for key,g in fishbins.groupby(['condition_id','group_name','group_type','group_order','time_s'],observed=True):
        row=dict(zip(['condition_id','group_name','group_type','group_order','time_s'],key))
        row['total_fish']=len(g)
        for s in signals:
            valid=g[s].dropna(); row[s+'_median']=valid.median(); row[s+'_q25']=valid.quantile(.25); row[s+'_q75']=valid.quantile(.75); row[s+'_fish']=len(valid); row[s+'_trials']=int(g.loc[valid.index,s+'_trials'].sum())
        rows.append(row)
    group=pd.DataFrame(rows)
    trial.to_parquet(OUT/'profile-trial-bins.parquet',index=False)
    fishbins.to_parquet(OUT/'profile-fish-bins.parquet',index=False)
    group.to_parquet(OUT/'profile-condition-bins.parquet',index=False)
    group.to_csv(OUT/'profile-condition-bins.csv',index=False)
    pd.DataFrame([dict(fish_id=r['recording_id'],**{k:v for k,v in report.items() if k.endswith('_bins')}) for r,report in zip(roster['selected'],reports)]).to_csv(OUT/'profile-fish-coverage.csv',index=False)
    save(OUT/'profile-source-manifest.json',{'recordings':reports,'profile_groups':profile_groups('all10sTrace'),'full_classifier_command_ran':False,'supported_calculation':'analysis.figure4._read_recording and verify_expected_us; condition-only adapter retains the canonical aggregation order','excluded_audit':'both excluded identities and all available trial outcomes retained in acquisition-scores-all63.csv and observed-trial-outcomes-all63.parquet; excluded profiles not silently inserted in selected summaries'})
    print('Summarized',len(trial),len(fishbins),len(group),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'fish':fish,'summarize':summarize}[sys.argv[1]]()
