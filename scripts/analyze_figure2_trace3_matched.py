"""Apply the selected Delay test families to the author-matched Trace cohort."""
from pathlib import Path
from dataclasses import replace
import json, os, sys, traceback
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.analysis.bout_block_statistics import corrected_b_statistics
from classical_conditioning.analysis.inference.learning_onset import LearningOnsetConfig, BLOCK_ORDER, block_global_interaction_test
from classical_conditioning.figures.pooled_log_heatmap import pool_equal_fish
from render_figure2_delay_logmedian import fit
from render_figure2_delay_phase_lmm import phase_features, contrasts, adequacy, PHASE, GLOBAL
from render_figure2_delay_legacy_metric_lme import BLOCK_FORMULA, LOCAL_FORMULA, bootstrap_trajectories, adjust_family

SOURCE=Path('J:/ClassicalConditioning Outputs/current/reviews/figure-02-3strace')
OUT=SOURCE/'matched-delay-statistics'
EXCLUDED=['20230426_01','20230802_01','20230803_12','20230822_04']
D_SOURCE=Path('J:/ClassicalConditioning-Archive/repository-cleanup/2026-10-10/payload/reviews/figure2_B_freeze_20261009/D_fish_blocks.parquet')


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def artifact(path):return {'path':str(Path(path).resolve()),'sha256':sha256_file(path)}
def save(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,default=str)


def prepare():
    if (OUT/'plan.json').exists():raise FileExistsError('Preserve existing statistical run')
    original=load(SOURCE/'plan.json')
    selected=[r for r in original['cohort'] if r['recording_id'] not in EXCLUDED]
    if len(selected)!=179:raise ValueError('Unexpected common roster')
    if sha256_file(D_SOURCE)!='65324bfb9cb50e5b690c99fd018a73ace4fb4ffc1428345b64dddd54a8e56c84':raise ValueError('Selected Delay block data changed')
    inputs=[SOURCE/'plan.json',SOURCE/'calculation.json',SOURCE/'window-logmedians.parquet',D_SOURCE,
            Path(__file__),REPO/'src/classical_conditioning/analysis/bout_block_statistics.py',
            REPO/'scripts/render_figure2_delay_logmedian.py',REPO/'scripts/render_figure2_delay_phase_lmm.py',
            REPO/'scripts/render_figure2_delay_legacy_metric_lme.py',REPO/'src/classical_conditioning/analysis/inference/learning_onset.py']
    for row in selected:
        rp=SOURCE/'fish'/(row['recording_id']+'.json');record=load(rp)
        if record['heatmap']!='complete':raise ValueError('Timing-flagged fish in common roster')
        for ref in record['outputs']:
            if sha256_file(Path(ref['path']))!=ref['sha256']:raise ValueError('Numerical cache changed')
        inputs.append(rp)
    save(OUT/'plan.json',{'selected':selected,'excluded_recordings':EXCLUDED,
         'exclusion_reason':'Author 2026-10-19: do not include timing-flagged fish; keep flagged; same fish across B/E/H',
         'inputs':[artifact(p) for p in inputs],
         'E':'Selected D/E joint 36 two-sided exact-sign / Brunner-Munzel t tests, Holm36; frozen D data + updated E data',
         'H':'Selected G block interaction Holm8, local condition difference BH9, local slope raw and BH9, phase-aware baseline-adjusted trial LMM contrast control-minus-Trace change from Pre5-14, BH90; no silent random-intercept fallback',
         'scientific_status':'expanded-cohort draft; no new freeze; the D/E joint family may change D adjusted annotations, not D data',
         'bootstrap_condition_alias':'Trace temporarily named delay only inside the existing descriptive bootstrap helper; output restored to trace; no data substitution'})
    windows=pd.read_parquet(SOURCE/'window-logmedians.parquet')
    windows=windows.loc[windows.fish_id.isin([r['recording_id'] for r in selected])].copy()
    if windows.fish_id.nunique()!=179 or len(windows)!=179*90:raise ValueError('Window roster mismatch')
    windows.to_parquet(OUT/'windows.parquet',index=False)
    parts=[]
    for order,(label,a,b) in enumerate([('PT',10,14),('ET',65,69),('LT',90,94)]):
        grouped=windows.loc[windows.trial_number.between(a,b)].groupby(['condition_id','fish_id']).legacy_log_median_difference
        part=grouped.agg(**{'Fish median response / baseline':'median','eligible_log_trials':'count'}).reset_index()
        part['Selected block order']=order;part['Selected block']=label;part['Eligible']=part.eligible_log_trials.ge(1)
        part['minimum_trials']=1;parts.append(part)
    blocks=pd.concat(parts,ignore_index=True);blocks.to_parquet(OUT/'E-fish-blocks.parquet',index=False)
    tests,changes=corrected_b_statistics({'D':pd.read_parquet(D_SOURCE),'E':blocks})
    tests.to_csv(OUT/'DE-statistics-Holm36.csv',index=False);changes.to_parquet(OUT/'DE-paired-changes.parquet',index=False)
    bins=pd.concat([pd.read_parquet(SOURCE/'fish'/(r['recording_id']+'-bins.parquet')) for r in selected],ignore_index=True)
    pooled=pool_equal_fish(bins,{r['recording_id']:r['condition'] for r in selected})
    pooled.to_parquet(OUT/'B-pooled-bins.parquet',index=False)
    boot=windows.rename(columns={'legacy_log_median_difference':'ratio'}).copy()
    boot.condition_id=boot.condition_id.replace({'trace':'delay'})
    trajectory,draws=bootstrap_trajectories(boot,n_boot=5000,seed=10)
    trajectory.condition_id=trajectory.condition_id.replace({'delay':'trace'})
    trajectory.to_parquet(OUT/'H-trajectory.parquet',index=False)
    np.savez_compressed(OUT/'H-bootstrap-draws.npz',**{f'{c}_{k}':v for c,d in draws.items() for k,v in d.items()})
    print('PREPARED: B/E/H common 179 fish; E joint Holm36 tests complete',flush=True)


def run_models():
    plan=load(OUT/'plan.json')
    for ref in plan['inputs']:
        if sha256_file(Path(ref['path']))!=ref['sha256']:raise ValueError('Sealed analysis input changed')
    windows=pd.read_parquet(OUT/'windows.parquet')
    data=windows.loc[np.isfinite(windows.baseline_log_median)&np.isfinite(windows.response_log_median)].copy()
    data['log_baseline']=data.baseline_log_median;data['log_response']=data.response_log_median
    data['fish_key']=data.fish_id.astype(str)
    data['trial_center']=float(data.trial_number.mean());data['trial_scale']=float(data.trial_number.std(ddof=0))
    data['trial_scaled']=(data.trial_number-data.trial_center)/data.trial_scale
    data['condition_id']=pd.Categorical(data.condition_id,categories=['control','trace'],ordered=True)
    data['block_10_name']=pd.Categorical([BLOCK_ORDER[(int(t)-5)//10] for t in data.trial_number],categories=BLOCK_ORDER,ordered=True)
    features=phase_features();data=data.merge(features,on='trial_number',validate='many_to_one')
    data=data.sort_values(['condition_id','fish_key','trial_number']).reset_index(drop=True)
    data.to_parquet(OUT/'H-model-input.parquet',index=False)
    config=LearningOnsetConfig(metric_id='legacy_distal_angular_speed',outcome_id='conditional-intensity',test_condition='trace',n_bootstrap=0,allow_random_intercept_fallback=False)
    diagnostics=[];failures=[];fits={}
    with threadpool_limits(limits=1):
        specs=[('block',BLOCK_FORMULA,config),('phase',PHASE,config),
               ('phase-powell',PHASE,replace(config,optimizer='powell')),
               ('phase-intercept-sensitivity',PHASE,replace(config,optimizer='powell',random_effects_formula='1'))]
        for name,formula,cfg in specs:
            print('FIT '+name,flush=True)
            try:
                result,diag=fit(data,formula,cfg,name,OUT)
                fits[name]=result;diagnostics.append({'model':name,**diag})
                residual,review=adequacy(result,data,name,diag)
                review['residual_sd_by_condition_phase']={str(k):v for k,v in review['residual_sd_by_condition_phase'].items()}
                save(OUT/(name+'-adequacy.json'),review)
                residual.to_parquet(OUT/(name+'-residuals.parquet'),index=False)
                if name!='block':
                    table,condition,change=contrasts(result,data,features,test_condition='trace')
                    table.to_csv(OUT/(name+'-contrasts.csv'),index=False)
                    np.savez_compressed(OUT/(name+'-contrast-matrices.npz'),condition=condition,change=change)
            except Exception as exc:
                traceback.print_exc();failures.append({'model':name,'error':repr(exc)})
        if 'block' in fits:
            result=fits['block'];rows=[]
            for term in result.fe_params.index:
                if 'condition_id' in term and ':' in term:
                    block=next(b for b in BLOCK_ORDER if '[T.'+b+']' in term)
                    rows.append({'block_10_name':block,'center':float(data.loc[data.block_10_name.eq(block),'trial_number'].mean()),'estimate':float(result.fe_params[term]),'p_raw':float(result.pvalues[term])})
            table=pd.DataFrame(rows);adjust_family(table,'p_raw','p_holm','holm');table.to_csv(OUT/'D-interaction-tests.csv',index=False)
            block_global_interaction_test(result).to_csv(OUT/'joint-interaction-test.csv',index=False)
        local=[]
        for block in BLOCK_ORDER:
            print('LOCAL '+block,flush=True)
            sub=data.loc[data.block_10_name.eq(block)].copy();sub['within_block_trial']=sub.trial_number-sub.trial_number.mean()
            row={'block_10_name':block,'center':float(sub.trial_number.mean()),'mean_estimate':np.nan,'mean_p_raw':np.nan,'slope_estimate':np.nan,'slope_p_raw':np.nan,'status':'unavailable'}
            try:
                result,diag=fit(sub,LOCAL_FORMULA,replace(config,random_effects_formula='1',optimizer='powell'),block.replace(' ','-')+'-local',OUT,allow_failed=True)
                diagnostics.append({'model':block+'-local',**diag})
                if result is not None:
                    row['status']='verified'
                    for term in result.fe_params.index:
                        if 'condition_id' in term:
                            label='slope' if ':' in term else 'mean'
                            row[label+'_estimate']=float(result.fe_params[term]);row[label+'_p_raw']=float(result.pvalues[term])
            except Exception as exc:
                failures.append({'model':block+'-local','error':repr(exc)});traceback.print_exc()
            local.append(row)
        table=pd.DataFrame(local)
        for label in ['mean','slope']:adjust_family(table,label+'_p_raw',label+'_p_fdr','fdr_bh')
        table.to_csv(OUT/'MR-local-block-tests.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'all-fit-diagnostics.csv',index=False)
    save(OUT/'completion.json',{'status':'models-complete' if not failures else 'model-review-required','failures':failures,
         'cohort_counts':{'control':90,'trace':89},'excluded_recordings':EXCLUDED,
         'outputs':[artifact(p) for p in OUT.iterdir() if p.is_file() and p.suffix in {'.csv','.parquet','.npz','.json'}],
         'primary_phase_available':(OUT/'phase-contrasts.csv').exists(),
         'limitations':'Exploratory; report diagnostics and unavailable models; no learning-onset claim. Statistical families match the selected Delay families; influence refits remain a separate audit.'})
    print('STATISTICAL ANALYSIS FINISHED',flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    prepare();run_models()
