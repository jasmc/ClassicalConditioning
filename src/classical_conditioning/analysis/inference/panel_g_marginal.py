"""Unsmoothed fish-level change comparator and inexpensive repeated calibration.

This is a different estimand from an adjusted-response LMM. Each contributing
fish is anchored by its own available Pre mean; no-bout vigor is never zero.
"""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .panel_g_review import PHASE_WINDOWS,save_json,timing_bounds,first_run


def fish_matrix(data):
    matrix=data.pivot(index=['condition_id','fish_id'],columns='trial_number',values='z').reindex(columns=range(5,95))
    values=matrix.to_numpy(float)
    pre=np.nanmean(values[:,:10],axis=1)
    change=values-pre[:,None]
    phases=np.column_stack([np.nanmean(values[:,a-5:b-4],axis=1)-pre for a,b in PHASE_WINDOWS.values()])
    return matrix.index,np.column_stack([change,phases])


def weighted_summary(values,weights):
    valid=np.isfinite(values);x=np.where(valid,values,0)
    counts=weights@valid.astype(float)
    total=weights@x;squares=weights@(x*x)
    means=np.divide(total,counts,out=np.full_like(total,np.nan),where=counts>0)
    variances=np.divide(squares-counts*means**2,counts-1,
        out=np.full_like(total,np.nan),where=counts>1)
    se=np.sqrt(np.maximum(variances,0)/counts)
    return means,se,counts


def marginal_band(data,seed,nboot=499):
    index,x=fish_matrix(data);rng=np.random.default_rng(seed)
    estimates=[];errors=[];boot_values=[];boot_errors=[];coverage=[]
    for condition in ['control','delay']:
        group=x[np.asarray(index.get_level_values('condition_id').astype(str))==condition]
        n=len(group)
        if n<2:raise ValueError('Insufficient fish for marginal comparator')
        m,se,count=weighted_summary(group,np.ones((1,n)))
        weights=rng.multinomial(n,np.repeat(1/n,n),size=nboot)
        bm,bse,_=weighted_summary(group,weights)
        estimates.append(m[0]);errors.append(se[0]);coverage.append(count[0])
        boot_values.append(bm);boot_errors.append(bse)
    values=estimates[0]-estimates[1];se=np.sqrt(errors[0]**2+errors[1]**2)
    draws=boot_values[0]-boot_values[1];drawse=np.sqrt(boot_errors[0]**2+boot_errors[1]**2)
    good=np.isfinite(draws).all(axis=1)&np.isfinite(drawse).all(axis=1)&(drawse>1e-10).all(axis=1)
    if np.any(~np.isfinite(se)) or np.any(se<=1e-10) or good.sum()<np.ceil(.95*nboot):
        return values,np.full(93,np.nan),np.full(93,np.nan),{'status':'band_failed','successful':int(good.sum())},coverage
    maxima=np.max(abs(draws[good]-values)/drawse[good],axis=1)
    critical=float(np.quantile(maxima,.95))
    return values,values-critical*se,values+critical*se,{'status':'exploratory','successful':int(good.sum()),
        'requested':nboot,'critical':critical},coverage


def run_marginal_candidates(out,config):
    data=pd.read_parquet(out/'model-input.parquet')
    v,lo,hi,status,coverage=marginal_band(data,config.seed+215,config.bootstrap)
    pd.DataFrame({'trial_number':range(5,95),'estimate':v[:90],'simultaneous_lower':lo[:90],
        'simultaneous_upper':hi[:90],'control_fish':coverage[0][:90],'delay_fish':coverage[1][:90]}).to_csv(out/'unsmoothed-fish-change-trajectory.csv',index=False)
    pd.DataFrame({'contrast':list(PHASE_WINDOWS),'estimate':v[90:],'simultaneous_lower':lo[90:],
        'simultaneous_upper':hi[90:]}).to_csv(out/'unsmoothed-fish-change-phases.csv',index=False)
    timing=[]
    if status['status']=='exploratory':
        for f in config.threshold_fractions:
            timing.append({'threshold_fraction':f,**timing_bounds(range(5,95),v[:90],lo[:90],hi[:90],-np.log1p(-f),config.persistence)})
    save_json(out/'unsmoothed-fish-change-result.json',{**status,'timing':timing,'paper_approved':False,
        'estimand':'condition difference of fish changes relative to each fish available Pre mean',
        'missingness_assumption':'conditional contribution; no-bout vigor remains undefined; informative availability stress required'})


def marginal_calibration_worker(payload):
    scenario,seed,nboot=payload
    from .panel_g_review_extensions import simulate
    with threadpool_limits(limits=1):
        data,truth=simulate(seed,scenario)
        v,lo,hi,status,_=marginal_band(data,seed+820,nboot)
        row={'scenario':scenario,'seed':seed,'band_status':status['status'],**status}
        if status['status']=='band_failed':return row
        threshold=-np.log(.90);true=first_run(range(5,95),truth,threshold)
        timing=timing_bounds(range(5,95),v[:90],lo[:90],hi[:90],threshold)
        phase=np.array([truth[a-5:b-4].mean() for a,b in PHASE_WINDOWS.values()])
        row.update(timing,true_crossing=true,simultaneous_trial_coverage=bool(((lo[:90]<=truth)&(hi[:90]>=truth)).all()),
            simultaneous_phase_coverage=bool(((lo[90:]<=phase)&(hi[90:]>=phase)).all()),
            false_supported_run=bool(not np.isfinite(true) and np.isfinite(timing['first_supported_trial'])),
            onset_interval_contains_truth=bool(timing['earliest_compatible_trial']<=true<=timing['latest_bound_trial']))
        return row


def run_marginal_calibration(out,config,workers=3,repetitions=100):
    from .panel_g_review_extensions import SCENARIOS
    jobs=[(s,config.seed+50000+SCENARIOS.index(s)*1000+r,499) for s in SCENARIOS for r in range(repetitions)]
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,row in enumerate(pool.map(marginal_calibration_worker,jobs,chunksize=2)):
            rows.append(row)
            if (i+1)%50==0:print(f'Unsmoothed calibration {i+1}/{len(jobs)}',flush=True)
    frame=pd.DataFrame(rows)
    frame.replace([np.inf,-np.inf],np.nan).to_csv(out/'unsmoothed-calibration.csv',index=False)
    summarize_marginal_calibration(out,repetitions)

def summarize_marginal_calibration(out,repetitions=100):
    from scipy.stats import beta
    frame=pd.read_csv(out/'unsmoothed-calibration.csv',keep_default_na=False,na_values=[''])
    accepted_status={'exploratory','bounded','crossing_uncertain','not_localized'}
    summary=[]
    for scenario,g in frame.groupby('scenario'):
        # Older replicate tables stored the timing status in this column.
        # Both schemas retain band failures explicitly.
        accepted=g.loc[g['band_status'].eq('exploratory')] if 'band_status' in g else g.loc[g.status.isin(accepted_status)]
        n=len(accepted)
        for field in ['simultaneous_trial_coverage','simultaneous_phase_coverage','false_supported_run','onset_interval_contains_truth']:
            values=accepted[field].map({True:True,False:False,'True':True,'False':False})
            if n and values.isna().any():raise ValueError('Missing metric in accepted simulation: '+field)
            successes=int(values.sum()) if n else 0
            summary.append({'scenario':scenario,'metric':field,'independent_experiments':n,
                'requested':len(g),'failures_withheld':len(g)-n,'successes':successes,'rate':successes/n if n else np.nan,
                'binomial_lower':float(beta.ppf(.025,successes,n-successes+1)) if successes else 0.,
                'binomial_upper':float(beta.ppf(.975,successes+1,n-successes)) if successes<n else 1.})
    pd.DataFrame(summary).to_csv(out/'unsmoothed-calibration-summary.csv',index=False)
    save_json(out/'unsmoothed-calibration-status.json',{'repetitions_per_scenario':repetitions,
        'bootstrap_per_simulation':499,'paper_approved':False,'estimator':'unsmoothed fish change, direct difference only',
        'summary_definition':'Accepted band fits counted independently of bounded/not-localized timing status',
        'interpretation':'Shape/availability sensitivity comparator; cannot validate adjusted LMM or conditional individual bands.'})
