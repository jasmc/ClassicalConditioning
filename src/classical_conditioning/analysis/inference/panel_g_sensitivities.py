"""Bounded smoothing, technical-coverage and recording-day sensitivities."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from patsy import build_design_matrices
from threadpoolctl import threadpool_limits
from .learning_onset import LearningOnsetConfig,_fit_mixed_model
from .panel_g_review import design_for,estimate,fit,prepare_input,save_json,simultaneous_band,PHASE_WINDOWS


def day_cluster_sample(data,rng):
    days=data.day.astype(str).unique();frames=[]
    for i,day in enumerate(rng.choice(days,len(days),replace=True)):
        rows=data.loc[data.day.astype(str)==day].copy()
        rows["resampled_key"]=str(i)+"::"+rows.fish_key.astype(str)
        frames.append(rows)
    return pd.concat(frames,ignore_index=True)


def day_worker(payload):
    data,target,seed,baseline=payload
    from .panel_g_review import formula
    with threadpool_limits(limits=1):
        sampled=day_cluster_sample(data,np.random.default_rng(seed))
        diagnostic={}
        for optimizer in ["lbfgs","powell"]:
            cfg=LearningOnsetConfig(metric_id="legacy_distal_angular_speed",optimizer=optimizer,allow_random_intercept_fallback=False)
            result,diagnostic=_fit_mixed_model(sampled,formula=formula(target),config=cfg,group_column="resampled_key",collect_extended_diagnostics=False)
            if result is not None:break
        if result is None:return {"success":False,"diagnostic":diagnostic}
        _,matrix,_=design_for(result,target,baseline);values,se=estimate(result,matrix)
        if not np.isfinite(values).all() or not np.all(se>1e-10):return {"success":False,"diagnostic":{"error":"invalid contrast"}}
        return {"success":True,"estimate":values,"se":se}


def categorical_sensitivity(data,target):
    adjustment="baseline_logmedian + " if target=="adjusted_response" else ""
    response="response_logmedian" if target=="adjusted_response" else "z"
    formula=f"{response} ~ {adjustment}C(condition_id, Treatment(reference='control')) * C(trial_number)"
    for optimizer in ["lbfgs","powell"]:
        cfg=LearningOnsetConfig(metric_id="legacy_distal_angular_speed",optimizer=optimizer,allow_random_intercept_fallback=False)
        result,diagnostic=_fit_mixed_model(data,formula=formula,config=cfg,collect_extended_diagnostics=True)
        if result is not None:break
    if result is None:return None,diagnostic
    designs=[]
    for condition in ["control","delay"]:
        cases=pd.DataFrame({"trial_number":range(5,95),"condition_id":condition,"baseline_logmedian":data.baseline_logmedian.mean()})
        designs.append(np.asarray(build_design_matrices([result.model.data.design_info],cases)[0]))
    matrix=designs[0]-designs[1];matrix-=matrix[:10].mean(axis=0)
    values,se=estimate(result,matrix)
    return pd.DataFrame({"trial_number":range(5,95),"estimate":values,"pointwise_lower":values-1.96*se,"pointwise_upper":values+1.96*se}),diagnostic


def run_sensitivities(out,config,workers=3,draw_count=199):
    data=pd.read_parquet(out / "model-input.parquet")
    days=data[["day","condition_id","fish_id"]].drop_duplicates().groupby(["day","condition_id"],observed=True).size().unstack(fill_value=0)
    days.to_csv(out / "day-condition-balance.csv")
    windows=pd.read_parquet(out / "scheduled-window-outcomes.parquet")
    technical=(windows.baseline_technical_valid_frames/windows.baseline_observed_frames>=.95)&(windows.response_technical_valid_frames/windows.response_observed_frames>=.95)
    retained=prepare_input(windows.loc[technical])
    save_json(out / "technical-coverage-sensitivity.json",{"threshold":.95,"approval_status":"unapproved_sensitivity_only",
        "original_eligible":len(data),"retained_eligible":len(retained),"cohort_membership_changed":False,
        "definition":"detector-valid adjacent finite frames / observed camera frames; not bout-bearing fraction",
        "classification":"signal_and_detector_eligibility_stress_not_independent_tracking_quality",
        "can_define_primary_membership":False})
    for target in ["difference","adjusted_response"]:
        result,diagnostic=fit(data,target)
        if result is None:continue
        _,matrix,_=design_for(result,target,float(data.baseline_logmedian.mean()))
        values,se=estimate(result,matrix)
        directory=out / target
        seeds=np.random.SeedSequence(config.seed+443).generate_state(draw_count)
        jobs=[(data,target,int(seed),float(data.baseline_logmedian.mean())) for seed in seeds]
        draws=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i,draw in enumerate(pool.map(day_worker,jobs)):
                draws.append(draw)
                if (i+1)%50==0:print(f"{target}: day-cluster sensitivity {i+1}/{draw_count}",flush=True)
        lower,upper,critical,maxima=simultaneous_band(values,se,draws)
        success=sum(x["success"] for x in draws)
        if success<np.ceil(.95*draw_count):lower[:]=np.nan;upper[:]=np.nan
        pd.DataFrame({"contrast":list(PHASE_WINDOWS),"estimate":values[90:],"simultaneous_lower":lower[90:],"simultaneous_upper":upper[90:]}).to_csv(directory / "day-cluster-phase-sensitivity.csv",index=False)
        pd.DataFrame({"trial_number":range(5,95),"estimate":values[:90],"simultaneous_lower":lower[:90],"simultaneous_upper":upper[:90]}).to_csv(directory / "day-cluster-trajectory-sensitivity.csv",index=False)
        save_json(directory / "day-cluster-diagnostics.json",{"draws":draw_count,"successful":success,"days":len(days),
            "role":"recording-day-cluster sensitivity; independence structure remains unapproved", "critical":critical})
        categorical,diagnostic=categorical_sensitivity(data,target)
        save_json(directory / "categorical-trial-diagnostics.json",diagnostic)
        if categorical is not None:categorical.to_csv(directory / "categorical-trial-sensitivity.csv",index=False)
    save_json(out / "sensitivities-complete.json",{"scope":"exploratory; technical quality, day clustering, categorical trial",
        "paper_approval":False,"original_G_changed":False})
