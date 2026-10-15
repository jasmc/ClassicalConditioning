"""Individual candidates, simulation pilots and isolated review reporting."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import html
import io
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from threadpoolctl import threadpool_limits

from .panel_g_review import (
    ReviewConfig, PHASE_WINDOWS, bootstrap_worker, clean_json, design_for, estimate,
    fit, fit_with_retry, first_run, phase_features, prepare_input, save_json, simultaneous_band,
    timing_bounds,
)
from .panel_g_individual import fit_individual, individual_curve, persistence_forecast
from .panel_g_missingness import scheduled_individual_timing


def credible_envelope(mean,cov,rng,draws=2000):
    values,vectors=np.linalg.eigh((cov+cov.T)/2)
    if values.min() < -1e-7:raise ValueError("Invalid individual covariance")
    factor=vectors@np.diag(np.sqrt(np.maximum(values,0)))
    se=np.sqrt(np.maximum(np.diag(cov),1e-12))
    samples=rng.normal(size=(draws,len(mean)))@factor.T
    critical=np.quantile(np.max(abs(samples)/se,axis=1),.95)
    return mean-critical*se,mean+critical*se,float(critical)


def run_individual(out,config):
    data=pd.read_parquet(out / "model-input.parquet")
    rng=np.random.default_rng(config.seed+601)
    table,trajectories,predictions,diagnostics,validation_trajectories=[],[],[],[],[]
    for method in ["spline","state_space"]:
        print(f"Individual {method}: training-only empirical-Bayes fit",flush=True)
        try:model=fit_individual(data,method)
        except ValueError as error:
            diagnostics.append({"method":method,"status":"failed","error":str(error)});continue
        diagnostics.append({"method":method,"status":"exploratory_conditional_empirical_bayes",
            "reml_objective":float(model.optimizer.fun),"covariance_parameters":{
                "random_intercept_variance":np.exp(model.theta[0]),"curve_variance":np.exp(model.theta[1]),
                "noise_variance":np.exp(model.theta[2]),"ar1":np.tanh(model.theta[3])},
            "iterations":model.optimizer.nit,"fit_trials":[5,64],
            "interval_limitations":"Covariance parameters are estimated and treated as fixed; calibration and hyperparameter uncertainty remain open",
            "baseline_target":"response-minus-baseline candidate only; author selection remains open"})
        for index,group in enumerate(model.groups):
            mean,cov=individual_curve(model,index)
            lower,upper,critical=credible_envelope(mean,cov,rng)
            trials=np.arange(15,65)
            observed=np.isin(trials,group["trials"])
            pre_count=int((group["trials"]<15).sum())
            information_ok=pre_count>=3 and observed.sum()>=10
            for t,m,l,u,seen in zip(trials,mean,lower,upper,observed):
                trajectories.append({"method":method,"condition_id":group["condition"],"fish_id":group["fish_id"],
                    "trial_number":t,"estimate":m,"conditional_lower":l,"conditional_upper":u,"observed":seen,
                    "interval_type":"conditional_empirical_Bayes_joint_credible_envelope"})
            for fraction in config.threshold_fractions:
                timing=scheduled_individual_timing(trials,mean,lower,upper,observed,
                    -np.log1p(-fraction),config.persistence,pre_count=pre_count)
                table.append({"method":method,"condition_id":group["condition"],"fish_id":group["fish_id"],
                    "pre_count":pre_count,"training_count":int(observed.sum()),"threshold_fraction":fraction,
                    "threshold_status":"candidate_not_author_approved",**timing})
            # Hold-out availability does not affect construction or onset eligibility.
            heldout=data.loc[(data.fish_key==group["key"])&data.trial_number.between(65,69)]
            if len(heldout):
                prediction,variance=persistence_forecast(model,index,heldout.trial_number.to_numpy(int))
                observed_mean=float(heldout.z.mean())
                predictions.append({"method":method,"condition_id":group["condition"],"fish_id":group["fish_id"],
                    "heldout_count":len(heldout),"predicted_mean_z":prediction,"observed_mean_z":observed_mean,
                    "conditional_lower":prediction-1.96*np.sqrt(variance),"conditional_upper":prediction+1.96*np.sqrt(variance),
                    "prediction_task":"training-end persistence into available trials65-69; reinforcement transition may invalidate this forecast"})
        # Controls are evaluated using a reference/hyperparameters fitted without
        # their training response. Reinsert only for conditional individual updating.
        controls=[(i,g) for i,g in enumerate(model.groups) if g["condition"]=="control"]
        for j,(index,group) in enumerate(controls):
            excluded=data.loc[data.fish_key!=group["key"]]
            try:
                reference=fit_individual(excluded,method,start=model.theta)
                # Gaussian posterior against independently estimated control drift.
                from scipy.linalg import cho_factor,cho_solve
                from .panel_g_individual import cross_coordinates,coordinates,covariance
                own=reference.cases("control");tr=np.arange(5,65)
                anchor=np.eye(60)-np.ones((60,1))@np.r_[np.repeat(.1,10),np.zeros(50)][None,:]
                intercept,wiggle,noise=np.exp(reference.theta[:3])
                cross=anchor@(intercept+wiggle*cross_coordinates(tr,group["trials"],method))
                prior=anchor@(intercept+wiggle*coordinates(tr,method))@anchor.T
                factor=cho_factor(covariance(group,reference.theta),lower=True)
                residual=group["y"]-group["x"]@reference.beta
                mean=-cross@cho_solve(factor,residual)
                coefficient=cross@cho_solve(factor,group["x"])
                # Own population mean minus external control drift cancels in G;
                # its uncertainty still enters through residual conditioning.
                cov=prior-cross@cho_solve(factor,cross.T)+coefficient@reference.beta_cov@coefficient.T
                lower,upper,_=credible_envelope(mean[10:],cov[10:,10:],rng)
                observed=np.isin(np.arange(15,65),group["trials"])
                for t,m,l,u,seen in zip(np.arange(15,65),mean[10:],lower,upper,observed):
                    validation_trajectories.append({"method":method,"condition_id":"control",
                        "fish_id":group["fish_id"],"trial_number":t,"estimate":m,
                        "conditional_lower":l,"conditional_upper":u,"observed":seen,
                        "validation":"leave_one_control_out"})
                for fraction in config.threshold_fractions:
                    result=scheduled_individual_timing(np.arange(15,65),mean[10:],lower,upper,
                        observed,-np.log1p(-fraction),config.persistence,
                        pre_count=int((group["trials"]<15).sum()))
                    result.update(method=method,condition_id="control",fish_id=group["fish_id"],threshold_fraction=fraction)
                    table.append({**result,"validation":"leave_one_control_out","threshold_status":"candidate_not_author_approved"})
            except ValueError as error:
                table.append({"method":method,"fish_id":group["fish_id"],"condition_id":"control",
                              "validation":"leave_one_control_out","status":"failed","error":str(error)})
            if (j+1)%7==0:print(f"Individual {method}: excluded-control validation {j+1}/{len(controls)}",flush=True)
    table_frame=pd.DataFrame(table)
    for column in ['method','condition_id','fish_id','threshold_fraction','status','validation','timing_definition']:
        if column not in table_frame:table_frame[column]=pd.Series(dtype='object')
    table_frame.replace([np.inf,-np.inf],np.nan).to_csv(out / "individual-timing-candidates.csv",index=False)
    trajectory_columns=['method','condition_id','fish_id','trial_number','estimate','conditional_lower','conditional_upper','observed','interval_type']
    pd.DataFrame(trajectories,columns=trajectory_columns).to_parquet(out / "individual-training-trajectories.parquet",index=False)
    pd.DataFrame(validation_trajectories,columns=trajectory_columns[:-1]+['validation']).to_parquet(out / "individual-control-validation-trajectories.parquet",index=False)
    prediction_columns=['method','condition_id','fish_id','heldout_count','predicted_mean_z','observed_mean_z','conditional_lower','conditional_upper','prediction_task']
    pd.DataFrame(predictions,columns=prediction_columns).to_csv(out / "individual-heldout-persistence.csv",index=False)
    save_json(out / "individual-model-diagnostics.json",diagnostics)


SCENARIOS=("null","preexisting","abrupt","gradual","transient","unequal_coverage",
           "heavy_tail_serial","influential","informative_availability","tracking_gaps","singular_random_effect")


def simulate(seed,scenario,counts=(28,29)):
    rng=np.random.default_rng(seed);trials=np.arange(5,95);rows=[];individual_truth={}
    t=np.clip((trials-20)/25,0,1)
    effect=np.zeros(90)
    if scenario=="abrupt":effect=np.where((trials>=30)&(trials<65),.25,.20)
    elif scenario in {"gradual","unequal_coverage","heavy_tail_serial","influential",
                      "informative_availability","tracking_gaps"}:
        effect=.25*t;effect[trials>=65]=.25-.15*(trials[trials>=65]-65)/29
    elif scenario=="transient":effect=np.where(np.isin(trials,[30,31]),.25,0)
    effect[trials<15]=0
    if scenario=="abrupt":effect[trials<30]=0
    for c,n in zip(["control","delay"],counts):
        for i in range(n):
            intercept=rng.normal(0,.08);slope=rng.normal(0,.025)
            if scenario=="singular_random_effect":intercept=0.;slope=0.
            baseline=-5+rng.normal(0,.15)+.10*(trials-5)/89+rng.normal(0,.06,90)
            innovation=(rng.standard_t(5,90)/np.sqrt(5/3) if scenario=="heavy_tail_serial" else rng.normal(size=90))*.12
            residual=innovation.copy()
            for j in range(1,90):residual[j]+=.25*residual[j-1]
            z=intercept+slope*(trials-49.5)/26+residual
            if c=="delay":z-=effect
            if scenario=="preexisting" and c=="delay":z-=.25
            if scenario=="influential" and c=="delay" and i==0:z-=.4*np.maximum((trials-15)/49,0)
            individual_effect=(effect.copy() if c=="delay" else np.zeros(90))-slope*(trials-9.5)/26
            if scenario=="influential" and c=="delay" and i==0:
                individual_effect+=.4*np.maximum((trials-15)/49,0)
            individual_truth[f"{c}-{i:02d}"]=individual_effect[10:60].tolist()
            missing=rng.random(90)<(.25 if scenario=="unequal_coverage" and c=="control" else .06)
            if scenario=="informative_availability":
                from scipy.special import expit
                # Deliberate selection stress, not an imputation of undefined
                # bout vigor: availability depends on response and baseline.
                probability=expit(-2.4+5*(z-intercept)+2*(baseline+5)
                                  +(.8 if c=="control" else 0))
                missing=rng.random(90)<probability
            if scenario=="tracking_gaps" and i%4==0:
                start=int(rng.integers(15,55));missing[(trials>=start)&(trials<start+8)]=True
            for j,trial in enumerate(trials):
                rows.append({"fish_id":f"{c}-{i:02d}","condition_id":c,"trial_number":trial,
                    "baseline_logmedian":np.nan if missing[j] else baseline[j],
                    "response_logmedian":np.nan if missing[j] else baseline[j]+z[j]})
    if scenario=="influential":effect=effect+.4*np.maximum((trials-15)/49,0)/counts[1]
    result=prepare_input(pd.DataFrame(rows));result.attrs["individual_truth"]=individual_truth
    return result,effect


def simulation_worker(payload):
    seed,scenario,target,nboot,threshold=payload
    with threadpool_limits(limits=1):
        data,truth=simulate(seed,scenario)
        result,diagnostic=fit_with_retry(data,target,extended=False)
        row={"seed":seed,"scenario":scenario,"target":target,"threshold":threshold,
             "true_crossing":first_run(np.arange(5,95),truth,threshold),"status":diagnostic["diagnostic_status"]}
        if result is None:return row
        baseline=float(data.baseline_logmedian.mean())
        _,matrix,_=design_for(result,target,baseline);values,errors=estimate(result,matrix)
        # Intentionally small, explicitly labelled pilot; no paper gate passes.
        draws=[]
        for i in range(nboot):
            payload=(data,target,seed*1000+i,baseline)
            draw=bootstrap_worker(payload)
            draws.append(draw)
        lo,hi,critical,maxima=simultaneous_band(values,errors,draws)
        success=sum(x["success"] for x in draws)
        row["bootstrap_successful"]=success;row["bootstrap_requested"]=nboot
        if success<np.ceil(.95*nboot):row["status"]="bootstrap_failed";return row
        coverage=(lo[:90]<=truth)&(hi[:90]>=truth)
        row["simultaneous_trial_coverage"]=bool(coverage.all())
        phase_truth=np.array([truth[a-5:b-4].mean() for a,b in PHASE_WINDOWS.values()])
        row["simultaneous_phase_coverage"]=bool(((lo[90:]<=phase_truth)&(hi[90:]>=phase_truth)).all())
        row.update(timing_bounds(np.arange(5,95),values[:90],lo[:90],hi[:90],threshold))
        row["false_supported_run"]=bool(not np.isfinite(row["true_crossing"]) and np.isfinite(row["first_supported_trial"]))
        return row


def run_calibration(out,config,workers,*,repetitions=2,bootstrap_per_simulation=39):
    # Two repeats per scenario/target is feasibility evidence only, not a
    # scientifically useful estimate of a 5% false-localization rate.
    if repetitions<1 or bootstrap_per_simulation<19:raise ValueError("Invalid pilot settings")
    payloads=[(config.seed+10000+i*1000+r,s,t,bootstrap_per_simulation,-np.log(.90))
              for i,s in enumerate(SCENARIOS) for r in range(repetitions)
              for t in ["difference","adjusted_response"]]
    progress=out/'calibration-progress.json'
    settings={'seed':config.seed,'repetitions':repetitions,'bootstrap_per_simulation':bootstrap_per_simulation}
    rows=[]
    if progress.exists():
        checkpoint=json.loads(progress.read_text())
        if checkpoint['settings']!=settings:raise ValueError('Calibration checkpoint settings changed')
        rows=checkpoint['rows']
    completed={(r['seed'],r['scenario'],r['target']) for r in rows}
    pending=[p for p in payloads if (p[0],p[1],p[2]) not in completed]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(simulation_worker,pending):
            rows.append(row)
            save_json(progress,{'settings':settings,'rows':rows})
            print(f"Calibration pilot {len(rows)}/{len(payloads)}: {row['scenario']} / {row['target']}",flush=True)
    pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).to_csv(out / "calibration-pilot.csv",index=False)
    save_json(out / "calibration-status.json",{"status":"pilot_only_not_paper_calibrated",
        "repetitions_per_scenario_target":repetitions,"bootstrap_per_simulation":bootstrap_per_simulation,"threshold_fraction":.10,
        "scenarios":SCENARIOS,"paper_gate_passed":False,
        "remaining":["hundreds of independent simulations per scenario","informative missingness stress tests",
                     "singular-fit and failure-propagation calibration","individual timing coverage and prior/hyperparameter sensitivity",
                     "joint phase-contrast coverage and model-family misspecification"]})
    individual_rows=[]
    rng=np.random.default_rng(config.seed+702)
    for scenario in ["null","gradual","abrupt","transient","informative_availability","tracking_gaps"]:
        data,truth=simulate(config.seed+20100+SCENARIOS.index(scenario),scenario)
        for method in ["spline","state_space"]:
            try:
                model=fit_individual(data,method)
                for index,g in enumerate(model.groups):
                    mean,cov=individual_curve(model,index);lo,hi,_=credible_envelope(mean,cov,rng)
                    real=data.attrs["individual_truth"][g["fish_id"]]
                    timing=scheduled_individual_timing(np.arange(15,65),mean,lo,hi,
                        np.isin(np.arange(15,65),g["trials"]),-np.log(.90),config.persistence,
                        pre_count=int((g["trials"]<15).sum()))
                    real_cross=first_run(np.arange(15,65),real,-np.log(.90))
                    individual_rows.append({"scenario":scenario,"method":method,"fish_id":g["fish_id"],
                        "true_crossing":real_cross,"simultaneous_coverage":bool(((lo<=real)&(hi>=real)).all()),
                        "false_supported_run":bool(not np.isfinite(real_cross) and
                            timing.get("first_supported_trial") is not None and np.isfinite(timing["first_supported_trial"])),**timing})
            except ValueError as error:individual_rows.append({"scenario":scenario,"method":method,"status":"failed","error":str(error)})
    individual_frame=pd.DataFrame(individual_rows)
    for column in ['scenario','method','fish_id','simultaneous_coverage','false_supported_run','status']:
        if column not in individual_frame:individual_frame[column]=np.nan
    individual_frame.replace([np.inf,-np.inf],np.nan).to_csv(out / "individual-calibration-pilot.csv",index=False)


def optimizer_retry_worker(payload):
    """Retry a failed fit numerically, preserving its fixed/random structure."""
    data,target,seed,baseline=payload
    from .panel_g_review import resample,formula
    from .learning_onset import LearningOnsetConfig,_fit_mixed_model
    with threadpool_limits(limits=1):
        sampled=resample(data,np.random.default_rng(seed))
        config=LearningOnsetConfig(metric_id="legacy_distal_angular_speed",optimizer="powell",
                                  allow_random_intercept_fallback=False)
        result,diagnostic=_fit_mixed_model(sampled,formula=formula(target),config=config,
            group_column="resampled_key",collect_extended_diagnostics=False)
        if result is None:return {"success":False,"diagnostic":diagnostic}
        _,matrix,_=design_for(result,target,baseline);values,errors=estimate(result,matrix)
        if not np.isfinite(values).all() or not np.all(errors>1e-10):
            return {"success":False,"diagnostic":{"error":"nonfinite retry contrast/SE"}}
        return {"success":True,"estimate":values,"se":errors,"optimizer":"powell","diagnostic":diagnostic}


def run_optimizer_stability(out,config,workers=3):
    data=pd.read_parquet(out / "model-input.parquet")
    seeds=np.random.SeedSequence(config.seed).generate_state(config.bootstrap)
    for target in ["difference","adjusted_response"]:
        directory=out / target
        if (directory / "optimizer-stability.json").exists():continue
        original=json.loads((directory / "result.json").read_text())
        if original.get("status")=="failed":
            save_json(directory / "optimizer-stability.json",{
                "status":"full_model_failed_no_bootstrap_to_stabilize","inference_withheld":True})
            continue
        if original.get("bootstrap_optimizer_policy")=="lbfgs_then_powell_same_structure":
            records=json.loads((directory/'bootstrap-fit-diagnostics.json').read_text())
            save_json(directory/'optimizer-stability.json',{'status':'numeric_retries_completed_inside_bootstrap',
                'requested':config.bootstrap,'successful':original['bootstrap_successful'],
                'successful_Powell_retries':sum(e['success'] and e['optimizer']=='powell' for e in records),
                'structure_changed':False,'failed_refits_retained_as_failures':True})
            continue
        failures=json.loads((directory / "bootstrap-failures.json").read_text())
        if not failures:
            save_json(directory / "optimizer-stability.json",{"retry_count":0});continue
        # Preserve the unpublished first numerical result byte-for-byte.
        history=directory / "lbfgs-first-attempt";history.mkdir(exist_ok=True)
        names=["result.json","bootstrap-failures.json","bootstrap-draws.npz","population-trajectory.csv","planned-phase-contrasts.csv"]
        for name in names:
            p=directory / name;saved=history / name
            if not saved.exists():saved.write_bytes(p.read_bytes())
        arrays=np.load(history / "bootstrap-draws.npz")
        failed={row["replicate"] for row in failures}
        draws=[None]*config.bootstrap
        for i,values,errors in zip([j for j in range(config.bootstrap) if j not in failed],arrays["estimates"],arrays["standard_errors"]):
            draws[i]={"success":True,"estimate":values,"se":errors,"optimizer":"lbfgs"}
        indices=sorted(failed)
        payloads=[(data,target,int(seeds[i]),float(data.baseline_logmedian.mean())) for i in indices]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for j,(index,draw) in enumerate(zip(indices,pool.map(optimizer_retry_worker,payloads))):
                draws[index]=draw
                if (j+1)%25==0:print(f"{target}: optimizer retry {j+1}/{len(indices)}",flush=True)
        if any(x is None for x in draws):raise ValueError("Missing replicate after numerical retry")
        result,diagnostic=fit(data,target)
        if result is None:raise ValueError("Original full fit changed on retry")
        features,matrix,_=design_for(result,target,float(data.baseline_logmedian.mean()))
        values,errors=estimate(result,matrix)
        lower,upper,critical,maxima=simultaneous_band(values,errors,draws)
        successful=sum(x["success"] for x in draws)
        accepted=successful>=np.ceil(.95*config.bootstrap)
        if not accepted:lower[:]=np.nan;upper[:]=np.nan
        table=pd.read_csv(history / "population-trajectory.csv")
        table["simultaneous_lower"]=lower[:90];table["simultaneous_upper"]=upper[:90]
        table.to_csv(directory / "population-trajectory.csv",index=False)
        phases=pd.read_csv(history / "planned-phase-contrasts.csv")
        phases["simultaneous_lower"]=lower[90:];phases["simultaneous_upper"]=upper[90:]
        phases.to_csv(directory / "planned-phase-contrasts.csv",index=False)
        timings=[]
        if accepted:
            for fraction in config.threshold_fractions:
                timing=timing_bounds(np.arange(5,95),values[:90],lower[:90],upper[:90],-np.log1p(-fraction),config.persistence,table.coverage_adequate.to_numpy(bool))
                timing.update(threshold_fraction=fraction,threshold_log_units=-np.log1p(-fraction),threshold_status="exploratory_not_author_approved",persistence=config.persistence)
                timings.append(timing)
        original.update(status="exploratory_candidate" if accepted else "band_failed",timing=timings,
            bootstrap_successful=successful,critical=critical,
            joint_global_curve_and_phase_p=float((1+np.sum(np.asarray(maxima)>=np.max(abs(values/errors))))/(len(maxima)+1)) if accepted else None,
            numerical_retry="lbfgs then powell on same resampled fish/model; first attempt retained")
        save_json(directory / "result.json",original)
        save_json(directory / "optimizer-stability.json",{"requested":config.bootstrap,"first_attempt_successful":config.bootstrap-len(indices),
            "retried":len(indices),"retry_successful":sum(draws[i]["success"] for i in indices),"final_successful":successful,
            "structure_changed":False,"optimizer_order":["lbfgs","powell"]})
        save_json(directory / "bootstrap-failures.json",[{"replicate":i,**x} for i,x in enumerate(draws) if not x["success"]])
        np.savez_compressed(directory / "bootstrap-draws.npz",estimates=np.array([x["estimate"] for x in draws if x["success"]]),
                            standard_errors=np.array([x["se"] for x in draws if x["success"]]),maxima=maxima)


def pulse_details(out):
    windows=pd.read_parquet(out / "scheduled-window-outcomes.parquet")
    lineage=json.loads((out / "input-lineage.json").read_text())
    protocols={Path(x["path"]).parent.name:x for x in lineage if Path(x["path"]).name=="stimulus_events.parquet"}
    rows=[]
    for fish,g in windows.groupby("fish_id",observed=True):
        record=protocols[fish];p=Path(record["path"])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record["sha256"]:raise ValueError("Changed protocol")
        events=pd.read_parquet(p);events=events.loc[events.Type.astype(str)=="Reinforcer"]
        for r in g.itertuples():
            start=r.event_start_absolute_time_ms
            for prefix,left,right in [("baseline",start-15000,start),("response",start,start+9000)]:
                selected=events.loc[(events.Beg<right)&(events.End>left)]
                for e in selected.itertuples():
                    rows.append({"fish_id":fish,"condition_id":r.condition_id,"trial_number":r.trial_number,
                        "window":prefix,"US_start_relative_CS_ms":int(e.Beg-start),"US_duration_ms":int(e.End-e.Beg),
                        "overlap_ms":int(min(e.End,right)-max(e.Beg,left))})
    result=pd.DataFrame(rows);result.to_csv(out / "actual-US-window-overlaps.csv",index=False)
    return result


def build_report(out,config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    plt.rcParams.update({"svg.fonttype":"none","font.family":"DejaVu Sans","font.size":9})
    data=pd.read_parquet(out / "model-input.parquet")
    windows=pd.read_parquet(out / "scheduled-window-outcomes.parquet")
    overlap=pulse_details(out)
    fig,axes=plt.subplots(4,1,figsize=(10,12),layout="constrained",sharex=True)
    palette={"difference":"#7543a1","adjusted_response":"#147d92"}
    results={}
    for target in palette:
        directory=out / target
        if not (directory / "result.json").exists():continue
        result=json.loads((directory / "result.json").read_text());results[target]=result
        if result["status"]=="failed":continue
        table=pd.read_csv(directory / "population-trajectory.csv")
        axis=axes[0 if target=="difference" else 1]
        for i,(phase,g) in enumerate(table.groupby("fit_phase",sort=False)):
            axis.plot(g.trial_number,g.estimate,color=palette[target],lw=1.6,label="Phase-aware LMM" if i==0 else None)
            axis.fill_between(g.trial_number,g.simultaneous_lower,g.simultaneous_upper,color=palette[target],alpha=.2)
        categorical=directory / "categorical-trial-sensitivity.csv"
        if categorical.exists():
            points=pd.read_csv(categorical)
            axis.scatter(points.trial_number,points.estimate,s=6,color=".65",alpha=.65,label="Categorical-trial sensitivity")
        axis.legend(frameon=False,fontsize=8)
        for fraction in config.threshold_fractions:
            axis.axhline(-np.log1p(-fraction),ls=":" if fraction else "-",lw=.6,color=".55")
        axis.set_title("Response − baseline LogMedian" if target=="difference" else "Response LogMedian adjusted for baseline")
        axis.set_ylabel("Control − Delay change\nrelative to Pre (log units)")
    for condition,color in [("control","#009fd5"),("delay","#d83983")]:
        mean=data.groupby(["condition_id","trial_number"],observed=True).z.mean().loc[condition]
        axes[2].plot(mean.index,mean.values,color=color,label=condition.title(),lw=1)
        counts=data.loc[data.condition_id==condition].groupby("trial_number",observed=True).fish_id.nunique().reindex(range(5,95),fill_value=0)
        axes[3].plot(counts.index,counts.values,color=color,label=condition.title(),lw=1)
    axes[2].axhline(0,color=".5",lw=.6);axes[2].set_ylabel("Observed fish mean\nresponse − baseline");axes[2].legend(frameon=False)
    axes[3].set_ylabel("Contributing fish");axes[3].set_xlabel("Original scheduled CS trial");axes[3].legend(frameon=False)
    for axis in axes:
        axis.axvline(14.5,color=".6",ls=":",lw=.6);axis.axvline(64.5,color=".6",ls=":",lw=.6)
        axis.spines[["top","right"]].set_visible(False);axis.set_xlim(4,95)
    fig.suptitle("Independent G analysis — exploratory candidates; original G preserved",fontsize=12)
    svg=out / "population-candidates.svg";fig.savefig(svg);plt.close(fig)
    sections=[]
    for target,result in results.items():
        sections.append(f"<h2>{html.escape(target.replace('_',' ').title())}</h2>")
        phase=out / target / "planned-phase-contrasts.csv"
        if phase.exists():sections.append(pd.read_csv(phase).round(4).to_html(index=False,border=0))
        sections.append(pd.DataFrame(result.get("timing",[])).to_html(index=False,border=0,na_rep="unresolved",float_format=lambda x:f"{x:.3g}"))
        sections.append(f"<p>Successful bootstrap refits: {result.get('bootstrap_successful',0)} / {config.bootstrap}. "
                        f"Residual excess kurtosis: {result.get('model_adequacy',{}).get('residual_excess_kurtosis','unavailable')}. "
                        "These are exploratory results, not approved paper inference.</p>")
        day=out / target / "day-cluster-phase-sensitivity.csv"
        if day.exists():sections.extend(["<h3>Recording-day cluster sensitivity</h3>",pd.read_csv(day).round(4).to_html(index=False,border=0),
            "<p>Whole recording dates are resampled together, retaining their condition balance. These sensitivity bands do not approve the independence structure.</p>"])
    qc=windows.groupby(["condition_id","eligibility_reason"],observed=True).size().rename("observations").reset_index()
    coverage=pd.read_csv(out / "fish-block-coverage.csv")
    sparse=coverage.loc[coverage.eligible<6].copy()
    inventory=json.loads((out / "recording-inventory.json").read_text())
    sections.extend(["<h2>Recording and outcome availability</h2>",
        f"<p>{inventory['record_count']} raw recording identities; {html.escape(str(inventory['status_counts']))}. "
        "Raw inventory verifies presence and headers, not a new technical policy. The main G cohort retains every selected fish; sensitivity removals are listed separately.</p>",qc.to_html(index=False,border=0),
        "<h3>Blocks with fewer than six eligible outcomes — review flags only</h3>",sparse.to_html(index=False,border=0),
        "<h3>Actual US overlaps with fixed windows</h3>",overlap.groupby(["condition_id","window"],observed=True).agg(
            events=("fish_id","size"),median_overlap_ms=("overlap_ms","median"),max_overlap_ms=("overlap_ms","max")).reset_index().to_html(index=False,border=0),
        "<p>Overlap is an event-timing flag, not proof of motor contamination. The sensitivity omits overlapped windows; "
        "the primary candidates retain the frozen windows.</p>"])
    quality=pd.read_csv(out / "fish-technical-audit.csv")
    quality["coordinate_valid_fraction"]=quality.frame_valid_count/quality.matched_frames
    stress=json.loads((out / "technical-coverage-sensitivity.json").read_text())
    sections.extend(["<h3>Recording quality and detector eligibility are different</h3>",
        f"<p>The captured coordinate-valid fractions range from {quality.coordinate_valid_fraction.min():.3f} to "
        f"{quality.coordinate_valid_fraction.max():.3f}. This is limited technical evidence, not a complete quality approval. "
        f"The unapproved 95% detector-valid frame stress criterion would retain {stress['retained_eligible']} of "
        f"{stress['original_eligible']} eligible outcomes and is not applied.</p>",
        "<p>The saved detector-valid mask also requires a finite rolling envelope and derivative/tail coverage. "
        "It is not equivalent to coordinate tracking validity. No detected valid bout does not establish biological immobility; "
        "bout detection and coverage remain scientific review requirements.</p>"])
    if (out / "individual-timing-candidates.csv").exists():
        individual=pd.read_csv(out / "individual-timing-candidates.csv")
        selected=individual.loc[individual.validation.isna()] if "validation" in individual else individual
        sections.extend(["<h2>Individual expression candidates</h2>",
            "<p>Two training-only models: hierarchical spline and random-walk state space. Intervals are conditional "
            "empirical-Bayes credible envelopes with estimated covariance parameters, not calibrated individual acquisition claims. "
            "A row with a missing upper bound has unresolved timing; it is not a non-learner label. "
            "Scheduled-grid model timing remains possible at unavailable trials; actually observed supported runs are recorded separately.</p>",
            selected.groupby(["method","condition_id","threshold_fraction","status"],observed=True).size().rename("fish").reset_index().to_html(index=False,border=0),
            "<details><summary>Individual timing table</summary>",individual.to_html(index=False,border=0,na_rep="unresolved"),"</details>"])
        from .panel_g_review_visuals import individual_gallery
        sections.append(individual_gallery(out))
        predictions=pd.read_csv(out / "individual-heldout-persistence.csv")
        summary=[]
        for (method,condition),g in predictions.groupby(["method","condition_id"]):
            summary.append({"method":method,"condition":condition,"fish":len(g),
                "rmse":np.sqrt(np.mean((g.predicted_mean_z-g.observed_mean_z)**2)),
                "predictive_coverage":np.mean((g.observed_mean_z>=g.conditional_lower)&(g.observed_mean_z<=g.conditional_upper)),
                "spearman":spearmanr(g.predicted_mean_z,g.observed_mean_z).statistic})
        pd.DataFrame(summary).to_csv(out / "individual-heldout-summary.csv",index=False)
        sections.extend(["<h3>Held-out early-test persistence forecast</h3>",pd.DataFrame(summary).round(4).to_html(index=False,border=0),
            "<p>This evaluates a persistence forecast across a reinforcement transition, not an extinction model.</p>"])
    if (out / "calibration-pilot.csv").exists():
        calibration=pd.read_csv(out / "calibration-pilot.csv",keep_default_na=False,na_values=[""])
        settings=json.loads((out/'calibration-status.json').read_text())
        sections.extend(["<h2>Simulation feasibility pilot</h2>",
            f"<p>{settings['repetitions_per_scenario_target']} repeats per scenario and model, with "
            f"{settings['bootstrap_per_simulation']} refits per simulated experiment. This is insufficient to approve "
            "a false-onset rate or interval coverage. No paper calibration gate has passed.</p>",calibration.to_html(index=False,border=0)])
        pilot=out / "individual-calibration-pilot.csv"
        if pilot.exists():
            checks=pd.read_csv(pilot,keep_default_na=False,na_values=[""])
            summary=checks.groupby(["scenario","method"],observed=True).agg(fish=("fish_id","count"),
                conditional_band_coverage=("simultaneous_coverage","mean"),false_supported_runs=("false_supported_run","sum")).reset_index()
            sections.extend(["<h3>Individual conditional-model feasibility pilot</h3>",summary.to_html(index=False,border=0),
                "<p>Fish within the same simulated experiment are not independent simulation repetitions. These fractions cannot establish nominal coverage.</p>"])
    from .panel_g_review_visuals import diagnostic_views
    sections.extend(["<h2>Model adequacy plots</h2>",diagnostic_views(out)])
    from .panel_g_companions import companion_views
    sections.append(companion_views(out))
    from .panel_g_followup_report import followup_sections
    sections.append(followup_sections(out))
    legend=("G independent scientific draft. Bout-window medians of natural-log positive valid adjacent detected-bout vigor; "
        "baseline [−15,0) s, response [0,9) s; no second log. Models use fish random intercept/slope and separate phase bases. "
        "Shading is a nominal 95% bootstrap-t simultaneous band jointly covering 90 scheduled trial contrasts and three phase averages. "
        "Candidate thresholds 0%, 5%, 10%, 20% translate to −ln(1−fraction) relative normalized log-vigor change; none is "
        "approved as biologically meaningful. Timing requires three consecutive scheduled training trials and adequate coverage. "
        "Whole-fish trajectory resampling retains missing outcomes. Gaussian adequacy, informative bout availability, actual US "
        "overlap and recording/day independence require review. Curve coverage failed the abrupt/transient simulation pilots; "
        "these bands and timing outputs are not calibrated paper inference. Movement-occurrence context includes valid no-bout trials. "
        "Original G and its selection remain unchanged.")
    (out / "legend.md").write_text(legend+"\n")
    content=("<!doctype html><html><head><meta charset='utf-8'><title>Independent G review</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#222}"
        "svg{width:100%;height:auto}table{border-collapse:collapse;font-size:12px;margin:15px 0}"
        "td,th{padding:5px 9px;border-bottom:1px solid #ddd;text-align:left}details{overflow:auto}"
        ".notice{background:#fff2d5;padding:16px;border-radius:8px}</style></head><body>"
        "<h1>Independent Figure 2 G review</h1><p class='notice'>Exploratory scientific draft. The current G version "
        "is preserved. Both baseline treatments and all thresholds are candidates; no cohort or paper approval is implied. "
        "The simulation pilots expose curve-coverage failures for abrupt and transient changes, so timing inference remains provisional.</p>"
        +svg.read_text()+"<p>"+html.escape(legend)+"</p>"+"".join(sections)+"</body></html>")
    (out / "review.html").write_text(content)
    artifacts=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name!="artifact-manifest.json":artifacts.append({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    save_json(out / "artifact-manifest.json",{"status":"independent_exploratory_review","artifacts":artifacts,
        "original_selection_changed":False,"cohort_exclusions":0,"primary_target_selected":False,
        "meaningful_threshold_selected":False,"paper_calibration_passed":False})
