"""Keep latent timing uncertainty separate from observed-run evidence.

An unavailable trial must not be silently ruled out as a possible onset. The
conditional model curve supplies a scheduled-grid timing interval; observed
rows separately determine whether a supported run was actually sampled.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .panel_g_review import first_run,timing_bounds,save_json


def scheduled_individual_timing(trials,mean,lower,upper,observed,threshold,persistence=3,
                                pre_count=None):
    if pre_count is not None and (pre_count<3 or np.asarray(observed).sum()<10):
        return {"status":"insufficient_information","estimate_trial":None,
                "earliest_compatible_trial":None,"latest_bound_trial":None,
                "first_supported_trial":None,"first_observed_supported_trial":None,
                "prediction_in_estimated_run":None,
                "timing_definition":"conditional_model_scheduled_grid_not_next_available_rows"}
    result=timing_bounds(trials,mean,lower,upper,threshold,persistence)
    result["first_observed_supported_trial"]=first_run(trials,lower,threshold,persistence,observed)
    point=result["estimate_trial"]
    if np.isfinite(point):
        mask=(np.asarray(trials)>=point)&(np.asarray(trials)<point+persistence)
        result["prediction_in_estimated_run"]=bool(not np.asarray(observed)[mask].all())
    else:result["prediction_in_estimated_run"]=None
    result["timing_definition"]="conditional_model_scheduled_grid_not_next_available_rows"
    return result


def review_individual_missingness(out,config):
    out=Path(out)
    marker=out / "individual-missingness-review.json"
    if marker.exists():return
    path=out / "individual-timing-candidates.csv"
    current=pd.read_csv(path)
    evaluable=current.loc[current.status.ne('failed')]
    if "timing_definition" in current and evaluable.timing_definition.eq(
            "conditional_model_scheduled_grid_not_next_available_rows").all():
        save_json(marker,{"status":"unified_scheduled_grid_estimator",
            "full_fit_control_validation_and_simulation_use_same_function":True,
            "failed_fits_withheld":int(current.status.eq('failed').sum()),
            "conditional_intervals_calibrated":False})
        return
    saved=out / "individual-observed-run-first-attempt.csv"
    if not saved.exists():saved.write_bytes(path.read_bytes())
    original=pd.read_csv(saved)
    trajectories=pd.read_parquet(out / "individual-training-trajectories.parquet")
    # Leave-one-control-out summaries quantify sampled-run evidence only; they
    # are retained rather than inventing unsaved LOOCV posterior trajectories.
    validation=original.loc[original.validation.notna()].copy()
    validation["timing_definition"]="leave_one_control_out_observed_run_evidence"
    rows=[]
    for (method,condition,fish),g in trajectories.groupby(["method","condition_id","fish_id"],observed=True):
        g=g.sort_values("trial_number")
        old=original.loc[(original.method==method)&(original.fish_id==fish)&original.validation.isna()]
        for fraction in config.threshold_fractions:
            row=old.loc[np.isclose(old.threshold_fraction,fraction)].iloc[0].to_dict()
            if row["status"]!="insufficient_information":
                result=scheduled_individual_timing(g.trial_number.to_numpy(),g.estimate.to_numpy(),
                    g.conditional_lower.to_numpy(),g.conditional_upper.to_numpy(),g.observed.to_numpy(bool),
                    -np.log1p(-fraction),config.persistence)
                row.update(result)
            rows.append(row)
    revised=pd.concat([pd.DataFrame(rows),validation],ignore_index=True)
    revised.replace([np.inf,-np.inf],np.nan).to_csv(path,index=False)
    save_json(marker,{"status":"exploratory_conditional_model_timing",
        "correction":"Unavailable trials remain possible onset locations. Model-supported timing and actually observed supported runs are distinct.",
        "posterior_intervals":"conditional on estimated covariance; no frequentist calibration approval",
        "legacy_LOOCV_rows":"observed-run validation summaries; not latent timing intervals",
        "original_results_preserved":str(saved),
        "prototype_information_rule":{"pre_trials":3,"training_trials":10,"approval_status":"unapproved_exploratory"}})
