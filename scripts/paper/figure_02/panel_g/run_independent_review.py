"""Run the isolated Figure 2 G analysis without changing any selected figure.

Run with .venv/bin/python. All numerical payloads and review media are written
to the explicit new external output directory, never to a frozen source.
"""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import html
import io
import json
import os
from pathlib import Path, PureWindowsPath
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = _COMMAND_REPOSITORY
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from classical_conditioning.analysis.inference.panel_g_review import (
    ReviewConfig, PHASE_WINDOWS, bootstrap_worker, clean_json, design_for, estimate,
    first_run, fit, fit_with_retry, fish_phase_effects, phase_features, prepare_input,
    residual_diagnostics, save_json, simultaneous_band, timing_bounds,
)
from classical_conditioning.inventory import build_recording_inventory

FREEZE = ROOT / "records/frozen-analyses/figure-02/panel-g/2026-10-09-historical-log-median"
MOUNT = Path("/Volumes/JOAQUIM")
FROZEN_EXTERNAL = MOUNT / "ClassicalConditioning-Mac-20261009/pc-outputs/figure2-assembly/frozen/20261009-G-historical-logmedian"
ANALYSIS_ROOT = MOUNT / 'ClassicalConditioning Outputs/figures/figure-02/panel-g/versions/figure2-G-independent-review'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""): digest.update(block)
    return digest.hexdigest()


def mapped(path):
    value = Path(path)
    if value.exists(): return value
    return MOUNT.joinpath(*PureWindowsPath(path).parts[1:])


def authenticate(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"Missing or changed authenticated input: {path}")
    return {"path": str(path), "sha256": expected}


def protected_files():
    paths = [ROOT / "configs/paper-figures/selections/figure2-G-logmedian-freeze-20261009.json",
             FREEZE / "manifests/freeze.json", FREEZE / "manifests/scientific-selection.json",
             FREEZE / "sources/selected-source-inventory.json",
             FROZEN_EXTERNAL / "Fig2_G_Historical_LogMedian.svg"]
    return [{"path": str(p), "sha256": sha(p)} for p in paths]


def verify_protected(out):
    for record in json.loads((out / "protected-originals.json").read_text()):
        authenticate(Path(record["path"]), record["sha256"])


def prepare(out):
    records = json.loads((FREEZE / "sources/selected-source-inventory.json").read_text())
    manifest_record = next(r for r in records if r["path"].endswith("\\frame-source-manifest.json"))
    original_manifest = FROZEN_EXTERNAL / "data/frame-source-manifest.json"
    authenticate(original_manifest, manifest_record["sha256"])
    sources = json.loads(original_manifest.read_text())
    grouped = {}
    for item in sources:
        original = PureWindowsPath(item["path"])
        grouped.setdefault(original.parent.name, {})[original.name] = item
    frozen_record = next(r for r in records if r["path"].endswith("\\fish-logmedian-differences.parquet"))
    # A byte-identical replica is selected by hash, not by directory date.
    replica = MOUNT / 'ClassicalConditioning Outputs/figures/figure-02/panels-g-h-i/versions/row3-trial-ratio-review/20261009T145647405899Z-delay-logmedian/fish-logmedian-differences.parquet'
    authenticate(replica, frozen_record["sha256"])
    frozen_values = pd.read_parquet(replica)
    inventory = build_recording_inventory(MOUNT / "Raw Data/allDelay", hash_files=False,
                                          inspect_tracking_headers=True)
    save_json(out / "recording-inventory.json", inventory)
    lineage = [authenticate(original_manifest, manifest_record["sha256"]),
               authenticate(replica, frozen_record["sha256"])]
    rows, qc, legacy = [], [], []
    for index, (fish, files) in enumerate(sorted(grouped.items())):
        processed = MOUNT / "Digested Data/allDelay-full-v1/Processed data" / fish
        quality = MOUNT / "Digested Data/allDelay-full-v1/Quality checks" / fish
        summary_path = quality / "candidate-trial-outcomes-corrected-v1_summary.json"
        metadata = json.loads(summary_path.read_text())
        lineage.append({"path": str(summary_path), "sha256": sha(summary_path)})
        trial_path = processed / "candidate-trial-outcomes-corrected-v1.parquet"
        lineage.append(authenticate(trial_path, metadata["artifacts"]["outcomes"]["sha256"]))
        trials = pd.read_parquet(trial_path)
        trials = trials.loc[trials.metric_id.eq("legacy_distal_angular_speed")
                            & trials.alignment.eq("CS") & trials.trial_number.between(5,94)]
        if len(trials) != 90 or trials.trial_number.nunique() != 90:
            raise ValueError(f"Unexpected scheduled trial map: {fish}")
        frame_path = mapped(files["frame_activity_candidates-corrected-v1.parquet"]["path"])
        movement_path = mapped(files["movement_state_candidates-corrected-v2.parquet"]["path"])
        lineage.append(authenticate(frame_path, files[frame_path.name]["sha256"]))
        lineage.append(authenticate(movement_path, files[movement_path.name]["sha256"]))
        if metadata["inputs"]["candidate_frames"] != files[frame_path.name]["sha256"] or metadata["inputs"]["movement_state"] != files[movement_path.name]["sha256"]:
            raise ValueError(f"Outcome/frame lineage disagrees: {fish}")
        protocol_path = processed / "stimulus_events.parquet"
        lineage.append(authenticate(protocol_path, metadata["inputs"]["protocol"]))
        protocol = pd.read_parquet(protocol_path)
        us = protocol.loc[protocol.Type.astype(str).eq("Reinforcer")].sort_values("Beg")
        frames = pd.read_parquet(frame_path, columns=["FrameID", "AbsoluteTime", "FrameStep", "legacy_distal_angular_speed_rad_per_ms"])
        movement = pd.read_parquet(movement_path, columns=["FrameID", "AbsoluteTime", "valid", "moving", "bout_id"])
        np.testing.assert_array_equal(frames.FrameID, movement.FrameID)
        np.testing.assert_array_equal(frames.AbsoluteTime, movement.AbsoluteTime)
        np.testing.assert_array_equal(movement.moving.to_numpy(bool), movement.bout_id.to_numpy()>0)
        time = frames.AbsoluteTime.to_numpy(np.int64)
        if np.any(np.diff(time)<0): raise ValueError(f"Nonmonotonic frame time: {fish}")
        value = frames.legacy_distal_angular_speed_rad_per_ms.to_numpy(float)
        technical = movement.valid.to_numpy(bool) & frames.FrameStep.eq(1).to_numpy() & np.isfinite(value)
        moving = movement.moving.to_numpy(bool)
        bout = technical & moving
        bout_times = time[bout]
        for _, trial in trials.sort_values("trial_number").iterrows():
            start = int(trial.event_start_absolute_time_ms)
            limits = [start-15000, start, start+9000]
            a,b,c = np.searchsorted(time, limits, side="left")
            row = trial.to_dict()
            for prefix, left, right in [("baseline",a,b),("response",b,c)]:
                values = value[left:right][bout[left:right]]
                positive = values[values>0]
                row[prefix+"_logmedian"] = float(np.median(np.log(positive))) if len(positive) else np.nan
                row[prefix+"_positive_frames"] = len(positive)
                row[prefix+"_observed_frames"] = int(right-left)
                row[prefix+"_technical_valid_frames"] = int(technical[left:right].sum())
                row[prefix+"_bout_frames"] = len(values)
                expected = trial.baseline_conditional_intensity if prefix=="baseline" else trial.conditional_intensity
                np.testing.assert_allclose(np.mean(values) if len(values) else np.nan,
                                           expected, rtol=1e-10, atol=1e-12, equal_nan=True)
            reasons = []
            for prefix in ["baseline","response"]:
                if row[prefix+"_technical_valid_frames"]==0: reasons.append(prefix+"_no_valid_frames")
                elif row[prefix+"_bout_frames"]==0: reasons.append(prefix+"_no_bout")
                elif row[prefix+"_positive_frames"]==0: reasons.append(prefix+"_no_positive_bout_vigor")
            row["eligibility_reason"] = ";".join(reasons) or "eligible"
            row["baseline_us_overlap"] = int(((us.Beg<start)&(us.End>start-15000)).sum())
            row["response_us_overlap"] = int(((us.Beg<start+9000)&(us.End>start)).sum())
            # Count actual prior long pairings within the Delay CS->US window;
            # short independent pulses are not inferred to be conditioning pairs.
            row["eligible"] = not reasons
            rows.append(row)
        observed_us_bouts = []
        for number, event in enumerate(us.itertuples(index=False),start=1):
            left,right = np.searchsorted(bout_times,[int(event.Beg),int(event.Beg)+5000],side="left")
            observed_us_bouts.append({"fish_id": fish,"us_ordinal":number,
                "raw_us_duration_ms":int(event.End-event.Beg),"bout_in_following_5s":bool(right>left),
                "status":"behavioral_projection_not_technical_exclusion"})
        legacy.extend(observed_us_bouts)
        acquisition_path = quality / "acquisition_summary.json"
        preprocessing_path = quality / "corrected-v1_preprocessing_summary.json"
        acquisition = json.loads(acquisition_path.read_text()) if acquisition_path.exists() else {}
        preprocessing = json.loads(preprocessing_path.read_text()) if preprocessing_path.exists() else {}
        for path in [acquisition_path,preprocessing_path]:
            if path.exists():lineage.append({"path":str(path),"sha256":sha(path),"scope":"technical evidence snapshot; not separately freeze-bound"})
        qc.append({"fish_id":fish,"condition_id":str(trials.condition_id.iloc[0]),
            "membership":"retained_selected_G_scope_no_new_exclusion",
            "acquisition_status":acquisition.get("status","unavailable"),
            "events_outside_acquisition":preprocessing.get("protocol_timing",{}).get("events_outside_acquisition"),
            "matched_frames":preprocessing.get("matched_frame_count"),
            "frame_valid_count":preprocessing.get("frame_valid_count"),
            "technical_policy_status":"unapproved_audit_only",
            "legacy_last_us_status":"not_evaluable_original_frame_annotated_end_semantics",
            "original_preprocessing_discard_default":False,
            "original_logmedian_apply_discard_default":False})
        del frames,movement,time,value,technical,moving,bout,bout_times
        gc.collect()
        print(f"Authenticated/rebuilt {index+1}/{len(grouped)} fish: {fish}",flush=True)
    windows = pd.DataFrame(rows)
    data = prepare_input(windows)
    merged = data[["condition_id","fish_id","trial_number","z"]].merge(
        frozen_values, on=["condition_id","fish_id","trial_number"],validate="one_to_one",how="outer",indicator=True)
    if not merged._merge.eq("both").all(): raise ValueError("Rebuilt eligibility differs from frozen G")
    np.testing.assert_allclose(merged.z,merged.ratio,rtol=0,atol=1e-12)
    windows.to_parquet(out / "scheduled-window-outcomes.parquet", index=False)
    data.to_parquet(out / "model-input.parquet", index=False)
    pd.DataFrame(qc).to_csv(out / "fish-technical-audit.csv",index=False)
    pd.DataFrame(legacy).to_csv(out / "us-bout-projection.csv",index=False)
    coverage = windows.groupby(["condition_id","fish_id","block_10_name"],observed=True).agg(
        scheduled=("trial_number","size"),eligible=("eligible","sum"),
        baseline_valid=("baseline_technical_valid_frames","sum"),response_valid=("response_technical_valid_frames","sum"),
        baseline_us_overlap=("baseline_us_overlap","sum"),response_us_overlap=("response_us_overlap","sum")).reset_index()
    coverage.to_csv(out / "fish-block-coverage.csv",index=False)
    save_json(out / "input-lineage.json",lineage)
    save_json(out / "input-verification.json",{"scheduled":len(windows),"eligible":len(data),
        "fish":data.fish_key.nunique(),"frozen_G_numeric_match":True,"maximum_difference":float(abs(merged.z-merged.ratio).max()),
        "membership_approval":"selected G scope only; not a newly approved whole-paper cohort",
        "complete_recording_inventory":"recording-inventory.json; extra records remain explicit and are not silently added"})


def bootstrap(data,target,config,workers):
    seeds=np.random.SeedSequence(config.seed).generate_state(config.bootstrap)
    baseline=float(data.baseline_logmedian.mean())
    payloads=[(data,target,int(seed),baseline) for seed in seeds]
    draws=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,item in enumerate(pool.map(bootstrap_worker,payloads,chunksize=1)):
            draws.append(item)
            if (i+1)%50==0:print(f"{target}: bootstrap {i+1}/{config.bootstrap}",flush=True)
    return draws


def population(out,config,workers):
    data=pd.read_parquet(out / "model-input.parquet")
    counts=data.groupby(["condition_id","trial_number"],observed=True).fish_key.nunique().unstack(0).reindex(range(5,95),fill_value=0).fillna(0)
    adequate=counts.min(axis=1).to_numpy()>=config.min_local_fish
    for target in ["difference","adjusted_response"]:
        directory=out / target;directory.mkdir(exist_ok=True)
        result,diagnostic=fit_with_retry(data,target)
        save_json(directory / "fit-diagnostics.json",diagnostic)
        if result is None:
            save_json(directory / "result.json",{"status":"failed","timing":[],"error":diagnostic.get("error")});continue
        features,matrix,designs=design_for(result,target,float(data.baseline_logmedian.mean()))
        values,errors=estimate(result,matrix)
        residuals,adequacy=residual_diagnostics(result,data)
        residuals.to_parquet(directory / "residuals.parquet",index=False)
        save_json(directory / "model-adequacy.json",adequacy)
        pd.DataFrame({"term":result.fe_params.index,"estimate":result.fe_params,"se":result.bse_fe}).to_csv(directory / "coefficients.csv",index=False)
        draws=bootstrap(data,target,config,workers)
        lower,upper,critical,maxima=simultaneous_band(values,errors,draws)
        successful=sum(x["success"] for x in draws)
        band_accepted=successful>=np.ceil(config.min_bootstrap_success_fraction*config.bootstrap)
        if not band_accepted:lower[:]=np.nan;upper[:]=np.nan
        table=features[["trial_number","fit_phase"]].copy()
        table["estimate"]=values[:90];table["se"]=errors[:90]
        table["simultaneous_lower"]=lower[:90];table["simultaneous_upper"]=upper[:90]
        table["control_fish"]=counts.control.to_numpy();table["delay_fish"]=counts.delay.to_numpy()
        table["coverage_adequate"]=adequate
        for c,x in zip(["control","delay"],designs):table[c+"_fitted"]=x@np.asarray(result.fe_params)
        table.to_csv(directory / "population-trajectory.csv",index=False)
        phase=pd.DataFrame({"contrast":list(PHASE_WINDOWS),"estimate":values[90:],"se":errors[90:],
            "simultaneous_lower":lower[90:],"simultaneous_upper":upper[90:]})
        phase.to_csv(directory / "planned-phase-contrasts.csv",index=False)
        timings=[]
        if band_accepted:
            for fraction in config.threshold_fractions:
                threshold=-np.log1p(-fraction)
                timing=timing_bounds(np.arange(5,95),values[:90],lower[:90],upper[:90],threshold,config.persistence,adequate)
                timing.update(threshold_fraction=fraction,threshold_log_units=threshold,
                    threshold_status="exploratory_not_author_approved",persistence=config.persistence)
                timings.append(timing)
        np.savez_compressed(directory / "bootstrap-draws.npz",
            estimates=np.asarray([x["estimate"] for x in draws if x["success"]]),
            standard_errors=np.asarray([x["se"] for x in draws if x["success"]]),maxima=maxima)
        save_json(directory / "bootstrap-failures.json",[{"replicate":i,**x} for i,x in enumerate(draws) if not x["success"]])
        save_json(directory / "bootstrap-fit-diagnostics.json",[{"replicate":i,"success":x["success"],
            "optimizer":x.get("optimizer"),"diagnostic":x.get("diagnostic")} for i,x in enumerate(draws)])
        observed=float(np.max(abs(values/errors)))
        global_p=float((1+np.sum(np.asarray(maxima)>=observed))/(len(maxima)+1)) if band_accepted else None
        influence=[]
        for key in data.fish_key.unique():
            reduced=data.loc[data.fish_key!=key]
            omitted,diag=fit_with_retry(reduced,target,extended=False)
            row={"omitted_fish":str(key),"status":diag["diagnostic_status"]}
            row["numerical_attempts_json"]=json.dumps(clean_json(diag.get("numerical_attempts",[])))
            if omitted is not None:
                _,cm,_=design_for(omitted,target,float(data.baseline_logmedian.mean()))
                vi,_=estimate(omitted,cm)
                row["max_contrast_shift"]=float(abs(vi-values).max())
                for i,name in enumerate(PHASE_WINDOWS):row[name+"_estimate"]=float(vi[90+i])
                for fraction in config.threshold_fractions:
                    row[f"point_crossing_{fraction:g}"]=first_run(np.arange(5,95),vi[:90],-np.log1p(-fraction),config.persistence,adequate)
            influence.append(row)
        pd.DataFrame(influence).to_csv(directory / "leave-one-fish-out.csv",index=False)
        alternatives=[]
        for random in ["1"]:
            alternative,diag=fit_with_retry(data,target,random=random)
            alt={"random_effects":random,"diagnostic":diag}
            if alternative is not None:
                _,cm,_=design_for(alternative,target,float(data.baseline_logmedian.mean()))
                vi,_=estimate(alternative,cm);alt["max_contrast_shift"]=float(abs(vi-values).max())
            alternatives.append(alt)
        windows=pd.read_parquet(out / "scheduled-window-outcomes.parquet")
        clean=windows.loc[(windows.baseline_us_overlap==0)&(windows.response_us_overlap==0)]
        clean_data=prepare_input(clean)
        clean_result,clean_diag=fit_with_retry(clean_data,target)
        us_alternative={"sensitivity":"exclude_US_overlap_windows_unapproved",
                        "retained_rows":len(clean_data),"diagnostic":clean_diag}
        if clean_result is not None:
            _,cm,_=design_for(clean_result,target,float(data.baseline_logmedian.mean()))
            vi,_=estimate(clean_result,cm)
            us_alternative["phase_estimates"]=dict(zip(PHASE_WINDOWS,vi[90:]))
            us_alternative["max_contrast_shift"]=float(abs(vi-values).max())
        alternatives.append(us_alternative)
        save_json(directory / "random-effects-sensitivity.json",alternatives)
        save_json(directory / "result.json",{"status":"exploratory_candidate" if band_accepted else "band_failed",
            "bootstrap_requested":config.bootstrap,"bootstrap_successful":successful,"critical":critical,
            "joint_global_curve_and_phase_p":global_p,"timing":timings,
            "bootstrap_optimizer_policy":"lbfgs_then_powell_same_structure",
            "paper_inference_approved":False,"baseline_target_approved":False,
            "remaining_gates":["primary effect target","meaningful threshold","technical cohort policy","model adequacy","paper-scale calibration"],
            "model_adequacy":adequacy,"independence_assumption":"fish; recording/day dependence requires review"})
        print(f"{target}: complete ({successful}/{config.bootstrap} bootstrap fits)",flush=True)
    robust=fish_phase_effects(data);robust.to_csv(out / "fish-phase-effects.csv",index=False)
    rng=np.random.default_rng(config.seed+81);summaries=[]
    for name,g in robust.groupby("contrast"):
        groups=[g.loc[g.condition_id==c,"suppression_change"].dropna().to_numpy() for c in ["control","delay"]]
        if any(len(v)<2 for v in groups):continue
        samples=np.array([rng.choice(groups[1],len(groups[1]),replace=True).mean()-rng.choice(groups[0],len(groups[0]),replace=True).mean() for _ in range(5000)])
        summaries.append({"contrast":name,"estimate":groups[1].mean()-groups[0].mean(),
            "pointwise_lower":np.quantile(samples,.025),"pointwise_upper":np.quantile(samples,.975),
            "control_fish":len(groups[0]),"delay_fish":len(groups[1]),"role":"robustness_only_pointwise_not_joint_claim_family"})
    pd.DataFrame(summaries).to_csv(out / "fish-phase-robustness.csv",index=False)


def individual(out,config):
    from classical_conditioning.analysis.inference.panel_g_review_extensions import run_individual
    run_individual(out,config)


def calibrate(out,config,workers,repetitions=2,bootstrap_per_simulation=39):
    from classical_conditioning.analysis.inference.panel_g_review_extensions import run_calibration
    run_calibration(out,config,workers,repetitions=repetitions,bootstrap_per_simulation=bootstrap_per_simulation)


def stabilize(out,config,workers):
    from classical_conditioning.analysis.inference.panel_g_review_extensions import run_optimizer_stability
    run_optimizer_stability(out,config,workers)


def report(out,config):
    from classical_conditioning.analysis.inference.panel_g_review_extensions import build_report
    build_report(out,config)


def timing_review(out,config):
    from classical_conditioning.analysis.inference.panel_g_missingness import review_individual_missingness
    review_individual_missingness(out,config)


def sensitivities(out,config,workers):
    from classical_conditioning.analysis.inference.panel_g_sensitivities import run_sensitivities
    run_sensitivities(out,config,workers)


def main():
    from classical_conditioning.analysis.inference.panel_g_run import main as guarded_main
    guarded_main(sys.modules[__name__])


if __name__=="__main__":main()
