"""Independent, explicitly exploratory review of Figure 2 G.

This module does not edit figure selections, frozen records or cohort membership.
LogMedian is computed once on positive bout frames. Undefined vigor remains
missing; scheduled trial numbers, not the next available rows, define time.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from scipy.stats import kurtosis, skew

from classical_conditioning.analysis.inference.learning_onset import (
    LearningOnsetConfig, _fit_mixed_model, _fixed_covariance,
)

PHASE_TERMS = (
    "C(fit_phase, Treatment(reference='Pre')) + pre_trial + "
    "train_b0 + train_b1 + train_b2 + train_b3 + "
    "test_b0 + test_b1 + test_b2 + test_b3"
)
PHASE_WINDOWS = {"late_training": (55, 64), "early_testing": (65, 74),
                 "late_testing": (85, 94)}


@dataclass(frozen=True)
class ReviewConfig:
    bootstrap: int = 999
    seed: int = 20261014
    persistence: int = 3
    threshold_fractions: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    min_local_fish: int = 6
    min_bootstrap_success_fraction: float = 0.95

    def __post_init__(self):
        if self.bootstrap < 19 or self.persistence < 1 or self.min_local_fish < 2:
            raise ValueError("Invalid review resampling/persistence/coverage settings")
        if any(not 0 <= v < 1 for v in self.threshold_fractions):
            raise ValueError("Threshold fractions must be in [0,1)")


def phase_features() -> pd.DataFrame:
    trials = np.arange(5, 95)
    out = pd.DataFrame({"trial_number": trials})
    out["fit_phase"] = pd.Categorical(
        np.where(trials < 15, "Pre", np.where(trials < 65, "Train", "Test")),
        categories=["Pre", "Train", "Test"], ordered=True,
    )
    out["trial_scaled"] = (trials - 49.5) / np.std(trials)
    out["pre_trial"] = np.where(trials < 15, (trials - 9.5) / 9, 0)
    for start, end, prefix in [(15, 64, "train"), (65, 94, "test")]:
        selected = (trials >= start) & (trials <= end)
        u = (trials[selected] - start) / (end - start)
        basis = np.asarray(dmatrix(
            "bs(u, df=4, degree=3, include_intercept=False, "
            "lower_bound=0, upper_bound=1)-1", {"u": u},
        ))
        for i in range(4):
            out[f"{prefix}_b{i}"] = 0.0
            out.loc[selected, f"{prefix}_b{i}"] = basis[:, i]
    return out


def prepare_input(windows: pd.DataFrame) -> pd.DataFrame:
    required = {"fish_id", "condition_id", "trial_number", "baseline_logmedian",
                "response_logmedian"}
    if not required.issubset(windows):
        raise ValueError(f"Missing input columns: {required - set(windows)}")
    if windows.duplicated(["condition_id", "fish_id", "trial_number"]).any():
        raise ValueError("Duplicate fish/trial identity")
    if not set(windows.condition_id.astype(str)).issubset({"control", "delay"}):
        raise ValueError("This scoped review supports Control versus Delay only")
    d = windows.loc[np.isfinite(windows.baseline_logmedian)
                    & np.isfinite(windows.response_logmedian)].copy()
    d = d.sort_values(["condition_id", "fish_id", "trial_number"])
    d["z"] = d.response_logmedian - d.baseline_logmedian
    d["fish_key"] = d.condition_id.astype(str) + "::" + d.fish_id.astype(str)
    d["condition_id"] = pd.Categorical(d.condition_id, ["control", "delay"])
    return d.merge(phase_features(), on="trial_number", validate="many_to_one")


def formula(target: str) -> str:
    if target not in {"difference", "adjusted_response"}:
        raise ValueError(target)
    response = "z" if target == "difference" else "response_logmedian"
    adjustment = "" if target == "difference" else "baseline_logmedian + "
    return (f"{response} ~ {adjustment}"
            f"C(condition_id, Treatment(reference='control')) * ({PHASE_TERMS})")


def fit(data: pd.DataFrame, target: str, *, group="fish_key", random="1 + trial_scaled",
        extended=True, optimizer="lbfgs", backend="fast"):
    if backend=="fast":
        from .panel_g_fast_lmm import fast_fit
        return fast_fit(data,formula(target),group=group,random=random,optimizer=optimizer)
    if backend!="statsmodels":raise ValueError(backend)
    config = LearningOnsetConfig(metric_id="legacy_distal_angular_speed",
        random_effects_formula=random, allow_random_intercept_fallback=False,
        optimizer=optimizer)
    return _fit_mixed_model(data, formula=formula(target), config=config,
        group_column=group, collect_extended_diagnostics=extended)


def fit_with_retry(data, target, **kwargs):
    """Retry numerical optimization only; retain every diagnostic/structure."""
    attempts = []
    for optimizer in ("lbfgs", "powell"):
        result, diagnostic = fit(data, target, optimizer=optimizer, **kwargs)
        attempts.append(diagnostic)
        if result is not None:
            return result, {**diagnostic, "numerical_attempts": attempts}
    return None, {**diagnostic, "numerical_attempts": attempts}


def design_for(result, target: str, baseline_reference: float):
    features = phase_features()
    designs = []
    for condition in ["control", "delay"]:
        cases = features.copy()
        cases["condition_id"] = condition
        cases["baseline_logmedian"] = baseline_reference
        designs.append(np.asarray(build_design_matrices(
            [result.model.data.design_info], cases)[0]))
    difference = designs[0] - designs[1]
    change = difference - difference[:10].mean(axis=0)
    summaries = np.vstack([change[(features.trial_number >= a)
                                 & (features.trial_number <= b)].mean(axis=0)
                           for a, b in PHASE_WINDOWS.values()])
    return features, np.vstack([change, summaries]), designs


def estimate(result, matrix):
    values = matrix @ np.asarray(result.fe_params)
    cov = _fixed_covariance(result)
    errors = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", matrix, cov, matrix), 0))
    return values, errors


def first_run(trials, values, threshold, persistence=3, eligible=None):
    """Return a scheduled-grid crossing or infinity, without dropping gaps."""
    trials, values = np.asarray(trials), np.asarray(values)
    order = np.argsort(trials)
    trials, values = trials[order], values[order]
    eligible = np.ones(len(trials), bool) if eligible is None else np.asarray(eligible)[order]
    for start in range(len(trials) - persistence + 1):
        stop = start + persistence
        ts = trials[start:stop]
        if (ts[0] >= 15 and ts[-1] <= 64 and np.all(np.diff(ts) == 1)
            and np.all(eligible[start:stop]) and np.all(np.isfinite(values[start:stop]))
            and np.all(values[start:stop] > threshold)):
            return float(ts[0])
    return float("inf")


def timing_bounds(trials, estimate_, lower, upper, threshold, persistence=3, eligible=None):
    possible = first_run(trials, upper, threshold, persistence, eligible)
    point = first_run(trials, estimate_, threshold, persistence, eligible)
    supported = first_run(trials, lower, threshold, persistence, eligible)
    if not possible <= point <= supported:
        raise ValueError("Timing bounds are not ordered")
    return {"estimate_trial": point, "earliest_compatible_trial": possible,
            "latest_bound_trial": supported, "first_supported_trial": supported,
            "status": "bounded" if np.isfinite(supported) else
                      "crossing_uncertain" if np.isfinite(point) else "not_localized"}


def resample(data, rng):
    frames = []
    for condition, group in data.groupby("condition_id", observed=True):
        ids = group.fish_key.unique()
        for i, fish in enumerate(rng.choice(ids, len(ids), replace=True)):
            rows = group.loc[group.fish_key == fish].copy()
            rows["resampled_key"] = f"{condition}::{i}::{fish}"
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


_BOOTSTRAP_CACHE={}


def cached_bootstrap_fit(data,target,seed):
    """Integer cluster multiplicities are exactly duplicated independent fish."""
    import hashlib
    from .panel_g_fast_lmm import prepare_fast,fast_fit
    columns=['fish_key','condition_id','trial_number','z','baseline_logmedian','response_logmedian',
             'trial_scaled','fit_phase','pre_trial',*[f'{phase}_b{i}' for phase in ['train','test'] for i in range(4)]]
    frame=data.copy(deep=False);frame.attrs={}
    signature=hashlib.sha256(pd.util.hash_pandas_object(frame[columns],index=False).to_numpy().tobytes()).hexdigest()
    key=(target,signature)
    if key not in _BOOTSTRAP_CACHE:
        _BOOTSTRAP_CACHE.clear();_BOOTSTRAP_CACHE[key]=prepare_fast(frame,formula(target))
    prepared=_BOOTSTRAP_CACHE[key];rng=np.random.default_rng(seed)
    weights=np.zeros(len(prepared.keys),int)
    for condition in ['control','delay']:
        positions=np.flatnonzero(prepared.conditions==condition)
        if len(positions):
            choices=rng.choice(positions,len(positions),replace=True)
            weights+=np.bincount(choices,minlength=len(weights))
    attempts=[]
    for optimizer in ['lbfgs','powell']:
        fitted,diagnostic=fast_fit(frame,formula(target),prepared=prepared,weights=weights,optimizer=optimizer)
        attempts.append(diagnostic)
        if fitted is not None:return fitted,{**diagnostic,'numerical_attempts':attempts}
    return None,{**diagnostic,'numerical_attempts':attempts}


def bootstrap_worker(payload):
    """One independently seeded whole-fish refit; failures are never no-onset."""
    data, target, seed, baseline_reference = payload
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        try:result, diagnostic = cached_bootstrap_fit(data,target,seed)
        except (ValueError,np.linalg.LinAlgError) as error:
            return {'success':False,'diagnostic':{'error':str(error)}}
        if result is None:
            return {"success": False, "diagnostic": diagnostic}
        _, matrix, _ = design_for(result, target, baseline_reference)
        values, errors = estimate(result, matrix)
        if not np.isfinite(values).all() or not np.all(errors > 1e-10):
            return {"success": False, "diagnostic": {"error": "nonfinite contrast or SE"}}
        return {"success": True, "estimate": values, "se": errors,
                "optimizer":diagnostic["optimizer"],"diagnostic":diagnostic}


def simultaneous_band(values, errors, draws):
    accepted = [d for d in draws if d["success"]]
    if not accepted:
        return np.full_like(values, np.nan), np.full_like(values, np.nan), np.nan, []
    maxima = np.array([np.max(np.abs((d["estimate"] - values) / d["se"]))
                       for d in accepted])
    critical = float(np.quantile(maxima, .95))
    return values - critical * errors, values + critical * errors, critical, maxima


def residual_diagnostics(result, data):
    out = data[["condition_id", "fish_id", "fish_key", "trial_number", "fit_phase"]].copy()
    out["residual"] = np.asarray(result.resid)
    out["fitted"] = np.asarray(result.fittedvalues)
    shifted = out.copy()
    shifted.trial_number -= 1
    adjacent = out.merge(shifted, on="fish_key trial_number".split(), suffixes=("_now", "_next"))
    correlations = []
    for _, g in adjacent.groupby("fish_key", observed=True):
        if len(g) > 3 and g.residual_now.std() > 0 and g.residual_next.std() > 0:
            correlations.append(g.residual_now.corr(g.residual_next))
    return out, {"residual_skewness": float(skew(out.residual)),
                 "residual_excess_kurtosis": float(kurtosis(out.residual)),
                 "median_true_adjacent_lag1": float(np.median(correlations)) if correlations else None,
                 "residual_sd_by_condition_phase": {
                     f"{c}/{p}": float(g.residual.std()) for (c,p), g in
                     out.groupby(["condition_id", "fit_phase"], observed=True)},
                 "adequacy_status": "requires_scientific_review"}


def fish_phase_effects(data):
    rows = []
    for (condition, fish), g in data.groupby(["condition_id", "fish_id"], observed=True):
        pre = g.loc[g.trial_number.between(5,14), "z"]
        for name, (a,b) in PHASE_WINDOWS.items():
            phase = g.loc[g.trial_number.between(a,b), "z"]
            rows.append({"condition_id": str(condition), "fish_id": str(fish),
                         "contrast": name, "pre_trials": len(pre), "phase_trials": len(phase),
                         "suppression_change": float(pre.mean()-phase.mean())
                             if len(pre) and len(phase) else np.nan})
    return pd.DataFrame(rows)


def clean_json(value):
    if isinstance(value, dict): return {str(k): clean_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray): return clean_json(value.tolist())
    if isinstance(value, (float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    return value


def save_json(path: Path, value: Any):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+"\n")
