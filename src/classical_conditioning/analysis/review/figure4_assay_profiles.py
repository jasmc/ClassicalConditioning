"""Assay-bounded extension of the Delay CS-time spline LMM adapter.

This pilot conditions on saved legacy groups. Its refit bootstrap describes
estimator variability; it does not validate learner labels or peak timing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from classical_conditioning.analysis.inference.panel_g_fast_lmm import prepare_fast, fast_fit
from classical_conditioning.analysis.review.figure4_profiles import (
    GROUPS, CONTRASTS, prediction_design, differences, peaks, stratified_weights,
)

@dataclass(frozen=True)
class ProfileSpec:
    lower: float = -5.
    upper: float = 16.

    def __post_init__(self):
        if not np.isfinite([self.lower, self.upper]).all() or not self.lower < 0 < self.upper:
            raise ValueError("Finite ordered bounds spanning CS onset are required")

    @property
    def formula(self):
        return ("signed ~ C(display_group) * bs(time_s, df=6, degree=3, "
                f"include_intercept=False, lower_bound={self.lower}, upper_bound={self.upper})")

TRACE3 = ProfileSpec()



def model_data(fish_bins, spec=TRACE3):
    data = fish_bins.loc[fish_bins.display_group.isin(GROUPS)
                         & fish_bins.time_s.between(spec.lower, spec.upper, inclusive="neither")
                         & np.isfinite(fish_bins.signed)].copy()
    if data.duplicated(["fish_id", "time_s"]).any():
        raise ValueError("Each fish must have at most one catch summary per time bin")
    data["display_group"] = pd.Categorical(data.display_group, categories=GROUPS)
    data["fish_key"] = data.fish_id
    # Backend name is historical: this covariate represents CS-relative seconds.
    data["trial_scaled"] = data.time_s / spec.upper
    return data.sort_values(["display_group", "fish_id", "time_s"]).reset_index(drop=True)


def fit(data, random="1 + trial_scaled", prepared=None, weights=None, spec=TRACE3):
    attempts = []
    for optimizer in ("lbfgs", "powell"):
        result, diagnostics = fast_fit(data, spec.formula, random=random, prepared=prepared,
                                       weights=weights, optimizer=optimizer)
        attempts.append(diagnostics)
        if result is not None:
            return result, attempts
    return None, attempts


def fit_bootstrap(data, times, repetitions=399, seed=20261022, progress=None, spec=TRACE3):
    mapping = data.groupby("fish_id", observed=True).display_group.first().astype(str).to_dict()
    attempts, structures = [], []
    # A failed full fit or <95% stable refits rejects the entire structure.
    # Restart the complete bootstrap for the reduced model; never mix models.
    for random in ("1 + trial_scaled", "1"):
        result, history = fit(data, random=random, spec=spec)
        attempts += history
        original_history = history
        if result is None:
            structures.append({"random": random, "status": "full fit failed", "full_fit_attempts": history})
            continue
        prepared = prepare_fast(data, spec.formula, random=random)
        grid, matrix = prediction_design(result, times)
        estimates = (matrix @ result.fe_params.to_numpy()).reshape(len(GROUPS), -1)
        rng = np.random.default_rng(seed)
        curves, diagnostics = [], []
        for replicate in range(repetitions):
            weights = stratified_weights(prepared.keys, mapping, rng)
            fitted, history = fit(data, random=random, prepared=prepared, weights=weights, spec=spec)
            diagnostics.append({"replicate": replicate, "success": fitted is not None,
                                "attempts": history})
            if fitted is not None:
                curves.append((matrix @ fitted.fe_params.to_numpy()).reshape(len(GROUPS), -1))
            if progress and ((replicate + 1) % 25 == 0 or replicate + 1 == repetitions):
                progress(replicate + 1, len(curves))
        accepted = len(curves) >= np.ceil(.95 * repetitions)
        structures.append({"random": random, "accepted": len(curves), "requested": repetitions,
                           "status": "accepted" if accepted else "unstable refit covariance",
                           "full_fit_attempts": original_history, "bootstrap_fits": diagnostics})
        if accepted:
            break
    else:
        raise ValueError("Neither declared structure supports stable bootstrap refits: " + str(structures))
    draws = np.asarray(curves)
    return result, grid, estimates, draws, {"random": random, "full_fit_attempts": attempts,
        "bootstrap_requested": repetitions, "bootstrap_accepted": len(curves),
        "structure_comparison": structures, "seed": seed, "formula": spec.formula, "fit_window_s": [spec.lower, spec.upper],
        "covariance": "Gaussian fish effects " + random + "; independent residuals conditional on effects",
        "limitations": ["Exploratory; labels fixed and retrospective", "No population or peak calibration completed",
                        "Fish bootstrap preserves each observed profile; it cannot repair informative bout availability",
                        "Fixed basis spline LMM, not a completed penalized GAMM comparison"]}


def individual_predictions(result, data, times, random, residual_variance, spec=TRACE3):
    """Conditional means; no calibrated individual intervals are implied."""
    grid, matrix = prediction_design(result, times)
    means = (matrix @ result.fe_params.to_numpy()).reshape(len(GROUPS), -1)
    residual = data.signed.to_numpy() - result.model.exog @ result.fe_params.to_numpy()
    covariance = np.asarray(result.cov_re)
    q = len(covariance)
    predicted = {}
    for fish, positions in data.groupby("fish_id", sort=False).indices.items():
        part = data.iloc[positions]
        z = np.ones((len(part), 1)) if q == 1 else np.column_stack([np.ones(len(part)), part.trial_scaled])
        gamma = np.linalg.solve(np.eye(q) + covariance @ (z.T @ z) / residual_variance,
                                covariance @ z.T @ residual[positions] / residual_variance)
        zg = np.ones((len(times), 1)) if q == 1 else np.column_stack([np.ones(len(times)), np.asarray(times) / spec.upper])
        group = GROUPS.index(str(part.display_group.iloc[0]))
        predicted[fish] = means[group] + zg @ gamma
    return predicted
