"""Declared per-trial scaling for single-fish and population vigor heatmaps.

The signal is corrected conditional movement intensity: each temporal bin
summarizes metric values only while the shared detector reports movement.
Within each fish, metric, and CS trial, baseline bins before -15 s define
P10 and P90. All eligible bins are scaled against those two numbers. The
single-fish display clips to 0-1; the pooled legacy-order route first keeps
the unbounded values, averages fish, then applies a second pooled-trial
pre-CS P10/P90 scaling and clips. Missing or degenerate baselines stay missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Recording ID", "Trial type", "Trial number", "Time bin center (s)",
    "Metric ID", "Conditional intensity mean", "Valid expected fraction",
}


def scale_per_trial_conditional_vigor(
    profiles: pd.DataFrame,
    *,
    minimum_coverage: float = 0.9,
    baseline_end_s: float = -15.0,
    baseline_start_s: float | None = None,
    minimum_baseline_bins: int = 3,
    clip: bool = True,
    transform: str = "linear",
) -> pd.DataFrame:
    """Return CS profiles with one explicit vigor scale per fish/trial.

    Reference quantiles use finite, covered pre-baseline bins from the full
    profile window. Display cropping to ±20 s happens after this function.
    Set ``clip=False`` before the pooled legacy-order aggregation.
    """
    missing = REQUIRED_COLUMNS.difference(profiles.columns)
    if missing:
        raise ValueError(f"Profiles lack required columns: {sorted(missing)}")
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("Minimum coverage must be between zero and one")
    if minimum_baseline_bins < 2:
        raise ValueError("At least two baseline bins are required")
    if transform not in {"linear", "log"}:
        raise ValueError("transform must be 'linear' or 'log'")
    if baseline_start_s is not None and baseline_start_s >= baseline_end_s:
        raise ValueError("Baseline start must precede baseline end")
    selected = profiles.loc[
        profiles["Trial type"].astype(str).eq("CS")
        & profiles["Trial number"].between(5, 94)
    ].copy()
    keys = ["Recording ID", "Metric ID", "Trial number"]
    if selected.duplicated([*keys, "Time bin center (s)"]).any():
        raise ValueError("More than one profile value exists per fish/trial/time bin")
    signal = pd.to_numeric(selected["Conditional intensity mean"], errors="coerce")
    coverage = pd.to_numeric(selected["Valid expected fraction"], errors="coerce")
    selected["Eligible vigor"] = signal.where(
        np.isfinite(signal) & (coverage >= minimum_coverage)
    )
    if transform == "log":
        positive = selected["Eligible vigor"].gt(0)
        selected["Eligible vigor"] = np.log(selected["Eligible vigor"].where(positive))
    selected["Vigor transform"] = transform
    baseline_window = selected["Time bin center (s)"].lt(baseline_end_s)
    if baseline_start_s is not None:
        baseline_window &= selected["Time bin center (s)"].ge(baseline_start_s)
    baseline = selected.loc[
        baseline_window
        & selected["Eligible vigor"].notna(),
        [*keys, "Eligible vigor"],
    ]
    grouped = baseline.groupby(keys, observed=True)["Eligible vigor"]
    quantiles = grouped.quantile([0.1, 0.9]).unstack(level=-1)
    quantiles.columns = ["Baseline P10", "Baseline P90"]
    quantiles["Baseline bins"] = grouped.size()
    selected = selected.merge(quantiles.reset_index(), on=keys, how="left", validate="many_to_one")
    low = selected["Baseline P10"].to_numpy(dtype=float)
    high = selected["Baseline P90"].to_numpy(dtype=float)
    values = selected["Eligible vigor"].to_numpy(dtype=float)
    enough = selected["Baseline bins"].fillna(0).to_numpy(dtype=int) >= minimum_baseline_bins
    valid = enough & np.isfinite(values) & np.isfinite(low) & np.isfinite(high) & (high > low)
    scaled = np.full(len(selected), np.nan, dtype=float)
    scaled[valid] = (values[valid] - low[valid]) / (high[valid] - low[valid])
    if clip:
        scaled[valid] = np.clip(scaled[valid], 0, 1)
    selected["Per-trial scaled vigor"] = scaled
    selected["Baseline start (s)"] = baseline_start_s
    selected["Baseline end (s)"] = baseline_end_s
    selected["Scale valid"] = enough & np.isfinite(low) & np.isfinite(high) & (high > low)
    return selected


def summarize_equal_fish_scaled_vigor(
    scaled: pd.DataFrame,
    *,
    metric_id: str,
    condition_by_recording: dict[str, str],
) -> pd.DataFrame:
    """Average one scaled trial/bin value per fish; retain fish coverage."""
    selected = scaled.loc[
        scaled["Metric ID"].astype(str).eq(metric_id)
        & scaled["Time bin center (s)"].between(-20, 20)
    ].copy()
    if selected.empty:
        raise ValueError(f"No scaled vigor for metric {metric_id}")
    selected["condition_id"] = selected["Recording ID"].astype(str).map(condition_by_recording)
    if selected["condition_id"].isna().any():
        raise ValueError("Scaled vigor includes recordings absent from the cohort")
    fish_counts = selected.groupby("condition_id", observed=True)["Recording ID"].nunique()
    group = (
        selected.groupby(["condition_id", "Trial number", "Time bin center (s)"], observed=True)
        ["Per-trial scaled vigor"]
        .agg([("Mean per-trial scaled vigor", "mean"), ("Contributing fish", "count")])
        .reset_index()
    )
    group["Total cohort fish"] = group["condition_id"].map(fish_counts).astype(int)
    group["Fish coverage fraction"] = group["Contributing fish"] / group["Total cohort fish"]
    group["Metric ID"] = metric_id
    group["Signal semantics"] = "conditional_movement_intensity_per_trial_pre_minus15_p10_p90"
    return group


def summarize_legacy_pooled_scaled_vigor(
    scaled: pd.DataFrame,
    *,
    metric_id: str,
    condition_by_recording: dict[str, str],
    minimum_baseline_bins: int = 3,
) -> pd.DataFrame:
    """Pool unbounded fish-scaled vigor, then rescale each pooled trial.

    The historical pooled script first averaged movement-only fish values in
    each trial/time bin, then applied a second P10/P90 scale across the pooled
    pre-CS bins of that trial. This corrected-profile analogue preserves that
    operation order. It uses fish bin means rather than old frame/bin medians.
    """
    if minimum_baseline_bins < 2:
        raise ValueError("At least two baseline bins are required")
    selected = scaled.loc[
        scaled["Metric ID"].astype(str).eq(metric_id)
        & scaled["Time bin center (s)"].ge(-20)
        & scaled["Time bin center (s)"].lt(20)
    ].copy()
    if selected.empty:
        raise ValueError(f"No scaled vigor for metric {metric_id}")
    selected["condition_id"] = selected["Recording ID"].astype(str).map(condition_by_recording)
    if selected["condition_id"].isna().any():
        raise ValueError("Scaled vigor includes recordings absent from the cohort")
    fish_counts = selected.groupby("condition_id", observed=True)["Recording ID"].nunique()
    keys = ["condition_id", "Trial number", "Time bin center (s)"]
    pooled = (
        selected.groupby(keys, observed=True)["Per-trial scaled vigor"]
        .agg([("Pooled pre-rescale vigor", "mean"), ("Contributing fish", "count")])
        .reset_index()
    )
    pooled["Total cohort fish"] = pooled["condition_id"].map(fish_counts).astype(int)
    pooled["Fish coverage fraction"] = pooled["Contributing fish"] / pooled["Total cohort fish"]
    baseline_keys = ["condition_id", "Trial number"]
    baseline = pooled.loc[
        pooled["Time bin center (s)"].lt(0)
        & pooled["Pooled pre-rescale vigor"].notna(),
        [*baseline_keys, "Pooled pre-rescale vigor"],
    ]
    grouped = baseline.groupby(baseline_keys, observed=True)["Pooled pre-rescale vigor"]
    quantiles = grouped.quantile([0.1, 0.9]).unstack(level=-1)
    quantiles.columns = ["Pooled baseline P10", "Pooled baseline P90"]
    quantiles["Pooled baseline bins"] = grouped.size()
    pooled = pooled.merge(quantiles.reset_index(), on=baseline_keys, how="left", validate="many_to_one")
    low = pooled["Pooled baseline P10"].to_numpy(dtype=float)
    high = pooled["Pooled baseline P90"].to_numpy(dtype=float)
    value = pooled["Pooled pre-rescale vigor"].to_numpy(dtype=float)
    enough = pooled["Pooled baseline bins"].fillna(0).to_numpy(dtype=int) >= minimum_baseline_bins
    valid = enough & np.isfinite(value) & np.isfinite(low) & np.isfinite(high) & (high > low)
    final = np.full(len(pooled), np.nan, dtype=float)
    final[valid] = np.clip((value[valid] - low[valid]) / (high[valid] - low[valid]), 0, 1)
    pooled["Mean per-trial scaled vigor"] = final
    pooled["Metric ID"] = metric_id
    transform = (
        set(selected["Vigor transform"].dropna().astype(str))
        if "Vigor transform" in selected else {"linear"}
    )
    if len(transform) != 1:
        raise ValueError("Pooled vigor inputs mix transforms")
    pooled["Signal semantics"] = (
        "legacy_order_fish_scaled_then_pooled_trial_pre0_p10_p90"
        if transform == {"linear"}
        else "log_vigor_then_fish_scaled_then_pooled_trial_pre0_p10_p90"
    )
    return pooled
