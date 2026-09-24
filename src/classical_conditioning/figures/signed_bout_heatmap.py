"""Shared signed bout-log-vigor panel data for example and pooled heatmaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from classical_conditioning.analysis.temporal_profiles import _signed_bout_log_vigor
from classical_conditioning.figures.example_traces import METRIC_COLUMNS


WINDOW_S = (-20.0, 20.0)
BIN_WIDTH_S = 0.5
SIGNAL = "signed_bout_log_vigor_pre20_median"


def calculate_fish_heatmaps(
    metrics: pd.DataFrame,
    movement: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    recording_id: str,
    baseline_end_s: float = 0.0,
    metric_ids: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Calculate one signed value per CS trial and half-second bin."""
    if not WINDOW_S[0] < baseline_end_s <= 0:
        raise ValueError("Baseline end must be after -20 s and no later than CS onset")
    metric_ids = tuple(METRIC_COLUMNS) if metric_ids is None else metric_ids
    if not metric_ids or any(metric not in METRIC_COLUMNS for metric in metric_ids):
        raise ValueError("At least one known metric is required")
    absolute = metrics["AbsoluteTime"].to_numpy(dtype=np.int64)
    if np.any(np.diff(absolute) < 0):
        raise ValueError(f"Nonchronological frame times for {recording_id}")
    if len(metrics) != len(movement) or not np.array_equal(
        metrics[["FrameID", "AbsoluteTime"]].to_numpy(),
        movement[["FrameID", "AbsoluteTime"]].to_numpy(),
    ):
        raise ValueError(f"Metric and movement frames do not align for {recording_id}")
    detector_valid = movement["valid"].to_numpy(dtype=bool)
    moving = movement["moving"].to_numpy(dtype=bool)
    bout_ids = movement["bout_id"].to_numpy(dtype=np.int32)
    metric_values = {
        metric_id: metrics[METRIC_COLUMNS[metric_id]].to_numpy(dtype=float)
        for metric_id in metric_ids
    }
    bins = np.arange(WINDOW_S[0], WINDOW_S[1], BIN_WIDTH_S) + BIN_WIDTH_S / 2
    rows = []
    for trial in range(5, 95):
        onset = int(cycles.iloc[trial - 1]["Beg"])
        start = np.searchsorted(absolute, onset + int(WINDOW_S[0] * 1000), side="left")
        stop = np.searchsorted(absolute, onset + int(WINDOW_S[1] * 1000), side="left")
        seconds = (absolute[start:stop] - onset) / 1000.0
        indices = np.floor((seconds - WINDOW_S[0]) / BIN_WIDTH_S).astype(np.int32)
        in_window = (indices >= 0) & (indices < len(bins))
        indices = indices[in_window]
        for metric_id, values in metric_values.items():
            signal = _signed_bout_log_vigor(
                values[start:stop][in_window], seconds[in_window], indices,
                detector_valid[start:stop][in_window], moving[start:stop][in_window],
                bout_ids[start:stop][in_window], bin_count=len(bins),
                baseline_end_s=baseline_end_s,
            )
            rows.extend({
                "Recording ID": recording_id,
                "Metric ID": metric_id,
                "Trial number": trial,
                "Time bin center (s)": float(time),
                "Signed log vigor": float(value),
                "Baseline start (s)": WINDOW_S[0],
                "Baseline end (s)": baseline_end_s,
            } for time, value in zip(bins, signal))
    return pd.DataFrame(rows)


def summarize_equal_fish_signed_log_vigor(
    fish_bins: pd.DataFrame,
    *,
    metric_id: str,
    condition_by_recording: dict[str, str],
) -> pd.DataFrame:
    """Pool fish-level signed bins without a second normalization stage."""
    selected = fish_bins.loc[fish_bins["Metric ID"].eq(metric_id)].copy()
    if selected.empty:
        raise ValueError(f"No signed vigor for metric {metric_id}")
    if selected.duplicated(["Recording ID", "Trial number", "Time bin center (s)"]).any():
        raise ValueError("Duplicate fish/trial/time bin")
    if not selected["Baseline start (s)"].eq(WINDOW_S[0]).all() or not selected[
        "Baseline end (s)"
    ].eq(0.0).all():
        raise ValueError("Pooled heatmap requires the -20 to 0 s baseline")
    selected["condition_id"] = selected["Recording ID"].map(condition_by_recording)
    if selected["condition_id"].isna().any():
        raise ValueError("Signed vigor includes recordings absent from the cohort")
    cohort_counts = selected.groupby("condition_id")["Recording ID"].nunique()
    pooled = (
        selected.groupby(["condition_id", "Trial number", "Time bin center (s)"], observed=True)
        ["Signed log vigor"]
        .agg([("Mean signed log vigor", "mean"), ("Contributing fish", "count")])
        .reset_index()
    )
    pooled["Total cohort fish"] = pooled["condition_id"].map(cohort_counts).astype(int)
    pooled["Fish coverage fraction"] = pooled["Contributing fish"] / pooled["Total cohort fish"]
    pooled["Metric ID"] = metric_id
    pooled["Signal semantics"] = SIGNAL
    return pooled
