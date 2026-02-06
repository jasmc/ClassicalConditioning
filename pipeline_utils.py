"""Shared pipeline utilities used across analysis scripts 1–6.

Centralizes:
- Fish discard filtering
- Filename sanitization helpers (delegates to figure_saving)
- Selected-fish suffix management
- Pooled data loading
- Common data loading boilerplate
- Scaled vigor computation
- Normalized vigor (per-trial) computation
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import analysis_utils
import figure_saving
from general_configuration import config as gen_config


# ---------------------------------------------------------------------------
# Filename helpers (thin wrappers around figure_saving to avoid duplication)
# ---------------------------------------------------------------------------

SELECTED_FISH_SUFFIX = "_selectedFish"


def stringify_for_filename(value: object) -> str:
    """Convert common objects (lists/arrays) into filename-friendly strings."""
    return figure_saving.stringify_for_filename(value)


def sanitize_filename(name: str) -> str:
    """Sanitize a filename component for Windows filesystems."""
    return figure_saving.sanitize_filename_component(name)


def maybe_append_selected_fish_stem(stem: str, apply_discard: bool) -> str:
    """Append ``_selectedFish`` to a filename stem when discard is enabled."""
    if not apply_discard:
        return str(stem)
    stem = str(stem)
    return stem if stem.endswith(SELECTED_FISH_SUFFIX) else f"{stem}{SELECTED_FISH_SUFFIX}"


def maybe_selected_fish_path(path_out: Path | str, apply_discard: bool) -> Path:
    """Append ``_selectedFish`` to a Path stem when discard is enabled."""
    p = Path(path_out)
    if not apply_discard:
        return p
    if p.stem.endswith(SELECTED_FISH_SUFFIX):
        return p
    return p.with_name(f"{p.stem}{SELECTED_FISH_SUFFIX}{p.suffix}")


# ---------------------------------------------------------------------------
# Fish discard filtering
# ---------------------------------------------------------------------------

def filter_discarded_fish(
    df: pd.DataFrame,
    fish_ids_to_discard: list[str],
    *,
    fish_col: str = "Fish",
    source: str = "",
    apply_discard: bool = True,
) -> pd.DataFrame:
    """Drop rows whose Fish ID is in the discarded list.

    Prints unique fish count before and after discarding.

    Parameters
    ----------
    df : DataFrame
        Data to filter.
    fish_ids_to_discard : list[str]
        IDs to remove.
    fish_col : str
        Column name containing fish identifiers.
    source : str
        Label for logging.
    apply_discard : bool
        If *False*, skip filtering and return *df* unchanged.
    """
    if df is None or df.empty:
        return df

    before = df[fish_col].nunique()
    prefix = f"  [{source}] " if source else "  "
    print(f"{prefix}Fish unique before discard: {before}")

    if not apply_discard or not fish_ids_to_discard:
        return df

    df_filtered = df[~df[fish_col].isin(fish_ids_to_discard)].copy()
    after = df_filtered[fish_col].nunique()
    print(f"{prefix}Fish unique after discard: {after}")
    return df_filtered


# ---------------------------------------------------------------------------
# Common data loading helpers
# ---------------------------------------------------------------------------

def load_fish_pickle(
    fish_path: Path,
    *,
    compression: str = "gzip",
    reset_index: bool = True,
) -> pd.DataFrame | None:
    """Load a per-fish pickle with standard error handling.

    Returns *None* on failure.
    """
    try:
        data = pd.read_pickle(str(fish_path), compression=compression)
    except Exception as exc:
        print(f"Failed to read {fish_path.name}: {exc}")
        return None
    if reset_index:
        data.reset_index(drop=True, inplace=True)
    return data


def ensure_time_col(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure the data has a ``Trial time (s)`` column.

    If only the frame-based column exists, converts it.
    """
    if "Trial time (s)" not in data.columns:
        data = analysis_utils.convert_time_from_frame_to_s(data)
    return data


def densify_sparse_columns(
    data: pd.DataFrame,
    columns: Sequence[str] = ("Vigor (deg/ms)", "Bout beg", "Bout end", "Bout"),
) -> pd.DataFrame:
    """Convert SparseDtype columns to dense; silently skips failures."""
    for col in columns:
        if col in data.columns:
            try:
                data[col] = data[col].sparse.to_dense()
            except Exception:
                pass
    return data


# ---------------------------------------------------------------------------
# Scaled vigor computation  (used by scripts 1 & 2)
# ---------------------------------------------------------------------------

def compute_scaled_vigor_for_heatmap(
    data: pd.DataFrame,
    *,
    vigor_col: str = "Vigor (deg/ms)",
    bout_col: str = "Bout",
    bout_beg_col: str = "Bout beg",
    bout_end_col: str = "Bout end",
    time_col: str = "Trial time (s)",
    trial_col: str = "Trial number",
    block_col: str = "Block name",
    baseline_window: float | None = None,
    quantile_range: tuple[float, float] = (0.1, 0.9),
) -> pd.DataFrame:
    """Compute bout-mean replaced, baseline-scaled vigor for heatmap display.

    Steps per trial:
    1. Set vigor to NaN outside bouts.
    2. Replace each bout's vigor with the bout-mean vigor.
    3. Scale vigor to [0, 1] using baseline quantiles.

    Returns a *copy* of the input with the vigor column overwritten.
    """
    if baseline_window is None:
        baseline_window = gen_config.baseline_window

    data = data.copy(deep=True)

    # Remove non-block data
    if block_col in data.columns:
        data = data[data[block_col] != ""]

    data = data[data[time_col].notna()]

    # Mask out non-bout samples
    if bout_col in data.columns:
        data.loc[~data[bout_col], vigor_col] = np.nan

    for t in data[trial_col].unique():
        mask_trial = data[trial_col] == t
        data_trial = data.loc[mask_trial].copy(deep=True)

        beg_bouts, end_bouts = analysis_utils.find_events(
            data_trial, bout_beg_col, bout_end_col, time_col
        )

        # Replace samples within each bout by bout-mean vigor
        for bout_b, bout_e in zip(beg_bouts, end_bouts):
            mask_bout = data_trial[time_col].between(bout_b, bout_e)
            mean_vigor = data_trial.loc[mask_bout, vigor_col].mean()
            data_trial.loc[mask_bout, vigor_col] = mean_vigor

        # Baseline quantile scaling
        mask_baseline = data_trial[time_col] < -baseline_window
        baseline_vigor = data_trial.loc[mask_baseline, vigor_col].dropna().values

        if baseline_vigor.size == 0:
            continue

        q_lo, q_hi = np.quantile(baseline_vigor, list(quantile_range))

        if np.isnan(q_hi) or q_lo == q_hi:
            continue

        data_trial.loc[mask_trial, vigor_col] = (
            (data_trial.loc[mask_trial, vigor_col] - q_lo) / (q_hi - q_lo)
        )
        data_trial[vigor_col] = data_trial[vigor_col].clip(0, 1)
        data.loc[mask_trial] = data_trial

    # Final mask
    if bout_col in data.columns:
        data.loc[~data[bout_col], vigor_col] = np.nan

    return data


# ---------------------------------------------------------------------------
# Per-trial normalized vigor computation  (used by scripts 1 & 2)
# ---------------------------------------------------------------------------

def compute_normalized_vigor_per_trial(
    data: pd.DataFrame,
    trials_range: np.ndarray,
    cr_window: list[float],
    *,
    csus: str = "CS",
    vigor_col: str = "Vigor (deg/ms)",
    bout_col: str = "Bout",
    time_col: str = "Trial time (s)",
    trial_col: str = "Trial number",
    block_col: str = "Block name",
    baseline_window: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-trial before, during, and normalized vigor.

    Returns ``(vigor_before, vigor_during, normalized_vigor)`` arrays of
    shape ``(len(trials_range),)``.
    """
    if baseline_window is None:
        baseline_window = gen_config.baseline_window

    data = data.copy(deep=True)
    if block_col in data.columns:
        data = data[data[block_col] != ""]
    if bout_col in data.columns:
        data.loc[~data[bout_col], vigor_col] = np.nan

    n = len(trials_range)
    bef = np.ones(n)
    dur = np.ones(n)
    nv = np.ones(n)

    for t_i, t in enumerate(trials_range):
        mask_trial = data[trial_col] == t
        data_trial = data.loc[mask_trial]

        if csus == "CS":
            bef[t_i] = data_trial.loc[
                data_trial[time_col].between(-baseline_window, 0), vigor_col
            ].mean()
            dur[t_i] = data_trial.loc[
                data_trial[time_col].between(cr_window[0], cr_window[1]), vigor_col
            ].mean()
        else:
            bef[t_i] = data_trial.loc[
                data_trial[time_col].between(
                    -baseline_window - cr_window[1], -cr_window[1]
                ),
                vigor_col,
            ].mean()
            dur[t_i] = data_trial.loc[
                data_trial[time_col].between(cr_window[0] - cr_window[1], 0),
                vigor_col,
            ].mean()

        nv[t_i] = dur[t_i] / bef[t_i]

    return bef, dur, nv


# ---------------------------------------------------------------------------
# Block name construction  (used by scripts 5 & 6)
# ---------------------------------------------------------------------------

def build_5_trial_block_names(number_blocks_original: int) -> list[str]:
    """Return ordered 5-trial block names based on original 10-trial block count.

    Supported counts: 7 (standard), 9 (extended test), 12 (re-train).
    """
    if number_blocks_original == 7:
        return [
            "Early Pre-Train", "Late Pre-Train",
            "Early Train", "Train 2", "Train 3", "Train 4",
            "Train 5", "Train 6", "Train 7", "Train 8", "Train 9", "Late Train",
            "Early Test", "Late Test",
        ]
    elif number_blocks_original == 9:
        return [
            "Early Pre-Train", "Late Pre-Train",
            "Early Train", "Train 2", "Train 3", "Train 4",
            "Train 5", "Train 6", "Train 7", "Train 8", "Train 9", "Late Train",
            "Early Test", "Test 2", "Test 3", "Test 4", "Test 5", "Late Test",
        ]
    elif number_blocks_original == 12:
        return [
            "Early Pre-Train", "Late Pre-Train",
            "Early Train", "Train 2", "Train 3", "Train 4",
            "Train 5", "Train 6", "Train 7", "Train 8", "Train 9", "Late Train",
            "Early Test", "Test 2", "Test 3", "Test 4", "Test 5", "Late Test",
            "Early Re-Train", "Re-Train 2", "Re-Train 3", "Re-Train 4",
            "Re-Train 5", "Late Re-Train",
        ]
    else:
        raise ValueError(
            f"Unsupported number of original blocks: {number_blocks_original}. "
            "Expected 7, 9, or 12."
        )


def get_block_config(number_blocks_original: int) -> tuple[
    list[str], list[str], list[str], int
]:
    """Return ``(block_names, blocks_chosen, blocks_chosen_labels, n_trials_per_block)``.

    Used by scripts 5 and 6 when setting up block-level analyses.
    """
    number_trials_block = 1
    blocks_chosen: list[str] = ["Train"]
    blocks_chosen_labels: list[str] = []
    block_names: list[str] = []

    if number_blocks_original == 7:
        block_names = [
            "Pre-Train", "Early Train", "Train 2", "Train 3",
            "Train 4", "Late Train", "Test",
        ]
        number_trials_block = 10
        blocks_chosen = ["Pre-Train", "Test"]
        blocks_chosen_labels = blocks_chosen
    elif number_blocks_original == 9:
        block_names = build_5_trial_block_names(9)
        number_trials_block = 5
        blocks_chosen = ["Early Pre-Train", "Early Test", "Late Test"]
        blocks_chosen_labels = ["PTr", "ETe", "LTe"]
    elif number_blocks_original == 12:
        block_names = build_5_trial_block_names(12)
        number_trials_block = 5
        blocks_chosen = ["Late Pre-Train", "Early Test", "Late Test"]
        blocks_chosen_labels = ["PT", "ET", "LT"]

    if not blocks_chosen_labels:
        blocks_chosen_labels = blocks_chosen

    return block_names, blocks_chosen, blocks_chosen_labels, number_trials_block
