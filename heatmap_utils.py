"""Shared heatmap helpers used across analysis scripts 1, 2, and 4.

Centralizes:
- Vigor-to-heatmap matrix pivoting
- Per-fish heatmap figure scaffolding (axes creation + stimulus lines)
- Phase-based heatmap block rendering (shared layout logic)

These utilities factor out the ~200-line pattern that was duplicated across
scripts for both raw and scaled vigor heatmaps.
"""
from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def pivot_vigor_to_heatmap_matrix(
    data: pd.DataFrame,
    *,
    time_col: str = "Trial time (s)",
    trial_col: str = "Trial number",
    vigor_col: str = "Vigor (deg/ms)",
) -> pd.DataFrame:
    """Pivot a long-form DataFrame into a trial × time-bin matrix for heatmap display.

    Returns a DataFrame with trial numbers as *rows* and time-bin centres as
    integer *columns*.
    """
    df = (
        data[[time_col, trial_col, vigor_col]]
        .pivot(index=time_col, columns=trial_col)
        .reset_index()
        .set_index(time_col)
        .droplevel(0, axis=1)
        .T
    )
    df.columns = df.columns.astype("int")
    return df


# ---------------------------------------------------------------------------
# Figure scaffolding
# ---------------------------------------------------------------------------

def make_phase_heatmap_axes(
    phases_trial_numbers: Sequence[Sequence[int]],
    *,
    ncols: int = 1,
    figsize: tuple[float, float] = (5 / 2.54, 6 / 2.54),
    hspace: float = 0.025,
    wspace: float = 0.25,
    sharey: bool | Literal["none", "all", "row", "col"] = False,
    sharex: bool | Literal["none", "all", "row", "col"] = False,
) -> tuple[Figure, np.ndarray]:
    """Create stacked subplot axes with height proportional to block size.

    Returns ``(fig, axs)`` where *axs* has shape ``(n_blocks, ncols)``.
    """
    height_ratios = [len(b) for b in phases_trial_numbers]
    fig_width = figsize[0] * max(ncols, 1)
    fig, axs = plt.subplots(
        len(phases_trial_numbers),
        ncols,
        facecolor="white",
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": hspace,
            "wspace": wspace,
        },
        squeeze=False,
        constrained_layout=False,
        figsize=(fig_width, figsize[1]),
        sharey=sharey,
        sharex=sharex,
    )
    return fig, axs


def add_stimulus_lines(
    ax: Axes,
    stim_color: Sequence[float],
    stim_duration: float,
    window_half_width: float,
    *,
    alpha: float = 0.7,
    lw: float = 1.5,
) -> None:
    """Draw vertical onset/offset lines for the stimulus on a heatmap axis."""
    xlims = ax.get_xlim()
    middle = np.mean(xlims)
    factor = (xlims[-1] - xlims[0]) / (window_half_width * 2)
    ax.axvline(middle, color=stim_color, alpha=alpha, lw=lw, linestyle="-")
    ax.axvline(
        middle + stim_duration * factor,
        color=stim_color,
        alpha=alpha,
        lw=lw,
        linestyle="-",
    )


# ---------------------------------------------------------------------------
# Block-wise heatmap rendering
# ---------------------------------------------------------------------------

def render_phase_blocks(
    axs: np.ndarray,
    heatmap_matrix: pd.DataFrame,
    phases_trial_numbers: Sequence[Sequence[int]],
    phases_block_names: Sequence[str],
    *,
    xtick_step: int | bool = False,
    stim_color: Sequence[float] | None = None,
    stim_duration: float = 0.0,
    window_half_width: float = 15.0,
    show_ylabel: bool = True,
    show_yticklabels: bool = False,
    col: int = 0,
    **heatmap_kwargs,
) -> None:
    """Render a series of phase-based heatmap blocks onto *axs*.

    Parameters
    ----------
    axs : ndarray of shape (n_phases, n_cols)
        Axes array from :func:`make_phase_heatmap_axes`.
    heatmap_matrix : DataFrame
        Trial × time matrix from :func:`pivot_vigor_to_heatmap_matrix`.
    phases_trial_numbers : list of lists of ints
        Trial numbers in each block/phase.
    phases_block_names : list of str
        Display name for each block.
    xtick_step : int or False
        X-tick label step for the bottom-most block.
    stim_color : RGB tuple, optional
        If given, stimulus onset/offset lines are drawn.
    stim_duration : float
        Duration of stimulus in seconds (for offset line).
    window_half_width : float
        Half the total time window in seconds (for line scaling).
    col : int
        Column index in the *axs* grid (for multi-column layouts).
    **heatmap_kwargs
        Passed directly to ``sns.heatmap``.
    """
    # Ensure we don't pass conflicting vmin/vmax when norm is present
    kw = dict(heatmap_kwargs)
    if "norm" in kw:
        kw.pop("vmin", None)
        kw.pop("vmax", None)

    for b_i, (trials, bname) in enumerate(zip(phases_trial_numbers, phases_block_names)):
        is_bottom = b_i == len(phases_trial_numbers) - 1
        show_xticks_here = xtick_step if is_bottom else False

        block_data = heatmap_matrix[heatmap_matrix.index.isin(trials)]

        sns.heatmap(
            block_data,
            xticklabels=show_xticks_here,
            yticklabels=show_yticklabels,
            ax=axs[b_i][col],
            **kw,
        )

        if not is_bottom:
            axs[b_i][col].set_xlabel("")

        if show_ylabel:
            axs[b_i][col].set_ylabel(bname, va="center", ha="center")
        else:
            axs[b_i][col].set_ylabel("")

        if stim_color is not None:
            add_stimulus_lines(
                axs[b_i][col], stim_color, stim_duration, window_half_width
            )

        axs[b_i][col].set_facecolor("k")
        axs[b_i][col].set_rasterization_zorder(0)

    # Format bottom x-tick labels as clean numbers
    if xtick_step:
        bottom_ax = axs[-1][col]
        labels = []
        for lbl in bottom_ax.get_xticklabels():
            text = lbl.get_text()
            if not text:
                labels.append("")
                continue
            try:
                labels.append(f"{float(text):g}")
            except ValueError:
                labels.append(text)
        bottom_ax.set_xticklabels(labels)
