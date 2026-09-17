"""Final learning trajectory, trial contrast, and block contrast figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_conditioning.analysis.inference.learning_onset import (
    load_learning_onset_analysis,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.figures.export import (
    FigureExportResult,
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.figures.metric_comparison import (
    CONDITION_DISPLAY,
    _condition_colors,
)
from classical_conditioning.figures.theme import (
    DEFAULT_THEME,
    DOUBLE_COLUMN_MM,
    apply_theme,
    mm_to_in,
    style_axes,
)


def _count_label(condition: str, summary: pd.DataFrame) -> str:
    counts = summary["fish_count"].astype(int)
    display = CONDITION_DISPLAY.get(condition, condition)
    if counts.empty:
        return f"{display} (n=0)"
    low, high = int(counts.min()), int(counts.max())
    return f"{display} (n={low})" if low == high else f"{display} (n={low}–{high})"


def _plot_learning_onset(
    frames: dict[str, pd.DataFrame],
    summary: dict[str, Any],
) -> tuple[Any, list[str], dict[str, dict[str, Any]]]:
    model_input = frames["model_input"]
    trials = frames["trial_contrasts"]
    blocks = frames["block_contrasts"]
    onset = frames["onset"]
    if model_input.empty:
        raise ConfigurationError("Learning-onset model input is empty.")
    fish = frames["fish_trajectory"]
    group = frames["group_trajectory"]
    if fish.empty or group.empty:
        raise ConfigurationError("Saved learning-trajectory panel data are empty.")
    experiment_ids = model_input["experiment_id"].astype(str).unique()
    experiment_id = experiment_ids[0] if len(experiment_ids) == 1 else None
    colors = _condition_colors(experiment_id)
    apply_theme(DEFAULT_THEME)
    width, _ = mm_to_in(DOUBLE_COLUMN_MM, 180.0)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(width, 180.0 / 25.4),
        layout="constrained",
        height_ratios=(1.25, 1.0, 0.9),
    )
    mappings: dict[str, dict[str, Any]] = {}
    trial_blocks = (
        model_input.loc[:, ["trial_number", "block_10_name"]]
        .drop_duplicates()
        .sort_values("trial_number")
    )
    block_changes = trial_blocks.loc[
        trial_blocks["block_10_name"].astype(str).ne(
            trial_blocks["block_10_name"].astype(str).shift()
        )
    ]
    boundaries = block_changes["trial_number"].astype(float).iloc[1:] - 0.5
    phase_ranges = []
    for phase, block_prefixes in (
        ("Pre-train", ("Pre-train",)),
        ("Training", ("Train",)),
        ("Test", ("Test",)),
    ):
        phase_trials = trial_blocks.loc[
            trial_blocks["block_10_name"].astype(str).str.startswith(
                block_prefixes
            ),
            "trial_number",
        ].astype(float)
        if not phase_trials.empty:
            phase_ranges.append((phase, phase_trials.min(), phase_trials.max()))

    trajectory_axis = axes[0]
    trajectory_axis.axhline(1.0, color="0.65", linewidth=0.6, zorder=1)
    for boundary in boundaries:
        trajectory_axis.axvline(
            boundary, color="0.85", linewidth=0.45, zorder=1
        )
    for condition in dict.fromkeys(fish["condition_id"].astype(str)):
        color = colors.get(condition, "0.2")
        condition_fish = fish.loc[fish["condition_id"].astype(str) == condition]
        for fish_key, frame in condition_fish.groupby("fish_key", observed=True):
            frame = frame.sort_values("trial_number")
            artist = trajectory_axis.plot(
                frame["trial_number"],
                frame["activity_ratio"],
                color=color,
                alpha=0.12,
                linewidth=0.45,
                zorder=2,
            )[0]
            artist_id = f"fish-trajectory__{condition}__{fish_key}"
            artist.set_gid(artist_id)
            mappings[artist_id] = {
                "condition": condition,
                "fish_key": str(fish_key),
                "value_field": "activity_ratio",
            }
        condition_group = group.loc[
            group["condition_id"].astype(str) == condition
        ].sort_values("trial_number")
        x = condition_group["trial_number"].to_numpy(dtype=float)
        ribbon = trajectory_axis.fill_between(
            x,
            condition_group["q25"].to_numpy(dtype=float),
            condition_group["q75"].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            zorder=3,
        )
        ribbon_id = f"fish-iqr__{condition}"
        ribbon.set_gid(ribbon_id)
        mappings[ribbon_id] = {
            "condition": condition,
            "aggregation": "fish_IQR",
        }
        line = trajectory_axis.plot(
            x,
            condition_group["median"].to_numpy(dtype=float),
            color=color,
            linewidth=1.25,
            zorder=4,
            label=_count_label(condition, condition_group),
        )[0]
        line_id = f"condition-median__{condition}"
        line.set_gid(line_id)
        mappings[line_id] = {
            "condition": condition,
            "aggregation": "equal_fish_median",
        }
    style_axes(
        trajectory_axis,
        theme=DEFAULT_THEME,
        xlabel="CS trial number",
        ylabel="Response / pre-CS baseline",
    )
    trajectory_axis.set_title("A  Learning trajectory (lower = stronger suppression)")
    for phase, start, end in phase_ranges:
        trajectory_axis.text(
            (start + end) / 2.0,
            1.01,
            phase,
            transform=trajectory_axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=DEFAULT_THEME.font_size - 1,
            clip_on=False,
        )
    trajectory_axis.legend(loc="best")

    contrast_axis = axes[1]
    contrast_axis.axhline(0.0, color="0.45", linewidth=0.7, zorder=1)
    for boundary in boundaries:
        contrast_axis.axvline(
            boundary, color="0.85", linewidth=0.45, zorder=1
        )
    delta_min = float(summary["config"]["delta_min"])
    contrast_axis.axhline(
        delta_min,
        color="0.65",
        linewidth=0.7,
        linestyle="--",
        zorder=1,
    )
    if not trials.empty:
        trials = trials.sort_values("trial_number")
        x = trials["trial_number"].to_numpy(dtype=float)
        lower = trials["simultaneous_lower"].to_numpy(dtype=float)
        upper = trials["simultaneous_upper"].to_numpy(dtype=float)
        ribbon = contrast_axis.fill_between(
            x, lower, upper, color="0.35", alpha=0.18, zorder=2
        )
        ribbon.set_gid("trial-contrast-simultaneous-band")
        mappings["trial-contrast-simultaneous-band"] = {
            "value_fields": ["simultaneous_lower", "simultaneous_upper"],
            "uncertainty": "fish_bootstrap_simultaneous",
        }
        line = contrast_axis.plot(
            x,
            trials["learning_contrast"].to_numpy(dtype=float),
            color="0.1",
            linewidth=1.2,
            zorder=3,
        )[0]
        line.set_gid("trial-learning-contrast")
        mappings["trial-learning-contrast"] = {
            "value_field": "learning_contrast",
            "direction": "positive_is_stronger_test_learning",
        }
        if not onset.empty and bool(onset.iloc[0]["localized"]):
            onset_trial = float(onset.iloc[0]["onset_trial"])
            onset_lower = onset.iloc[0].get("bootstrap_onset_lower")
            onset_upper = onset.iloc[0].get("bootstrap_onset_upper")
            if pd.notna(onset_lower) and pd.notna(onset_upper):
                contrast_axis.axvspan(
                    float(onset_lower),
                    float(onset_upper),
                    color=DEFAULT_THEME.us_color,
                    alpha=0.10,
                    zorder=1,
                )
            contrast_axis.axvline(
                onset_trial,
                color=DEFAULT_THEME.us_color,
                linewidth=1.0,
                zorder=4,
                label=f"Onset: trial {int(onset_trial)}",
            )
            contrast_axis.legend(loc="best")
    style_axes(
        contrast_axis,
        theme=DEFAULT_THEME,
        xlabel="CS trial number",
        ylabel="Adjusted learning contrast",
    )
    contrast_axis.set_title("B  Test-versus-control change from pre-training")

    block_axis = axes[2]
    block_axis.axvline(0.0, color="0.45", linewidth=0.7, zorder=1)
    if not blocks.empty:
        blocks = blocks.reset_index(drop=True)
        y = np.arange(len(blocks), dtype=float)
        estimates = blocks["learning_contrast"].to_numpy(dtype=float)
        lower = estimates - blocks["ci_lower"].to_numpy(dtype=float)
        upper = blocks["ci_upper"].to_numpy(dtype=float) - estimates
        points = block_axis.errorbar(
            estimates,
            y,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color="0.15",
            markersize=3.5,
            capsize=2,
            zorder=3,
        )
        points.lines[0].set_gid("block-learning-contrasts")
        mappings["block-learning-contrasts"] = {
            "value_field": "learning_contrast",
            "interval_fields": ["ci_lower", "ci_upper"],
        }
        block_axis.set_yticks(y)
        block_axis.set_yticklabels(blocks["block_10_name"])
        supported_rows = blocks["supported"].fillna(False).astype(bool)
        supported_blocks = blocks.loc[supported_rows, "block_10_name"].astype(str)
        if supported_rows.any():
            supported_indices = np.flatnonzero(supported_rows.to_numpy())
            block_axis.scatter(
                estimates[supported_indices],
                y[supported_indices],
                marker="o",
                s=16,
                color=DEFAULT_THEME.us_color,
                zorder=4,
                label=f"First supported: {supported_blocks.iloc[0]}",
            )
        primary_block = str(summary["config"]["late_blocks"][-1])
        primary_rows = blocks["block_10_name"].astype(str).eq(primary_block)
        if primary_rows.any():
            primary_index = int(np.flatnonzero(primary_rows.to_numpy())[0])
            block_axis.scatter(
                [estimates[primary_index]],
                [y[primary_index]],
                marker="s",
                s=28,
                facecolors="none",
                edgecolors=DEFAULT_THEME.us_color,
                linewidths=0.9,
                zorder=4,
                label=f"Primary: {primary_block}",
            )
        if supported_rows.any() or primary_rows.any():
            block_axis.legend(loc="best")
        block_axis.invert_yaxis()
    style_axes(
        block_axis,
        theme=DEFAULT_THEME,
        xlabel="Adjusted learning contrast",
        ylabel="Prespecified block",
    )
    block_axis.set_title("C  Planned block contrasts")
    return figure, ["A", "B", "C"], mappings


def build_learning_onset_figure(
    project_dir: Path,
    analysis_id: str,
    *,
    mode: FigureMode,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render the final three-component learning-onset figure."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Learning-onset figures are Matplotlib-only.")
    project_dir = project_dir.resolve()
    frames, summary = load_learning_onset_analysis(project_dir, analysis_id)
    diagnostics = frames["diagnostics"]
    required = diagnostics.loc[
        diagnostics["required_for_publication"].fillna(True).astype(bool)
    ]
    if required.empty or not required["diagnostic_status"].eq("ok").all():
        raise ConfigurationError(
            "Learning-onset figure requires accepted block and longitudinal fits."
        )
    figure, panel_ids, mappings = _plot_learning_onset(frames, summary)
    source_path = Path(__file__).resolve()
    input_artifacts = tuple(
        {
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
        }
        for record in summary["artifacts"].values()
    )
    provenance = FigureProvenance(
        figure_id="learning-onset",
        analysis_recipe="learning-onset",
        source_file=str(source_path),
        source_symbol="build_learning_onset_figure",
        source_hash=sha256_file(source_path),
        reproduction_snippet=(
            "python -m classical_conditioning figure-learning-onset "
            f"--project-dir \"{project_dir}\" --analysis-id {analysis_id} "
            f"--mode {mode.value}"
        ),
        input_artifacts=input_artifacts,
        cohort_hash=str(summary["cohort_hash"]),
        artist_mappings=mappings,
    )
    output_base = (
        project_dir
        / "Figures"
        / ("Publication" if mode == FigureMode.PUBLICATION else "PNG")
        / "Analyses"
        / analysis_id
        / "learning-onset"
    )
    try:
        return export_matplotlib_figure(
            figure,
            output_base,
            provenance,
            mode=mode,
            panel_ids=panel_ids,
            overwrite=overwrite,
        )
    finally:
        plt.close(figure)
