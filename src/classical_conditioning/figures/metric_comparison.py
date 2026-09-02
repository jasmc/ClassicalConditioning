"""Cohort-level candidate metric comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.analysis.metric_comparison import OUTCOME_COLUMNS
from classical_conditioning.analysis.movement_state import (
    resolve_candidate_metric_source,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import ConfigurationError, SchemaValidationError
from classical_conditioning.figures.export import (
    FigureExportResult,
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.figures.temporal_profiles import METRIC_LABELS
from classical_conditioning.figures.theme import (
    DEFAULT_THEME,
    DOUBLE_COLUMN_MM,
    apply_theme,
    condition_color,
    mm_to_in,
    style_axes,
)

SHORT_METRIC_LABELS = {
    "segment_absolute_angular_speed_sum": "Angular-speed sum",
    "all_segment_angular_rms": "Angular RMS",
    "whole_tail_xy_rms_speed": "XY RMS speed",
    "whole_tail_xy_mean_speed": "XY mean speed",
    "curvature_change_rms": "Curvature RMS",
}

CONDITION_DISPLAY = {
    "control": "Control",
    "fixedtrace": "3 s Trace",
}


def _figure_version_tag(comparison_recipe: str) -> str:
    if comparison_recipe.endswith("-corrected-v1"):
        return "corrected-v1"
    return "v1"


def _condition_colors(experiment_name: str | None) -> dict[str, tuple[float, float, float]]:
    colors: dict[str, tuple[float, float, float]] = {}
    if experiment_name:
        spec = get_experiment_spec(experiment_name)
        for condition in spec.conditions:
            colors[condition.condition_id] = condition_color(condition)
    if "control" not in colors:
        colors["control"] = (0 / 255, 174 / 255, 239 / 255)
    if "fixedtrace" not in colors:
        colors["fixedtrace"] = (241 / 255, 90 / 255, 41 / 255)
    return colors


def _cohort_standardized_figure(
    recording_summary: pd.DataFrame,
    *,
    trial_type: str,
    outcome_id: str,
    experiment_name: str | None = None,
) -> tuple:
    required = {
        "Recording ID",
        "Trial type",
        "Metric ID",
        "Outcome ID",
        "Standardized difference",
        "Condition ID",
    }
    missing = required.difference(recording_summary.columns)
    if missing:
        raise SchemaValidationError(
            f"Recording metric summary is missing columns: {sorted(missing)}"
        )
    if outcome_id not in OUTCOME_COLUMNS:
        raise ValueError(f"Unknown candidate outcome: {outcome_id}")
    selected = recording_summary.loc[
        (recording_summary["Trial type"].astype(str) == trial_type)
        & (recording_summary["Outcome ID"].astype(str) == outcome_id)
    ].copy()
    if selected.empty:
        raise ConfigurationError(
            f"No recording rows for {trial_type} {outcome_id}."
        )
    metrics = [metric for metric in METRIC_LABELS if metric in set(selected["Metric ID"])]
    conditions = [
        condition
        for condition in ("control", "fixedtrace")
        if condition in set(selected["Condition ID"].astype(str))
    ]
    if not conditions:
        conditions = sorted(
            {
                str(value)
                for value in selected["Condition ID"]
                if str(value).strip() and str(value) != "all"
            }
        )
    colors = _condition_colors(experiment_name)
    apply_theme(DEFAULT_THEME)
    width, _ = mm_to_in(DOUBLE_COLUMN_MM, 95.0)
    figure, axis = plt.subplots(figsize=(width, 95.0 / 25.4), layout="constrained")
    axis.axhline(0.0, color="0.6", linewidth=0.6, zorder=1)
    n_metrics = len(metrics)
    n_conditions = max(len(conditions), 1)
    group_width = 0.72
    bar_width = group_width / n_conditions
    rng = np.random.default_rng(0)
    artist_mappings: dict[str, dict[str, str]] = {}
    for condition_index, condition in enumerate(conditions):
        offset = (condition_index - (n_conditions - 1) / 2) * bar_width
        means: list[float] = []
        xs: list[float] = []
        for metric_index, metric_id in enumerate(metrics):
            rows = selected.loc[
                (selected["Metric ID"].astype(str) == metric_id)
                & (selected["Condition ID"].astype(str) == condition)
            ]
            values = rows["Standardized difference"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            x = metric_index + offset
            xs.append(x)
            means.append(float(np.mean(finite)) if finite.size else np.nan)
            if finite.size:
                jitter = rng.uniform(-0.08 * bar_width, 0.08 * bar_width, size=finite.size)
                points = axis.scatter(
                    np.full(finite.size, x) + jitter,
                    finite,
                    s=18,
                    color=colors.get(condition, "0.2"),
                    edgecolors="0.15",
                    linewidths=0.3,
                    zorder=4,
                    alpha=0.9,
                )
                point_id = f"points__{condition}__{metric_id}"
                points.set_gid(point_id)
                artist_mappings[point_id] = {
                    "condition": condition,
                    "metric_id": metric_id,
                    "value_field": "Standardized difference",
                }
        bars = axis.bar(
            xs,
            means,
            width=bar_width * 0.9,
            color=colors.get(condition, "0.7"),
            edgecolor="none",
            alpha=0.35,
            zorder=2,
            label=CONDITION_DISPLAY.get(condition, condition),
        )
        bar_id = f"bars__{condition}"
        for patch in bars.patches:
            patch.set_gid(bar_id)
        artist_mappings[bar_id] = {
            "condition": condition,
            "value_field": "Standardized difference",
            "aggregation": "equal_recording_mean",
        }
    axis.set_xticks(range(n_metrics))
    axis.set_xticklabels(
        [SHORT_METRIC_LABELS.get(metric, metric) for metric in metrics],
        rotation=20,
        ha="right",
    )
    style_axes(
        axis,
        theme=DEFAULT_THEME,
        xlabel="",
        ylabel="Standardized difference\n(response − baseline) / baseline SD",
    )
    axis.legend(frameon=False, loc="best")
    outcome_title = OUTCOME_COLUMNS[outcome_id]
    axis.set_title(
        f"{trial_type}-aligned {outcome_title} — fish-equal standardized difference"
    )
    return figure, ["A"], artist_mappings


def build_metric_comparison_figure(
    project_dir: Path,
    analysis_id: str,
    *,
    mode: FigureMode,
    trial_type: str = "CS",
    outcome_id: str = "movement-probability",
    comparison_recipe: str | None = None,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render a cohort metric-comparison figure from saved recording summaries."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Cohort metric-comparison figures are Matplotlib-only.")
    project_dir = project_dir.resolve()
    route = resolve_candidate_metric_source(comparison_recipe=comparison_recipe)
    recipe_id = route.comparison_recipe
    recording_path = (
        project_dir
        / "Processed data"
        / "Analyses"
        / analysis_id
        / f"{recipe_id}_recording_summary.parquet"
    )
    summary_path = (
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{recipe_id}_summary.json"
    )
    if not recording_path.is_file():
        raise FileNotFoundError(f"Missing recording metric summary: {recording_path}")
    recording_summary = pq.read_table(recording_path).to_pandas()
    experiment_name = None
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        experiment_name = payload.get("experiment_name")
    figure, panel_ids, artist_mappings = _cohort_standardized_figure(
        recording_summary,
        trial_type=trial_type,
        outcome_id=outcome_id,
        experiment_name=experiment_name,
    )
    version = _figure_version_tag(recipe_id)
    output_root = (
        project_dir
        / "Figures"
        / ("Publication" if mode == FigureMode.PUBLICATION else "PNG")
        / "Analyses"
        / analysis_id
    )
    output_base = (
        output_root
        / f"metric-comparison_{trial_type.lower()}_{outcome_id}-{version}"
    )
    source_path = Path(__file__).resolve()
    provenance = FigureProvenance(
        figure_id=(
            f"metric-comparison-{trial_type.lower()}-{outcome_id}-{version}"
        ),
        analysis_recipe=recipe_id,
        source_file=str(source_path),
        source_symbol="build_metric_comparison_figure",
        source_hash=sha256_file(source_path),
        reproduction_snippet=(
            "python -m classical_conditioning figure-metric-comparison "
            f"--project-dir \"{project_dir}\" --analysis-id {analysis_id} "
            f"--recipe {recipe_id} --trial-type {trial_type} "
            f"--outcome {outcome_id} --mode {mode.value}"
        ),
        input_artifacts=(
            {
                "path": str(recording_path),
                "sha256": sha256_file(recording_path),
            },
        ),
        artist_mappings=artist_mappings,
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
