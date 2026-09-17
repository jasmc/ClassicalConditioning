"""Residual diagnostic figure for learning-onset mixed models."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np

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
from classical_conditioning.figures.theme import (
    DEFAULT_THEME,
    DOUBLE_COLUMN_MM,
    apply_theme,
    mm_to_in,
    style_axes,
)


def build_learning_diagnostics_figure(
    project_dir: Path,
    analysis_id: str,
    *,
    mode: FigureMode,
    overwrite: bool = False,
) -> FigureExportResult:
    """Plot residual-versus-fit and normal-quantile diagnostics."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Learning diagnostic figures are Matplotlib-only.")
    project_dir = project_dir.resolve()
    frames, summary = load_learning_onset_analysis(project_dir, analysis_id)
    residuals = frames["residuals"]
    model_names = ("block", "longitudinal")
    if residuals.empty:
        raise ConfigurationError("No accepted model produced residual diagnostics.")

    apply_theme(DEFAULT_THEME)
    width, _ = mm_to_in(DOUBLE_COLUMN_MM, 150.0)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(width, 150.0 / 25.4),
        layout="constrained",
    )
    panel_ids = ["A", "B", "C", "D"]
    artist_mappings = {}
    for row, model_name in enumerate(model_names):
        model = residuals.loc[residuals["model"] == model_name].copy()
        fitted = model["fitted_log_response"].to_numpy(dtype=float)
        standardized = model["standardized_residual"].to_numpy(dtype=float)
        finite = np.isfinite(fitted) & np.isfinite(standardized)
        fitted = fitted[finite]
        standardized = standardized[finite]
        if not len(standardized):
            for column, label in enumerate(("Residual vs fit", "Normal quantiles")):
                axis = axes[row, column]
                axis.text(
                    0.5,
                    0.5,
                    "No accepted fit",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
                axis.set_title(
                    f"{panel_ids[row * 2 + column]}  "
                    f"{model_name.title()}: {label}"
                )
                axis.set_axis_off()
            continue

        residual_axis = axes[row, 0]
        residual_axis.axhline(0.0, color="0.55", linewidth=0.7)
        points = residual_axis.scatter(
            fitted,
            standardized,
            s=5,
            alpha=0.25,
            color="0.15",
            linewidths=0,
        )
        points.set_gid(f"{model_name}-residual-versus-fit")
        artist_mappings[f"{model_name}-residual-versus-fit"] = {
            "model": model_name,
            "x": "fitted_log_response",
            "y": "standardized_residual",
        }
        style_axes(
            residual_axis,
            theme=DEFAULT_THEME,
            xlabel="Fitted log response",
            ylabel="Standardized residual",
        )
        residual_axis.set_title(
            f"{panel_ids[row * 2]}  {model_name.title()}: residual vs fit"
        )

        quantile_axis = axes[row, 1]
        ordered = np.sort(standardized)
        probabilities = (np.arange(len(ordered), dtype=float) + 0.5) / len(ordered)
        theoretical = np.array(
            [NormalDist().inv_cdf(float(value)) for value in probabilities]
        )
        low = float(min(theoretical.min(), ordered.min()))
        high = float(max(theoretical.max(), ordered.max()))
        quantile_axis.plot([low, high], [low, high], color="0.55", linewidth=0.7)
        quantiles = quantile_axis.scatter(
            theoretical,
            ordered,
            s=5,
            alpha=0.35,
            color="0.15",
            linewidths=0,
        )
        quantiles.set_gid(f"{model_name}-normal-quantiles")
        artist_mappings[f"{model_name}-normal-quantiles"] = {
            "model": model_name,
            "value": "standardized_residual",
        }
        style_axes(
            quantile_axis,
            theme=DEFAULT_THEME,
            xlabel="Normal theoretical quantile",
            ylabel="Observed residual quantile",
        )
        quantile_axis.set_title(
            f"{panel_ids[row * 2 + 1]}  {model_name.title()}: normal quantiles"
        )

    source_path = Path(__file__).resolve()
    provenance = FigureProvenance(
        figure_id="learning-diagnostics",
        analysis_recipe="learning-onset",
        source_file=str(source_path),
        source_symbol="build_learning_diagnostics_figure",
        source_hash=sha256_file(source_path),
        reproduction_snippet=(
            "python -m classical_conditioning figure-learning-diagnostics "
            f"--project-dir \"{project_dir}\" --analysis-id {analysis_id} "
            f"--mode {mode.value}"
        ),
        input_artifacts=tuple(
            {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
            }
            for record in summary["artifacts"].values()
        ),
        cohort_hash=str(summary["cohort_hash"]),
        artist_mappings=artist_mappings,
    )
    output_base = (
        project_dir
        / "Figures"
        / ("Publication" if mode == FigureMode.PUBLICATION else "PNG")
        / "Analyses"
        / analysis_id
        / "learning-diagnostics"
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
