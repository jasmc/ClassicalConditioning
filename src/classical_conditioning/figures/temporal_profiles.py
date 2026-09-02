"""Static, publication, and interactive candidate temporal-profile figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from classical_conditioning.analysis.movement_state import (
    CandidateMetricSource,
    resolve_candidate_metric_source,
)
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.figures.export import (
    FigureExportResult,
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.figures.theme import (
    DOUBLE_COLUMN_MM,
    add_stimulus_window,
    apply_theme,
    heatmap_cmap,
    stacked_subplots,
    stimulus_duration_s,
    style_axes,
)

METRIC_LABELS = {
    "segment_absolute_angular_speed_sum": "Segment angular-speed sum (rad/ms)",
    "all_segment_angular_rms": "Segment angular RMS (rad/ms)",
    "whole_tail_xy_rms_speed": "Whole-tail XY RMS (px/ms)",
    "whole_tail_xy_mean_speed": "Whole-tail XY mean (px/ms)",
    "curvature_change_rms": "Curvature-change RMS (rad/px/ms)",
}

OUTCOME_SPECS = {
    "total-activity": {
        "column": "Total activity mean",
        "title": "total activity",
        "coverage": "Valid expected fraction",
        "cmap_family": "intensity",
        "plotly_colorscale": "Viridis",
        "scale": "quantile-0.99",
        "colorbar": "Relative intensity (panel 99th percentile)",
    },
    "movement-probability": {
        "column": "Movement probability",
        "title": "movement probability",
        "coverage": "Detector valid fraction",
        "cmap_family": "probability",
        "plotly_colorscale": "Magma",
        "scale": "fixed",
        "colorbar": "Movement probability",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    "fraction-time-moving": {
        "column": "Fraction time moving",
        "title": "fraction time moving",
        "coverage": "Detector valid fraction",
        "cmap_family": "probability",
        "plotly_colorscale": "Magma",
        "scale": "fixed",
        "colorbar": "Fraction time moving",
        "vmin": 0.0,
        "vmax": 1.0,
    },
    "conditional-intensity": {
        "column": "Conditional intensity mean",
        "title": "conditional movement intensity",
        "coverage": "Detector valid fraction",
        "cmap_family": "intensity",
        "plotly_colorscale": "Viridis",
        "scale": "quantile-0.99",
        "colorbar": "Relative intensity (panel 99th percentile)",
    },
    "bout-rate": {
        "column": "Bout rate per minute",
        "title": "bout initiation rate",
        "coverage": "Detector valid fraction",
        "cmap_family": "intensity",
        "plotly_colorscale": "Plasma",
        "scale": "quantile-0.99",
        "colorbar": "Relative intensity (panel 99th percentile)",
    },
}


def _outcome_scale(
    values: np.ndarray,
    outcome: dict[str, str | float],
) -> tuple[float, float, str]:
    finite = values[np.isfinite(values)]
    vmin = float(outcome.get("vmin", 0.0))
    if outcome["scale"] == "fixed":
        vmax = float(outcome["vmax"])
        description = f"linear, fixed [{vmin:g}, {vmax:g}]"
    else:
        vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
        vmax = max(vmax, np.finfo(float).eps)
        description = (
            f"linear, vmin={vmin:g}, vmax={vmax:g} "
            "(metric 99th percentile)"
        )
    return vmin, vmax, description


def figure_version_tag(source: CandidateMetricSource) -> str:
    """Stable filename suffix for a candidate temporal recipe."""
    if source.requires_corrected_preprocess:
        return "corrected-v2"
    return "v2"


def _load_profiles(
    project_dir: Path,
    recording_id: str,
    *,
    source: CandidateMetricSource,
) -> tuple[pd.DataFrame, dict, tuple[int, int], Path]:
    source_dir = project_dir / "Processed data" / recording_id
    profiles_path = source_dir / source.temporal_artifact_name
    marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{source.temporal_marker_suffix}"
    )
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.temporal_summary_name
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "complete"
        or marker.get("recipe") != source.temporal_recipe
        or marker.get("recording_id") != recording_id
    ):
        raise ValueError("Candidate temporal profile marker is incomplete.")
    profile_state = (
        profiles_path.stat().st_size,
        profiles_path.stat().st_mtime_ns,
    )
    if sha256_file(profiles_path) != marker.get("profiles_sha256"):
        raise ValueError("Candidate temporal profile hash differs from its marker.")
    if sha256_file(summary_path) != marker.get("summary_sha256"):
        raise ValueError("Candidate temporal summary hash differs from its marker.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("recording_id") != recording_id
        or summary.get("recipe") != source.temporal_recipe
    ):
        raise ValueError("Candidate temporal summary identity is invalid.")
    profiles = pq.read_table(profiles_path).to_pandas()
    if (
        profile_state
        != (profiles_path.stat().st_size, profiles_path.stat().st_mtime_ns)
        or sha256_file(profiles_path) != marker.get("profiles_sha256")
    ):
        raise RuntimeError("Candidate temporal profile changed while loading.")
    if recording_id_from_profiles(profiles) != recording_id:
        raise ValueError("Candidate temporal profile table identity is invalid.")
    return profiles, marker, profile_state, profiles_path


def _verify_profile_unchanged(
    profile_path: Path,
    marker: dict,
    profile_state: tuple[int, int],
) -> None:
    if profile_state != (
        profile_path.stat().st_size,
        profile_path.stat().st_mtime_ns,
    ) or sha256_file(profile_path) != marker["profiles_sha256"]:
        raise RuntimeError("Candidate temporal profile changed during figure build.")


def _outcome_cmap_name(outcome: dict[str, str | float], theme) -> str:
    if outcome["cmap_family"] == "probability":
        return theme.probability_cmap
    return theme.intensity_cmap


def _candidate_heatmap_figure(
    profiles: pd.DataFrame,
    trial_type: str,
    outcome_id: str,
) -> tuple[plt.Figure, list[str], dict[str, dict[str, str]]]:
    if outcome_id not in OUTCOME_SPECS:
        raise ValueError(f"Unknown candidate outcome: {outcome_id}")
    outcome = OUTCOME_SPECS[outcome_id]
    theme = apply_theme()
    selected = profiles[profiles["Trial type"].astype(str) == trial_type]
    metrics = list(METRIC_LABELS)
    figure, axes = stacked_subplots(
        len(metrics),
        width_mm=DOUBLE_COLUMN_MM,
        row_height_mm=26.0,
        sharex=True,
        theme=theme,
    )
    cmap = heatmap_cmap(_outcome_cmap_name(outcome, theme), theme)
    duration_s = stimulus_duration_s(trial_type)
    panel_ids: list[str] = []
    artist_mappings: dict[str, dict[str, str]] = {}
    last_scale: tuple[float, float] | None = None
    for index, (axis, metric_id) in enumerate(zip(axes, metrics)):
        panel_id = chr(ord("A") + index)
        panel_ids.append(panel_id)
        metric = selected[selected["Metric ID"].astype(str) == metric_id]
        metric = metric.copy()
        metric.loc[
            metric[outcome["coverage"]] < 0.9,
            outcome["column"],
        ] = np.nan
        pivot = metric.pivot(
            index="Trial number",
            columns="Time bin center (s)",
            values=outcome["column"],
        )
        values = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        vmin, vmax, scale_description = _outcome_scale(
            np.asarray(values.filled(np.nan)),
            outcome,
        )
        last_scale = (vmin, vmax)
        time_centers = pivot.columns.to_numpy(dtype=float)
        trial_centers = pivot.index.to_numpy(dtype=float)
        time_step = (
            float(np.median(np.diff(time_centers)))
            if len(time_centers) > 1
            else 1.0
        )
        trial_step = (
            float(np.median(np.diff(trial_centers)))
            if len(trial_centers) > 1
            else 1.0
        )
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            extent=(
                float(time_centers.min() - time_step / 2),
                float(time_centers.max() + time_step / 2),
                float(trial_centers.min() - trial_step / 2),
                float(trial_centers.max() + trial_step / 2),
            ),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            rasterized=True,
        )
        heatmap_id = (
            f"heatmap__{panel_id.lower()}__{metric_id}__{outcome_id}"
        )
        image.set_gid(heatmap_id)
        artist_mappings[heatmap_id] = {
            "x_field": "Time bin center (s)",
            "y_field": "Trial number",
            "value_field": outcome["column"],
            "coverage_field": outcome["coverage"],
            "coverage_threshold": "0.9",
            "display_scale": scale_description,
            "cmap": cmap.name if hasattr(cmap, "name") else str(cmap),
        }
        stimulus_id = f"stimulus__{panel_id.lower()}__window"
        add_stimulus_window(
            axis,
            trial_type,
            onset_s=0.0,
            duration_s=duration_s,
            theme=theme,
            gid=stimulus_id,
        )
        artist_mappings[stimulus_id] = {
            "source": "protocol alignment",
            "time_s": "0",
            "duration_s": str(duration_s),
            "alignment": trial_type,
        }
        is_last = index == len(metrics) - 1
        style_axes(
            axis,
            theme=theme,
            show_xticks=is_last,
            show_yticks=True,
            xlabel=f"Time from {trial_type} onset (s)",
            ylabel="Trial",
        )
        axis.set_title(METRIC_LABELS[metric_id], loc="left")
    if last_scale is None:
        raise ValueError("Candidate heatmap has no metric panels.")
    if outcome["scale"] == "fixed":
        colorbar_norm = Normalize(vmin=last_scale[0], vmax=last_scale[1])
    else:
        colorbar_norm = Normalize(vmin=0.0, vmax=1.0)
    mappable = ScalarMappable(cmap=cmap, norm=colorbar_norm)
    mappable.set_array([])
    colorbar = figure.colorbar(
        mappable,
        ax=list(axes),
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label(str(outcome["colorbar"]))
    colorbar.ax.set_gid("axes__colorbar__main")
    artist_mappings["colorbar"] = {
        "role": "colorbar",
        "label": str(outcome["colorbar"]),
        "cmap": cmap.name if hasattr(cmap, "name") else str(cmap),
        "shared": "true",
    }
    figure.suptitle(
        f"{recording_id_from_profiles(profiles)} — {trial_type}-aligned "
        f"{outcome['title']}"
    )
    return figure, panel_ids + ["colorbar"], artist_mappings


def recording_id_from_profiles(profiles: pd.DataFrame) -> str:
    values = profiles["Recording ID"].astype(str).unique()
    if len(values) != 1:
        raise ValueError("Profiles must contain exactly one recording ID.")
    return str(values[0])


def build_candidate_profile_figure(
    project_dir: Path,
    recording_id: str,
    *,
    mode: FigureMode,
    trial_type: str = "CS",
    outcome_id: str = "total-activity",
    metric_recipe: str | None = None,
    temporal_recipe: str | None = None,
    overwrite: bool = False,
) -> FigureExportResult | Path:
    """Render candidate temporal profiles from one immutable panel-data table."""
    project_dir = project_dir.resolve()
    route = resolve_candidate_metric_source(
        metric_recipe=metric_recipe,
        temporal_recipe=temporal_recipe,
    )
    profiles, marker, profile_state, input_path = _load_profiles(
        project_dir,
        recording_id,
        source=route,
    )
    if trial_type not in {"CS", "US"}:
        raise ValueError("Trial type must be CS or US.")
    if outcome_id not in OUTCOME_SPECS:
        raise ValueError(f"Unknown candidate outcome: {outcome_id}")
    outcome = OUTCOME_SPECS[outcome_id]
    source_path = Path(__file__).resolve()
    version = figure_version_tag(route)
    reproduction = (
        "python -m classical_conditioning figure-candidate-profiles "
        f"--project-dir \"{project_dir}\" --recording-id {recording_id} "
        f"--trial-type {trial_type} --outcome {outcome_id} --mode {mode.value} "
        f"--recipe {route.temporal_recipe}"
    )
    if mode == FigureMode.INTERACTIVE:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        output = (
            project_dir
            / "Figures"
            / "Interactive"
            / recording_id
            / f"candidate_{trial_type.lower()}_{outcome_id}-{version}.html"
        )
        if output.exists() and not overwrite:
            raise FileExistsError(f"Interactive figure exists: {output}")
        selected = profiles[profiles["Trial type"].astype(str) == trial_type]
        metrics = list(METRIC_LABELS)
        figure = make_subplots(
            rows=len(metrics),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[METRIC_LABELS[metric] for metric in metrics],
        )
        for row, metric_id in enumerate(metrics, start=1):
            metric = selected[selected["Metric ID"].astype(str) == metric_id]
            metric = metric.copy()
            metric.loc[
                metric[outcome["coverage"]] < 0.9,
                outcome["column"],
            ] = np.nan
            pivot = metric.pivot(
                index="Trial number",
                columns="Time bin center (s)",
                values=outcome["column"],
            )
            values = pivot.to_numpy(dtype=float)
            vmin, vmax, _ = _outcome_scale(values, outcome)
            yaxis_name = "yaxis" if row == 1 else f"yaxis{row}"
            domain = getattr(figure.layout, yaxis_name).domain
            figure.add_trace(
                go.Heatmap(
                    x=pivot.columns,
                    y=pivot.index,
                    z=values,
                    colorscale=outcome["plotly_colorscale"],
                    zmin=vmin,
                    zmax=vmax,
                    colorbar={
                        "title": (
                            METRIC_LABELS[metric_id]
                            if outcome_id
                            in {"total-activity", "conditional-intensity"}
                            else outcome["colorbar"]
                        ),
                        "x": 1.02,
                        "y": (domain[0] + domain[1]) / 2,
                        "len": domain[1] - domain[0],
                        "yanchor": "middle",
                    },
                    name=metric_id,
                ),
                row=row,
                col=1,
            )
        figure.update_layout(
            title=f"{recording_id} — {trial_type}-aligned {outcome['title']}",
            height=1_400,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        sidecar = output.with_suffix(".figure.json")
        with artifact_staging(
            output.parent,
            prefix=f".{output.stem}-interactive-",
        ) as staging_root:
            staged_html = staging_root / output.name
            staged_sidecar = staging_root / sidecar.name
            figure.write_html(
                staged_html,
                include_plotlyjs=True,
                full_html=True,
                auto_open=False,
                config={"displaylogo": False},
            )
            write_json_atomic(
                staged_sidecar,
                {
                    "figure_id": (
                        f"candidate-{trial_type.lower()}-{outcome_id}-{version}"
                    ),
                    "analysis_recipe": route.temporal_recipe,
                    "mode": "interactive",
                    "source_file": str(source_path),
                    "source_symbol": "build_candidate_profile_figure",
                    "source_hash": sha256_file(source_path),
                    "reproduction_snippet": reproduction,
                    "input_artifacts": [
                        {
                            "path": str(input_path),
                            "sha256": marker["profiles_sha256"],
                        }
                    ],
                    "outputs": [
                        {
                            "path": str(output),
                            "sha256": sha256_file(staged_html),
                        }
                    ],
                },
            )
            _verify_profile_unchanged(input_path, marker, profile_state)
            publish_transaction(
                ((staged_html, output), (staged_sidecar, sidecar)),
                staging_root,
                overwrite=overwrite,
            )
        return output

    figure, panel_ids, artist_mappings = _candidate_heatmap_figure(
        profiles, trial_type, outcome_id
    )
    output_root = (
        project_dir
        / "Figures"
        / ("Publication" if mode == FigureMode.PUBLICATION else "PNG")
        / recording_id
    )
    output_base = (
        output_root / f"candidate_{trial_type.lower()}_{outcome_id}-{version}"
    )
    provenance = FigureProvenance(
        figure_id=f"candidate-{trial_type.lower()}-{outcome_id}-{version}",
        analysis_recipe=route.temporal_recipe,
        source_file=str(source_path),
        source_symbol="build_candidate_profile_figure",
        source_hash=sha256_file(source_path),
        reproduction_snippet=reproduction,
        input_artifacts=(
            {
                "path": str(input_path),
                "sha256": marker["profiles_sha256"],
            },
        ),
        artist_mappings=artist_mappings,
    )
    try:
        _verify_profile_unchanged(input_path, marker, profile_state)
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
