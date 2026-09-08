"""Static, publication, and interactive candidate temporal-profile figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    "legacy_distal_angular_speed": "Legacy distal angular speed (rad/ms)",
}

METRIC_UNITS = {
    "segment_absolute_angular_speed_sum": "rad/ms",
    "all_segment_angular_rms": "rad/ms",
    "whole_tail_xy_rms_speed": "px/ms",
    "whole_tail_xy_mean_speed": "px/ms",
    "curvature_change_rms": "rad/px/ms",
    "legacy_distal_angular_speed": "rad/ms",
}

DETECTOR_COVERAGE = "Detector valid fraction"
COVERAGE_THRESHOLD = 0.9


@dataclass(frozen=True)
class PanelSpec:
    """One heatmap row: which column to draw and how to scale its colour."""

    key: str
    title: str
    column: str
    coverage: str
    colorbar_label: str
    cmap_family: str = "intensity"
    plotly_colorscale: str = "Viridis"
    scale: str = "quantile-0.99"
    vmin: float = 0.0
    vmax: float | None = None
    metric_id: str | None = None


@dataclass(frozen=True)
class FigureSpec:
    """One figure: its rows, its title, and whether rows share a colour scale."""

    figure_id: str
    title: str
    panels: tuple[PanelSpec, ...]
    shared_colorbar: bool
    description: str


def _metric_panels(
    column: str,
    coverage: str,
    *,
    scaled: bool,
) -> tuple[PanelSpec, ...]:
    """One row per metric, in registry order."""
    panels = []
    for metric_id, label in METRIC_LABELS.items():
        if scaled:
            colorbar_label = "Scaled activity (0-1)"
            spec = PanelSpec(
                key=metric_id,
                title=label,
                column=column,
                coverage=coverage,
                colorbar_label=colorbar_label,
                scale="fixed",
                vmax=1.0,
                metric_id=metric_id,
            )
        else:
            # The row title already names the metric and its unit, so the
            # per-row colorbar carries only the unit and stays legible.
            spec = PanelSpec(
                key=metric_id,
                title=label,
                column=column,
                coverage=coverage,
                colorbar_label=METRIC_UNITS[metric_id],
                scale="quantile-0.99",
                metric_id=metric_id,
            )
        panels.append(spec)
    return tuple(panels)


# Bout-derived outcomes come from the single shared detector, so they are rows
# of one metric-free figure rather than six per-metric variants.
BOUT_OUTCOME_PANELS = (
    PanelSpec(
        key="movement-probability",
        title="Movement probability",
        column="Movement probability",
        coverage=DETECTOR_COVERAGE,
        colorbar_label="Probability (0-1)",
        cmap_family="probability",
        plotly_colorscale="Magma",
        scale="fixed",
        vmax=1.0,
    ),
    PanelSpec(
        key="fraction-time-moving",
        title="Fraction time moving",
        column="Fraction time moving",
        coverage=DETECTOR_COVERAGE,
        colorbar_label="Fraction (0-1)",
        cmap_family="probability",
        plotly_colorscale="Magma",
        scale="fixed",
        vmax=1.0,
    ),
    PanelSpec(
        key="bout-rate",
        title="Bout initiation rate",
        column="Bout rate per minute",
        coverage=DETECTOR_COVERAGE,
        colorbar_label="Bouts per minute",
        plotly_colorscale="Plasma",
        scale="quantile-0.99",
    ),
)

FIGURE_SPECS = {
    "total-activity-raw": FigureSpec(
        figure_id="total-activity-raw",
        title="total activity (raw)",
        panels=_metric_panels(
            "Total activity mean",
            "Valid expected fraction",
            scaled=False,
        ),
        shared_colorbar=False,
        description=(
            "Mean metric value per trial and time bin, in native units. Each "
            "row has its own colour scale because the metrics are not "
            "commensurable."
        ),
    ),
    "total-activity-scaled": FigureSpec(
        figure_id="total-activity-scaled",
        title="total activity (two-layer scaled)",
        panels=_metric_panels(
            "Scaled total activity",
            "Valid expected fraction",
            scaled=True,
        ),
        shared_colorbar=True,
        description=(
            "Per-trial two-layer scaled activity on a common 0-1 scale: "
            "frame-level P10-P90 from samples earlier than -15 s, then a "
            "second P10-P90 over pre-onset bins, clipped to the unit interval."
        ),
    ),
    "conditional-intensity-raw": FigureSpec(
        figure_id="conditional-intensity-raw",
        title="conditional movement intensity (raw)",
        panels=_metric_panels(
            "Conditional intensity mean",
            DETECTOR_COVERAGE,
            scaled=False,
        ),
        shared_colorbar=False,
        description=(
            "Mean metric value over frames inside a detected bout: intensity "
            "given that the animal was moving, in native units."
        ),
    ),
    "bout-outcomes": FigureSpec(
        figure_id="bout-outcomes",
        title="bout-detection outcomes",
        panels=BOUT_OUTCOME_PANELS,
        shared_colorbar=False,
        description=(
            "Outcomes that depend only on the shared metric-independent "
            "detector. Rows are different outcomes, not different metrics."
        ),
    ),
}


def _panel_scale(
    values: np.ndarray,
    panel: PanelSpec,
) -> tuple[float, float, str]:
    finite = values[np.isfinite(values)]
    vmin = float(panel.vmin)
    if panel.scale == "fixed":
        vmax = float(panel.vmax if panel.vmax is not None else 1.0)
        description = f"linear, fixed [{vmin:g}, {vmax:g}]"
    else:
        vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
        vmax = max(vmax, np.finfo(float).eps)
        description = (
            f"linear, vmin={vmin:g}, vmax={vmax:g} (panel 99th percentile)"
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


def _panel_cmap_name(panel: PanelSpec, theme) -> str:
    if panel.cmap_family == "probability":
        return theme.probability_cmap
    return theme.intensity_cmap


def _panel_pivot(
    selected: pd.DataFrame,
    panel: PanelSpec,
) -> pd.DataFrame:
    """Pivot one panel's values into trials x time bins, masking low coverage."""
    if panel.metric_id is not None:
        rows = selected[selected["Metric ID"].astype(str) == panel.metric_id]
    else:
        # Bout outcomes are identical for every metric because one detector
        # produced them; take a single metric's rows to avoid duplicates.
        metric_ids = [
            metric_id
            for metric_id in METRIC_LABELS
            if metric_id in set(selected["Metric ID"].astype(str))
        ]
        if not metric_ids:
            raise ValueError("Profiles contain no recognized metric rows.")
        rows = selected[selected["Metric ID"].astype(str) == metric_ids[0]]
    rows = rows.copy()
    rows.loc[rows[panel.coverage] < COVERAGE_THRESHOLD, panel.column] = np.nan
    return rows.pivot(
        index="Trial number",
        columns="Time bin center (s)",
        values=panel.column,
    )


def _candidate_heatmap_figure(
    profiles: pd.DataFrame,
    trial_type: str,
    figure_id: str,
) -> tuple[plt.Figure, list[str], dict[str, dict[str, str]]]:
    if figure_id not in FIGURE_SPECS:
        raise ValueError(f"Unknown candidate figure: {figure_id}")
    spec = FIGURE_SPECS[figure_id]
    theme = apply_theme()
    selected = profiles[profiles["Trial type"].astype(str) == trial_type]
    figure, axes = stacked_subplots(
        len(spec.panels),
        width_mm=DOUBLE_COLUMN_MM,
        row_height_mm=26.0,
        sharex=True,
        theme=theme,
    )
    duration_s = stimulus_duration_s(trial_type)
    panel_ids: list[str] = []
    artist_mappings: dict[str, dict[str, str]] = {}
    images: list[tuple[str, object, PanelSpec, tuple[float, float]]] = []
    for index, (axis, panel) in enumerate(zip(axes, spec.panels)):
        panel_id = chr(ord("A") + index)
        panel_ids.append(panel_id)
        cmap = heatmap_cmap(_panel_cmap_name(panel, theme), theme)
        pivot = _panel_pivot(selected, panel)
        values = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
        vmin, vmax, scale_description = _panel_scale(
            np.asarray(values.filled(np.nan)),
            panel,
        )
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
        heatmap_id = f"heatmap__{panel_id.lower()}__{panel.key}__{figure_id}"
        image.set_gid(heatmap_id)
        images.append((heatmap_id, image, panel, (vmin, vmax)))
        artist_mappings[heatmap_id] = {
            "x_field": "Time bin center (s)",
            "y_field": "Trial number",
            "value_field": panel.column,
            "coverage_field": panel.coverage,
            "coverage_threshold": str(COVERAGE_THRESHOLD),
            "display_scale": scale_description,
            "cmap": cmap.name if hasattr(cmap, "name") else str(cmap),
            "shared_detector": str(panel.metric_id is None).lower(),
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
        style_axes(
            axis,
            theme=theme,
            show_xticks=index == len(spec.panels) - 1,
            show_yticks=True,
            xlabel=f"Time from {trial_type} onset (s)",
            ylabel="Trial",
        )
        axis.set_title(panel.title, loc="left")

    if not images:
        raise ValueError("Candidate heatmap has no panels.")
    if spec.shared_colorbar:
        # Every row is already on the same scale, so one colorbar is honest.
        heatmap_id, image, panel, _ = images[0]
        colorbar = figure.colorbar(
            image,
            ax=list(axes),
            fraction=0.035,
            pad=0.02,
        )
        colorbar.set_label(panel.colorbar_label)
        colorbar.ax.set_gid("axes__colorbar__main")
        artist_mappings["colorbar"] = {
            "role": "colorbar",
            "label": panel.colorbar_label,
            "shared": "true",
        }
        colorbar_ids = ["colorbar"]
    else:
        # Rows carry different units or scales, so each gets its own colorbar
        # and no cross-row colour comparison is implied.
        colorbar_ids = []
        for axis, (heatmap_id, image, panel, scale) in zip(axes, images):
            colorbar = figure.colorbar(
                image,
                ax=axis,
                fraction=0.035,
                pad=0.02,
            )
            colorbar.set_label(panel.colorbar_label)
            colorbar_id = f"colorbar__{panel.key}"
            colorbar.ax.set_gid(f"axes__{colorbar_id}")
            artist_mappings[colorbar_id] = {
                "role": "colorbar",
                "label": panel.colorbar_label,
                "shared": "false",
                "vmin": f"{scale[0]:g}",
                "vmax": f"{scale[1]:g}",
            }
            colorbar_ids.append(colorbar_id)
    figure.suptitle(
        f"{recording_id_from_profiles(profiles)} — {trial_type}-aligned "
        f"{spec.title}"
    )
    return figure, panel_ids + colorbar_ids, artist_mappings


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
    figure_id: str = "total-activity-raw",
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
    if figure_id not in FIGURE_SPECS:
        raise ValueError(f"Unknown candidate figure: {figure_id}")
    spec = FIGURE_SPECS[figure_id]
    source_path = Path(__file__).resolve()
    version = figure_version_tag(route)
    reproduction = (
        "python -m classical_conditioning figure-candidate-profiles "
        f"--project-dir \"{project_dir}\" --recording-id {recording_id} "
        f"--trial-type {trial_type} --figure {figure_id} --mode {mode.value} "
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
            / f"candidate_{trial_type.lower()}_{figure_id}-{version}.html"
        )
        if output.exists() and not overwrite:
            raise FileExistsError(f"Interactive figure exists: {output}")
        selected = profiles[profiles["Trial type"].astype(str) == trial_type]
        figure = make_subplots(
            rows=len(spec.panels),
            cols=1,
            shared_xaxes=True,
            subplot_titles=[panel.title for panel in spec.panels],
        )
        for row, panel in enumerate(spec.panels, start=1):
            pivot = _panel_pivot(selected, panel)
            values = pivot.to_numpy(dtype=float)
            vmin, vmax, _ = _panel_scale(values, panel)
            yaxis_name = "yaxis" if row == 1 else f"yaxis{row}"
            domain = getattr(figure.layout, yaxis_name).domain
            figure.add_trace(
                go.Heatmap(
                    x=pivot.columns,
                    y=pivot.index,
                    z=values,
                    colorscale=panel.plotly_colorscale,
                    zmin=vmin,
                    zmax=vmax,
                    colorbar={
                        "title": panel.colorbar_label,
                        "x": 1.02,
                        "y": (domain[0] + domain[1]) / 2,
                        "len": domain[1] - domain[0],
                        "yanchor": "middle",
                    },
                    name=panel.key,
                ),
                row=row,
                col=1,
            )
        figure.update_layout(
            title=f"{recording_id} — {trial_type}-aligned {spec.title}",
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
                        f"candidate-{trial_type.lower()}-{figure_id}-{version}"
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
        profiles, trial_type, figure_id
    )
    output_root = (
        project_dir
        / "Figures"
        / ("Publication" if mode == FigureMode.PUBLICATION else "PNG")
        / recording_id
    )
    output_base = (
        output_root / f"candidate_{trial_type.lower()}_{figure_id}-{version}"
    )
    provenance = FigureProvenance(
        figure_id=f"candidate-{trial_type.lower()}-{figure_id}-{version}",
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
