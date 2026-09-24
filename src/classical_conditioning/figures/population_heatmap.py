"""Frozen-cohort CS heatmaps with visible contributing-fish coverage.

These are descriptive Figure 2 precursors, not approved manuscript panels.
Each fish contributes at most one value per trial/time bin. The displayed
signal is 0–1 scaled *total activity across valid frames*, including valid
inactivity; it is not vigor conditional on movement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.candidate_runner import _verify_temporal
from classical_conditioning.analysis.movement_state import resolve_candidate_metric_source
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.exceptions import ConfigurationError, SchemaValidationError
from classical_conditioning.figures.cohort_response import _load_primary_cohort, _metric_slug
from classical_conditioning.figures.export import (
    FigureExportResult,
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.figures.theme import DOUBLE_COLUMN_MM, apply_theme, heatmap_cmap, mm_to_in, style_axes


def summarize_population_heatmap(
    profiles: pd.DataFrame,
    *,
    metric_id: str,
    condition_by_recording: dict[str, str],
    fish_by_recording: dict[str, str],
    minimum_coverage: float = 0.9,
) -> pd.DataFrame:
    """Average equally weighted fish and retain coverage for every observed bin."""
    required = {
        "Recording ID", "Trial type", "Trial number", "Time bin center (s)",
        "Metric ID", "Scaled total activity", "Valid expected fraction",
    }
    missing = required.difference(profiles.columns)
    if missing:
        raise SchemaValidationError(f"Population profiles lack fields: {sorted(missing)}")
    if not 0 <= minimum_coverage <= 1:
        raise ConfigurationError("minimum_coverage must be between 0 and 1.")
    selected = profiles.loc[
        (profiles["Trial type"].astype(str) == "CS")
        & (profiles["Metric ID"].astype(str) == metric_id),
        ["Recording ID", "Trial number", "Time bin center (s)",
         "Scaled total activity", "Valid expected fraction"],
    ].copy()
    if selected.empty:
        raise ConfigurationError(f"No CS temporal profiles for metric {metric_id!r}.")
    selected["condition_id"] = selected["Recording ID"].astype(str).map(condition_by_recording)
    selected["fish_id"] = selected["Recording ID"].astype(str).map(fish_by_recording)
    if selected[["condition_id", "fish_id"]].isna().any().any():
        raise SchemaValidationError("Population profile has fish absent from the reviewed cohort.")
    identity = ["condition_id", "fish_id", "Trial number", "Time bin center (s)"]
    if selected.duplicated(identity).any():
        raise SchemaValidationError("More than one value exists for a fish/trial/time bin.")
    values = pd.to_numeric(selected["Scaled total activity"], errors="coerce")
    coverage = pd.to_numeric(selected["Valid expected fraction"], errors="coerce")
    if ((values.notna()) & (~values.between(0, 1))).any():
        raise SchemaValidationError("Scaled total activity must lie within 0–1.")
    selected["eligible_activity"] = values.where(coverage >= minimum_coverage)
    totals = {
        condition: len(set(fish_ids))
        for condition, fish_ids in selected.groupby("condition_id")["fish_id"]
    }
    group = (
        selected.groupby(["condition_id", "Trial number", "Time bin center (s)"], observed=True)
        ["eligible_activity"]
        .agg([("Mean scaled total activity", "mean"), ("Contributing fish", "count")])
        .reset_index()
    )
    group["Total cohort fish"] = group["condition_id"].map(totals).astype(int)
    group["Fish coverage fraction"] = group["Contributing fish"] / group["Total cohort fish"]
    group["Signal semantics"] = "scaled_total_activity_all_valid_frames_including_zero"
    group["Metric ID"] = metric_id
    group["Minimum valid-frame fraction"] = minimum_coverage
    return group


def _matrix(frame: pd.DataFrame, field: str, trials: np.ndarray, times: np.ndarray) -> np.ndarray:
    return (
        frame.pivot(index="Trial number", columns="Time bin center (s)", values=field)
        .reindex(index=trials, columns=times)
        .to_numpy(dtype=float)
    )


def build_population_heatmap_figure(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    mode: FigureMode,
    minimum_coverage: float = 0.9,
    overwrite: bool = False,
) -> FigureExportResult:
    """Render a matched-control population heatmap plus fish-coverage strips."""
    if mode == FigureMode.INTERACTIVE:
        raise ConfigurationError("Population heatmaps support static or publication mode.")
    project_dir = project_dir.resolve()
    cohort = _load_primary_cohort(project_dir, cohort_id)
    if cohort.metric_recipe != "tail-candidate-corrected":
        raise ConfigurationError("Population heatmap requires corrected cohort outcomes.")
    source = resolve_candidate_metric_source(metric_recipe=cohort.metric_recipe)
    frames: list[pd.DataFrame] = []
    input_artifacts: list[dict[str, Any]] = list(cohort.input_artifacts)
    for recording_id in cohort.recording_ids:
        # The same verifier used by the runner authenticates the temporal
        # artifact against corrected metrics, movement state, and protocol.
        _verify_temporal(project_dir, recording_id, cohort.experiment_id, source)
        path = (
            project_dir / "Processed data" / recording_id / source.temporal_artifact_name
        )
        frames.append(pq.read_table(path).to_pandas())
        input_artifacts.append({
            "recipe": source.temporal_recipe, "path": str(path), "sha256": sha256_file(path)
        })
    data = summarize_population_heatmap(
        pd.concat(frames, ignore_index=True),
        metric_id=metric_id,
        condition_by_recording=cohort.condition_by_recording,
        fish_by_recording=cohort.fish_by_recording,
        minimum_coverage=minimum_coverage,
    )
    # The draft comparison uses a fixed ±20-s CS window. This display choice
    # does not alter the authenticated full-window per-fish temporal profiles.
    data = data.loc[data["Time bin center (s)"].between(-20, 20)].copy()
    if data.empty:
        raise ConfigurationError("No population heatmap bins fall in the ±20-s CS window.")
    conditions = tuple(dict.fromkeys(cohort.condition_by_recording.values()))
    paired = tuple(condition for condition in conditions if condition != "control")
    if "control" not in conditions or len(paired) != 1:
        raise ConfigurationError("Population heatmap requires exactly one paired condition and its matched control.")
    ordered = ("control", paired[0])
    panel_data = (
        project_dir / "Processed data" / "Analyses" / analysis_id
        / f"population-heatmap_{_metric_slug(metric_id)}.parquet"
    )
    panel_summary = panel_data.with_suffix(".json")
    metric_code = (
        Path(__file__).resolve().parents[1]
        / "preprocessing" / "candidate_metric_kernel.py"
    )
    metric_definition_hash = hashlib.sha256(
        metric_id.encode("utf-8") + b"\0" + metric_code.read_bytes()
    ).hexdigest()
    with artifact_staging(panel_data.parent, prefix=f".{panel_data.stem}-") as staging:
        staged_data = staging / panel_data.name
        staged_summary = staging / panel_summary.name
        pq.write_table(pa.Table.from_pandas(data, preserve_index=False), staged_data, compression="zstd")
        write_json_atomic(staged_summary, {
            "recipe": "cohort-population-heatmap",
            "cohort_id": cohort_id,
            "cohort_hash": cohort.cohort_hash,
            "metric_id": metric_id,
            "metric_sha256": metric_definition_hash,
            "signal_semantics": "scaled_total_activity_all_valid_frames_including_zero",
            "aggregation": "one bin per fish, then equal-fish mean",
            "minimum_valid_frame_fraction": minimum_coverage,
            "panel_data_sha256": sha256_file(staged_data),
            "input_artifacts": input_artifacts,
        })
        publish_transaction(
            ((staged_data, panel_data), (staged_summary, panel_summary)),
            staging, overwrite=overwrite,
        )
    input_artifacts.append({"recipe": "cohort-population-heatmap", "path": str(panel_data), "sha256": sha256_file(panel_data)})

    theme = apply_theme()
    figure, axes = plt.subplots(
        2, 2, figsize=mm_to_in(DOUBLE_COLUMN_MM, 112),
        gridspec_kw={"height_ratios": [4, 1]}, sharex="col", sharey="row",
        constrained_layout=True,
    )
    mappings: dict[str, dict[str, Any]] = {}
    main_image = coverage_image = None
    trials = np.sort(data["Trial number"].unique())
    times = np.sort(data["Time bin center (s)"].unique())
    time_step = float(np.median(np.diff(times))) if len(times) > 1 else 0.5
    trial_step = float(np.median(np.diff(trials))) if len(trials) > 1 else 1.0
    extent = (times[0] - time_step / 2, times[-1] + time_step / 2,
              trials[0] - trial_step / 2, trials[-1] + trial_step / 2)
    try:
        for index, condition in enumerate(ordered):
            subset = data.loc[data["condition_id"] == condition]
            activity = _matrix(subset, "Mean scaled total activity", trials, times)
            coverage = _matrix(subset, "Fish coverage fraction", trials, times)
            main_image = axes[0, index].imshow(
                np.ma.masked_invalid(activity), origin="lower", aspect="auto",
                extent=extent, vmin=0, vmax=1,
                cmap="managua_r",
                interpolation="nearest",
            )
            main_id = f"heatmap__{condition}__scaled-total-activity"
            main_image.set_gid(main_id)
            mappings[main_id] = {
                "value_field": "Mean scaled total activity",
                "coverage_field": "Contributing fish",
                "x_field": "Time bin center (s)", "y_field": "Trial number",
                "units": "0–1 scaled activity across all valid frames",
                "cmap": main_image.get_cmap().name,
                "display_scale": "linear, fixed [0, 1]",
                "cohort_hash": cohort.cohort_hash,
            }
            coverage_image = axes[1, index].imshow(
                np.ma.masked_invalid(coverage), origin="lower", aspect="auto",
                extent=extent, vmin=0, vmax=1, cmap="managua_r", interpolation="nearest",
            )
            coverage_id = f"heatmap__{condition}__fish-coverage"
            coverage_image.set_gid(coverage_id)
            mappings[coverage_id] = {
                "value_field": "Fish coverage fraction",
                "count_field": "Contributing fish",
                "denominator_field": "Total cohort fish",
                "x_field": "Time bin center (s)", "y_field": "Trial number",
                "units": "fraction of frozen cohort fish",
                "cmap": coverage_image.get_cmap().name,
            }
            fish_count = int(subset["Total cohort fish"].iloc[0])
            axes[0, index].set_title(f"{condition} (n={fish_count}) — scaled total activity")
            axes[1, index].set_title("Contributing-fish coverage")
            axes[1, index].set_xlabel("Time relative to CS onset (s)")
            for axis in axes[:, index]:
                axis.axvline(0, color="#009E73", linewidth=0.7)
                axis.axvline(10, color="#009E73", linewidth=0.7, linestyle="--")
            style_axes(axes[0, index], theme=theme, show_xticks=False, show_yticks=index == 0)
            style_axes(axes[1, index], theme=theme, show_yticks=index == 0)
        axes[0, 0].set_ylabel("Global CS trial")
        axes[1, 0].set_ylabel("Global CS trial")
        figure.colorbar(main_image, ax=axes[0, :], label="Mean scaled total activity (0–1)")
        figure.colorbar(coverage_image, ax=axes[1, :], label="Fish coverage fraction")
        output_base = (
            project_dir / "Figures"
            / ("Publication" if mode == FigureMode.PUBLICATION else "PNG")
            / "Analyses" / analysis_id
            / f"cohort-population-heatmap_{_metric_slug(metric_id)}"
        )
        source_path = Path(__file__).resolve()
        return export_matplotlib_figure(
            figure,
            output_base,
            FigureProvenance(
                figure_id="cohort-population-heatmap",
                analysis_recipe="cohort-population-heatmap",
                source_file=str(source_path),
                source_symbol="build_population_heatmap_figure",
                source_hash=sha256_file(source_path),
                reproduction_snippet=(
                    "from classical_conditioning.figures.population_heatmap import "
                    "build_population_heatmap_figure"
                ),
                input_artifacts=tuple(input_artifacts),
                cohort_hash=cohort.cohort_hash,
                artist_mappings=mappings,
            ),
            mode=mode,
            panel_ids=["control", paired[0], "control-coverage", f"{paired[0]}-coverage", "activity-colorbar", "coverage-colorbar"],
            overwrite=overwrite,
        )
    finally:
        plt.close(figure)
