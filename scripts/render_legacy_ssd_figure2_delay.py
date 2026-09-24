"""Render descriptive Figure 2 A/D/G Delay-control panels from the SSD cohort.

The SSD stores completed corrected artifacts with versioned names. This adapter
authenticates them, uses the current pipeline's equal-fish summary functions,
and exports one A/D/G set for each candidate vigor metric. The A heatmap uses
log conditional vigor before fish/trial and pooled/trial P10/P90 scaling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import classical_conditioning.figures.cohort_response as response_module
import classical_conditioning.figures.per_trial_scaled_vigor as scaling_module
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.cohort_response import (
    _plot_selected_blocks,
    summarize_selected_block_ratios,
    summarize_trial_ratios,
)
from classical_conditioning.figures.example_traces import METRIC_COLUMNS, METRIC_DISPLAY_NAMES
from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
from classical_conditioning.figures.per_trial_scaled_vigor import (
    scale_per_trial_conditional_vigor,
    summarize_legacy_pooled_scaled_vigor,
)
from classical_conditioning.figures.theme import apply_theme, mm_to_in, style_axes


COHORT_ID = "allDelay-full-v1"
MINIMUM_COVERAGE = 0.9
PROFILE_COLUMNS = [
    "Recording ID", "Trial type", "Trial number", "Time bin center (s)",
    "Metric ID", "Conditional intensity mean", "Valid expected fraction",
]
OUTCOME_COLUMNS = [
    "recording_id", "fish_id", "condition_id", "alignment", "trial_number",
    "metric_id", "response_total_activity", "baseline_total_activity",
]


def _verify_cohort(project: Path):
    cohort_root = project / "Processed data" / "Cohorts" / COHORT_ID
    manifest_path = cohort_root / "cohort-manifest-v1.parquet"
    review_path = cohort_root / "cohort-manifest-v1_review.csv"
    marker_path = project / "Metadata" / f"{COHORT_ID}_cohort-manifest-v1_complete.json"
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "complete" or marker.get("cohort_id") != COHORT_ID:
        raise ValueError(f"Incomplete cohort manifest: {marker_path}")
    if sha256_file(manifest_path) != marker["manifest_sha256"]:
        raise ValueError("Cohort manifest hash mismatch")
    if sha256_file(review_path) != marker["review_copy_sha256"]:
        raise ValueError("Reviewed cohort copy hash mismatch")
    cohort = pd.read_csv(review_path)
    selected = cohort.loc[cohort["primary_included"].eq(True)].copy()
    if selected.empty or not selected["review_status"].eq("approved").all():
        raise ValueError("Cohort contains no approved primary fish")
    if selected["recording_id"].duplicated().any() or selected["fish_id"].duplicated().any():
        raise ValueError("Cohort fish identities are not unique")
    if set(selected["condition_id"]) != {"control", "delay"}:
        raise ValueError("Expected a Delay/control matched cohort")
    inputs = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in (manifest_path, review_path, marker_path)
    ]
    return selected, marker["logical_content_sha256"], inputs


def _read_verified_recording(project: Path, row: pd.Series):
    recording_id = str(row["recording_id"])
    root = project / "Processed data" / recording_id
    specs = (
        (
            root / "candidate_temporal_outcomes-corrected-v3.parquet",
            project / "Metadata" / f"{recording_id}_candidate-temporal-outcomes-corrected-v3_complete.json",
            "profiles_sha256",
        ),
        (
            root / "candidate-trial-outcomes-corrected-v1.parquet",
            project / "Metadata" / f"{recording_id}_candidate-trial-outcomes-corrected-v1_complete.json",
            "artifact_sha256",
        ),
    )
    inputs = []
    for path, marker_path, hash_key in specs:
        marker = json.loads(marker_path.read_text())
        expected = marker[hash_key]
        if isinstance(expected, dict):
            expected = expected["outcomes"]
        if marker.get("status") != "complete" or marker.get("recording_id") != recording_id:
            raise ValueError(f"Incomplete artifact marker: {marker_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch: {path}")
        inputs.append({"path": str(path), "sha256": actual})
        inputs.append({"path": str(marker_path), "sha256": sha256_file(marker_path)})
    profile = pq.read_table(specs[0][0], columns=PROFILE_COLUMNS).to_pandas()
    profile = profile.loc[
        profile["Trial type"].eq("CS")
        & profile["Trial number"].between(5, 94)
    ].copy()
    outcome = pq.read_table(specs[1][0], columns=OUTCOME_COLUMNS).to_pandas()
    if set(profile["Recording ID"].astype(str)) != {recording_id}:
        raise ValueError(f"Temporal-profile recording identity mismatch: {recording_id}")
    if set(outcome["recording_id"].astype(str)) != {recording_id}:
        raise ValueError(f"Trial-outcome recording identity mismatch: {recording_id}")
    if set(outcome["fish_id"].astype(str)) != {str(row["fish_id"])}:
        raise ValueError(f"Trial-outcome fish identity mismatch: {recording_id}")
    if set(outcome["condition_id"].astype(str)) != {str(row["condition_id"])}:
        raise ValueError(f"Trial-outcome condition mismatch: {recording_id}")
    return profile, outcome, inputs


def _heatmap_matrix(frame: pd.DataFrame, field: str, trials, times):
    return frame.pivot(index="Trial number", columns="Time bin center (s)", values=field).reindex(
        index=trials, columns=times
    ).to_numpy(dtype=float)


def _plot_heatmap(data: pd.DataFrame, metric_id: str, counts: dict[str, int]):
    theme = apply_theme()
    figure, axes = plt.subplots(
        1, 2, figsize=mm_to_in(183, 95), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    trials = np.arange(5, 95)
    times = np.sort(data["Time bin center (s)"].unique())
    extent = (float(times.min() - 0.25), float(times.max() + 0.25), 94.5, 4.5)
    mappings = {}
    main = None
    for column, condition in enumerate(("control", "delay")):
        subset = data.loc[data["condition_id"].eq(condition)]
        activity = np.ma.masked_invalid(_heatmap_matrix(subset, "Mean per-trial scaled vigor", trials, times))
        axis = axes[0, column]
        main = axis.imshow(
            activity, origin="upper", aspect="auto", interpolation="nearest",
            extent=extent, vmin=0, vmax=1, cmap="managua_r", rasterized=True,
        )
        main_id = f"heatmap__{condition}__per_trial_scaled_vigor"
        main.set_gid(main_id)
        mappings[main_id] = {
            "condition": condition, "metric_id": metric_id,
            "signal": "unbounded fish scaled conditional vigor pooled by bin then trial pre-CS P10-P90 rescaled",
            "minimum_valid_expected_fraction": str(MINIMUM_COVERAGE),
            "cmap": "managua_r", "vmin": "0", "vmax": "1",
        }
        axis.set_title(f"{condition.capitalize()} (n={counts[condition]})")
        axis.set_xlabel("Time relative to CS onset (s)")
        for seconds in (0, 10):
            axis.axvline(seconds, color=theme.cs_color, linewidth=0.7)
        for boundary in (14.5, 64.5):
            axis.axhline(boundary, color="white", linewidth=0.7)
        axis.set_xlim(-20, 20)
        axis.set_ylim(94.5, 4.5)
        style_axes(axis, theme=theme, show_xticks=True, show_yticks=column == 0)
    axes[0, 0].set_ylabel("Global CS trial")
    figure.colorbar(main, ax=axes[0, :], fraction=0.025, pad=0.025,
                    label="Pooled per-trial scaled vigor (0–1)")
    figure.suptitle(f"Figure 2A review · {METRIC_DISPLAY_NAMES[metric_id]}")
    return figure, ["control", "delay", "vigor_colorbar"], mappings


def _plot_coverage(data: pd.DataFrame, metric_id: str, counts: dict[str, int]):
    theme = apply_theme()
    figure, axes = plt.subplots(
        1, 2, figsize=mm_to_in(183, 95), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    trials = np.arange(5, 95)
    times = np.sort(data["Time bin center (s)"].unique())
    extent = (float(times.min() - 0.25), float(times.max() + 0.25), 94.5, 4.5)
    mappings = {}
    image = None
    for column, condition in enumerate(("control", "delay")):
        subset = data.loc[data["condition_id"].eq(condition)]
        coverage = np.ma.masked_invalid(_heatmap_matrix(subset, "Fish coverage fraction", trials, times))
        axis = axes[0, column]
        image = axis.imshow(
            coverage, origin="upper", aspect="auto", interpolation="nearest",
            extent=extent, vmin=0, vmax=1, cmap="cividis", rasterized=True,
        )
        gid = f"heatmap__{condition}__fish_coverage"
        image.set_gid(gid)
        mappings[gid] = {
            "condition": condition, "metric_id": metric_id,
            "signal": "contributing fish / cohort fish", "cmap": "cividis",
        }
        axis.set_title(f"{condition.capitalize()} (n={counts[condition]})")
        axis.set_xlabel("Time relative to CS onset (s)")
        for seconds in (0, 10):
            axis.axvline(seconds, color=theme.cs_color, linewidth=0.7)
        for boundary in (14.5, 64.5):
            axis.axhline(boundary, color="white", linewidth=0.7)
        axis.set_xlim(-20, 20)
        axis.set_ylim(94.5, 4.5)
        style_axes(axis, theme=theme, show_xticks=True, show_yticks=column == 0)
    axes[0, 0].set_ylabel("Global CS trial")
    figure.colorbar(image, ax=axes[0, :], fraction=0.025, pad=0.025,
                    label="Contributing fish fraction")
    figure.suptitle(f"Figure 2A supplementary coverage · {METRIC_DISPLAY_NAMES[metric_id]}")
    return figure, ["control_coverage", "delay_coverage", "coverage_colorbar"], mappings


def _save_frame(frame: pd.DataFrame, path: Path) -> dict[str, str]:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")
    return {"path": str(path), "sha256": sha256_file(path)}


def _plot_trial_summary(group: pd.DataFrame, metric_id: str):
    """Show equal-fish cohort summaries; retain all fish data in source files."""
    apply_theme()
    figure, axis = plt.subplots(figsize=mm_to_in(183, 83), layout="constrained")
    axis.axhline(1.0, color="0.65", linewidth=0.7)
    colors = {"control": "#00AEEF", "delay": "#EC008C"}
    mappings = {}
    for condition in ("control", "delay"):
        selected = group.loc[group["condition_id"].eq(condition)].sort_values("trial_number")
        x = selected["trial_number"].to_numpy(dtype=float)
        median = selected["Cohort median response / baseline"].to_numpy(dtype=float)
        low = selected["Cohort Q25 response / baseline"].to_numpy(dtype=float)
        high = selected["Cohort Q75 response / baseline"].to_numpy(dtype=float)
        color = colors[condition]
        ribbon = axis.fill_between(x, low, high, color=color, alpha=0.18)
        ribbon_id = f"cohort_iqr__{condition}"
        ribbon.set_gid(ribbon_id)
        line = axis.plot(x, median, color=color, linewidth=1.4,
                         label=f"{condition.capitalize()} (n={int(selected['Fish count'].max())})")[0]
        line_id = f"cohort_median__{condition}"
        line.set_gid(line_id)
        mappings[ribbon_id] = {"condition": condition, "metric_id": metric_id,
                               "spread": "fish IQR", "aggregation": "equal-fish"}
        mappings[line_id] = {"condition": condition, "metric_id": metric_id,
                             "value_field": "Cohort median response / baseline",
                             "aggregation": "equal-fish"}
    axis.set_xlim(1, 94)
    axis.set_ylim(0.65, 1.35)
    axis.set_xticks([5, 15, 25, 35, 45, 55, 65, 75, 85, 94])
    axis.legend(loc="upper right")
    style_axes(axis, xlabel="Global CS trial", ylabel="Activity response / pre-CS baseline")
    axis.set_title(f"Figure 2G review · {METRIC_DISPLAY_NAMES[metric_id]} · cohort median [IQR]")
    return figure, ["G"], mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort, cohort_hash, inputs = _verify_cohort(project)
    profiles, outcomes = [], []
    for _, row in cohort.iterrows():
        profile, outcome, fish_inputs = _read_verified_recording(project, row)
        profiles.append(profile)
        outcomes.append(outcome)
        inputs.extend(fish_inputs)
    profiles = pd.concat(profiles, ignore_index=True)
    scaled_profiles = scale_per_trial_conditional_vigor(
        profiles, minimum_coverage=MINIMUM_COVERAGE, clip=False, transform="log",
    )
    outcomes = pd.concat(outcomes, ignore_index=True)
    by_recording = cohort.set_index("recording_id")["condition_id"].to_dict()
    fish_by_recording = cohort.set_index("recording_id")["fish_id"].to_dict()
    counts = cohort.groupby("condition_id")["fish_id"].nunique().to_dict()
    inputs.extend({"path": str(path), "sha256": sha256_file(path)} for path in (
        Path(response_module.__file__).resolve(),
        Path(scaling_module.__file__).resolve(),
    ))
    source = Path(__file__).resolve()
    reproduction = (
        f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src ./.venv/bin/python "
        f"scripts/render_legacy_ssd_figure2_delay.py --project-dir '{project}' "
        f"--output-dir '{output}' --mode {args.mode} --overwrite"
    )
    mode = FigureMode(args.mode)
    for metric_id in METRIC_COLUMNS:
        heatmap = summarize_legacy_pooled_scaled_vigor(
            scaled_profiles, metric_id=metric_id,
            condition_by_recording=by_recording,
        )
        heatmap = heatmap.loc[heatmap["Trial number"].between(5, 94)].copy()
        block_fish, block_group = summarize_selected_block_ratios(
            outcomes, metric_id=metric_id,
        )
        trial_fish, trial_group = summarize_trial_ratios(outcomes, metric_id=metric_id)
        panel_frames = (
            ("2A", heatmap, _plot_heatmap(heatmap, metric_id, counts)),
            ("2A-coverage", heatmap, _plot_coverage(heatmap, metric_id, counts)),
            ("2D", block_group, _plot_selected_blocks(block_fish, block_group, experiment_name="allDelay")),
            ("2G", trial_group, _plot_trial_summary(trial_group, metric_id)),
        )
        for panel, data, (figure, panel_ids, mappings) in panel_frames:
            panel_output = output / "supplementary" if panel == "2A-coverage" else output
            panel_output.mkdir(parents=True, exist_ok=True)
            data_path = panel_output / f"figure-{panel}_{metric_id}_panel-data.parquet"
            if data_path.exists() and not args.overwrite:
                raise FileExistsError(data_path)
            data_input = _save_frame(data, data_path)
            fish_path = None
            fish_input = None
            if panel in {"2D", "2G"}:
                fish_path = output / f"figure-{panel}_{metric_id}_fish-data.parquet"
                fish_input = _save_frame(
                    block_fish if panel == "2D" else trial_fish, fish_path,
                )
            base = panel_output / f"figure-{panel}_delay-control_{metric_id}"
            try:
                result = export_matplotlib_figure(
                    figure, base,
                    FigureProvenance(
                        figure_id=f"figure-{panel}-delay-control-review",
                        analysis_recipe="corrected-candidate-cohort-descriptive",
                        source_file=str(source), source_symbol="main",
                        source_hash=sha256_file(source),
                        reproduction_snippet=reproduction,
                        input_artifacts=tuple((*inputs, data_input, *((fish_input,) if fish_input else ()))),
                        cohort_hash=cohort_hash, artist_mappings=mappings,
                    ),
                    mode=mode, panel_ids=panel_ids, overwrite=args.overwrite,
                )
            finally:
                plt.close(figure)
            for path in (*result.outputs, result.sidecar, data_path,
                         *((fish_path,) if fish_path else ())):
                print(path)
    print(f"Cohort: {counts['delay']} Delay, {counts['control']} Control")


if __name__ == "__main__":
    main()
