"""Render archived bin-first Figure 1 example heatmaps for comparison.

The paper-review Figure 1E uses frame-first scaling in
``render_log_scaled_vigor_heatmaps.py``.
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

import classical_conditioning.figures.per_trial_scaled_vigor as scaling_module
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import METRIC_COLUMNS, METRIC_DISPLAY_NAMES
from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
from classical_conditioning.figures.per_trial_scaled_vigor import scale_per_trial_conditional_vigor
from classical_conditioning.figures.theme import apply_theme, mm_to_in


FISH = (("20221115_07", "Delay", "delay"), ("20221115_09", "Control", "control"))
PHASES = (("Pre-Train", 5, 14), ("Train", 15, 64), ("Test", 65, 94))
SELECTED_TRIALS = (9, 17, 63, 66, 93)
PROFILE_COLUMNS = [
    "Recording ID", "Trial type", "Trial number", "Time bin center (s)",
    "Metric ID", "Conditional intensity mean", "Valid expected fraction",
]


def _read_verified_profiles(project: Path, fish_id: str, expected_condition: str):
    path = (
        project / "Processed data" / fish_id
        / "candidate_temporal_outcomes-corrected-v3.parquet"
    )
    marker_path = (
        project / "Metadata"
        / f"{fish_id}_candidate-temporal-outcomes-corrected-v3_complete.json"
    )
    manifest_path = project / "Metadata" / f"{fish_id}_source_manifest.json"
    marker = json.loads(marker_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if marker.get("status") != "complete" or marker.get("recording_id") != fish_id:
        raise ValueError(f"Incomplete temporal artifact: {fish_id}")
    if sha256_file(path) != marker["profiles_sha256"]:
        raise ValueError(f"Temporal artifact hash mismatch: {fish_id}")
    if manifest["recording_id"] != fish_id or manifest["condition_id"] != expected_condition:
        raise ValueError(f"Fish condition differs from source manifest: {fish_id}")
    profiles = pq.read_table(path, columns=PROFILE_COLUMNS).to_pandas()
    if set(profiles["Recording ID"].astype(str)) != {fish_id}:
        raise ValueError(f"Temporal artifact has another fish identity: {fish_id}")
    inputs = [
        {"path": str(item), "sha256": sha256_file(item)}
        for item in (path, marker_path, manifest_path)
    ]
    return profiles, inputs


def _matrix(data: pd.DataFrame, fish_id: str, metric_id: str, start: int, end: int):
    selected = data.loc[
        data["Recording ID"].eq(fish_id) & data["Metric ID"].eq(metric_id)
    ]
    times = np.arange(-20.0, 20.0, 0.5) + 0.25
    return selected.pivot(
        index="Trial number", columns="Time bin center (s)", values="Per-trial scaled vigor"
    ).reindex(index=range(start, end + 1), columns=times).to_numpy(dtype=float)


def _render(data: pd.DataFrame, metric_id: str, *, cmap_name: str = "managua_r"):
    theme = apply_theme()
    figure, axes = plt.subplots(
        3, 2, figsize=mm_to_in(183, 165),
        gridspec_kw={"height_ratios": [10, 50, 30]},
        sharex=True, constrained_layout=True,
    )
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("black")
    mappings = {}
    panel_ids = []
    image = None
    for row, (phase, start, end) in enumerate(PHASES):
        for column, (fish_id, name, _) in enumerate(FISH):
            axis = axes[row, column]
            panel_id = f"{name.lower()}_{phase.lower().replace('-', '_')}"
            panel_ids.append(panel_id)
            values = np.ma.masked_invalid(_matrix(data, fish_id, metric_id, start, end))
            image = axis.imshow(
                values, origin="upper", aspect="auto", interpolation="nearest",
                extent=(-20, 20, end - start + 1, 0), vmin=0, vmax=1,
                cmap=cmap, rasterized=True,
            )
            image_id = f"heatmap__{panel_id}__{metric_id}"
            image.set_gid(image_id)
            mappings[image_id] = {
                "recording_id": fish_id, "metric_id": metric_id,
                "value_field": "Per-trial scaled vigor",
                "baseline": (
                    "pre-CS -20 to 0 s; log bout-mean movement frames P10-P90; scale frames before binning"
                    if "Signal semantics" in data.columns and
                    data["Signal semantics"].astype(str).str.contains("pre_minus20_to_0").all()
                    else "log bout-mean movement frames P10-P90; scale frames before binning"
                    if "Signal semantics" in data.columns and
                    data["Signal semantics"].astype(str).str.contains("frame_scaled_per_trial").all()
                    else "pre-CS time < -15 s; covered bin P10-P90"
                ),
                "cmap": cmap_name, "vmin": "0", "vmax": "1",
            }
            for seconds in (0, 10):
                axis.axvline(seconds, color=theme.cs_color, linewidth=0.7)
            for trial in SELECTED_TRIALS:
                if start <= trial <= end:
                    axis.plot(20.35, trial - start + 0.5, marker="<", color="black",
                              markersize=3.5, clip_on=False)
            axis.set_xlim(-20, 20)
            axis.set_ylim(end - start + 1, 0)
            axis.set_yticks([])
            axis.set_ylabel(phase if column == 0 else "")
            if row == 0:
                axis.set_title(f"{name}  {fish_id}")
            if row == 2:
                axis.set_xlabel("Time relative to CS onset (s)")
            else:
                axis.tick_params(labelbottom=False)
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.025,
                    label="Per-trial scaled vigor (0–1)")
    panel_ids.append("colorbar")
    figure.suptitle(f"Figure 1 example fish · {METRIC_DISPLAY_NAMES[metric_id]}")
    return figure, panel_ids, mappings


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
    profiles = []
    inputs = []
    for fish_id, _, condition in FISH:
        data, verified = _read_verified_profiles(project, fish_id, condition)
        profiles.append(data)
        inputs.extend(verified)
    scaled = scale_per_trial_conditional_vigor(
        pd.concat(profiles, ignore_index=True), transform="log",
    )
    scaled = scaled.loc[scaled["Time bin center (s)"].between(-20, 20)].copy()
    data_path = output / "figure-1-example-per-trial-scaled-vigor_panel-data.parquet"
    if data_path.exists() and not args.overwrite:
        raise FileExistsError(data_path)
    pq.write_table(pa.Table.from_pandas(scaled, preserve_index=False), data_path, compression="zstd")
    inputs.extend({"path": str(path), "sha256": sha256_file(path)} for path in (
        Path(scaling_module.__file__).resolve(), data_path,
    ))
    source = Path(__file__).resolve()
    reproduction = (
        f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src ./.venv/bin/python "
        f"scripts/render_ssd_per_trial_example_heatmaps.py --project-dir '{project}' "
        f"--output-dir '{output}' --mode {args.mode} --overwrite"
    )
    for metric_id in METRIC_COLUMNS:
        figure, panel_ids, mappings = _render(scaled, metric_id)
        base = output / f"figure-1-E_delay-control_{metric_id}"
        try:
            result = export_matplotlib_figure(
                figure, base,
                FigureProvenance(
                    figure_id="figure-1-E-delay-control-per-trial-scaled-vigor",
                    analysis_recipe="corrected-per-trial-conditional-vigor-scaling",
                    source_file=str(source), source_symbol="main",
                    source_hash=sha256_file(source),
                    reproduction_snippet=reproduction,
                    input_artifacts=tuple(inputs), artist_mappings=mappings,
                ),
                mode=FigureMode(args.mode), panel_ids=panel_ids, overwrite=args.overwrite,
            )
        finally:
            plt.close(figure)
        for path in (*result.outputs, result.sidecar):
            print(path)
    print(data_path)


if __name__ == "__main__":
    main()
