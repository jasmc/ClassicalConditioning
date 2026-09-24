"""Continuous Figure 1 heatmap candidate from all valid frame vigor bins.

Each 0.5 s bin is the mean of valid corrected frame metric values, regardless
of the experimental bout detector. P10/P90 scaling is separately recomputed
for every fish, metric, and trial from covered bins before -15 s. Bins with
fewer than 90% valid expected frames are retained but flagged, so the visual
step trace never invents missing values. This is a distinct signal from the
earlier movement-conditional heatmap candidate.
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

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import METRIC_DISPLAY_NAMES
from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
from classical_conditioning.figures.theme import apply_theme, mm_to_in


FISH = (("20221115_07", "Delay", "delay"), ("20221115_09", "Control", "control"))
PHASES = (("Pre-Train", 5, 14), ("Train", 15, 64), ("Test", 65, 94))
SELECTED_TRIALS = (9, 17, 63, 66, 93)
COLUMNS = ["Recording ID", "Trial type", "Trial number", "Time bin center (s)",
           "Metric ID", "Total activity mean", "Valid expected fraction"]


def load_verified_profile(project: Path, fish: str, condition: str):
    data_path = project / "Processed data" / fish / "candidate_temporal_outcomes-corrected-v3.parquet"
    marker_path = project / "Metadata" / f"{fish}_candidate-temporal-outcomes-corrected-v3_complete.json"
    manifest_path = project / "Metadata" / f"{fish}_source_manifest.json"
    marker, manifest = json.loads(marker_path.read_text()), json.loads(manifest_path.read_text())
    if marker.get("status") != "complete" or marker.get("recording_id") != fish:
        raise ValueError(f"Incomplete temporal profile: {fish}")
    if sha256_file(data_path) != marker["profiles_sha256"]:
        raise ValueError(f"Temporal profile hash mismatch: {fish}")
    if manifest["recording_id"] != fish or manifest["condition_id"] != condition:
        raise ValueError(f"Manifest identity mismatch: {fish}")
    data = pq.read_table(data_path, columns=COLUMNS).to_pandas()
    data = data.loc[data["Trial type"].eq("CS") &
                    data["Trial number"].between(5, 94) &
                    data["Time bin center (s)"].ge(-20) &
                    data["Time bin center (s)"].lt(20)].copy()
    return data, (data_path, marker_path, manifest_path)


def scale_total_vigor(profiles: pd.DataFrame) -> pd.DataFrame:
    keys = ["Recording ID", "Metric ID", "Trial number"]
    if profiles.duplicated([*keys, "Time bin center (s)"]).any():
        raise ValueError("Duplicate fish/metric/trial/bin")
    rows = []
    for _, trial in profiles.groupby(keys, observed=True, sort=False):
        trial = trial.sort_values("Time bin center (s)").copy()
        if len(trial) != 80 or not np.allclose(np.diff(trial["Time bin center (s)"]), .5):
            raise ValueError("A complete 80-bin CS window is required")
        baseline = trial.loc[
            trial["Time bin center (s)"].lt(-15)
            & trial["Valid expected fraction"].ge(.9), "Total activity mean"
        ].dropna()
        if len(baseline) < 3:
            raise ValueError("Too few covered baseline bins")
        low, high = np.quantile(baseline, [.1, .9])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("Degenerate all-frame vigor baseline")
        values = trial["Total activity mean"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("All-frame vigor has missing bins; cannot render continuous steps")
        trial["Baseline P10"] = low
        trial["Baseline P90"] = high
        trial["Baseline bins"] = len(baseline)
        trial["Per-trial scaled vigor"] = np.clip((values - low) / (high - low), 0, 1)
        trial["Low valid-frame coverage"] = trial["Valid expected fraction"].lt(.9)
        trial["Signal semantics"] = "all_valid_frame_mean_per_trial_pre_minus15_p10_p90"
        rows.append(trial)
    return pd.concat(rows, ignore_index=True)


def render(data: pd.DataFrame, metric_id: str):
    theme = apply_theme()
    fig, axes = plt.subplots(3, 2, figsize=mm_to_in(183, 165),
        gridspec_kw={"height_ratios": [10, 50, 30]}, sharex=True, constrained_layout=True)
    mappings, panels = {}, []
    image = None
    for row, (phase, first, last) in enumerate(PHASES):
        for col, (fish, label, _) in enumerate(FISH):
            axis = axes[row, col]
            part = data.loc[data["Recording ID"].eq(fish) & data["Metric ID"].eq(metric_id)]
            times = np.arange(-20, 20, .5) + .25
            matrix = part.pivot(index="Trial number", columns="Time bin center (s)",
                values="Per-trial scaled vigor").reindex(index=range(first,last+1),columns=times)
            if matrix.isna().any().any():
                raise ValueError(f"Incomplete heatmap {fish}, {metric_id}, {phase}")
            image = axis.imshow(matrix.to_numpy(dtype=float), origin="upper", aspect="auto",
                interpolation="nearest", extent=(-20, 20, last-first+1, 0),
                vmin=0, vmax=1, cmap="managua_r", rasterized=True)
            panel = f"{label.lower()}_{phase.lower().replace('-', '_')}"
            panels.append(panel)
            gid = f"heatmap__{panel}__{metric_id}"
            image.set_gid(gid)
            mappings[gid] = {"recording_id": fish, "metric_id": metric_id,
                "value_field": "Per-trial scaled vigor", "signal": "all valid frame vigor",
                "cmap": "managua_r", "vmin": "0", "vmax": "1"}
            for second in (0, 10):
                axis.axvline(second, color=theme.cs_color, linewidth=.7)
            for trial in SELECTED_TRIALS:
                if first <= trial <= last:
                    axis.plot(20.35, trial-first+.5, marker="<", color="black",
                              markersize=3.5, clip_on=False)
            axis.set_xlim(-20,20)
            axis.set_ylim(last-first+1,0)
            axis.set_yticks([])
            axis.set_ylabel(phase if col == 0 else "")
            if row == 0:
                axis.set_title(f"{label}  {fish}")
            if row == 2:
                axis.set_xlabel("Time relative to CS onset (s)")
            else:
                axis.tick_params(labelbottom=False)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=.025, pad=.025,
                 label="All-frame per-trial scaled vigor (0–1)")
    panels.append("colorbar")
    fig.suptitle(f"Figure 1E all-frame review · {METRIC_DISPLAY_NAMES[metric_id]}")
    return fig, panels, mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
        default=Path("outputs/figure1-heatmaps/all-frame-review"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, paths = [], []
    for fish, _, condition in FISH:
        frame, used = load_verified_profile(args.project_dir, fish, condition)
        frames.append(frame)
        paths.extend(used)
    data = scale_total_vigor(pd.concat(frames, ignore_index=True))
    panel_path = args.output_dir / "figure-1-all-frame-scaled-vigor_panel-data.parquet"
    if panel_path.exists() and not args.overwrite:
        raise FileExistsError(panel_path)
    pq.write_table(pa.Table.from_pandas(data, preserve_index=False), panel_path,
                   compression="zstd")
    source = Path(__file__).resolve()
    inputs = tuple({"path": str(path.resolve()), "sha256": sha256_file(path)}
                   for path in (*paths, panel_path))
    snippet = f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src .venv/bin/python scripts/render_figure1_all_frame_heatmaps.py --project-dir '{args.project_dir}' --output-dir '{args.output_dir}' --overwrite"
    for metric_id in METRIC_DISPLAY_NAMES:
        fig, panels, mappings = render(data, metric_id)
        base = args.output_dir / f"figure-1-E_delay-control_{metric_id}_all-frame"
        try:
            result = export_matplotlib_figure(fig, base,
                FigureProvenance(figure_id="figure-1-E-all-frame-per-trial-vigor-review",
                    analysis_recipe="all-valid-frame-vigor-per-trial-p10-p90",
                    source_file=str(source), source_symbol="render", source_hash=sha256_file(source),
                    reproduction_snippet=snippet, input_artifacts=inputs,
                    artist_mappings=mappings),
                mode=FigureMode.STATIC, panel_ids=panels, overwrite=args.overwrite)
        finally:
            plt.close(fig)
        print(*result.outputs)
    print(panel_path)


if __name__ == "__main__":
    main()
