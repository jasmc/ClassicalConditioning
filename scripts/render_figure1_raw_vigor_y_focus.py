"""Render Figure 1D with raw frames and matching single-fish heatmap bins.

Black is the measured frame-by-frame candidate metric. Orange is the exact
movement-conditional heatmap signal in 0.5 s bins, on a separate axis.
Finite runs have vertical boundaries at adjacent missing bins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import (
    METRIC_COLUMNS, METRIC_DISPLAY_NAMES, prepare_example_trace_data,
)
from classical_conditioning.figures.export import (
    FigureMode, FigureProvenance, export_matplotlib_figure,
)
from classical_conditioning.figures.temporal_profiles import METRIC_UNITS
from classical_conditioning.figures.theme import apply_theme, mm_to_in
from render_legacy_ssd_example_traces import _file_hash, _read_windows, _verified_paths


FISH = (("20221115_07", "Delay"), ("20221115_09", "Control"))
TRIALS = (9, 17, 63, 66, 93)
LABELS = ("Pre-Train", "Early Train", "Late Train", "Early Test", "Late Test")
WINDOW = (-20.0, 20.0)


def _read_fish(project: Path, fish: str, metric_id: str):
    corrected_path, metrics_path, protocol_path = _verified_paths(project, fish)
    protocol = pq.read_table(protocol_path).to_pandas()
    cycles = protocol.loc[protocol["Type"].eq("Cycle")].sort_values("Beg").reset_index(drop=True)
    intervals = [
        (int(cycles.iloc[trial - 1]["Beg"] + WINDOW[0] * 1000),
         int(cycles.iloc[trial - 1]["Beg"] + WINDOW[1] * 1000))
        for trial in TRIALS
    ]
    corrected = _read_windows(
        corrected_path, ["FrameID", "AbsoluteTime", *(f"angle{i}" for i in range(16))],
        intervals,
    )
    metrics = _read_windows(
        metrics_path, ["FrameID", "AbsoluteTime", METRIC_COLUMNS[metric_id]], intervals,
    )
    frames, events = prepare_example_trace_data(
        corrected, metrics, protocol, trial_numbers=TRIALS, metric_id=metric_id,
        tail_point=15, window_s=WINDOW, cs_duration_s=10.0,
    )
    frames["Recording ID"] = fish
    events["Recording ID"] = fish
    return frames, events, (corrected_path, metrics_path, protocol_path)


def _render(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    heatmap_data: pd.DataFrame,
    metric_id: str,
    *,
    focus_y: bool,
):
    theme = apply_theme()
    figure, axes = plt.subplots(
        len(TRIALS), len(FISH), figsize=mm_to_in(183, 150),
        sharex=True, sharey=True, constrained_layout=True,
    )
    all_vigor = frames["Vigor"].to_numpy(dtype=float)
    finite = all_vigor[np.isfinite(all_vigor) & (all_vigor >= 0)]
    if not finite.size:
        raise ValueError(f"No finite raw vigor for {metric_id}")
    y_max = float(np.quantile(finite, 0.995)) if focus_y else float(np.max(finite))
    if y_max <= 0:
        raise ValueError(f"Raw vigor range is degenerate for {metric_id}")
    mappings = {}
    panels = []
    for row, (trial, label) in enumerate(zip(TRIALS, LABELS)):
        for col, (fish, condition) in enumerate(FISH):
            axis = axes[row, col]
            part = frames.loc[frames["Recording ID"].eq(fish) & frames["Trial number"].eq(trial)]
            if part.empty:
                raise ValueError(f"No frames for {fish}, trial {trial}, {metric_id}")
            x = part["Time relative to CS onset (s)"].to_numpy(dtype=float)
            y = part["Vigor"].to_numpy(dtype=float)
            line, = axis.plot(x, y, color="0.12", linewidth=0.55, rasterized=True)
            gid = f"raw_vigor__{fish}__trial_{trial}__{metric_id}"
            line.set_gid(gid)
            mappings[gid] = {
                "recording_id": fish, "trial_number": str(trial),
                "metric_id": metric_id, "value_field": "Vigor",
                "units": METRIC_UNITS[metric_id],
                "display_y_max": str(y_max), "display_cap": "99.5th percentile" if focus_y else "none",
            }
            if focus_y:
                bins = heatmap_data.loc[
                    heatmap_data["Recording ID"].astype(str).eq(fish)
                    & heatmap_data["Trial number"].eq(trial)
                    & heatmap_data["Metric ID"].eq(metric_id)
                    & heatmap_data["Time bin center (s)"].ge(WINDOW[0])
                    & heatmap_data["Time bin center (s)"].lt(WINDOW[1])
                ].sort_values("Time bin center (s)")
                if len(bins) != 80:
                    raise ValueError(f"Expected 80 heatmap bins: {fish}, {metric_id}, trial {trial}")
                signed = "Signed log vigor" in bins.columns
                value_field = "Signed log vigor" if signed else "Per-trial scaled vigor"
                values = bins[value_field].to_numpy(dtype=float)
                transform = ("signed log relative to pre-CS median" if signed else
                             str(bins["Vigor transform"].iloc[0]) if "Vigor transform" in bins else "linear")
                centers = bins["Time bin center (s)"].to_numpy(dtype=float)
                if not np.allclose(centers, np.arange(-20, 20, .5) + .25):
                    raise ValueError("Heatmap bins do not span the full 40 s window")
                right = axis.twinx()
                edges = np.arange(-20, 20.5, .5)
                finite_mask = np.isfinite(values)
                if not finite_mask.any():
                    raise ValueError(f"No finite heatmap bins: {fish}, {metric_id}, trial {trial}")
                transitions = np.diff(np.r_[False, finite_mask, False].astype(int))
                starts = np.flatnonzero(transitions == 1)
                stops = np.flatnonzero(transitions == -1)
                for run, (start, stop) in enumerate(zip(starts, stops)):
                    step = right.stairs(
                        values[start:stop], edges[start:stop + 1], baseline=0,
                        color="#DD5300", linewidth=1.0, alpha=0.9,
                        rasterized=True,
                    )
                    step_id = f"heatmap_steps__{fish}__trial_{trial}__{metric_id}__run_{run}"
                    step.set_gid(step_id)
                    mappings[step_id] = {
                        "recording_id": fish, "trial_number": str(trial),
                        "metric_id": metric_id,
                        "value_field": value_field,
                        "units": "signed log" if signed else "0-1",
                        "source_panel": "figure-1-E movement-conditional",
                        "time_bin_width_s": "0.5", "first_bin": str(start),
                        "last_bin_exclusive": str(stop),
                        "missing_bins": "not drawn; run boundaries descend to zero",
                        "vigor_transform_before_scaling": transform,
                    }
                if signed:
                    right.set_ylim(-0.27, 0.27)
                else:
                    right.set_ylim(0, 1.05)
                right.set_xlim(*WINDOW)
                right.spines[["top", "left"]].set_visible(False)
                right.tick_params(axis="y", labelsize=7, colors="#B64700",
                                  right=col == 1, labelright=col == 1)
                if col == 1 and row == len(TRIALS) // 2:
                    right.set_ylabel("Signed log vigor" if signed else "Heatmap scaled vigor (0–1)",
                                     color="#B64700", fontsize=8)
            axis.axvspan(0, 10, color=theme.cs_color, alpha=0.055, zorder=-1)
            for time in (0, 10):
                axis.axvline(time, color=theme.cs_color, linewidth=0.75)
            us = events.loc[
                events["Recording ID"].eq(fish)
                & events["Trial number"].eq(trial)
                & events["Event"].eq("actual US onset"), "Time (s)",
            ]
            for us_time in us:
                axis.axvline(float(us_time), color=theme.us_color, linewidth=0.85)
            axis.set_xlim(*WINDOW)
            axis.set_ylim(0, y_max * 1.03)
            axis.set_xticks((-20, -10, 0, 10, 20))
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelbottom=row == len(TRIALS) - 1,
                             labelleft=col == 0, labelsize=8)
            if col == 0:
                axis.set_ylabel(f"{label}\ntrial {trial}", fontsize=8)
            if row == 0:
                axis.set_title(f"{condition}  {fish}", fontsize=10)
            if row == len(TRIALS) - 1:
                axis.set_xlabel("Time relative to CS onset (s)", fontsize=9)
            panels.append(f"D_{condition.lower()}_trial_{trial}")
    transform_label = (
        "signed log " if "Signed log vigor" in heatmap_data.columns else
        "frame-scaled log "
        if "Signal semantics" in heatmap_data.columns and
        set(heatmap_data["Signal semantics"].astype(str)) ==
        {"bout_mean_log_frame_scaled_per_trial_pre_minus20_to_0_then_binned"}
        else (str(heatmap_data["Vigor transform"].iloc[0]) + " "
              if "Vigor transform" in heatmap_data else "")
    )
    cap_text = (f" · Y focus; orange = {transform_label}conditional heatmap bins (right axis)"
                if focus_y else " · full Y range")
    figure.suptitle(
        f"Figure 1D raw vigor · {METRIC_DISPLAY_NAMES[metric_id]}{cap_text}", fontsize=10,
    )
    figure.supylabel(f"Frame-level vigor ({METRIC_UNITS[metric_id]})", fontsize=9)
    if focus_y:
        panels.extend(f"heatmap_scale__{condition.lower()}_trial_{trial}"
                      for trial, _ in zip(TRIALS, LABELS)
                      for _, condition in FISH)
    return figure, panels, mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heatmap-panel-data", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.heatmap_panel_data.resolve()
    heatmap_data = pq.read_table(heatmap_path).to_pandas()
    required = {"Recording ID", "Trial number", "Metric ID", "Time bin center (s)"}
    missing = required.difference(heatmap_data.columns)
    if missing:
        raise ValueError(f"Heatmap panel data lack columns: {sorted(missing)}")
    if not ({"Signed log vigor", "Per-trial scaled vigor"} & set(heatmap_data.columns)):
        raise ValueError("Heatmap panel data contain no vigor value column")
    expected_semantics = {
        "bout_mean_log_frame_scaled_per_trial_pre_minus20_to_0_then_binned",
        "bout_mean_log_frame_scaled_per_trial_then_binned",
    }
    frame_scaled = ("Signal semantics" in heatmap_data.columns and
                    set(heatmap_data["Signal semantics"].astype(str)).issubset(expected_semantics))
    if not frame_scaled and "Conditional intensity mean" not in heatmap_data.columns and "Signed log vigor" not in heatmap_data.columns:
        raise ValueError("Figure 1D requires movement-conditional heatmap panel data")
    if heatmap_data.duplicated(["Recording ID", "Trial number", "Metric ID", "Time bin center (s)"]).any():
        raise ValueError("Heatmap panel data contain duplicate fish/trial/metric/bin rows")
    source = Path(__file__).resolve()
    helper = Path(_verified_paths.__wrapped__.__code__.co_filename).resolve()
    snippet = (
        f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src ./.venv/bin/python "
        f"scripts/render_figure1_raw_vigor_y_focus.py --project-dir '{project}' "
        f"--output-dir '{output}' --heatmap-panel-data '{heatmap_path}' "
        f"--mode {args.mode} --overwrite"
    )
    for metric_id in METRIC_COLUMNS:
        frames, events, paths = [], [], set()
        for fish, _ in FISH:
            fish_frames, fish_events, fish_paths = _read_fish(project, fish, metric_id)
            frames.append(fish_frames)
            events.append(fish_events)
            paths.update(fish_paths)
        joined_frames = pd.concat(frames, ignore_index=True)
        joined_events = pd.concat(events, ignore_index=True)
        inputs = tuple(
            {"path": str(path), "sha256": _file_hash(path)}
            for path in sorted((*paths, helper, heatmap_path))
        )
        for focus_y in (False, True):
            figure, panels, mappings = _render(joined_frames, joined_events, heatmap_data,
                                                metric_id,
                                                focus_y=focus_y)
            variant = "focus-y" if focus_y else "full-y"
            base = output / f"figure-1-D-raw-vigor-{variant}_{metric_id}_9-17-63-66-93"
            try:
                result = export_matplotlib_figure(
                    figure, base,
                    FigureProvenance(
                        figure_id=f"figure-1-D-raw-vigor-{variant}",
                        analysis_recipe="verified-corrected-frame-vigor-example",
                        source_file=str(source), source_symbol="_render",
                        source_hash=sha256_file(source), reproduction_snippet=snippet,
                        input_artifacts=inputs, artist_mappings=mappings,
                    ),
                    mode=FigureMode(args.mode), panel_ids=panels,
                    overwrite=args.overwrite,
                )
            finally:
                plt.close(figure)
            for path in (*result.outputs, result.sidecar):
                print(path)


if __name__ == "__main__":
    main()
