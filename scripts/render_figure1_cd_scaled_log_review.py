"""Compare Figure 1 C/D traces with trial-scaled log vigor baselines.

This is a review renderer. It leaves the paper registry and existing heatmap
recipes unchanged until one baseline and signal definition are approved.
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
from classical_conditioning.figures.example_traces import (
    METRIC_COLUMNS, METRIC_DISPLAY_NAMES, prepare_example_trace_data,
)
from classical_conditioning.figures.export import (
    FigureMode, FigureProvenance, export_matplotlib_figure,
)
from classical_conditioning.figures.theme import apply_theme, mm_to_in

from render_legacy_ssd_example_traces import _read_windows, _verified_paths


WINDOW_S = (-20.0, 20.0)
TRIALS = (9, 17, 63, 66, 93)
STAGES = ("Pre-Train", "Early Train", "Late Train", "Early Test", "Late Test")
BIN_WIDTH_S = 0.5
BIN_CENTERS = np.arange(WINDOW_S[0], WINDOW_S[1], BIN_WIDTH_S) + BIN_WIDTH_S / 2


def scale_trial_log_vigor(
    values: np.ndarray,
    seconds: np.ndarray,
    valid: np.ndarray,
    moving: np.ndarray,
    bout_ids: np.ndarray,
    *,
    baseline_start_s: float,
    baseline_end_s: float = 0.0,
    moving_bouts_only: bool = True,
) -> tuple[np.ndarray, float, float, int]:
    """P10/P90-scale log vigor using one explicit interval in one CS trial.

    For the movement-conditional recipe, each positive moving frame receives
    the mean raw vigor of its detected bout before the natural log is taken.
    Quantiles weight frames, as in the retained frame-first heatmap review.
    """
    arrays = tuple(np.asarray(item) for item in
                   (values, seconds, valid, moving, bout_ids))
    if len({len(item) for item in arrays}) != 1:
        raise ValueError("Trial metric, time, and movement arrays must align")
    if not WINDOW_S[0] <= baseline_start_s < baseline_end_s <= 0:
        raise ValueError("Baseline must lie within the displayed pre-CS window")
    values, seconds, valid, moving, bout_ids = arrays
    usable = valid.astype(bool) & np.isfinite(values) & (values > 0)
    positive = np.full(len(values), np.nan, dtype=float)
    if moving_bouts_only:
        usable &= moving.astype(bool) & (bout_ids > 0)
        if usable.any():
            frame = pd.DataFrame({"bout": bout_ids[usable], "value": values[usable]})
            positive[usable] = frame.groupby("bout", sort=False)["value"].transform("mean")
    else:
        positive[usable] = values[usable]
    logged = np.full(len(values), np.nan, dtype=float)
    finite_positive = np.isfinite(positive) & (positive > 0)
    logged[finite_positive] = np.log(positive[finite_positive])
    reference = logged[(seconds >= baseline_start_s) & (seconds < baseline_end_s)
                       & np.isfinite(logged)]
    scaled = np.full(len(values), np.nan, dtype=float)
    if len(reference) < 3:
        return scaled, np.nan, np.nan, len(reference)
    low, high = np.quantile(reference, [0.1, 0.9])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return scaled, float(low), float(high), len(reference)
    finite = np.isfinite(logged)
    scaled[finite] = np.clip((logged[finite] - low) / (high - low), 0, 1)
    return scaled, float(low), float(high), len(reference)


def half_second_bin_mean(seconds: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Average finite frame-scaled values; leave empty bins missing."""
    indices = np.floor((seconds - WINDOW_S[0]) / BIN_WIDTH_S).astype(np.int32)
    finite = (indices >= 0) & (indices < len(BIN_CENTERS)) & np.isfinite(values)
    count = np.bincount(indices[finite], minlength=len(BIN_CENTERS))
    total = np.bincount(indices[finite], weights=values[finite],
                        minlength=len(BIN_CENTERS))
    return np.divide(total, count, out=np.full(len(BIN_CENTERS), np.nan),
                     where=count > 0)


def _read_trial_frames(project: Path, fish: str, metric_id: str,
                       trials: tuple[int, ...]):
    corrected_path, metrics_path, protocol_path = _verified_paths(project, fish)
    movement_path = (project / "Processed data" / fish
                     / "movement_state_candidates-corrected-v2.parquet")
    marker_path = (project / "Metadata"
                   / f"{fish}_movement-candidate-corrected-v2_complete.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("recording_id") != fish:
        raise ValueError(f"Incomplete movement marker: {marker_path}")
    if sha256_file(movement_path) != marker["movement_sha256"]:
        raise ValueError(f"Movement artifact hash mismatch: {movement_path}")
    protocol = pq.read_table(protocol_path).to_pandas()
    cycles = protocol.loc[protocol["Type"].eq("Cycle")].sort_values("Beg").reset_index(drop=True)
    intervals = [(int(cycles.iloc[t - 1]["Beg"] + WINDOW_S[0] * 1000),
                  int(cycles.iloc[t - 1]["Beg"] + WINDOW_S[1] * 1000))
                 for t in trials]
    corrected = _read_windows(corrected_path,
                              ["FrameID", "AbsoluteTime", *(f"angle{i}" for i in range(16))],
                              intervals)
    metrics = _read_windows(metrics_path,
                            ["FrameID", "AbsoluteTime", METRIC_COLUMNS[metric_id]], intervals)
    movement = _read_windows(movement_path,
                             ["FrameID", "AbsoluteTime", "valid", "moving", "bout_id"],
                             intervals)
    if not np.array_equal(
        metrics[["FrameID", "AbsoluteTime"]].to_numpy(),
        movement[["FrameID", "AbsoluteTime"]].to_numpy(),
    ):
        raise ValueError("Metric and movement frames do not align")
    frames, events = prepare_example_trace_data(
        corrected, metrics, protocol, trial_numbers=trials, metric_id=metric_id,
        tail_point=15, window_s=WINDOW_S, cs_duration_s=10.0,
    )
    if movement["FrameID"].duplicated().any():
        raise ValueError("Duplicate movement FrameID")
    selected = frames.merge(
        movement[["FrameID", "valid", "moving", "bout_id"]],
        on="FrameID", how="left", validate="many_to_one",
    )
    if selected[["valid", "moving", "bout_id"]].isna().any().any():
        raise ValueError("Selected frames lack movement state")
    paths = (corrected_path, metrics_path, movement_path, marker_path, protocol_path)
    return selected, events, paths


def _render(data: pd.DataFrame, events: pd.DataFrame, metric_id: str,
            baseline_start_s: float, *, moving_bouts_only: bool,
            trials: tuple[int, ...]):
    theme = apply_theme()
    fig, axes = plt.subplots(len(trials), 2,
                             figsize=mm_to_in(183, max(55, 29 * len(trials))),
                             sharex=True, sharey="col", constrained_layout=True,
                             squeeze=False)
    mappings = {}
    panels = []
    stage_by_trial = dict(zip(TRIALS, STAGES))
    for row, trial in enumerate(trials):
        stage = stage_by_trial.get(trial, f"Trial {trial}")
        subset = data.loc[data["Trial number"].eq(trial)].sort_values(
            "Time relative to CS onset (s)"
        )
        x = subset["Time relative to CS onset (s)"].to_numpy(dtype=float)
        for col, (field, label) in enumerate((
            ("Tail angle (rad)", "C · tail angle (rad)"),
            ("Scaled log vigor", "D · 0.5 s mean scaled log vigor (0–1)"),
        )):
            axis = axes[row, col]
            if col == 0:
                line, = axis.plot(x, subset[field], color="black", linewidth=.48,
                                  rasterized=True)
            else:
                binned = half_second_bin_mean(
                    x, subset[field].to_numpy(dtype=float)
                )
                line, = axis.plot(BIN_CENTERS, binned, color="black",
                                  linewidth=.75, marker=".", markersize=2)
            gid = f"trace__{'C' if col == 0 else 'D'}__trial_{trial}"
            line.set_gid(gid)
            mappings[gid] = {"trial_number": str(trial), "value_field": field,
                             "baseline_start_s": str(baseline_start_s) if col else "",
                             "baseline_end_s": "0" if col else "",
                             "aggregation": "mean of finite scaled frames per 0.5 s bin" if col else "none"}
            for event in events.loc[events["Trial number"].eq(trial)].itertuples(index=False):
                kind = str(event.Event)
                axis.axvline(float(event[2]),
                             color=theme.us_color if kind == "actual US onset" else theme.cs_color,
                             linestyle="--" if kind == "CS offset" else "-", linewidth=.7)
            axis.set_xlim(*WINDOW_S)
            if col == 1:
                axis.set_ylim(0, 1)
                axis.axvspan(baseline_start_s, 0, color=".7", alpha=.1, zorder=-2)
            if row == 0:
                axis.set_title(label)
            if col == 0:
                axis.set_ylabel(f"{stage}\ntrial {trial}")
            if row == len(trials) - 1:
                axis.set_xlabel("Time relative to CS onset (s)")
            else:
                axis.tick_params(labelbottom=False)
            panels.append(f"{'C' if col == 0 else 'D'}_trial_{trial}")
    fig.suptitle(
        f"Figure 1 C/D review · {METRIC_DISPLAY_NAMES[metric_id]} · "
        f"baseline [{baseline_start_s:g}, 0) s · "
        f"{'moving bouts' if moving_bouts_only else 'all valid frames'}"
    )
    return fig, panels, mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recording-id", default="20221115_07")
    parser.add_argument("--metric", choices=tuple(METRIC_COLUMNS),
                        default="tail_length_weighted_angular_l1")
    parser.add_argument("--trial", type=int, action="append",
                        help="Global CS trial; defaults to 9, 17, 63, 66, 93")
    parser.add_argument("--signal", choices=("moving-bouts", "all-valid"),
                        default="moving-bouts")
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    trials = tuple(args.trial or TRIALS)
    if not trials or min(trials) < 1 or len(trials) != len(set(trials)):
        raise ValueError("Choose distinct positive global CS trials")
    project = args.project_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames, events, sources = _read_trial_frames(
        project, args.recording_id, args.metric, trials
    )
    moving_bouts_only = args.signal == "moving-bouts"
    source = Path(__file__).resolve()
    for baseline_start_s in (-20.0, -15.0):
        pieces = []
        for trial in trials:
            piece = frames.loc[frames["Trial number"].eq(trial)].copy()
            scaled, low, high, count = scale_trial_log_vigor(
                piece["Vigor"].to_numpy(dtype=float),
                piece["Time relative to CS onset (s)"].to_numpy(dtype=float),
                piece["valid"].to_numpy(dtype=bool),
                piece["moving"].to_numpy(dtype=bool),
                piece["bout_id"].to_numpy(dtype=np.int32),
                baseline_start_s=baseline_start_s,
                moving_bouts_only=moving_bouts_only,
            )
            piece["Scaled log vigor"] = scaled
            piece["Baseline log P10"] = low
            piece["Baseline log P90"] = high
            piece["Baseline moving frames"] = count
            piece["Baseline start (s)"] = baseline_start_s
            piece["Baseline end (s)"] = 0.0
            piece["Metric ID"] = args.metric
            piece["Signal"] = args.signal
            pieces.append(piece)
        data = pd.concat(pieces, ignore_index=True)
        trial_tag = "" if trials == TRIALS else f"_{'-'.join(map(str, trials))}"
        stem = (f"figure-1-CD-scaled-log_{args.recording_id}_{args.metric}_"
                f"{args.signal}{trial_tag}_"
                f"pre{abs(int(baseline_start_s))}")
        panel_path = output / f"{stem}_panel-data.parquet"
        if panel_path.exists() and not args.overwrite:
            raise FileExistsError(panel_path)
        pq.write_table(pa.Table.from_pandas(data, preserve_index=False), panel_path,
                       compression="zstd")
        fig, panels, mappings = _render(data, events, args.metric, baseline_start_s,
                                         moving_bouts_only=moving_bouts_only,
                                         trials=trials)
        try:
            result = export_matplotlib_figure(
                fig, output / stem,
                FigureProvenance(
                    figure_id="figure-1-CD-scaled-log-baseline-review",
                    analysis_recipe="per-trial-log-vigor-preCS-P10-P90-frame-scale-then-0p5s-bin-mean",
                    source_file=str(source), source_symbol="main",
                    source_hash=sha256_file(source),
                    reproduction_snippet=(
                        f"python scripts/render_figure1_cd_scaled_log_review.py "
                        f"--project-dir '{project}' --output-dir '{output}' "
                        f"--recording-id {args.recording_id} --metric {args.metric} "
                        f"--signal {args.signal} --mode {args.mode} "
                        + " ".join(f"--trial {trial}" for trial in trials)
                    ),
                    input_artifacts=tuple(
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in (*sources, panel_path)
                    ),
                    artist_mappings=mappings,
                    analysis_identity={
                        "recording_id": args.recording_id, "metric_id": args.metric,
                        "signal": args.signal, "baseline_start_s": baseline_start_s,
                        "baseline_end_s": 0.0, "scaling": "log P10/P90, clipped 0–1",
                        "scientific_status": "baseline_comparison_only",
                    },
                ),
                mode=FigureMode(args.mode), panel_ids=panels, overwrite=args.overwrite,
            )
        finally:
            plt.close(fig)
        print(panel_path)
        for path in (*result.outputs, result.sidecar):
            print(path)


if __name__ == "__main__":
    main()
