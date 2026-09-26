"""Build separate vector Figure 1D/E traces from the same measured Delay fish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from classical_conditioning.figures.example_traces import (  # noqa: E402
    METRIC_COLUMNS, prepare_example_trace_data,
)
from classical_conditioning.figures.theme import apply_theme  # noqa: E402
from render_legacy_ssd_example_traces import _read_windows, _verified_paths  # noqa: E402

PROJECT = Path("J:/Digested Data/allDelay-full-v1")
OUTPUT = Path("J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/traces")
FISH = "20221115_07"
METRIC = "legacy_distal_angular_speed"
TRIALS = (9, 17, 63, 66, 93)
STAGES = ("Pre-Train", "Early Train", "Late Train", "Early Test", "Late Test")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    corrected_path, metrics_path, protocol_path = _verified_paths(PROJECT, FISH)
    manifest_path = PROJECT / "Metadata" / f"{FISH}_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["recording_id"] != FISH or manifest["condition_id"] != "delay":
        raise ValueError("Trace fish is not the selected Delay example")
    protocol = pq.read_table(protocol_path).to_pandas()
    cycles = protocol.loc[protocol["Type"].eq("Cycle")].sort_values("Beg").reset_index(drop=True)
    if len(cycles) != 94:
        raise ValueError("Expected 94 CS cycles")
    intervals = [(int(cycles.iloc[trial - 1]["Beg"] - 20_000),
                  int(cycles.iloc[trial - 1]["Beg"] + 20_000))
                 for trial in TRIALS]
    corrected = _read_windows(corrected_path,
                              ["FrameID", "AbsoluteTime", *[f"angle{i}" for i in range(16)]],
                              intervals)
    metrics = _read_windows(metrics_path,
                            ["FrameID", "AbsoluteTime", METRIC_COLUMNS[METRIC]], intervals)
    frames, events = prepare_example_trace_data(
        corrected, metrics, protocol, trial_numbers=TRIALS, metric_id=METRIC,
        tail_point=15, window_s=(-20, 20), cs_duration_s=10,
    )
    inputs = [{"path": str(path), "sha256": digest(path)}
              for path in (protocol_path, manifest_path)]
    for path, marker_name, field in (
        (corrected_path, f"{FISH}_corrected-preprocess-v1_complete.json", "frames_sha256"),
        (metrics_path, f"{FISH}_candidate-corrected-v1_complete.json", "metrics_sha256"),
    ):
        marker = json.loads((PROJECT / "Metadata" / marker_name).read_text(encoding="utf-8"))
        inputs.append({"path": str(path), "sha256": marker[field]})
    return frames, events, inputs


def draw(frames: pd.DataFrame, events: pd.DataFrame, *, panel: str,
         output: Path) -> None:
    theme = apply_theme()
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["path.simplify"] = False
    fig, axes = plt.subplots(5, 1, figsize=(8.15, 4.8), sharex=True,
                             layout="none")
    fig.subplots_adjust(left=.21, right=.975, top=.86, bottom=.14, hspace=.13)
    is_angle = panel == "D"
    if is_angle:
        header = "Tail angle (°) · Delay fish 20221115_07"
        subtitle = "Distal cumulative angle; pre-CS median centred"
        limits = (-240.0, 240.0)
        ticks = (-200, 0, 200)
    else:
        header = "Frame vigor (rad/ms) · Delay fish 20221115_07"
        subtitle = "Raw legacy distal angular speed"
        finite = frames["Vigor"].to_numpy(dtype=float)
        ceiling = math.ceil(float(np.nanmax(finite)) * 2) / 2
        limits = (0.0, ceiling)
        ticks = (0, ceiling/2, ceiling)
    fig.text(.09, .96, header, fontsize=13, fontweight="bold", va="top")
    fig.text(.09, .91, subtitle, fontsize=9, color="#52606a", va="top")
    for axis, trial, stage in zip(axes, TRIALS, STAGES, strict=True):
        subset = frames.loc[frames["Trial number"].eq(trial)]
        x = subset["Time relative to CS onset (s)"].to_numpy(dtype=float)
        y = subset["Tail angle (rad)"].to_numpy(dtype=float) * (180 / np.pi) if is_angle \
            else subset["Vigor"].to_numpy(dtype=float)
        axis.plot(x, y, color="#111111", lw=.48, rasterized=False)
        for event in events.loc[events["Trial number"].eq(trial)].itertuples(index=False):
            kind = str(event.Event)
            axis.axvline(float(event[2]),
                         color=theme.us_color if kind == "actual US onset" else theme.cs_color,
                         lw=.85 if kind == "actual US onset" else .7,
                         linestyle="--" if kind == "CS offset" else "-")
        axis.set_xlim(-20, 20)
        axis.set_ylim(*limits)
        axis.set_yticks(ticks)
        axis.tick_params(axis="both", labelsize=8, length=2.5, width=.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(.65)
        axis.spines["bottom"].set_linewidth(.65)
        axis.text(-.19, .5, stage, transform=axis.transAxes,
                  ha="right", va="center", fontsize=9, fontweight="bold")
        if axis is not axes[-1]:
            axis.tick_params(labelbottom=False)
    axes[-1].set_xticks((-20, -10, 0, 10, 20))
    axes[-1].set_xlabel("Time relative to CS onset (s)", fontsize=10, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Title": header,
                "Description": "Measured fish traces, global CS trials 9, 17, 63, 66, 93."})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    frames, events, inputs = load()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "Fig1_PanelsD-E_Delay_legacy-vigor_frames_v1.parquet"
    events_path = args.output_dir / "Fig1_PanelsD-E_Delay_events_v1.parquet"
    pq.write_table(pa.Table.from_pandas(frames, preserve_index=False), data_path)
    pq.write_table(pa.Table.from_pandas(events, preserve_index=False), events_path)
    for panel, name in (("D", "TailAngle"), ("E", "RawVigor")):
        svg = args.output_dir / f"Fig1_Panel{panel}_{name}_legacy-vigor_v1.svg"
        draw(frames, events, panel=panel, output=svg)
        sidecar = {
            "panel": panel, "recording_id": FISH, "condition_id": "delay",
            "global_cs_trials": TRIALS, "trial_stages": STAGES,
            "window_s": [-20, 20], "tail_point": 15,
            "metric_id": METRIC if panel == "E" else "distal_cumulative_tail_angle",
            "signal": "raw frame vigor" if panel == "E" else "pre-CS-median-centered tail angle",
            "input_artifacts": inputs,
            "frames": str(data_path), "frames_sha256": digest(data_path),
            "events": str(events_path), "events_sha256": digest(events_path),
            "svg": str(svg), "svg_sha256": digest(svg),
        }
        svg.with_suffix(".svg.json").write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        print(svg)


if __name__ == "__main__":
    main()
