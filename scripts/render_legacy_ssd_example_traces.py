"""Render Figure 1 example traces from the versioned allDelay-full-v1 SSD export.

This compatibility entry point uses the shared Figure 1 trace preparation and
renderer. It reads only row groups overlapping the requested CS windows, and
checks the SSD completion markers before using their artifacts.
"""

from __future__ import annotations

import argparse
import json
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import classical_conditioning.figures.example_traces as shared_renderer
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import (
    METRIC_COLUMNS,
    prepare_example_trace_data,
    render_example_trace_figure,
)
from classical_conditioning.figures.export import (
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)


@cache
def _file_hash(path: Path) -> str:
    return sha256_file(path)


@cache
def _verified_paths(project_dir: Path, recording_id: str) -> tuple[Path, Path, Path]:
    root = project_dir / "Processed data" / recording_id
    corrected = root / "frame_preprocessed_corrected-v1.parquet"
    metrics = root / "frame_activity_candidates-corrected-v1.parquet"
    protocol = root / "stimulus_events.parquet"
    checks = (
        (corrected, project_dir / "Metadata" / f"{recording_id}_corrected-preprocess-v1_complete.json", "frames_sha256"),
        (metrics, project_dir / "Metadata" / f"{recording_id}_candidate-corrected-v1_complete.json", "metrics_sha256"),
    )
    for artifact, marker_path, hash_key in checks:
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "complete" or marker.get("recording_id") != recording_id:
            raise ValueError(f"Incomplete or mismatched marker: {marker_path}")
        if _file_hash(artifact) != marker[hash_key]:
            raise ValueError(f"Artifact hash differs from completion marker: {artifact}")
    manifest_path = project_dir / "Metadata" / f"{recording_id}_source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["artifacts"]["protocol"]["sha256"] != _file_hash(protocol):
        raise ValueError(f"Protocol hash differs from source manifest: {protocol}")
    return corrected, metrics, protocol


def _read_windows(path: Path, columns: list[str], intervals: list[tuple[int, int]]) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    time_index = parquet.schema.names.index("AbsoluteTime")
    selected = []
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(time_index).statistics
        if stats is None or stats.min is None or stats.max is None:
            raise ValueError(f"Missing AbsoluteTime row-group statistics: {path}")
        if any(stats.max >= start and stats.min <= end for start, end in intervals):
            selected.append(row_group)
    if not selected:
        raise ValueError(f"No frames overlap the selected trials in {path}")
    data = parquet.read_row_groups(selected, columns=columns).to_pandas()
    time = data["AbsoluteTime"].to_numpy(dtype=np.int64)
    keep = np.zeros(len(data), dtype=bool)
    for start, end in intervals:
        keep |= (time >= start) & (time <= end)
    return data.loc[keep].reset_index(drop=True)


def render_from_ssd(
    project_dir: Path,
    output_dir: Path,
    recording_id: str,
    *,
    trials: tuple[int, ...],
    metric_id: str,
    mode: FigureMode = FigureMode.STATIC,
    window_s: tuple[float, float] = (-20.0, 20.0),
    tail_point: int = 15,
    overwrite: bool = False,
):
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Only static and publication exports are supported")
    if metric_id not in METRIC_COLUMNS:
        raise ValueError(f"Unknown metric: {metric_id}")
    corrected_path, metrics_path, protocol_path = _verified_paths(project_dir, recording_id)
    protocol = pq.read_table(protocol_path).to_pandas()
    cycles = protocol.loc[protocol["Type"].eq("Cycle")].sort_values("Beg").reset_index(drop=True)
    if any(trial < 1 or trial > len(cycles) for trial in trials):
        raise ValueError(f"Trials must be between 1 and {len(cycles)}")
    intervals = [
        (int(cycles.iloc[trial - 1]["Beg"] + window_s[0] * 1000),
         int(cycles.iloc[trial - 1]["Beg"] + window_s[1] * 1000))
        for trial in trials
    ]
    angle_columns = [f"angle{index}" for index in range(tail_point + 1)]
    corrected = _read_windows(corrected_path, ["FrameID", "AbsoluteTime", *angle_columns], intervals)
    metrics = _read_windows(metrics_path, ["FrameID", "AbsoluteTime", METRIC_COLUMNS[metric_id]], intervals)
    frames, events = prepare_example_trace_data(
        corrected, metrics, protocol, trial_numbers=trials, metric_id=metric_id,
        tail_point=tail_point, window_s=window_s, cs_duration_s=10.0,
    )
    figure, panel_ids, mappings = render_example_trace_figure(
        frames, events, trial_numbers=trials, metric_id=metric_id, window_s=window_s,
        trial_labels=(
            ("Pre-Train", "Early Train", "Late Train", "Early Test", "Late Test")
            if trials == (9, 17, 63, 66, 93) else None
        ),
    )
    source = Path(__file__).resolve()
    base = output_dir / recording_id / f"figure-1-CD_{metric_id}_{'-'.join(map(str, trials))}"
    snippet = (
        f"python scripts/render_legacy_ssd_example_traces.py --project-dir '{project_dir}' "
        f"--output-dir '{output_dir}' --recording-id {recording_id} "
        f"--metric {metric_id} " + " ".join(f"--trial {n}" for n in trials)
        + f" --mode {mode.value} --window-start {window_s[0]} --window-end {window_s[1]}"
        + f" --tail-point {tail_point}"
    )
    inputs = tuple({"path": str(path), "sha256": _file_hash(path)} for path in
                   (corrected_path, metrics_path, protocol_path,
                    Path(shared_renderer.__file__).resolve()))
    try:
        return export_matplotlib_figure(
            figure, base,
            FigureProvenance(
                figure_id="figure-1-CD-example-traces",
                analysis_recipe="selected-fish-corrected-example-traces-ssd-v1",
                source_file=str(source), source_symbol="render_from_ssd",
                source_hash=sha256_file(source), reproduction_snippet=snippet,
                input_artifacts=inputs, artist_mappings=mappings,
            ),
            mode=mode, panel_ids=panel_ids, overwrite=overwrite,
        )
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--trial", type=int, action="append", required=True,
                        help="Global CS trial number; historical labels add four.")
    metric_group = parser.add_mutually_exclusive_group(required=True)
    metric_group.add_argument("--metric", choices=tuple(METRIC_COLUMNS))
    metric_group.add_argument("--all-metrics", action="store_true")
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--tail-point", type=int, default=15)
    parser.add_argument("--window-start", type=float, default=-20.0)
    parser.add_argument("--window-end", type=float, default=20.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for metric_id in (tuple(METRIC_COLUMNS) if args.all_metrics else (args.metric,)):
        result = render_from_ssd(
            args.project_dir, args.output_dir, args.recording_id,
            trials=tuple(args.trial), metric_id=metric_id,
            mode=FigureMode(args.mode), tail_point=args.tail_point,
            window_s=(args.window_start, args.window_end), overwrite=args.overwrite,
        )
        for path in (*result.outputs, result.sidecar):
            print(path)


if __name__ == "__main__":
    main()
