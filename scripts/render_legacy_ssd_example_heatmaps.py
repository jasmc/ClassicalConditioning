"""Render paired single-fish Figure 1 heatmaps from versioned SSD artifacts.

The signal is the current signed bout-log-vigor candidate, computed from the
SSD's corrected activity and shared movement detector. The colour scale is the
March 2026 historical ``managua_r`` scale at -0.25 to +0.25.
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

import classical_conditioning.analysis.temporal_profiles as profile_module
import classical_conditioning.figures.theme as theme_module
from classical_conditioning.analysis.temporal_profiles import _signed_bout_log_vigor
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import METRIC_COLUMNS, METRIC_DISPLAY_NAMES
from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
from classical_conditioning.figures.theme import apply_theme, heatmap_cmap, mm_to_in

from render_legacy_ssd_example_traces import _file_hash, _verified_paths


PHASES = (("Pre-Train", 5, 14), ("Train", 15, 64), ("Test", 65, 94))
SELECTED_TRIALS = (9, 17, 63, 66, 93)
WINDOW_S = (-20.0, 20.0)
BIN_WIDTH_S = 0.5


def _read_selected_windows(
    path: Path, columns: list[str], intervals: list[tuple[int, int]],
) -> pd.DataFrame:
    """Prune row groups, then keep only requested CS windows in each group."""
    parquet = pq.ParquetFile(path)
    time_index = parquet.schema.names.index("AbsoluteTime")
    pieces = []
    for row_group in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(row_group).column(time_index).statistics
        if stats is None or stats.min is None or stats.max is None:
            raise ValueError(f"Missing AbsoluteTime row-group statistics in {path}")
        overlapping = [
            (start, end) for start, end in intervals
            if stats.max >= start and stats.min < end
        ]
        if not overlapping:
            continue
        piece = parquet.read_row_group(row_group, columns=columns).to_pandas()
        times = piece["AbsoluteTime"].to_numpy(dtype=np.int64)
        keep = np.zeros(len(piece), dtype=bool)
        for start, end in overlapping:
            keep |= (times >= start) & (times < end)
        if keep.any():
            pieces.append(piece.loc[keep])
    if not pieces:
        raise ValueError(f"No CS-window frames in {path}")
    return pd.concat(pieces, ignore_index=True)


def _load_fish(project_dir: Path, recording_id: str):
    _, metrics_path, protocol_path = _verified_paths(project_dir, recording_id)
    movement_path = (
        project_dir / "Processed data" / recording_id
        / "movement_state_candidates-corrected-v2.parquet"
    )
    marker_path = (
        project_dir / "Metadata"
        / f"{recording_id}_movement-candidate-corrected-v2_complete.json"
    )
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "complete" or marker.get("recording_id") != recording_id:
        raise ValueError(f"Incomplete movement artifact: {marker_path}")
    if _file_hash(movement_path) != marker["movement_sha256"]:
        raise ValueError(f"Movement artifact differs from completion marker: {movement_path}")
    protocol = pq.read_table(protocol_path).to_pandas()
    cycles = protocol.loc[protocol["Type"].eq("Cycle")].sort_values("Beg").reset_index(drop=True)
    if len(cycles) < 94:
        raise ValueError(f"Expected at least 94 CS cycles for {recording_id}")
    intervals = [
        (int(cycles.iloc[trial - 1]["Beg"] + WINDOW_S[0] * 1000),
         int(cycles.iloc[trial - 1]["Beg"] + WINDOW_S[1] * 1000))
        for trial in range(5, 95)
    ]
    metric_columns = [METRIC_COLUMNS[metric_id] for metric_id in METRIC_COLUMNS]
    metrics = _read_selected_windows(
        metrics_path, ["FrameID", "AbsoluteTime", *metric_columns], intervals,
    )
    movement = _read_selected_windows(
        movement_path, ["FrameID", "AbsoluteTime", "valid", "moving", "bout_id"], intervals,
    )
    if len(metrics) != len(movement) or not np.array_equal(
        metrics[["FrameID", "AbsoluteTime"]].to_numpy(),
        movement[["FrameID", "AbsoluteTime"]].to_numpy(),
    ):
        raise ValueError(f"Metric and movement frames do not align for {recording_id}")
    return metrics, movement, cycles, (metrics_path, movement_path, protocol_path)


def calculate_fish_heatmaps(
    metrics: pd.DataFrame,
    movement: pd.DataFrame,
    cycles: pd.DataFrame,
    *,
    recording_id: str,
) -> pd.DataFrame:
    """Use the pipeline signed-vigor recipe for each metric and CS trial."""
    absolute = metrics["AbsoluteTime"].to_numpy(dtype=np.int64)
    if np.any(np.diff(absolute) < 0):
        raise ValueError(f"Nonchronological frame times for {recording_id}")
    detector_valid = movement["valid"].to_numpy(dtype=bool)
    moving = movement["moving"].to_numpy(dtype=bool)
    bout_ids = movement["bout_id"].to_numpy(dtype=np.int32)
    metric_values = {
        metric_id: metrics[column].to_numpy(dtype=float)
        for metric_id, column in METRIC_COLUMNS.items()
    }
    bins = np.arange(WINDOW_S[0], WINDOW_S[1], BIN_WIDTH_S) + BIN_WIDTH_S / 2
    bin_count = len(bins)
    rows = []
    for trial in range(5, 95):
        onset = int(cycles.iloc[trial - 1]["Beg"])
        start = np.searchsorted(absolute, onset + int(WINDOW_S[0] * 1000), side="left")
        stop = np.searchsorted(absolute, onset + int(WINDOW_S[1] * 1000), side="left")
        seconds = (absolute[start:stop] - onset) / 1000.0
        indices = np.floor((seconds - WINDOW_S[0]) / BIN_WIDTH_S).astype(np.int32)
        in_window = (indices >= 0) & (indices < bin_count)
        indices = indices[in_window]
        for metric_id, values in metric_values.items():
            signal = _signed_bout_log_vigor(
                values[start:stop][in_window], seconds[in_window], indices,
                detector_valid[start:stop][in_window], moving[start:stop][in_window],
                bout_ids[start:stop][in_window], bin_count=bin_count,
                baseline_end_s=-15.0,
            )
            rows.extend({
                "Recording ID": recording_id,
                "Metric ID": metric_id,
                "Trial number": trial,
                "Time bin center (s)": float(time),
                "Signed log vigor": float(value),
            } for time, value in zip(bins, signal))
    return pd.DataFrame(rows)


def _matrix(data: pd.DataFrame, metric_id: str, recording_id: str, start: int, end: int):
    selected = data.loc[
        data["Metric ID"].eq(metric_id) & data["Recording ID"].eq(recording_id)
    ]
    centers = np.arange(WINDOW_S[0], WINDOW_S[1], BIN_WIDTH_S) + BIN_WIDTH_S / 2
    return selected.pivot(
        index="Trial number", columns="Time bin center (s)", values="Signed log vigor"
    ).reindex(index=range(start, end + 1), columns=centers).to_numpy(dtype=float)


def render_comparison(data: pd.DataFrame, metric_id: str):
    theme = apply_theme()
    fig, axes = plt.subplots(
        3, 2, figsize=mm_to_in(183, 165),
        gridspec_kw={"height_ratios": [10, 50, 30]},
        sharex=True, constrained_layout=True,
    )
    palette = heatmap_cmap(theme.single_fish_scaled_vigor_cmap, theme)
    palette.set_bad("black")
    panel_ids = []
    mappings = {}
    image = None
    for row, (phase, start, end) in enumerate(PHASES):
        for column, (recording_id, fish_name) in enumerate((
            ("20221115_07", "Delay"), ("20221115_09", "Control"),
        )):
            axis = axes[row, column]
            panel_id = f"{fish_name.lower()}_{phase.lower().replace('-', '_')}"
            panel_ids.append(panel_id)
            values = np.ma.masked_invalid(_matrix(data, metric_id, recording_id, start, end))
            image = axis.imshow(
                values, aspect="auto", interpolation="nearest", origin="upper",
                extent=(WINDOW_S[0], WINDOW_S[1], end - start + 1, 0),
                cmap=palette, vmin=theme.single_fish_scaled_vigor_vmin,
                vmax=theme.single_fish_scaled_vigor_vmax, rasterized=True,
            )
            image_id = f"heatmap__{panel_id}__{metric_id}"
            image.set_gid(image_id)
            mappings[image_id] = {
                "recording_id": recording_id, "metric_id": metric_id,
                "phase": phase, "trial_start": str(start), "trial_end": str(end),
                "value_field": "Signed log vigor", "cmap": palette.name,
                "vmin": str(theme.single_fish_scaled_vigor_vmin),
                "vmax": str(theme.single_fish_scaled_vigor_vmax),
                "missing_color": "black",
            }
            for seconds in (0, 10):
                axis.axvline(seconds, color=theme.cs_color, linewidth=0.75)
            for trial in SELECTED_TRIALS:
                if start <= trial <= end:
                    axis.plot(20.35, trial - start + 0.5, marker="<", color="black",
                              markersize=3.5, clip_on=False)
            axis.set_xlim(*WINDOW_S)
            axis.set_ylim(end - start + 1, 0)
            axis.set_yticks([])
            axis.set_ylabel(phase if column == 0 else "")
            if row == 0:
                axis.set_title(f"{fish_name}  {recording_id}")
            if row == 2:
                axis.set_xlabel("Time relative to CS onset (s)")
            else:
                axis.tick_params(labelbottom=False)
            for spine in axis.spines.values():
                spine.set_linewidth(theme.axes_linewidth)
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.025)
    colorbar.set_label("Signed log vigor relative to pre-CS median")
    panel_ids.append("colorbar")
    fig.suptitle(f"Example fish · {METRIC_DISPLAY_NAMES[metric_id]}")
    return fig, panel_ids, mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = args.output_dir.resolve()
    frames = []
    inputs = []
    for fish in ("20221115_07", "20221115_09"):
        metrics, movement, cycles, paths = _load_fish(project, fish)
        frames.append(calculate_fish_heatmaps(metrics, movement, cycles, recording_id=fish))
        inputs.extend({"path": str(path), "sha256": _file_hash(path)} for path in paths)
        del metrics, movement
    data = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "figure-1-example-signed-log-vigor_panel-data.parquet"
    if panel_path.exists() and not args.overwrite:
        raise FileExistsError(panel_path)
    pq.write_table(pa.Table.from_pandas(data, preserve_index=False), panel_path)
    inputs.extend({"path": str(path), "sha256": _file_hash(path)} for path in (
        Path(profile_module.__file__).resolve(), Path(theme_module.__file__).resolve(),
        Path(__file__).with_name("render_legacy_ssd_example_traces.py").resolve(),
        panel_path,
    ))
    source = Path(__file__).resolve()
    for metric_id in METRIC_COLUMNS:
        fig, panel_ids, mappings = render_comparison(data, metric_id)
        base = output / f"figure-1-E_delay-control_{metric_id}"
        try:
            result = export_matplotlib_figure(
                fig, base,
                FigureProvenance(
                    figure_id="figure-1-E-delay-control-example-heatmaps",
                    analysis_recipe="signed-bout-log-vigor-corrected-ssd-v1",
                    source_file=str(source), source_symbol="main",
                    source_hash=sha256_file(source),
                    reproduction_snippet=(
                        f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src ./.venv/bin/python "
                        f"scripts/render_legacy_ssd_example_heatmaps.py "
                        f"--project-dir '{project}' --output-dir '{output}' "
                        f"--mode {args.mode} --overwrite"
                    ),
                    input_artifacts=tuple(inputs), artist_mappings=mappings,
                ),
                mode=FigureMode(args.mode), panel_ids=panel_ids, overwrite=args.overwrite,
            )
        finally:
            plt.close(fig)
        for path in (*result.outputs, result.sidecar):
            print(path)
    print(panel_path)


if __name__ == "__main__":
    main()
