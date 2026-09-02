"""Balanced local trace windows for human movement-detector review."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.artifacts import (
    artifact_staging,
    load_and_verify_source_manifest,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.analysis.movement_state import (
    METRIC_IDS,
    smooth_contiguous_median,
)
from classical_conditioning.config.domain import Alignment
from classical_conditioning.figures.theme import (
    SINGLE_COLUMN_MM,
    add_stimulus_window,
    apply_theme,
    qualitative_color,
    stacked_subplots,
    style_axes,
)
from classical_conditioning.preprocessing.candidates_v1 import CANDIDATE_COLUMNS


@dataclass(frozen=True)
class TraceReviewResult:
    recording_id: str
    traces_path: Path
    manifest_path: Path
    annotation_path: Path
    static_figure_path: Path
    interactive_figure_path: Path
    window_count: int


def select_review_windows(
    frames: pd.DataFrame,
    movement: pd.DataFrame,
    protocol: pd.DataFrame,
    *,
    half_window_ms: float = 2_000.0,
) -> pd.DataFrame:
    """Select deterministic quiet, strong, disagreement, and US windows."""
    if not np.array_equal(frames["FrameID"], movement["FrameID"]):
        raise ValueError("Candidate and movement frames do not align.")
    absolute = frames["AbsoluteTime"].to_numpy(dtype=np.int64)
    elapsed = frames["ElapsedTime"].to_numpy(dtype=float)
    xy_rms = frames[CANDIDATE_COLUMNS[2]].to_numpy(dtype=float)
    second = np.floor((elapsed - elapsed[0]) / 1_000).astype(np.int64)
    full_window = (
        (absolute >= absolute[0] + half_window_ms)
        & (absolute <= absolute[-1] - half_window_ms)
    )
    metric_valid = movement[
        [f"{metric_id}__valid" for metric_id in METRIC_IDS.values()]
    ].to_numpy(dtype=bool)
    jointly_valid = np.all(metric_valid, axis=1) & full_window
    xy_valid = (
        movement["whole_tail_xy_rms_speed__valid"].to_numpy(dtype=bool)
        & np.isfinite(xy_rms)
        & full_window
    )

    quiet_median = (
        pd.DataFrame(
            {
                "second": second[xy_valid],
                "value": xy_rms[xy_valid],
            }
        )
        .groupby("second", sort=True)["value"]
        .median()
    )
    quiet_candidates: list[int] = []
    for quiet_second in quiet_median.sort_values(kind="stable").index:
        candidates = np.flatnonzero((second == quiet_second) & jointly_valid)
        if candidates.size:
            quiet_candidates.append(int(candidates[len(candidates) // 2]))

    strong_valid_indices = np.flatnonzero(xy_valid)
    top_count = min(1_000, len(strong_valid_indices))
    if top_count:
        top_local = np.argpartition(
            -xy_rms[strong_valid_indices],
            top_count - 1,
        )[:top_count]
        strong_candidates = strong_valid_indices[
            top_local[np.argsort(-xy_rms[strong_valid_indices][top_local])]
        ].tolist()
    else:
        strong_candidates = []

    moving_columns = [
        f"{metric_id}__moving" for metric_id in METRIC_IDS.values()
    ]
    moving_matrix = movement[moving_columns].to_numpy(dtype=float)
    disagreement = np.var(moving_matrix, axis=1)
    disagreement_mean = (
        pd.DataFrame(
            {
                "second": second[jointly_valid],
                "value": disagreement[jointly_valid],
            }
        )
        .groupby("second", sort=True)["value"]
        .mean()
    )
    disagreement_candidates: list[int] = []
    for disagreement_second in disagreement_mean.sort_values(
        ascending=False,
        kind="stable",
    ).index:
        candidates = np.flatnonzero(
            (second == disagreement_second) & jointly_valid
        )
        if candidates.size:
            disagreement_candidates.append(
                int(candidates[np.argmax(disagreement[candidates])])
            )

    requests: list[tuple[str, str, list[int], int]] = []
    us_events = (
        protocol.loc[protocol["Type"].astype(str) == "Reinforcer", "Beg"]
        .sort_values(kind="stable")
        .to_numpy(dtype=np.int64)
    )
    if us_events.size:
        eligible_events = us_events[
            (us_events >= absolute[0] + half_window_ms)
            & (us_events <= absolute[-1] - half_window_ms)
        ]
        if eligible_events.size:
            for order, (label, event_index) in enumerate(
                (
                    ("us-early", 0),
                    ("us-middle", len(eligible_events) // 2),
                    ("us-late", len(eligible_events) - 1),
                ),
                start=3,
            ):
                center = int(eligible_events[event_index])
                frame_index = int(np.searchsorted(absolute, center))
                if jointly_valid[frame_index]:
                    requests.append(
                        (
                            label,
                            f"US event at {center}",
                            [frame_index],
                            order,
                        )
                    )

    requests.extend(
        [
            (
                "quiet",
                "lowest eligible 1 s whole-tail XY RMS median",
                quiet_candidates,
                0,
            ),
            (
                "strong",
                "highest eligible whole-tail XY RMS frame",
                strong_candidates,
                1,
            ),
            (
                "disagreement",
                "eligible 1 s interval with greatest detector disagreement",
                disagreement_candidates,
                2,
            ),
        ]
    )

    rows: list[dict[str, Any]] = []
    used_centers: list[int] = []
    minimum_center_distance = 2 * half_window_ms
    for label, reason, frame_indices, order in sorted(
        requests,
        key=lambda item: (item[0] not in {"us-early", "us-middle", "us-late"}, item[3]),
    ):
        selected_index = next(
            (
                index
                for index in frame_indices
                if all(
                    abs(int(absolute[index]) - used) >= minimum_center_distance
                    for used in used_centers
                )
            ),
            None,
        )
        if selected_index is None:
            continue
        frame_index = int(selected_index)
        center = int(absolute[frame_index])
        used_centers.append(center)
        rows.append(
            {
                "Window ID": label,
                "Selection reason": reason,
                "Center frame ID": int(frames["FrameID"].iloc[frame_index]),
                "Center absolute time (ms)": center,
                "Display order": order,
            }
        )
    return pd.DataFrame(rows).sort_values("Display order").reset_index(drop=True)


def extract_review_traces(
    frames: pd.DataFrame,
    movement: pd.DataFrame,
    windows: pd.DataFrame,
    calibration: dict[str, Any],
    *,
    half_window_ms: float = 2_000.0,
    smoothing_window_samples: int = 7,
) -> pd.DataFrame:
    """Extract long-form normalized traces around selected review centers."""
    absolute = frames["AbsoluteTime"].to_numpy(dtype=np.int64)
    frame_steps = frames["FrameStep"].to_numpy(dtype=np.int64)
    detector_inputs = {
        source_column: smooth_contiguous_median(
            frames[source_column].to_numpy(dtype=float),
            frame_steps,
            window_samples=smoothing_window_samples,
        )
        for source_column in CANDIDATE_COLUMNS
    }
    pieces: list[pd.DataFrame] = []
    for window in windows.itertuples(index=False):
        center = int(getattr(window, "_3"))
        start = int(np.searchsorted(absolute, center - half_window_ms, side="left"))
        end = int(np.searchsorted(absolute, center + half_window_ms, side="left"))
        for source_column, metric_id in METRIC_IDS.items():
            high = float(calibration[metric_id]["high_threshold"])
            low = float(calibration[metric_id]["low_threshold"])
            values = frames[source_column].iloc[start:end].to_numpy(dtype=float)
            detector_input = detector_inputs[source_column][start:end]
            pieces.append(
                pd.DataFrame(
                    {
                        "Window ID": getattr(window, "_0"),
                        "Selection reason": getattr(window, "_1"),
                        "Center absolute time (ms)": center,
                        "FrameID": frames["FrameID"].iloc[start:end].to_numpy(
                            dtype=np.int64
                        ),
                        "Relative time (ms)": (
                            absolute[start:end] - center
                        ).astype(np.int32),
                        "Metric ID": metric_id,
                        "Value": values,
                        "Detector input": detector_input,
                        "Detector input / high threshold": detector_input / high,
                        "Low / high threshold": low / high,
                        "Low threshold": low,
                        "High threshold": high,
                        "Valid": movement[f"{metric_id}__valid"]
                        .iloc[start:end]
                        .to_numpy(dtype=bool),
                        "Moving": movement[f"{metric_id}__moving"]
                        .iloc[start:end]
                        .to_numpy(dtype=bool),
                        "Bout ID": movement[f"{metric_id}__bout_id"]
                        .iloc[start:end]
                        .to_numpy(dtype=np.int32),
                    }
                )
            )
    return pd.concat(pieces, ignore_index=True)


def _static_figure(traces: pd.DataFrame, windows: pd.DataFrame) -> plt.Figure:
    theme = apply_theme()
    metric_ids = list(METRIC_IDS.values())
    colors = {
        metric_id: qualitative_color(index, theme)
        for index, metric_id in enumerate(metric_ids)
    }
    figure, axes = stacked_subplots(
        len(windows),
        width_mm=SINGLE_COLUMN_MM * 1.6,
        row_height_mm=32.0,
        sharex=True,
        theme=theme,
    )
    for axis, window in zip(axes, windows.itertuples(index=False)):
        window_id = str(getattr(window, "_0"))
        selected = traces[traces["Window ID"] == window_id]
        for metric_id, group in selected.groupby("Metric ID", observed=True):
            axis.plot(
                group["Relative time (ms)"] / 1_000,
                group["Detector input / high threshold"],
                color=colors[str(metric_id)],
                linewidth=theme.lines_linewidth,
                label=str(metric_id),
            )
            axis.axhline(
                float(group["Low / high threshold"].iloc[0]),
                color=colors[str(metric_id)],
                linestyle=":",
                linewidth=theme.lines_linewidth,
                alpha=0.5,
            )
            moving = group["Moving"].to_numpy(dtype=bool)
            axis.fill_between(
                group["Relative time (ms)"] / 1_000,
                0,
                1,
                where=moving,
                transform=axis.get_xaxis_transform(),
                color=colors[str(metric_id)],
                alpha=0.08,
            )
        axis.axhline(
            1,
            color=theme.baseline_color,
            linestyle="--",
            linewidth=theme.lines_linewidth,
            alpha=0.5,
        )
        if window_id.startswith("us-"):
            add_stimulus_window(
                axis,
                Alignment.US,
                onset_s=0.0,
                theme=theme,
            )
        else:
            axis.axvline(
                0,
                color=theme.baseline_color,
                linewidth=theme.lines_linewidth,
                alpha=0.7,
            )
        is_last = axis is axes[-1]
        style_axes(
            axis,
            theme=theme,
            show_xticks=is_last,
            show_yticks=True,
            xlabel="Time from selected center (s)",
            ylabel="Metric / high threshold",
        )
        axis.set_title(
            f"{window_id}: {getattr(window, '_1')}",
            loc="left",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=theme.legend_fontsize,
        frameon=False,
    )
    return figure


def _interactive_figure(traces: pd.DataFrame, windows: pd.DataFrame):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=len(windows),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"{getattr(window, '_0')}: {getattr(window, '_1')}"
            for window in windows.itertuples(index=False)
        ],
    )
    for row, window in enumerate(windows.itertuples(index=False), start=1):
        window_id = getattr(window, "_0")
        selected = traces[traces["Window ID"] == window_id]
        for metric_id, group in selected.groupby("Metric ID", observed=True):
            figure.add_trace(
                go.Scattergl(
                    x=group["Relative time (ms)"] / 1_000,
                    y=group["Detector input / high threshold"],
                    customdata=np.column_stack(
                        [
                            group["Moving"],
                            group["Valid"],
                            group["Bout ID"],
                            group["Value"],
                        ]
                    ),
                    hovertemplate=(
                        "t=%{x:.3f}s<br>detector/high=%{y:.3f}"
                        "<br>moving=%{customdata[0]}"
                        "<br>valid=%{customdata[1]}"
                        "<br>bout=%{customdata[2]}"
                        "<br>raw=%{customdata[3]:.5g}<extra>%{fullData.name}</extra>"
                    ),
                    mode="lines",
                    name=str(metric_id),
                    legendgroup=str(metric_id),
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )
        figure.add_hline(y=1, line_dash="dash", row=row, col=1)
        figure.add_vline(x=0, line_color="#FF3C00", row=row, col=1)
    figure.update_layout(
        height=max(500, len(windows) * 280),
        title="Candidate movement-detector trace review",
    )
    return figure


def build_trace_review(
    project_dir: Path,
    recording_id: str,
    *,
    overwrite: bool = False,
) -> TraceReviewResult:
    """Build local review data, figures, and an annotation template."""
    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    metric_path = source_dir / "frame_activity_candidates-v1.parquet"
    movement_path = source_dir / "movement_state_candidates-v1.parquet"
    protocol_path = source_dir / "stimulus_events.parquet"
    movement_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "movement-candidate-v1_summary.json"
    )
    metric_marker = json.loads(
        (
            project_dir
            / "Metadata"
            / f"{recording_id}_candidate-v1_complete.json"
        ).read_text(encoding="utf-8")
    )
    movement_marker = json.loads(
        (
            project_dir
            / "Metadata"
            / f"{recording_id}_movement-candidate-v1_complete.json"
        ).read_text(encoding="utf-8")
    )
    if sha256_file(metric_path) != metric_marker["metrics_sha256"]:
        raise ValueError("Candidate metric hash differs from its marker.")
    if sha256_file(movement_path) != movement_marker["movement_sha256"]:
        raise ValueError("Movement-state hash differs from its marker.")
    if sha256_file(movement_summary_path) != movement_marker["summary_sha256"]:
        raise ValueError("Movement-state summary hash differs from its marker.")
    movement_summary = json.loads(
        movement_summary_path.read_text(encoding="utf-8")
    )
    metric_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "candidate-v1_activity_summary.json"
    )
    metric_summary = json.loads(metric_summary_path.read_text(encoding="utf-8"))
    if (
        metric_marker.get("status") != "complete"
        or metric_marker.get("recipe") != "tail-candidate-development-v1"
        or metric_marker.get("recording_id") != recording_id
        or sha256_file(metric_summary_path) != metric_marker.get("summary_sha256")
    ):
        raise ValueError("Candidate metric marker or summary is invalid.")
    if (
        movement_marker.get("status") != "complete"
        or movement_marker.get("recipe") != "movement-candidate-v1"
        or movement_marker.get("recording_id") != recording_id
        or movement_summary.get("recipe") != "movement-candidate-v1"
        or movement_summary.get("recording_id") != recording_id
        or movement_summary["inputs"]["candidate_metrics"]["sha256"]
        != metric_marker["metrics_sha256"]
    ):
        raise ValueError("Movement-state lineage is invalid.")
    _, intake_artifacts, input_state = load_and_verify_source_manifest(
        project_dir,
        recording_id,
    )
    for kind, record in intake_artifacts.items():
        if metric_summary["input_artifacts"][kind]["sha256"] != record["sha256"]:
            raise ValueError(
                f"Candidate metrics were built from a different {kind} artifact."
            )
    if sha256_file(protocol_path) != intake_artifacts["protocol"]["sha256"]:
        raise ValueError("Protocol artifact differs from the source manifest.")

    output_dir = (
        project_dir / "Quality checks" / recording_id / "Trace review"
    )
    traces_path = output_dir / "trace_review_windows-v1.parquet"
    manifest_path = output_dir / "trace_review_manifest-v1.json"
    annotation_path = output_dir / "trace_review_annotations-v1.csv"
    static_path = output_dir / "trace_review-v1.png"
    interactive_path = output_dir / "trace_review-v1.html"
    outputs = (
        traces_path,
        manifest_path,
        annotation_path,
        static_path,
        interactive_path,
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Trace review outputs already exist: {existing}")
    if annotation_path.exists() and overwrite:
        raise FileExistsError(
            "Refusing to overwrite a human annotation file. Move or version it first."
        )

    frame_columns = [
        "FrameID",
        "ElapsedTime",
        "AbsoluteTime",
        "FrameStep",
        *CANDIDATE_COLUMNS,
    ]
    frames = pq.read_table(metric_path, columns=frame_columns).to_pandas()
    movement_columns = ["FrameID"]
    for metric_id in METRIC_IDS.values():
        movement_columns.extend(
            [
                f"{metric_id}__valid",
                f"{metric_id}__moving",
                f"{metric_id}__bout_id",
            ]
        )
    movement = pq.read_table(movement_path, columns=movement_columns).to_pandas()
    protocol = pq.read_table(protocol_path).to_pandas()
    metric_state = (metric_path.stat().st_size, metric_path.stat().st_mtime_ns)
    movement_state = (
        movement_path.stat().st_size,
        movement_path.stat().st_mtime_ns,
    )
    windows = select_review_windows(frames, movement, protocol)
    traces = extract_review_traces(
        frames,
        movement,
        windows,
        movement_summary["calibration"],
        smoothing_window_samples=int(
            movement_summary["resolved_smoothing_window_samples"]
        ),
    )
    annotations = windows.copy()
    annotations["Reviewer"] = ""
    annotations["Movement/rest"] = ""
    annotations["Bout boundaries acceptable"] = ""
    annotations["Relative strength"] = ""
    annotations["Coordinated/irregular"] = ""
    annotations["Tracking usable"] = ""
    annotations["Notes"] = ""

    output_dir.mkdir(parents=True, exist_ok=True)
    with artifact_staging(
        output_dir,
        prefix=f".{recording_id}-trace-review-v1-",
    ) as staging_root:
        staged_traces = staging_root / traces_path.name
        staged_manifest = staging_root / manifest_path.name
        staged_annotations = staging_root / annotation_path.name
        staged_static = staging_root / static_path.name
        staged_interactive = staging_root / interactive_path.name
        pq.write_table(
            pa.Table.from_pandas(traces, preserve_index=False),
            staged_traces,
            compression="zstd",
            write_statistics=True,
        )
        annotations.to_csv(staged_annotations, index=False)
        figure = _static_figure(traces, windows)
        try:
            figure.savefig(staged_static, dpi=200)
        finally:
            plt.close(figure)
        interactive = _interactive_figure(traces, windows)
        interactive.write_html(
            staged_interactive,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
            config={"displaylogo": False},
        )
        manifest = {
            "recipe": "trace-review-v1",
            "scientific_status": "candidate_development",
            "recording_id": recording_id,
            "window_count": len(windows),
            "windows": windows.to_dict(orient="records"),
            "annotation_status": "unreviewed",
            "inputs": {
                "candidate_metrics": {
                    "path": str(metric_path),
                    "sha256": metric_marker["metrics_sha256"],
                },
                "movement_state": {
                    "path": str(movement_path),
                    "sha256": movement_marker["movement_sha256"],
                },
                "metric_summary": {
                    "path": str(metric_summary_path),
                    "sha256": metric_marker["summary_sha256"],
                },
                "movement_summary": {
                    "path": str(movement_summary_path),
                    "sha256": movement_marker["summary_sha256"],
                },
                "protocol": intake_artifacts["protocol"],
            },
            "outputs": {
                "traces": {
                    "path": str(traces_path),
                    "sha256": sha256_file(staged_traces),
                },
                "annotations": {"path": str(annotation_path)},
                "static": {
                    "path": str(static_path),
                    "sha256": sha256_file(staged_static),
                },
                "interactive": {
                    "path": str(interactive_path),
                    "sha256": sha256_file(staged_interactive),
                },
            },
        }
        write_json_atomic(staged_manifest, manifest)
        if (
            metric_state
            != (metric_path.stat().st_size, metric_path.stat().st_mtime_ns)
            or movement_state
            != (movement_path.stat().st_size, movement_path.stat().st_mtime_ns)
            or sha256_file(metric_path) != metric_marker["metrics_sha256"]
            or sha256_file(movement_path) != movement_marker["movement_sha256"]
        ):
            raise RuntimeError("Trace-review inputs changed during generation.")
        protocol_stat = protocol_path.stat()
        if (
            input_state["protocol"]
            != (protocol_stat.st_size, protocol_stat.st_mtime_ns)
            or sha256_file(protocol_path) != intake_artifacts["protocol"]["sha256"]
        ):
            raise RuntimeError("Protocol changed during trace-review generation.")

        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_created = False
        try:
            with annotation_path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(staged_annotations.read_text(encoding="utf-8"))
            annotation_created = True
            publish_transaction(
                (
                    (staged_traces, traces_path),
                    (staged_manifest, manifest_path),
                    (staged_static, static_path),
                    (staged_interactive, interactive_path),
                ),
                staging_root,
                overwrite=overwrite,
            )
        except Exception:
            if annotation_created and annotation_path.exists():
                annotation_path.unlink()
            raise
    return TraceReviewResult(
        recording_id=recording_id,
        traces_path=traces_path,
        manifest_path=manifest_path,
        annotation_path=annotation_path,
        static_figure_path=static_path,
        interactive_figure_path=interactive_path,
        window_count=len(windows),
    )
