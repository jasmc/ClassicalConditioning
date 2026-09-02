"""Exploratory whole-tail activity metrics from measured tracking coordinates."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.artifacts import (
    artifact_staging,
    load_and_verify_source_manifest as _load_and_verify_source_manifest,
    publish_transaction as _publish_transaction,
    sha256_file as _sha256_file,
    write_json_atomic as _write_json_atomic,
)

CANDIDATE_COLUMNS = (
    "segment_absolute_angular_speed_sum_rad_per_ms",
    "all_segment_angular_rms_rad_per_ms",
    "whole_tail_xy_rms_speed_px_per_ms",
    "whole_tail_xy_mean_speed_px_per_ms",
    "curvature_change_rms_rad_per_px_per_ms",
)


@dataclass(frozen=True)
class CandidateMetricConfig:
    point_count: int = 16
    minimum_valid_tail_fraction: float = 0.8
    terminal_angle_is_placeholder: bool = True
    body_translation_correction: str = "subtract_tail_base"
    body_rotation_correction: str = "none"
    time_source: str = "camera_elapsed_time_ms"
    gap_policy: str = "invalidate_derivative_when_frame_step_is_not_one"
    maximum_mean_local_bend_geometry_error_rad: float = 0.001
    minimum_geometry_validation_fraction: float = 0.99


@dataclass(frozen=True)
class CandidateMetricResult:
    recording_id: str
    metrics_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int
    valid_derivative_count: int


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _validate_frame_order(
    frame_ids: np.ndarray,
    previous_frame_id: int | None,
) -> int:
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if frame_ids.size == 0:
        raise ValueError("Tracking frame batch is empty.")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("Tracking FrameID values must be strictly increasing.")
    if previous_frame_id is not None and previous_frame_id >= int(frame_ids[0]):
        raise ValueError("Tracking FrameID order is invalid across chunk boundaries.")
    return int(frame_ids[-1])


def _geometry_agreement(
    x: np.ndarray,
    y: np.ndarray,
    local_angles: np.ndarray,
) -> tuple[float, int, int]:
    delta_x = np.diff(x, axis=1)
    delta_y = np.diff(y, axis=1)
    segment_lengths = np.hypot(delta_x, delta_y)
    segment_orientation = np.arctan2(delta_y, delta_x)
    observed_bend = _wrap_angle(np.diff(segment_orientation, axis=1))
    stated_bend = local_angles[:, 1:-1]
    eligible = (
        np.isfinite(observed_bend)
        & np.isfinite(stated_bend)
        & np.isfinite(segment_lengths[:, :-1])
        & np.isfinite(segment_lengths[:, 1:])
        & (segment_lengths[:, :-1] > 0)
        & (segment_lengths[:, 1:] > 0)
    )
    error = np.abs(_wrap_angle(observed_bend - stated_bend))
    return float(np.sum(error[eligible])), int(np.count_nonzero(eligible)), int(error.size)


def _tail_point_weights(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    segment_lengths = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    valid_lengths = np.where(
        np.isfinite(segment_lengths) & (segment_lengths > 0),
        segment_lengths,
        np.nan,
    )
    with np.errstate(all="ignore"):
        typical_length = np.nanmedian(valid_lengths, axis=1)
    reference_lengths = np.where(
        np.isfinite(valid_lengths),
        valid_lengths,
        typical_length[:, None],
    )
    weights = np.empty_like(x, dtype=np.float64)
    weights[:, 0] = 0.5 * reference_lengths[:, 0]
    weights[:, -1] = 0.5 * reference_lengths[:, -1]
    weights[:, 1:-1] = 0.5 * (
        reference_lengths[:, :-1] + reference_lengths[:, 1:]
    )
    return weights, reference_lengths


def _weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    minimum_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    total_weight = np.nansum(np.where(np.isfinite(weights), weights, 0.0), axis=1)
    valid_weight = np.sum(np.where(valid, weights, 0.0), axis=1)
    fraction = np.divide(
        valid_weight,
        total_weight,
        out=np.zeros_like(valid_weight),
        where=total_weight > 0,
    )
    numerator = np.sum(np.where(valid, values * weights, 0.0), axis=1)
    result = np.divide(
        numerator,
        valid_weight,
        out=np.full_like(valid_weight, np.nan),
        where=(valid_weight > 0) & (fraction >= minimum_fraction),
    )
    return result, fraction


def _weighted_rms(
    values: np.ndarray,
    weights: np.ndarray,
    minimum_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean_square, fraction = _weighted_mean(
        np.square(values),
        weights,
        minimum_fraction,
    )
    return np.sqrt(mean_square), fraction


def calculate_candidate_metrics(
    frame_ids: np.ndarray,
    elapsed_time_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    local_angles: np.ndarray,
    *,
    config: CandidateMetricConfig,
    previous: dict[str, np.ndarray | float | int] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | float | int]]:
    """Calculate candidate metrics for one ordered chunk with carry state."""
    if x.shape != y.shape or x.shape != local_angles.shape:
        raise ValueError("x, y, and local-angle arrays must have identical shapes.")
    if x.shape[1] != config.point_count:
        raise ValueError(
            f"Expected {config.point_count} tail points, received {x.shape[1]}."
        )
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    elapsed_time_ms = np.asarray(elapsed_time_ms, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    local_angles = np.asarray(local_angles, dtype=np.float64)
    _validate_frame_order(
        frame_ids,
        int(previous["frame_id"]) if previous is not None else None,
    )

    base_x = x[:, [0]]
    base_y = y[:, [0]]
    body_x = x - base_x
    body_y = y - base_y

    if previous is not None:
        frame_with_previous = np.concatenate(
            [np.array([previous["frame_id"]], dtype=np.int64), frame_ids]
        )
        time_with_previous = np.concatenate(
            [np.array([previous["elapsed_time_ms"]], dtype=np.float64), elapsed_time_ms]
        )
        x_with_previous = np.vstack([previous["body_x"], body_x])
        y_with_previous = np.vstack([previous["body_y"], body_y])
    else:
        frame_with_previous = frame_ids
        time_with_previous = elapsed_time_ms
        x_with_previous = body_x
        y_with_previous = body_y

    frame_steps = np.diff(frame_with_previous)
    delta_time = np.diff(time_with_previous)
    derivative_valid = (frame_steps == 1) & np.isfinite(delta_time) & (delta_time > 0)
    if previous is None:
        frame_step_for_rows = np.concatenate([[0], frame_steps])
        derivative_valid = np.concatenate([[False], derivative_valid])
        delta_time_for_rows = np.concatenate([[np.nan], delta_time])
        delta_x = np.vstack([np.full((1, x.shape[1]), np.nan), np.diff(body_x, axis=0)])
        delta_y = np.vstack([np.full((1, y.shape[1]), np.nan), np.diff(body_y, axis=0)])
    else:
        frame_step_for_rows = frame_steps
        delta_time_for_rows = delta_time
        delta_x = np.diff(x_with_previous, axis=0)
        delta_y = np.diff(y_with_previous, axis=0)

    valid_time = delta_time_for_rows[:, None]
    point_speed = np.hypot(delta_x, delta_y) / valid_time
    point_weights, segment_lengths = _tail_point_weights(body_x, body_y)
    xy_rms, valid_tail_fraction = _weighted_rms(
        point_speed,
        point_weights,
        config.minimum_valid_tail_fraction,
    )
    xy_mean, _ = _weighted_mean(
        point_speed,
        point_weights,
        config.minimum_valid_tail_fraction,
    )

    orientation_count = config.point_count - 1
    current_segment_orientation = np.arctan2(
        np.diff(body_y, axis=1),
        np.diff(body_x, axis=1),
    )
    if previous is None:
        orientation_delta = np.vstack(
            [
                np.full((1, orientation_count), np.nan),
                _wrap_angle(np.diff(current_segment_orientation, axis=0)),
            ]
        )
    else:
        orientation_source = np.vstack(
            [previous["segment_orientation"], current_segment_orientation]
        )
        orientation_delta = _wrap_angle(np.diff(orientation_source, axis=0))
    angular_speed = np.abs(orientation_delta) / valid_time
    segment_weights = segment_lengths[:, :orientation_count]
    angular_rms, angular_valid_fraction = _weighted_rms(
        angular_speed,
        segment_weights,
        config.minimum_valid_tail_fraction,
    )
    manuscript_sum = np.sum(angular_speed, axis=1)
    manuscript_sum[~np.all(np.isfinite(angular_speed), axis=1)] = np.nan

    local_bend_count = config.point_count - 2
    bend_arc_length = 0.5 * (
        segment_lengths[:, :-1] + segment_lengths[:, 1:]
    )
    current_curvature = (
        local_angles[:, 1 : 1 + local_bend_count] / bend_arc_length
    )
    if previous is None:
        curvature_delta = np.vstack(
            [
                np.full((1, local_bend_count), np.nan),
                np.diff(current_curvature, axis=0),
            ]
        )
    else:
        curvature_source = np.vstack(
            [previous["local_curvature"], current_curvature]
        )
        curvature_delta = np.diff(curvature_source, axis=0)
    curvature_rate = np.abs(curvature_delta) / valid_time
    curvature_rms, curvature_valid_fraction = _weighted_rms(
        curvature_rate,
        bend_arc_length,
        config.minimum_valid_tail_fraction,
    )

    for values in (xy_rms, xy_mean, angular_rms, manuscript_sum, curvature_rms):
        values[~derivative_valid] = np.nan

    result = pd.DataFrame(
        {
            "FrameID": frame_ids,
            "ElapsedTime": elapsed_time_ms,
            "FrameStep": frame_step_for_rows.astype(np.int64),
            "DeltaTimeMs": delta_time_for_rows,
            "valid_derivative": derivative_valid,
            "xy_valid_tail_fraction": valid_tail_fraction,
            "angular_valid_tail_fraction": angular_valid_fraction,
            "curvature_valid_tail_fraction": curvature_valid_fraction,
            CANDIDATE_COLUMNS[0]: manuscript_sum,
            CANDIDATE_COLUMNS[1]: angular_rms,
            CANDIDATE_COLUMNS[2]: xy_rms,
            CANDIDATE_COLUMNS[3]: xy_mean,
            CANDIDATE_COLUMNS[4]: curvature_rms,
        }
    )
    state: dict[str, np.ndarray | float | int] = {
        "frame_id": int(frame_ids[-1]),
        "elapsed_time_ms": float(elapsed_time_ms[-1]),
        "body_x": body_x[-1],
        "body_y": body_y[-1],
        "local_angles": local_angles[-1],
        "local_curvature": current_curvature[-1],
        "segment_orientation": current_segment_orientation[-1],
    }
    return result, state


def _tracking_columns(point_count: int) -> list[str]:
    return ["FrameID"] + [
        f"{prefix}{index}"
        for prefix in ("x", "y", "angle")
        for index in range(point_count)
    ]


def _extract_arrays(
    tracking: pd.DataFrame,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = tracking[[f"x{index}" for index in range(point_count)]].to_numpy(
        dtype=np.float64
    )
    y = tracking[[f"y{index}" for index in range(point_count)]].to_numpy(
        dtype=np.float64
    )
    angles = tracking[
        [f"angle{index}" for index in range(point_count)]
    ].to_numpy(dtype=np.float64)
    return x, y, angles


def build_candidate_activity_metrics(
    project_dir: Path,
    recording_id: str,
    *,
    config: CandidateMetricConfig | None = None,
    batch_size: int = 250_000,
    overwrite: bool = False,
) -> CandidateMetricResult:
    """Build candidate frame metrics from measured local tracking data."""
    config = config or CandidateMetricConfig()
    if config != CandidateMetricConfig():
        raise ValueError(
            "tail-candidate-development-v1 uses a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    camera_path = source_dir / "camera.parquet"
    tracking_path = source_dir / "tracking.parquet"
    recording_name, input_artifacts, input_state = _load_and_verify_source_manifest(
        project_dir,
        recording_id,
    )

    metrics_path = source_dir / "frame_activity_candidates-v1.parquet"
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "candidate-v1_activity_summary.json"
    )
    marker_path = (
        project_dir
        / "Metadata"
        / f"{recording_id}_candidate-v1_complete.json"
    )
    existing = [
        path for path in (metrics_path, summary_path, marker_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Candidate metric outputs already exist: {existing}")

    camera = pq.read_table(
        camera_path,
        columns=["FrameID", "ElapsedTime", "AbsoluteTime"],
    ).to_pandas()
    camera_ids = camera["FrameID"].to_numpy(dtype=np.int64)
    camera_elapsed = camera["ElapsedTime"].to_numpy(dtype=np.float64)
    camera_absolute = camera["AbsoluteTime"].to_numpy(dtype=np.int64)
    if np.any(np.diff(camera_ids) <= 0):
        raise ValueError("Camera FrameID must be strictly increasing.")

    tracking_file = pq.ParquetFile(tracking_path)
    point_count = config.point_count
    columns = _tracking_columns(point_count)
    state: dict[str, np.ndarray | float | int] | None = None
    row_count = 0
    valid_derivative_count = 0
    invalid_gap_count = 0
    metric_minima = {column: np.inf for column in CANDIDATE_COLUMNS}
    metric_maxima = {column: -np.inf for column in CANDIDATE_COLUMNS}
    geometry_error_sum = 0.0
    geometry_error_count = 0
    geometry_total_count = 0
    terminal_angle_nonzero_count = 0
    previous_raw_tracking_frame_id: int | None = None

    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-candidate-v1-",
    ) as staging_root:
        staged_metrics = staging_root / "frame_activity_candidates-v1.parquet"
        staged_summary = staging_root / "candidate-v1_activity_summary.json"
        staged_marker = staging_root / "candidate-v1_complete.json"
        writer: pq.ParquetWriter | None = None
        try:
            for batch in tracking_file.iter_batches(
                batch_size=batch_size,
                columns=columns,
            ):
                tracking = batch.to_pandas()
                frame_ids = tracking["FrameID"].to_numpy(dtype=np.int64)
                previous_raw_tracking_frame_id = _validate_frame_order(
                    frame_ids,
                    previous_raw_tracking_frame_id,
                )
                camera_indices = np.searchsorted(camera_ids, frame_ids)
                matched = camera_indices < len(camera_ids)
                matched[matched] &= (
                    camera_ids[camera_indices[matched]] == frame_ids[matched]
                )
                if not np.any(matched):
                    continue
                tracking = tracking.loc[matched].reset_index(drop=True)
                camera_indices = camera_indices[matched]
                frame_ids = frame_ids[matched]
                elapsed = camera_elapsed[camera_indices]
                absolute = camera_absolute[camera_indices]
                x, y, angles = _extract_arrays(tracking, point_count)

                error_sum, error_count, total_count = _geometry_agreement(
                    x, y, angles
                )
                geometry_error_sum += error_sum
                geometry_error_count += error_count
                geometry_total_count += total_count
                terminal_angle_nonzero_count += int(
                    np.count_nonzero(np.abs(angles[:, -1]) > 1e-12)
                )

                metrics, state = calculate_candidate_metrics(
                    frame_ids,
                    elapsed,
                    x,
                    y,
                    angles,
                    config=config,
                    previous=state,
                )
                metrics.insert(2, "AbsoluteTime", absolute)
                if writer is None:
                    schema = pa.Table.from_pandas(
                        metrics,
                        preserve_index=False,
                    ).schema.with_metadata(
                        {
                            b"recipe": b"tail-candidate-development-v1",
                            b"scientific_status": b"candidate_development",
                            b"recording_id": recording_id.encode("utf-8"),
                            b"source_tracking_sha256": input_artifacts["tracking"][
                                "sha256"
                            ].encode("ascii"),
                        }
                    )
                    writer = pq.ParquetWriter(
                        staged_metrics,
                        schema,
                        compression="zstd",
                        write_statistics=True,
                    )
                table = pa.Table.from_pandas(
                    metrics,
                    schema=writer.schema,
                    preserve_index=False,
                    safe=True,
                )
                writer.write_table(table, row_group_size=len(metrics))
                row_count += len(metrics)
                valid_derivative_count += int(metrics["valid_derivative"].sum())
                invalid_gap_count += int(
                    (
                        metrics["FrameStep"].ne(1)
                        & metrics["FrameStep"].ne(0)
                    ).sum()
                )
                for column in CANDIDATE_COLUMNS:
                    values = metrics[column].to_numpy(dtype=float)
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        metric_minima[column] = min(
                            metric_minima[column], float(np.min(finite))
                        )
                        metric_maxima[column] = max(
                            metric_maxima[column], float(np.max(finite))
                        )
        finally:
            tracking_file.close()
            if writer is not None:
                writer.close()

        if writer is None or row_count == 0:
            raise ValueError("No overlapping camera/tracking rows were available.")
        if config.terminal_angle_is_placeholder and terminal_angle_nonzero_count:
            raise ValueError(
                "Terminal angle was configured as a placeholder but contains "
                f"{terminal_angle_nonzero_count} non-zero values."
            )
        mean_geometry_error = (
            geometry_error_sum / geometry_error_count
            if geometry_error_count
            else None
        )
        geometry_coverage = (
            geometry_error_count / geometry_total_count
            if geometry_total_count
            else 0.0
        )
        if (
            mean_geometry_error is None
            or mean_geometry_error
            > config.maximum_mean_local_bend_geometry_error_rad
            or geometry_coverage < config.minimum_geometry_validation_fraction
        ):
            raise ValueError(
                "Raw angle semantics do not meet the frozen measured-geometry "
                f"criteria: mean_error={mean_geometry_error}, "
                f"coverage={geometry_coverage}."
            )

        for kind, path in {
            "camera": camera_path,
            "tracking": tracking_path,
            "protocol": source_dir / "stimulus_events.parquet",
        }.items():
            stat = path.stat()
            if input_state[kind] != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"{kind} intake artifact changed during metric build.")
            if _sha256_file(path) != input_artifacts[kind]["sha256"]:
                raise RuntimeError(f"{kind} intake artifact hash changed during metric build.")

        metric_hash = _sha256_file(staged_metrics)
        summary: dict[str, Any] = {
            "recipe": "tail-candidate-development-v1",
            "scientific_status": "candidate_development",
            "recording_id": recording_id,
            "recording_name": recording_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "row_count": row_count,
            "valid_derivative_count": valid_derivative_count,
            "invalid_derivative_count": row_count - valid_derivative_count,
            "invalid_gap_count": invalid_gap_count,
            "metric_ranges": {
                column: {
                    "minimum": (
                        metric_minima[column]
                        if np.isfinite(metric_minima[column])
                        else None
                    ),
                    "maximum": (
                        metric_maxima[column]
                        if np.isfinite(metric_maxima[column])
                        else None
                    ),
                }
                for column in CANDIDATE_COLUMNS
            },
            "tracking_semantics": {
                "mean_absolute_local_bend_geometry_error_rad": (
                    mean_geometry_error
                ),
                "maximum_accepted_mean_error_rad": (
                    config.maximum_mean_local_bend_geometry_error_rad
                ),
                "semantics_validated": True,
                "geometry_comparison_count": geometry_error_count,
                "geometry_potential_count": geometry_total_count,
                "geometry_validation_fraction": geometry_coverage,
                "minimum_geometry_validation_fraction": (
                    config.minimum_geometry_validation_fraction
                ),
                "terminal_angle_nonzero_count": terminal_angle_nonzero_count,
                "interpretation": (
                    "angle1..angle14 are compared with changes between adjacent "
                    "measured segment orientations and used for curvature; "
                    "angular-speed metrics are derived directly from measured "
                    "XY segment orientations; angle15 is a terminal placeholder."
                ),
            },
            "known_limitations": [
                "Candidate metrics are exploratory and not paper-approved.",
                "Body translation is removed by subtracting tail-base position.",
                "No independent body-axis measurement is available, so rotation is not corrected.",
                "No temporal or spatial smoothing is applied in candidate-v1.",
                "Movement-state thresholds and bouts are not defined in this artifact.",
            ],
            "input_artifacts": input_artifacts,
            "artifact": {
                "path": str(metrics_path.resolve()),
                "sha256": metric_hash,
                "rows": row_count,
                "compression": "zstd",
                "compression_lossless": True,
            },
        }
        _write_json_atomic(staged_summary, summary)
        summary_hash = _sha256_file(staged_summary)
        _write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": "tail-candidate-development-v1",
                "recording_id": recording_id,
                "metrics_sha256": metric_hash,
                "summary_sha256": summary_hash,
            },
        )
        _publish_transaction(
            (
                (staged_metrics, metrics_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return CandidateMetricResult(
        recording_id=recording_id,
        metrics_path=metrics_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_count=row_count,
        valid_derivative_count=valid_derivative_count,
    )
