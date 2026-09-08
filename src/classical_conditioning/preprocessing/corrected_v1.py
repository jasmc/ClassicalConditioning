"""Corrected measured-time preprocessing contract (Step 06 foundation).

Distinct from frozen ``legacy-paper-v1``. This recipe preserves measured camera
timestamps, does not apply the legacy startup discard / uniform 700 Hz grid /
AbsoluteTime rebuild, and records explicit validity masks. Interpolation and
filtering remain disabled until separately versioned policies are approved
(Gate P). Body rotation is explicitly ``none`` when no independent body axis
exists (Gate T0).
"""

from __future__ import annotations

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

RECIPE_ID = "corrected-preprocess-v1"
SCIENTIFIC_STATUS = "candidate_development"
ARTIFACT_NAME = "frame_preprocessed_corrected-v1.parquet"
SUMMARY_NAME = "corrected-v1_preprocessing_summary.json"
MARKER_SUFFIX = "corrected-preprocess-v1_complete.json"


@dataclass(frozen=True)
class CorrectedPreprocessConfig:
    """Frozen policy set for ``corrected-preprocess-v1``.

    Changing any field requires a new recipe identity.
    """

    point_count: int = 16
    time_source: str = "camera_elapsed_time_ms"
    frame_gap_policy: str = "invalidate_cross_gap_derivative"
    maximum_valid_derivative_interval_ms: float = 10.0
    long_gap_policy: str = "invalidate"
    interpolation_enabled: bool = False
    interpolation_method: str = "none"
    timestamp_jitter_tolerance_ms: float = 0.5
    duplicate_frame_policy: str = "reject"
    nonmonotonic_timestamp_policy: str = "reject"
    temporal_filter_enabled: bool = False
    spatial_filter_enabled: bool = False
    filter_edge_policy: str = "not_applicable"
    body_translation_correction: str = "subtract_tail_base"
    body_rotation_correction: str = "none"
    minimum_valid_point_fraction: float = 0.8
    drop_initial_camera_rows: int = 0
    drop_trailing_tracking_summary_row: bool = False
    coordinate_source: str = "measured"


@dataclass(frozen=True)
class CorrectedPreprocessResult:
    recording_id: str
    frames_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int
    derivative_valid_count: int


def _validate_strictly_increasing_ids(
    frame_ids: np.ndarray,
    *,
    previous_frame_id: int | None,
    label: str,
) -> int:
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if frame_ids.size == 0:
        raise ValueError(f"{label} frame batch is empty.")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError(
            f"{label} FrameID values must be strictly increasing "
            f"(duplicate_frame_policy=reject)."
        )
    if previous_frame_id is not None and previous_frame_id >= int(frame_ids[0]):
        raise ValueError(
            f"{label} FrameID order is invalid across chunk boundaries."
        )
    return int(frame_ids[-1])


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


def summarize_protocol_timing(
    protocol: pd.DataFrame,
    camera_frame_ids: np.ndarray,
    camera_absolute_ms: np.ndarray,
) -> dict[str, Any]:
    """Classify protocol events relative to observed camera AbsoluteTime."""
    if protocol.empty:
        return {
            "event_count": 0,
            "events_outside_acquisition": 0,
            "events_between_observed_frames": 0,
            "events_on_observed_frames": 0,
        }
    required = {"Type", "Beg", "End"}
    missing = required.difference(protocol.columns)
    if missing:
        raise ValueError(
            "Protocol table requires columns "
            f"{sorted(required)}; missing {sorted(missing)}."
        )
    starts = protocol["Beg"].to_numpy(dtype=np.float64)
    first = float(camera_absolute_ms[0])
    last = float(camera_absolute_ms[-1])
    outside = (starts < first) | (starts > last)
    indices = np.searchsorted(camera_absolute_ms, starts, side="right") - 1
    in_range = ~outside
    on_frame = np.zeros(len(starts), dtype=bool)
    between = np.zeros(len(starts), dtype=bool)
    for index, start in enumerate(starts):
        if not in_range[index]:
            continue
        camera_index = int(indices[index])
        if camera_index < 0 or camera_index >= len(camera_absolute_ms):
            outside[index] = True
            continue
        if abs(float(camera_absolute_ms[camera_index]) - float(start)) <= 1e-9:
            on_frame[index] = True
        elif camera_index + 1 < len(camera_absolute_ms):
            between[index] = True
        else:
            on_frame[index] = True
    return {
        "event_count": int(len(starts)),
        "events_outside_acquisition": int(np.count_nonzero(outside)),
        "events_between_observed_frames": int(np.count_nonzero(between)),
        "events_on_observed_frames": int(np.count_nonzero(on_frame)),
        "camera_frame_id_span": [
            int(camera_frame_ids[0]),
            int(camera_frame_ids[-1]),
        ],
        "camera_absolute_ms_span": [first, last],
    }


def calculate_corrected_frames(
    frame_ids: np.ndarray,
    elapsed_time_ms: np.ndarray,
    absolute_time_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    local_angles: np.ndarray,
    *,
    config: CorrectedPreprocessConfig,
    previous: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one corrected frame table chunk with explicit validity masks."""
    if config.interpolation_enabled:
        raise ValueError(
            "corrected-preprocess-v1 keeps interpolation_enabled=False until "
            "Gate P freezes an interpolation policy."
        )
    if config.temporal_filter_enabled or config.spatial_filter_enabled:
        raise ValueError(
            "corrected-preprocess-v1 keeps filtering disabled until Gate P "
            "freezes filter policies."
        )
    if config.body_rotation_correction != "none":
        raise ValueError(
            "corrected-preprocess-v1 records body_rotation_correction='none' "
            "(no independent body-axis field in current tracking)."
        )
    if config.body_translation_correction != "subtract_tail_base":
        raise ValueError(
            "corrected-preprocess-v1 only supports subtract_tail_base "
            "translation."
        )
    if x.shape != y.shape or x.shape != local_angles.shape:
        raise ValueError("x, y, and local-angle arrays must have identical shapes.")
    if x.shape[1] != config.point_count:
        raise ValueError(
            f"Expected {config.point_count} tail points, received {x.shape[1]}."
        )

    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    elapsed_time_ms = np.asarray(elapsed_time_ms, dtype=np.float64)
    absolute_time_ms = np.asarray(absolute_time_ms, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    local_angles = np.asarray(local_angles, dtype=np.float64)

    _validate_strictly_increasing_ids(
        frame_ids,
        previous_frame_id=(
            int(previous["frame_id"]) if previous is not None else None
        ),
        label="Tracking",
    )
    if previous is None:
        if np.any(np.diff(elapsed_time_ms) <= 0):
            raise ValueError(
                "ElapsedTime must be strictly increasing "
                "(nonmonotonic_timestamp_policy=reject)."
            )
    else:
        previous_elapsed = float(previous["elapsed_time_ms"])
        if elapsed_time_ms[0] <= previous_elapsed or np.any(
            np.diff(elapsed_time_ms) <= 0
        ):
            raise ValueError(
                "ElapsedTime must be strictly increasing "
                "(nonmonotonic_timestamp_policy=reject)."
            )

    base_x = x[:, 0].copy()
    base_y = y[:, 0].copy()
    body_x = x - base_x[:, None]
    body_y = y - base_y[:, None]

    point_valid = np.isfinite(x) & np.isfinite(y)
    valid_point_count = np.count_nonzero(point_valid, axis=1).astype(np.int64)
    valid_point_fraction = valid_point_count.astype(np.float64) / float(
        config.point_count
    )
    timestamp_valid = np.isfinite(elapsed_time_ms) & np.isfinite(absolute_time_ms)
    frame_valid = timestamp_valid & (
        valid_point_fraction >= config.minimum_valid_point_fraction
    )

    if previous is not None:
        frame_with_previous = np.concatenate(
            [np.array([previous["frame_id"]], dtype=np.int64), frame_ids]
        )
        time_with_previous = np.concatenate(
            [
                np.array([previous["elapsed_time_ms"]], dtype=np.float64),
                elapsed_time_ms,
            ]
        )
        previous_ok = np.empty(len(frame_ids), dtype=bool)
        previous_ok[0] = bool(previous["timestamp_valid"]) and bool(
            previous["frame_valid"]
        )
        if len(frame_ids) > 1:
            previous_ok[1:] = timestamp_valid[:-1] & frame_valid[:-1]
        frame_steps = np.diff(frame_with_previous)
        delta_time = np.diff(time_with_previous)
        frame_step_for_rows = frame_steps.astype(np.int64)
        delta_time_for_rows = delta_time
    else:
        frame_steps = np.diff(frame_ids)
        delta_time = np.diff(elapsed_time_ms)
        frame_step_for_rows = np.concatenate(
            [np.array([0], dtype=np.int64), frame_steps.astype(np.int64)]
        )
        delta_time_for_rows = np.concatenate([np.array([np.nan]), delta_time])
        previous_ok = np.concatenate(
            [np.array([False]), timestamp_valid[:-1] & frame_valid[:-1]]
        )

    frame_id_contiguous = frame_step_for_rows == 1
    interval_ok = (
        np.isfinite(delta_time_for_rows)
        & (delta_time_for_rows > 0)
        & (delta_time_for_rows <= config.maximum_valid_derivative_interval_ms)
    )
    long_gap_invalid = (frame_step_for_rows > 1) | (
        np.isfinite(delta_time_for_rows)
        & (delta_time_for_rows > config.maximum_valid_derivative_interval_ms)
    )
    # First row without previous is not a long-gap invalidation event.
    if previous is None:
        long_gap_invalid = long_gap_invalid.copy()
        long_gap_invalid[0] = False

    derivative_valid = (
        frame_id_contiguous
        & interval_ok
        & timestamp_valid
        & previous_ok
        & frame_valid
    )

    payload: dict[str, Any] = {
        "FrameID": frame_ids,
        "ElapsedTime": elapsed_time_ms,
        "AbsoluteTime": absolute_time_ms,
        "FrameStep": frame_step_for_rows,
        "DeltaTimeMs": delta_time_for_rows,
        "frame_id_contiguous": frame_id_contiguous,
        "timestamp_valid": timestamp_valid,
        "frame_valid": frame_valid,
        "derivative_valid": derivative_valid,
        "long_gap_invalid": long_gap_invalid,
        "valid_point_count": valid_point_count,
        "valid_point_fraction": valid_point_fraction,
        "base_x": base_x,
        "base_y": base_y,
        "coordinate_source": np.full(len(frame_ids), config.coordinate_source),
        "body_translation_applied": np.full(len(frame_ids), True),
        "body_rotation_applied": np.full(len(frame_ids), False),
    }
    for index in range(config.point_count):
        payload[f"x{index}"] = body_x[:, index]
        payload[f"y{index}"] = body_y[:, index]
        payload[f"angle{index}"] = local_angles[:, index]
        payload[f"point{index}_valid"] = point_valid[:, index]

    result = pd.DataFrame(payload)
    state: dict[str, Any] = {
        "frame_id": int(frame_ids[-1]),
        "elapsed_time_ms": float(elapsed_time_ms[-1]),
        "timestamp_valid": bool(timestamp_valid[-1]),
        "frame_valid": bool(frame_valid[-1]),
    }
    return result, state


def build_corrected_preprocessing(
    project_dir: Path,
    recording_id: str,
    *,
    config: CorrectedPreprocessConfig | None = None,
    batch_size: int = 250_000,
    overwrite: bool = False,
) -> CorrectedPreprocessResult:
    """Build the corrected measured-time frame artifact for one recording."""
    config = config or CorrectedPreprocessConfig()
    if config != CorrectedPreprocessConfig():
        raise ValueError(
            "corrected-preprocess-v1 uses a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.drop_initial_camera_rows != 0:
        raise ValueError(
            "corrected-preprocess-v1 does not discard initial camera rows."
        )
    if config.drop_trailing_tracking_summary_row:
        raise ValueError(
            "corrected-preprocess-v1 does not drop a trailing tracking row; "
            "intake Parquet is the source of truth."
        )

    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    camera_path = source_dir / "camera.parquet"
    tracking_path = source_dir / "tracking.parquet"
    protocol_path = source_dir / "stimulus_events.parquet"
    for path in (camera_path, tracking_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing intake artifact: {path}")

    recording_name, input_artifacts, _input_state = _load_and_verify_source_manifest(
        project_dir,
        recording_id,
    )

    frames_path = source_dir / ARTIFACT_NAME
    summary_path = (
        project_dir / "Quality checks" / recording_id / SUMMARY_NAME
    )
    marker_path = project_dir / "Metadata" / f"{recording_id}_{MARKER_SUFFIX}"
    existing = [
        path for path in (frames_path, summary_path, marker_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Corrected preprocessing outputs already exist: {existing}"
        )

    camera = pq.read_table(
        camera_path,
        columns=["FrameID", "ElapsedTime", "AbsoluteTime"],
    ).to_pandas()
    camera_ids = camera["FrameID"].to_numpy(dtype=np.int64)
    camera_elapsed = camera["ElapsedTime"].to_numpy(dtype=np.float64)
    camera_absolute = camera["AbsoluteTime"].to_numpy(dtype=np.float64)
    _validate_strictly_increasing_ids(
        camera_ids,
        previous_frame_id=None,
        label="Camera",
    )
    if np.any(np.diff(camera_elapsed) <= 0):
        raise ValueError(
            "Camera ElapsedTime must be strictly increasing "
            "(nonmonotonic_timestamp_policy=reject)."
        )

    protocol = pq.read_table(protocol_path).to_pandas()
    protocol_timing = summarize_protocol_timing(
        protocol,
        camera_ids,
        camera_absolute,
    )

    tracking_file = pq.ParquetFile(tracking_path)
    point_count = config.point_count
    columns = _tracking_columns(point_count)
    state: dict[str, Any] | None = None
    previous_raw_tracking_frame_id: int | None = None
    row_count = 0
    derivative_valid_count = 0
    long_gap_count = 0
    frame_valid_count = 0
    matched_frame_count = 0
    tracking_frame_count = 0
    delta_time_samples: list[float] = []

    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-corrected-v1-",
    ) as staging_root:
        staged_frames = staging_root / ARTIFACT_NAME
        staged_summary = staging_root / SUMMARY_NAME
        staged_marker = staging_root / MARKER_SUFFIX
        writer: pq.ParquetWriter | None = None
        try:
            for batch in tracking_file.iter_batches(
                batch_size=batch_size,
                columns=columns,
            ):
                tracking = batch.to_pandas()
                frame_ids = tracking["FrameID"].to_numpy(dtype=np.int64)
                tracking_frame_count += int(len(frame_ids))
                previous_raw_tracking_frame_id = _validate_strictly_increasing_ids(
                    frame_ids,
                    previous_frame_id=previous_raw_tracking_frame_id,
                    label="Tracking",
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
                matched_frame_count += int(len(frame_ids))

                frames, state = calculate_corrected_frames(
                    frame_ids,
                    elapsed,
                    absolute,
                    x,
                    y,
                    angles,
                    config=config,
                    previous=state,
                )
                row_count += int(len(frames))
                derivative_valid_count += int(frames["derivative_valid"].sum())
                long_gap_count += int(frames["long_gap_invalid"].sum())
                frame_valid_count += int(frames["frame_valid"].sum())
                finite_deltas = frames["DeltaTimeMs"].to_numpy(dtype=np.float64)
                finite_deltas = finite_deltas[np.isfinite(finite_deltas)]
                if finite_deltas.size:
                    # Reservoir-style downsample for summary percentiles.
                    if len(delta_time_samples) < 200_000:
                        delta_time_samples.extend(finite_deltas.tolist())
                    else:
                        delta_time_samples.extend(
                            finite_deltas[:: max(1, finite_deltas.size // 1000)].tolist()
                        )

                if writer is None:
                    schema = pa.Table.from_pandas(
                        frames,
                        preserve_index=False,
                    ).schema.with_metadata(
                        {
                            b"recipe": RECIPE_ID.encode("ascii"),
                            b"scientific_status": SCIENTIFIC_STATUS.encode("ascii"),
                            b"recording_id": recording_id.encode("utf-8"),
                            b"source_tracking_sha256": input_artifacts["tracking"][
                                "sha256"
                            ].encode("ascii"),
                            b"source_camera_sha256": input_artifacts["camera"][
                                "sha256"
                            ].encode("ascii"),
                        }
                    )
                    writer = pq.ParquetWriter(
                        staged_frames,
                        schema,
                        compression="zstd",
                        write_statistics=True,
                    )
                table = pa.Table.from_pandas(
                    frames,
                    schema=writer.schema,
                    preserve_index=False,
                    safe=True,
                )
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

        if row_count == 0:
            raise ValueError(
                "No overlapping camera/tracking frames were available for "
                "corrected preprocessing."
            )

        delta_array = np.asarray(delta_time_samples, dtype=np.float64)
        if delta_array.size:
            median_dt = float(np.median(delta_array))
            mad = float(np.median(np.abs(delta_array - median_dt)))
            jitter_exceed_count = int(
                np.count_nonzero(
                    np.abs(delta_array - median_dt)
                    > config.timestamp_jitter_tolerance_ms
                )
            )
        else:
            median_dt = float("nan")
            mad = float("nan")
            jitter_exceed_count = 0

        summary: dict[str, Any] = {
            "recipe": RECIPE_ID,
            "scientific_status": SCIENTIFIC_STATUS,
            "paper_approved": False,
            "recording_id": recording_id,
            "recording_name": recording_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "row_count": row_count,
            "camera_row_count": int(len(camera_ids)),
            "tracking_row_count": tracking_frame_count,
            "matched_frame_count": matched_frame_count,
            "frame_valid_count": frame_valid_count,
            "derivative_valid_count": derivative_valid_count,
            "long_gap_invalid_count": long_gap_count,
            "timing": {
                "median_delta_time_ms": median_dt,
                "delta_time_mad_ms": mad,
                "jitter_tolerance_ms": config.timestamp_jitter_tolerance_ms,
                "sampled_delta_count": int(delta_array.size),
                "sampled_jitter_exceedances": jitter_exceed_count,
            },
            "protocol_timing": protocol_timing,
            "policies": {
                "interpolation": "disabled",
                "temporal_filter": "disabled",
                "spatial_filter": "disabled",
                "body_translation": config.body_translation_correction,
                "body_rotation": config.body_rotation_correction,
                "legacy_startup_discard": False,
                "legacy_trailing_tracking_discard": False,
                "legacy_uniform_700hz_grid": False,
            },
            "known_limitations": [
                "Interpolation is disabled; long gaps invalidate derivatives only.",
                "No temporal or spatial filtering is applied in this recipe.",
                "Body-axis rotation is not corrected (field absent; Gate T0).",
                "Gate P has not approved this contract for paper results.",
                "Candidate metrics still read intake Parquet until an adapter "
                "switches them onto this artifact.",
            ],
            "input_artifacts": {
                "camera": input_artifacts["camera"],
                "tracking": input_artifacts["tracking"],
                "protocol": input_artifacts["protocol"],
            },
            "artifact": {
                "path": str(frames_path),
                "sha256": _sha256_file(staged_frames),
            },
        }
        _write_json_atomic(staged_summary, summary)
        _write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "recording_id": recording_id,
                "frames_sha256": summary["artifact"]["sha256"],
                "summary_sha256": _sha256_file(staged_summary),
            },
        )
        _publish_transaction(
            (
                (staged_frames, frames_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return CorrectedPreprocessResult(
        recording_id=recording_id,
        frames_path=frames_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_count=row_count,
        derivative_valid_count=derivative_valid_count,
    )
