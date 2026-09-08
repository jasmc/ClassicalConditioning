"""Candidate activity metrics sourced from corrected-preprocess-v1 frames.

Distinct from ``tail-candidate-development-v1``, which still reads intake
camera/tracking Parquet directly. This recipe requires the corrected
measured-time artifact and intersects metric derivatives with its validity
masks (including long-interval invalidation).
"""

from __future__ import annotations

import json
from dataclasses import asdict
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
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
)
from classical_conditioning.preprocessing.candidates_v1 import (
    CANDIDATE_COLUMNS,
    CandidateMetricConfig,
    CandidateMetricResult,
    _extract_arrays,
    _geometry_agreement,
    _tracking_columns,
    _validate_frame_order,
    calculate_candidate_metrics,
)
from classical_conditioning.preprocessing.corrected_v1 import (
    ARTIFACT_NAME as CORRECTED_ARTIFACT_NAME,
    MARKER_SUFFIX as CORRECTED_MARKER_SUFFIX,
    RECIPE_ID as CORRECTED_RECIPE_ID,
    SUMMARY_NAME as CORRECTED_SUMMARY_NAME,
)

RECIPE_ID = "tail-candidate-corrected-v1"
SCIENTIFIC_STATUS = "candidate_development"
METRICS_NAME = "frame_activity_candidates-corrected-v1.parquet"
SUMMARY_NAME = "candidate-corrected-v1_activity_summary.json"
MARKER_SUFFIX = "candidate-corrected-v1_complete.json"


def verify_corrected_preprocess_source(
    project_dir: Path,
    recording_id: str,
) -> dict[str, Any]:
    """Verify corrected-preprocess-v1 lineage for one recording."""
    source_dir = project_dir / "Processed data" / recording_id
    frames_path = source_dir / CORRECTED_ARTIFACT_NAME
    summary_path = (
        project_dir / "Quality checks" / recording_id / CORRECTED_SUMMARY_NAME
    )
    marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{CORRECTED_MARKER_SUFFIX}"
    )
    missing = [
        path for path in (frames_path, summary_path, marker_path) if not path.is_file()
    ]
    if missing:
        raise ArtifactNotFoundError(
            f"Missing corrected-preprocess-v1 artifacts: {missing}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError(
            f"Corrected preprocess metadata is invalid JSON: {error}"
        ) from error
    if not isinstance(marker, dict) or not isinstance(summary, dict):
        raise ArtifactIntegrityError(
            "Corrected preprocess metadata must be JSON objects."
        )

    frames_hash = _sha256_file(frames_path)
    if (
        marker.get("status") != "complete"
        or marker.get("recipe") != CORRECTED_RECIPE_ID
        or marker.get("recording_id") != recording_id
        or summary.get("recipe") != CORRECTED_RECIPE_ID
        or summary.get("recording_id") != recording_id
        or marker.get("frames_sha256") != frames_hash
        or marker.get("summary_sha256") != _sha256_file(summary_path)
        or summary.get("artifact", {}).get("sha256") != frames_hash
    ):
        raise ArtifactIntegrityError(
            f"Completed {CORRECTED_RECIPE_ID} lineage is invalid for {recording_id}."
        )
    return {
        "frames_path": frames_path,
        "summary_path": summary_path,
        "marker_path": marker_path,
        "summary": summary,
        "frames_sha256": frames_hash,
    }


def apply_corrected_validity_mask(
    metrics: pd.DataFrame,
    *,
    corrected_derivative_valid: np.ndarray,
    corrected_frame_step: np.ndarray,
    corrected_delta_time_ms: np.ndarray,
) -> pd.DataFrame:
    """Intersect candidate derivatives with corrected preprocess masks."""
    if len(metrics) != len(corrected_derivative_valid):
        raise ValueError("Corrected validity length does not match metrics rows.")
    out = metrics.copy()
    combined = (
        out["valid_derivative"].to_numpy(dtype=bool)
        & np.asarray(corrected_derivative_valid, dtype=bool)
    )
    out["valid_derivative"] = combined
    out["FrameStep"] = np.asarray(corrected_frame_step, dtype=np.int64)
    out["DeltaTimeMs"] = np.asarray(corrected_delta_time_ms, dtype=np.float64)
    for column in CANDIDATE_COLUMNS:
        values = out[column].to_numpy(dtype=np.float64, copy=True)
        values[~combined] = np.nan
        out[column] = values
    return out


def build_candidate_activity_metrics_from_corrected(
    project_dir: Path,
    recording_id: str,
    *,
    config: CandidateMetricConfig | None = None,
    batch_size: int = 250_000,
    overwrite: bool = False,
) -> CandidateMetricResult:
    """Build candidate metrics from corrected-preprocess-v1 frames."""
    config = config or CandidateMetricConfig()
    if config != CandidateMetricConfig():
        raise ValueError(
            f"{RECIPE_ID} uses a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    recording_name, input_artifacts, _input_state = _load_and_verify_source_manifest(
        project_dir,
        recording_id,
    )
    corrected = verify_corrected_preprocess_source(project_dir, recording_id)
    frames_path = Path(corrected["frames_path"])

    metrics_path = source_dir / METRICS_NAME
    summary_path = (
        project_dir / "Quality checks" / recording_id / SUMMARY_NAME
    )
    marker_path = project_dir / "Metadata" / f"{recording_id}_{MARKER_SUFFIX}"
    existing = [
        path for path in (metrics_path, summary_path, marker_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Corrected-source candidate outputs already exist: {existing}"
        )

    point_count = config.point_count
    columns = [
        "FrameID",
        "ElapsedTime",
        "AbsoluteTime",
        "FrameStep",
        "DeltaTimeMs",
        "derivative_valid",
        *_tracking_columns(point_count)[1:],
    ]
    frames_file = pq.ParquetFile(frames_path)
    state: dict[str, np.ndarray | float | int] | None = None
    previous_frame_id: int | None = None
    row_count = 0
    valid_derivative_count = 0
    invalid_gap_count = 0
    corrected_mask_extra_invalid = 0
    metric_minima = {column: np.inf for column in CANDIDATE_COLUMNS}
    metric_maxima = {column: -np.inf for column in CANDIDATE_COLUMNS}
    geometry_error_sum = 0.0
    geometry_error_count = 0
    geometry_total_count = 0
    terminal_angle_nonzero_count = 0

    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-candidate-corrected-v1-",
    ) as staging_root:
        staged_metrics = staging_root / METRICS_NAME
        staged_summary = staging_root / SUMMARY_NAME
        staged_marker = staging_root / MARKER_SUFFIX
        writer: pq.ParquetWriter | None = None
        try:
            for batch in frames_file.iter_batches(
                batch_size=batch_size,
                columns=columns,
            ):
                frames = batch.to_pandas()
                frame_ids = frames["FrameID"].to_numpy(dtype=np.int64)
                previous_frame_id = _validate_frame_order(
                    frame_ids,
                    previous_frame_id,
                )
                elapsed = frames["ElapsedTime"].to_numpy(dtype=np.float64)
                absolute = frames["AbsoluteTime"].to_numpy(dtype=np.float64)
                corrected_valid = frames["derivative_valid"].to_numpy(dtype=bool)
                corrected_step = frames["FrameStep"].to_numpy(dtype=np.int64)
                corrected_delta = frames["DeltaTimeMs"].to_numpy(dtype=np.float64)
                x, y, angles = _extract_arrays(frames, point_count)

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
                before_valid = metrics["valid_derivative"].to_numpy(dtype=bool)
                metrics = apply_corrected_validity_mask(
                    metrics,
                    corrected_derivative_valid=corrected_valid,
                    corrected_frame_step=corrected_step,
                    corrected_delta_time_ms=corrected_delta,
                )
                after_valid = metrics["valid_derivative"].to_numpy(dtype=bool)
                corrected_mask_extra_invalid += int(
                    np.count_nonzero(before_valid & ~after_valid)
                )
                metrics.insert(2, "AbsoluteTime", absolute)

                if writer is None:
                    schema = pa.Table.from_pandas(
                        metrics,
                        preserve_index=False,
                    ).schema.with_metadata(
                        {
                            b"recipe": RECIPE_ID.encode("ascii"),
                            b"scientific_status": SCIENTIFIC_STATUS.encode("ascii"),
                            b"recording_id": recording_id.encode("utf-8"),
                            b"source_corrected_sha256": str(
                                corrected["frames_sha256"]
                            ).encode("ascii"),
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
                valid_derivative_count += int(after_valid.sum())
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
            if writer is not None:
                writer.close()

        if writer is None or row_count == 0:
            raise ValueError("No corrected-preprocess frames were available.")
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
                "Corrected-frame angle semantics do not meet the frozen "
                f"measured-geometry criteria: mean_error={mean_geometry_error}, "
                f"coverage={geometry_coverage}."
            )

        if _sha256_file(frames_path) != corrected["frames_sha256"]:
            raise RuntimeError(
                "Corrected preprocess artifact hash changed during metric build."
            )

        metric_hash = _sha256_file(staged_metrics)
        summary: dict[str, Any] = {
            "recipe": RECIPE_ID,
            "scientific_status": SCIENTIFIC_STATUS,
            "paper_approved": False,
            "recording_id": recording_id,
            "recording_name": recording_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "source_recipe": CORRECTED_RECIPE_ID,
            "row_count": row_count,
            "valid_derivative_count": valid_derivative_count,
            "invalid_derivative_count": row_count - valid_derivative_count,
            "invalid_gap_count": invalid_gap_count,
            "corrected_mask_extra_invalid_count": corrected_mask_extra_invalid,
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
                "mean_absolute_local_bend_geometry_error_rad": mean_geometry_error,
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
                    "Metrics are calculated on corrected-preprocess-v1 "
                    "body-translated measured XY; derivative validity is the "
                    "intersection of candidate adjacency rules and corrected "
                    "gap/long-interval masks."
                ),
            },
            "known_limitations": [
                "Exploratory; not paper-approved.",
                "Depends on corrected-preprocess-v1 (interpolation/filtering still disabled).",
                "Does not replace tail-candidate-development-v1 intake-sourced outputs.",
                "Paired downstream recipes use the corrected-* route IDs.",
            ],
            "input_artifacts": {
                "corrected_preprocess": {
                    "path": str(frames_path),
                    "sha256": corrected["frames_sha256"],
                    "recipe": CORRECTED_RECIPE_ID,
                },
                "camera": input_artifacts["camera"],
                "tracking": input_artifacts["tracking"],
                "protocol": input_artifacts["protocol"],
            },
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
                "recipe": RECIPE_ID,
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
