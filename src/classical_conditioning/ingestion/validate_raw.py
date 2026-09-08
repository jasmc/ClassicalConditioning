"""Validate one local acquisition triplet without writing Parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from classical_conditioning.artifacts import write_json_atomic
from classical_conditioning.ingestion.frame_sequence import validate_frame_sequence
from classical_conditioning.ingestion.readers import (
    read_camera,
    read_protocol,
    read_tracking,
)
from classical_conditioning.ingestion.tracking_audit import classify_tracking_columns


@dataclass(frozen=True)
class RawValidationResult:
    recording_id: str
    recording_name: str
    summary: dict[str, Any]
    report_path: Path | None


def validate_raw_triplet(
    input_dir: Path,
    *,
    output: Path | None = None,
    overwrite: bool = False,
) -> RawValidationResult:
    """Read and validate one camera/tracking/protocol triplet locally."""
    from classical_conditioning.intake import discover_recording

    sources = discover_recording(input_dir)
    camera = read_camera(sources.camera)
    tracking = read_tracking(sources.tracking, mode="full")
    protocol = read_protocol(sources.protocol)
    frame_report = validate_frame_sequence(camera.frame)
    classification = classify_tracking_columns(list(tracking.schema.columns))

    protocol_outside = 0
    if not camera.frame.empty and not protocol.frame.empty:
        start = int(camera.frame["AbsoluteTime"].min())
        stop = int(camera.frame["AbsoluteTime"].max())
        protocol_outside = int(
            (
                (protocol.frame["End"] < start) | (protocol.frame["Beg"] > stop)
            ).sum()
        )

    summary: dict[str, Any] = {
        "artifact_kind": "raw-acquisition-validation-v1",
        "recording_id": sources.recording_id,
        "recording_name": sources.recording_name,
        "camera": {
            "path": str(camera.source_path),
            "separator": camera.separator,
            "decimal": camera.decimal,
            "row_count": int(len(camera.frame)),
            "frame_sequence": frame_report.to_dict(),
        },
        "tracking": {
            "path": str(tracking.source_path),
            "mode": tracking.mode,
            "row_count": int(len(tracking.frame)),
            "point_count": tracking.schema.point_count,
            "dropped_trailing_summary_row": tracking.dropped_trailing_summary_row,
            "classification": {
                "has_xy": classification.has_xy,
                "angle_indices": list(classification.angle_indices),
                "confidence_columns": list(classification.confidence_columns),
                "unrecognized_columns": list(classification.unrecognized_columns),
            },
        },
        "protocol": {
            "path": str(protocol.source_path),
            "row_count": int(len(protocol.frame)),
            "invalid_duration_count": protocol.invalid_duration_count,
            "event_type_counts": protocol.event_type_counts,
            "events_outside_camera_absolute_time": protocol_outside,
        },
        "status": (
            "PASS"
            if (
                frame_report.duplicate_frame_id_count == 0
                and frame_report.reverse_event_count == 0
                and frame_report.non_finite_elapsed_count == 0
                and protocol.invalid_duration_count == 0
            )
            else "WARN"
        ),
    }

    report_path: Path | None = None
    if output is not None:
        report_path = output.resolve()
        if report_path.exists() and not overwrite:
            raise FileExistsError(
                f"Raw validation report already exists; pass overwrite to replace: "
                f"{report_path}"
            )
        write_json_atomic(report_path, summary)

    return RawValidationResult(
        recording_id=sources.recording_id,
        recording_name=sources.recording_name,
        summary=summary,
        report_path=report_path,
    )
