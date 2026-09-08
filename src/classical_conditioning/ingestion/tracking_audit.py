"""Header and sample audit for raw tail-tracking files.

This audit inventories available fields without assuming every non-frame column
is an angle. Gate T0 defaults below come from the legacy reader comments and
pilot geometry checks; they are not a physical calibration certificate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from classical_conditioning.artifacts import write_json_atomic
from classical_conditioning.exceptions import ConfigurationError

# Evidence: data_io.read_tail_tracking_data and legacy my_functions convert
# raw angleN columns with ``* (180/pi)`` under the comment
# "Convert radian to degree". Candidate metrics keep radians and name units
# ``*_rad_*``. Measured xN/yN are present; package curvature uses ``rad_per_px``
# naming. Absolute micrometre calibration is not required for relative metrics.
GATE_T0_LEGACY_EVIDENCE = {
    "raw_angle_units": "radian",
    "legacy_analysis_angle_units": "degree",
    "angle_semantics": (
        "local_intersegment_bend_for_angle1_to_angle14; "
        "angle15_terminal_placeholder; "
        "angle0_retained_but_not_used_as_a_bend"
    ),
    "coordinate_frame": "tracking_image_pixels",
    "spatial_units": "pixel",
    "absolute_calibration_required": False,
    "evidence_sources": [
        "data_io.read_tail_tracking_data: Convert radian to degree",
        "legacy/modules/my_functions.py: Vectorized conversion from radian to degree",
        "preprocessing.legacy_v1.prepare_legacy_tracking",
        "preprocessing.candidates_v1 geometry agreement and rad_per_px naming",
        "Plans/DECISIONS.md Gate T0",
    ],
}

_ANGLE = re.compile(r"^angle(\d+)$", re.IGNORECASE)
_X = re.compile(r"^x(\d+)$", re.IGNORECASE)
_Y = re.compile(r"^y(\d+)$", re.IGNORECASE)
_CONFIDENCE = re.compile(
    r"^(confidence|conf|score|likelihood|prob)(\d+)?$",
    re.IGNORECASE,
)
_BODY = re.compile(
    r"^(body|base|axis|heading|orientation).*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TrackingColumnClassification:
    frame_id_columns: tuple[str, ...]
    angle_columns: tuple[str, ...]
    x_columns: tuple[str, ...]
    y_columns: tuple[str, ...]
    confidence_columns: tuple[str, ...]
    body_reference_columns: tuple[str, ...]
    unrecognized_columns: tuple[str, ...]
    angle_indices: tuple[int, ...]
    x_indices: tuple[int, ...]
    y_indices: tuple[int, ...]

    @property
    def has_xy(self) -> bool:
        return bool(self.x_indices) and self.x_indices == self.y_indices

    @property
    def point_count_from_angles(self) -> int | None:
        if not self.angle_indices:
            return None
        return max(self.angle_indices) + 1


def classify_tracking_columns(
    columns: list[str] | tuple[str, ...],
) -> TrackingColumnClassification:
    frame_id: list[str] = []
    angles: list[str] = []
    xs: list[str] = []
    ys: list[str] = []
    confidence: list[str] = []
    body: list[str] = []
    unrecognized: list[str] = []
    angle_indices: list[int] = []
    x_indices: list[int] = []
    y_indices: list[int] = []

    for column in columns:
        if column == "FrameID" or column.lower() in {"frameid", "frame_id", "id"}:
            frame_id.append(column)
            continue
        if match := _ANGLE.fullmatch(column):
            angles.append(column)
            angle_indices.append(int(match.group(1)))
            continue
        if match := _X.fullmatch(column):
            xs.append(column)
            x_indices.append(int(match.group(1)))
            continue
        if match := _Y.fullmatch(column):
            ys.append(column)
            y_indices.append(int(match.group(1)))
            continue
        if _CONFIDENCE.fullmatch(column):
            confidence.append(column)
            continue
        if _BODY.fullmatch(column):
            body.append(column)
            continue
        unrecognized.append(column)

    return TrackingColumnClassification(
        frame_id_columns=tuple(frame_id),
        angle_columns=tuple(angles),
        x_columns=tuple(xs),
        y_columns=tuple(ys),
        confidence_columns=tuple(confidence),
        body_reference_columns=tuple(body),
        unrecognized_columns=tuple(unrecognized),
        angle_indices=tuple(sorted(angle_indices)),
        x_indices=tuple(sorted(x_indices)),
        y_indices=tuple(sorted(y_indices)),
    )


def _numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce")
    finite = values[values.notna() & values.map(math.isfinite)]
    if finite.empty:
        return {
            "non_null_count": 0,
            "finite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "non_null_count": int(values.notna().sum()),
        "finite_count": int(len(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def audit_tracking_file(
    path: Path,
    *,
    sample_rows: int = 2_000,
) -> dict[str, Any]:
    """Inventory tracking fields from the header and a bounded sample."""
    if sample_rows < 1:
        raise ConfigurationError("Tracking audit sample_rows must be at least 1.")
    tracking_path = path.resolve()
    if not tracking_path.is_file():
        raise FileNotFoundError(f"Tracking file does not exist: {tracking_path}")

    sample = pd.read_csv(
        tracking_path,
        sep=r"\s+",
        nrows=sample_rows,
        engine="c",
    )
    columns = [str(column) for column in sample.columns]
    classification = classify_tracking_columns(columns)

    angle_contiguous = (
        list(classification.angle_indices)
        == list(range(classification.point_count_from_angles or 0))
        if classification.angle_indices
        else False
    )
    xy_match_angles = (
        classification.has_xy
        and classification.x_indices == classification.angle_indices
    )

    column_summaries = {
        column: _numeric_summary(sample[column])
        for column in columns
        if column in sample.columns
    }

    return {
        "artifact_kind": "tracking-field-audit-v1",
        "path": str(tracking_path),
        "sample_rows_requested": sample_rows,
        "sample_rows_read": int(len(sample)),
        "columns": columns,
        "classification": {
            "frame_id_columns": list(classification.frame_id_columns),
            "angle_columns": list(classification.angle_columns),
            "x_columns": list(classification.x_columns),
            "y_columns": list(classification.y_columns),
            "confidence_columns": list(classification.confidence_columns),
            "body_reference_columns": list(classification.body_reference_columns),
            "unrecognized_columns": list(classification.unrecognized_columns),
            "angle_indices": list(classification.angle_indices),
            "x_indices": list(classification.x_indices),
            "y_indices": list(classification.y_indices),
            "has_xy": classification.has_xy,
            "point_count_from_angles": classification.point_count_from_angles,
            "angles_contiguous_from_zero": angle_contiguous,
            "xy_indices_match_angles": xy_match_angles,
        },
        "gate_t0_observations": {
            "raw_xy_present": classification.has_xy,
            "confidence_fields_present": bool(classification.confidence_columns),
            "body_reference_fields_present": bool(
                classification.body_reference_columns
            ),
            "unrecognized_fields_present": bool(
                classification.unrecognized_columns
            ),
            "raw_angle_units": GATE_T0_LEGACY_EVIDENCE["raw_angle_units"],
            "legacy_analysis_angle_units": GATE_T0_LEGACY_EVIDENCE[
                "legacy_analysis_angle_units"
            ],
            "angle_semantics": GATE_T0_LEGACY_EVIDENCE["angle_semantics"],
            "coordinate_frame": GATE_T0_LEGACY_EVIDENCE["coordinate_frame"],
            "spatial_units": GATE_T0_LEGACY_EVIDENCE["spatial_units"],
            "absolute_calibration_required": GATE_T0_LEGACY_EVIDENCE[
                "absolute_calibration_required"
            ],
            "synchronized_video": "optional_deferred",
            "evidence_sources": list(GATE_T0_LEGACY_EVIDENCE["evidence_sources"]),
            "notes": [
                "Raw angleN values are radians; legacy vigor converts them to degrees.",
                "Measured xN/yN are treated as tracking-image pixels.",
                "Absolute micrometre scale is not required for relative activity metrics.",
                "Synchronized video remains optional for blinded clip validation.",
            ],
        },
        "column_summaries": column_summaries,
    }


def write_tracking_audit(
    path: Path,
    output: Path,
    *,
    sample_rows: int = 2_000,
    overwrite: bool = False,
) -> Path:
    output_path = output.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Tracking audit already exists; pass overwrite to replace: {output_path}"
        )
    write_json_atomic(output_path, audit_tracking_file(path, sample_rows=sample_rows))
    return output_path
