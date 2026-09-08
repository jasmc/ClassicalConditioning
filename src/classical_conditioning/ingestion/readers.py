"""Explicit local readers for camera, tracking, and protocol TXT files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from classical_conditioning.exceptions import SchemaValidationError
from classical_conditioning.ingestion.schemas import (
    CAMERA_COLUMNS,
    PROTOCOL_COLUMNS,
    TrackingSchema,
    normalize_camera_columns,
    validate_camera_columns,
    validate_protocol_columns,
    validate_tracking_columns,
)

TrackingMode = Literal["full", "legacy_angles"]


@dataclass(frozen=True)
class CameraReadResult:
    frame: pd.DataFrame
    source_path: Path
    separator: str
    decimal: str


@dataclass(frozen=True)
class TrackingReadResult:
    frame: pd.DataFrame
    source_path: Path
    schema: TrackingSchema
    mode: TrackingMode
    dropped_trailing_summary_row: bool
    angles_converted_to_degrees: bool


@dataclass(frozen=True)
class ProtocolReadResult:
    frame: pd.DataFrame
    source_path: Path
    invalid_duration_count: int
    event_type_counts: dict[str, int]


def _require_file(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Raw acquisition file does not exist: {resolved}")
    return resolved


def _read_whitespace_table(
    path: Path,
    *,
    decimal: str = ".",
) -> tuple[pd.DataFrame, str, str]:
    """Try space-separated then tab-separated acquisition text."""
    errors: list[str] = []
    for separator, label in ((" ", "space"), ("\t", "tab")):
        try:
            frame = pd.read_csv(
                path,
                sep=separator,
                header=0,
                decimal=decimal,
                engine="c",
            )
        except Exception as exc:  # noqa: BLE001 - collect parse attempts
            errors.append(f"{label}/{decimal}: {exc}")
            continue
        if frame.shape[1] == 1 and separator == " ":
            errors.append(f"{label}/{decimal}: collapsed to a single column")
            continue
        return frame, label, decimal
    raise SchemaValidationError(
        f"Could not parse whitespace table {path}: {'; '.join(errors)}"
    )


def read_camera(path: Path) -> CameraReadResult:
    """Read canonical camera timing without discarding leading rows."""
    source = _require_file(path)
    last_parse_error: Exception | None = None
    frame: pd.DataFrame | None = None
    separator = "space"
    used_decimal = "."
    for decimal in (".", ","):
        try:
            frame, separator, used_decimal = _read_whitespace_table(
                source,
                decimal=decimal,
            )
            break
        except SchemaValidationError as exc:
            last_parse_error = exc
            frame = None
    if frame is None:
        raise SchemaValidationError(
            f"Camera file could not be parsed: {source}: {last_parse_error}"
        )

    working = frame.copy()
    working.columns = normalize_camera_columns([str(c) for c in working.columns])
    validate_camera_columns(list(working.columns))
    working = working.loc[:, list(CAMERA_COLUMNS)].copy()
    try:
        working["FrameID"] = pd.to_numeric(working["FrameID"], errors="raise").astype(
            "int64"
        )
        working["ElapsedTime"] = pd.to_numeric(
            working["ElapsedTime"],
            errors="raise",
        ).astype("float64")
        working["AbsoluteTime"] = pd.to_numeric(
            working["AbsoluteTime"],
            errors="raise",
        ).astype("int64")
    except (ValueError, TypeError) as exc:
        raise SchemaValidationError(
            f"Camera numeric columns are invalid in {source}: {exc}"
        ) from exc
    if working.empty:
        raise SchemaValidationError("Camera table is empty.")
    if working["FrameID"].duplicated().any():
        raise SchemaValidationError("Camera FrameID values must be unique.")
    if not working["ElapsedTime"].map(np.isfinite).all():
        raise SchemaValidationError("Camera ElapsedTime must be finite.")
    return CameraReadResult(
        frame=working,
        source_path=source,
        separator=separator,
        decimal=used_decimal,
    )


def read_tracking(
    path: Path,
    *,
    mode: TrackingMode = "full",
    drop_trailing_summary_row: bool = True,
    convert_angles_to_degrees: bool = False,
    legacy_angle_point_count: int | None = None,
) -> TrackingReadResult:
    """Read tracking TXT in full-field or legacy angle-only mode."""
    source = _require_file(path)
    frame, _, _ = _read_whitespace_table(source, decimal=".")
    try:
        if frame.shape[1] == 1:
            frame, _, _ = _read_whitespace_table(source, decimal=",")
    except SchemaValidationError:
        pass

    columns = [str(column) for column in frame.columns]
    schema = validate_tracking_columns(columns)
    working = frame.copy()
    working.columns = columns

    dropped = False
    if drop_trailing_summary_row:
        if len(working) < 2:
            raise SchemaValidationError(
                "Tracking table needs at least two rows to drop the summary row."
            )
        working = working.iloc[:-1].copy()
        dropped = True

    # When the summary row is retained for legacy prepare(), FrameID may be
    # non-numeric on that final row. Coerce rather than failing the whole file.
    working["FrameID"] = pd.to_numeric(working["FrameID"], errors="coerce")
    if dropped:
        if working["FrameID"].isna().any():
            raise SchemaValidationError(
                "Tracking FrameID values must be numeric after dropping the summary row."
            )
        working["FrameID"] = working["FrameID"].astype("int64")
    else:
        if len(working) >= 2 and working["FrameID"].iloc[:-1].isna().any():
            raise SchemaValidationError(
                "Tracking FrameID values before the trailing summary row must be numeric."
            )
        working["FrameID"] = working["FrameID"].astype("float64")

    for column in schema.angle_columns + schema.x_columns + schema.y_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").astype(
            "float64"
        )

    angles_to_degrees = False
    if mode == "full":
        selected = working.loc[
            :,
            ["FrameID", *schema.x_columns, *schema.y_columns, *schema.angle_columns],
        ].copy()
    elif mode == "legacy_angles":
        point_count = legacy_angle_point_count or schema.point_count
        if point_count < 2 or point_count > schema.point_count:
            raise SchemaValidationError(
                f"legacy_angle_point_count={point_count} is outside available "
                f"point count {schema.point_count}."
            )
        angle_columns = tuple(f"angle{index}" for index in range(point_count))
        selected = working.loc[:, ["FrameID", *angle_columns]].copy()
        if convert_angles_to_degrees:
            selected.loc[:, list(angle_columns)] = (
                selected.loc[:, list(angle_columns)].to_numpy(dtype=np.float64)
                * (180.0 / np.pi)
            )
            angles_to_degrees = True
            selected = selected.rename(
                columns={
                    column: f"Angle of point {index} (deg)"
                    for index, column in enumerate(angle_columns)
                }
            )
    else:
        raise SchemaValidationError(f"Unsupported tracking mode: {mode!r}")

    if selected.empty:
        raise SchemaValidationError(
            "Tracking table is empty after summary-row handling."
        )

    return TrackingReadResult(
        frame=selected,
        source_path=source,
        schema=schema,
        mode=mode,
        dropped_trailing_summary_row=dropped,
        angles_converted_to_degrees=angles_to_degrees,
    )


def read_protocol(path: Path) -> ProtocolReadResult:
    """Read stimulus-control events with explicit duration checks."""
    source = _require_file(path)
    frame, _, _ = _read_whitespace_table(source, decimal=".")
    columns = [str(column) for column in frame.columns]
    validate_protocol_columns(columns)
    working = frame.loc[:, list(PROTOCOL_COLUMNS)].copy()
    working["Type"] = working["Type"].astype("string")
    working["Beg"] = pd.to_numeric(working["Beg"], errors="raise").astype("int64")
    working["End"] = pd.to_numeric(working["End"], errors="raise").astype("int64")
    if working.empty:
        raise SchemaValidationError("Protocol table is empty.")
    if working["Type"].isna().any():
        raise SchemaValidationError("Protocol event types cannot be missing.")

    invalid_duration = working["Beg"] >= working["End"]
    invalid_duration_count = int(invalid_duration.sum())
    ordered = working.sort_values(["Beg", "End", "Type"], kind="mergesort").reset_index(
        drop=True
    )
    counts = {
        str(key): int(value)
        for key, value in ordered["Type"].value_counts(sort=False).items()
    }
    return ProtocolReadResult(
        frame=ordered,
        source_path=source,
        invalid_duration_count=invalid_duration_count,
        event_type_counts=counts,
    )
