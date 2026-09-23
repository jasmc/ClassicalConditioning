"""Explicit local readers for camera, tracking, and protocol TXT files.

Review note: readers parse and type local raw inputs but do not write derived
data. Intake owns conversion to Parquet and provenance publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from classical_conditioning.exceptions import SchemaValidationError
# Schema constants and validators separate file syntax from table topology.
from classical_conditioning.ingestion.schemas import (
    CAMERA_COLUMNS,
    PROTOCOL_COLUMNS,
    TrackingSchema,
    normalize_camera_columns,
    validate_camera_columns,
    validate_protocol_columns,
    validate_tracking_columns,
)

# Typed reader results retain parsing decisions as evidence for later QC.
@dataclass(frozen=True)
class CameraReadResult:
    frame: pd.DataFrame
    source_path: Path
    separator: str
    decimal: str


# Tracking results additionally record topology and summary-row treatment.
@dataclass(frozen=True)
class TrackingReadResult:
    frame: pd.DataFrame
    source_path: Path
    schema: TrackingSchema
    mode: str
    dropped_trailing_summary_row: bool


# Protocol results retain duration warnings and an event-type inventory.
@dataclass(frozen=True)
class ProtocolReadResult:
    frame: pd.DataFrame
    source_path: Path
    invalid_duration_count: int
    event_type_counts: dict[str, int]


def _require_file(path: Path) -> Path:
    # Resolve first so errors and stored provenance refer to one absolute path.
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
    # Try the two supported delimiter conventions and preserve failed attempts.
    errors: list[str] = []
    for separator, label in ((" ", "space"), ("\t", "tab")):
        # Parsing with the C engine gives deterministic whitespace-table handling.
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
        # A space parse that collapses a tab file is not a valid success.
        if frame.shape[1] == 1 and separator == " ":
            errors.append(f"{label}/{decimal}: collapsed to a single column")
            continue
        return frame, label, decimal
    raise SchemaValidationError(
        f"Could not parse whitespace table {path}: {'; '.join(errors)}"
    )


def read_camera(path: Path) -> CameraReadResult:
    """Read canonical camera timing without discarding leading rows."""
    # Camera exports may use dot or comma decimals, so try both deliberately.
    source = _require_file(path)
    last_error: Exception | None = None
    # A parser can read comma decimals as strings without raising. Therefore a
    # decimal convention succeeds only after schema *and* numeric validation.
    for decimal in (".", ","):
        try:
            frame, separator, used_decimal = _read_whitespace_table(
                source,
                decimal=decimal,
            )
            working = frame.copy()
            working.columns = normalize_camera_columns(
                [str(column) for column in working.columns]
            )
            validate_camera_columns(list(working.columns))
            working = working.loc[:, list(CAMERA_COLUMNS)].copy()
            working["FrameID"] = pd.to_numeric(
                working["FrameID"], errors="raise"
            ).astype("int64")
            working["ElapsedTime"] = pd.to_numeric(
                working["ElapsedTime"], errors="raise"
            ).astype("float64")
            working["AbsoluteTime"] = pd.to_numeric(
                working["AbsoluteTime"], errors="raise"
            ).astype("int64")
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
        except (SchemaValidationError, ValueError, TypeError) as exc:
            last_error = exc
    raise SchemaValidationError(
        f"Camera file could not be parsed and typed: {source}: {last_error}"
    ) from last_error


def read_tracking(
    path: Path,
    *,
    mode: str = "full",
    drop_trailing_summary_row: bool = True,
) -> TrackingReadResult:
    """Read full-field tracking TXT for the supported candidate route."""
    # Tracking normally uses decimal dots. Unlike camera data, coordinate values
    # can safely become NaN later, so inspect raw cells before choosing commas.
    source = _require_file(path)
    frame, _, _ = _read_whitespace_table(source, decimal=".")
    decimal_comma_pattern = r"[+-]?(?:\d+,\d*|\d*,\d+)(?:[eE][+-]?\d+)?"
    has_decimal_comma = any(
        column.str.fullmatch(decimal_comma_pattern).fillna(False).any()
        for _, column in frame.astype("string").items()
    )
    if has_decimal_comma:
        frame, _, _ = _read_whitespace_table(source, decimal=",")

    # Validate field topology before numeric conversion or row removal.
    columns = [str(column) for column in frame.columns]
    schema = validate_tracking_columns(columns)
    working = frame.copy()
    working.columns = columns

    # Historical exports end with an optional summary row rather than a frame.
    dropped = False
    if drop_trailing_summary_row:
        trailing_frame_id = pd.to_numeric(
            working["FrameID"].iloc[-1], errors="coerce"
        )
        if pd.isna(trailing_frame_id):
            working = working.iloc[:-1].copy()
            dropped = True

    # The source's optional trailing summary row may have a non-numeric FrameID.
    working["FrameID"] = pd.to_numeric(working["FrameID"], errors="coerce")
    if working["FrameID"].isna().any():
        raise SchemaValidationError(
            "Tracking FrameID values must be numeric after optional summary-row handling."
        )
    working["FrameID"] = working["FrameID"].astype("int64")

    # Invalid tracking coordinates remain NaN for downstream validity masking.
    for column in schema.angle_columns + schema.x_columns + schema.y_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").astype(
            "float64"
        )

    # Only full-field tracking is supported by the active candidate workflow.
    if mode != "full":
        raise SchemaValidationError(f"Unsupported tracking mode: {mode!r}")
    selected = working.loc[
        :,
        ["FrameID", *schema.x_columns, *schema.y_columns, *schema.angle_columns],
    ].copy()

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
    )


def read_protocol(path: Path) -> ProtocolReadResult:
    """Read stimulus-control events with explicit duration checks."""
    # Protocol values use the normal decimal convention and strict header schema.
    source = _require_file(path)
    frame, _, _ = _read_whitespace_table(source, decimal=".")
    columns = [str(column) for column in frame.columns]
    validate_protocol_columns(columns)
    # Select canonical fields and convert event times/labels to explicit types.
    working = frame.loc[:, list(PROTOCOL_COLUMNS)].copy()
    working["Type"] = working["Type"].astype("string")
    working["Beg"] = pd.to_numeric(working["Beg"], errors="raise").astype("int64")
    working["End"] = pd.to_numeric(working["End"], errors="raise").astype("int64")
    if working.empty:
        raise SchemaValidationError("Protocol table is empty.")
    if working["Type"].isna().any():
        raise SchemaValidationError("Protocol event types cannot be missing.")

    # Record (but do not discard) invalid durations for QC and caller policy.
    invalid_duration = working["Beg"] >= working["End"]
    invalid_duration_count = int(invalid_duration.sum())
    ordered = working.sort_values(["Beg", "End", "Type"], kind="mergesort").reset_index(
        drop=True
    )
    # Stable sorting and counts make protocol artifacts deterministic to inspect.
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
