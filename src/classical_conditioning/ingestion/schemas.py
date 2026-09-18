"""Raw acquisition table schemas and column validation.

Review note: these checks validate only column naming and topology. Value-level
validation is owned by readers and later intake/analysis stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from classical_conditioning.exceptions import SchemaValidationError

# Canonical schemas emitted by readers and used by intake artifacts.
CAMERA_COLUMNS = ("FrameID", "ElapsedTime", "AbsoluteTime")
PROTOCOL_COLUMNS = ("Type", "Beg", "End")

# Exact regexes prevent similarly named columns from being silently accepted.
_ANGLE = re.compile(r"^angle(\d+)$")
_X = re.compile(r"^x(\d+)$")
_Y = re.compile(r"^y(\d+)$")


# Parsed topology of a valid tracking table, reused by reader/intake code.
@dataclass(frozen=True)
class TrackingSchema:
    columns: tuple[str, ...]
    point_count: int
    angle_columns: tuple[str, ...]
    x_columns: tuple[str, ...]
    y_columns: tuple[str, ...]


def normalize_camera_columns(columns: list[str]) -> list[str]:
    """Map legacy camera aliases onto the canonical intake names."""
    # Translate only documented legacy aliases; preserve unknown names for error.
    renamed = []
    for column in columns:
        if column in {"ID", "FrameID"}:
            renamed.append("FrameID")
        elif column in {"TotalTime", "ElapsedTime"}:
            renamed.append("ElapsedTime")
        elif column == "AbsoluteTime":
            renamed.append("AbsoluteTime")
        else:
            renamed.append(column)
    return renamed


def validate_camera_columns(columns: list[str]) -> tuple[str, ...]:
    # Canonicalise accepted aliases, then require the exact expected order.
    normalized = normalize_camera_columns(columns)
    if normalized != list(CAMERA_COLUMNS):
        raise SchemaValidationError(
            f"Unexpected camera columns: {columns}; expected {list(CAMERA_COLUMNS)} "
            f"(aliases ID->FrameID and TotalTime->ElapsedTime are accepted)."
        )
    return CAMERA_COLUMNS


def validate_protocol_columns(columns: list[str]) -> tuple[str, ...]:
    # Protocol headers have no legacy aliases, so exact order is required.
    if columns != list(PROTOCOL_COLUMNS):
        raise SchemaValidationError(
            f"Unexpected protocol columns: {columns}; expected {list(PROTOCOL_COLUMNS)}."
        )
    return PROTOCOL_COLUMNS


def validate_tracking_columns(columns: list[str]) -> TrackingSchema:
    # Frame identity is mandatory and must occupy the first field.
    if not columns or columns[0] != "FrameID":
        raise SchemaValidationError("Tracking data must begin with FrameID.")

    # Extract numeric suffixes for each recognized coordinate/angle family.
    coordinate_x = {
        int(match.group(1))
        for column in columns
        if (match := _X.fullmatch(column))
    }
    coordinate_y = {
        int(match.group(1))
        for column in columns
        if (match := _Y.fullmatch(column))
    }
    angles = {
        int(match.group(1))
        for column in columns
        if (match := _ANGLE.fullmatch(column))
    }
    # Build the complete allowed header set and the required contiguous indices.
    recognized = {"FrameID"} | {
        f"{prefix}{index}"
        for prefix in ("x", "y", "angle")
        for index in coordinate_x | coordinate_y | angles
    }
    contiguous = set(range(max(angles) + 1)) if angles else set()
    # Reject missing, mismatched, sparse, or unrecognized tracking fields.
    if (
        len(angles) < 2
        or coordinate_x != coordinate_y
        or coordinate_x != angles
        or angles != contiguous
        or set(columns) != recognized
    ):
        raise SchemaValidationError(
            "Tracking data must contain only matching, contiguous xN, yN, "
            "and angleN columns starting at zero for at least two tail points."
        )

    # Return names in numeric point order, independent of input header ordering.
    point_count = max(angles) + 1
    return TrackingSchema(
        columns=tuple(columns),
        point_count=point_count,
        angle_columns=tuple(f"angle{index}" for index in range(point_count)),
        x_columns=tuple(f"x{index}" for index in range(point_count)),
        y_columns=tuple(f"y{index}" for index in range(point_count)),
    )
