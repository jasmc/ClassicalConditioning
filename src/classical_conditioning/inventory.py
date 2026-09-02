"""Read-only discovery and hashing of local raw recording triplets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from classical_conditioning.artifacts import sha256_file, write_json_atomic
from classical_conditioning.exceptions import ConfigurationError, SchemaValidationError
from classical_conditioning.ingestion.schemas import validate_tracking_columns
from classical_conditioning.intake import (
    SOURCE_SUFFIXES,
    SourceKind,
    recording_id_from_name,
)
from classical_conditioning.paths import (
    assert_output_outside_raw_or_in_paper_data,
    condition_from_recording_name,
    is_reserved_derived_path,
)


def _inventory_hash(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def inspect_tracking_header(path: Path) -> dict[str, Any]:
    """Read only the tracking header to recover point-count schema facts."""
    tracking_path = path.resolve()
    if not tracking_path.is_file():
        raise FileNotFoundError(f"Tracking file does not exist: {tracking_path}")
    header = pd.read_csv(tracking_path, sep=r"\s+", nrows=0, engine="c")
    columns = [str(column) for column in header.columns]
    try:
        schema = validate_tracking_columns(columns)
    except SchemaValidationError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "columns": columns,
            "point_count": None,
            "has_xy": None,
        }
    return {
        "ok": True,
        "error": None,
        "columns": columns,
        "point_count": schema.point_count,
        "has_xy": True,
        "angle_columns": list(schema.angle_columns),
    }


def _tracking_schema_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    point_counts: dict[str, int] = {}
    failures = 0
    inspected = 0
    for record in records:
        schema = record.get("tracking_schema")
        if not schema:
            continue
        inspected += 1
        if not schema.get("ok"):
            failures += 1
            continue
        key = str(schema.get("point_count"))
        point_counts[key] = point_counts.get(key, 0) + 1
    return {
        "inspected_count": inspected,
        "failure_count": failures,
        "point_count_histogram": point_counts,
        "stable_point_count": (
            len(point_counts) == 1 and failures == 0 and inspected > 0
        ),
    }


def build_recording_inventory(
    input_dir: Path,
    *,
    hash_files: bool = True,
    inspect_tracking_headers: bool = False,
) -> dict[str, Any]:
    """Discover supported raw files recursively without modifying them."""
    root = input_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")

    grouped: dict[str, dict[SourceKind, set[Path]]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or is_reserved_derived_path(path, root):
            continue
        for kind, suffix in SOURCE_SUFFIXES.items():
            if path.name.endswith(suffix):
                recording_name = path.name[: -len(suffix)]
                grouped.setdefault(recording_name, {}).setdefault(kind, set()).add(
                    path.resolve()
                )
                break

    parsed_recording_ids: dict[str, str | None] = {}
    recording_ids: dict[str, list[str]] = {}
    for recording_name in grouped:
        try:
            recording_id = recording_id_from_name(recording_name)
        except ValueError:
            parsed_recording_ids[recording_name] = None
            continue
        parsed_recording_ids[recording_name] = recording_id
        recording_ids.setdefault(recording_id, []).append(recording_name)

    records: list[dict[str, Any]] = []
    for recording_name, sources in sorted(grouped.items()):
        recording_id = parsed_recording_ids[recording_name]
        missing = [
            kind for kind in SOURCE_SUFFIXES if len(sources.get(kind, set())) == 0
        ]
        duplicate = [
            kind for kind in SOURCE_SUFFIXES if len(sources.get(kind, set())) > 1
        ]
        colliding_names = (
            sorted(recording_ids[recording_id])
            if recording_id is not None
            else []
        )
        if recording_id is None:
            status = "INVALID_RECORDING_NAME"
        elif duplicate:
            status = "AMBIGUOUS_COMPONENT"
        elif len(colliding_names) > 1:
            status = "AMBIGUOUS_RECORDING_ID"
        elif missing:
            status = "INCOMPLETE"
        else:
            status = "COMPLETE"

        components: dict[str, list[dict[str, Any]]] = {}
        for kind in SOURCE_SUFFIXES:
            component_records: list[dict[str, Any]] = []
            for path in sorted(sources.get(kind, set())):
                component = {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                }
                if hash_files:
                    component["sha256"] = sha256_file(path)
                component_records.append(component)
            components[kind] = component_records

        record: dict[str, Any] = {
            "recording_id": recording_id,
            "recording_name": recording_name,
            "condition_id": None,
            "status": status,
            "missing_components": missing,
            "duplicate_components": duplicate,
            "recording_id_collisions": (
                colliding_names if len(colliding_names) > 1 else []
            ),
            "components": components,
        }
        try:
            record["condition_id"] = condition_from_recording_name(recording_name)
        except ConfigurationError:
            record["condition_id"] = None
        if inspect_tracking_headers and status == "COMPLETE":
            tracking_paths = sorted(sources.get("tracking", set()))
            if len(tracking_paths) == 1:
                record["tracking_schema"] = inspect_tracking_header(tracking_paths[0])
        records.append(record)

    status_counts = {
        status: sum(record["status"] == status for record in records)
        for status in (
            "COMPLETE",
            "INCOMPLETE",
            "AMBIGUOUS_COMPONENT",
            "AMBIGUOUS_RECORDING_ID",
            "INVALID_RECORDING_NAME",
        )
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root),
        "source_files_modified": False,
        "source_hashes_included": hash_files,
        "tracking_headers_inspected": inspect_tracking_headers,
        "record_count": len(records),
        "status_counts": status_counts,
        "records_sha256": _inventory_hash(records),
        "records": records,
    }
    if inspect_tracking_headers:
        payload["tracking_schema_summary"] = _tracking_schema_summary(records)
    return payload


def write_recording_inventory(
    input_dir: Path,
    output: Path,
    *,
    hash_files: bool = True,
    inspect_tracking_headers: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write the inventory outside the immutable raw-data tree."""
    root = input_dir.resolve()
    destination = output.resolve()
    assert_output_outside_raw_or_in_paper_data(root, destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Recording inventory already exists: {destination}")
    write_json_atomic(
        destination,
        build_recording_inventory(
            root,
            hash_files=hash_files,
            inspect_tracking_headers=inspect_tracking_headers,
        ),
    )
    return destination


def complete_recording_ids(
    inventory: dict[str, Any],
    *,
    keep_conditions: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return COMPLETE recording IDs, optionally filtered by filename condition."""
    keep = (
        {token.strip().lower() for token in keep_conditions}
        if keep_conditions
        else None
    )
    selected: list[str] = []
    for record in inventory.get("records", []):
        if record.get("status") != "COMPLETE":
            continue
        condition = record.get("condition_id")
        if keep is not None and condition not in keep:
            continue
        recording_id = record.get("recording_id")
        if recording_id:
            selected.append(str(recording_id))
    return tuple(dict.fromkeys(selected))
