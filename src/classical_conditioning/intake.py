"""Lossless local ingestion and acquisition-integrity reporting."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.artifacts import (
    artifact_staging as _artifact_staging,
    publish_transaction as _publish_transaction,
    sha256_file as _sha256_file,
    write_json_atomic as _write_json,
)
from classical_conditioning.exceptions import ConfigurationError, SchemaValidationError
from classical_conditioning.paths import (
    assert_project_dir_allowed,
    condition_from_recording_name,
    is_reserved_derived_path,
)
from classical_conditioning.ingestion.schemas import (
    validate_camera_columns,
    validate_protocol_columns,
    validate_tracking_columns,
)

SourceKind = Literal["camera", "tracking", "protocol"]

SOURCE_SUFFIXES: dict[SourceKind, str] = {
    "camera": "_cam.txt",
    "tracking": "_mp tail tracking.txt",
    "protocol": "_stim control.txt",
}


@dataclass(frozen=True)
class RecordingSources:
    recording_name: str
    recording_id: str
    camera: Path
    tracking: Path
    protocol: Path


@dataclass(frozen=True)
class IntakeResult:
    recording_id: str
    status: str
    processed_dir: Path
    quality_dir: Path
    report_path: Path
    summary_path: Path


@dataclass
class FrameSequenceStats:
    row_count: int = 0
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    minimum_frame_id: int | None = None
    maximum_frame_id: int | None = None
    gap_events: int = 0
    missing_frame_count: int = 0
    duplicate_events: int = 0
    reverse_events: int = 0
    anomaly_examples: list[dict[str, int]] = field(default_factory=list)
    _previous_frame_id: int | None = field(default=None, repr=False)

    def update(self, frame_ids: np.ndarray) -> None:
        if frame_ids.size == 0:
            return
        frame_ids = frame_ids.astype(np.int64, copy=False)
        if self.first_frame_id is None:
            self.first_frame_id = int(frame_ids[0])
        chunk_minimum = int(np.min(frame_ids))
        chunk_maximum = int(np.max(frame_ids))
        self.minimum_frame_id = (
            chunk_minimum
            if self.minimum_frame_id is None
            else min(self.minimum_frame_id, chunk_minimum)
        )
        self.maximum_frame_id = (
            chunk_maximum
            if self.maximum_frame_id is None
            else max(self.maximum_frame_id, chunk_maximum)
        )

        if self._previous_frame_id is None:
            differences = np.diff(frame_ids)
            previous = frame_ids[:-1]
            current = frame_ids[1:]
        else:
            values = np.concatenate(
                [np.array([self._previous_frame_id], dtype=np.int64), frame_ids]
            )
            differences = np.diff(values)
            previous = values[:-1]
            current = values[1:]

        self.gap_events += int(np.count_nonzero(differences > 1))
        self.missing_frame_count += int(
            np.sum(differences[differences > 1] - 1, dtype=np.int64)
        )
        self.duplicate_events += int(np.count_nonzero(differences == 0))
        self.reverse_events += int(np.count_nonzero(differences < 0))

        if len(self.anomaly_examples) < 100:
            anomaly_indices = np.flatnonzero(differences != 1)
            for index in anomaly_indices[: 100 - len(self.anomaly_examples)]:
                self.anomaly_examples.append(
                    {
                        "previous_frame_id": int(previous[index]),
                        "frame_id": int(current[index]),
                        "difference": int(differences[index]),
                    }
                )

        self.row_count += int(frame_ids.size)
        self.last_frame_id = int(frame_ids[-1])
        self._previous_frame_id = int(frame_ids[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if not key.startswith("_")
        }


@dataclass
class NumericColumnStats:
    null_count: int = 0
    nonfinite_count: int = 0
    minimum: float | None = None
    maximum: float | None = None

    def update(self, values: pd.Series) -> None:
        self.null_count += int(values.isna().sum())
        array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        self.nonfinite_count += int(np.count_nonzero(~np.isfinite(array)))
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return
        current_minimum = float(np.min(finite))
        current_maximum = float(np.max(finite))
        self.minimum = (
            current_minimum
            if self.minimum is None
            else min(self.minimum, current_minimum)
        )
        self.maximum = (
            current_maximum
            if self.maximum is None
            else max(self.maximum, current_maximum)
        )


@dataclass
class OrderedSequenceStats:
    first_value: float | None = None
    last_value: float | None = None
    duplicate_steps: int = 0
    reverse_steps: int = 0
    _previous_value: float | None = field(default=None, repr=False)

    def update(self, values: pd.Series) -> None:
        array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return
        if self.first_value is None:
            self.first_value = float(finite[0])
        if self._previous_value is not None:
            finite = np.concatenate(
                [np.array([self._previous_value], dtype=np.float64), finite]
            )
        differences = np.diff(finite)
        self.duplicate_steps += int(np.count_nonzero(differences == 0))
        self.reverse_steps += int(np.count_nonzero(differences < 0))
        self.last_value = float(finite[-1])
        self._previous_value = float(finite[-1])

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if not key.startswith("_")
        }


@dataclass
class TableStats:
    kind: SourceKind
    columns: list[str]
    dtypes: dict[str, str]
    frames: FrameSequenceStats = field(default_factory=FrameSequenceStats)
    numeric: dict[str, NumericColumnStats] = field(default_factory=dict)
    ordered_sequences: dict[str, OrderedSequenceStats] = field(default_factory=dict)
    sample_rows: list[pd.DataFrame] = field(default_factory=list, repr=False)

    def update(self, frame: pd.DataFrame, sample_stride: int) -> None:
        if "FrameID" in frame.columns:
            self.frames.update(frame["FrameID"].to_numpy(dtype=np.int64))
        else:
            self.frames.row_count += int(len(frame))
        for column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column].dtype):
                self.numeric.setdefault(column, NumericColumnStats()).update(frame[column])
        if self.kind == "camera":
            for column in ("ElapsedTime", "AbsoluteTime"):
                self.ordered_sequences.setdefault(
                    column, OrderedSequenceStats()
                ).update(frame[column])
        sampled = frame.iloc[::sample_stride]
        if not sampled.empty:
            self.sample_rows.append(sampled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "frames": self.frames.to_dict(),
            "numeric": {
                column: asdict(stats) for column, stats in self.numeric.items()
            },
            "ordered_sequences": {
                column: stats.to_dict()
                for column, stats in self.ordered_sequences.items()
            },
        }


def recording_id_from_name(recording_name: str) -> str:
    parts = recording_name.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"Recording name must start with date and fish number: {recording_name}"
        )
    return "_".join(parts[:2])


def _group_raw_triplets(input_dir: Path) -> dict[str, dict[SourceKind, Path]]:
    grouped: dict[str, dict[SourceKind, Path]] = {}
    for path in input_dir.rglob("*"):
        if not path.is_file() or is_reserved_derived_path(path, input_dir):
            continue
        for kind, suffix in SOURCE_SUFFIXES.items():
            if path.name.endswith(suffix):
                recording_name = path.name[: -len(suffix)]
                existing = grouped.setdefault(recording_name, {})
                if kind in existing and existing[kind] != path.resolve():
                    raise ValueError(
                        f"Duplicate {kind} files for {recording_name}: "
                        f"{existing[kind]} and {path.resolve()}"
                    )
                existing[kind] = path.resolve()
                break
    return grouped


def discover_recordings(input_dir: Path) -> tuple[RecordingSources, ...]:
    """Find every complete camera/tracking/protocol triplet under input_dir."""
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    complete: list[RecordingSources] = []
    for recording_name, sources in sorted(_group_raw_triplets(input_dir).items()):
        if set(sources) != set(SOURCE_SUFFIXES):
            continue
        complete.append(
            RecordingSources(
                recording_name=recording_name,
                recording_id=recording_id_from_name(recording_name),
                camera=sources["camera"],
                tracking=sources["tracking"],
                protocol=sources["protocol"],
            )
        )
    return tuple(complete)


def discover_recording(
    input_dir: Path,
    *,
    recording_id: str | None = None,
) -> RecordingSources:
    """Find one complete camera/tracking/protocol triplet."""
    complete = discover_recordings(input_dir)
    if recording_id is not None:
        matches = [item for item in complete if item.recording_id == recording_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one complete triplet for {recording_id!r}; "
                f"found {len(matches)}."
            )
        return matches[0]
    if len(complete) != 1:
        details = {
            item.recording_name: item.recording_id for item in complete
        }
        raise ValueError(
            "Expected exactly one complete camera/tracking/protocol triplet; "
            f"found {details or 'no supported files'}."
        )
    return complete[0]


def inspect_table_structure(
    path: Path,
    *,
    rows: int = 25,
) -> dict[str, Any]:
    """Read a small local preview and return structure without row values."""
    if rows < 1:
        raise ValueError("Preview rows must be at least 1.")
    preview = pd.read_csv(path, sep=r"\s+", nrows=rows, engine="c")
    return {
        "path": str(path.resolve()),
        "preview_rows": int(len(preview)),
        "columns": list(preview.columns),
        "dtypes": {column: str(dtype) for column, dtype in preview.dtypes.items()},
    }


def _schema_for(kind: SourceKind, columns: list[str]) -> pa.Schema:
    if kind == "camera":
        try:
            validate_camera_columns(columns)
        except SchemaValidationError as exc:
            raise ValueError(str(exc)) from exc
        return pa.schema(
            [
                pa.field("FrameID", pa.int64(), nullable=False),
                pa.field("ElapsedTime", pa.float64(), nullable=False),
                pa.field("AbsoluteTime", pa.int64(), nullable=False),
            ]
        )
    if kind == "tracking":
        try:
            tracking = validate_tracking_columns(columns)
        except SchemaValidationError as exc:
            raise ValueError(str(exc)) from exc
        return pa.schema(
            [pa.field("FrameID", pa.int64(), nullable=False)]
            + [
                pa.field(column, pa.float64(), nullable=True)
                for column in tracking.columns[1:]
            ]
        )
    if kind == "protocol":
        try:
            validate_protocol_columns(columns)
        except SchemaValidationError as exc:
            raise ValueError(str(exc)) from exc
        return pa.schema(
            [
                pa.field("Type", pa.string(), nullable=False),
                pa.field("Beg", pa.int64(), nullable=False),
                pa.field("End", pa.int64(), nullable=False),
            ]
        )
    raise ValueError(f"Unsupported source kind: {kind}")


def _dtype_for(schema: pa.Schema) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in schema:
        if pa.types.is_int64(field.type):
            result[field.name] = "int64"
        elif pa.types.is_float64(field.type):
            result[field.name] = "float64"
        elif pa.types.is_string(field.type):
            result[field.name] = "string"
        else:
            raise TypeError(f"Unsupported Arrow type: {field.type}")
    return result


def _read_chunks(
    path: Path,
    kind: SourceKind,
    schema: pa.Schema,
    chunk_rows: int,
) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        sep=r"\s+",
        dtype=_dtype_for(schema),
        chunksize=chunk_rows,
        engine="c",
    )


def _new_logical_digest(schema: pa.Schema) -> hashlib._Hash:
    digest = hashlib.sha256()
    digest.update(str(schema).encode("utf-8"))
    return digest


def _update_logical_digest(digest: Any, frame: pd.DataFrame) -> None:
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(
        dtype=np.uint64
    )
    digest.update(row_hashes.tobytes())


def _convert_table(
    source_path: Path,
    output_path: Path,
    kind: SourceKind,
    *,
    chunk_rows: int,
    preview_rows: int,
    overwrite: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if chunk_rows < 1:
        raise ValueError("Chunk rows must be at least 1.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Derived artifact already exists: {output_path}. "
            "Use --overwrite only when intentionally rebuilding it."
        )

    source_stat_before = source_path.stat()
    source_hash = _sha256_file(source_path)
    structure = inspect_table_structure(source_path, rows=preview_rows)
    schema = _schema_for(kind, structure["columns"])
    schema = schema.with_metadata(
        {
            b"source_path": str(source_path.resolve()).encode("utf-8"),
            b"source_sha256": source_hash.encode("ascii"),
            b"compression": b"zstd-lossless",
            b"schema_version": b"1.0",
        }
    )

    temporary = output_path.with_suffix(output_path.suffix + ".incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()

    stats: TableStats | None = None
    logical_digest = _new_logical_digest(schema.remove_metadata())
    writer: pq.ParquetWriter | None = None
    sample_stride = 500 if kind == "camera" else 1_000
    try:
        writer = pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            use_dictionary=kind == "protocol",
            write_statistics=True,
        )
        for frame in _read_chunks(source_path, kind, schema, chunk_rows):
            if list(frame.columns) != structure["columns"]:
                raise ValueError(
                    f"{kind} columns changed during reading: {list(frame.columns)}"
                )
            if stats is None:
                stats = TableStats(
                    kind=kind,
                    columns=list(frame.columns),
                    dtypes={
                        column: str(dtype) for column, dtype in frame.dtypes.items()
                    },
                )
            stats.update(frame, sample_stride)
            _update_logical_digest(logical_digest, frame)
            table = pa.Table.from_pandas(
                frame,
                schema=schema,
                preserve_index=False,
                safe=True,
            )
            writer.write_table(table, row_group_size=len(frame))
    except Exception:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        assert writer is not None
        writer.close()

    if stats is None or stats.frames.row_count == 0:
        if temporary.exists():
            temporary.unlink()
        raise ValueError(f"{kind} source contains no data rows: {source_path}")

    source_stat_after = source_path.stat()
    if (
        source_stat_before.st_size != source_stat_after.st_size
        or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
    ):
        temporary.unlink()
        raise RuntimeError(f"Raw source changed during ingestion: {source_path}")

    parquet_file: pq.ParquetFile | None = None
    try:
        parquet_file = pq.ParquetFile(temporary)
        if parquet_file.metadata.num_rows != stats.frames.row_count:
            raise RuntimeError(
                f"Parquet row count mismatch for {kind}: "
                f"{parquet_file.metadata.num_rows} != {stats.frames.row_count}"
            )

        parquet_digest = _new_logical_digest(schema.remove_metadata())
        for row_group in range(parquet_file.num_row_groups):
            restored = parquet_file.read_row_group(row_group).to_pandas()
            restored = restored.astype(_dtype_for(schema))
            _update_logical_digest(parquet_digest, restored)
        if parquet_digest.hexdigest() != logical_digest.hexdigest():
            raise RuntimeError(f"Lossless Parquet verification failed for {kind}.")
        row_group_count = parquet_file.num_row_groups
        parquet_size = temporary.stat().st_size
        parquet_hash = _sha256_file(temporary)
    except Exception:
        if parquet_file is not None:
            parquet_file.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        parquet_file.close()

    os.replace(temporary, output_path)

    samples = pd.concat(stats.sample_rows, ignore_index=True)
    summary = {
        "source": {
            "path": str(source_path.resolve()),
            "size_bytes": source_stat_before.st_size,
            "modified_time_utc": datetime.fromtimestamp(
                source_stat_before.st_mtime, timezone.utc
            ).isoformat(),
            "sha256": source_hash,
        },
        "artifact": {
            "path": str(output_path.resolve()),
            "size_bytes": parquet_size,
            "sha256": parquet_hash,
            "logical_sha256": logical_digest.hexdigest(),
            "compression": "zstd",
            "compression_lossless": True,
            "schema_version": "1.0",
            "row_groups": row_group_count,
        },
        "preview": structure,
        "statistics": stats.to_dict(),
    }
    return summary, samples


def _read_protocol(
    source_path: Path,
    output_path: Path,
    *,
    preview_rows: int,
    overwrite: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    summary, _ = _convert_table(
        source_path,
        output_path,
        "protocol",
        chunk_rows=max(preview_rows, 10_000),
        preview_rows=preview_rows,
        overwrite=overwrite,
    )
    protocol = pq.read_table(output_path).to_pandas()
    invalid_duration = protocol["Beg"] >= protocol["End"]
    summary["statistics"]["event_counts"] = {
        str(key): int(value)
        for key, value in protocol["Type"].value_counts(sort=False).items()
    }
    summary["statistics"]["invalid_duration_count"] = int(invalid_duration.sum())
    return summary, protocol


def _plot_camera(
    camera: pd.DataFrame,
    output_path: Path,
) -> None:
    elapsed = camera["ElapsedTime"].to_numpy(dtype=float)
    frames = camera["FrameID"].to_numpy(dtype=np.int64)
    frame_steps = np.diff(frames)
    elapsed_steps = np.diff(elapsed)
    valid_steps = frame_steps > 0
    sampled_interval = np.full(frame_steps.shape, np.nan, dtype=float)
    sampled_interval[valid_steps] = elapsed_steps[valid_steps] / frame_steps[valid_steps]
    frame_axis = frames[1:]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(frame_axis, sampled_interval, color="black", linewidth=0.5)
    axes[0, 0].set(title="Sampled interval per frame", ylabel="Milliseconds")
    finite = sampled_interval[np.isfinite(sampled_interval)]
    axes[0, 1].hist(finite, bins=100, color="#4696FF")
    axes[0, 1].set(title="Inter-frame interval distribution", xlabel="Milliseconds")
    effective_rate = np.full(sampled_interval.shape, np.nan, dtype=float)
    positive_interval = sampled_interval > 0
    effective_rate[positive_interval] = 1_000 / sampled_interval[positive_interval]
    axes[1, 0].plot(frame_axis, effective_rate, color="#FF6D00", linewidth=0.6)
    axes[1, 0].set(
        title="Sampled effective frame rate",
        xlabel="Frame ID",
        ylabel="Frames per second",
    )
    expected = np.nanmedian(finite) if finite.size else math.nan
    residual = elapsed - (elapsed[0] + (frames - frames[0]) * expected)
    axes[1, 1].plot(frames, residual, color="#2DB757", linewidth=0.5)
    axes[1, 1].set(
        title="Timing residual from median cadence",
        xlabel="Frame ID",
        ylabel="Milliseconds",
    )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_tracking(
    tracking: pd.DataFrame,
    output_path: Path,
) -> None:
    angle_columns = [column for column in tracking if column.startswith("angle")]
    if not angle_columns:
        return
    angles = tracking[angle_columns].to_numpy(dtype=float).T
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    image = axes[0].imshow(
        angles,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap="coolwarm",
    )
    axes[0].set(
        title="Downsampled raw tail angles",
        xlabel="Sample across recording",
        ylabel="Tail point",
    )
    fig.colorbar(image, ax=axes[0], label="Radians")
    valid_count = np.isfinite(angles).sum(axis=0)
    axes[1].plot(valid_count, color="black", linewidth=0.8)
    axes[1].set(
        title="Valid angle points",
        xlabel="Sample across recording",
        ylabel="Count",
        ylim=(-0.5, len(angle_columns) + 0.5),
    )
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_protocol(
    protocol: pd.DataFrame,
    acquisition_start: int,
    acquisition_end: int,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 3), constrained_layout=True)
    colors = {"Cycle": "#2DB757", "Reinforcer": "#750E5C"}
    levels = {name: index for index, name in enumerate(protocol["Type"].unique())}
    for event_type, group in protocol.groupby("Type", sort=False):
        axis.eventplot(
            (group["Beg"] - acquisition_start) / 60_000,
            lineoffsets=levels[event_type],
            linelengths=0.7,
            colors=colors.get(str(event_type), "#4696FF"),
            label=str(event_type),
        )
    axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
    axis.axvline(
        (acquisition_end - acquisition_start) / 60_000,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    axis.set(
        title="Stimulus protocol within acquisition",
        xlabel="Minutes from acquisition start",
        yticks=list(levels.values()),
        yticklabels=list(levels),
    )
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _quality_status(summary: dict[str, Any]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    for kind in ("camera", "tracking"):
        frame_stats = summary[kind]["statistics"]["frames"]
        for field_name in ("gap_events", "duplicate_events", "reverse_events"):
            if frame_stats[field_name]:
                warnings.append(f"{kind}: {field_name}={frame_stats[field_name]}")
        numeric = summary[kind]["statistics"]["numeric"]
        nonfinite = sum(value["nonfinite_count"] for value in numeric.values())
        if nonfinite:
            warnings.append(f"{kind}: nonfinite_values={nonfinite}")

    camera_sequences = summary["camera"]["statistics"]["ordered_sequences"]
    for column in ("ElapsedTime", "AbsoluteTime"):
        reverse_steps = camera_sequences[column]["reverse_steps"]
        if reverse_steps:
            warnings.append(f"camera: {column}_reverse_steps={reverse_steps}")
    elapsed_duplicates = camera_sequences["ElapsedTime"]["duplicate_steps"]
    if elapsed_duplicates:
        warnings.append(f"camera: ElapsedTime_duplicate_steps={elapsed_duplicates}")

    protocol = summary["protocol"]["statistics"]
    if protocol["invalid_duration_count"]:
        warnings.append(
            f"protocol: invalid_duration_count={protocol['invalid_duration_count']}"
        )
    if summary["alignment"]["protocol_events_outside_acquisition"]:
        warnings.append(
            "protocol: events extend outside the camera acquisition interval"
        )
    if summary["alignment"]["stream_nonoverlap_frame_span"]:
        warnings.append(
            "camera/tracking frame ranges differ at their boundaries"
        )
    return ("REVIEW" if warnings else "PASS"), warnings


def _write_html_report(
    result_path: Path,
    recording_name: str,
    summary: dict[str, Any],
) -> None:
    status = html.escape(summary["status"])
    warnings = summary["warnings"]
    warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in warnings)
    if not warning_items:
        warning_items = "<li>No integrity warnings.</li>"
    table_rows = []
    for kind in ("camera", "tracking", "protocol"):
        stats = summary[kind]["statistics"]
        row_count = stats["frames"]["row_count"]
        table_rows.append(
            f"<tr><td>{kind}</td><td>{row_count:,}</td>"
            f"<td>{len(stats['columns'])}</td></tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Acquisition integrity: {html.escape(recording_name)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 2rem auto; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #c4c4cd; padding: .45rem .8rem; text-align: left; }}
    img {{ max-width: 100%; margin: 1rem 0; border: 1px solid #c4c4cd; }}
    .status {{ display: inline-block; padding: .4rem .8rem; background: #ffe600; }}
  </style>
</head>
<body>
  <h1>Acquisition integrity check</h1>
  <p><strong>Recording:</strong> {html.escape(recording_name)}</p>
  <p class="status"><strong>Status:</strong> {status}</p>
  <h2>Source tables</h2>
  <table><thead><tr><th>Table</th><th>Rows</th><th>Columns</th></tr></thead>
  <tbody>{''.join(table_rows)}</tbody></table>
  <h2>Warnings</h2><ul>{warning_items}</ul>
  <h2>Camera timing</h2><img src="figures/camera_timing.png" alt="Camera timing">
  <h2>Tracking overview</h2><img src="figures/tracking_overview.png" alt="Tracking overview">
  <h2>Protocol</h2><img src="figures/protocol_timeline.png" alt="Protocol timeline">
</body>
</html>
"""
    result_path.write_text(document, encoding="utf-8")


def _stage_and_publish_intake(
    sources: RecordingSources,
    planned_outputs: tuple[Path, ...],
    staged_outputs: tuple[Path, ...],
    staging_root: Path,
    processed_dir: Path,
    quality_dir: Path,
    *,
    chunk_rows: int,
    preview_rows: int,
    overwrite: bool,
) -> IntakeResult:
    source_stats_before = {
        kind: (path.stat().st_size, path.stat().st_mtime_ns)
        for kind, path in (
            ("camera", sources.camera),
            ("tracking", sources.tracking),
            ("protocol", sources.protocol),
        )
    }

    camera_summary, camera_sample = _convert_table(
        sources.camera,
        staged_outputs[0],
        "camera",
        chunk_rows=chunk_rows,
        preview_rows=preview_rows,
        overwrite=overwrite,
    )
    tracking_summary, tracking_sample = _convert_table(
        sources.tracking,
        staged_outputs[1],
        "tracking",
        chunk_rows=chunk_rows,
        preview_rows=preview_rows,
        overwrite=overwrite,
    )
    protocol_summary, protocol_sample = _read_protocol(
        sources.protocol,
        staged_outputs[2],
        preview_rows=preview_rows,
        overwrite=overwrite,
    )
    for summary_item, final_path in (
        (camera_summary, planned_outputs[0]),
        (tracking_summary, planned_outputs[1]),
        (protocol_summary, planned_outputs[2]),
    ):
        summary_item["artifact"]["path"] = str(final_path.resolve())

    camera_frames = camera_summary["statistics"]["frames"]
    tracking_frames = tracking_summary["statistics"]["frames"]
    acquisition_start = int(
        camera_summary["statistics"]["numeric"]["AbsoluteTime"]["minimum"]
    )
    acquisition_end = int(
        camera_summary["statistics"]["numeric"]["AbsoluteTime"]["maximum"]
    )
    protocol_beg_min = int(protocol_summary["statistics"]["numeric"]["Beg"]["minimum"])
    protocol_end_max = int(protocol_summary["statistics"]["numeric"]["End"]["maximum"])
    camera_minimum = int(camera_frames["minimum_frame_id"])
    camera_maximum = int(camera_frames["maximum_frame_id"])
    tracking_minimum = int(tracking_frames["minimum_frame_id"])
    tracking_maximum = int(tracking_frames["maximum_frame_id"])
    overlap_start = max(camera_minimum, tracking_minimum)
    overlap_end = min(camera_maximum, tracking_maximum)
    camera_span = (
        camera_maximum - camera_minimum + 1
    )
    tracking_span = (
        tracking_maximum - tracking_minimum + 1
    )
    overlap_span = max(0, overlap_end - overlap_start + 1)
    nonoverlap_frame_span = camera_span + tracking_span - 2 * overlap_span

    summary: dict[str, Any] = {
        "recording_name": sources.recording_name,
        "recording_id": sources.recording_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_only": True,
        "raw_files_immutable": True,
        "camera": camera_summary,
        "tracking": tracking_summary,
        "protocol": protocol_summary,
        "alignment": {
            "overlap_first_frame_id": overlap_start,
            "overlap_last_frame_id": overlap_end,
            "overlap_frame_span": overlap_span,
            "stream_nonoverlap_frame_span": nonoverlap_frame_span,
            "protocol_events_outside_acquisition": bool(
                protocol_beg_min < acquisition_start
                or protocol_end_max > acquisition_end
            ),
        },
    }
    status, warnings = _quality_status(summary)
    summary["status"] = status
    summary["warnings"] = warnings

    source_manifest = {
        "recording_name": sources.recording_name,
        "recording_id": sources.recording_id,
        "condition_id": condition_from_recording_name(sources.recording_name),
        "raw_files_immutable": True,
        "sources": {
            "camera": camera_summary["source"],
            "tracking": tracking_summary["source"],
            "protocol": protocol_summary["source"],
        },
        "artifacts": {
            "camera": camera_summary["artifact"],
            "tracking": tracking_summary["artifact"],
            "protocol": protocol_summary["artifact"],
        },
    }
    _write_json(staged_outputs[3], summary)
    _write_json(staged_outputs[8], source_manifest)

    _plot_camera(camera_sample, staged_outputs[5])
    _plot_tracking(tracking_sample, staged_outputs[6])
    _plot_protocol(
        protocol_sample,
        acquisition_start,
        acquisition_end,
        staged_outputs[7],
    )
    _write_html_report(staged_outputs[4], sources.recording_name, summary)

    source_stats_after = {
        kind: (path.stat().st_size, path.stat().st_mtime_ns)
        for kind, path in (
            ("camera", sources.camera),
            ("tracking", sources.tracking),
            ("protocol", sources.protocol),
        )
    }
    if source_stats_after != source_stats_before:
        raise RuntimeError("One or more raw files changed during intake.")

    _publish_transaction(
        tuple(zip(staged_outputs, planned_outputs)),
        staging_root,
        overwrite=overwrite,
    )

    return IntakeResult(
        recording_id=sources.recording_id,
        status=status,
        processed_dir=processed_dir,
        quality_dir=quality_dir,
        report_path=planned_outputs[4],
        summary_path=planned_outputs[3],
    )


def intake_recording(
    input_dir: Path,
    project_dir: Path,
    *,
    recording_id: str | None = None,
    chunk_rows: int = 250_000,
    preview_rows: int = 25,
    overwrite: bool = False,
) -> IntakeResult:
    """Convert one local immutable recording triplet and report integrity."""
    sources = discover_recording(input_dir, recording_id=recording_id)
    project_dir = project_dir.resolve()
    input_dir = input_dir.resolve()
    assert_project_dir_allowed(input_dir, project_dir)

    processed_dir = project_dir / "Processed data" / sources.recording_id
    quality_dir = project_dir / "Quality checks" / sources.recording_id
    figures_dir = quality_dir / "figures"
    metadata_dir = project_dir / "Metadata"
    planned_outputs = (
        processed_dir / "camera.parquet",
        processed_dir / "tracking.parquet",
        processed_dir / "stimulus_events.parquet",
        quality_dir / "acquisition_summary.json",
        quality_dir / "acquisition_report.html",
        figures_dir / "camera_timing.png",
        figures_dir / "tracking_overview.png",
        figures_dir / "protocol_timeline.png",
        metadata_dir / f"{sources.recording_id}_source_manifest.json",
    )
    existing_outputs = [path for path in planned_outputs if path.exists()]
    if existing_outputs and not overwrite:
        rendered = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "One or more derived intake artifacts already exist. "
            f"Use --overwrite only when intentionally rebuilding them: {rendered}"
        )

    project_dir.mkdir(parents=True, exist_ok=True)
    with _artifact_staging(
        project_dir,
        prefix=f".{sources.recording_id}-intake-",
    ) as staging_root:
        staged_outputs = (
            staging_root / "Processed data" / "camera.parquet",
            staging_root / "Processed data" / "tracking.parquet",
            staging_root / "Processed data" / "stimulus_events.parquet",
            staging_root / "Quality checks" / "acquisition_summary.json",
            staging_root / "Quality checks" / "acquisition_report.html",
            staging_root / "Quality checks" / "figures" / "camera_timing.png",
            staging_root / "Quality checks" / "figures" / "tracking_overview.png",
            staging_root / "Quality checks" / "figures" / "protocol_timeline.png",
            staging_root
            / "Metadata"
            / f"{sources.recording_id}_source_manifest.json",
        )
        for staged_path in staged_outputs:
            staged_path.parent.mkdir(parents=True, exist_ok=True)
        return _stage_and_publish_intake(
            sources,
            planned_outputs,
            staged_outputs,
            staging_root,
            processed_dir,
            quality_dir,
            chunk_rows=chunk_rows,
            preview_rows=preview_rows,
            overwrite=overwrite,
        )


@dataclass(frozen=True)
class IntakeBatchResult:
    recording_ids: tuple[str, ...]
    completed: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]


def intake_recordings(
    input_dir: Path,
    project_dir: Path,
    *,
    keep_conditions: Iterable[str] | None = None,
    recording_ids: Iterable[str] | None = None,
    chunk_rows: int = 250_000,
    preview_rows: int = 25,
    overwrite: bool = False,
    progress: "PipelineProgress | None" = None,
) -> IntakeBatchResult:
    """Intake every selected complete triplet, skipping existing unless overwrite."""
    from classical_conditioning.progress import default_progress

    progress = progress or default_progress(enabled=False)
    keep = (
        {token.strip().lower() for token in keep_conditions}
        if keep_conditions
        else None
    )
    requested = (
        tuple(dict.fromkeys(recording_ids)) if recording_ids is not None else None
    )
    selected: list[RecordingSources] = []
    for sources in discover_recordings(input_dir):
        if requested is not None and sources.recording_id not in requested:
            continue
        if keep is not None:
            condition = condition_from_recording_name(sources.recording_name)
            if condition not in keep:
                continue
        selected.append(sources)
    if requested is not None:
        found = {item.recording_id for item in selected}
        missing = [item for item in requested if item not in found]
        if missing:
            raise ConfigurationError(
                "Requested recording IDs were not found among complete kept "
                f"triplets: {missing}"
            )

    completed: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    total = len(selected)
    for index, sources in enumerate(
        progress.iter_items(selected, description="Intake"),
        start=1,
    ):
        marker = (
            project_dir.resolve()
            / "Metadata"
            / f"{sources.recording_id}_source_manifest.json"
        )
        if marker.exists() and not overwrite:
            skipped.append(sources.recording_id)
            progress.item_done(
                index, total, sources.recording_id, status="skipped"
            )
            continue
        try:
            intake_recording(
                input_dir,
                project_dir,
                recording_id=sources.recording_id,
                chunk_rows=chunk_rows,
                preview_rows=preview_rows,
                overwrite=overwrite,
            )
            completed.append(sources.recording_id)
            progress.item_done(
                index, total, sources.recording_id, status="completed"
            )
        except Exception as error:
            failed.append((sources.recording_id, str(error)))
            progress.item_done(
                index,
                total,
                sources.recording_id,
                status=f"failed: {error}",
            )
    return IntakeBatchResult(
        recording_ids=tuple(item.recording_id for item in selected),
        completed=tuple(completed),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )
