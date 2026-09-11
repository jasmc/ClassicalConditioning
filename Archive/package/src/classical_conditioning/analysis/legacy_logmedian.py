"""Executable preservation of the historical downstream LogMedian transform."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_bool_dtype

from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    verify_completed_parquet,
    write_json_atomic,
)

HISTORICAL_SOURCE_COMMIT = "e4fe3f48d66916174f7fcaa4a5a18be5d49431f3"
TIME_COLUMN = "Trial time (frame) [700 FPS]"
VIGOR_COLUMN = "Vigor (deg/ms)"
SCALED_VIGOR_COLUMN = "Scaled vigor (AU)"


@dataclass(frozen=True)
class LegacyLogMedianConfig:
    rolling_window_frames: int = 10
    downsample_factor: int = 10
    baseline_window_frames: int = 15 * 700
    read_batch_rows: int = 250_000


@dataclass(frozen=True)
class LegacyLogMedianResult:
    recording_id: str
    samples_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int


def transform_legacy_logmedian(
    samples: pd.DataFrame,
    config: LegacyLogMedianConfig = LegacyLogMedianConfig(),
) -> pd.DataFrame:
    """Reproduce historical stage-3 LogMedian behavior on legacy trial samples."""
    if config.rolling_window_frames <= 0:
        raise ValueError("Rolling window must be positive.")
    if config.downsample_factor <= 0:
        raise ValueError("Downsample factor must be positive.")
    identifier_columns = (
        {"Fish"} if "Fish" in samples else {"Day", "Fish no."}
    )
    required = {
        "Trial type",
        "Trial number",
        TIME_COLUMN,
        VIGOR_COLUMN,
        "Bout",
    } | identifier_columns
    missing = required.difference(samples.columns)
    if missing:
        raise KeyError(f"Legacy samples are missing columns: {sorted(missing)}")
    if samples["Bout"].isna().any() or not is_bool_dtype(samples["Bout"].dtype):
        raise TypeError("Legacy Bout must be a complete boolean column.")

    grouping_columns = ["Fish", "Trial type", "Trial number"]
    result = samples.copy()
    if "Fish" not in result:
        result["Fish"] = (
            result["Day"].astype("string")
            + "_"
            + result["Fish no."].astype("string")
        )
    result["Fish"] = result["Fish"].astype("string")
    result["Trial number"] = result["Trial number"].astype("int32")
    result[TIME_COLUMN] = result[TIME_COLUMN].astype("int32")
    result[VIGOR_COLUMN] = result[VIGOR_COLUMN].astype("float32")
    if SCALED_VIGOR_COLUMN in result:
        result[SCALED_VIGOR_COLUMN] = result[SCALED_VIGOR_COLUMN].astype("float32")
    result.loc[~result["Bout"], VIGOR_COLUMN] = np.nan
    result.sort_values(
        [*grouping_columns, TIME_COLUMN],
        inplace=True,
    )
    result[VIGOR_COLUMN] = (
        result.groupby(
            grouping_columns,
            observed=True,
            sort=False,
        )[VIGOR_COLUMN]
        .transform(
            lambda values: values.rolling(
                config.rolling_window_frames,
                min_periods=1,
            ).median()
        )
        .astype("float32")
    )
    result = result.groupby(
        grouping_columns,
        observed=True,
        group_keys=False,
        sort=False,
    ).nth(slice(None, None, config.downsample_factor))

    vigor = result[VIGOR_COLUMN].to_numpy(dtype=np.float64)
    log_vigor = np.full(vigor.shape, np.nan, dtype=np.float64)
    positive = vigor > 0
    log_vigor[positive] = np.log(vigor[positive])
    result[VIGOR_COLUMN] = log_vigor
    baseline = (
        result.loc[result[TIME_COLUMN] < -config.baseline_window_frames]
        .groupby(grouping_columns, observed=True, sort=False)[VIGOR_COLUMN]
        .median()
        .rename("_legacy_logmedian_baseline")
    )
    result = result.merge(
        baseline,
        left_on=grouping_columns,
        right_index=True,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    result[SCALED_VIGOR_COLUMN] = (
        result[VIGOR_COLUMN] - result["_legacy_logmedian_baseline"]
    )
    result.drop(columns="_legacy_logmedian_baseline", inplace=True)
    return result.reset_index(drop=True)


def _iter_contiguous_trial_groups(
    parquet_file: pq.ParquetFile,
    *,
    batch_rows: int,
):
    if batch_rows <= 0:
        raise ValueError("Read batch size must be positive.")
    grouping_columns = ["Fish", "Trial type", "Trial number"]
    pending = pd.DataFrame()
    closed: set[tuple[Any, ...]] = set()
    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        frame = batch.to_pandas()
        if "Fish" not in frame:
            missing = {"Day", "Fish no."}.difference(frame.columns)
            if missing:
                raise KeyError(
                    "Legacy samples require Fish or Day and Fish no.; "
                    f"missing {sorted(missing)}"
                )
            frame["Fish"] = (
                frame["Day"].astype("string")
                + "_"
                + frame["Fish no."].astype("string")
            )
        if not pending.empty:
            frame = pd.concat([pending, frame], ignore_index=True)
        keys = frame[grouping_columns]
        transitions = np.flatnonzero(
            np.concatenate(
                (
                    [True],
                    np.any(
                        keys.iloc[1:].to_numpy()
                        != keys.iloc[:-1].to_numpy(),
                        axis=1,
                    ),
                )
            )
        )
        last_start = int(transitions[-1])
        for start, end in zip(transitions[:-1], transitions[1:]):
            group = frame.iloc[int(start) : int(end)].copy()
            key = tuple(group[column].iloc[0] for column in grouping_columns)
            if key in closed:
                raise ValueError(f"Legacy trial group is not contiguous: {key}")
            closed.add(key)
            yield group
        pending = frame.iloc[last_start:].copy()

    if not pending.empty:
        key = tuple(pending[column].iloc[0] for column in grouping_columns)
        if key in closed:
            raise ValueError(f"Legacy trial group is not contiguous: {key}")
        yield pending


def _transform_groups_to_parquet(
    source_path: Path,
    output_path: Path,
    config: LegacyLogMedianConfig,
) -> tuple[int, dict[str, int], int]:
    parquet_file = pq.ParquetFile(source_path)
    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    row_count = 0
    trial_numbers: dict[str, set[int]] = {}
    maximum_group_rows = 0
    group_keys: list[tuple[Any, ...]] = []
    try:
        for group in _iter_contiguous_trial_groups(
            parquet_file,
            batch_rows=config.read_batch_rows,
        ):
            times = group[TIME_COLUMN].to_numpy(dtype=np.int64)
            if np.any(np.diff(times) <= 0):
                key = (
                    group["Fish"].iloc[0],
                    group["Trial type"].iloc[0],
                    group["Trial number"].iloc[0],
                )
                raise ValueError(
                    f"Legacy trial time is not strictly increasing: {key}"
                )
            maximum_group_rows = max(maximum_group_rows, len(group))
            group_keys.append(
                (
                    group["Fish"].iloc[0],
                    group["Trial type"].iloc[0],
                    group["Trial number"].iloc[0],
                )
            )
            transformed = transform_legacy_logmedian(group, config)
            table = pa.Table.from_pandas(transformed, preserve_index=False)
            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(
                    output_path,
                    output_schema,
                    compression="zstd",
                    write_statistics=True,
                )
            elif table.schema != output_schema:
                table = table.cast(output_schema)
            writer.write_table(table)
            row_count += len(transformed)
            trial_type = str(group["Trial type"].iloc[0])
            trial_numbers.setdefault(trial_type, set()).add(
                int(group["Trial number"].iloc[0])
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("Frozen legacy input contains no trial samples.")
    if group_keys != sorted(group_keys):
        raise ValueError(
            "Legacy trial groups are not in the global stable-sort order required "
            "for bounded-memory equivalence."
        )
    return (
        row_count,
        {
            trial_type: len(numbers)
            for trial_type, numbers in trial_numbers.items()
        },
        maximum_group_rows,
    )


def _verify_legacy_input(
    project_dir: Path,
    recording_id: str,
) -> tuple[Path, dict[str, Any], tuple[int, int]]:
    source_dir = project_dir / "Processed data" / recording_id
    samples_path = source_dir / "samples_legacy-v1.parquet"
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "legacy-v1_preprocessing_summary.json"
    )
    marker_path = project_dir / "Metadata" / f"{recording_id}_legacy-v1_complete.json"
    verified = verify_completed_parquet(
        samples_path,
        summary_path,
        marker_path,
        recipe="legacy-paper-v1",
        recording_id=recording_id,
    )
    return samples_path, verified.marker, verified.data_state


def build_legacy_logmedian(
    project_dir: Path,
    recording_id: str,
    *,
    config: LegacyLogMedianConfig = LegacyLogMedianConfig(),
    overwrite: bool = False,
) -> LegacyLogMedianResult:
    """Build the historical LogMedian stage-3 artifact from frozen legacy samples."""
    project_dir = project_dir.resolve()
    source_path, source_marker, source_state = _verify_legacy_input(
        project_dir,
        recording_id,
    )
    output_dir = project_dir / "Processed data" / recording_id
    samples_path = output_dir / "samples_historical-logmedian-v1.parquet"
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "historical-logmedian-v1_summary.json"
    )
    marker_path = (
        project_dir
        / "Metadata"
        / f"{recording_id}_historical-logmedian-v1_complete.json"
    )
    outputs = (samples_path, summary_path, marker_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Historical LogMedian outputs already exist: {existing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-historical-logmedian-v1-",
    ) as staging_root:
        staged_samples = staging_root / samples_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        row_count, trial_counts, maximum_group_rows = _transform_groups_to_parquet(
            source_path,
            staged_samples,
            config,
        )
        summary = {
            "recipe": "historical-logmedian-v1",
            "scientific_status": "legacy_reproduction",
            "recording_id": recording_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "historical_source": {
                "commit": HISTORICAL_SOURCE_COMMIT,
                "file": "3_FishGrouping_LogMedian.py",
            },
            "config": asdict(config),
            "input": {
                "path": str(source_path),
                "recipe": "legacy-paper-v1",
                "sha256": source_marker["samples_sha256"],
            },
            "row_count": row_count,
            "trial_counts": trial_counts,
            "streaming": {
                "grouping_columns": ["Fish", "Trial type", "Trial number"],
                "source_groups_required_contiguous": True,
                "source_time_required_strictly_increasing_within_group": True,
                "maximum_source_group_rows": maximum_group_rows,
                "read_batch_rows": config.read_batch_rows,
            },
            "behavior": [
                "Vigor outside legacy bouts is missing.",
                "Vigor uses a right-aligned rolling median within fish and trial.",
                "Every tenth row is retained after smoothing.",
                "Only positive smoothed vigor is log-transformed.",
                "The median earlier than -15 seconds is subtracted per trial.",
            ],
        }
        artifact_hash = sha256_file(staged_samples)
        summary["artifact"] = {
            "path": str(samples_path),
            "sha256": artifact_hash,
            "compression": "zstd",
            "compression_lossless": True,
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": "historical-logmedian-v1",
                "recording_id": recording_id,
                "samples_sha256": artifact_hash,
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        source_stat = source_path.stat()
        if (
            source_state != (source_stat.st_size, source_stat.st_mtime_ns)
            or sha256_file(source_path) != source_marker["samples_sha256"]
        ):
            raise RuntimeError("Frozen legacy input changed during LogMedian generation.")
        publish_transaction(
            (
                (staged_samples, samples_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )
    return LegacyLogMedianResult(
        recording_id=recording_id,
        samples_path=samples_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_count=row_count,
    )
