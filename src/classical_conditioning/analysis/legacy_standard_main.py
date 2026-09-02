"""Frozen reproduction of the standard-main stage-3 grouping transform."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import CategoricalDtype, is_bool_dtype

from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    verify_completed_parquet,
    write_json_atomic,
)
from classical_conditioning.config import (
    config_hash,
    get_experiment_spec,
    get_legacy_paper_config,
)
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    SchemaValidationError,
)

RECIPE_ID = "legacy-standard-main-v1"
LEGACY_PAPER_CONFIG_SHA256 = (
    "0c4f133a874d9d87cf9fda1e7be7b3e07b148eaaf86a2b5e647bc4c63d24d6e9"
)
STANDARD_MAIN_CONFIG_SHA256 = (
    "8ee7ea95671f9dbafc8a150853c15ad301eba30647d06634b96eb53bd0408936"
)
STANDARD_MAIN_SOURCE_COMMIT = "b0dfcb345fc185343802f6072ff7b3740498345b"
STANDARD_MAIN_SOURCE_BLOB = "bc528071b07d10b9b5923d072d6fe48327507ec3"

TIME_COLUMN = "Trial time (frame) [700 FPS]"
VIGOR_COLUMN = "Vigor (deg/ms)"
SCALED_VIGOR_COLUMN = "Scaled vigor (AU)"
TAIL_ANGLE_COLUMN = "Angle of point 15 (deg)"
ALIGNMENTS = ("CS", "US")

_METADATA_COLUMNS = (
    "Strain",
    "Age (dpf)",
    "Exp.",
    "ProtocolRig",
    "Day",
    "Fish no.",
)
_EVENT_COLUMNS = ("CS beg", "CS end", "US beg", "US end")
_REQUIRED_COLUMNS = {
    *_METADATA_COLUMNS,
    *_EVENT_COLUMNS,
    "Trial type",
    "Trial number",
    "Block name",
    TIME_COLUMN,
    TAIL_ANGLE_COLUMN,
    VIGOR_COLUMN,
    SCALED_VIGOR_COLUMN,
    "Bout beg",
    "Bout end",
    "Bout",
}
_OUTPUT_COLUMNS = (
    *_METADATA_COLUMNS,
    TIME_COLUMN,
    *_EVENT_COLUMNS,
    "Trial number",
    "Block name",
    TAIL_ANGLE_COLUMN,
    VIGOR_COLUMN,
    SCALED_VIGOR_COLUMN,
    "Bout",
    "Fish",
)


@dataclass(frozen=True)
class LegacyStandardMainConfig:
    rolling_window_frames: int = 10
    downsample_factor: int = 10
    baseline_window_frames: int = 15 * 700
    discard_list_applied_to_rows: bool = False

    def __post_init__(self) -> None:
        if self.rolling_window_frames <= 0:
            raise ConfigurationError("Rolling window must be positive.")
        if self.downsample_factor <= 0:
            raise ConfigurationError("Downsample factor must be positive.")
        if self.baseline_window_frames <= 0:
            raise ConfigurationError("Baseline window must be positive.")
        if self.discard_list_applied_to_rows:
            raise ConfigurationError(
                f"{RECIPE_ID} must preserve the non-applied stage-3 discard list."
            )


@dataclass(frozen=True)
class LegacyStandardMainResult:
    recording_id: str
    samples_paths: dict[str, Path]
    summary_path: Path
    completion_marker_path: Path
    row_counts: dict[str, int]
    trial_counts: dict[str, int]


def _recipe_config_hash(config: LegacyStandardMainConfig) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_samples(samples: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS.difference(samples.columns)
    if missing:
        raise SchemaValidationError(
            f"Frozen legacy samples are missing columns: {sorted(missing)}"
        )
    if samples["Bout"].isna().any() or not is_bool_dtype(samples["Bout"].dtype):
        raise SchemaValidationError(
            "Frozen legacy Bout must be a complete boolean column."
        )
    invalid_alignments = set(samples["Trial type"].astype(str)).difference(ALIGNMENTS)
    if invalid_alignments:
        raise SchemaValidationError(
            f"Frozen legacy samples contain unknown alignments: "
            f"{sorted(invalid_alignments)}"
        )


def _block_categories(experiment_name: str) -> list[str]:
    return list(
        dict.fromkeys(
            trial.block_10_name
            for trial in get_experiment_spec(experiment_name).analysis_trials
        )
    )


def transform_legacy_standard_main(
    samples: pd.DataFrame,
    *,
    experiment_name: str = "allDelay",
    config: LegacyStandardMainConfig = LegacyStandardMainConfig(),
) -> dict[str, pd.DataFrame]:
    """Apply the frozen standard-main stage-3 operation order."""
    _validate_samples(samples)
    result = samples.loc[
        :,
        [
            *_METADATA_COLUMNS,
            TIME_COLUMN,
            *_EVENT_COLUMNS,
            "Trial type",
            "Trial number",
            "Block name",
            TAIL_ANGLE_COLUMN,
            VIGOR_COLUMN,
            SCALED_VIGOR_COLUMN,
            "Bout beg",
            "Bout end",
            "Bout",
        ],
    ].copy()

    result["Fish"] = (
        result["Day"].astype("string")
        + "_"
        + result["Fish no."].astype("string")
    )
    result["Exp."] = (
        result["Exp."].astype(str).str.split("-", n=1).str[0].str.lower()
    )
    result[TIME_COLUMN] = result[TIME_COLUMN].astype("int32")
    result["Trial number"] = result["Trial number"].astype("int32")
    result[TAIL_ANGLE_COLUMN] = result[TAIL_ANGLE_COLUMN].astype("float32")
    result[VIGOR_COLUMN] = result[VIGOR_COLUMN].astype("float32")
    result[SCALED_VIGOR_COLUMN] = result[SCALED_VIGOR_COLUMN].astype("float32")
    for column in _EVENT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int32")

    block_lookup = get_experiment_spec(experiment_name).block_lookup()
    block_values = [
        block_lookup.get((str(alignment), int(trial_number)))
        for alignment, trial_number in zip(
            result["Trial type"],
            result["Trial number"],
            strict=True,
        )
    ]
    result["Block name"] = pd.Categorical(
        block_values,
        dtype=CategoricalDtype(
            categories=_block_categories(experiment_name),
            ordered=True,
        ),
    )
    result = result.loc[result["Block name"].notna()].copy()
    if result.empty:
        return {}

    result.loc[~result["Bout"], VIGOR_COLUMN] = np.nan
    grouping_columns = ["Fish", "Trial type", "Trial number"]
    baseline = result.loc[
        result[TIME_COLUMN] < -config.baseline_window_frames
    ]
    baseline_stats = (
        baseline.groupby(grouping_columns, observed=True)[VIGOR_COLUMN]
        .quantile([0.1, 0.9])
        .unstack()
    )
    if baseline_stats.shape[1] == 2:
        baseline_stats.columns = ["_legacy_p10", "_legacy_p90"]
        result = result.merge(
            baseline_stats,
            left_on=grouping_columns,
            right_index=True,
            how="left",
            sort=False,
            validate="many_to_one",
        )
        denominator = result["_legacy_p90"] - result["_legacy_p10"]
        valid = denominator.gt(0) & denominator.notna()
        scaled = pd.Series(np.nan, index=result.index, dtype="float64")
        scaled.loc[valid] = (
            result.loc[valid, VIGOR_COLUMN] - result.loc[valid, "_legacy_p10"]
        ) / denominator.loc[valid]
        result[SCALED_VIGOR_COLUMN] = scaled
        result.drop(columns=["_legacy_p10", "_legacy_p90"], inplace=True)
    else:
        result[SCALED_VIGOR_COLUMN] = np.nan

    result.sort_values([*grouping_columns, TIME_COLUMN], inplace=True)
    for column in (VIGOR_COLUMN, SCALED_VIGOR_COLUMN):
        source_dtype = result[column].dtype
        result[column] = (
            result.groupby(
                grouping_columns,
                observed=True,
                sort=False,
            )[column]
            .transform(
                lambda values: values.rolling(
                    config.rolling_window_frames,
                    min_periods=1,
                ).mean()
            )
            .astype(source_dtype)
        )
    result = result.groupby(
        grouping_columns,
        observed=True,
        group_keys=False,
        sort=False,
    ).nth(slice(None, None, config.downsample_factor))
    result.loc[
        ~result["Bout"],
        [VIGOR_COLUMN, SCALED_VIGOR_COLUMN],
    ] = np.nan

    outputs: dict[str, pd.DataFrame] = {}
    for alignment in ALIGNMENTS:
        aligned = result.loc[result["Trial type"].astype(str) == alignment].copy()
        if aligned.empty:
            continue
        outputs[alignment] = aligned.loc[:, _OUTPUT_COLUMNS].reset_index(drop=True)
    return outputs


def _iter_contiguous_trials(
    parquet_file: pq.ParquetFile,
    *,
    batch_rows: int,
) -> Iterator[pd.DataFrame]:
    if batch_rows <= 0:
        raise ConfigurationError("Read batch size must be positive.")
    grouping_columns = ["Fish", "Trial type", "Trial number"]
    pending = pd.DataFrame()
    closed: set[tuple[Any, ...]] = set()
    previous_key: tuple[Any, ...] | None = None

    for batch in parquet_file.iter_batches(batch_size=batch_rows):
        frame = batch.to_pandas()
        if "Fish" not in frame:
            missing = {"Day", "Fish no."}.difference(frame.columns)
            if missing:
                raise SchemaValidationError(
                    "Frozen legacy samples require Fish or Day and Fish no.; "
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
                raise SchemaValidationError(
                    f"Frozen legacy trial group is not contiguous: {key}"
                )
            if previous_key is not None and key < previous_key:
                raise SchemaValidationError(
                    "Frozen legacy trial groups are not in stable sort order."
                )
            closed.add(key)
            previous_key = key
            yield group.drop(columns="Fish")
        pending = frame.iloc[last_start:].copy()

    if not pending.empty:
        key = tuple(pending[column].iloc[0] for column in grouping_columns)
        if key in closed:
            raise SchemaValidationError(
                f"Frozen legacy trial group is not contiguous: {key}"
            )
        if previous_key is not None and key < previous_key:
            raise SchemaValidationError(
                "Frozen legacy trial groups are not in stable sort order."
            )
        yield pending.drop(columns="Fish")


def _write_standard_main_outputs(
    source_path: Path,
    staging_root: Path,
    *,
    experiment_name: str,
    config: LegacyStandardMainConfig,
    read_batch_rows: int,
) -> tuple[dict[str, Path], dict[str, int], dict[str, int], int]:
    writers: dict[str, pq.ParquetWriter] = {}
    schemas: dict[str, pa.Schema] = {}
    staged_paths = {
        alignment: staging_root
        / f"samples_{RECIPE_ID}_{alignment}.parquet"
        for alignment in ALIGNMENTS
    }
    row_counts = {alignment: 0 for alignment in ALIGNMENTS}
    trial_numbers: dict[str, set[int]] = {
        alignment: set() for alignment in ALIGNMENTS
    }
    maximum_group_rows = 0
    try:
        parquet_file = pq.ParquetFile(source_path)
        try:
            for group in _iter_contiguous_trials(
                parquet_file,
                batch_rows=read_batch_rows,
            ):
                times = group[TIME_COLUMN].to_numpy(dtype=np.int64)
                if np.any(np.diff(times) <= 0):
                    key = (
                        str(group["Trial type"].iloc[0]),
                        int(group["Trial number"].iloc[0]),
                    )
                    raise SchemaValidationError(
                        f"Frozen legacy trial time is not strictly increasing: {key}"
                    )
                maximum_group_rows = max(maximum_group_rows, len(group))
                transformed = transform_legacy_standard_main(
                    group,
                    experiment_name=experiment_name,
                    config=config,
                )
                for alignment, frame in transformed.items():
                    table = pa.Table.from_pandas(
                        frame,
                        preserve_index=False,
                        safe=True,
                    )
                    if alignment not in writers:
                        schemas[alignment] = table.schema
                        writers[alignment] = pq.ParquetWriter(
                            staged_paths[alignment],
                            table.schema,
                            compression="zstd",
                            write_statistics=True,
                        )
                    elif table.schema != schemas[alignment]:
                        table = table.cast(schemas[alignment])
                    writers[alignment].write_table(table)
                    row_counts[alignment] += len(frame)
                    trial_numbers[alignment].add(
                        int(frame["Trial number"].iloc[0])
                    )
        finally:
            parquet_file.close()
    finally:
        for writer in writers.values():
            writer.close()

    if not writers:
        raise SchemaValidationError(
            "Frozen legacy input contains no configured analysis trials."
        )
    return (
        {
            alignment: path
            for alignment, path in staged_paths.items()
            if alignment in writers
        },
        {
            alignment: count
            for alignment, count in row_counts.items()
            if alignment in writers
        },
        {
            alignment: len(numbers)
            for alignment, numbers in trial_numbers.items()
            if alignment in writers
        },
        maximum_group_rows,
    )


def build_legacy_standard_main(
    project_dir: Path,
    recording_id: str,
    *,
    experiment_name: str = "allDelay",
    config: LegacyStandardMainConfig = LegacyStandardMainConfig(),
    read_batch_rows: int = 250_000,
    overwrite: bool = False,
) -> LegacyStandardMainResult:
    """Build authenticated standard-main stage-3 Parquet artifacts."""
    if config != LegacyStandardMainConfig():
        raise ConfigurationError(
            f"{RECIPE_ID} uses a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    recipe_hash = _recipe_config_hash(config)
    if recipe_hash != STANDARD_MAIN_CONFIG_SHA256:
        raise ConfigurationError(
            f"The frozen {RECIPE_ID} configuration hash changed. "
            "Use a new recipe identity for changed behavior."
        )
    resolved_config = get_legacy_paper_config(experiment_name)
    resolved_hash = config_hash(resolved_config)
    if resolved_hash != LEGACY_PAPER_CONFIG_SHA256:
        raise ConfigurationError(
            "The frozen legacy-paper-v1 configuration hash changed consciously "
            "before running this recipe."
        )
    if read_batch_rows <= 0:
        raise ConfigurationError("Read batch size must be positive.")

    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    source_path = source_dir / "samples_legacy-v1.parquet"
    source_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "legacy-v1_preprocessing_summary.json"
    )
    source_marker_path = (
        project_dir / "Metadata" / f"{recording_id}_legacy-v1_complete.json"
    )
    source = verify_completed_parquet(
        source_path,
        source_summary_path,
        source_marker_path,
        recipe="legacy-paper-v1",
        recording_id=recording_id,
    )
    if source.summary.get("experiment") != experiment_name:
        raise ArtifactIntegrityError(
            "Frozen legacy preprocessing experiment does not match the requested "
            f"experiment: {source.summary.get('experiment')!r} != {experiment_name!r}"
        )

    final_samples_paths = {
        alignment: source_dir / f"samples_{RECIPE_ID}_{alignment}.parquet"
        for alignment in ALIGNMENTS
    }
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / f"{RECIPE_ID}_summary.json"
    )
    marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{RECIPE_ID}_complete.json"
    )
    possible_outputs = (*final_samples_paths.values(), summary_path, marker_path)
    existing = [path for path in possible_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")

    source_dir.mkdir(parents=True, exist_ok=True)
    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_paths, row_counts, trial_counts, maximum_group_rows = (
            _write_standard_main_outputs(
                source_path,
                staging_root,
                experiment_name=experiment_name,
                config=config,
                read_batch_rows=read_batch_rows,
            )
        )
        artifacts: dict[str, dict[str, Any]] = {}
        for alignment, staged_path in staged_paths.items():
            final_path = final_samples_paths[alignment]
            parquet = pq.ParquetFile(staged_path)
            try:
                if parquet.metadata.num_rows != row_counts[alignment]:
                    raise RuntimeError(
                        f"{alignment} Parquet row-count verification failed."
                    )
                artifacts[alignment] = {
                    "path": str(final_path),
                    "sha256": sha256_file(staged_path),
                    "rows": row_counts[alignment],
                    "columns": len(parquet.schema_arrow),
                    "row_groups": parquet.num_row_groups,
                    "size_bytes": staged_path.stat().st_size,
                    "compression": "zstd",
                    "compression_lossless": True,
                }
            finally:
                parquet.close()

        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "legacy_reproduction",
            "recording_id": recording_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment_name,
            "historical_source": {
                "commit": STANDARD_MAIN_SOURCE_COMMIT,
                "blob": STANDARD_MAIN_SOURCE_BLOB,
                "file": "3_FishGrouping.py",
            },
            "config": asdict(config),
            "config_sha256": recipe_hash,
            "legacy_paper_config_sha256": resolved_hash,
            "input": {
                "path": str(source_path),
                "recipe": "legacy-paper-v1",
                "sha256": source.marker["samples_sha256"],
            },
            "row_counts": row_counts,
            "trial_counts": trial_counts,
            "streaming": {
                "grouping_columns": ["Fish", "Trial type", "Trial number"],
                "source_groups_required_contiguous": True,
                "source_time_required_strictly_increasing_within_group": True,
                "maximum_source_group_rows": maximum_group_rows,
                "read_batch_rows": read_batch_rows,
            },
            "behavior": [
                "Only trials mapped to configured 10-trial blocks are retained.",
                "The configured stage-3 discard list is not applied to data rows.",
                "Vigor outside legacy bouts is missing.",
                "P10/P90 scaling uses samples earlier than -15 seconds per trial.",
                "Vigor and scaled vigor use a right-aligned rolling mean.",
                "Every tenth row is retained after smoothing.",
                "Out-of-bout vigor and scaled vigor are missing in final outputs.",
                "CS and US are published as separate alignment artifacts.",
            ],
            "artifacts": artifacts,
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "recording_id": recording_id,
                "artifact_sha256": {
                    alignment: artifact["sha256"]
                    for alignment, artifact in artifacts.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )

        current_stat = source_path.stat()
        if source.data_state != (
            current_stat.st_size,
            current_stat.st_mtime_ns,
        ) or sha256_file(source_path) != source.marker["samples_sha256"]:
            raise ArtifactIntegrityError(
                "Frozen legacy input changed during standard-main generation."
            )

        stale_alignment_paths = tuple(
            final_path
            for alignment, final_path in final_samples_paths.items()
            if alignment not in staged_paths and final_path.exists()
        )
        publish_transaction(
            (
                *(
                    (staged_path, final_samples_paths[alignment])
                    for alignment, staged_path in staged_paths.items()
                ),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
            removals=stale_alignment_paths,
        )

    return LegacyStandardMainResult(
        recording_id=recording_id,
        samples_paths={
            alignment: final_samples_paths[alignment]
            for alignment in staged_paths
        },
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_counts=row_counts,
        trial_counts=trial_counts,
    )
