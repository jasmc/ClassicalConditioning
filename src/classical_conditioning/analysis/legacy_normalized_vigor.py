"""Frozen standard-main stage-5 per-trial normalized-vigor windows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import CategoricalDtype

from classical_conditioning.analysis.legacy_standard_main import (
    ALIGNMENTS,
    RECIPE_ID as STANDARD_MAIN_RECIPE_ID,
    TIME_COLUMN,
    VIGOR_COLUMN,
)
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    verify_completed_parquet_set,
    write_json_atomic,
)
from classical_conditioning.config import get_experiment_spec
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    SchemaValidationError,
)

RECIPE_ID = "legacy-normalized-vigor-v1"
CONFIG_SHA256 = "42d7d9fb285bc6813c70fe26e6b70fb9f46fe1fffdcc36e2b54c0a0203dcc850"
SOURCE_COMMIT = "b0dfcb345fc185343802f6072ff7b3740498345b"
SOURCE_BLOB = "f14095d53dec169478b410151dddeddf6ce6f326"

TIME_SECONDS_COLUMN = "Trial time (s)"
BASELINE_COLUMN = "Mean 15 s before"
RESPONSE_COLUMN = "Mean CR"
NORMALIZED_COLUMN = "Normalized vigor"
_GROUP_COLUMNS = (
    "Strain",
    "Age (dpf)",
    "Exp.",
    "ProtocolRig",
    "Day",
    "Fish no.",
    "Fish",
    "Block name",
    "Trial number",
)
_REQUIRED_COLUMNS = {
    *_GROUP_COLUMNS,
    TIME_COLUMN,
    VIGOR_COLUMN,
    "US beg",
}


@dataclass(frozen=True)
class LegacyNormalizedVigorConfig:
    expected_framerate_hz: float = 700.0
    trial_window_s: tuple[float, float] = (-21.0, 21.0)
    baseline_duration_s: float = 15.0
    response_window_s: tuple[float, float] = (0.0, 9.0)
    interval_closure: str = "both"
    apply_fish_discard: bool = False
    maximum_nan_fraction_per_window: float = 0.90
    apply_nan_fraction_filter: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.expected_framerate_hz) or self.expected_framerate_hz <= 0:
            raise ConfigurationError("Expected frame rate must be positive.")
        if self.trial_window_s[0] >= self.trial_window_s[1]:
            raise ConfigurationError("Stage-5 trial window must be increasing.")
        if not np.isfinite(self.baseline_duration_s) or self.baseline_duration_s <= 0:
            raise ConfigurationError("Stage-5 baseline duration must be positive.")
        if self.response_window_s[0] >= self.response_window_s[1]:
            raise ConfigurationError("Stage-5 response window must be increasing.")
        if self.interval_closure != "both":
            raise ConfigurationError(
                "Frozen stage-5 windows must include both endpoints."
            )
        if self.apply_fish_discard:
            raise ConfigurationError(
                f"{RECIPE_ID} must preserve disabled stage-5 fish exclusion."
            )
        if not 0 <= self.maximum_nan_fraction_per_window <= 1:
            raise ConfigurationError(
                "Stage-5 maximum missing fraction must be between zero and one."
            )
        if self.apply_nan_fraction_filter:
            raise ConfigurationError(
                f"{RECIPE_ID} must preserve disabled missing-window invalidation."
            )


@dataclass(frozen=True)
class LegacyNormalizedVigorResult:
    recording_id: str
    artifact_paths: dict[str, Path]
    summary_path: Path
    completion_marker_path: Path
    row_counts: dict[str, int]


def _config_hash(config: LegacyNormalizedVigorConfig) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _block_categories(experiment_name: str) -> list[str]:
    return list(
        dict.fromkeys(
            trial.block_10_name
            for trial in get_experiment_spec(experiment_name).analysis_trials
        )
    )


def _assign_blocks(
    frame: pd.DataFrame,
    *,
    alignment: str,
    experiment_name: str,
) -> pd.DataFrame:
    lookup = get_experiment_spec(experiment_name).block_lookup()
    result = frame.copy()
    result["Block name"] = pd.Categorical(
        [
            lookup.get((alignment, int(trial_number)))
            for trial_number in result["Trial number"]
        ],
        dtype=CategoricalDtype(
            categories=_block_categories(experiment_name),
            ordered=True,
        ),
    )
    return result


def aggregate_legacy_normalized_vigor(
    samples: pd.DataFrame,
    *,
    alignment: str,
    experiment_name: str = "allDelay",
    config: LegacyNormalizedVigorConfig = LegacyNormalizedVigorConfig(),
) -> pd.DataFrame:
    missing = _REQUIRED_COLUMNS.difference(samples.columns)
    if missing:
        raise SchemaValidationError(
            f"Standard-main samples are missing columns: {sorted(missing)}"
        )
    if alignment not in ALIGNMENTS:
        raise ConfigurationError(f"Unknown stage-5 alignment: {alignment!r}")
    if samples.empty:
        raise SchemaValidationError("Standard-main samples are empty.")
    conditions = samples["Exp."].astype(str).unique()
    if len(conditions) != 1:
        raise SchemaValidationError(
            "One stage-5 aggregation input must contain exactly one condition."
        )

    frame = samples.loc[
        :,
        [
            *_GROUP_COLUMNS,
            TIME_COLUMN,
            VIGOR_COLUMN,
            "US beg",
        ],
    ].copy()
    frame[TIME_SECONDS_COLUMN] = (
        frame[TIME_COLUMN].astype("float64") / config.expected_framerate_hz
    )
    frame = frame.loc[
        frame[TIME_SECONDS_COLUMN].between(
            config.trial_window_s[0],
            config.trial_window_s[1],
            inclusive=config.interval_closure,
        )
    ].copy()
    frame = _assign_blocks(
        frame,
        alignment=alignment,
        experiment_name=experiment_name,
    )
    frame.loc[
        frame[TIME_SECONDS_COLUMN].lt(0) & frame["US beg"].gt(0),
        VIGOR_COLUMN,
    ] = np.nan

    if alignment == "CS":
        baseline_bounds = (-config.baseline_duration_s, 0.0)
        response_bounds = config.response_window_s
    else:
        response_end = config.response_window_s[1]
        baseline_bounds = (
            -config.baseline_duration_s - response_end,
            -response_end,
        )
        response_bounds = (
            config.response_window_s[0] - response_end,
            0.0,
        )

    baseline_mask = frame[TIME_SECONDS_COLUMN].between(
        *baseline_bounds,
        inclusive=config.interval_closure,
    )
    response_mask = frame[TIME_SECONDS_COLUMN].between(
        *response_bounds,
        inclusive=config.interval_closure,
    )
    grouping = list(_GROUP_COLUMNS)
    baseline = (
        frame.loc[baseline_mask]
        .groupby(grouping, observed=True)[VIGOR_COLUMN]
        .mean()
    )
    response = (
        frame.loc[response_mask]
        .groupby(grouping, observed=True)[VIGOR_COLUMN]
        .mean()
    )
    baseline_non_missing = (
        frame.loc[baseline_mask, [*grouping, VIGOR_COLUMN]]
        .assign(_not_missing=lambda data: data[VIGOR_COLUMN].notna())
        .groupby(grouping, observed=True)["_not_missing"]
        .mean()
        .rename("_baseline_non_missing")
    )
    response_non_missing = (
        frame.loc[response_mask, [*grouping, VIGOR_COLUMN]]
        .assign(_not_missing=lambda data: data[VIGOR_COLUMN].notna())
        .groupby(grouping, observed=True)["_not_missing"]
        .mean()
        .rename("_response_non_missing")
    )
    result = pd.concat(
        [baseline, response, response / baseline],
        axis=1,
        keys=[BASELINE_COLUMN, RESPONSE_COLUMN, NORMALIZED_COLUMN],
    ).reset_index()
    result = result.merge(
        baseline_non_missing.reset_index(),
        on=grouping,
        how="left",
    )
    result = result.merge(
        response_non_missing.reset_index(),
        on=grouping,
        how="left",
    )
    if config.apply_nan_fraction_filter:
        baseline_missing = 1.0 - result["_baseline_non_missing"].fillna(0.0)
        response_missing = 1.0 - result["_response_non_missing"].fillna(0.0)
        invalid = baseline_missing.gt(
            config.maximum_nan_fraction_per_window
        ) | response_missing.gt(config.maximum_nan_fraction_per_window)
        result.loc[
            invalid,
            [BASELINE_COLUMN, RESPONSE_COLUMN, NORMALIZED_COLUMN],
        ] = np.nan
    result.drop(
        columns=["_baseline_non_missing", "_response_non_missing"],
        inplace=True,
    )
    result["Trial type"] = alignment
    result = _assign_blocks(
        result,
        alignment=alignment,
        experiment_name=experiment_name,
    )
    for column in (
        "Exp.",
        "ProtocolRig",
        "Age (dpf)",
        "Day",
        "Fish no.",
        "Strain",
        "Fish",
    ):
        result[column] = result[column].astype("category")
    return result.reset_index(drop=True)


def _verify_standard_main(
    project_dir: Path,
    recording_id: str,
):
    source_dir = project_dir / "Processed data" / recording_id
    return verify_completed_parquet_set(
        {
            alignment: source_dir
            / f"samples_{STANDARD_MAIN_RECIPE_ID}_{alignment}.parquet"
            for alignment in ALIGNMENTS
        },
        project_dir
        / "Quality checks"
        / recording_id
        / f"{STANDARD_MAIN_RECIPE_ID}_summary.json",
        project_dir
        / "Metadata"
        / f"{recording_id}_{STANDARD_MAIN_RECIPE_ID}_complete.json",
        recipe=STANDARD_MAIN_RECIPE_ID,
        recording_id=recording_id,
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        write_statistics=True,
    )
    parquet = pq.ParquetFile(path)
    try:
        if parquet.metadata.num_rows != len(frame):
            raise RuntimeError("Stage-5 Parquet row-count verification failed.")
        return {
            "sha256": sha256_file(path),
            "rows": len(frame),
            "columns": len(table.column_names),
            "row_groups": parquet.num_row_groups,
            "size_bytes": path.stat().st_size,
            "compression": "zstd",
            "compression_lossless": True,
        }
    finally:
        parquet.close()


def build_legacy_normalized_vigor(
    project_dir: Path,
    recording_id: str,
    *,
    experiment_name: str = "allDelay",
    config: LegacyNormalizedVigorConfig = LegacyNormalizedVigorConfig(),
    overwrite: bool = False,
) -> LegacyNormalizedVigorResult:
    if config != LegacyNormalizedVigorConfig():
        raise ConfigurationError(
            f"{RECIPE_ID} uses a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    recipe_hash = _config_hash(config)
    if recipe_hash != CONFIG_SHA256:
        raise ConfigurationError(
            f"The frozen {RECIPE_ID} configuration hash changed. "
            "Use a new recipe identity for changed behavior."
        )
    experiment = get_experiment_spec(experiment_name)
    source = _verify_standard_main(project_dir.resolve(), recording_id)
    if source.summary.get("experiment") != experiment_name:
        raise ArtifactIntegrityError(
            "Standard-main experiment does not match the requested experiment."
        )

    project_dir = project_dir.resolve()
    output_dir = project_dir / "Processed data" / recording_id
    possible_artifact_paths = {
        alignment: output_dir / f"{RECIPE_ID}_{alignment}.parquet"
        for alignment in ALIGNMENTS
    }
    artifact_paths = {
        alignment: possible_artifact_paths[alignment]
        for alignment in source.data_paths
    }
    summary_path = (
        project_dir / "Quality checks" / recording_id / f"{RECIPE_ID}_summary.json"
    )
    marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{RECIPE_ID}_complete.json"
    )
    outputs = (*possible_artifact_paths.values(), summary_path, marker_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")

    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_paths: dict[str, Path] = {}
        records: dict[str, dict[str, Any]] = {}
        for alignment, input_path in source.data_paths.items():
            samples = pq.read_table(input_path).to_pandas()
            outcomes = aggregate_legacy_normalized_vigor(
                samples,
                alignment=alignment,
                experiment_name=experiment_name,
                config=config,
            )
            staged_path = staging_root / artifact_paths[alignment].name
            record = _write_parquet(staged_path, outcomes)
            record["path"] = str(artifact_paths[alignment])
            staged_paths[alignment] = staged_path
            records[alignment] = record

        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "legacy_reproduction",
            "recording_id": recording_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": experiment_name,
            "historical_source": {
                "commit": SOURCE_COMMIT,
                "blob": SOURCE_BLOB,
                "file": "5_NormalizedVigorPlotting.py",
                "function": "run_data_aggregation",
            },
            "config": asdict(config),
            "config_sha256": recipe_hash,
            "input": {
                "recipe": STANDARD_MAIN_RECIPE_ID,
                "summary_sha256": sha256_file(source.summary_path),
                "artifact_sha256": {
                    key: source.marker["artifact_sha256"][key]
                    for key in source.data_paths
                },
            },
            "experiment_windows": {
                "conditioned_response_window_s": [
                    experiment.conditioned_response_window.start_s,
                    experiment.conditioned_response_window.end_s,
                ],
            },
            "artifacts": records,
            "behavior": [
                "Stage-5 fish exclusion is disabled.",
                "Only samples from -21 through +21 seconds are retained.",
                "CS baseline and response windows both include time zero.",
                "US baseline and response windows both include the shifted response endpoint.",
                "Window values are arithmetic means of bout-masked vigor.",
                "Normalized vigor is response divided by baseline without a zero guard.",
                "Per-window 90 percent missing-value invalidation is disabled.",
            ],
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "recording_id": recording_id,
                "artifact_sha256": {
                    alignment: record["sha256"]
                    for alignment, record in records.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )

        for alignment, input_path in source.data_paths.items():
            stat = input_path.stat()
            if source.data_states[alignment] != (
                stat.st_size,
                stat.st_mtime_ns,
            ) or sha256_file(input_path) != source.marker["artifact_sha256"][
                alignment
            ]:
                raise ArtifactIntegrityError(
                    "Standard-main input changed during stage-5 aggregation."
                )
        stale_artifact_paths = tuple(
            path
            for alignment, path in possible_artifact_paths.items()
            if alignment not in artifact_paths and path.exists()
        )
        publish_transaction(
            tuple(
                (staged_paths[key], artifact_paths[key])
                for key in staged_paths
            )
            + (
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
            removals=stale_artifact_paths,
        )

    return LegacyNormalizedVigorResult(
        recording_id=recording_id,
        artifact_paths=artifact_paths,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_counts={
            alignment: record["rows"] for alignment, record in records.items()
        },
    )
