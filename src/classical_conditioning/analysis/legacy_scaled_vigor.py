"""Frozen standard-main stage-4 scaled-vigor aggregation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.legacy_standard_main import (
    ALIGNMENTS,
    RECIPE_ID as STANDARD_MAIN_RECIPE_ID,
    SCALED_VIGOR_COLUMN,
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
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    SchemaValidationError,
)
RECIPE_ID = "legacy-scaled-vigor-v1"
CONFIG_SHA256 = "753b5408b7100b6933078948535c77f7f8ff62c8250b6521a4241c4eefa4d5a0"
SOURCE_COMMIT = "b0dfcb345fc185343802f6072ff7b3740498345b"
SOURCE_BLOB = "7b9c62b8228d9c29f0442aa6f42c27addcd0d49a"

TIME_SECONDS_COLUMN = "Trial time (s)"
_REQUIRED_COLUMNS = {
    "Exp.",
    TIME_COLUMN,
    "Trial number",
    "Block name",
    SCALED_VIGOR_COLUMN,
    "Fish",
    "Bout",
}


def _load_discarded_fish_ids(path: Path) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        for token in re.sub(r"[;,\t]", " ", line).split():
            token = token.strip().strip('"').strip("'")
            if token and token.lower() not in {
                "fish",
                "fish_id",
                "fishid",
                "id",
                "ids",
            }:
                tokens.append(token)
    return tuple(dict.fromkeys(tokens))


@dataclass(frozen=True)
class LegacyScaledVigorConfig:
    expected_framerate_hz: float = 700.0
    x_limits_s: tuple[float, float] = (-20.0, 20.0)
    bin_widths_s: tuple[float, ...] = (0.5, 1.0)
    apply_fish_discard: bool = True
    heatmap_baseline_end_s: float = 0.0
    heatmap_lower_quantile: float = 0.1
    heatmap_upper_quantile: float = 0.9
    heatmap_clip: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.expected_framerate_hz) or self.expected_framerate_hz <= 0:
            raise ConfigurationError("Expected frame rate must be positive.")
        if self.x_limits_s[0] >= self.x_limits_s[1]:
            raise ConfigurationError("Stage-4 x limits must be increasing.")
        if not self.bin_widths_s or any(
            not np.isfinite(width) or width <= 0 for width in self.bin_widths_s
        ):
            raise ConfigurationError("Stage-4 bin widths must be positive.")
        if not (
            0 <= self.heatmap_lower_quantile < self.heatmap_upper_quantile <= 1
        ):
            raise ConfigurationError("Stage-4 heatmap quantiles are invalid.")
        if self.heatmap_clip[0] >= self.heatmap_clip[1]:
            raise ConfigurationError("Stage-4 heatmap clip must be increasing.")


@dataclass(frozen=True)
class LegacyScaledVigorTables:
    line: pd.DataFrame
    count_heatmap: pd.DataFrame
    scaled_heatmap: pd.DataFrame
    fish_count: int


@dataclass(frozen=True)
class LegacyScaledVigorResult:
    recording_id: str
    artifact_paths: dict[str, Path]
    summary_path: Path
    completion_marker_path: Path
    row_counts: dict[str, int]


@dataclass(frozen=True)
class LegacyScaledVigorCohortResult:
    analysis_id: str
    recording_ids: tuple[str, ...]
    artifact_paths: dict[str, Path]
    summary_path: Path
    completion_marker_path: Path
    row_counts: dict[str, int]


def _config_hash(config: LegacyScaledVigorConfig) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_legacy_time_bins(
    x_limits_s: tuple[float, float],
    bin_width_s: float,
) -> list[float]:
    if bin_width_s <= 0:
        raise ConfigurationError("Stage-4 bin width must be positive.")
    start_target = x_limits_s[0] - 1
    end_target = x_limits_s[1] + 2
    start_negative = np.floor(start_target / bin_width_s) * bin_width_s
    negative = np.arange(start_negative, bin_width_s, bin_width_s)
    end_positive = np.ceil(end_target / bin_width_s) * bin_width_s
    positive = np.arange(0, end_positive + bin_width_s / 2, bin_width_s)
    return np.unique(np.concatenate((negative, positive))).tolist()


def aggregate_legacy_scaled_vigor(
    samples: pd.DataFrame,
    *,
    bin_width_s: float,
    discarded_fish_ids: tuple[str, ...] = (),
    config: LegacyScaledVigorConfig = LegacyScaledVigorConfig(),
) -> LegacyScaledVigorTables:
    missing = _REQUIRED_COLUMNS.difference(samples.columns)
    if missing:
        raise SchemaValidationError(
            f"Standard-main samples are missing columns: {sorted(missing)}"
        )
    if samples.empty:
        raise SchemaValidationError("Standard-main samples are empty.")
    if samples["Bout"].isna().any():
        raise SchemaValidationError("Standard-main Bout contains missing values.")
    conditions = samples["Exp."].astype(str).unique()
    if len(conditions) != 1:
        raise SchemaValidationError(
            "One stage-4 aggregation input must contain exactly one condition."
        )

    frame = samples.loc[
        :,
        [
            "Exp.",
            TIME_COLUMN,
            "Trial number",
            "Block name",
            SCALED_VIGOR_COLUMN,
            "Fish",
            "Bout",
        ],
    ].copy()
    if config.apply_fish_discard and discarded_fish_ids:
        frame = frame.loc[~frame["Fish"].isin(discarded_fish_ids)].copy()
    if frame.empty:
        raise SchemaValidationError(
            "Stage-4 fish exclusion removed every input row."
        )
    frame.loc[~frame["Bout"], SCALED_VIGOR_COLUMN] = np.nan
    frame[TIME_SECONDS_COLUMN] = (
        frame[TIME_COLUMN].astype("float64") / config.expected_framerate_hz
    )
    fish_count = int(frame["Fish"].nunique())
    if fish_count < 1:
        raise SchemaValidationError("Stage-4 input contains no fish identity.")

    exact = (
        frame.groupby(
            [TIME_SECONDS_COLUMN, "Trial number", "Block name"],
            observed=True,
        )
        .agg(
            **{
                SCALED_VIGOR_COLUMN: (SCALED_VIGOR_COLUMN, "median"),
                "Count": (SCALED_VIGOR_COLUMN, "count"),
            }
        )
        .reset_index()
    )
    exact["_bin"] = pd.cut(
        exact[TIME_SECONDS_COLUMN],
        compute_legacy_time_bins(config.x_limits_s, bin_width_s),
        include_lowest=True,
    )
    binned = (
        exact.groupby(
            ["_bin", "Trial number", "Block name"],
            observed=True,
        )
        .agg(
            **{
                SCALED_VIGOR_COLUMN: (SCALED_VIGOR_COLUMN, "mean"),
                "Count": ("Count", "sum"),
            }
        )
        .reset_index()
    )
    binned[TIME_SECONDS_COLUMN] = binned["_bin"].apply(
        lambda interval: interval.mid
    ).astype(float)
    binned.drop(columns="_bin", inplace=True)
    binned.dropna(subset=[TIME_SECONDS_COLUMN], inplace=True)
    binned["Count"] = binned["Count"] / fish_count
    binned["Exp."] = str(conditions[0])
    line = binned.loc[
        :,
        [
            "Trial number",
            "Block name",
            SCALED_VIGOR_COLUMN,
            "Count",
            TIME_SECONDS_COLUMN,
            "Exp.",
        ],
    ].reset_index(drop=True)

    count_heatmap = line.loc[
        :,
        ["Trial number", TIME_SECONDS_COLUMN, "Count", "Exp."],
    ].copy()

    baseline = line.loc[
        line[TIME_SECONDS_COLUMN] < config.heatmap_baseline_end_s
    ]
    baseline_stats = (
        baseline.groupby("Trial number", observed=True)[SCALED_VIGOR_COLUMN]
        .quantile(
            [config.heatmap_lower_quantile, config.heatmap_upper_quantile]
        )
        .unstack()
    )
    normalized = line.copy()
    if baseline_stats.empty:
        normalized[SCALED_VIGOR_COLUMN] = np.nan
    else:
        baseline_stats.columns = ["_legacy_p10", "_legacy_p90"]
        normalized = normalized.merge(
            baseline_stats,
            on="Trial number",
            how="left",
            validate="many_to_one",
        )
        numerator = (
            normalized[SCALED_VIGOR_COLUMN] - normalized["_legacy_p10"]
        )
        denominator = normalized["_legacy_p90"] - normalized["_legacy_p10"]
        valid = denominator.gt(0) & denominator.notna()
        normalized[SCALED_VIGOR_COLUMN] = np.nan
        normalized.loc[valid, SCALED_VIGOR_COLUMN] = (
            numerator.loc[valid] / denominator.loc[valid]
        ).clip(*config.heatmap_clip)
    scaled_heatmap = normalized.loc[
        normalized[SCALED_VIGOR_COLUMN].notna(),
        ["Trial number", TIME_SECONDS_COLUMN, SCALED_VIGOR_COLUMN, "Exp."],
    ].reset_index(drop=True)
    return LegacyScaledVigorTables(
        line=line,
        count_heatmap=count_heatmap.reset_index(drop=True),
        scaled_heatmap=scaled_heatmap,
        fish_count=fish_count,
    )


def _verify_standard_main(
    project_dir: Path,
    recording_id: str,
):
    source_dir = project_dir / "Processed data" / recording_id
    expected_paths = {
        alignment: source_dir
        / f"samples_{STANDARD_MAIN_RECIPE_ID}_{alignment}.parquet"
        for alignment in ALIGNMENTS
    }
    return verify_completed_parquet_set(
        expected_paths,
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


def _artifact_key(alignment: str, bin_width_s: float, table_name: str) -> str:
    milliseconds = int(round(bin_width_s * 1_000))
    return f"{alignment}_{milliseconds}ms_{table_name}"


def _cohort_artifact_key(
    condition: str,
    alignment: str,
    bin_width_s: float,
    table_name: str,
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", condition):
        raise SchemaValidationError(
            f"Condition ID is unsafe for cohort artifact naming: {condition!r}"
        )
    return f"{condition}_{_artifact_key(alignment, bin_width_s, table_name)}"


def _write_table(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
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
            raise RuntimeError("Stage-4 Parquet row-count verification failed.")
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


def build_legacy_scaled_vigor(
    project_dir: Path,
    recording_id: str,
    *,
    config: LegacyScaledVigorConfig = LegacyScaledVigorConfig(),
    overwrite: bool = False,
) -> LegacyScaledVigorResult:
    if config != LegacyScaledVigorConfig():
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

    project_dir = project_dir.resolve()
    source = _verify_standard_main(project_dir, recording_id)
    discard_path = project_dir / "Processed data" / "Discarded_fish_IDs.txt"
    discard_ids = (
        _load_discarded_fish_ids(discard_path)
        if config.apply_fish_discard and discard_path.is_file()
        else ()
    )
    discard_state = (
        (discard_path.stat().st_size, discard_path.stat().st_mtime_ns)
        if discard_path.is_file()
        else None
    )
    discard_hash = sha256_file(discard_path) if discard_path.is_file() else None

    output_dir = project_dir / "Processed data" / recording_id
    possible_artifact_paths: dict[str, Path] = {}
    for alignment in ALIGNMENTS:
        for bin_width_s in config.bin_widths_s:
            for table_name in ("line", "count_heatmap", "scaled_heatmap"):
                key = _artifact_key(alignment, bin_width_s, table_name)
                possible_artifact_paths[key] = (
                    output_dir / f"{RECIPE_ID}_{key}.parquet"
                )
    artifact_paths = {
        key: path
        for key, path in possible_artifact_paths.items()
        if key.split("_", 1)[0] in source.data_paths
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
        fish_counts: dict[str, int] = {}
        for alignment, input_path in source.data_paths.items():
            samples = pq.read_table(input_path).to_pandas()
            for bin_width_s in config.bin_widths_s:
                tables = aggregate_legacy_scaled_vigor(
                    samples,
                    bin_width_s=bin_width_s,
                    discarded_fish_ids=discard_ids,
                    config=config,
                )
                fish_counts[alignment] = tables.fish_count
                for table_name in ("line", "count_heatmap", "scaled_heatmap"):
                    key = _artifact_key(alignment, bin_width_s, table_name)
                    staged_path = staging_root / artifact_paths[key].name
                    record = _write_table(
                        staged_path,
                        getattr(tables, table_name),
                    )
                    record["path"] = str(artifact_paths[key])
                    staged_paths[key] = staged_path
                    records[key] = record

        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "legacy_reproduction",
            "recording_id": recording_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "historical_source": {
                "commit": SOURCE_COMMIT,
                "blob": SOURCE_BLOB,
                "file": "4_ScaledVigorPlotting.py",
                "function": "run_build_pooled_outputs",
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
            "fish_exclusion": {
                "applied": config.apply_fish_discard,
                "path": str(discard_path),
                "present": discard_path.is_file(),
                "sha256": discard_hash,
                "fish_ids": list(discard_ids),
            },
            "fish_counts": fish_counts,
            "artifacts": records,
            "behavior": [
                "Configured fish IDs are excluded before aggregation.",
                "Scaled vigor outside legacy bouts is missing.",
                "Exact-time values are pooled by median and non-missing count.",
                "Time-bin values average exact-time medians and sum counts.",
                "Counts are divided by the total retained fish count.",
                "Heatmap scaled vigor receives a second all-negative-time P10/P90 transform.",
                "Heatmap scaled vigor is clipped to the inclusive zero-to-one range.",
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
                    key: record["sha256"] for key, record in records.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )

        for key, input_path in source.data_paths.items():
            stat = input_path.stat()
            if source.data_states[key] != (
                stat.st_size,
                stat.st_mtime_ns,
            ) or sha256_file(input_path) != source.marker["artifact_sha256"][key]:
                raise ArtifactIntegrityError(
                    "Standard-main input changed during stage-4 aggregation."
                )
        if discard_state is not None:
            stat = discard_path.stat()
            if discard_state != (
                stat.st_size,
                stat.st_mtime_ns,
            ) or sha256_file(discard_path) != discard_hash:
                raise ArtifactIntegrityError(
                    "Fish discard list changed during stage-4 aggregation."
                )

        stale_artifact_paths = tuple(
            path
            for key, path in possible_artifact_paths.items()
            if key not in artifact_paths and path.exists()
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

    return LegacyScaledVigorResult(
        recording_id=recording_id,
        artifact_paths=artifact_paths,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_counts={key: record["rows"] for key, record in records.items()},
    )


def build_legacy_scaled_vigor_cohort(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    config: LegacyScaledVigorConfig = LegacyScaledVigorConfig(),
    overwrite: bool = False,
) -> LegacyScaledVigorCohortResult:
    """Pool authenticated stage-3 recordings before frozen stage-4 aggregation."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )
    recording_ids = tuple(dict.fromkeys(recording_ids))
    if not recording_ids:
        raise ConfigurationError("At least one recording ID is required.")
    if config != LegacyScaledVigorConfig():
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

    project_dir = project_dir.resolve()
    sources = {
        recording_id: _verify_standard_main(project_dir, recording_id)
        for recording_id in recording_ids
    }
    source_experiments = {
        source.summary.get("experiment") for source in sources.values()
    }
    if len(source_experiments) != 1 or None in source_experiments:
        raise ArtifactIntegrityError(
            "Cohort stage-4 inputs do not share one authenticated experiment."
        )
    experiment_name = next(iter(source_experiments))
    discard_path = project_dir / "Processed data" / "Discarded_fish_IDs.txt"
    discard_ids = (
        _load_discarded_fish_ids(discard_path)
        if config.apply_fish_discard and discard_path.is_file()
        else ()
    )
    discard_state = (
        (discard_path.stat().st_size, discard_path.stat().st_mtime_ns)
        if discard_path.is_file()
        else None
    )
    discard_hash = sha256_file(discard_path) if discard_path.is_file() else None

    pooled: dict[tuple[str, str], pd.DataFrame] = {}
    for alignment in ALIGNMENTS:
        frames = []
        fish_recordings: dict[str, str] = {}
        for recording_id, source in sources.items():
            input_path = source.data_paths.get(alignment)
            if input_path is None:
                continue
            frame = pq.read_table(input_path).to_pandas()
            for fish_id in frame["Fish"].astype(str).unique():
                previous = fish_recordings.setdefault(fish_id, recording_id)
                if previous != recording_id:
                    raise SchemaValidationError(
                        f"Fish ID {fish_id!r} occurs in multiple recordings: "
                        f"{previous!r} and {recording_id!r}."
                    )
            frames.append(frame)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            for condition, condition_frame in combined.groupby(
                "Exp.",
                observed=True,
                sort=False,
            ):
                pooled[(str(condition), alignment)] = condition_frame.reset_index(
                    drop=True
                )

    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    artifact_paths = {
        _cohort_artifact_key(
            condition,
            alignment,
            bin_width_s,
            table_name,
        ): (
            output_dir
            / f"{RECIPE_ID}_"
            f"{_cohort_artifact_key(condition, alignment, bin_width_s, table_name)}"
            ".parquet"
        )
        for condition, alignment in pooled
        for bin_width_s in config.bin_widths_s
        for table_name in ("line", "count_heatmap", "scaled_heatmap")
    }
    summary_path = (
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{RECIPE_ID}_summary.json"
    )
    marker_path = (
        project_dir
        / "Metadata"
        / f"{analysis_id}_{RECIPE_ID}_complete.json"
    )
    outputs = (*artifact_paths.values(), summary_path, marker_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} cohort outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{analysis_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_paths: dict[str, Path] = {}
        records: dict[str, dict[str, Any]] = {}
        fish_counts: dict[str, int] = {}
        for (condition, alignment), samples in pooled.items():
            for bin_width_s in config.bin_widths_s:
                tables = aggregate_legacy_scaled_vigor(
                    samples,
                    bin_width_s=bin_width_s,
                    discarded_fish_ids=discard_ids,
                    config=config,
                )
                fish_counts[f"{condition}_{alignment}"] = tables.fish_count
                for table_name in ("line", "count_heatmap", "scaled_heatmap"):
                    key = _cohort_artifact_key(
                        condition,
                        alignment,
                        bin_width_s,
                        table_name,
                    )
                    staged_path = staging_root / artifact_paths[key].name
                    record = _write_table(
                        staged_path,
                        getattr(tables, table_name),
                    )
                    record["path"] = str(artifact_paths[key])
                    staged_paths[key] = staged_path
                    records[key] = record

        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "legacy_reproduction",
            "analysis_id": analysis_id,
            "recording_ids": list(recording_ids),
            "experiment": experiment_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "historical_source": {
                "commit": SOURCE_COMMIT,
                "blob": SOURCE_BLOB,
                "file": "4_ScaledVigorPlotting.py",
                "function": "run_build_pooled_outputs",
            },
            "config": asdict(config),
            "config_sha256": recipe_hash,
            "inputs": {
                recording_id: {
                    "summary_sha256": sha256_file(source.summary_path),
                    "artifact_sha256": {
                        key: source.marker["artifact_sha256"][key]
                        for key in source.data_paths
                    },
                }
                for recording_id, source in sources.items()
            },
            "pooling": {
                "order": "recordings_then_exact_time_then_time_bin",
                "recording_count": len(recording_ids),
                "condition_ids": sorted(
                    {condition for condition, _ in pooled}
                ),
                "performed_before_nonlinear_heatmap_scaling": True,
                "performed_separately_within_condition": True,
            },
            "fish_exclusion": {
                "applied": config.apply_fish_discard,
                "path": str(discard_path),
                "present": discard_path.is_file(),
                "sha256": discard_hash,
                "fish_ids": list(discard_ids),
            },
            "fish_counts": fish_counts,
            "artifacts": records,
            "behavior": [
                "All requested recordings are pooled within condition before exact-time aggregation.",
                "Configured fish IDs are excluded before aggregation.",
                "Scaled vigor outside legacy bouts is missing.",
                "Exact-time values are pooled by median and non-missing count.",
                "Time-bin values average exact-time medians and sum counts.",
                "Counts are divided by the total retained fish count.",
                "Heatmap scaled vigor receives a second all-negative-time P10/P90 transform.",
                "Heatmap scaled vigor is clipped to the inclusive zero-to-one range.",
            ],
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "analysis_id": analysis_id,
                "recording_ids": list(recording_ids),
                "artifact_sha256": {
                    key: record["sha256"] for key, record in records.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )

        for recording_id, source in sources.items():
            for key, input_path in source.data_paths.items():
                stat = input_path.stat()
                if source.data_states[key] != (
                    stat.st_size,
                    stat.st_mtime_ns,
                ) or sha256_file(input_path) != source.marker["artifact_sha256"][key]:
                    raise ArtifactIntegrityError(
                        "Standard-main input changed during cohort stage-4 "
                        f"aggregation: {recording_id}"
                    )
        if discard_state is not None:
            stat = discard_path.stat()
            if discard_state != (
                stat.st_size,
                stat.st_mtime_ns,
            ) or sha256_file(discard_path) != discard_hash:
                raise ArtifactIntegrityError(
                    "Fish discard list changed during cohort stage-4 aggregation."
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
        )

    return LegacyScaledVigorCohortResult(
        analysis_id=analysis_id,
        recording_ids=recording_ids,
        artifact_paths=artifact_paths,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_counts={key: record["rows"] for key, record in records.items()},
    )
