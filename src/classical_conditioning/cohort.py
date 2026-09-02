"""Reviewed cohort-manifest validation, freezing, and strict application."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
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
    write_json_atomic,
)
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    SchemaValidationError,
    ScientificValidationError,
)

RECIPE_ID = "cohort-manifest-v1"
SCHEMA_VERSION = "cohort-manifest/1.0"
COHORT_COLUMNS = (
    "experiment_id",
    "recording_id",
    "fish_id",
    "condition_id",
    "technical_valid",
    "technical_exclusion_reason",
    "behavioral_engagement",
    "behavioral_engagement_reason",
    "primary_included",
    "sensitivity_population_ids",
    "review_status",
    "reviewer",
    "reviewed_at",
    "source_qc_artifact_id",
)


@dataclass(frozen=True)
class CohortManifestResult:
    cohort_id: str
    manifest_path: Path
    review_copy_path: Path
    summary_path: Path
    completion_marker_path: Path
    logical_content_sha256: str
    row_count: int
    primary_count: int


def _validate_identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ConfigurationError(
            f"{label} must use only letters, numbers, dot, underscore, or hyphen."
        )


def _normalize_population_ids(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [part.strip() for part in value.split(",") if part.strip()]
        value = decoded
    if not isinstance(value, (list, tuple, set, np.ndarray)):
        raise SchemaValidationError(
            "sensitivity_population_ids must be a list or JSON list."
        )
    normalized = sorted({str(item).strip() for item in value if str(item).strip()})
    for population_id in normalized:
        _validate_identifier(population_id, "Sensitivity population ID")
    return normalized


def _parse_review_timestamp(value: Any) -> Any:
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT


def canonicalize_reviewed_cohort(reviewed: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonically order a fully reviewed cohort table."""
    missing = set(COHORT_COLUMNS).difference(reviewed.columns)
    if missing:
        raise SchemaValidationError(
            f"Reviewed cohort is missing columns: {sorted(missing)}"
        )
    frame = reviewed.loc[:, COHORT_COLUMNS].copy()
    if frame.empty:
        raise ScientificValidationError(
            "A frozen cohort manifest must contain at least one reviewed fish."
        )
    identity_columns = (
        "experiment_id",
        "recording_id",
        "fish_id",
        "condition_id",
        "review_status",
        "reviewer",
        "reviewed_at",
        "source_qc_artifact_id",
    )
    for column in identity_columns:
        frame[column] = frame[column].astype("string").str.strip()
        if frame[column].isna().any() or frame[column].eq("").any():
            raise SchemaValidationError(
                f"Reviewed cohort column {column!r} cannot be missing or empty."
            )
    for column in ("technical_valid", "primary_included"):
        if frame[column].isna().any():
            raise SchemaValidationError(
                f"Reviewed cohort column {column!r} cannot be missing."
            )
        if not frame[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise SchemaValidationError(
                f"Reviewed cohort column {column!r} must contain booleans."
            )
        frame[column] = frame[column].astype(bool)
    engagement_values = frame["behavioral_engagement"]
    invalid_engagement = engagement_values.map(
        lambda value: not (
            value is None
            or value is pd.NA
            or (isinstance(value, float) and pd.isna(value))
            or isinstance(value, bool)
        )
    )
    if invalid_engagement.any():
        raise SchemaValidationError(
            "behavioral_engagement must contain booleans or missing values."
        )
    frame["behavioral_engagement"] = frame["behavioral_engagement"].astype(
        "boolean"
    )

    if frame.duplicated(["experiment_id", "fish_id"]).any():
        raise SchemaValidationError(
            "Reviewed cohort contains duplicate experiment_id/fish_id rows."
        )
    if frame.duplicated(["recording_id"]).any():
        raise SchemaValidationError(
            "Reviewed cohort contains duplicate recording_id rows."
        )
    if not frame["review_status"].eq("approved").all():
        raise ScientificValidationError(
            "Every cohort row must have review_status='approved' before freezing."
        )
    parsed_review_times = frame["reviewed_at"].map(_parse_review_timestamp)
    if parsed_review_times.map(
        lambda value: pd.isna(value) or value.tzinfo is None
    ).any():
        raise SchemaValidationError(
            "reviewed_at values must include an explicit timezone."
        )
    reviewed_at = pd.to_datetime(frame["reviewed_at"], utc=True, errors="coerce")
    if reviewed_at.isna().any():
        raise SchemaValidationError(
            "reviewed_at values must be parseable timestamps with UTC normalization."
        )
    frame["reviewed_at"] = reviewed_at.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    invalid_primary = frame["primary_included"] & ~frame["technical_valid"]
    if invalid_primary.any():
        raise ScientificValidationError(
            "Technically invalid fish cannot be included in the primary cohort."
        )
    if not frame["primary_included"].any():
        raise ScientificValidationError(
            "A frozen primary cohort must include at least one fish."
        )
    exclusion_reason = frame["technical_exclusion_reason"].astype("string").str.strip()
    if ((~frame["technical_valid"]) & (exclusion_reason.isna() | exclusion_reason.eq(""))).any():
        raise SchemaValidationError(
            "Technically invalid fish require a technical_exclusion_reason."
        )
    if (frame["technical_valid"] & exclusion_reason.notna() & exclusion_reason.ne("")).any():
        raise SchemaValidationError(
            "Technically valid fish cannot have a technical_exclusion_reason."
        )
    frame["technical_exclusion_reason"] = exclusion_reason
    frame["behavioral_engagement_reason"] = (
        frame["behavioral_engagement_reason"].astype("string").str.strip()
    )
    frame["sensitivity_population_ids"] = frame[
        "sensitivity_population_ids"
    ].map(_normalize_population_ids)
    frame = frame.sort_values(
        ["experiment_id", "condition_id", "recording_id", "fish_id"],
        kind="stable",
    ).reset_index(drop=True)
    return frame


def logical_cohort_hash(frame: pd.DataFrame) -> str:
    """Hash canonical cohort meaning independently of Parquet serialization."""
    canonical = canonicalize_reviewed_cohort(frame)
    records = []
    for row in canonical.itertuples(index=False, name=None):
        record = {}
        for column, value in zip(COHORT_COLUMNS, row, strict=True):
            if value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                value = None
            elif isinstance(value, np.bool_):
                value = bool(value)
            record[column] = value
        records.append(record)
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "columns": list(COHORT_COLUMNS),
            "records": records,
        },
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_cohort(
    data: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    include_column: str = "primary_included",
) -> pd.DataFrame:
    """Apply one cohort by a validated many-to-one experiment/fish join."""
    required_data = {"experiment_id", "fish_id"}
    missing_data = required_data.difference(data.columns)
    if missing_data:
        raise SchemaValidationError(
            f"Data is missing cohort join columns: {sorted(missing_data)}"
        )
    canonical = canonicalize_reviewed_cohort(manifest)
    if include_column not in canonical.columns:
        raise SchemaValidationError(
            f"Cohort include column does not exist: {include_column!r}"
        )
    if not is_bool_dtype(canonical[include_column].dtype):
        raise SchemaValidationError(
            f"Cohort include column must be boolean: {include_column!r}"
        )
    joined = data.merge(
        canonical[
            ["experiment_id", "fish_id", include_column]
        ].assign(_cohort_match=True),
        on=["experiment_id", "fish_id"],
        how="left",
        validate="many_to_one",
    )
    unmatched = joined["_cohort_match"].isna()
    if unmatched.any():
        identities = (
            joined.loc[unmatched, ["experiment_id", "fish_id"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise SchemaValidationError(
            f"Data contains rows absent from the cohort manifest: {identities[:10]}"
        )
    included = joined[include_column].astype(bool)
    return joined.loc[included].drop(columns=["_cohort_match", include_column])


def freeze_cohort_manifest(
    project_dir: Path,
    reviewed: pd.DataFrame,
    *,
    cohort_id: str,
    policy_id: str,
) -> CohortManifestResult:
    """Freeze a reviewed cohort under a new immutable cohort identity."""
    _validate_identifier(cohort_id, "Cohort ID")
    _validate_identifier(policy_id, "Policy ID")
    project_dir = project_dir.resolve()
    canonical = canonicalize_reviewed_cohort(reviewed)
    content_hash = logical_cohort_hash(canonical)

    output_dir = project_dir / "Processed data" / "Cohorts" / cohort_id
    manifest_path = output_dir / f"{RECIPE_ID}.parquet"
    review_copy_path = output_dir / f"{RECIPE_ID}_review.csv"
    summary_path = (
        project_dir
        / "Quality checks"
        / "Cohorts"
        / cohort_id
        / f"{RECIPE_ID}_summary.json"
    )
    marker_path = (
        project_dir / "Metadata" / f"{cohort_id}_{RECIPE_ID}_complete.json"
    )
    outputs = (manifest_path, review_copy_path, summary_path, marker_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Frozen cohort artifacts are immutable; use a new cohort ID: "
            f"{existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{cohort_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_manifest = staging_root / manifest_path.name
        staged_review = staging_root / review_copy_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        table = pa.Table.from_pandas(canonical, preserve_index=False, safe=True)
        pq.write_table(
            table,
            staged_manifest,
            compression="zstd",
            write_statistics=True,
        )
        review_copy = canonical.copy()
        review_copy["sensitivity_population_ids"] = review_copy[
            "sensitivity_population_ids"
        ].map(lambda values: json.dumps(values, ensure_ascii=True))
        review_copy.to_csv(staged_review, index=False, lineterminator="\n")
        summary = {
            "recipe": RECIPE_ID,
            "schema_version": SCHEMA_VERSION,
            "scientific_status": "approved_cohort_manifest",
            "cohort_id": cohort_id,
            "policy_id": policy_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "logical_content_sha256": content_hash,
            "row_count": len(canonical),
            "primary_count": int(canonical["primary_included"].sum()),
            "condition_counts": {
                str(key): int(value)
                for key, value in canonical.groupby(
                    "condition_id",
                    observed=True,
                ).size().items()
            },
            "artifacts": {
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_file(staged_manifest),
                    "compression": "zstd",
                    "compression_lossless": True,
                },
                "review_copy": {
                    "path": str(review_copy_path),
                    "sha256": sha256_file(staged_review),
                    "canonical": False,
                },
            },
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "schema_version": SCHEMA_VERSION,
                "cohort_id": cohort_id,
                "logical_content_sha256": content_hash,
                "manifest_sha256": sha256_file(staged_manifest),
                "review_copy_sha256": sha256_file(staged_review),
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        publish_transaction(
            (
                (staged_manifest, manifest_path),
                (staged_review, review_copy_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=False,
        )

    return CohortManifestResult(
        cohort_id=cohort_id,
        manifest_path=manifest_path,
        review_copy_path=review_copy_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        logical_content_sha256=content_hash,
        row_count=len(canonical),
        primary_count=int(canonical["primary_included"].sum()),
    )


def load_cohort_manifest(project_dir: Path, cohort_id: str) -> pd.DataFrame:
    """Load a frozen cohort only after byte and logical-content authentication."""
    _validate_identifier(cohort_id, "Cohort ID")
    project_dir = project_dir.resolve()
    output_dir = project_dir / "Processed data" / "Cohorts" / cohort_id
    manifest_path = output_dir / f"{RECIPE_ID}.parquet"
    review_copy_path = output_dir / f"{RECIPE_ID}_review.csv"
    summary_path = (
        project_dir
        / "Quality checks"
        / "Cohorts"
        / cohort_id
        / f"{RECIPE_ID}_summary.json"
    )
    marker_path = (
        project_dir / "Metadata" / f"{cohort_id}_{RECIPE_ID}_complete.json"
    )
    missing = [
        path
        for path in (manifest_path, review_copy_path, summary_path, marker_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing frozen cohort artifacts: {missing}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError(
            f"Frozen cohort metadata is invalid JSON: {error}"
        ) from error
    if (
        summary.get("recipe") != RECIPE_ID
        or summary.get("schema_version") != SCHEMA_VERSION
        or summary.get("cohort_id") != cohort_id
        or marker.get("status") != "complete"
        or marker.get("recipe") != RECIPE_ID
        or marker.get("schema_version") != SCHEMA_VERSION
        or marker.get("cohort_id") != cohort_id
        or marker.get("manifest_sha256") != sha256_file(manifest_path)
        or marker.get("review_copy_sha256") != sha256_file(review_copy_path)
        or marker.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ArtifactIntegrityError(
            f"Frozen cohort byte lineage is invalid for {cohort_id}."
        )
    frame = pq.read_table(manifest_path).to_pandas()
    logical_hash = logical_cohort_hash(frame)
    if (
        summary.get("logical_content_sha256") != logical_hash
        or marker.get("logical_content_sha256") != logical_hash
    ):
        raise ArtifactIntegrityError(
            f"Frozen cohort logical content is invalid for {cohort_id}."
        )
    return canonicalize_reviewed_cohort(frame)
