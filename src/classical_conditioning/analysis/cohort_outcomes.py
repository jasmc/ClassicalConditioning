"""Single cohort-applied population boundary for trial outcomes.

Review note: this authenticates per-recording outcome inputs, joins them to a
frozen reviewed cohort, and writes explicit eligibility evidence rather than
silently dropping fish or invalid baseline/response measurements.
"""

from __future__ import annotations

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

from classical_conditioning.analysis.inference.model_input import (
    load_authenticated_trial_outcomes,
)
from classical_conditioning.analysis.movement_state import (
    resolve_candidate_metric_source,
)
from classical_conditioning.analysis.trial_outcomes import (
    trial_outcome_settings_for_experiment,
)
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.cohort import (
    apply_cohort,
    load_cohort_manifest,
    logical_cohort_hash,
)
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    SchemaValidationError,
)

RECIPE_ID = "cohort-trial-outcomes"
ELIGIBILITY_RECIPE_ID = "analysis-eligibility"
SUPPORTED_ELIGIBILITY_OUTCOMES = {
    "total-activity": ("response_total_activity", "baseline_total_activity"),
    "conditional-intensity": (
        "conditional_intensity",
        "baseline_conditional_intensity",
    ),
}


@dataclass(frozen=True)
class CohortTrialOutcomesResult:
    # Index the frozen cohort outcome artifact and the accompanying sample-flow
    # evidence that explains its retained records.
    cohort_id: str
    cohort_hash: str
    outcomes_path: Path
    sample_flow_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int
    fish_count: int


# Identify the published eligibility table and its provenance artifacts.
@dataclass(frozen=True)
class AnalysisEligibilityResult:
    analysis_id: str
    cohort_id: str
    cohort_hash: str
    eligibility_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int
    eligible_count: int


# Ensure analysis and cohort identifiers are safe to embed in output paths.
def _validate_identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ConfigurationError(
            f"{label} must use only letters, numbers, dot, underscore, or hyphen."
        )


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    # Write one lossless table and return path/hash metadata for its parent summary.
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": len(table.column_names),
        "size_bytes": path.stat().st_size,
        "compression": "zstd",
        "compression_lossless": True,
    }


def _cohort_paths(project_dir: Path, cohort_id: str) -> tuple[Path, Path, Path, Path]:
    # The reviewed manifest is immutable; this applied outcome set is derived
    # and can be atomically refreshed at the same four canonical paths.
    root = project_dir / "Processed data" / "Cohorts" / cohort_id
    outcomes = root / f"{RECIPE_ID}.parquet"
    sample_flow = root / "cohort-sample-flow.parquet"
    summary = (
        project_dir
        / "Quality checks"
        / "Cohorts"
        / cohort_id
        / f"{RECIPE_ID}_summary.json"
    )
    marker = project_dir / "Metadata" / f"{cohort_id}_{RECIPE_ID}_complete.json"
    return outcomes, sample_flow, summary, marker


def _trial_artifact_paths(
    # Reconstruct expected trial outcome/QC/marker paths for one recording/route.
    project_dir: Path,
    recording_id: str,
    *,
    metric_recipe: str,
) -> tuple[Path, Path, Path]:
    route = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    return (
        project_dir / "Processed data" / recording_id / route.trial_outcomes_name,
        project_dir / "Quality checks" / recording_id / route.trial_summary_name,
        project_dir / "Metadata" / f"{recording_id}_{route.trial_marker_suffix}",
    )


def _validate_outcome_identity(
    # Ensure table identity columns agree with the reviewed frozen manifest.
    outcomes: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    # Compare output identity values to the frozen cohort manifest before the
    # tables are permitted to represent the same selected population.
    identity_columns = [
        "experiment_id",
        "recording_id",
        "fish_id",
        "condition_id",
    ]
    missing = set(identity_columns).difference(outcomes.columns)
    if missing:
        raise SchemaValidationError(
            f"Trial outcomes are missing identity columns: {sorted(missing)}"
        )
    expected = manifest.loc[:, identity_columns].copy()
    for column in identity_columns:
        expected[column] = expected[column].astype(str)
    observed = outcomes.loc[:, identity_columns].drop_duplicates().copy()
    for column in identity_columns:
        observed[column] = observed[column].astype(str)
    if observed["recording_id"].duplicated().any():
        raise SchemaValidationError(
            "A recording has inconsistent experiment, fish, or condition identity."
        )
    checked = observed.merge(
        expected.assign(_manifest_match=True),
        on=identity_columns,
        how="left",
        validate="one_to_one",
    )
    if checked["_manifest_match"].isna().any():
        invalid = checked.loc[
            checked["_manifest_match"].isna(), identity_columns
        ].to_dict("records")
        raise SchemaValidationError(
            "Trial-outcome identities disagree with the cohort manifest: "
            f"{invalid[:10]}"
        )


def build_cohort_trial_outcomes(
    # Authenticate all reviewed fish outcome artifacts then concatenate their rows.
    project_dir: Path,
    *,
    cohort_id: str,
    metric_recipe: str = "tail-candidate-corrected",
    overwrite: bool = False,
) -> CohortTrialOutcomesResult:
    """Apply a frozen cohort once and publish its canonical population table."""
    _validate_identifier(cohort_id, "Cohort ID")
    project_dir = project_dir.resolve()
    manifest = load_cohort_manifest(project_dir, cohort_id)
    cohort_hash = logical_cohort_hash(manifest)

    frames: list[pd.DataFrame] = []
    inputs: dict[str, dict[str, str]] = {}
    flow_rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        recording_id = str(row.recording_id)
        paths = _trial_artifact_paths(
            project_dir,
            recording_id,
            metric_recipe=metric_recipe,
        )
        existing = [path.is_file() for path in paths]
        if any(existing) and not all(existing):
            raise ArtifactIntegrityError(
                f"Partial trial-outcome artifact set for {recording_id}: {paths}"
            )
        available = all(existing)
        row_count = 0
        if available:
            loaded, records, _ = load_authenticated_trial_outcomes(
                project_dir,
                (recording_id,),
                metric_recipe=metric_recipe,
            )
            frames.append(loaded)
            inputs.update(records)
            row_count = len(loaded)
        elif bool(row.primary_included):
            raise FileNotFoundError(
                f"Primary-included recording has no trial outcomes: {recording_id}"
            )
        flow_rows.append(
            {
                "cohort_id": cohort_id,
                "cohort_hash": cohort_hash,
                "experiment_id": str(row.experiment_id),
                "recording_id": recording_id,
                "fish_id": str(row.fish_id),
                "condition_id": str(row.condition_id),
                "technical_valid": bool(row.technical_valid),
                "primary_included": bool(row.primary_included),
                "technical_exclusion_reason": row.technical_exclusion_reason,
                "trial_outcomes_available": available,
                "source_trial_rows": row_count,
                "cohort_trial_rows": row_count if bool(row.primary_included) else 0,
                "disposition": (
                    "included"
                    if bool(row.primary_included)
                    else "excluded_with_outcomes"
                    if available
                    else "excluded_without_outcomes"
                ),
            }
        )
    if not frames:
        raise ConfigurationError("The reviewed cohort has no trial-outcome artifacts.")
    all_outcomes = pd.concat(frames, ignore_index=True)
    _validate_outcome_identity(all_outcomes, manifest)
    included = apply_cohort(all_outcomes, manifest)
    included.insert(0, "cohort_hash", cohort_hash)
    included.insert(0, "cohort_id", cohort_id)
    sample_flow = pd.DataFrame(flow_rows)

    outcomes_path, flow_path, summary_path, marker_path = _cohort_paths(
        project_dir, cohort_id
    )
    existing_outputs = [
        path
        for path in (outcomes_path, flow_path, summary_path, marker_path)
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing_outputs}")
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{cohort_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_outcomes = staging_root / outcomes_path.name
        staged_flow = staging_root / flow_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        artifacts = {
            "outcomes": _write_parquet(staged_outcomes, included),
            "sample_flow": _write_parquet(staged_flow, sample_flow),
        }
        artifacts["outcomes"]["path"] = str(outcomes_path)
        artifacts["sample_flow"]["path"] = str(flow_path)
        fish_identity = included.loc[
            :, ["experiment_id", "fish_id", "condition_id"]
        ].drop_duplicates()
        condition_counts = {
            str(key): int(value)
            for key, value in fish_identity.groupby(
                "condition_id", observed=True
            ).size().items()
        }
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "cohort_population_boundary",
            "paper_approved": False,
            "cohort_id": cohort_id,
            "cohort_hash": cohort_hash,
            "metric_recipe": metric_recipe,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": len(included),
            "fish_count": len(fish_identity),
            "recording_count": int(included["recording_id"].nunique()),
            "condition_fish_counts": condition_counts,
            "inputs": inputs,
            "artifacts": artifacts,
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "cohort_id": cohort_id,
                "cohort_hash": cohort_hash,
                "artifact_sha256": {
                    name: record["sha256"] for name, record in artifacts.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        publish_transaction(
            (
                (staged_outcomes, outcomes_path),
                (staged_flow, flow_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return CohortTrialOutcomesResult(
        cohort_id=cohort_id,
        cohort_hash=cohort_hash,
        outcomes_path=outcomes_path,
        sample_flow_path=flow_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_count=len(included),
        fish_count=len(
            included.loc[:, ["experiment_id", "fish_id"]].drop_duplicates()
        ),
    )


def load_cohort_trial_outcomes(
    # Load only after marker/summary/table hashes and cohort identity agree.
    project_dir: Path,
    cohort_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the cohort population boundary after authenticating all outputs."""
    project_dir = project_dir.resolve()
    outcomes_path, flow_path, summary_path, marker_path = _cohort_paths(
        project_dir, cohort_id
    )
    missing = [
        path
        for path in (outcomes_path, flow_path, summary_path, marker_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing cohort trial-outcome artifacts: {missing}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest = load_cohort_manifest(project_dir, cohort_id)
    cohort_hash = logical_cohort_hash(manifest)
    expected = {
        "outcomes": sha256_file(outcomes_path),
        "sample_flow": sha256_file(flow_path),
    }
    if (
        summary.get("recipe") != RECIPE_ID
        or summary.get("cohort_id") != cohort_id
        or summary.get("cohort_hash") != cohort_hash
        or marker.get("status") != "complete"
        or marker.get("recipe") != RECIPE_ID
        or marker.get("cohort_id") != cohort_id
        or marker.get("cohort_hash") != cohort_hash
        or marker.get("artifact_sha256") != expected
        or marker.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ArtifactIntegrityError(
            f"Cohort trial-outcome lineage is invalid for {cohort_id}."
        )
    # A cohort table may still hash correctly after the code/config contract
    # changes. Check each source trial summary before an LME or figure reuses it.
    try:
        route = resolve_candidate_metric_source(metric_recipe=summary["metric_recipe"])
        source_inputs = summary["inputs"]
        if not isinstance(source_inputs, dict):
            raise KeyError("inputs")
        for row in manifest.itertuples(index=False):
            recording_id = str(row.recording_id)
            source = source_inputs.get(recording_id)
            if source is None:
                if bool(row.primary_included):
                    raise ArtifactIntegrityError(
                        f"Cohort trial outcomes lack an included source for {recording_id}."
                    )
                continue
            source_path = project_dir / "Processed data" / recording_id / route.trial_outcomes_name
            source_summary_path = (
                project_dir / "Quality checks" / recording_id / route.trial_summary_name
            )
            source_marker_path = (
                project_dir / "Metadata" / f"{recording_id}_{route.trial_marker_suffix}"
            )
            source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
            source_marker = json.loads(source_marker_path.read_text(encoding="utf-8"))
            expected_config = trial_outcome_settings_for_experiment(str(row.experiment_id))
            if (
                source.get("recipe") != route.trial_recipe
                or source.get("sha256") != sha256_file(source_path)
                or source_marker.get("status") != "complete"
                or source_marker.get("recipe") != route.trial_recipe
                or source_marker.get("recording_id") != recording_id
                or source_marker.get("artifact_sha256", {}).get("outcomes") != source["sha256"]
                or source_marker.get("summary_sha256") != sha256_file(source_summary_path)
                or source_summary.get("experiment") != str(row.experiment_id)
                or source_summary.get("config") != expected_config
                or source_summary.get("artifacts", {}).get("outcomes", {}).get("sha256")
                != source["sha256"]
            ):
                raise ArtifactIntegrityError(
                    f"Cohort trial outcomes use stale source settings for {recording_id}."
                )
    except (KeyError, FileNotFoundError, OSError, ConfigurationError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"Cohort trial outcomes have invalid source settings for {cohort_id}."
        ) from error
    outcomes = pq.read_table(outcomes_path).to_pandas()
    if (
        not outcomes["cohort_id"].astype(str).eq(cohort_id).all()
        or not outcomes["cohort_hash"].astype(str).eq(cohort_hash).all()
    ):
        raise ArtifactIntegrityError("Cohort identity columns do not match lineage.")
    return outcomes, summary


def build_analysis_eligibility(
    # Classify every cohort fish/outcome row with explicit technical/data reasons.
    outcomes: pd.DataFrame,
    *,
    metric_id: str,
    outcome_id: str,
    alignment: str = "CS",
    min_baseline_samples: int = 1,
    min_response_samples: int = 1,
) -> pd.DataFrame:
    """Label every candidate trial row without redefining cohort membership."""
    if outcome_id not in SUPPORTED_ELIGIBILITY_OUTCOMES:
        raise ConfigurationError(
            "Eligibility supports total-activity and conditional-intensity."
        )
    if alignment not in {"CS", "US"}:
        raise ConfigurationError("Alignment must be CS or US.")
    if min_baseline_samples < 1 or min_response_samples < 1:
        raise ConfigurationError("Eligibility sample minima must be positive.")
    response_column, baseline_column = SUPPORTED_ELIGIBILITY_OUTCOMES[outcome_id]
    required = {
        "cohort_id",
        "cohort_hash",
        "experiment_id",
        "recording_id",
        "fish_id",
        "condition_id",
        "trial_id",
        "alignment",
        "trial_number",
        "block_10_name",
        "metric_id",
        "baseline_valid_sample_count",
        "response_valid_sample_count",
        response_column,
        baseline_column,
    }
    missing = required.difference(outcomes.columns)
    if missing:
        raise SchemaValidationError(
            f"Cohort outcomes are missing eligibility columns: {sorted(missing)}"
        )
    frame = outcomes.loc[
        (outcomes["alignment"].astype(str) == alignment)
        & (outcomes["metric_id"].astype(str) == metric_id)
    ].copy()
    if frame.empty:
        raise ConfigurationError(
            f"No {alignment} outcomes for metric {metric_id!r}."
        )
    response = frame[response_column].to_numpy(dtype=float)
    baseline = frame[baseline_column].to_numpy(dtype=float)
    reasons = np.full(len(frame), "", dtype=object)
    masks = (
        (
            frame["baseline_valid_sample_count"].to_numpy(dtype=int)
            < min_baseline_samples,
            "missing_baseline_window",
        ),
        (
            frame["response_valid_sample_count"].to_numpy(dtype=int)
            < min_response_samples,
            "missing_response_window",
        ),
        (~np.isfinite(baseline), "nonfinite_baseline"),
        ((np.isfinite(baseline)) & (baseline <= 0), "nonpositive_ratio_baseline"),
        (~np.isfinite(response), "nonfinite_response"),
        (frame["block_10_name"].isna().to_numpy(), "missing_block_assignment"),
    )
    if outcome_id == "conditional-intensity":
        masks = (
            *masks[:-2],
            (~np.isfinite(response), "conditional_intensity_undefined_no_bout"),
            (
                np.isfinite(response) & (response <= 0),
                "nonpositive_conditional_intensity",
            ),
            masks[-1],
        )
    for mask, reason in masks:
        reasons[(reasons == "") & mask] = reason
    result = frame.loc[
        :,
        [
            "cohort_id",
            "cohort_hash",
            "experiment_id",
            "recording_id",
            "fish_id",
            "condition_id",
            "trial_id",
            "trial_number",
            "block_10_name",
            "metric_id",
        ],
    ].copy()
    result["fish_key"] = (
        result["experiment_id"].astype(str)
        + "::"
        + result["fish_id"].astype(str)
    )
    result["outcome_id"] = outcome_id
    result["eligible"] = reasons == ""
    result["ineligible_reason"] = reasons
    result["required_baseline_samples"] = min_baseline_samples
    result["observed_baseline_samples"] = frame[
        "baseline_valid_sample_count"
    ].to_numpy(dtype=int)
    result["required_response_samples"] = min_response_samples
    result["observed_response_samples"] = frame[
        "response_valid_sample_count"
    ].to_numpy(dtype=int)
    return result.reset_index(drop=True)


def build_analysis_eligibility_artifact(
    # Persist eligibility evidence, QC summary, and completion marker transactionally.
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    metric_id: str,
    outcome_id: str,
    alignment: str = "CS",
    min_baseline_samples: int = 1,
    min_response_samples: int = 1,
    overwrite: bool = False,
) -> AnalysisEligibilityResult:
    """Publish named row eligibility derived from the cohort population table."""
    _validate_identifier(analysis_id, "Analysis ID")
    project_dir = project_dir.resolve()
    outcomes, cohort_summary = load_cohort_trial_outcomes(project_dir, cohort_id)
    eligibility = build_analysis_eligibility(
        outcomes,
        metric_id=metric_id,
        outcome_id=outcome_id,
        alignment=alignment,
        min_baseline_samples=min_baseline_samples,
        min_response_samples=min_response_samples,
    )
    cohort_hash = str(cohort_summary["cohort_hash"])
    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    eligibility_path = output_dir / f"{ELIGIBILITY_RECIPE_ID}.parquet"
    summary_path = (
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{ELIGIBILITY_RECIPE_ID}_summary.json"
    )
    marker_path = (
        project_dir
        / "Metadata"
        / f"{analysis_id}_{ELIGIBILITY_RECIPE_ID}_complete.json"
    )
    existing = [
        path
        for path in (eligibility_path, summary_path, marker_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"{ELIGIBILITY_RECIPE_ID} outputs already exist: {existing}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with artifact_staging(
        project_dir,
        prefix=f".{analysis_id}-{ELIGIBILITY_RECIPE_ID}-",
    ) as staging_root:
        staged_table = staging_root / eligibility_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        artifact = _write_parquet(staged_table, eligibility)
        artifact["path"] = str(eligibility_path)
        reason_counts = {
            str(key): int(value)
            for key, value in eligibility.loc[~eligibility["eligible"]]
            .groupby("ineligible_reason", observed=True)
            .size()
            .items()
        }
        summary = {
            "recipe": ELIGIBILITY_RECIPE_ID,
            "scientific_status": "analysis_eligibility_not_cohort_membership",
            "paper_approved": False,
            "analysis_id": analysis_id,
            "cohort_id": cohort_id,
            "cohort_hash": cohort_hash,
            "metric_id": metric_id,
            "outcome_id": outcome_id,
            "alignment": alignment,
            "min_baseline_samples": min_baseline_samples,
            "min_response_samples": min_response_samples,
            "row_count": len(eligibility),
            "eligible_count": int(eligibility["eligible"].sum()),
            "ineligible_reason_counts": reason_counts,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cohort_outcomes_sha256": cohort_summary["artifacts"]["outcomes"][
                "sha256"
            ],
            "artifact": artifact,
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": ELIGIBILITY_RECIPE_ID,
                "analysis_id": analysis_id,
                "cohort_id": cohort_id,
                "cohort_hash": cohort_hash,
                "eligibility_sha256": artifact["sha256"],
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        publish_transaction(
            (
                (staged_table, eligibility_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )
    return AnalysisEligibilityResult(
        analysis_id=analysis_id,
        cohort_id=cohort_id,
        cohort_hash=cohort_hash,
        eligibility_path=eligibility_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_count=len(eligibility),
        eligible_count=int(eligibility["eligible"].sum()),
    )
