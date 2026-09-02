"""Descriptive frozen-legacy versus candidate trial-outcome comparison."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.legacy_normalized_vigor import (
    RECIPE_ID as LEGACY_RECIPE_ID,
)
from classical_conditioning.analysis.trial_outcomes import (
    METRIC_IDS,
    RECIPE_ID as CANDIDATE_RECIPE_ID,
    verify_candidate_trial_outcomes,
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
    SchemaValidationError,
)

RECIPE_ID = "legacy-candidate-outcome-comparison-v1"


@dataclass(frozen=True)
class OutcomeComparisonResult:
    recording_id: str
    matched_path: Path
    coverage_path: Path
    summary_path: Path
    completion_marker_path: Path
    matched_row_count: int


def compare_legacy_candidate_outcomes(
    legacy: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare response/baseline ratios only where trial identities overlap."""
    legacy_required = {
        "Fish",
        "Trial type",
        "Trial number",
        "Mean 15 s before",
        "Mean CR",
        "Normalized vigor",
    }
    candidate_required = {
        "fish_id",
        "alignment",
        "trial_number",
        "metric_id",
        "baseline_total_activity",
        "response_total_activity",
    }
    missing_legacy = legacy_required.difference(legacy.columns)
    missing_candidate = candidate_required.difference(candidate.columns)
    if missing_legacy or missing_candidate:
        raise SchemaValidationError(
            "Outcome comparison inputs are missing columns: "
            f"legacy={sorted(missing_legacy)}, "
            f"candidate={sorted(missing_candidate)}"
        )
    legacy = legacy.copy()
    candidate = candidate.copy()
    legacy["Fish"] = legacy["Fish"].astype(str)
    legacy["Trial type"] = legacy["Trial type"].astype(str)
    candidate["fish_id"] = candidate["fish_id"].astype(str)
    candidate["alignment"] = candidate["alignment"].astype(str)
    legacy["_fish_key"] = legacy["Fish"].str.casefold()
    candidate["_fish_key"] = candidate["fish_id"].str.casefold()
    legacy_key = ["_fish_key", "Trial type", "Trial number"]
    candidate_key = ["_fish_key", "alignment", "trial_number", "metric_id"]
    if legacy.duplicated(legacy_key).any():
        raise SchemaValidationError("Legacy trial identities are not unique.")
    if candidate.duplicated(candidate_key).any():
        raise SchemaValidationError("Candidate trial-metric identities are not unique.")
    metric_ids = set(candidate["metric_id"].dropna().astype(str))
    expected_metrics = set(METRIC_IDS.values())
    if metric_ids != expected_metrics:
        raise SchemaValidationError(
            "Candidate comparison requires all five metrics; "
            f"found={sorted(metric_ids)}"
        )

    matched = candidate.merge(
        legacy,
        how="inner",
        left_on=["_fish_key", "alignment", "trial_number"],
        right_on=legacy_key,
        validate="many_to_one",
    )
    if matched.empty and not legacy.empty and not candidate.empty:
        raise SchemaValidationError(
            "Legacy and candidate outcomes have no overlapping trial identities."
        )
    baseline = matched["baseline_total_activity"].to_numpy(dtype=float)
    response = matched["response_total_activity"].to_numpy(dtype=float)
    ratio = np.divide(
        response,
        baseline,
        out=np.full(len(matched), np.nan),
        where=np.isfinite(baseline) & np.isfinite(response) & (baseline != 0),
    )
    matched["candidate_response_baseline_ratio"] = ratio
    matched["candidate_minus_legacy_ratio"] = (
        ratio - matched["Normalized vigor"].to_numpy(dtype=float)
    )
    matched = matched[
        [
            "fish_id",
            "alignment",
            "trial_number",
            "metric_id",
            "Mean 15 s before",
            "Mean CR",
            "Normalized vigor",
            "baseline_total_activity",
            "response_total_activity",
            "candidate_response_baseline_ratio",
            "candidate_minus_legacy_ratio",
        ]
    ].rename(
        columns={
            "Mean 15 s before": "legacy_baseline_mean",
            "Mean CR": "legacy_response_mean",
            "Normalized vigor": "legacy_response_baseline_ratio",
        }
    )

    coverage_rows: list[dict[str, Any]] = []
    for alignment in ("CS", "US"):
        legacy_alignment = legacy[legacy["Trial type"] == alignment]
        candidate_alignment = candidate[candidate["alignment"] == alignment]
        legacy_trials = set(
            zip(
                legacy_alignment["_fish_key"],
                legacy_alignment["Trial number"],
                strict=True,
            )
        )
        for metric_id in sorted(expected_metrics):
            metric = candidate_alignment[
                candidate_alignment["metric_id"] == metric_id
            ]
            candidate_trials = set(
                zip(metric["_fish_key"], metric["trial_number"], strict=True)
            )
            overlap = legacy_trials & candidate_trials
            differences = matched.loc[
                (matched["alignment"] == alignment)
                & (matched["metric_id"] == metric_id),
                "candidate_minus_legacy_ratio",
            ].to_numpy(dtype=float)
            finite = differences[np.isfinite(differences)]
            coverage_rows.append(
                {
                    "alignment": alignment,
                    "metric_id": metric_id,
                    "legacy_trial_count": len(legacy_trials),
                    "candidate_trial_count": len(candidate_trials),
                    "overlap_trial_count": len(overlap),
                    "legacy_only_trial_count": len(legacy_trials - candidate_trials),
                    "candidate_only_trial_count": len(candidate_trials - legacy_trials),
                    "finite_difference_count": int(finite.size),
                    "mean_candidate_minus_legacy_ratio": (
                        float(np.mean(finite)) if finite.size else np.nan
                    ),
                    "median_absolute_ratio_difference": (
                        float(np.median(np.abs(finite))) if finite.size else np.nan
                    ),
                }
            )
    return matched, pd.DataFrame(coverage_rows)


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return {
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": len(table.column_names),
        "compression": "zstd",
        "compression_lossless": True,
    }


def build_legacy_candidate_outcome_comparison(
    project_dir: Path,
    recording_id: str,
    *,
    overwrite: bool = False,
) -> OutcomeComparisonResult:
    """Publish an authenticated, non-inferential per-recording comparison."""
    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    legacy_source = verify_completed_parquet_set(
        {
            alignment: source_dir / f"{LEGACY_RECIPE_ID}_{alignment}.parquet"
            for alignment in ("CS", "US")
        },
        project_dir
        / "Quality checks"
        / recording_id
        / f"{LEGACY_RECIPE_ID}_summary.json",
        project_dir
        / "Metadata"
        / f"{recording_id}_{LEGACY_RECIPE_ID}_complete.json",
        recipe=LEGACY_RECIPE_ID,
        recording_id=recording_id,
    )
    candidate_source = verify_candidate_trial_outcomes(project_dir, recording_id)
    legacy = pd.concat(
        [
            pq.read_table(legacy_source.data_paths[alignment]).to_pandas()
            for alignment in ("CS", "US")
        ],
        ignore_index=True,
    )
    candidate = pq.read_table(candidate_source.data_paths["outcomes"]).to_pandas()
    matched, coverage = compare_legacy_candidate_outcomes(legacy, candidate)

    matched_path = source_dir / f"{RECIPE_ID}_matched.parquet"
    coverage_path = source_dir / f"{RECIPE_ID}_coverage.parquet"
    summary_path = (
        project_dir / "Quality checks" / recording_id / f"{RECIPE_ID}_summary.json"
    )
    marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{RECIPE_ID}_complete.json"
    )
    outputs = (matched_path, coverage_path, summary_path, marker_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")

    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_matched = staging_root / matched_path.name
        staged_coverage = staging_root / coverage_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        artifacts = {
            "matched": _write_parquet(staged_matched, matched),
            "coverage": _write_parquet(staged_coverage, coverage),
        }
        artifacts["matched"]["path"] = str(matched_path)
        artifacts["coverage"]["path"] = str(coverage_path)
        write_json_atomic(
            staged_summary,
            {
                "recipe": RECIPE_ID,
                "scientific_status": "descriptive_difference_report",
                "paper_approved": False,
                "recording_id": recording_id,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {
                    "legacy": {
                        "recipe": LEGACY_RECIPE_ID,
                        "summary_sha256": sha256_file(legacy_source.summary_path),
                        "artifact_sha256": legacy_source.marker["artifact_sha256"],
                    },
                    "candidate": {
                        "recipe": CANDIDATE_RECIPE_ID,
                        "summary_sha256": sha256_file(candidate_source.summary_path),
                        "artifact_sha256": candidate_source.marker["artifact_sha256"],
                    },
                },
                "artifacts": artifacts,
                "interpretation": {
                    "comparison": (
                        "Candidate response/baseline activity ratio minus frozen "
                        "legacy normalized vigor on identical fish/alignment/trial IDs."
                    ),
                    "inference_performed": False,
                    "metric_selection_performed": False,
                    "limitations": [
                        "Legacy rest is missing while candidate rest is zero.",
                        "Legacy and candidate preprocessing and window boundaries differ.",
                        "Ratio differences are descriptive and do not establish superiority.",
                    ],
                },
            },
        )
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "recording_id": recording_id,
                "artifact_sha256": {
                    name: record["sha256"] for name, record in artifacts.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        for source in (legacy_source, candidate_source):
            for name, path in source.data_paths.items():
                stat = path.stat()
                if (
                    source.data_states[name] != (stat.st_size, stat.st_mtime_ns)
                    or sha256_file(path) != source.marker["artifact_sha256"][name]
                ):
                    raise ArtifactIntegrityError(
                        f"Outcome comparison input changed during execution: {path}"
                    )
        publish_transaction(
            (
                (staged_matched, matched_path),
                (staged_coverage, coverage_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )
    return OutcomeComparisonResult(
        recording_id=recording_id,
        matched_path=matched_path,
        coverage_path=coverage_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        matched_row_count=len(matched),
    )
