"""Deterministic batch work manifests for multi-recording corrected/development runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.movement_state import (
    CandidateMetricSource,
    resolve_candidate_metric_source,
)
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.exceptions import ConfigurationError

RECIPE_ID = "batch-work-manifest-v1"
SelectionMode = Literal["all", "pending", "failed"]

BATCH_COLUMNS = (
    "recording_id",
    "stage",
    "recipe",
    "expected_output",
    "status",
    "attempt",
    "failure_reason",
    "disposition",
)


@dataclass(frozen=True)
class BatchStage:
    stage: str
    recipe: str
    expected_output: str
    marker_suffix: str


@dataclass(frozen=True)
class BatchWorkManifestResult:
    batch_id: str
    manifest_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int
    pending_count: int
    complete_count: int


def _validate_identifier(value: str, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ConfigurationError(
            f"{label} must use only letters, numbers, dot, underscore, or hyphen."
        )


def corrected_or_development_stages(
    source: CandidateMetricSource,
) -> tuple[BatchStage, ...]:
    """Return the ordered per-recording stages for one frozen candidate route."""
    stages: list[BatchStage] = []
    if source.requires_corrected_preprocess:
        stages.append(
            BatchStage(
                stage="corrected-preprocess",
                recipe="corrected-preprocess-v1",
                expected_output="frame_preprocessed_corrected-v1.parquet",
                marker_suffix="corrected-preprocess-v1_complete.json",
            )
        )
    stages.extend(
        [
            BatchStage(
                stage="activity-metrics",
                recipe=source.metric_recipe,
                expected_output=source.metrics_name,
                marker_suffix=source.metric_marker_suffix,
            ),
            BatchStage(
                stage="movement-state",
                recipe=source.movement_recipe,
                expected_output=source.movement_artifact_name,
                marker_suffix=source.movement_marker_suffix,
            ),
            BatchStage(
                stage="temporal-profiles",
                recipe=source.temporal_recipe,
                expected_output=source.temporal_artifact_name,
                marker_suffix=source.temporal_marker_suffix,
            ),
            BatchStage(
                stage="trial-outcomes",
                recipe=source.trial_recipe,
                expected_output=source.trial_outcomes_name,
                marker_suffix=source.trial_marker_suffix,
            ),
        ]
    )
    return tuple(stages)


def _marker_status(project_dir: Path, recording_id: str, stage: BatchStage) -> str:
    marker = project_dir / "Metadata" / f"{recording_id}_{stage.marker_suffix}"
    output = (
        project_dir / "Processed data" / recording_id / stage.expected_output
    )
    if marker.is_file() and output.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "failed"
        if (
            payload.get("status") == "complete"
            and payload.get("recipe") == stage.recipe
            and payload.get("recording_id") == recording_id
        ):
            return "complete"
        return "failed"
    if marker.exists() or output.exists():
        return "failed"
    return "pending"


def plan_batch_work(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    metric_recipe: str = "tail-candidate-corrected-v1",
    selection: SelectionMode = "all",
) -> pd.DataFrame:
    """Build a deterministic work table from frozen route stages and local markers."""
    if selection not in ("all", "pending", "failed"):
        raise ConfigurationError(
            "Batch selection must be one of: all, pending, failed."
        )
    recording_ids = tuple(dict.fromkeys(recording_ids))
    if not recording_ids:
        raise ConfigurationError("At least one recording ID is required.")
    for recording_id in recording_ids:
        _validate_identifier(recording_id, "Recording ID")
    project_dir = project_dir.resolve()
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    stages = corrected_or_development_stages(source)
    rows: list[dict[str, object]] = []
    for recording_id in recording_ids:
        for stage in stages:
            status = _marker_status(project_dir, recording_id, stage)
            rows.append(
                {
                    "recording_id": recording_id,
                    "stage": stage.stage,
                    "recipe": stage.recipe,
                    "expected_output": stage.expected_output,
                    "status": status,
                    "attempt": 0,
                    "failure_reason": (
                        "incomplete or invalid stage artifacts"
                        if status == "failed"
                        else ""
                    ),
                    "disposition": "",
                }
            )
    frame = pd.DataFrame(rows, columns=list(BATCH_COLUMNS))
    if selection == "pending":
        frame = frame.loc[frame["status"] == "pending"].reset_index(drop=True)
    elif selection == "failed":
        frame = frame.loc[frame["status"] == "failed"].reset_index(drop=True)
    return frame


def write_batch_work_manifest(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    batch_id: str,
    metric_recipe: str = "tail-candidate-corrected-v1",
    selection: SelectionMode = "all",
    overwrite: bool = False,
) -> BatchWorkManifestResult:
    """Publish a batch work manifest plus summary for coverage accounting."""
    _validate_identifier(batch_id, "Batch ID")
    project_dir = project_dir.resolve()
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    work = plan_batch_work(
        project_dir,
        recording_ids,
        metric_recipe=metric_recipe,
        selection=selection,
    )
    output_dir = project_dir / "Processed data" / "Batches" / batch_id
    manifest_path = output_dir / f"{RECIPE_ID}.parquet"
    summary_path = (
        project_dir
        / "Quality checks"
        / "Batches"
        / batch_id
        / f"{RECIPE_ID}_summary.json"
    )
    marker_path = project_dir / "Metadata" / f"{batch_id}_{RECIPE_ID}_complete.json"
    existing = [
        path for path in (manifest_path, summary_path, marker_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{batch_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_manifest = staging_root / manifest_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        table = pa.Table.from_pandas(work, preserve_index=False, safe=True)
        pq.write_table(
            table,
            staged_manifest,
            compression="zstd",
            write_statistics=True,
        )
        digest = sha256_file(staged_manifest)
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": source.scientific_status,
            "paper_approved": False,
            "batch_id": batch_id,
            "metric_recipe": source.metric_recipe,
            "runner_recipe": source.runner_recipe,
            "selection": selection,
            "recording_ids": list(dict.fromkeys(recording_ids)),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(work)),
            "status_counts": {
                str(status): int(count)
                for status, count in work["status"].value_counts().items()
            },
            "artifact": {
                "path": str(manifest_path),
                "sha256": digest,
                "rows": int(len(work)),
                "compression": "zstd",
                "compression_lossless": True,
            },
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "batch_id": batch_id,
                "manifest_sha256": digest,
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        publish_transaction(
            (
                (staged_manifest, manifest_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    status_counts = work["status"].value_counts()
    return BatchWorkManifestResult(
        batch_id=batch_id,
        manifest_path=manifest_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_count=int(len(work)),
        pending_count=int(status_counts.get("pending", 0)),
        complete_count=int(status_counts.get("complete", 0)),
    )


@dataclass(frozen=True)
class BatchExecuteResult:
    batch_id: str
    metric_recipe: str
    selection: SelectionMode
    recording_ids: tuple[str, ...]
    analysis_id: str | None
    runner_manifest_path: Path | None
    before_pending_count: int
    before_failed_count: int
    after_pending_count: int
    after_failed_count: int
    after_complete_count: int
    refreshed_manifest_path: Path


def execute_batch_work(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    batch_id: str,
    metric_recipe: str = "tail-candidate-corrected-v1",
    selection: SelectionMode = "pending",
    analysis_id: str | None = None,
    experiment_name: str = "allDelay",
    batch_size: int = 250_000,
    overwrite_failed: bool = True,
) -> BatchExecuteResult:
    """Execute pending or failed per-recording stages, then refresh the work manifest.

    Pending stages resume without overwrite. Failed stages rebuild the affected
    recordings when ``overwrite_failed`` is true. Cohort comparison is always
    refreshed through the candidate runner for the selected recordings.
    """
    if selection not in ("pending", "failed", "all"):
        raise ConfigurationError(
            "Batch execute selection must be one of: pending, failed, all."
        )
    _validate_identifier(batch_id, "Batch ID")
    recording_ids = tuple(dict.fromkeys(recording_ids))
    if not recording_ids:
        raise ConfigurationError("At least one recording ID is required.")
    for recording_id in recording_ids:
        _validate_identifier(recording_id, "Recording ID")
    project_dir = project_dir.resolve()
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    before = plan_batch_work(
        project_dir,
        recording_ids,
        metric_recipe=metric_recipe,
        selection="all",
    )
    before_pending = int((before["status"] == "pending").sum())
    before_failed = int((before["status"] == "failed").sum())

    if selection == "all":
        target_ids = recording_ids
        overwrite = False
    elif selection == "pending":
        target_ids = tuple(
            before.loc[before["status"] == "pending", "recording_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        overwrite = False
    else:
        target_ids = tuple(
            before.loc[before["status"] == "failed", "recording_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        overwrite = bool(overwrite_failed)

    runner_manifest: Path | None = None
    resolved_analysis_id = analysis_id
    if target_ids:
        from classical_conditioning.analysis.candidate_runner import (
            run_candidate_development_pipeline,
        )

        if resolved_analysis_id is None:
            resolved_analysis_id = f"{batch_id}-run"
        _validate_identifier(resolved_analysis_id, "Analysis ID")
        runner = run_candidate_development_pipeline(
            project_dir,
            target_ids,
            analysis_id=resolved_analysis_id,
            experiment_name=experiment_name,
            batch_size=batch_size,
            overwrite=overwrite,
            metric_recipe=source.metric_recipe,
            runner_recipe=source.runner_recipe,
            continue_on_error=True,
        )
        runner_manifest = runner.manifest_path

    refreshed = write_batch_work_manifest(
        project_dir,
        recording_ids,
        batch_id=batch_id,
        metric_recipe=metric_recipe,
        selection="all",
        overwrite=True,
    )
    after = plan_batch_work(
        project_dir,
        recording_ids,
        metric_recipe=metric_recipe,
        selection="all",
    )
    return BatchExecuteResult(
        batch_id=batch_id,
        metric_recipe=metric_recipe,
        selection=selection,
        recording_ids=recording_ids,
        analysis_id=resolved_analysis_id,
        runner_manifest_path=runner_manifest,
        before_pending_count=before_pending,
        before_failed_count=before_failed,
        after_pending_count=int((after["status"] == "pending").sum()),
        after_failed_count=int((after["status"] == "failed").sum()),
        after_complete_count=int((after["status"] == "complete").sum()),
        refreshed_manifest_path=refreshed.manifest_path,
    )
