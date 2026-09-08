"""Authenticated orchestration for non-approved candidate-development routes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from classical_conditioning.analysis.metric_comparison import (
    build_candidate_metric_comparison,
)
from classical_conditioning.analysis.movement_state import (
    CandidateMetricSource,
    resolve_candidate_metric_source,
    build_candidate_movement_state,
)
from classical_conditioning.analysis.temporal_profiles import (
    build_candidate_temporal_profiles,
)
from classical_conditioning.analysis.trial_outcomes import (
    build_candidate_trial_outcomes,
    verify_candidate_trial_outcomes,
)
from classical_conditioning.artifacts import (
    load_and_verify_source_manifest,
    sha256_file,
    verify_completed_analysis_parquet_set,
    write_json_atomic,
)
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
)
from classical_conditioning.preprocessing.candidates_corrected_v1 import (
    build_candidate_activity_metrics_from_corrected,
)
from classical_conditioning.preprocessing.candidates_v1 import (
    build_candidate_activity_metrics,
)
from classical_conditioning.preprocessing.corrected_v1 import (
    RECIPE_ID as CORRECTED_PREPROCESS_RECIPE_ID,
    build_corrected_preprocessing,
)

RUNNER_RECIPE_ID = "candidate-development-runner-v1"
CORRECTED_RUNNER_RECIPE_ID = "candidate-corrected-runner-v1"
METRIC_RECIPE_ID = "tail-candidate-development-v1"
MOVEMENT_RECIPE_ID = "movement-candidate-v2"
TEMPORAL_RECIPE_ID = "candidate-temporal-outcomes-v3"


@dataclass(frozen=True)
class CandidateRunnerResult:
    analysis_id: str
    recording_ids: tuple[str, ...]
    manifest_path: Path
    step_status: dict[str, dict[str, str]]


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )


def _verify_single_artifact(
    *,
    data_path: Path,
    summary_path: Path,
    marker_path: Path,
    recipe: str,
    recording_id: str,
    marker_hash_key: str,
    experiment_name: str | None = None,
) -> str:
    missing = [
        path
        for path in (data_path, summary_path, marker_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing completed {recipe} artifacts: {missing}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError(
            f"Completed {recipe} metadata is invalid JSON: {error}"
        ) from error
    digest = sha256_file(data_path)
    if (
        summary.get("recipe") != recipe
        or summary.get("recording_id") != recording_id
        or summary.get("artifact", {}).get("sha256") != digest
        or marker.get("status") != "complete"
        or marker.get("recipe") != recipe
        or marker.get("recording_id") != recording_id
        or marker.get(marker_hash_key) != digest
        or marker.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ArtifactIntegrityError(
            f"Completed {recipe} lineage is invalid for {recording_id}."
        )
    if experiment_name is not None and summary.get("experiment") != experiment_name:
        raise ArtifactIntegrityError(
            f"Completed {recipe} experiment differs from the requested experiment."
        )
    return sha256_file(marker_path)


def _verify_corrected_preprocess(project_dir: Path, recording_id: str) -> str:
    return _verify_single_artifact(
        data_path=project_dir
        / "Processed data"
        / recording_id
        / "frame_preprocessed_corrected-v1.parquet",
        summary_path=project_dir
        / "Quality checks"
        / recording_id
        / "corrected-v1_preprocessing_summary.json",
        marker_path=project_dir
        / "Metadata"
        / f"{recording_id}_corrected-preprocess-v1_complete.json",
        recipe=CORRECTED_PREPROCESS_RECIPE_ID,
        recording_id=recording_id,
        marker_hash_key="frames_sha256",
    )


def _verify_metrics(
    project_dir: Path,
    recording_id: str,
    source: CandidateMetricSource,
) -> str:
    marker_digest = _verify_single_artifact(
        data_path=project_dir
        / "Processed data"
        / recording_id
        / source.metrics_name,
        summary_path=project_dir
        / "Quality checks"
        / recording_id
        / source.metric_summary_name,
        marker_path=project_dir
        / "Metadata"
        / f"{recording_id}_{source.metric_marker_suffix}",
        recipe=source.metric_recipe,
        recording_id=recording_id,
        marker_hash_key="metrics_sha256",
    )
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.metric_summary_name
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if source.requires_corrected_preprocess:
        corrected_marker = json.loads(
            (
                project_dir
                / "Metadata"
                / f"{recording_id}_corrected-preprocess-v1_complete.json"
            ).read_text(encoding="utf-8")
        )
        if (
            summary.get("input_artifacts", {})
            .get("corrected_preprocess", {})
            .get("sha256")
            != corrected_marker.get("frames_sha256")
        ):
            raise ArtifactIntegrityError(
                f"Candidate metrics use stale corrected preprocess for {recording_id}."
            )
        return marker_digest

    _, source_artifacts, _ = load_and_verify_source_manifest(
        project_dir,
        recording_id,
    )
    if any(
        summary.get("input_artifacts", {}).get(kind, {}).get("sha256")
        != artifact["sha256"]
        for kind, artifact in source_artifacts.items()
    ):
        raise ArtifactIntegrityError(
            f"Candidate metrics use stale intake lineage for {recording_id}."
        )
    return marker_digest


def _verify_movement(
    project_dir: Path,
    recording_id: str,
    source: CandidateMetricSource | None = None,
) -> str:
    route = source or resolve_candidate_metric_source()
    marker_digest = _verify_single_artifact(
        data_path=project_dir
        / "Processed data"
        / recording_id
        / route.movement_artifact_name,
        summary_path=project_dir
        / "Quality checks"
        / recording_id
        / route.movement_summary_name,
        marker_path=project_dir
        / "Metadata"
        / f"{recording_id}_{route.movement_marker_suffix}",
        recipe=route.movement_recipe,
        recording_id=recording_id,
        marker_hash_key="movement_sha256",
    )
    candidate_marker = json.loads(
        (
            project_dir
            / "Metadata"
            / f"{recording_id}_{route.metric_marker_suffix}"
        ).read_text(encoding="utf-8")
    )
    summary = json.loads(
        (
            project_dir
            / "Quality checks"
            / recording_id
            / route.movement_summary_name
        ).read_text(encoding="utf-8")
    )
    if (
        summary.get("inputs", {})
        .get("candidate_metrics", {})
        .get("sha256")
        != candidate_marker.get("metrics_sha256")
    ):
        raise ArtifactIntegrityError(
            f"Movement state uses stale candidate metrics for {recording_id}."
        )
    return marker_digest


def _verify_temporal(
    project_dir: Path,
    recording_id: str,
    experiment_name: str,
    source: CandidateMetricSource,
) -> str:
    marker_digest = _verify_single_artifact(
        data_path=project_dir
        / "Processed data"
        / recording_id
        / source.temporal_artifact_name,
        summary_path=project_dir
        / "Quality checks"
        / recording_id
        / source.temporal_summary_name,
        marker_path=project_dir
        / "Metadata"
        / f"{recording_id}_{source.temporal_marker_suffix}",
        recipe=source.temporal_recipe,
        recording_id=recording_id,
        marker_hash_key="profiles_sha256",
        experiment_name=experiment_name,
    )
    candidate_marker = json.loads(
        (
            project_dir
            / "Metadata"
            / f"{recording_id}_{source.metric_marker_suffix}"
        ).read_text(encoding="utf-8")
    )
    movement_marker = json.loads(
        (
            project_dir
            / "Metadata"
            / f"{recording_id}_{source.movement_marker_suffix}"
        ).read_text(encoding="utf-8")
    )
    summary = json.loads(
        (
            project_dir
            / "Quality checks"
            / recording_id
            / source.temporal_summary_name
        ).read_text(encoding="utf-8")
    )
    if (
        summary.get("inputs", {}).get("candidate_frames", {}).get("sha256")
        != candidate_marker.get("metrics_sha256")
        or summary.get("inputs", {}).get("movement_state", {}).get("sha256")
        != movement_marker.get("movement_sha256")
    ):
        raise ArtifactIntegrityError(
            f"Temporal outcomes use stale candidate inputs for {recording_id}."
        )
    return marker_digest


def _verify_comparison(
    project_dir: Path,
    analysis_id: str,
    recording_ids: tuple[str, ...],
    source: CandidateMetricSource,
) -> str:
    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    marker_path = (
        project_dir
        / "Metadata"
        / f"{analysis_id}_{source.comparison_recipe}_complete.json"
    )
    verified = verify_completed_analysis_parquet_set(
        {
            "recording_summary": (
                output_dir / f"{source.comparison_recipe}_recording_summary.parquet"
            ),
            "cohort_summary": (
                output_dir / f"{source.comparison_recipe}_cohort_summary.parquet"
            ),
        },
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{source.comparison_recipe}_summary.json",
        marker_path,
        recipe=source.comparison_recipe,
        analysis_id=analysis_id,
        recording_ids=recording_ids,
    )
    for recording_id in recording_ids:
        temporal_path = (
            project_dir
            / "Processed data"
            / recording_id
            / source.temporal_artifact_name
        )
        source_meta = verified.summary.get("inputs", {}).get(recording_id, {})
        if (
            source_meta.get("recipe") != source.temporal_recipe
            or source_meta.get("sha256") != sha256_file(temporal_path)
        ):
            raise ArtifactIntegrityError(
                "Five-metric comparison uses stale temporal outcomes for "
                f"{recording_id}."
            )
    return sha256_file(marker_path)


def _verify_trial_outcomes(
    project_dir: Path,
    recording_id: str,
    source: CandidateMetricSource,
) -> str:
    verified = verify_candidate_trial_outcomes(
        project_dir,
        recording_id,
        metric_recipe=source.metric_recipe,
    )
    return sha256_file(verified.marker_path)


def _run_recording_candidate_stages(
    project_dir: Path,
    recording_id: str,
    *,
    experiment_name: str,
    batch_size: int,
    overwrite: bool,
    route: CandidateMetricSource,
    steps: dict[str, dict[str, str]],
    lineage: dict[str, str],
    progress: "PipelineProgress | None" = None,
) -> None:
    from classical_conditioning.progress import default_progress

    progress = progress or default_progress(enabled=False)
    if route.requires_corrected_preprocess:
        preprocess_marker = (
            project_dir
            / "Metadata"
            / f"{recording_id}_corrected-preprocess-v1_complete.json"
        )
        if not overwrite and preprocess_marker.exists():
            progress.step("corrected-preprocess", status="existing")
            state = "existing"
        else:
            with progress.step_timer("corrected-preprocess"):
                build_corrected_preprocessing(
                    project_dir,
                    recording_id,
                    batch_size=batch_size,
                    overwrite=overwrite,
                )
            state = "completed"
        lineage[f"{recording_id}:corrected-preprocess"] = (
            _verify_corrected_preprocess(project_dir, recording_id)
        )
        steps[recording_id]["corrected-preprocess"] = state

    metric_marker = (
        project_dir / "Metadata" / f"{recording_id}_{route.metric_marker_suffix}"
    )
    if not overwrite and metric_marker.exists():
        progress.step("activity-metrics", status="existing")
        state = "existing"
    else:
        with progress.step_timer("activity-metrics"):
            if route.requires_corrected_preprocess:
                build_candidate_activity_metrics_from_corrected(
                    project_dir,
                    recording_id,
                    batch_size=batch_size,
                    overwrite=overwrite,
                )
            else:
                build_candidate_activity_metrics(
                    project_dir,
                    recording_id,
                    batch_size=batch_size,
                    overwrite=overwrite,
                )
        state = "completed"
    lineage[f"{recording_id}:activity-metrics"] = _verify_metrics(
        project_dir,
        recording_id,
        route,
    )
    steps[recording_id]["activity-metrics"] = state

    movement_marker = (
        project_dir / "Metadata" / f"{recording_id}_{route.movement_marker_suffix}"
    )
    if not overwrite and movement_marker.exists():
        progress.step("movement-state", status="existing")
        state = "existing"
    else:
        with progress.step_timer("movement-state"):
            build_candidate_movement_state(
                project_dir,
                recording_id,
                metric_recipe=route.metric_recipe,
                overwrite=overwrite,
            )
        state = "completed"
    lineage[f"{recording_id}:movement-state"] = _verify_movement(
        project_dir,
        recording_id,
        route,
    )
    steps[recording_id]["movement-state"] = state

    temporal_marker = (
        project_dir / "Metadata" / f"{recording_id}_{route.temporal_marker_suffix}"
    )
    if not overwrite and temporal_marker.exists():
        progress.step("temporal-profiles", status="existing")
        state = "existing"
    else:
        with progress.step_timer("temporal-profiles"):
            build_candidate_temporal_profiles(
                project_dir,
                recording_id,
                experiment_name=experiment_name,
                metric_recipe=route.metric_recipe,
                overwrite=overwrite,
            )
        state = "completed"
    lineage[f"{recording_id}:temporal-profiles"] = _verify_temporal(
        project_dir,
        recording_id,
        experiment_name,
        route,
    )
    steps[recording_id]["temporal-profiles"] = state

    trial_marker = (
        project_dir / "Metadata" / f"{recording_id}_{route.trial_marker_suffix}"
    )
    if not overwrite and trial_marker.exists():
        progress.step("trial-outcomes", status="existing")
        state = "existing"
    else:
        with progress.step_timer("trial-outcomes"):
            build_candidate_trial_outcomes(
                project_dir,
                recording_id,
                experiment_name=experiment_name,
                metric_recipe=route.metric_recipe,
                overwrite=overwrite,
            )
        state = "completed"
    lineage[f"{recording_id}:trial-outcomes"] = _verify_trial_outcomes(
        project_dir,
        recording_id,
        route,
    )
    steps[recording_id]["trial-outcomes"] = state


def run_candidate_development_pipeline(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    experiment_name: str = "allDelay",
    batch_size: int = 250_000,
    overwrite: bool = False,
    metric_recipe: str | None = None,
    runner_recipe: str | None = None,
    continue_on_error: bool = False,
    progress: "PipelineProgress | None" = None,
) -> CandidateRunnerResult:
    """Run candidate metrics through identical non-approved outcomes."""
    from classical_conditioning.progress import default_progress

    progress = progress or default_progress(enabled=False)
    _validate_analysis_id(analysis_id)
    if batch_size <= 0:
        raise ConfigurationError("Candidate metric batch size must be positive.")
    recording_ids = tuple(dict.fromkeys(recording_ids))
    if not recording_ids:
        raise ConfigurationError("At least one recording ID is required.")
    route = resolve_candidate_metric_source(
        metric_recipe=metric_recipe,
        runner_recipe=runner_recipe,
    )
    project_dir = project_dir.resolve()
    manifest_path = (
        project_dir
        / "Metadata"
        / f"{analysis_id}_{route.runner_recipe}_manifest.json"
    )
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ArtifactIntegrityError(
                f"Existing candidate runner manifest is invalid: {manifest_path}"
            ) from error
        identity = {
            "recipe": route.runner_recipe,
            "analysis_id": analysis_id,
            "experiment_name": experiment_name,
            "recording_ids": list(recording_ids),
        }
        previous_metric = previous.get(
            "metric_recipe",
            "tail-candidate-development-v1",
        )
        if (
            any(previous.get(key) != value for key, value in identity.items())
            or previous_metric != route.metric_recipe
        ):
            raise ArtifactIntegrityError(
                "Existing candidate runner identity differs from the requested run. "
                "Use a new analysis ID."
            )

    steps: dict[str, dict[str, str]] = {}
    lineage: dict[str, str] = {}
    successful: list[str] = []
    total = len(recording_ids)
    for index, recording_id in enumerate(
        progress.iter_items(recording_ids, description="candidate recordings"),
        start=1,
    ):
        steps[recording_id] = {}
        progress.info(f"recording {recording_id} [{index}/{total}]")
        try:
            _run_recording_candidate_stages(
                project_dir,
                recording_id,
                experiment_name=experiment_name,
                batch_size=batch_size,
                overwrite=overwrite,
                route=route,
                steps=steps,
                lineage=lineage,
                progress=progress,
            )
            successful.append(recording_id)
            progress.item_done(
                index, total, recording_id, status="completed"
            )
        except Exception as error:
            progress.item_done(
                index, total, recording_id, status=f"failed: {error}"
            )
            if not continue_on_error:
                raise
            steps[recording_id]["failed"] = str(error)

    if not successful:
        raise ConfigurationError(
            "No recordings completed candidate stages; cohort comparison was skipped."
        )

    comparison_marker = (
        project_dir
        / "Metadata"
        / f"{analysis_id}_{route.comparison_recipe}_complete.json"
    )
    if not overwrite and comparison_marker.exists():
        progress.step("cohort metric comparison", status="existing")
        state = "existing"
    else:
        with progress.step_timer("cohort metric comparison"):
            build_candidate_metric_comparison(
                project_dir,
                successful,
                analysis_id=analysis_id,
                experiment_name=experiment_name,
                metric_recipe=route.metric_recipe,
                overwrite=overwrite,
            )
        state = "completed"
    lineage["cohort:five-metric-comparison"] = _verify_comparison(
        project_dir,
        analysis_id,
        tuple(successful),
        route,
    )
    steps["cohort"] = {"five-metric-comparison": state}

    write_json_atomic(
        manifest_path,
        {
            "recipe": route.runner_recipe,
            "scientific_status": route.scientific_status,
            "paper_approved": False,
            "analysis_id": analysis_id,
            "experiment_name": experiment_name,
            "recording_ids": list(recording_ids),
            "metric_recipe": route.metric_recipe,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "steps": steps,
            "completion_marker_sha256": lineage,
            "blocked_next_steps": [
                "reviewed trace annotations",
                "prespecified metric scorecard",
                "independent validation partition",
                "approved primary cohort",
                "approved confirmatory statistical model",
            ],
        },
    )
    return CandidateRunnerResult(
        analysis_id=analysis_id,
        recording_ids=recording_ids,
        manifest_path=manifest_path,
        step_status=steps,
    )
