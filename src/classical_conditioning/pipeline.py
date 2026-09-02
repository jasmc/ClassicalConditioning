"""Config-driven orchestration for intake and analysis routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from classical_conditioning.analysis.candidate_runner import (
    run_candidate_development_pipeline,
)
from classical_conditioning.analysis.legacy_runner import run_legacy_analysis_pipeline
from classical_conditioning.analysis.movement_state import RUNNER_RECIPE_TO_METRIC_SOURCE
from classical_conditioning.artifacts import write_json_atomic
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.figures.export import FigureMode
from classical_conditioning.figures.metric_comparison import (
    build_metric_comparison_figure,
)
from classical_conditioning.intake import discover_recordings, intake_recordings
from classical_conditioning.inventory import write_recording_inventory
from classical_conditioning.paths import condition_from_recording_name
from classical_conditioning.preprocessing.legacy_v1 import preprocess_legacy_recording
from classical_conditioning.run_config import PipelineRunConfig, pipeline_config_to_dict


@dataclass(frozen=True)
class PipelineRunResult:
    recording_ids: tuple[str, ...]
    intake_completed: tuple[str, ...] = ()
    intake_skipped: tuple[str, ...] = ()
    intake_failed: tuple[tuple[str, str], ...] = ()
    legacy_preprocess: dict[str, str] = field(default_factory=dict)
    legacy_runner_status: str | None = None
    candidate_runner_status: str | None = None
    figure_paths: tuple[Path, ...] = ()
    summary_path: Path | None = None


def resolve_pipeline_recording_ids(config: PipelineRunConfig) -> tuple[str, ...]:
    """Discover recording IDs from raw_dir using optional filters."""
    if config.recording_ids is not None:
        return config.recording_ids

    keep = (
        {token.strip().lower() for token in config.keep_conditions}
        if config.keep_conditions
        else None
    )
    selected: list[str] = []
    for sources in discover_recordings(config.raw_dir):
        if keep is not None:
            condition = condition_from_recording_name(sources.recording_name)
            if condition not in keep:
                continue
        selected.append(sources.recording_id)
    if not selected:
        raise ConfigurationError(
            "No complete triplets matched the requested filters under raw_dir."
        )
    return tuple(dict.fromkeys(selected))


def run_pipeline(config: PipelineRunConfig) -> PipelineRunResult:
    """Execute a relocatable pipeline from a validated run configuration."""
    recording_ids = resolve_pipeline_recording_ids(config)
    config.save_dir.mkdir(parents=True, exist_ok=True)

    intake_completed: tuple[str, ...] = ()
    intake_skipped: tuple[str, ...] = ()
    intake_failed: tuple[tuple[str, str], ...] = ()
    legacy_preprocess: dict[str, str] = {}
    legacy_runner_status: str | None = None
    candidate_runner_status: str | None = None
    figure_paths: list[Path] = []

    if config.run_inventory:
        inventory_path = (
            config.save_dir / "Metadata" / "recording_inventory.json"
        )
        write_recording_inventory(
            config.raw_dir,
            inventory_path,
            hash_files=True,
            overwrite=config.overwrite,
        )

    if config.run_intake:
        intake_result = intake_recordings(
            config.raw_dir,
            config.save_dir,
            keep_conditions=config.keep_conditions,
            recording_ids=recording_ids,
            overwrite=config.overwrite,
        )
        intake_completed = intake_result.completed
        intake_skipped = intake_result.skipped
        intake_failed = intake_result.failed
        if intake_failed and not config.continue_on_error:
            raise ConfigurationError(
                "Intake failed for one or more recordings; see pipeline summary."
            )
        active_ids = tuple(
            recording_id
            for recording_id in recording_ids
            if recording_id in set(intake_completed).union(intake_skipped)
        )
        if not active_ids:
            raise ConfigurationError(
                "No recordings are available after intake; pipeline stopped."
            )
        recording_ids = active_ids

    if "legacy" in config.routes:
        for recording_id in recording_ids:
            marker = (
                config.save_dir
                / "Metadata"
                / f"{recording_id}_legacy-v1_complete.json"
            )
            if not config.overwrite and marker.exists():
                legacy_preprocess[recording_id] = "existing"
                continue
            try:
                preprocess_legacy_recording(
                    project_dir=config.save_dir,
                    recording_id=recording_id,
                    experiment_name=config.experiment,
                    overwrite=config.overwrite,
                )
                legacy_preprocess[recording_id] = "completed"
            except Exception as error:
                legacy_preprocess[recording_id] = f"failed: {error}"
                if not config.continue_on_error:
                    raise

        legacy_ok = [
            recording_id
            for recording_id, state in legacy_preprocess.items()
            if state in {"existing", "completed"}
        ]
        if not legacy_ok:
            raise ConfigurationError(
                "Legacy preprocessing did not succeed for any recording."
            )
        legacy_result = run_legacy_analysis_pipeline(
            config.save_dir,
            legacy_ok,
            analysis_id=config.resolved_legacy_analysis_id(),
            experiment_name=config.experiment,
            alignment=config.legacy_alignment,
            read_batch_rows=config.batch_size,
            overwrite=config.overwrite,
            run_statistics=config.legacy_run_statistics,
        )
        legacy_runner_status = legacy_result.status

    if "candidate" in config.routes:
        metric_recipe = RUNNER_RECIPE_TO_METRIC_SOURCE[config.candidate_runner_recipe]
        candidate_result = run_candidate_development_pipeline(
            config.save_dir,
            recording_ids,
            analysis_id=config.resolved_candidate_analysis_id(),
            experiment_name=config.experiment,
            batch_size=config.batch_size,
            overwrite=config.overwrite,
            metric_recipe=metric_recipe,
            runner_recipe=config.candidate_runner_recipe,
            continue_on_error=config.continue_on_error,
        )
        candidate_runner_status = candidate_result.manifest_path.name

        if config.run_figures:
            comparison_recipe = (
                "candidate-metric-comparison-corrected-v1"
                if config.candidate_runner_recipe.endswith("-corrected-v1")
                else "candidate-metric-comparison-v1"
            )
            for outcome in config.figure_outcomes:
                figure_result = build_metric_comparison_figure(
                    config.save_dir,
                    config.resolved_candidate_analysis_id(),
                    mode=FigureMode(config.figure_mode),
                    trial_type="CS",
                    outcome_id=outcome,
                    comparison_recipe=comparison_recipe,
                    overwrite=config.overwrite,
                )
                figure_paths.extend(
                    path for path in figure_result.outputs if path.suffix == ".png"
                )

    summary_path = config.save_dir / "Metadata" / f"{config.analysis_id}_pipeline_run.json"
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": pipeline_config_to_dict(config),
        "recording_ids": list(recording_ids),
        "intake_completed": list(intake_completed),
        "intake_skipped": list(intake_skipped),
        "intake_failed": [
            {"recording_id": recording_id, "error": error}
            for recording_id, error in intake_failed
        ],
        "legacy_preprocess": legacy_preprocess,
        "legacy_runner_status": legacy_runner_status,
        "candidate_runner_status": candidate_runner_status,
        "figure_paths": [str(path) for path in figure_paths],
    }
    write_json_atomic(summary_path, payload)

    return PipelineRunResult(
        recording_ids=recording_ids,
        intake_completed=intake_completed,
        intake_skipped=intake_skipped,
        intake_failed=intake_failed,
        legacy_preprocess=legacy_preprocess,
        legacy_runner_status=legacy_runner_status,
        candidate_runner_status=candidate_runner_status,
        figure_paths=tuple(figure_paths),
        summary_path=summary_path,
    )
