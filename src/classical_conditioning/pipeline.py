"""Config-driven orchestration for intake and analysis routes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from classical_conditioning.analysis.candidate_runner import (
    run_candidate_development_pipeline,
)
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
from classical_conditioning.progress import PipelineProgress, default_progress
from classical_conditioning.run_config import PipelineRunConfig, pipeline_config_to_dict


@dataclass(frozen=True)
class PipelineRunResult:
    """High-level record of what this invocation did, not an analysis artifact.

    Per-recording scientific outputs and their provenance live in their
    respective stage directories.  This result, and the JSON summary written
    at the end of :func:`run_pipeline`, make partial completion visible to a
    caller without re-reading those artifacts.
    """
    recording_ids: tuple[str, ...]
    intake_completed: tuple[str, ...] = ()
    intake_skipped: tuple[str, ...] = ()
    intake_failed: tuple[tuple[str, str], ...] = ()
    candidate_runner_status: str | None = None
    figure_paths: tuple[Path, ...] = ()
    summary_path: Path | None = None


def resolve_pipeline_recording_ids(config: PipelineRunConfig) -> tuple[str, ...]:
    """Return the requested cohort, preserving discovery order.

    An explicit list is authoritative.  Otherwise this function discovers only
    complete raw triplets and applies the optional filename-condition filter.
    It deliberately does not apply scientific exclusions: those need a
    separately reviewed cohort policy rather than an implicit discovery rule.
    """
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
    # ``dict`` preserves insertion order, so duplicate source discoveries do
    # not make one recording run twice while discovery order remains stable.
    return tuple(dict.fromkeys(selected))


def run_pipeline(
    config: PipelineRunConfig,
    *,
    progress: PipelineProgress | None = None,
) -> PipelineRunResult:
    """Execute the configured intake, analysis routes, and optional figures.

    This is orchestration only.  It selects a cohort, calls independently
    versioned stages, and records their status. The candidate runner enforces
    its own artifact lineage.
    """
    progress = progress or default_progress(enabled=config.show_progress)
    progress.stage(
        "Pipeline start",
        detail=(
            f"experiment={config.experiment} analysis_id={config.analysis_id} "
            f"routes={list(config.routes)}"
        ),
    )
    recording_ids = resolve_pipeline_recording_ids(config)
    progress.info(f"selected {len(recording_ids)} recording(s)")
    # ``raw_dir`` is an immutable input.  Every generated artifact belongs
    # below the separately configured, writable save tree.
    config.save_dir.mkdir(parents=True, exist_ok=True)

    intake_completed: tuple[str, ...] = ()
    intake_skipped: tuple[str, ...] = ()
    intake_failed: tuple[tuple[str, str], ...] = ()
    candidate_runner_status: str | None = None
    figure_paths: list[Path] = []

    if config.run_inventory:
        with progress.stage_timer("Inventory"):
            # Inventory is optional QC/provenance.  Intake independently
            # validates its inputs, so an existing inventory is not required
            # to run the analysis.
            inventory_path = (
                config.save_dir / "Metadata" / "recording_inventory.json"
            )
            write_recording_inventory(
                config.raw_dir,
                inventory_path,
                hash_files=True,
                overwrite=config.overwrite,
            )
            progress.info(f"wrote {inventory_path}")

    if config.run_intake:
        with progress.stage_timer(
            "Intake",
            detail=f"{len(recording_ids)} recording(s)",
        ):
            intake_result = intake_recordings(
                config.raw_dir,
                config.save_dir,
                keep_conditions=config.keep_conditions,
                recording_ids=recording_ids,
                overwrite=config.overwrite,
                progress=progress,
            )
            intake_completed = intake_result.completed
            intake_skipped = intake_result.skipped
            intake_failed = intake_result.failed
            progress.info(
                f"completed={len(intake_completed)} "
                f"skipped={len(intake_skipped)} "
                f"failed={len(intake_failed)}"
            )
        if intake_failed and not config.continue_on_error:
            raise ConfigurationError(
                "Intake failed for one or more recordings; see pipeline summary."
            )
        # Downstream stages must only receive recordings with a current intake
        # artifact.  "Skipped" means an existing intake artifact was accepted;
        # it is therefore as eligible as one completed in this invocation.
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

    if "candidate" in config.routes:
        # A runner recipe selects one frozen, internally compatible set of
        # metric, detector, temporal-profile, trial-outcome, and comparison
        # recipes.  Passing both selectors below prevents accidental mixing of
        # the development and corrected artifact families.
        metric_recipe = RUNNER_RECIPE_TO_METRIC_SOURCE[config.candidate_runner_recipe]
        with progress.stage_timer(
            "Candidate analysis runner",
            detail=config.resolved_candidate_analysis_id(),
        ):
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
                progress=progress,
            )
            candidate_runner_status = candidate_result.manifest_path.name
            progress.info(f"manifest={candidate_runner_status}")

        if config.run_figures:
            # Figures consume the cohort comparison, not raw frames.  Keep the
            # comparison recipe paired with the runner family for the same
            # reason that the runner pairs its per-recording stages.
            comparison_recipe = (
                "candidate-metric-comparison-corrected-v1"
                if config.candidate_runner_recipe.endswith("-corrected-v1")
                else "candidate-metric-comparison-v1"
            )
            with progress.stage_timer(
                "Figure generation",
                detail=f"{len(config.figure_outcomes)} outcome(s)",
            ):
                for outcome in progress.iter_items(
                    config.figure_outcomes,
                    description="figures",
                ):
                    progress.info(f"building metric-comparison outcome={outcome}")
                    try:
                        figure_result = build_metric_comparison_figure(
                            config.save_dir,
                            config.resolved_candidate_analysis_id(),
                            mode=FigureMode(config.figure_mode),
                            trial_type="CS",
                            outcome_id=outcome,
                            comparison_recipe=comparison_recipe,
                            overwrite=config.overwrite,
                        )
                    except FileNotFoundError as error:
                        # Convert an implementation-level missing artifact
                        # error into an actionable orchestration error: figures
                        # are valid only after their matching comparison route.
                        raise ConfigurationError(
                            "Figure generation could not find the candidate "
                            "metric-comparison summary. Preprocessing may have "
                            "finished, but the comparison parquet is missing. "
                            "Confirm the candidate route completed and that "
                            f"run_figures uses a matching analysis_id. Details: {error}"
                        ) from error
                    figure_paths.extend(
                        path
                        for path in figure_result.outputs
                        if path.suffix == ".png"
                    )
                    progress.info(
                        "wrote "
                        + ", ".join(
                            path.name
                            for path in figure_result.outputs
                            if path.suffix == ".png"
                        )
                    )

    with progress.stage_timer("Writing pipeline summary"):
        summary_path = (
            config.save_dir / "Metadata" / f"{config.analysis_id}_pipeline_run.json"
        )
        # This is intentionally a run ledger, not a scientific result.  Store
        # the resolved config and failures so a partial run remains auditable.
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
            "candidate_runner_status": candidate_runner_status,
            "figure_paths": [str(path) for path in figure_paths],
        }
        write_json_atomic(summary_path, payload)
        progress.info(f"wrote {summary_path}")

    progress.stage("Pipeline complete")
    return PipelineRunResult(
        recording_ids=recording_ids,
        intake_completed=intake_completed,
        intake_skipped=intake_skipped,
        intake_failed=intake_failed,
        candidate_runner_status=candidate_runner_status,
        figure_paths=tuple(figure_paths),
        summary_path=summary_path,
    )
