"""Run inventory, verified intake, corrected analysis, and required figures."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from classical_conditioning.analysis.candidate_runner import run_candidate_development_pipeline
from classical_conditioning.analysis.cohort_outcomes import build_cohort_trial_outcomes
from classical_conditioning.analysis.discarding import DEFAULT_METRIC, assess_discarding
from classical_conditioning.analysis.inference.learning_onset import (
    LearningOnsetConfig,
    build_learning_onset_analysis,
    load_learning_onset_analysis,
)
from classical_conditioning.analysis.metric_comparison import OUTCOME_COLUMNS
from classical_conditioning.analysis.review.trace_review import build_trace_review
from classical_conditioning.artifacts import sha256_file, write_json_atomic
from classical_conditioning.cohort import load_cohort_manifest
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.figures.cohort_response import (
    build_block_profile_figure,
    build_catch_profile_figure,
    build_event_aligned_ratio_figure,
    build_selected_block_ratio_figure,
    build_trial_ratio_figure,
)
from classical_conditioning.figures.export import FigureExportResult, FigureMode
from classical_conditioning.figures.learning_diagnostics import build_learning_diagnostics_figure
from classical_conditioning.figures.learning_onset import build_learning_onset_figure
from classical_conditioning.figures.metric_comparison import build_metric_comparison_figure
from classical_conditioning.figures.population_heatmap import build_population_heatmap_figure
from classical_conditioning.figures.temporal_profiles import (
    FIGURE_SPECS,
    build_candidate_profile_figure,
)
from classical_conditioning.intake import IntakeBatchResult, intake_recordings
from classical_conditioning.inventory import build_recording_inventory
from classical_conditioning.progress import PipelineProgress, default_progress
from classical_conditioning.run_config import PipelineRunConfig, pipeline_config_to_dict


COHORT_FIGURES: dict[str, Callable[..., FigureExportResult]] = {
    "selected-block-ratio": build_selected_block_ratio_figure,
    "trial-ratio": build_trial_ratio_figure,
    "event-aligned-ratio": build_event_aligned_ratio_figure,
    "catch-profile": build_catch_profile_figure,
    "block-profile": build_block_profile_figure,
}
INTAKE_QC = ("camera_timing.png", "tracking_overview.png", "protocol_timeline.png")
PAPER_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "configs" / "paper-figures" / "behavior-paper.json"
)


# Return the artifacts and stage outcomes needed for the final run summary.
@dataclass(frozen=True)
class PipelineRunResult:
    recording_ids: tuple[str, ...]
    intake_completed: tuple[str, ...] = ()
    intake_skipped: tuple[str, ...] = ()
    intake_failed: tuple[tuple[str, str], ...] = ()
    candidate_runner_status: str | None = None
    figure_paths: tuple[Path, ...] = ()
    summary_path: Path | None = None


def resolve_pipeline_recording_ids(
    config: PipelineRunConfig, inventory: dict[str, Any] | None = None
) -> tuple[str, ...]:
    """Select all matching raw groups, including incomplete recordings."""
    inventory = inventory or build_recording_inventory(config.raw_dir, hash_files=True)
    if config.recording_ids is not None:
        return config.recording_ids
    keep = set(config.keep_conditions or ())
    selected = tuple(dict.fromkeys(
        str(record["recording_id"])
        for record in inventory["records"]
        if record.get("recording_id") is not None
        and (not keep or record.get("condition_id") in keep)
    ))
    if not selected:
        raise ConfigurationError("No raw recordings matched the requested filters.")
    return selected


# Normalize figure builder return types before collecting their output paths.
def _figure_paths(result: Any) -> tuple[Path, ...]:
    if isinstance(result, FigureExportResult):
        return result.outputs
    if isinstance(result, Path):
        return (result,)
    if hasattr(result, "static_figure_path"):
        return (result.static_figure_path,)
    raise TypeError(f"Unexpected figure result: {type(result).__name__}")


def _paper_panel_statuses() -> dict[str, dict[str, Any]]:
    """Use the proposed registry; a routine plot never implies paper approval."""
    registry = json.loads(PAPER_REGISTRY.read_text(encoding="utf-8"))
    expected = {
        f"fig-{number}{letter}"
        for number, letters in (("1", "ABCDEFGH"), ("2", "ABCDEFGHI"),
                                ("3", "ABCDEFG"), ("4", "ABC"))
        for letter in letters
    }
    panels = registry["panels"]
    if set(panels) != expected:
        raise ConfigurationError("Paper figure registry has missing or unexpected panel IDs.")
    if any(item["status"] not in {"completed", "failed", "blocked"} for item in panels.values()):
        raise ConfigurationError("Paper figure registry has an invalid panel status.")
    return panels


def _profile_figure_ids(alignment: str) -> tuple[str, ...]:
    """The signed, pre-CS-centered candidate heatmap is CS-only."""
    return tuple(
        figure_id for figure_id in FIGURE_SPECS
        if alignment == "CS" or figure_id != "signed-log-vigor"
    )


def _figure_authentication(
    config: PipelineRunConfig, inventory: dict[str, Any] | None,
    selection_assessment: Path | None = None,
) -> dict[str, str | None]:
    """Bind panel status to the chosen cohort, metric code, settings, and raw scan."""
    cohort_path = (
        config.save_dir / "Processed data" / "Cohorts" / config.cohort_id
        / "cohort-manifest.parquet"
        if config.cohort_id else None
    )
    metric_code = (
        Path(__file__).resolve().parent
        / "preprocessing" / "candidate_metric_kernel.py"
    )
    settings = json.dumps(
        pipeline_config_to_dict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    metric_definition = (
        config.metric.encode("utf-8") + b"\0" + metric_code.read_bytes()
        if config.metric else None
    )
    return {
        "cohort_sha256": sha256_file(cohort_path) if cohort_path and cohort_path.is_file() else None,
        "metric_sha256": hashlib.sha256(metric_definition).hexdigest() if metric_definition else None,
        "settings_sha256": hashlib.sha256(settings).hexdigest(),
        "upstream_sha256": inventory.get("records_sha256") if inventory else None,
        "selection_assessment_sha256": (
            sha256_file(selection_assessment)
            if selection_assessment and selection_assessment.is_file() else None
        ),
    }


def run_pipeline(
    config: PipelineRunConfig,
    *,
    progress: PipelineProgress | None = None,
) -> PipelineRunResult:
    """Complete the routine route and persist every recording and figure status."""
    progress = progress or default_progress(enabled=config.show_progress)
    mode = FigureMode(config.figure_mode)
    config.save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.save_dir / "Metadata" / f"{config.analysis_id}_pipeline_run.json"
    selected: tuple[str, ...] = ()
    active: tuple[str, ...] = ()
    intake: IntakeBatchResult | None = None
    candidate_status: str | None = None
    figures: dict[str, dict[str, Any]] = {}
    paper_panels: dict[str, dict[str, Any]] = {}
    inventory: dict[str, Any] | None = None
    selection_assessment: Path | None = None
    stage_errors: list[str] = []
    figure_paths: list[Path] = []
    pending: dict[Future[Any], str] = {}

    # Record unavailable figures with a reason for the run summary.
    def blocked(key: str, reason: str) -> None:
        figures[key] = {"status": "blocked", "reason": reason, "outputs": []}

    try:
        paper_panels = _paper_panel_statuses()
        for alignment in ("CS", "US"):
            for outcome in OUTCOME_COLUMNS:
                blocked(f"metric-comparison:{alignment}:{outcome}", "cohort comparison is not available")
        for name in COHORT_FIGURES:
            blocked(f"cohort:{name}", "reviewed cohort and metric are not available")
        blocked("cohort:population-heatmap", "matched reviewed cohort and metric are not available")
        blocked("learning:diagnostics", "learning model is not available")
        blocked("learning:onset", "learning model is not available")
        blocked("learner:figures", "frozen learner representation is not available")
        # Inventory is unconditional and includes incomplete, ambiguous, and
        # failed raw groups before any analysis-stage selection is made.
        with progress.stage_timer("Inventory"):
            inventory = build_recording_inventory(
                config.raw_dir, hash_files=True, inspect_tracking_headers=True
            )
            write_json_atomic(
                config.save_dir / "Metadata" / "recording_inventory.json", inventory
            )
            selected = resolve_pipeline_recording_ids(config, inventory)
            for recording_id in selected:
                for filename in INTAKE_QC:
                    blocked(
                        f"{recording_id}:intake-qc:{filename.removesuffix('.png')}",
                        "verified intake has not completed",
                    )
                blocked(f"{recording_id}:detector-review", "movement state is not available")
                for alignment in ("CS", "US"):
                    for figure_id in _profile_figure_ids(alignment):
                        blocked(
                            f"{recording_id}:profile:{alignment}:{figure_id}",
                            "temporal profile is not available",
                        )
        with progress.stage_timer("Verified intake"):
            intake = intake_recordings(
                config.raw_dir,
                config.save_dir,
                keep_conditions=config.keep_conditions,
                recording_ids=selected,
                chunk_rows=config.batch_size,
                overwrite=config.overwrite,
                inventory=inventory,
                progress=progress,
            )
        active_set = set(intake.completed) | set(intake.skipped)
        active = tuple(recording_id for recording_id in selected if recording_id in active_set)
        problem_by_id = {
            recording_id: reason for recording_id, reason in (*intake.failed, *intake.incomplete)
        }
        if intake.failed:
            stage_errors.extend(f"intake {name}: {reason}" for name, reason in intake.failed)
        if intake.incomplete:
            stage_errors.extend(f"incomplete {name}: {reason}" for name, reason in intake.incomplete)

        # Intake QC is published transactionally with its three Parquet files.
        for recording_id in selected:
            for filename in INTAKE_QC:
                key = f"{recording_id}:intake-qc:{filename.removesuffix('.png')}"
                path = config.save_dir / "Quality checks" / recording_id / "figures" / filename
                if recording_id in active_set and path.is_file():
                    figures[key] = {"status": "completed", "reason": None, "outputs": [str(path)]}
                    figure_paths.append(path)
                else:
                    blocked(key, problem_by_id.get(recording_id, "intake QC artifact is missing"))

        with ProcessPoolExecutor(max_workers=2) as pool:
            # Track each submitted figure job by the key used in the report.
            def schedule(key: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
                figures[key] = {"status": "pending", "reason": None, "outputs": []}
                pending[pool.submit(function, *args, **kwargs)] = key

            # Submit figures as soon as their prerequisite processing stage finishes.
            def stage_ready(recording_id: str, stage: str) -> None:
                if stage == "movement-state":
                    schedule(
                        f"{recording_id}:detector-review",
                        build_trace_review,
                        config.save_dir,
                        recording_id,
                        metric_recipe="tail-candidate-corrected",
                        overwrite=True,
                    )
                if stage == "temporal-profiles":
                    for alignment in ("CS", "US"):
                        for figure_id in _profile_figure_ids(alignment):
                            schedule(
                                f"{recording_id}:profile:{alignment}:{figure_id}",
                                build_candidate_profile_figure,
                                config.save_dir,
                                recording_id,
                                mode=mode,
                                trial_type=alignment,
                                figure_id=figure_id,
                                metric_recipe="tail-candidate-corrected",
                                overwrite=True,
                            )

            runner_ok = False
            # Include per-fish processing failures before assessing cohort eligibility.
            def assess_after_fish(
                _successful: tuple[str, ...],
                steps: dict[str, dict[str, str]],
            ) -> None:
                nonlocal selection_assessment
                ledger = json.loads(intake.ledger_path.read_text(encoding="utf-8"))
                statuses = {
                    name: value.get("status", "unknown")
                    for name, value in ledger["recordings"].items()
                }
                for name, stage in steps.items():
                    if stage.get("failed"):
                        statuses[name] = "candidate_failed"
                with progress.stage_timer("Technical and exploratory discarding assessment"):
                    assessment = assess_discarding(
                        config.raw_dir, config.save_dir,
                        analysis_id=config.analysis_id, experiment=config.experiment,
                        metric_id=config.assessment_metric or config.metric or DEFAULT_METRIC,
                        technical_policy_path=config.technical_policy,
                        disabled_rules=config.disabled_discard_checks,
                        recording_ids=selected, inventory=inventory,
                        processing_statuses=statuses,
                    )
                    selection_assessment = assessment.summary_path

            if active:
                try:
                    with progress.stage_timer("Corrected three-metric analysis"):
                        runner = run_candidate_development_pipeline(
                            config.save_dir,
                            active,
                            analysis_id=config.analysis_id,
                            experiment_name=config.experiment,
                            batch_size=config.batch_size,
                            overwrite=config.overwrite,
                            metric_recipe="tail-candidate-corrected",
                            runner_recipe="candidate-corrected-runner",
                            continue_on_error=config.continue_on_error,
                            on_stage_ready=stage_ready,
                            before_comparison=assess_after_fish,
                            progress=progress,
                        )
                    candidate_status = runner.manifest_path.name
                    runner_ok = "cohort" in runner.step_status
                    for name in active:
                        failed = runner.step_status.get(name, {}).get("failed")
                        if failed:
                            stage_errors.append(f"candidate {name}: {failed}")
                except Exception as error:
                    stage_errors.append(f"candidate runner: {error}")
            else:
                stage_errors.append("No recordings have verified intake.")
            if selection_assessment is None:
                try:
                    assess_after_fish((), {})
                except Exception as error:
                    stage_errors.append(f"discarding assessment: {error}")

            for recording_id in selected:
                for key in (
                    f"{recording_id}:detector-review",
                    *(f"{recording_id}:profile:{alignment}:{figure_id}"
                      for alignment in ("CS", "US")
                      for figure_id in _profile_figure_ids(alignment)),
                ):
                    if key not in figures:
                        blocked(key, problem_by_id.get(recording_id, "candidate stage did not complete"))

            for alignment in ("CS", "US"):
                for outcome in OUTCOME_COLUMNS:
                    key = f"metric-comparison:{alignment}:{outcome}"
                    if runner_ok:
                        schedule(
                            key, build_metric_comparison_figure,
                            config.save_dir, config.analysis_id,
                            mode=mode, trial_type=alignment, outcome_id=outcome,
                            comparison_recipe="candidate-metric-comparison-corrected",
                            overwrite=True,
                        )
                    else:
                        blocked(key, "cohort metric comparison did not complete")

            # A reviewed manifest is immutable, while its applied population
            # table is a replaceable derived artifact authenticated by that
            # manifest and every included fish's trial outcome marker.
            cohort_ready = False
            cohort_error: str | None = None
            if config.cohort_id and config.metric:
                try:
                    build_cohort_trial_outcomes(
                        config.save_dir,
                        cohort_id=config.cohort_id,
                        metric_recipe="tail-candidate-corrected",
                        overwrite=True,
                    )
                    cohort_ready = True
                except Exception as error:
                    cohort_error = str(error)
                    stage_errors.append(f"cohort outcomes: {error}")

            # These five figures use one reviewed cohort and one selected metric.
            for name, builder in COHORT_FIGURES.items():
                key = f"cohort:{name}"
                if cohort_ready:
                    schedule(
                        key, builder, config.save_dir,
                        cohort_id=config.cohort_id,
                        analysis_id=config.analysis_id,
                        metric_id=config.metric,
                        mode=mode,
                        overwrite=True,
                    )
                else:
                    blocked(
                        key,
                        f"cohort outcomes unavailable: {cohort_error}"
                        if cohort_error else "requires a reviewed cohort_id and selected metric",
                    )

            if cohort_ready:
                manifest = load_cohort_manifest(config.save_dir, config.cohort_id)
                included = manifest.loc[manifest["primary_included"].astype(bool)]
                conditions = set(included["condition_id"].astype(str))
                if "control" in conditions and len(conditions - {"control"}) == 1:
                    schedule(
                        "cohort:population-heatmap", build_population_heatmap_figure,
                        config.save_dir, cohort_id=config.cohort_id,
                        analysis_id=config.analysis_id, metric_id=config.metric,
                        mode=mode, overwrite=True,
                    )
                else:
                    blocked(
                        "cohort:population-heatmap",
                        "requires a frozen cohort containing one paired condition and its matched control",
                    )
            elif cohort_error:
                blocked("cohort:population-heatmap", f"cohort outcomes unavailable: {cohort_error}")

            if cohort_ready:
                try:
                    conditioned = next(
                        item.condition_id
                        for item in get_experiment_spec(config.experiment).conditions
                        if item.condition_id != "control"
                    )
                    build_learning_onset_analysis(
                        config.save_dir,
                        cohort_id=config.cohort_id,
                        analysis_id=config.analysis_id,
                        config=LearningOnsetConfig(
                            metric_id=config.metric, test_condition=conditioned
                        ),
                        overwrite=True,
                    )
                    schedule(
                        "learning:diagnostics", build_learning_diagnostics_figure,
                        config.save_dir, config.analysis_id, mode=mode, overwrite=True,
                    )
                    frames, _ = load_learning_onset_analysis(config.save_dir, config.analysis_id)
                    diagnostics = frames["diagnostics"]
                    required = diagnostics.loc[
                        diagnostics["required_for_publication"].fillna(True).astype(bool)
                    ]
                    if not required.empty and required["diagnostic_status"].eq("ok").all():
                        schedule(
                            "learning:onset", build_learning_onset_figure,
                            config.save_dir, config.analysis_id, mode=mode, overwrite=True,
                        )
                    else:
                        blocked("learning:onset", "required block or longitudinal diagnostic gate failed")
                except Exception as error:
                    stage_errors.append(f"learning onset: {error}")
                    blocked("learning:diagnostics", f"model fit unavailable: {error}")
                    blocked("learning:onset", f"model fit unavailable: {error}")
            else:
                reason = (
                    f"cohort outcomes unavailable: {cohort_error}"
                    if cohort_error else "requires a reviewed cohort and selected metric"
                )
                blocked("learning:diagnostics", reason)
                blocked("learning:onset", reason)
            blocked(
                "learner:figures",
                "requires Gate L, a frozen learner representation, and its validated renderer",
            )

            for future in as_completed(pending):
                key = pending[future]
                try:
                    outputs = _figure_paths(future.result())
                    figures[key] = {
                        "status": "completed", "reason": None,
                        "outputs": [str(path) for path in outputs],
                    }
                    figure_paths.extend(outputs)
                    if key == "cohort:population-heatmap":
                        panel_id = {
                            "allDelay": "fig-2A", "all3sTrace": "fig-2B",
                            "all10sTrace": "fig-2C",
                        }.get(config.experiment)
                        if panel_id:
                            paper_panels[panel_id]["descriptive_precursor"] = [str(path) for path in outputs]
                except Exception as error:
                    figures[key] = {"status": "failed", "reason": str(error), "outputs": []}
                    stage_errors.append(f"figure {key}: {error}")
    except Exception as error:
        stage_errors.append(f"pipeline: {error}")
    finally:
        # An invocation is auditable even if inventory, intake, analysis, or
        # rendering stops before all downstream artifacts become available.
        # The pool has exited here, so every submitted render has a final
        # outcome even when an earlier stage interrupted normal collection.
        for future, key in pending.items():
            if figures.get(key, {}).get("status") != "pending":
                continue
            try:
                outputs = _figure_paths(future.result())
                figures[key] = {
                    "status": "completed", "reason": None,
                    "outputs": [str(path) for path in outputs],
                }
                figure_paths.extend(outputs)
            except Exception as error:
                figures[key] = {"status": "failed", "reason": str(error), "outputs": []}
                stage_errors.append(f"figure {key}: {error}")
        write_json_atomic(summary_path, {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if stage_errors else "complete",
            "config": pipeline_config_to_dict(config),
            "recording_ids": list(selected),
            "active_recording_ids": list(active),
            "intake_completed": list(intake.completed) if intake else [],
            "intake_skipped": list(intake.skipped) if intake else [],
            "intake_failed": [
                {"recording_id": name, "error": reason}
                for name, reason in (intake.failed if intake else ())
            ],
            "intake_incomplete": [
                {"recording_id": name, "reason": reason}
                for name, reason in (intake.incomplete if intake else ())
            ],
            "intake_failed_skipped": list(intake.failed_skipped) if intake else [],
            "intake_status_ledger": str(intake.ledger_path) if intake else None,
            "selection_assessment": str(selection_assessment) if selection_assessment else None,
            "candidate_runner_status": candidate_status,
            "figures": figures,
            "paper_panels": paper_panels,
            "paper_registry": str(PAPER_REGISTRY),
            "figure_authentication": _figure_authentication(
                config, inventory, selection_assessment
            ),
            "stage_errors": stage_errors,
        })
    if stage_errors:
        raise ConfigurationError(
            f"Pipeline finished with {len(stage_errors)} issue(s); see {summary_path}"
        )
    return PipelineRunResult(
        recording_ids=selected,
        intake_completed=intake.completed if intake else (),
        intake_skipped=intake.skipped if intake else (),
        intake_failed=intake.failed if intake else (),
        candidate_runner_status=candidate_status,
        figure_paths=tuple(figure_paths),
        summary_path=summary_path,
    )
