"""Local orchestration for the frozen legacy stage-3 through stage-5 pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from classical_conditioning.analysis.legacy_normalized_vigor import (
    ALIGNMENTS,
    RECIPE_ID as NORMALIZED_RECIPE_ID,
    build_legacy_normalized_vigor,
)
from classical_conditioning.analysis.legacy_scaled_vigor import (
    LegacyScaledVigorConfig,
    RECIPE_ID as SCALED_VIGOR_RECIPE_ID,
    build_legacy_scaled_vigor_cohort,
)
from classical_conditioning.analysis.legacy_standard_main import (
    RECIPE_ID as STANDARD_MAIN_RECIPE_ID,
    build_legacy_standard_main,
)
from classical_conditioning.analysis.legacy_statistics import (
    RECIPE_ID as STATISTICS_RECIPE_ID,
    build_legacy_statistics,
)
from classical_conditioning.config import get_experiment_spec
from classical_conditioning.artifacts import (
    sha256_file,
    verify_completed_analysis_parquet_set,
    verify_completed_parquet_set,
    write_json_atomic,
)
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
)

RUNNER_RECIPE_ID = "legacy-runner-v1"
_STAGE_SEQUENCE = (
    (STANDARD_MAIN_RECIPE_ID, "legacy-standard-main"),
    (NORMALIZED_RECIPE_ID, "legacy-normalized-vigor"),
)
_STATISTICS_TABLES = (
    "model_input",
    "block_medians",
    "nonparametric",
    "global_model",
    "block_models",
    "trial_models",
)


@dataclass(frozen=True)
class LegacyRunnerResult:
    analysis_id: str
    recording_ids: tuple[str, ...]
    manifest_path: Path
    status: str
    step_status: dict[str, dict[str, str]]


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )


def _stage_marker_path(project_dir: Path, recording_id: str, recipe_id: str) -> Path:
    return project_dir / "Metadata" / f"{recording_id}_{recipe_id}_complete.json"


def _statistics_marker_path(project_dir: Path, analysis_id: str, alignment: str) -> Path:
    return (
        project_dir
        / "Metadata"
        / f"{analysis_id}_{STATISTICS_RECIPE_ID}_{alignment}_complete.json"
    )


def _verify_recording_stage(
    project_dir: Path,
    recording_id: str,
    recipe_id: str,
    experiment_name: str,
) -> None:
    source_dir = project_dir / "Processed data" / recording_id
    if recipe_id == STANDARD_MAIN_RECIPE_ID:
        paths = {
            alignment: source_dir / f"samples_{recipe_id}_{alignment}.parquet"
            for alignment in ALIGNMENTS
        }
    elif recipe_id == NORMALIZED_RECIPE_ID:
        paths = {
            alignment: source_dir / f"{recipe_id}_{alignment}.parquet"
            for alignment in ALIGNMENTS
        }
    else:
        raise AssertionError(f"Unsupported recording stage: {recipe_id}")
    verified = verify_completed_parquet_set(
        paths,
        project_dir / "Quality checks" / recording_id / f"{recipe_id}_summary.json",
        _stage_marker_path(project_dir, recording_id, recipe_id),
        recipe=recipe_id,
        recording_id=recording_id,
    )
    if verified.summary.get("experiment") != experiment_name:
        raise ArtifactIntegrityError(
            f"Completed {recipe_id} experiment does not match the requested "
            f"experiment: {verified.summary.get('experiment')!r} != "
            f"{experiment_name!r}"
        )


def _scaled_vigor_paths(
    project_dir: Path,
    analysis_id: str,
    experiment_name: str,
) -> dict[str, Path]:
    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    return {
        f"{condition.condition_id}_{alignment}_"
        f"{int(round(width * 1_000))}ms_{table_name}": (
            output_dir
            / f"{SCALED_VIGOR_RECIPE_ID}_{condition.condition_id}_{alignment}_"
            f"{int(round(width * 1_000))}ms_{table_name}.parquet"
        )
        for condition in get_experiment_spec(experiment_name).conditions
        for alignment in ALIGNMENTS
        for width in LegacyScaledVigorConfig().bin_widths_s
        for table_name in ("line", "count_heatmap", "scaled_heatmap")
    }


def _verify_scaled_vigor(
    project_dir: Path,
    analysis_id: str,
    recording_ids: tuple[str, ...],
    experiment_name: str,
) -> Path:
    marker_path = (
        project_dir / "Metadata" / f"{analysis_id}_{SCALED_VIGOR_RECIPE_ID}_complete.json"
    )
    verify_completed_analysis_parquet_set(
        _scaled_vigor_paths(project_dir, analysis_id, experiment_name),
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{SCALED_VIGOR_RECIPE_ID}_summary.json",
        marker_path,
        recipe=SCALED_VIGOR_RECIPE_ID,
        analysis_id=analysis_id,
        recording_ids=recording_ids,
    )
    return marker_path


def _verify_statistics(
    project_dir: Path,
    analysis_id: str,
    recording_ids: tuple[str, ...],
    alignment: str,
) -> Path:
    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    marker_path = _statistics_marker_path(project_dir, analysis_id, alignment)
    verify_completed_analysis_parquet_set(
        {
            name: output_dir / f"{STATISTICS_RECIPE_ID}_{alignment}_{name}.parquet"
            for name in _STATISTICS_TABLES
        },
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{STATISTICS_RECIPE_ID}_{alignment}_summary.json",
        marker_path,
        recipe=STATISTICS_RECIPE_ID,
        analysis_id=analysis_id,
        recording_ids=recording_ids,
        alignment=alignment,
    )
    return marker_path


def run_legacy_analysis_pipeline(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    experiment_name: str = "allDelay",
    alignment: str = "CS",
    read_batch_rows: int = 250_000,
    overwrite: bool = False,
    run_statistics: bool = True,
) -> LegacyRunnerResult:
    """Run the frozen legacy stage-3 -> stage-5 pipeline for explicit recording IDs."""
    _validate_analysis_id(analysis_id)
    project_dir = project_dir.resolve()
    ordered_ids = tuple(dict.fromkeys(str(recording_id) for recording_id in recording_ids))
    if not ordered_ids:
        raise ConfigurationError("At least one recording ID is required.")

    manifest_path = (
        project_dir
        / "Metadata"
        / f"{analysis_id}_{RUNNER_RECIPE_ID}_manifest.json"
    )
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ArtifactIntegrityError(
                f"Existing runner manifest is invalid JSON: {manifest_path}"
            ) from error
        expected_identity = {
            "recipe": RUNNER_RECIPE_ID,
            "analysis_id": analysis_id,
            "experiment_name": experiment_name,
            "alignment": alignment,
            "recording_ids": list(ordered_ids),
        }
        if any(
            previous_manifest.get(key) != value
            for key, value in expected_identity.items()
        ):
            raise ArtifactIntegrityError(
                "Existing runner manifest identity differs from the requested run. "
                "Use a new analysis ID."
            )

    step_status: dict[str, dict[str, str]] = {}
    lineage: dict[str, str] = {}

    for recording_id in ordered_ids:
        step_status[recording_id] = {}
        for recipe_id, step_name in _STAGE_SEQUENCE:
            marker_path = _stage_marker_path(project_dir, recording_id, recipe_id)
            if not overwrite and marker_path.exists():
                _verify_recording_stage(
                    project_dir,
                    recording_id,
                    recipe_id,
                    experiment_name,
                )
                step_status[recording_id][step_name] = "existing"
            else:
                if step_name == "legacy-standard-main":
                    build_legacy_standard_main(
                        project_dir,
                        recording_id,
                        experiment_name=experiment_name,
                        read_batch_rows=read_batch_rows,
                        overwrite=overwrite,
                    )
                elif step_name == "legacy-normalized-vigor":
                    build_legacy_normalized_vigor(
                        project_dir,
                        recording_id,
                        experiment_name=experiment_name,
                        overwrite=overwrite,
                    )
                else:
                    raise AssertionError(f"Unsupported pipeline step: {step_name}")
                _verify_recording_stage(
                    project_dir,
                    recording_id,
                    recipe_id,
                    experiment_name,
                )
                step_status[recording_id][step_name] = "completed"
            lineage[f"{recording_id}:{step_name}"] = sha256_file(marker_path)

    scaled_marker = (
        project_dir / "Metadata" / f"{analysis_id}_{SCALED_VIGOR_RECIPE_ID}_complete.json"
    )
    if not overwrite and scaled_marker.exists():
        _verify_scaled_vigor(
            project_dir,
            analysis_id,
            ordered_ids,
            experiment_name,
        )
        step_status["cohort"] = {"legacy-scaled-vigor": "existing"}
    else:
        build_legacy_scaled_vigor_cohort(
            project_dir,
            ordered_ids,
            analysis_id=analysis_id,
            overwrite=overwrite,
        )
        _verify_scaled_vigor(
            project_dir,
            analysis_id,
            ordered_ids,
            experiment_name,
        )
        step_status["cohort"] = {"legacy-scaled-vigor": "completed"}
    lineage["cohort:legacy-scaled-vigor"] = sha256_file(scaled_marker)

    if run_statistics:
        statistics_marker = _statistics_marker_path(project_dir, analysis_id, alignment)
        if not overwrite and statistics_marker.exists():
            _verify_statistics(
                project_dir,
                analysis_id,
                ordered_ids,
                alignment,
            )
            step_status["cohort"]["legacy-statistics"] = "existing"
        else:
            build_legacy_statistics(
                project_dir,
                ordered_ids,
                analysis_id=analysis_id,
                alignment=alignment,
                experiment_name=experiment_name,
                overwrite=overwrite,
            )
            _verify_statistics(
                project_dir,
                analysis_id,
                ordered_ids,
                alignment,
            )
            step_status["cohort"]["legacy-statistics"] = "completed"
        lineage["cohort:legacy-statistics"] = sha256_file(statistics_marker)

    manifest = {
        "recipe": RUNNER_RECIPE_ID,
        "scientific_status": "legacy_reproduction",
        "analysis_id": analysis_id,
        "experiment_name": experiment_name,
        "alignment": alignment,
        "statistics_requested": run_statistics,
        "recording_ids": list(ordered_ids),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "steps": step_status,
        "completion_marker_sha256": lineage,
    }
    write_json_atomic(manifest_path, manifest)

    return LegacyRunnerResult(
        analysis_id=analysis_id,
        recording_ids=ordered_ids,
        manifest_path=manifest_path,
        status=manifest["status"],
        step_status=step_status,
    )
