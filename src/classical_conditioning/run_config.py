"""Strict JSON settings for the routine corrected analysis run."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.analysis.movement_state import METRIC_IDS
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.paths import assert_project_dir_allowed


_FIELDS = frozenset({
    "raw_dir", "save_dir", "experiment", "analysis_id", "keep_conditions",
    "recording_ids", "overwrite", "continue_on_error", "batch_size",
    "figure_mode", "show_progress", "cohort_id", "metric",
    "learner_representation_id",
    "assessment_metric", "technical_policy", "disabled_discard_checks",
})
_OBSOLETE = frozenset({
    "routes", "candidate_runner_recipe", "candidate_analysis_id",
    "run_inventory", "run_intake", "run_figures", "figure_outcomes",
    "legacy_alignment", "legacy_run_statistics", "legacy_analysis_id",
})


def _identifier(value: Any, field: str) -> str:
    identifier = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identifier):
        raise ConfigurationError(
            f"{field} must use only letters, numbers, dot, underscore, or hyphen."
        )
    if re.search(r"[-_]v\d+(?:$|[-_.])", identifier, re.IGNORECASE):
        raise ConfigurationError(f"{field} must not contain a version suffix.")
    return identifier


def _optional_list(value: Any, field: str, *, lower: bool = False) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ConfigurationError(f"{field} must be a non-empty list of strings or null.")
    values = (item.strip().lower() if lower else item.strip() for item in value)
    return tuple(dict.fromkeys(values))


def _optional_identifier(value: Any, field: str) -> str | None:
    """Accept either null or a path-safe, non-empty JSON identifier."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string or null.")
    return _identifier(value, field)


@dataclass(frozen=True)
class PipelineRunConfig:
    raw_dir: Path
    save_dir: Path
    experiment: str
    analysis_id: str
    keep_conditions: tuple[str, ...] | None = None
    recording_ids: tuple[str, ...] | None = None
    overwrite: bool = False
    continue_on_error: bool = True
    batch_size: int = 250_000
    figure_mode: str = "static"
    show_progress: bool = True
    cohort_id: str | None = None
    metric: str | None = None
    learner_representation_id: str | None = None
    assessment_metric: str | None = None
    technical_policy: Path | None = None
    disabled_discard_checks: tuple[str, ...] = ()

    @property
    def input_dir(self) -> Path:
        return self.raw_dir

    @property
    def project_dir(self) -> Path:
        return self.save_dir

    def resolved_candidate_analysis_id(self) -> str:
        # The routine has one analysis identity shared by results and figures.
        return self.analysis_id


def load_pipeline_run_config(path: Path) -> PipelineRunConfig:
    """Load strict JSON and reject obsolete switches before any work begins."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ConfigurationError("Pipeline config must be a JSON object.")
    obsolete = sorted(set(payload) & _OBSOLETE)
    if obsolete:
        raise ConfigurationError(
            f"Obsolete pipeline settings: {obsolete}. The corrected candidate "
            "route always inventories, intakes, and renders available figures."
        )
    unknown = sorted(set(payload) - _FIELDS)
    if unknown:
        raise ConfigurationError(f"Unknown pipeline settings: {unknown}.")
    for field in ("raw_dir", "save_dir", "experiment", "analysis_id"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ConfigurationError(f"{field} is required as a non-empty string.")
    experiment = payload["experiment"].strip()
    get_experiment_spec(experiment)
    figure_mode = str(payload.get("figure_mode", "static")).strip().lower()
    if figure_mode not in {"static", "publication"}:
        raise ConfigurationError("figure_mode must be static or publication.")
    batch_size = payload.get("batch_size", 250_000)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ConfigurationError("batch_size must be a positive integer.")
    for field in ("overwrite", "continue_on_error", "show_progress"):
        if field in payload and not isinstance(payload[field], bool):
            raise ConfigurationError(f"{field} must be a boolean.")
    if "technical_policy" in payload and payload["technical_policy"] is not None and (
        not isinstance(payload["technical_policy"], str)
        or not payload["technical_policy"].strip()
    ):
        raise ConfigurationError("technical_policy must be a non-empty path or null.")
    config = PipelineRunConfig(
        raw_dir=Path(payload["raw_dir"]).expanduser().resolve(),
        save_dir=Path(payload["save_dir"]).expanduser().resolve(),
        experiment=experiment,
        analysis_id=_identifier(payload["analysis_id"], "analysis_id"),
        keep_conditions=_optional_list(payload.get("keep_conditions"), "keep_conditions", lower=True),
        recording_ids=_optional_list(payload.get("recording_ids"), "recording_ids"),
        overwrite=payload.get("overwrite", False),
        continue_on_error=payload.get("continue_on_error", True),
        batch_size=batch_size,
        figure_mode=figure_mode,
        show_progress=payload.get("show_progress", True),
        cohort_id=_optional_identifier(payload.get("cohort_id"), "cohort_id"),
        metric=_optional_identifier(payload.get("metric"), "metric"),
        learner_representation_id=_optional_identifier(
            payload.get("learner_representation_id"), "learner_representation_id"
        ),
        assessment_metric=_optional_identifier(payload.get("assessment_metric"), "assessment_metric"),
        technical_policy=(Path(payload["technical_policy"]).expanduser().resolve()
                          if isinstance(payload.get("technical_policy"), str) and payload["technical_policy"].strip()
                          else None),
        disabled_discard_checks=_optional_list(
            payload.get("disabled_discard_checks"), "disabled_discard_checks"
        ) or (),
    )
    if bool(config.cohort_id) != bool(config.metric):
        raise ConfigurationError("cohort_id and metric must be supplied together.")
    if config.metric and config.metric not in METRIC_IDS.values():
        raise ConfigurationError(
            f"metric must be one of {sorted(METRIC_IDS.values())}."
        )
    valid_conditions = {
        condition.condition_id for condition in get_experiment_spec(experiment).conditions
    }
    if config.keep_conditions and set(config.keep_conditions) - valid_conditions:
        raise ConfigurationError(
            f"keep_conditions must be chosen from {sorted(valid_conditions)}."
        )
    assert_project_dir_allowed(config.raw_dir, config.save_dir)
    return config


def pipeline_config_to_dict(config: PipelineRunConfig) -> dict[str, Any]:
    """Record every effective setting in the invocation summary."""
    return {
        "raw_dir": str(config.raw_dir),
        "save_dir": str(config.save_dir),
        "experiment": config.experiment,
        "analysis_id": config.analysis_id,
        "keep_conditions": list(config.keep_conditions or ()),
        "recording_ids": list(config.recording_ids or ()),
        "overwrite": config.overwrite,
        "continue_on_error": config.continue_on_error,
        "batch_size": config.batch_size,
        "figure_mode": config.figure_mode,
        "show_progress": config.show_progress,
        "cohort_id": config.cohort_id,
        "metric": config.metric,
        "learner_representation_id": config.learner_representation_id,
        "assessment_metric": config.assessment_metric,
        "technical_policy": str(config.technical_policy) if config.technical_policy else None,
        "disabled_discard_checks": list(config.disabled_discard_checks),
    }
