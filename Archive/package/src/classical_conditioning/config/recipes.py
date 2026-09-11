"""Versioned, deterministic analysis recipes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from classical_conditioning.config.domain import (
    ConfigurationStage,
    ExperimentSpec,
    ScientificStatus,
)
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import ConfigurationError


@dataclass(frozen=True)
class LegacyPreprocessingSettings:
    expected_framerate_hz: float = 700.0
    camera_rows_discarded_at_start: int = 13_999
    maximum_interval_deviation_ms: float = 0.005
    frame_loss_buffer_frames: int = 700
    tracking_error_threshold_deg: float = 2 * 180 / math.pi
    angle_point_count: int = 16
    temporal_filter_frames: int = 10
    spatial_filter_segments_configured_but_not_applied: int = 3
    bout_max_window_frames: int = 20
    bout_min_window_frames: int = 400
    bout_threshold_primary_deg_per_ms: float = 4.0
    bout_threshold_secondary_configured_but_not_applied_deg_per_ms: float = 1.0
    minimum_bout_duration_frames: int = 40
    minimum_interbout_frames: int = 10
    trial_start_frames: int = -45 * 700
    trial_end_frames: int = 45 * 700
    baseline_window_frames: int = 15 * 700


@dataclass(frozen=True)
class TemporalOutcomeSettings:
    window_start_s: float = -45.0
    window_end_s: float = 45.0
    bin_width_s: float = 0.5
    interval_closure: str = "left"
    aggregation: str = "mean_total_activity"

    def __post_init__(self) -> None:
        if self.window_start_s >= self.window_end_s:
            raise ConfigurationError("Outcome window start must precede its end.")
        if not math.isfinite(self.bin_width_s) or self.bin_width_s <= 0:
            raise ConfigurationError("Outcome bin width must be finite and positive.")
        if self.interval_closure != "left":
            raise ConfigurationError("Only left-closed outcome bins are supported.")


@dataclass(frozen=True)
class ConfigurationSource:
    section: str
    source: str


@dataclass(frozen=True)
class ResolvedAnalysisConfig:
    recipe_id: str
    scientific_status: ScientificStatus
    experiment: ExperimentSpec
    preprocessing: LegacyPreprocessingSettings
    outcomes: TemporalOutcomeSettings
    source_trace: tuple[ConfigurationSource, ...]

    def __post_init__(self) -> None:
        if not self.recipe_id:
            raise ConfigurationError("Recipe ID cannot be empty.")
        if not isinstance(self.scientific_status, ScientificStatus):
            raise ConfigurationError(
                f"Unknown scientific status: {self.scientific_status!r}"
            )
        traced_sections = [entry.section for entry in self.source_trace]
        required_sections = {"experiment", "preprocessing", "outcomes"}
        if not required_sections.issubset(traced_sections):
            missing = sorted(required_sections.difference(traced_sections))
            raise ConfigurationError(
                f"Configuration source trace is missing sections: {missing}"
            )


def get_legacy_paper_config(
    experiment_name: str = "allDelay",
) -> ResolvedAnalysisConfig:
    return ResolvedAnalysisConfig(
        recipe_id="legacy-paper-v1",
        scientific_status=ScientificStatus.LEGACY,
        experiment=get_experiment_spec(experiment_name),
        preprocessing=LegacyPreprocessingSettings(),
        outcomes=TemporalOutcomeSettings(),
        source_trace=(
            ConfigurationSource(
                section="experiment",
                source="experiment_configuration.get_experiment_config('allDelay')",
            ),
            ConfigurationSource(
                section="preprocessing",
                source=(
                    "general_configuration.config and "
                    "preprocessing.legacy_v1.LegacyPreprocessingConfig"
                ),
            ),
            ConfigurationSource(
                section="outcomes",
                source="analysis.temporal_profiles.TemporalProfileConfig",
            ),
        ),
    )


def _json_primitive(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_primitive(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_primitive(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_primitive(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ConfigurationError("Configuration cannot contain non-finite numbers.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ConfigurationError(
        f"Configuration value of type {type(value).__name__} is not serializable."
    )


def config_to_dict(config: ResolvedAnalysisConfig) -> dict[str, Any]:
    result = _json_primitive(config)
    if not isinstance(result, dict):
        raise ConfigurationError("Resolved configuration must serialize to an object.")
    return result


def config_to_json(config: ResolvedAnalysisConfig) -> str:
    return json.dumps(
        config_to_dict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def config_hash(config: ResolvedAnalysisConfig) -> str:
    return hashlib.sha256(config_to_json(config).encode("utf-8")).hexdigest()


def _scientific_experiment_settings(experiment: ExperimentSpec) -> dict[str, Any]:
    return {
        "experiment_id": experiment.experiment_id,
        "paradigm": experiment.paradigm,
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "source_name": condition.source_name,
                "role": condition.role,
                "us_latency_s": condition.us_latency_s,
            }
            for condition in experiment.conditions
        ],
        "analysis_trials": experiment.analysis_trials,
        "minimum_cs_trials": experiment.minimum_cs_trials,
        "minimum_us_trials": experiment.minimum_us_trials,
        "cs_duration_s": experiment.cs_duration_s,
        "conditioned_response_window": experiment.conditioned_response_window,
    }


def stage_config_hash(
    config: ResolvedAnalysisConfig,
    stage: ConfigurationStage,
) -> str:
    common = {
        "recipe_id": config.recipe_id,
        "scientific_status": config.scientific_status,
        "experiment": _scientific_experiment_settings(config.experiment),
    }
    if stage is ConfigurationStage.EXPERIMENT:
        selected: Mapping[str, Any] = {
            **common,
            "display_conditions": [
                {
                    "condition_id": condition.condition_id,
                    "display_name": condition.display_name,
                    "color_rgb_255": condition.color_rgb_255,
                }
                for condition in config.experiment.conditions
            ],
        }
    elif stage is ConfigurationStage.PREPROCESSING:
        selected = {**common, "preprocessing": config.preprocessing}
    elif stage is ConfigurationStage.OUTCOMES:
        selected = {**common, "outcomes": config.outcomes}
    else:
        raise ConfigurationError(f"Unsupported configuration stage: {stage!r}")
    payload = json.dumps(
        _json_primitive(selected),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
