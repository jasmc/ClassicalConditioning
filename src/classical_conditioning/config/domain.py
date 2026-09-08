"""Immutable domain definitions for versioned analysis configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from classical_conditioning.exceptions import ConfigurationError


class Alignment(str, Enum):
    CS = "CS"
    US = "US"


class Phase(str, Enum):
    PRE = "Pre"
    TRAIN = "Train"
    TEST = "Test"


class Paradigm(str, Enum):
    DELAY = "delay"
    TRACE = "trace"


class ConditionRole(str, Enum):
    CONTROL = "control"
    CONDITIONED = "conditioned"


class ScientificStatus(str, Enum):
    LEGACY = "legacy"
    CANDIDATE = "candidate"
    CORRECTED = "corrected"


class ConfigurationStage(str, Enum):
    EXPERIMENT = "experiment"
    PREPROCESSING = "preprocessing"
    OUTCOMES = "outcomes"


@dataclass(frozen=True, order=True)
class FishKey:
    experiment_id: str
    day: str
    fish_number: str

    def __post_init__(self) -> None:
        for field_name in ("experiment_id", "day", "fish_number"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"FishKey {field_name} cannot be empty.")


@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ConfigurationError("Time-window bounds must be finite.")
        if self.start_s >= self.end_s:
            raise ConfigurationError("Time-window start must precede its end.")


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    display_name: str
    source_name: str
    role: ConditionRole
    color_rgb_255: tuple[int, int, int]
    us_latency_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, ConditionRole):
            raise ConfigurationError(f"Unknown condition role: {self.role!r}")
        if not self.condition_id or not self.display_name or not self.source_name:
            raise ConfigurationError("Condition identifiers and names cannot be empty.")
        if len(self.color_rgb_255) != 3 or any(
            value < 0 or value > 255 for value in self.color_rgb_255
        ):
            raise ConfigurationError("Condition color must contain three 0-255 values.")
        if any(not math.isfinite(value) or value < 0 for value in self.us_latency_s):
            raise ConfigurationError("US latency values must be finite and non-negative.")


@dataclass(frozen=True)
class TrialSpec:
    alignment: Alignment
    trial_number: int
    phase: Phase
    block_10_id: int
    block_10_name: str
    catch: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.alignment, Alignment):
            raise ConfigurationError(f"Unknown trial alignment: {self.alignment!r}")
        if not isinstance(self.phase, Phase):
            raise ConfigurationError(f"Unknown trial phase: {self.phase!r}")
        if self.trial_number < 1:
            raise ConfigurationError("Trial number must be positive.")
        if self.block_10_id < 1 or not self.block_10_name:
            raise ConfigurationError("Trial block identity must be positive and named.")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    paradigm: Paradigm
    conditions: tuple[ConditionSpec, ...]
    analysis_trials: tuple[TrialSpec, ...]
    minimum_cs_trials: int
    minimum_us_trials: int
    cs_duration_s: float
    conditioned_response_window: TimeWindow

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ConfigurationError("Experiment ID cannot be empty.")
        if not isinstance(self.paradigm, Paradigm):
            raise ConfigurationError(f"Unknown experiment paradigm: {self.paradigm!r}")
        if self.minimum_cs_trials < 1 or self.minimum_us_trials < 1:
            raise ConfigurationError("Minimum trial counts must be positive.")
        if not math.isfinite(self.cs_duration_s) or self.cs_duration_s <= 0:
            raise ConfigurationError("CS duration must be finite and positive.")

        condition_ids = [condition.condition_id for condition in self.conditions]
        source_names = [condition.source_name for condition in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ConfigurationError("Experiment condition IDs must be unique.")
        if len(source_names) != len(set(source_names)):
            raise ConfigurationError("Experiment condition source names must be unique.")

        trial_keys = [
            (trial.alignment, trial.trial_number) for trial in self.analysis_trials
        ]
        if len(trial_keys) != len(set(trial_keys)):
            raise ConfigurationError(
                "Experiment trial mappings must be unique by alignment and number."
            )
        for alignment in Alignment:
            numbers = sorted(
                trial.trial_number
                for trial in self.analysis_trials
                if trial.alignment is alignment
            )
            if numbers and numbers != list(range(numbers[0], numbers[-1] + 1)):
                raise ConfigurationError(
                    f"{alignment.value} analysis trial mapping has internal gaps."
                )

        block_definitions: dict[tuple[Alignment, int], tuple[str, Phase]] = {}
        for trial in self.analysis_trials:
            key = (trial.alignment, trial.block_10_id)
            definition = (trial.block_10_name, trial.phase)
            existing = block_definitions.setdefault(key, definition)
            if existing != definition:
                raise ConfigurationError(
                    "One trial block ID cannot have conflicting names or phases."
                )

    def block_lookup(self) -> dict[tuple[str, int], str]:
        return {
            (trial.alignment.value, trial.trial_number): trial.block_10_name
            for trial in self.analysis_trials
        }
