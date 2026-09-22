"""Immutable domain definitions for versioned analysis configuration.

Review note: this module contains declarative values only. Validation happens
when an immutable object is constructed, so downstream code can rely on these
invariants instead of repeatedly checking raw dictionaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from classical_conditioning.exceptions import ConfigurationError


# Alignment is the event to which a trial-derived result is time-locked.
class Alignment(str, Enum):
    CS = "CS"
    US = "US"


# Phase supplies the coarse experimental grouping used in analysis and figures.
class Phase(str, Enum):
    PRE = "Pre"
    TRAIN = "Train"
    TEST = "Test"


# Paradigm distinguishes the temporal relationship between CS and US.
class Paradigm(str, Enum):
    DELAY = "delay"
    TRACE = "trace"


# Role is analytical (control vs conditioned), independent of source filenames.
class ConditionRole(str, Enum):
    CONTROL = "control"
    CONDITIONED = "conditioned"


# A fish key is immutable and sortable so it can reliably index cohort tables.
@dataclass(frozen=True, order=True)
class FishKey:
    experiment_id: str
    day: str
    fish_number: str

    def __post_init__(self) -> None:
        # Reject blank components at the domain boundary; whitespace-only IDs
        # would otherwise produce distinct but unusable output paths and rows.
        for field_name in ("experiment_id", "day", "fish_number"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"FishKey {field_name} cannot be empty.")


# A time interval whose start must precede its end.
@dataclass(frozen=True)
class TimeWindow:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        # NaN/infinite endpoints make comparisons and time-bin construction unsafe.
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ConfigurationError("Time-window bounds must be finite.")
        if self.start_s >= self.end_s:
            raise ConfigurationError("Time-window start must precede its end.")


# One condition's stable ID, labels, role, visual colour, and US timing.
@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    display_name: str
    source_name: str
    role: ConditionRole
    color_rgb_255: tuple[int, int, int]
    us_latency_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        # Require enum values rather than arbitrary strings to prevent misspelled
        # roles from reaching comparison and figure code.
        if not isinstance(self.role, ConditionRole):
            raise ConfigurationError(f"Unknown condition role: {self.role!r}")
        # Every condition needs a machine ID, a label, and a raw-source token.
        if not self.condition_id or not self.display_name or not self.source_name:
            raise ConfigurationError("Condition identifiers and names cannot be empty.")
        # RGB channels use the conventional inclusive 0..255 scale.
        if len(self.color_rgb_255) != 3 or any(
            value < 0 or value > 255 for value in self.color_rgb_255
        ):
            raise ConfigurationError("Condition color must contain three 0-255 values.")
        # US latency is a physical duration, hence finite and never negative.
        if any(not math.isfinite(value) or value < 0 for value in self.us_latency_s):
            raise ConfigurationError("US latency values must be finite and non-negative.")


# TrialSpec maps one numbered trial/alignment pair into a phase and ten-trial block.
@dataclass(frozen=True)
class TrialSpec:
    alignment: Alignment
    trial_number: int
    phase: Phase
    block_10_id: int
    block_10_name: str
    catch: bool = False

    def __post_init__(self) -> None:
        # Validate enum-typed fields before testing their scalar properties.
        if not isinstance(self.alignment, Alignment):
            raise ConfigurationError(f"Unknown trial alignment: {self.alignment!r}")
        if not isinstance(self.phase, Phase):
            raise ConfigurationError(f"Unknown trial phase: {self.phase!r}")
        # Trial and block numbering begins at one in the experimental protocol.
        if self.trial_number < 1:
            raise ConfigurationError("Trial number must be positive.")
        if self.block_10_id < 1 or not self.block_10_name:
            raise ConfigurationError("Trial block identity must be positive and named.")


# ExperimentSpec is the complete frozen recipe-facing description of an assay.
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
        # Validate mandatory scalar identity and timing properties first.
        if not self.experiment_id:
            raise ConfigurationError("Experiment ID cannot be empty.")
        if not isinstance(self.paradigm, Paradigm):
            raise ConfigurationError(f"Unknown experiment paradigm: {self.paradigm!r}")
        if self.minimum_cs_trials < 1 or self.minimum_us_trials < 1:
            raise ConfigurationError("Minimum trial counts must be positive.")
        if not math.isfinite(self.cs_duration_s) or self.cs_duration_s <= 0:
            raise ConfigurationError("CS duration must be finite and positive.")

        # Condition machine IDs and raw-source names must each be unambiguous.
        condition_ids = [condition.condition_id for condition in self.conditions]
        source_names = [condition.source_name for condition in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ConfigurationError("Experiment condition IDs must be unique.")
        if len(source_names) != len(set(source_names)):
            raise ConfigurationError("Experiment condition source names must be unique.")

        # A numbered trial can appear only once for a given alignment.
        trial_keys = [
            (trial.alignment, trial.trial_number) for trial in self.analysis_trials
        ]
        if len(trial_keys) != len(set(trial_keys)):
            raise ConfigurationError(
                "Experiment trial mappings must be unique by alignment and number."
            )
        # Within an alignment, configured trial numbers must have no internal
        # gaps; gaps would make block assignment and expected coverage unclear.
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

        # Reuse of a block ID is valid only when its name and phase agree.
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
        # Export the compact lookup shape consumed by trial-mapping callers.
        return {
            (trial.alignment.value, trial.trial_number): trial.block_10_name
            for trial in self.analysis_trials
        }

    def catch_trial_numbers(self, alignment: Alignment = Alignment.CS) -> tuple[int, ...]:
        """Return configured catch trials in protocol order for one alignment."""
        return tuple(
            trial.trial_number
            for trial in self.analysis_trials
            if trial.alignment is alignment and trial.catch
        )

    def trial_blocks(
        self,
        alignment: Alignment = Alignment.CS,
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Return declared ten-trial blocks without reconstructing plot lists."""
        ordered = sorted(
            (trial for trial in self.analysis_trials if trial.alignment is alignment),
            key=lambda trial: (trial.block_10_id, trial.trial_number),
        )
        blocks: list[tuple[str, tuple[int, ...]]] = []
        for block_id in dict.fromkeys(trial.block_10_id for trial in ordered):
            members = tuple(
                trial.trial_number for trial in ordered if trial.block_10_id == block_id
            )
            name = next(
                trial.block_10_name for trial in ordered if trial.block_10_id == block_id
            )
            blocks.append((name, members))
        return tuple(blocks)
