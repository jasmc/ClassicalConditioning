"""Validated experiment specifications migrated from legacy configuration."""

from __future__ import annotations

from classical_conditioning.config.domain import (
    Alignment,
    ConditionRole,
    ConditionSpec,
    ExperimentSpec,
    Paradigm,
    Phase,
    TimeWindow,
    TrialSpec,
)
from classical_conditioning.exceptions import ConfigurationError


_CS_GROUPS = (
    ("Pre-train", Phase.PRE, range(5, 15)),
    ("Train 1", Phase.TRAIN, range(15, 25)),
    ("Train 2", Phase.TRAIN, range(25, 35)),
    ("Train 3", Phase.TRAIN, range(35, 45)),
    ("Train 4", Phase.TRAIN, range(45, 55)),
    ("Train 5", Phase.TRAIN, range(55, 65)),
    ("Test 1", Phase.TEST, range(65, 75)),
    ("Test 2", Phase.TEST, range(75, 85)),
    ("Test 3", Phase.TEST, range(85, 95)),
)
_US_GROUPS = (
    ("Train 1", Phase.TRAIN, range(18, 28)),
    ("Train 2", Phase.TRAIN, range(28, 37)),
    ("Train 3", Phase.TRAIN, range(37, 46)),
    ("Train 4", Phase.TRAIN, range(46, 55)),
    ("Train 5", Phase.TRAIN, range(55, 64)),
)


def _analysis_trials() -> tuple[TrialSpec, ...]:
    trials: list[TrialSpec] = []
    for alignment, groups in (
        (Alignment.CS, _CS_GROUPS),
        (Alignment.US, _US_GROUPS),
    ):
        for block_id, (block_name, phase, trial_numbers) in enumerate(groups, start=1):
            trials.extend(
                TrialSpec(
                    alignment=alignment,
                    trial_number=int(trial_number),
                    phase=phase,
                    block_10_id=block_id,
                    block_10_name=block_name,
                )
                for trial_number in trial_numbers
            )
    return tuple(trials)


_SHARED_TRIAL_STRUCTURE = dict(
    analysis_trials=_analysis_trials(),
    minimum_cs_trials=94,
    minimum_us_trials=78,
    cs_duration_s=10.0,
)

_ALL_DELAY = ExperimentSpec(
    experiment_id="allDelay",
    paradigm=Paradigm.DELAY,
    conditions=(
        ConditionSpec(
            condition_id="control",
            display_name="Control",
            source_name="control",
            role=ConditionRole.CONTROL,
            color_rgb_255=(0, 174, 239),
        ),
        ConditionSpec(
            condition_id="delay",
            display_name="Delay",
            source_name="delay",
            role=ConditionRole.CONDITIONED,
            color_rgb_255=(236, 0, 140),
            us_latency_s=(9.0,) * 46,
        ),
    ),
    conditioned_response_window=TimeWindow(0.0, 9.0),
    **_SHARED_TRIAL_STRUCTURE,
)

_FIXED_VS_INCREASING_TRACE = ExperimentSpec(
    experiment_id="fixedVsIncreasingTrace",
    paradigm=Paradigm.TRACE,
    conditions=(
        ConditionSpec(
            condition_id="control",
            display_name="Control",
            source_name="control",
            role=ConditionRole.CONTROL,
            color_rgb_255=(0, 174, 239),
        ),
        ConditionSpec(
            condition_id="fixedtrace",
            display_name="Trace CC fixed",
            source_name="fixedTrace",
            role=ConditionRole.CONDITIONED,
            color_rgb_255=(241, 90, 41),
            us_latency_s=(9.0,) * 46,
        ),
    ),
    conditioned_response_window=TimeWindow(0.0, 13.0),
    **_SHARED_TRIAL_STRUCTURE,
)

_EXPERIMENTS = {
    spec.experiment_id: spec
    for spec in (_ALL_DELAY, _FIXED_VS_INCREASING_TRACE)
}


def get_experiment_spec(experiment_name: str) -> ExperimentSpec:
    try:
        return _EXPERIMENTS[experiment_name]
    except KeyError as error:
        raise ConfigurationError(
            f"Experiment {experiment_name!r} is not yet migrated to package config."
        ) from error


def get_trial_block_lookup(experiment_name: str) -> dict[tuple[str, int], str]:
    return get_experiment_spec(experiment_name).block_lookup()
