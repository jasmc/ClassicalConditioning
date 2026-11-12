"""Validated experiment specifications migrated from legacy configuration.

Review note: these are the package's frozen, in-code assay definitions. They
are values, not user inputs; consumers obtain one by experiment ID below.
"""

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
from classical_conditioning.config.protocols import SESSION


# CS and US trial numbering are historically different, so each gets its own
# ordered block definition: display name, phase, and inclusive trial range.
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
# All 18 pre-training US events are retained. Training starts at raw US 19.
# Derive each block's US count from its non-catch CS trials, avoiding an
# independent, shifted copy of the pairing schedule.
_us_cursor = SESSION.first_training_us
_us_groups = []
for _name, _phase, _cs_ids in _CS_GROUPS:
    if _phase is Phase.TRAIN:
        _count = sum(i not in SESSION.training_catch_cs for i in _cs_ids)
        _us_groups.append((_name, _phase, range(_us_cursor, _us_cursor + _count)))
        _us_cursor += _count
_US_GROUPS = tuple(_us_groups)

# Protocol-defined CS omissions used by cohort and learner temporal profiles.
# Trial 65 is intentionally included: the first Early Test trial is analysed as
# a catch trial even though it is outside the historical training-only list.
_CS_CATCH_TRIALS = frozenset(SESSION.analysis_catch_cs)


def _analysis_trials() -> tuple[TrialSpec, ...]:
    # Expand compact ranges into one validated immutable TrialSpec per trial.
    trials: list[TrialSpec] = []
    for alignment, groups in (
        (Alignment.CS, _CS_GROUPS),
        (Alignment.US, _US_GROUPS),
    ):
        # Block IDs are assigned independently within each alignment.
        for block_id, (block_name, phase, trial_numbers) in enumerate(groups, start=1):
            trials.extend(
                TrialSpec(
                    alignment=alignment,
                    trial_number=int(trial_number),
                    phase=phase,
                    block_10_id=block_id,
                    block_10_name=block_name,
                    catch=(
                        alignment is Alignment.CS
                        and int(trial_number) in _CS_CATCH_TRIALS
                    ),
                )
                for trial_number in trial_numbers
            )
    return tuple(trials)


# The currently supported experiments share protocol trial numbering and CS duration.
_SHARED_TRIAL_STRUCTURE = dict(
    analysis_trials=_analysis_trials(),
    expected_cs_count=SESSION.cs_count,
    expected_us_count=SESSION.complete_us_count,
    cs_duration_s=SESSION.cs_duration_s,
)

# Delay conditioning has its own condition label and response window.
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
            us_latency_s=(9.0,) * SESSION.paired_count,
        ),
    ),
    conditioned_response_window=TimeWindow(0.0, 9.0),
    **_SHARED_TRIAL_STRUCTURE,
)

# The 3-second trace assay retains the same control role but uses a longer
# response window.
_ALL_3S_TRACE = ExperimentSpec(
    experiment_id="all3sTrace",
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
            condition_id="trace",
            display_name="3sTrace",
            source_name="trace",
            role=ConditionRole.CONDITIONED,
            color_rgb_255=(241, 90, 41),
            us_latency_s=(13.0,) * SESSION.paired_count,
        ),
    ),
    conditioned_response_window=TimeWindow(0.0, 13.0),
    **_SHARED_TRIAL_STRUCTURE,
)

# The 10-second trace assay has a 20-second conditioned-response window and
# source data named after the fixed 10-second trace protocol.
_ALL_10S_TRACE = ExperimentSpec(
    experiment_id="all10sTrace",
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
            condition_id="trace",
            display_name="10sTrace",
            source_name="10sFixedTrace",
            role=ConditionRole.CONDITIONED,
            color_rgb_255=(145, 54, 25),
            us_latency_s=(20.0,) * SESSION.paired_count,
        ),
    ),
    conditioned_response_window=TimeWindow(0.0, 20.0),
    **_SHARED_TRIAL_STRUCTURE,
)

# Keep lookup construction next to the immutable definitions to avoid aliases.
_EXPERIMENTS = {
    spec.experiment_id: spec
    for spec in (_ALL_DELAY, _ALL_3S_TRACE, _ALL_10S_TRACE)
}


def get_experiment_spec(experiment_name: str) -> ExperimentSpec:
    # Only named, migrated specifications may be used by the active package.
    try:
        return _EXPERIMENTS[experiment_name]
    except KeyError as error:
        raise ConfigurationError(
            f"Experiment {experiment_name!r} is not yet migrated to package config."
        ) from error


def get_trial_block_lookup(experiment_name: str) -> dict[tuple[str, int], str]:
    # Delegate both experiment validation and the canonical lookup construction.
    return get_experiment_spec(experiment_name).block_lookup()


def condition_aliases(experiment_name: str) -> dict[str, str]:
    """Normalize supported acquisition tokens without changing stored IDs."""
    spec = get_experiment_spec(experiment_name)
    aliases = {c.source_name.lower(): c.condition_id for c in spec.conditions}
    if experiment_name == "all3sTrace":
        aliases.update({"3strace": "trace", "tracenomk801": "trace",
                        "controlnomk801": "control"})
    elif experiment_name == "all10sTrace":
        aliases.update({"10strace": "trace", "trace": "trace"})
    return aliases


def resolve_condition(experiment_name: str, token: str) -> str:
    try:
        return condition_aliases(experiment_name)[token.strip().lower()]
    except KeyError as exc:
        raise ConfigurationError(
            f"Recording condition is not configured for {experiment_name}: {token!r}"
        ) from exc
