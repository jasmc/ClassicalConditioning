"""Package-owned immutable analysis configuration."""

from classical_conditioning.config.domain import (
    Alignment,
    ConditionRole,
    ConditionSpec,
    ExperimentSpec,
    FishKey,
    Paradigm,
    Phase,
    TimeWindow,
    TrialSpec,
)
from classical_conditioning.config.experiments import (
    get_experiment_spec,
    get_trial_block_lookup,
)
from classical_conditioning.config.identity import (
    fish_key_from_recording_id,
    recording_id_from_fish_key,
)
from classical_conditioning.config.trial_map import (
    build_trial_map,
    get_experiment_trial_map,
)

__all__ = [
    "Alignment",
    "ConditionRole",
    "ConditionSpec",
    "ExperimentSpec",
    "FishKey",
    "Paradigm",
    "Phase",
    "ResolvedConfigExport",
    "TimeWindow",
    "TrialSpec",
    "build_trial_map",
    "export_resolved_config",
    "fish_key_from_recording_id",
    "get_experiment_spec",
    "get_experiment_trial_map",
    "get_trial_block_lookup",
    "recording_id_from_fish_key",
]


def __getattr__(name: str):
    if name in {"ResolvedConfigExport", "export_resolved_config"}:
        from classical_conditioning.config import export

        return getattr(export, name)
    raise AttributeError(name)
