"""Public configuration API with lazy export support.

Review note: core immutable definitions load eagerly, while the export module
loads only on demand because it imports several pipeline-stage configurations.
"""

# Re-export immutable domain vocabulary used throughout the package.
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
# Re-export the supported experiment catalogue and its trial-block lookup.
from classical_conditioning.config.experiments import (
    get_experiment_spec,
    get_trial_block_lookup,
)
# Re-export lossless conversion between recording IDs and biological fish keys.
from classical_conditioning.config.identity import (
    fish_key_from_recording_id,
    recording_id_from_fish_key,
)
# Re-export construction of self-describing expected-trial maps.
from classical_conditioning.config.trial_map import (
    build_trial_map,
    get_experiment_trial_map,
)

# This is the intentionally small public configuration surface.
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
    # Delay importing the heavy export module until one of its public names is
    # actually requested; this prevents configuration imports from pulling in
    # analysis-stage dependencies.
    if name in {"ResolvedConfigExport", "export_resolved_config"}:
        from classical_conditioning.config import export

        return getattr(export, name)
    raise AttributeError(name)
