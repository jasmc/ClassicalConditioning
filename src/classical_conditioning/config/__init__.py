"""Package-owned immutable analysis configuration."""

from classical_conditioning.config.domain import (
    Alignment,
    ConditionRole,
    ConditionSpec,
    ConfigurationStage,
    ExperimentSpec,
    FishKey,
    Paradigm,
    Phase,
    ScientificStatus,
    TimeWindow,
    TrialSpec,
)
from classical_conditioning.config.experiments import (
    get_experiment_spec,
    get_trial_block_lookup,
)
from classical_conditioning.config.export import (
    ResolvedConfigExport,
    export_resolved_config,
)
from classical_conditioning.config.identity import (
    fish_key_from_recording_id,
    recording_id_from_fish_key,
)
from classical_conditioning.config.recipes import (
    ConfigurationSource,
    LegacyPreprocessingSettings,
    ResolvedAnalysisConfig,
    TemporalOutcomeSettings,
    config_hash,
    config_to_dict,
    config_to_json,
    get_legacy_paper_config,
    stage_config_hash,
)
from classical_conditioning.config.trial_map import (
    build_trial_map,
    get_experiment_trial_map,
)

__all__ = [
    "Alignment",
    "ConditionRole",
    "ConditionSpec",
    "ConfigurationSource",
    "ConfigurationStage",
    "ExperimentSpec",
    "FishKey",
    "LegacyPreprocessingSettings",
    "Paradigm",
    "Phase",
    "ResolvedAnalysisConfig",
    "ResolvedConfigExport",
    "ScientificStatus",
    "TemporalOutcomeSettings",
    "TimeWindow",
    "TrialSpec",
    "build_trial_map",
    "config_hash",
    "config_to_dict",
    "config_to_json",
    "export_resolved_config",
    "fish_key_from_recording_id",
    "get_experiment_spec",
    "get_experiment_trial_map",
    "get_legacy_paper_config",
    "get_trial_block_lookup",
    "recording_id_from_fish_key",
    "stage_config_hash",
]
