"""Export the frozen configuration of the supported candidate route.

Review note: this module writes an auditable snapshot of code-defined settings;
it does not load or mutate a run configuration supplied by a user.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from classical_conditioning.analysis.metric_comparison import (
    comparison_config_for_experiment,
)
from classical_conditioning.analysis.movement_state import (
    RUNNER_RECIPE_TO_METRIC_SOURCE,
    MovementCalibrationConfig,
    resolve_candidate_metric_source,
)
from classical_conditioning.analysis.temporal_profiles import TemporalProfileConfig
from classical_conditioning.analysis.trial_outcomes import TrialOutcomeConfig
from classical_conditioning.artifacts import write_json_atomic
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.config.trial_map import get_experiment_trial_map
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.preprocessing.candidate_metric_kernel import CandidateMetricConfig
from classical_conditioning.preprocessing.corrected_frame_preprocessing import CorrectedPreprocessConfig


# Return paths and identity together so a CLI caller can report all three files.
@dataclass(frozen=True)
class ResolvedConfigExport:
    recipe_id: str
    experiment_id: str
    config_hash: str
    config_path: Path
    trial_map_path: Path
    source_report_path: Path


def _json_value(value: Any) -> Any:
    # Recursively turn dataclasses/enums/paths into values accepted by json.dumps.
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    # Preserve nested structure while normalising JSON-incompatible mapping keys.
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def export_resolved_config(
    project_dir: Path,
    *,
    experiment_name: str = "allDelay",
    runner_recipe: str = "candidate-corrected-runner-v1",
    overwrite: bool = False,
) -> ResolvedConfigExport:
    """Write a reproducible description of the supported candidate route."""
    # Reject unsupported route labels before resolving dependent recipe settings.
    if runner_recipe not in RUNNER_RECIPE_TO_METRIC_SOURCE:
        raise ConfigurationError(f"Unsupported candidate runner recipe: {runner_recipe}")
    route = resolve_candidate_metric_source(runner_recipe=runner_recipe)
    experiment = get_experiment_spec(experiment_name)
    # Gather every stage's defaults into one normalized, serializable structure.
    resolved = _json_value(
        {
            "runner_recipe": route.runner_recipe,
            "scientific_status": route.scientific_status,
            "metric_recipe": route.metric_recipe,
            "movement_recipe": route.movement_recipe,
            "temporal_recipe": route.temporal_recipe,
            "trial_recipe": route.trial_recipe,
            "comparison_recipe": route.comparison_recipe,
            "experiment": experiment,
            "corrected_preprocess": CorrectedPreprocessConfig(),
            "candidate_metrics": CandidateMetricConfig(),
            "movement": MovementCalibrationConfig(),
            "temporal_profiles": TemporalProfileConfig(),
            "trial_outcomes": TrialOutcomeConfig(),
            "metric_comparison": comparison_config_for_experiment(experiment_name),
        }
    )
    # Hash canonical compact JSON, making the identifier independent of spacing.
    payload_text = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    # All configuration evidence belongs in the project's metadata namespace.
    metadata = project_dir.resolve() / "Metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    stem = f"{route.runner_recipe}_{experiment.experiment_id}"
    config_path = metadata / f"resolved_config_{stem}.json"
    trial_map_path = metadata / f"trial_map_{experiment.experiment_id}.json"
    source_report_path = metadata / f"config_source_report_{stem}.json"
    targets = (config_path, trial_map_path, source_report_path)
    # Refuse partial replacement: the three files form one coherent export set.
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Resolved configuration artifacts already exist; pass overwrite=True.")
    # Atomically publish the resolved settings, trial map, and concise provenance.
    write_json_atomic(config_path, {"artifact_kind": "resolved-candidate-config-v1", "config_hash": config_hash, "resolved": resolved})
    write_json_atomic(trial_map_path, get_experiment_trial_map(experiment_name))
    write_json_atomic(
        source_report_path,
        {
            "artifact_kind": "configuration-source-report-v1",
            "recipe_id": route.runner_recipe,
            "experiment_id": experiment.experiment_id,
            "scientific_status": route.scientific_status,
            "config_hash": config_hash,
            "notes": [
                "This export describes the supported candidate route.",
                "Historical legacy execution is archived outside the package.",
            ],
        },
    )
    # Return the same identity recorded in the files for programmatic callers.
    return ResolvedConfigExport(
        recipe_id=route.runner_recipe,
        experiment_id=experiment.experiment_id,
        config_hash=config_hash,
        config_path=config_path,
        trial_map_path=trial_map_path,
        source_report_path=source_report_path,
    )
