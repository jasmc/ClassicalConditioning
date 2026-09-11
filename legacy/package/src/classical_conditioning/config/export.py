"""Write resolved configuration and trial-map artifacts locally."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from classical_conditioning.artifacts import write_json_atomic
from classical_conditioning.config.recipes import (
    ResolvedAnalysisConfig,
    config_hash,
    config_to_dict,
    get_legacy_paper_config,
    stage_config_hash,
)
from classical_conditioning.config.domain import ConfigurationStage
from classical_conditioning.config.trial_map import get_experiment_trial_map
from classical_conditioning.exceptions import ConfigurationError


@dataclass(frozen=True)
class ResolvedConfigExport:
    recipe_id: str
    experiment_id: str
    config_hash: str
    config_path: Path
    trial_map_path: Path
    source_report_path: Path


def _metadata_dir(project_dir: Path) -> Path:
    root = project_dir.resolve()
    metadata = root / "Metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    return metadata


def _source_report(config: ResolvedAnalysisConfig) -> dict[str, Any]:
    return {
        "artifact_kind": "configuration-source-report-v1",
        "recipe_id": config.recipe_id,
        "experiment_id": config.experiment.experiment_id,
        "scientific_status": config.scientific_status.value,
        "config_hash": config_hash(config),
        "stage_hashes": {
            stage.value: stage_config_hash(config, stage)
            for stage in ConfigurationStage
        },
        "source_trace": [
            {"section": entry.section, "source": entry.source}
            for entry in config.source_trace
        ],
        "notes": [
            "Legacy recipe values are preserved even when scientifically suspect.",
            "Display-only fields do not change preprocessing or outcome stage hashes.",
        ],
    }


def export_resolved_config(
    project_dir: Path,
    *,
    experiment_name: str = "allDelay",
    recipe_id: str = "legacy-paper-v1",
    overwrite: bool = False,
) -> ResolvedConfigExport:
    """Serialize the resolved recipe, trial map, and source-trace report."""
    if recipe_id != "legacy-paper-v1":
        raise ConfigurationError(
            f"Resolved-config export does not yet support recipe {recipe_id!r}."
        )

    config = get_legacy_paper_config(experiment_name)
    metadata = _metadata_dir(project_dir)
    stem = f"{config.recipe_id}_{config.experiment.experiment_id}"
    config_path = metadata / f"resolved_config_{stem}.json"
    trial_map_path = metadata / f"trial_map_{config.experiment.experiment_id}.json"
    source_report_path = metadata / f"config_source_report_{stem}.json"

    targets = (config_path, trial_map_path, source_report_path)
    if not overwrite and any(path.exists() for path in targets):
        existing = [str(path) for path in targets if path.exists()]
        raise FileExistsError(
            "Resolved configuration artifacts already exist; "
            f"pass overwrite=True to replace: {existing}"
        )

    payload = {
        "artifact_kind": "resolved-analysis-config-v1",
        "config_hash": config_hash(config),
        "stage_hashes": {
            stage.value: stage_config_hash(config, stage)
            for stage in ConfigurationStage
        },
        "resolved": config_to_dict(config),
    }
    write_json_atomic(config_path, payload)
    write_json_atomic(trial_map_path, get_experiment_trial_map(experiment_name))
    write_json_atomic(source_report_path, _source_report(config))

    return ResolvedConfigExport(
        recipe_id=config.recipe_id,
        experiment_id=config.experiment.experiment_id,
        config_hash=config_hash(config),
        config_path=config_path,
        trial_map_path=trial_map_path,
        source_report_path=source_report_path,
    )
