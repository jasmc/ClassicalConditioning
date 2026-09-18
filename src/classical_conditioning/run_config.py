"""JSON run configuration for relocatable batch pipelines.

Review note: this is the trust boundary for a user-authored pipeline JSON file.
It validates and normalizes input once, then pipeline.py consumes the immutable
PipelineRunConfig rather than re-reading arbitrary JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.paths import assert_project_dir_allowed

# Only these active candidate recipe families can be selected by a run config.
VALID_RUNNER_RECIPES = frozenset(
    {
        "candidate-development-runner-v1",
        "candidate-corrected-runner-v1",
    }
)
# The installable workflow is candidate-only; old legacy fields are rejected.
VALID_ROUTES = frozenset({"candidate"})
RETIRED_LEGACY_FIELDS = frozenset(
    {"legacy_alignment", "legacy_run_statistics", "legacy_analysis_id"}
)


def _validate_analysis_id(analysis_id: str) -> None:
    # This ID becomes a directory/filename component, so prohibit separators and
    # whitespace while retaining ordinary descriptive punctuation.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "analysis_id must use only letters, numbers, dot, underscore, or hyphen."
        )


# Normalized, immutable settings passed from CLI/config loading to the pipeline.
@dataclass(frozen=True)
class PipelineRunConfig:
    raw_dir: Path
    save_dir: Path
    experiment: str
    analysis_id: str
    routes: tuple[str, ...] = ("candidate",)
    keep_conditions: tuple[str, ...] | None = None
    recording_ids: tuple[str, ...] | None = None
    overwrite: bool = False
    continue_on_error: bool = True
    run_inventory: bool = False
    run_intake: bool = True
    run_figures: bool = False
    candidate_runner_recipe: str = "candidate-corrected-runner-v1"
    candidate_analysis_id: str | None = None
    batch_size: int = 250_000
    figure_mode: str = "static"
    figure_outcomes: tuple[str, ...] = ("movement-probability",)
    show_progress: bool = True

    @property
    def input_dir(self) -> Path:
        # Compatibility alias for lower-level functions that call raw_dir input.
        return self.raw_dir

    @property
    def project_dir(self) -> Path:
        # Compatibility alias for stages that call save_dir the project directory.
        return self.save_dir

    def resolved_candidate_analysis_id(self) -> str:
        # A route suffix prevents ambiguous collision with a caller's base ID.
        return self.candidate_analysis_id or f"{self.analysis_id}-candidate"


def _optional_string_list(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    # Optional filters are either absent or a non-empty normalised list.
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{field_name} must be a non-empty list or null.")
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _optional_recording_ids(value: Any) -> tuple[str, ...] | None:
    # Preserve first-seen order while removing duplicate explicitly requested IDs.
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ConfigurationError("recording_ids must be a non-empty list or null.")
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def load_pipeline_run_config(path: Path) -> PipelineRunConfig:
    """Load and validate a pipeline run configuration JSON file."""
    # Decode the file before any field-level validation can be meaningful.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigurationError("Pipeline config must be a JSON object.")
    # Fail explicitly rather than silently ignoring retired legacy semantics.
    retired_fields = sorted(RETIRED_LEGACY_FIELDS.intersection(payload))
    if retired_fields:
        raise ConfigurationError(
            "Legacy pipeline settings have been retired: "
            f"{retired_fields}. Remove them and use the candidate route."
        )

    # Coerce required paths once, then resolve them after constructing the config.
    raw_dir = Path(str(payload.get("raw_dir", ""))).expanduser()
    save_dir = Path(str(payload.get("save_dir", ""))).expanduser()
    if not str(raw_dir).strip():
        raise ConfigurationError("raw_dir is required.")
    if not str(save_dir).strip():
        raise ConfigurationError("save_dir is required.")

    # Confirm that the requested experiment has a frozen package specification.
    experiment = str(payload.get("experiment", "")).strip()
    if not experiment:
        raise ConfigurationError("experiment is required.")
    get_experiment_spec(experiment)

    # Validate the run label before it can become derived output path components.
    analysis_id = str(payload.get("analysis_id", "")).strip()
    if not analysis_id:
        raise ConfigurationError("analysis_id is required.")
    _validate_analysis_id(analysis_id)

    # Route is a list for forward compatibility, although only candidate exists.
    routes_raw = payload.get("routes", ["candidate"])
    if not isinstance(routes_raw, list) or not routes_raw:
        raise ConfigurationError("routes must be a non-empty list.")
    routes = tuple(str(item).strip().lower() for item in routes_raw)
    unknown = set(routes).difference(VALID_ROUTES)
    if unknown:
        if "legacy" in unknown:
            raise ConfigurationError(
                "The legacy route has been retired from the installable package. "
                "Use the candidate route; historical source is under legacy/."
            )
        raise ConfigurationError(f"Unknown routes: {sorted(unknown)}")

    # Select one of the frozen, internally compatible candidate recipe families.
    candidate_runner_recipe = str(
        payload.get("candidate_runner_recipe", "candidate-corrected-runner-v1")
    ).strip()
    if candidate_runner_recipe not in VALID_RUNNER_RECIPES:
        raise ConfigurationError(
            f"candidate_runner_recipe must be one of {sorted(VALID_RUNNER_RECIPES)}."
        )

    # Figure mode controls renderer outputs and is constrained to supported modes.
    figure_mode = str(payload.get("figure_mode", "static")).strip().lower()
    if figure_mode not in {"static", "publication"}:
        raise ConfigurationError("figure_mode must be static or publication.")

    # Batch size must stay positive because stages use it for chunked table reads.
    batch_size = int(payload.get("batch_size", 250_000))
    if batch_size <= 0:
        raise ConfigurationError("batch_size must be positive.")

    # Materialize one normalized, resolved, immutable configuration object.
    config = PipelineRunConfig(
        raw_dir=raw_dir.resolve(),
        save_dir=save_dir.resolve(),
        experiment=experiment,
        analysis_id=analysis_id,
        routes=routes,
        keep_conditions=_optional_string_list(
            payload.get("keep_conditions"),
            field_name="keep_conditions",
        ),
        recording_ids=_optional_recording_ids(payload.get("recording_ids")),
        overwrite=bool(payload.get("overwrite", False)),
        continue_on_error=bool(payload.get("continue_on_error", True)),
        run_inventory=bool(payload.get("run_inventory", False)),
        run_intake=bool(payload.get("run_intake", True)),
        run_figures=bool(payload.get("run_figures", False)),
        candidate_runner_recipe=candidate_runner_recipe,
        candidate_analysis_id=(
            str(payload["candidate_analysis_id"]).strip()
            if payload.get("candidate_analysis_id")
            else None
        ),
        batch_size=batch_size,
        figure_mode=figure_mode,
        figure_outcomes=_optional_string_list(
            payload.get("figure_outcomes", ["movement-probability"]),
            field_name="figure_outcomes",
        )
        or ("movement-probability",),
        show_progress=bool(payload.get("show_progress", True)),
    )
    # Enforce the raw-versus-derived filesystem boundary before any output exists.
    assert_project_dir_allowed(config.raw_dir, config.save_dir)
    return config


def pipeline_config_to_dict(config: PipelineRunConfig) -> dict[str, Any]:
    """Serialize a run config for provenance sidecars."""
    # Serialize every effective setting, including defaults, for a run ledger.
    return {
        "raw_dir": str(config.raw_dir),
        "save_dir": str(config.save_dir),
        "experiment": config.experiment,
        "analysis_id": config.analysis_id,
        "routes": list(config.routes),
        "keep_conditions": list(config.keep_conditions or ()),
        "recording_ids": list(config.recording_ids or ()),
        "overwrite": config.overwrite,
        "continue_on_error": config.continue_on_error,
        "run_inventory": config.run_inventory,
        "run_intake": config.run_intake,
        "run_figures": config.run_figures,
        "candidate_runner_recipe": config.candidate_runner_recipe,
        "candidate_analysis_id": config.resolved_candidate_analysis_id(),
        "batch_size": config.batch_size,
        "figure_mode": config.figure_mode,
        "figure_outcomes": list(config.figure_outcomes),
        "show_progress": config.show_progress,
    }
