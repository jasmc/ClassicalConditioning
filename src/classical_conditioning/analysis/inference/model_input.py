"""Shared frozen model-input tables for candidate population inference.

Both the exploratory LME scaffold and the fish-permutation alternative consume
the same authenticated trial-outcome → model-input transform (Step 10.2).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.movement_state import (
    resolve_candidate_metric_source,
)
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    SchemaValidationError,
)

RECIPE_ID = "candidate-model-input-v1"
OUTCOME_SPECS = {
    "total-activity": (
        "response_total_activity",
        "baseline_total_activity",
    ),
    "conditional-intensity": (
        "conditional_intensity",
        "baseline_total_activity",
    ),
    "bout-rate": (
        "bout_rate_per_minute",
        "baseline_total_activity",
    ),
}
MODEL_INPUT_COLUMNS = (
    "experiment_id",
    "recording_id",
    "fish_id",
    "condition_id",
    "trial_id",
    "alignment",
    "trial_number",
    "block_10_name",
    "metric_id",
    "outcome_id",
    "response",
    "baseline",
    "log_response",
    "log_baseline",
)
DIAGNOSTIC_STATUS_VALUES = ("ok", "failed", "singular")


@dataclass(frozen=True)
class ModelInputConfig:
    alignment: str = "CS"
    min_response_valid_samples: int = 1
    min_baseline_valid_samples: int = 1
    activity_offset: float = 1e-6
    require_block_label: bool = True

    def __post_init__(self) -> None:
        if self.alignment not in {"CS", "US"}:
            raise ConfigurationError("Model-input alignment must be CS or US.")
        if self.min_response_valid_samples < 1 or self.min_baseline_valid_samples < 1:
            raise ConfigurationError("Coverage minima must be at least 1.")
        if self.activity_offset <= 0:
            raise ConfigurationError("Activity offset must be positive.")


@dataclass(frozen=True)
class ModelInputArtifactResult:
    analysis_id: str
    recording_ids: tuple[str, ...]
    model_input_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int
    fish_count: int


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )


def _config_hash(config: ModelInputConfig) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_authenticated_trial_outcomes(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    metric_recipe: str = "tail-candidate-corrected-v1",
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], str]:
    """Load and authenticate candidate trial-outcome Parquet for many recordings."""
    recording_ids = tuple(dict.fromkeys(recording_ids))
    if not recording_ids:
        raise ConfigurationError("At least one recording ID is required.")
    route = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    project_dir = project_dir.resolve()
    frames: list[pd.DataFrame] = []
    inputs: dict[str, dict[str, str]] = {}
    for recording_id in recording_ids:
        outcomes_path = (
            project_dir
            / "Processed data"
            / recording_id
            / route.trial_outcomes_name
        )
        summary_path = (
            project_dir
            / "Quality checks"
            / recording_id
            / route.trial_summary_name
        )
        marker_path = (
            project_dir
            / "Metadata"
            / f"{recording_id}_{route.trial_marker_suffix}"
        )
        missing = [
            path
            for path in (outcomes_path, summary_path, marker_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing trial outcomes for {recording_id}: {missing}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        digest = sha256_file(outcomes_path)
        outcomes_hash = summary.get("artifacts", {}).get("outcomes", {}).get("sha256")
        if (
            summary.get("recipe") != route.trial_recipe
            or summary.get("recording_id") != recording_id
            or outcomes_hash != digest
            or marker.get("status") != "complete"
            or marker.get("recipe") != route.trial_recipe
            or marker.get("recording_id") != recording_id
            or marker.get("artifact_sha256", {}).get("outcomes") != digest
            or marker.get("summary_sha256") != sha256_file(summary_path)
        ):
            raise ArtifactIntegrityError(
                f"Trial outcome lineage is invalid for {recording_id}."
            )
        frames.append(pq.read_table(outcomes_path).to_pandas())
        inputs[recording_id] = {
            "recipe": route.trial_recipe,
            "path": str(outcomes_path),
            "sha256": digest,
        }
    return pd.concat(frames, ignore_index=True), inputs, route.metric_recipe


def build_candidate_model_input(
    outcomes: pd.DataFrame,
    *,
    config: ModelInputConfig = ModelInputConfig(),
) -> pd.DataFrame:
    """Build one long model-input table for every metric and continuous outcome."""
    required = {
        "experiment_id",
        "recording_id",
        "fish_id",
        "condition_id",
        "trial_id",
        "alignment",
        "trial_number",
        "block_10_name",
        "metric_id",
        "baseline_valid_sample_count",
        "response_valid_sample_count",
        *(column for pair in OUTCOME_SPECS.values() for column in pair),
    }
    missing = required.difference(outcomes.columns)
    if missing:
        raise SchemaValidationError(
            f"Trial outcomes are missing columns: {sorted(missing)}"
        )

    frame = outcomes.loc[outcomes["alignment"].astype(str) == config.alignment].copy()
    rows: list[dict[str, Any]] = []
    for outcome_id, (response_column, baseline_column) in OUTCOME_SPECS.items():
        subset = frame.loc[
            (frame["response_valid_sample_count"] >= config.min_response_valid_samples)
            & (
                frame["baseline_valid_sample_count"]
                >= config.min_baseline_valid_samples
            )
        ]
        for _, row in subset.iterrows():
            response = float(row[response_column])
            baseline = float(row[baseline_column])
            if not np.isfinite(response) or not np.isfinite(baseline):
                continue
            if response < 0 or baseline < 0:
                continue
            rows.append(
                {
                    "experiment_id": str(row["experiment_id"]),
                    "recording_id": str(row["recording_id"]),
                    "fish_id": str(row["fish_id"]),
                    "condition_id": str(row["condition_id"]),
                    "trial_id": str(row["trial_id"]),
                    "alignment": str(row["alignment"]),
                    "trial_number": int(row["trial_number"]),
                    "block_10_name": (
                        None
                        if pd.isna(row["block_10_name"])
                        else str(row["block_10_name"])
                    ),
                    "metric_id": str(row["metric_id"]),
                    "outcome_id": outcome_id,
                    "response": response,
                    "baseline": baseline,
                    "log_response": float(np.log(response + config.activity_offset)),
                    "log_baseline": float(np.log(baseline + config.activity_offset)),
                }
            )
    if not rows:
        raise ConfigurationError("No retained rows for candidate model input.")
    frame_out = pd.DataFrame(rows).loc[:, list(MODEL_INPUT_COLUMNS)]
    if config.require_block_label:
        frame_out = frame_out.loc[frame_out["block_10_name"].notna()].reset_index(
            drop=True
        )
        if frame_out.empty:
            raise ConfigurationError(
                "No retained rows with non-null block_10_name for model input."
            )
    return frame_out


def model_input_coverage(model_input: pd.DataFrame) -> dict[str, Any]:
    """Compact inclusion counts for summaries and diagnostics."""
    return {
        "row_count": int(len(model_input)),
        "fish_count": int(model_input["fish_id"].nunique()) if len(model_input) else 0,
        "recording_count": (
            int(model_input["recording_id"].nunique()) if len(model_input) else 0
        ),
        "trial_count": int(model_input["trial_id"].nunique()) if len(model_input) else 0,
        "metric_ids": sorted(model_input["metric_id"].astype(str).unique().tolist())
        if len(model_input)
        else [],
        "outcome_ids": sorted(model_input["outcome_id"].astype(str).unique().tolist())
        if len(model_input)
        else [],
        "block_10_names": sorted(
            model_input["block_10_name"].dropna().astype(str).unique().tolist()
        )
        if len(model_input)
        else [],
    }


def empty_fit_diagnostics_frame() -> pd.DataFrame:
    """Canonical diagnostic columns for population-fit recipes."""
    return pd.DataFrame(
        columns=[
            "metric_id",
            "outcome_id",
            "observation_count",
            "fish_count",
            "converged",
            "singular_covariance",
            "diagnostic_status",
            "error",
            "warnings",
        ]
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": len(table.column_names),
        "size_bytes": path.stat().st_size,
        "compression": "zstd",
        "compression_lossless": True,
    }


def build_candidate_model_input_artifact(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    metric_recipe: str = "tail-candidate-corrected-v1",
    config: ModelInputConfig = ModelInputConfig(),
    overwrite: bool = False,
) -> ModelInputArtifactResult:
    """Publish a standalone frozen model-input artifact (Step 10.2)."""
    _validate_analysis_id(analysis_id)
    recording_ids = tuple(dict.fromkeys(recording_ids))
    project_dir = project_dir.resolve()
    outcomes, inputs, resolved_metric = load_authenticated_trial_outcomes(
        project_dir,
        recording_ids,
        metric_recipe=metric_recipe,
    )
    model_input = build_candidate_model_input(outcomes, config=config)
    coverage = model_input_coverage(model_input)

    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    model_input_path = output_dir / f"{RECIPE_ID}.parquet"
    summary_path = (
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{RECIPE_ID}_summary.json"
    )
    marker_path = project_dir / "Metadata" / f"{analysis_id}_{RECIPE_ID}_complete.json"
    existing = [
        path for path in (model_input_path, summary_path, marker_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{analysis_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_table = staging_root / model_input_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        record = _write_parquet(staged_table, model_input)
        record["path"] = str(model_input_path)
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "exploratory_model_input",
            "paper_approved": False,
            "gate_s_frozen": False,
            "analysis_id": analysis_id,
            "model_input_id": analysis_id,
            "metric_recipe": resolved_metric,
            "recording_ids": list(recording_ids),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "config_sha256": _config_hash(config),
            "transforms": {
                "log_response": "log(response + activity_offset)",
                "log_baseline": "log(baseline + activity_offset)",
                "activity_offset": config.activity_offset,
            },
            "coverage": coverage,
            "inputs": inputs,
            "artifact": record,
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "analysis_id": analysis_id,
                "recording_ids": list(recording_ids),
                "model_input_sha256": record["sha256"],
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        publish_transaction(
            (
                (staged_table, model_input_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return ModelInputArtifactResult(
        analysis_id=analysis_id,
        recording_ids=recording_ids,
        model_input_path=model_input_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_count=int(coverage["row_count"]),
        fish_count=int(coverage["fish_count"]),
    )
