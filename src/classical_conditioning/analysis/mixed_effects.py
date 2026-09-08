"""Exploratory fish-aware mixed-effects scaffold for candidate trial outcomes.

This is an engineering default aligned with Gate S in DECISIONS.md, not an
approved confirmatory model. See Plans/STATISTICS_METHODOLOGY_WORKSHOP.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.model_input import (
    OUTCOME_SPECS,
    ModelInputConfig,
    build_candidate_model_input,
    load_authenticated_trial_outcomes,
    model_input_coverage,
)
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.exceptions import ConfigurationError

RECIPE_ID = "candidate-mixed-effects-v1"


@dataclass(frozen=True)
class MixedEffectsConfig:
    alignment: str = "CS"
    min_response_valid_samples: int = 1
    min_baseline_valid_samples: int = 1
    activity_offset: float = 1e-6
    fixed_effects_formula: str = "log_response ~ log_baseline + C(block_10_name)"
    random_effects_formula: str = "1"
    groups_column: str = "fish_id"
    optimizer: str = "lbfgs"
    reml: bool = False

    def __post_init__(self) -> None:
        if self.alignment not in {"CS", "US"}:
            raise ConfigurationError("Mixed-effects alignment must be CS or US.")
        if self.min_response_valid_samples < 1 or self.min_baseline_valid_samples < 1:
            raise ConfigurationError("Coverage minima must be at least 1.")
        if self.activity_offset <= 0:
            raise ConfigurationError("Activity offset must be positive.")
        if self.groups_column != "fish_id":
            raise ConfigurationError(
                "Population models must group repeated measures by fish_id."
            )

    def model_input_config(self) -> ModelInputConfig:
        return ModelInputConfig(
            alignment=self.alignment,
            min_response_valid_samples=self.min_response_valid_samples,
            min_baseline_valid_samples=self.min_baseline_valid_samples,
            activity_offset=self.activity_offset,
            require_block_label=True,
        )


@dataclass(frozen=True)
class MixedEffectsResult:
    analysis_id: str
    recording_ids: tuple[str, ...]
    model_input_path: Path
    coefficients_path: Path
    diagnostics_path: Path
    summary_path: Path
    completion_marker_path: Path


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )


def _config_hash(config: MixedEffectsConfig) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_mixed_effects_model_input(
    outcomes: pd.DataFrame,
    *,
    config: MixedEffectsConfig = MixedEffectsConfig(),
) -> pd.DataFrame:
    """Build one long model-input table for every metric and continuous outcome."""
    return build_candidate_model_input(outcomes, config=config.model_input_config())


def _fit_one_metric_outcome(
    model_input: pd.DataFrame,
    *,
    metric_id: str,
    outcome_id: str,
    config: MixedEffectsConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subset = model_input.loc[
        (model_input["metric_id"] == metric_id)
        & (model_input["outcome_id"] == outcome_id)
    ].copy()
    diagnostic: dict[str, Any] = {
        "metric_id": metric_id,
        "outcome_id": outcome_id,
        "observation_count": int(len(subset)),
        "fish_count": int(subset["fish_id"].nunique()) if len(subset) else 0,
        "converged": False,
        "singular_covariance": False,
        "diagnostic_status": "failed",
        "error": None,
        "warnings": [],
    }
    if subset.empty:
        diagnostic["error"] = "No retained rows."
        return [], diagnostic
    if diagnostic["fish_count"] < 2:
        diagnostic["error"] = "At least two fish are required for population models."
        return [], diagnostic

    try:
        import statsmodels.formula.api as smf
    except ImportError as error:
        diagnostic["error"] = f"statsmodels unavailable: {error}"
        return [], diagnostic

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                config.fixed_effects_formula,
                subset,
                groups=subset[config.groups_column],
                re_formula=config.random_effects_formula,
            )
            result = model.fit(reml=config.reml, method=config.optimizer)
            diagnostic["warnings"] = [str(item.message) for item in caught]
    except Exception as error:  # noqa: BLE001 - capture fit failures as diagnostics
        diagnostic["error"] = str(error)
        return [], diagnostic

    converged = bool(getattr(result, "converged", False))
    singular = any(
        "singular" in message.lower() for message in diagnostic["warnings"]
    )
    diagnostic_status = "ok"
    if not converged:
        diagnostic_status = "failed"
    elif singular:
        diagnostic_status = "singular"
    diagnostic.update(
        {
            "converged": converged,
            "diagnostic_status": diagnostic_status,
            "singular_covariance": singular,
            "optimizer": config.optimizer,
            "reml": config.reml,
            "llf": float(result.llf) if np.isfinite(result.llf) else None,
            "aic": float(result.aic)
            if np.isfinite(getattr(result, "aic", np.nan))
            else None,
            "random_effect_variance": float(result.cov_re.iloc[0, 0])
            if getattr(result, "cov_re", None) is not None
            else None,
            "scale": float(result.scale) if np.isfinite(result.scale) else None,
            "error": None
            if diagnostic_status == "ok"
            else (
                "Model reported non-convergence."
                if not converged
                else "Random-effects covariance is singular."
            ),
        }
    )
    coefficients = [
        {
            "metric_id": metric_id,
            "outcome_id": outcome_id,
            "term": str(term),
            "estimate": float(result.params[term]),
            "standard_error": float(result.bse[term]),
            "p_value": float(result.pvalues[term]),
            "converged": converged,
            "diagnostic_status": diagnostic["diagnostic_status"],
        }
        for term in result.params.index
    ]
    return coefficients, diagnostic


def fit_candidate_mixed_effects(
    model_input: pd.DataFrame,
    *,
    config: MixedEffectsConfig = MixedEffectsConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the same exploratory mixed model for every metric × outcome."""
    coefficient_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    pairs = (
        model_input.loc[:, ["metric_id", "outcome_id"]]
        .drop_duplicates()
        .sort_values(["metric_id", "outcome_id"])
    )
    for _, pair in pairs.iterrows():
        coefficients, diagnostic = _fit_one_metric_outcome(
            model_input,
            metric_id=str(pair["metric_id"]),
            outcome_id=str(pair["outcome_id"]),
            config=config,
        )
        coefficient_rows.extend(coefficients)
        diagnostic_rows.append(diagnostic)
    return pd.DataFrame(coefficient_rows), pd.DataFrame(diagnostic_rows)


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


def build_candidate_mixed_effects(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    metric_recipe: str = "tail-candidate-corrected-v1",
    config: MixedEffectsConfig = MixedEffectsConfig(),
    overwrite: bool = False,
) -> MixedEffectsResult:
    """Fit exploratory mixed-effects models on authenticated trial outcomes."""
    _validate_analysis_id(analysis_id)
    recording_ids = tuple(dict.fromkeys(recording_ids))
    if not recording_ids:
        raise ConfigurationError("At least one recording ID is required.")
    project_dir = project_dir.resolve()
    outcomes, inputs, resolved_metric = load_authenticated_trial_outcomes(
        project_dir,
        recording_ids,
        metric_recipe=metric_recipe,
    )
    model_input = build_mixed_effects_model_input(outcomes, config=config)
    coverage = model_input_coverage(model_input)
    coefficients, diagnostics = fit_candidate_mixed_effects(model_input, config=config)

    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    model_input_path = output_dir / f"{RECIPE_ID}_model_input.parquet"
    coefficients_path = output_dir / f"{RECIPE_ID}_coefficients.parquet"
    diagnostics_path = output_dir / f"{RECIPE_ID}_diagnostics.parquet"
    summary_path = (
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{RECIPE_ID}_summary.json"
    )
    marker_path = project_dir / "Metadata" / f"{analysis_id}_{RECIPE_ID}_complete.json"
    existing = [
        path
        for path in (
            model_input_path,
            coefficients_path,
            diagnostics_path,
            summary_path,
            marker_path,
        )
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{analysis_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged = {
            "model_input": staging_root / model_input_path.name,
            "coefficients": staging_root / coefficients_path.name,
            "diagnostics": staging_root / diagnostics_path.name,
        }
        records = {
            "model_input": _write_parquet(staged["model_input"], model_input),
            "coefficients": _write_parquet(staged["coefficients"], coefficients),
            "diagnostics": _write_parquet(staged["diagnostics"], diagnostics),
        }
        for name, final_path in {
            "model_input": model_input_path,
            "coefficients": coefficients_path,
            "diagnostics": diagnostics_path,
        }.items():
            records[name]["path"] = str(final_path)

        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        failed = (
            int((diagnostics["diagnostic_status"] != "ok").sum())
            if len(diagnostics)
            else 0
        )
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "exploratory_mixed_effects",
            "paper_approved": False,
            "gate_s_frozen": False,
            "analysis_id": analysis_id,
            "model_input_id": f"{analysis_id}:model-input",
            "metric_recipe": resolved_metric,
            "recording_ids": list(recording_ids),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "config_sha256": _config_hash(config),
            "coverage": coverage,
            "outcome_ids": list(OUTCOME_SPECS),
            "workshop_note": "Plans/STATISTICS_METHODOLOGY_WORKSHOP.md",
            "inputs": inputs,
            "artifacts": records,
            "fit_summary": {
                "model_count": int(len(diagnostics)),
                "failed_or_nonconverged_count": failed,
            },
            "inference": {
                "performed": True,
                "confirmatory": False,
                "reason": (
                    "Engineering default fish-grouped LME for pipeline tests and "
                    "five-metric plumbing. Gate S and the statistics workshop must "
                    "approve any confirmatory claim."
                ),
            },
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "analysis_id": analysis_id,
                "recording_ids": list(recording_ids),
                "artifact_sha256": {
                    name: record["sha256"] for name, record in records.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        publish_transaction(
            (
                (staged["model_input"], model_input_path),
                (staged["coefficients"], coefficients_path),
                (staged["diagnostics"], diagnostics_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return MixedEffectsResult(
        analysis_id=analysis_id,
        recording_ids=recording_ids,
        model_input_path=model_input_path,
        coefficients_path=coefficients_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
    )
