"""Condition-aware block, trial, and learning-onset inference.

Review note: this module carries the exploratory model/diagnostic decisions in
explicit tables and artifacts. It does not make an approval claim merely because
a model converges or a plotted trajectory appears plausible.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.cohort_outcomes import (
    build_analysis_eligibility,
    load_cohort_trial_outcomes,
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

RECIPE_ID = "learning-onset"
BLOCK_ORDER = (
    "Pre-train",
    "Train 1",
    "Train 2",
    "Train 3",
    "Train 4",
    "Train 5",
    "Test 1",
    "Test 2",
    "Test 3",
)
OUTCOME_COLUMNS = {
    "total-activity": ("response_total_activity", "baseline_total_activity"),
    "conditional-intensity": (
        "conditional_intensity",
        "baseline_conditional_intensity",
    ),
}


@dataclass(frozen=True)
class LearningOnsetConfig:
    # Freeze the scientific comparison, model specification, and resampling
    # settings so an onset estimate always carries its full interpretation.
    metric_id: str
    outcome_id: str = "total-activity"
    alignment: str = "CS"
    control_condition: str = "control"
    test_condition: str = "delay"
    pretraining_block: str = "Pre-train"
    late_blocks: tuple[str, ...] = ("Test 2", "Test 3")
    delta_min: float = 0.0
    persistence_trials: int = 3
    confidence_level: float = 0.95
    spline_df: int = 5
    run_categorical_sensitivity: bool = True
    activity_offset: float = 1e-6
    min_baseline_samples: int = 1
    min_response_samples: int = 1
    random_effects_formula: str = "1 + trial_scaled"
    allow_random_intercept_fallback: bool = True
    optimizer: str = "lbfgs"
    sensitivity_optimizer: str | None = "powell"
    run_random_intercept_sensitivity: bool = True
    n_bootstrap: int = 499
    min_successful_bootstrap: int = 100
    min_bootstrap_success_fraction: float = 0.8
    n_permutations: int = 9_999
    seed: int = 20260917

    def __post_init__(self) -> None:
        # Reject incompatible conditions, unknown blocks, or insufficient
        # resampling settings before any fitted artifacts are written.
        if self.outcome_id not in OUTCOME_COLUMNS:
            raise ConfigurationError(
                "Learning onset supports total-activity and conditional-intensity."
            )
        if self.alignment not in {"CS", "US"}:
            raise ConfigurationError("Learning-onset alignment must be CS or US.")
        if self.control_condition == self.test_condition:
            raise ConfigurationError("Control and test conditions must differ.")
        if self.pretraining_block not in BLOCK_ORDER:
            raise ConfigurationError("Pre-training block is not recognized.")
        if not self.late_blocks or not set(self.late_blocks).issubset(BLOCK_ORDER):
            raise ConfigurationError("Late blocks must be named configured blocks.")
        if self.persistence_trials < 1:
            raise ConfigurationError("Persistence length must be positive.")
        if not 0.5 < self.confidence_level < 1.0:
            raise ConfigurationError("Confidence level must be in (0.5, 1).")
        if self.spline_df < 3:
            raise ConfigurationError("Spline degrees of freedom must be at least 3.")
        if self.activity_offset <= 0:
            raise ConfigurationError("Activity offset must be positive.")
        if self.n_bootstrap < 0 or self.n_permutations < 99:
            raise ConfigurationError(
                "Bootstrap count cannot be negative; use at least 99 permutations."
            )
        if self.min_successful_bootstrap < 1:
            raise ConfigurationError(
                "Minimum successful bootstrap count must be positive."
            )
        if (
            self.n_bootstrap > 0
            and self.min_successful_bootstrap > self.n_bootstrap
        ):
            raise ConfigurationError(
                "Minimum successful bootstrap count exceeds requested replicates."
            )
        if not 0 < self.min_bootstrap_success_fraction <= 1:
            raise ConfigurationError(
                "Bootstrap success fraction must be in (0, 1]."
            )


@dataclass(frozen=True)
class LearningOnsetResult:
    # Return a typed index of all output artifacts rather than ambiguous paths
    # assembled again by downstream consumers.
    analysis_id: str
    cohort_id: str
    cohort_hash: str
    model_input_path: Path
    eligibility_path: Path
    block_global_test_path: Path
    block_coefficients_path: Path
    block_contrasts_path: Path
    longitudinal_coefficients_path: Path
    trial_contrasts_path: Path
    adjusted_trajectories_path: Path
    categorical_trial_contrasts_path: Path
    onset_path: Path
    fish_effects_path: Path
    robustness_path: Path
    diagnostics_path: Path
    model_sensitivity_path: Path
    residuals_path: Path
    coverage_path: Path
    influence_path: Path
    fish_trajectory_path: Path
    group_trajectory_path: Path
    bootstrap_onsets_path: Path
    summary_path: Path
    completion_marker_path: Path


def _analysis_paths(
    project_dir: Path,
    analysis_id: str,
) -> tuple[dict[str, Path], Path, Path]:
    # Centralize the stable analysis and quality-check layout used for writing
    # and for later integrity verification.
    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    quality_dir = project_dir / "Quality checks" / "Analyses" / analysis_id
    paths = {
        "model_input": output_dir / "learning-model-input.parquet",
        "eligibility": output_dir / "analysis-eligibility.parquet",
        "block_global_test": output_dir / "block-global-test.parquet",
        "block_coefficients": output_dir / "block-model-coefficients.parquet",
        "block_contrasts": output_dir / "block-contrasts.parquet",
        "longitudinal_coefficients": (
            output_dir / "longitudinal-model-coefficients.parquet"
        ),
        "trial_contrasts": output_dir / "trial-contrasts.parquet",
        "adjusted_trajectories": output_dir / "adjusted-trajectories.parquet",
        "categorical_trial_contrasts": (
            output_dir / "categorical-trial-contrasts.parquet"
        ),
        "onset": output_dir / "learning-onset.parquet",
        "fish_effects": output_dir / "fish-learning-effects.parquet",
        "robustness": output_dir / "fish-robustness.parquet",
        "diagnostics": quality_dir / "learning-model-diagnostics.parquet",
        "model_sensitivity": quality_dir / "learning-model-sensitivity.parquet",
        "residuals": quality_dir / "learning-model-residuals.parquet",
        "coverage": quality_dir / "learning-model-coverage.parquet",
        "influence": quality_dir / "leave-one-fish-out.parquet",
        "fish_trajectory": output_dir / "figure-fish-trajectories.parquet",
        "group_trajectory": output_dir / "figure-group-trajectories.parquet",
        "bootstrap_trials": output_dir / "bootstrap-trial-contrasts.parquet",
        "bootstrap_onsets": output_dir / "bootstrap-onsets.parquet",
    }
    summary = quality_dir / f"{RECIPE_ID}_summary.json"
    marker = project_dir / "Metadata" / f"{analysis_id}_{RECIPE_ID}_complete.json"
    return paths, summary, marker


def load_learning_onset_analysis(
    project_dir: Path,
    analysis_id: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load and authenticate every learning-onset result table."""
    _validate_identifier(analysis_id, "Analysis ID")
    project_dir = project_dir.resolve()
    paths, summary_path, marker_path = _analysis_paths(project_dir, analysis_id)
    missing = [
        path
        for path in (*paths.values(), summary_path, marker_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing learning-onset artifacts: {missing}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    observed_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if (
        summary.get("recipe") != RECIPE_ID
        or summary.get("analysis_id") != analysis_id
        or marker.get("status") != "complete"
        or marker.get("recipe") != RECIPE_ID
        or marker.get("analysis_id") != analysis_id
        or marker.get("artifact_sha256") != observed_hashes
        or marker.get("summary_sha256") != sha256_file(summary_path)
        or marker.get("cohort_hash") != summary.get("cohort_hash")
        or marker.get("config_sha256") != summary.get("config_sha256")
    ):
        raise ArtifactIntegrityError(
            f"Learning-onset lineage is invalid for {analysis_id}."
        )
    frames = {name: pq.read_table(path).to_pandas() for name, path in paths.items()}
    return frames, summary


def _validate_identifier(value: str, label: str) -> None:
    # Allow only portable identifier characters before a value reaches paths.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ConfigurationError(
            f"{label} must use only letters, numbers, dot, underscore, or hyphen."
        )


def _config_hash(config: LearningOnsetConfig) -> str:
    # Hash canonical JSON so the recipe fingerprint is repeatable and inspectable.
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_learning_model_input(
    outcomes: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    config: LearningOnsetConfig,
) -> pd.DataFrame:
    """Create one eligible, directed model table without changing the cohort."""
    response_column, baseline_column = OUTCOME_COLUMNS[config.outcome_id]
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
        response_column,
        baseline_column,
    }
    missing = required.difference(outcomes.columns)
    if missing:
        raise SchemaValidationError(
            f"Cohort outcomes are missing model columns: {sorted(missing)}"
        )
    eligible_keys = eligibility.loc[
        eligibility["eligible"], ["trial_id", "metric_id", "outcome_id"]
    ]
    selected = outcomes.loc[
        (outcomes["alignment"].astype(str) == config.alignment)
        & (outcomes["metric_id"].astype(str) == config.metric_id)
        & outcomes["condition_id"].astype(str).isin(
            [config.control_condition, config.test_condition]
        )
    ].copy()
    selected = selected.merge(
        eligible_keys,
        left_on=["trial_id", "metric_id"],
        right_on=["trial_id", "metric_id"],
        how="inner",
        validate="one_to_one",
    )
    if selected.empty:
        raise ConfigurationError("No eligible rows remain for learning-onset analysis.")
    conditions = set(selected["condition_id"].astype(str))
    required_conditions = {config.control_condition, config.test_condition}
    if conditions != required_conditions:
        raise ConfigurationError(
            "Learning-onset data must contain exactly the configured control and "
            f"test conditions; observed={sorted(conditions)}."
        )
    selected["response"] = selected[response_column].astype(float)
    selected["baseline"] = selected[baseline_column].astype(float)
    selected["fish_key"] = (
        selected["experiment_id"].astype(str)
        + "::"
        + selected["fish_id"].astype(str)
    )
    offset = (
        0.0
        if config.outcome_id == "conditional-intensity"
        else config.activity_offset
    )
    selected["log_response"] = np.log(selected["response"] + offset)
    selected["log_baseline"] = np.log(selected["baseline"] + offset)
    if not (
        np.isfinite(selected["log_response"]).all()
        and np.isfinite(selected["log_baseline"]).all()
    ):
        raise SchemaValidationError(
            "Eligibility admitted a response or baseline that cannot be logged."
        )
    selected["cr_score"] = selected["log_baseline"] - selected["log_response"]
    trial_mean = float(selected["trial_number"].mean())
    trial_scale = float(selected["trial_number"].std(ddof=0))
    if not np.isfinite(trial_scale) or trial_scale <= 0:
        raise ConfigurationError("Trial number has no usable variation.")
    selected["trial_center"] = trial_mean
    selected["trial_scale"] = trial_scale
    selected["trial_scaled"] = (
        selected["trial_number"].astype(float) - trial_mean
    ) / trial_scale
    selected["condition_id"] = pd.Categorical(
        selected["condition_id"],
        categories=[config.control_condition, config.test_condition],
        ordered=True,
    )
    present_blocks = [
        block for block in BLOCK_ORDER if block in set(selected["block_10_name"])
    ]
    selected["block_10_name"] = pd.Categorical(
        selected["block_10_name"], categories=present_blocks, ordered=True
    )
    return selected.reset_index(drop=True)


def _fit_mixed_model(
    data: pd.DataFrame,
    *,
    formula: str,
    config: LearningOnsetConfig,
    group_column: str = "fish_key",
    collect_extended_diagnostics: bool = True,
) -> tuple[Any | None, dict[str, Any]]:
    # Attempt the requested mixed model and preserve convergence or fallback
    # information as diagnostics instead of concealing fit failures.
    diagnostic: dict[str, Any] = {
        "formula": formula,
        "requested_random_effects": config.random_effects_formula,
        "used_random_effects": None,
        "optimizer": config.optimizer,
        "observation_count": len(data),
        "fish_count": int(data[group_column].nunique()),
        "condition_fish_counts_json": json.dumps(
            {
                str(key): int(value)
                for key, value in data.groupby("condition_id", observed=True)[
                    group_column
                ].nunique().items()
            },
            sort_keys=True,
        ),
        "converged": False,
        "singular": False,
        "fallback_used": False,
        "warnings_json": "[]",
        "error": None,
        "diagnostic_status": "failed",
    }
    try:
        import statsmodels.formula.api as smf
    except ImportError as error:
        diagnostic["error"] = f"statsmodels unavailable: {error}"
        return None, diagnostic
    candidates = [config.random_effects_formula]
    if config.allow_random_intercept_fallback and config.random_effects_formula != "1":
        candidates.append("1")
    errors: list[str] = []
    for index, random_formula in enumerate(candidates):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    data,
                    groups=data[group_column],
                    re_formula=random_formula,
                )
                result = model.fit(reml=False, method=config.optimizer)
            covariance = np.asarray(result.cov_re, dtype=float)
            eigenvalues = np.linalg.eigvalsh(covariance)
            largest = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
            smallest = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
            singular = bool(
                eigenvalues.size
                and (smallest <= 1e-8 or smallest <= largest * 1e-6)
            )
            converged = bool(getattr(result, "converged", False))
            fixed_rank = int(np.linalg.matrix_rank(model.exog))
            fixed_columns = int(model.exog.shape[1])
            rank_deficient = fixed_rank < fixed_columns
            hessian_min_eigenvalue = np.nan
            hessian_max_eigenvalue = np.nan
            hessian_error = None
            if collect_extended_diagnostics:
                try:
                    hessian_output = model.hessian(result.params)
                    hessian = np.asarray(
                        hessian_output[0]
                        if isinstance(hessian_output, tuple)
                        else hessian_output,
                        dtype=float,
                    )
                    hessian_eigenvalues = np.linalg.eigvalsh(
                        (hessian + hessian.T) / 2.0
                    )
                    hessian_min_eigenvalue = float(np.min(hessian_eigenvalues))
                    hessian_max_eigenvalue = float(np.max(hessian_eigenvalues))
                except Exception as error:  # noqa: BLE001 - diagnostic only
                    hessian_error = str(error)
            diagnostic.update(
                {
                    "used_random_effects": random_formula,
                    "fallback_used": index > 0,
                    "converged": converged,
                    "singular": singular,
                    "random_effect_min_eigenvalue": smallest,
                    "random_effect_max_eigenvalue": largest,
                    "random_effect_covariance_json": json.dumps(
                        covariance.tolist()
                    ),
                    "hessian_min_eigenvalue": hessian_min_eigenvalue,
                    "hessian_max_eigenvalue": hessian_max_eigenvalue,
                    "hessian_error": hessian_error,
                    "optimizer_details_json": json.dumps(
                        getattr(result, "mle_retvals", {}),
                        default=str,
                        sort_keys=True,
                    ),
                    "warnings_json": json.dumps(
                        [str(item.message) for item in caught]
                    ),
                    "fixed_design_rank": fixed_rank,
                    "fixed_design_columns": fixed_columns,
                    "rank_deficient": rank_deficient,
                    "aic": float(result.aic) if np.isfinite(result.aic) else None,
                    "bic": float(result.bic) if np.isfinite(result.bic) else None,
                    "llf": float(result.llf) if np.isfinite(result.llf) else None,
                }
            )
            diagnostic["diagnostic_status"] = (
                "ok"
                if converged and not singular and not rank_deficient
                else "failed"
            )
            if diagnostic["diagnostic_status"] == "ok":
                return result, diagnostic
            errors.append(
                f"{random_formula}: converged={converged}, singular={singular}, "
                f"rank_deficient={rank_deficient}"
            )
        except Exception as error:  # noqa: BLE001 - recorded model failure
            errors.append(f"{random_formula}: {error}")
    diagnostic["error"] = "; ".join(errors)
    return None, diagnostic


def _fixed_design_row(result: Any, frame: pd.DataFrame) -> np.ndarray:
    # Recreate the fitted fixed-effects design row for a requested covariate case.
    try:
        from patsy import build_design_matrices
    except ImportError as error:
        raise RuntimeError(f"patsy unavailable: {error}") from error
    design_info = result.model.data.design_info
    matrix = build_design_matrices([design_info], frame, return_type="dataframe")[0]
    return matrix.to_numpy(dtype=float)[0]


def _fixed_covariance(result: Any) -> np.ndarray:
    # Select only fixed-effect covariance entries, excluding random-effect terms.
    names = list(result.fe_params.index)
    covariance = result.cov_params().loc[names, names]
    return covariance.to_numpy(dtype=float)


def model_coefficients(
    result: Any | None,
    *,
    model_name: str,
    confidence_level: float,
) -> pd.DataFrame:
    """Return a complete named fixed-effect coefficient table."""
    columns = [
        "model",
        "term",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "p_value",
    ]
    if result is None:
        return pd.DataFrame(columns=columns)
    covariance = _fixed_covariance(result)
    estimates = result.fe_params.to_numpy(dtype=float)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    rows = []
    for index, term in enumerate(result.fe_params.index):
        lower, upper, p_value = _normal_interval(
            float(estimates[index]),
            float(standard_errors[index]),
            confidence_level,
        )
        rows.append(
            {
                "model": model_name,
                "term": str(term),
                "estimate": float(estimates[index]),
                "standard_error": float(standard_errors[index]),
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def adjusted_condition_trajectories(
    result: Any | None,
    model_input: pd.DataFrame,
    *,
    config: LearningOnsetConfig,
) -> pd.DataFrame:
    """Predict adjusted fixed-effect trajectories for both conditions."""
    columns = [
        "condition_id",
        "trial_number",
        "adjusted_log_response",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "reference_log_baseline",
    ]
    if result is None:
        return pd.DataFrame(columns=columns)
    beta = result.fe_params.to_numpy(dtype=float)
    covariance = _fixed_covariance(result)
    baseline = float(model_input["log_baseline"].mean())
    center = float(model_input["trial_center"].iloc[0])
    scale = float(model_input["trial_scale"].iloc[0])
    rows = []
    for condition in (config.control_condition, config.test_condition):
        for trial in sorted(model_input["trial_number"].astype(int).unique()):
            design = _fixed_design_row(
                result,
                pd.DataFrame(
                    {
                        "condition_id": [condition],
                        "trial_number": [trial],
                        "trial_scaled": [(float(trial) - center) / scale],
                        "log_baseline": [baseline],
                    }
                ),
            )
            estimate = float(design @ beta)
            variance = float(design @ covariance @ design)
            standard_error = float(np.sqrt(max(variance, 0.0)))
            lower, upper, _ = _normal_interval(
                estimate, standard_error, config.confidence_level
            )
            rows.append(
                {
                    "condition_id": condition,
                    "trial_number": int(trial),
                    "adjusted_log_response": estimate,
                    "standard_error": standard_error,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "reference_log_baseline": baseline,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def model_residuals(
    result: Any | None,
    model_input: pd.DataFrame,
    *,
    model_name: str,
) -> pd.DataFrame:
    """Publish observation-level fitted values and residuals for review."""
    columns = [
        "model",
        "fish_key",
        "experiment_id",
        "fish_id",
        "condition_id",
        "trial_id",
        "trial_number",
        "block_10_name",
        "observed_log_response",
        "fitted_log_response",
        "residual",
        "standardized_residual",
    ]
    if result is None:
        return pd.DataFrame(columns=columns)
    fitted = np.asarray(result.fittedvalues, dtype=float)
    observed = model_input["log_response"].to_numpy(dtype=float)
    residual = observed - fitted
    scale = float(np.sqrt(getattr(result, "scale", np.nan)))
    standardized = residual / scale if np.isfinite(scale) and scale > 0 else np.nan
    frame = model_input.loc[
        :,
        [
            "fish_key",
            "experiment_id",
            "fish_id",
            "condition_id",
            "trial_id",
            "trial_number",
            "block_10_name",
        ],
    ].copy()
    frame.insert(0, "model", model_name)
    frame["observed_log_response"] = observed
    frame["fitted_log_response"] = fitted
    frame["residual"] = residual
    frame["standardized_residual"] = standardized
    return frame.loc[:, columns]


def model_sensitivity_checks(
    primary_result: Any | None,
    model_input: pd.DataFrame,
    *,
    model_name: str,
    formula: str,
    config: LearningOnsetConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Refit approved optimizer/random-structure sensitivities."""
    columns = [
        "model",
        "sensitivity",
        "term",
        "primary_estimate",
        "sensitivity_estimate",
        "absolute_difference",
        "diagnostic_status",
        "error",
    ]
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    variants: list[tuple[str, LearningOnsetConfig]] = []
    if config.sensitivity_optimizer and config.sensitivity_optimizer != config.optimizer:
        variants.append(
            (
                "alternate_optimizer",
                replace(
                    config,
                    optimizer=config.sensitivity_optimizer,
                    allow_random_intercept_fallback=False,
                ),
            )
        )
    if config.run_random_intercept_sensitivity:
        variants.append(
            (
                "random_intercept_only",
                replace(
                    config,
                    random_effects_formula="1",
                    allow_random_intercept_fallback=False,
                ),
            )
        )
    for sensitivity_name, sensitivity_config in variants:
        if primary_result is None:
            diagnostic = {
                "diagnostic_status": "not_run",
                "error": "Primary model failed.",
            }
            sensitivity_result = None
        else:
            sensitivity_result, diagnostic = _fit_mixed_model(
                model_input,
                formula=formula,
                config=sensitivity_config,
            )
        diagnostics.append(
            {
                "model": f"{model_name}_{sensitivity_name}",
                "required_for_publication": False,
                **diagnostic,
            }
        )
        if primary_result is None or sensitivity_result is None:
            rows.append(
                {
                    "model": model_name,
                    "sensitivity": sensitivity_name,
                    "term": None,
                    "primary_estimate": np.nan,
                    "sensitivity_estimate": np.nan,
                    "absolute_difference": np.nan,
                    "diagnostic_status": diagnostic["diagnostic_status"],
                    "error": diagnostic.get("error"),
                }
            )
            continue
        primary = primary_result.fe_params
        alternative = sensitivity_result.fe_params.reindex(primary.index)
        for term in primary.index:
            primary_value = float(primary.loc[term])
            alternative_value = float(alternative.loc[term])
            rows.append(
                {
                    "model": model_name,
                    "sensitivity": sensitivity_name,
                    "term": str(term),
                    "primary_estimate": primary_value,
                    "sensitivity_estimate": alternative_value,
                    "absolute_difference": abs(
                        alternative_value - primary_value
                    ),
                    "diagnostic_status": diagnostic["diagnostic_status"],
                    "error": diagnostic.get("error"),
                }
            )
    if not variants:
        rows.append(
            {
                "model": model_name,
                "sensitivity": "none_configured",
                "term": None,
                "primary_estimate": np.nan,
                "sensitivity_estimate": np.nan,
                "absolute_difference": np.nan,
                "diagnostic_status": "not_run",
                "error": None,
            }
        )
    return pd.DataFrame(rows, columns=columns), diagnostics


def _normal_interval(
    estimate: float,
    standard_error: float,
    confidence_level: float,
) -> tuple[float, float, float]:
    # Calculate a two-sided normal-approximation interval and matching p-value.
    quantile = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    lower = estimate - quantile * standard_error
    upper = estimate + quantile * standard_error
    if standard_error <= 0 or not np.isfinite(standard_error):
        p_value = np.nan
    else:
        z_value = abs(estimate / standard_error)
        p_value = 2.0 * (1.0 - NormalDist().cdf(z_value))
    return float(lower), float(upper), float(p_value)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    # Apply Holm's step-down correction while retaining missing values as missing.
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if not finite_indices.size:
        return adjusted
    ordered = finite_indices[np.argsort(p_values[finite_indices])]
    running = 0.0
    total = len(ordered)
    for rank, index in enumerate(ordered):
        value = min(1.0, float(p_values[index]) * (total - rank))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def block_contrasts(
    result: Any,
    model_input: pd.DataFrame,
    *,
    config: LearningOnsetConfig,
) -> pd.DataFrame:
    """Calculate named control-minus-test block contrasts and pre-change."""
    beta = result.fe_params.to_numpy(dtype=float)
    covariance = _fixed_covariance(result)
    baseline_reference = float(model_input["log_baseline"].mean())
    trial_reference = float(model_input["trial_scaled"].mean())
    blocks = [
        block for block in BLOCK_ORDER if block in set(model_input["block_10_name"])
    ]
    vectors: dict[str, np.ndarray] = {}
    raw_estimates: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for block in blocks:
        common = {
            "block_10_name": [block],
            "log_baseline": [baseline_reference],
            "trial_scaled": [trial_reference],
        }
        control = _fixed_design_row(
            result,
            pd.DataFrame({**common, "condition_id": [config.control_condition]}),
        )
        test = _fixed_design_row(
            result,
            pd.DataFrame({**common, "condition_id": [config.test_condition]}),
        )
        vector = control - test
        vectors[block] = vector
        raw_estimates[block] = float(vector @ beta)
    if config.pretraining_block not in vectors:
        raise ConfigurationError("Pre-training block is absent from model input.")
    pre_vector = vectors[config.pretraining_block]
    pre_estimate = raw_estimates[config.pretraining_block]
    for block in blocks:
        change_vector = vectors[block] - pre_vector
        estimate = float(change_vector @ beta)
        variance = float(change_vector @ covariance @ change_vector)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        lower, upper, p_value = _normal_interval(
            estimate, standard_error, config.confidence_level
        )
        fish_counts = model_input.loc[
            model_input["block_10_name"].astype(str) == block
        ].groupby("condition_id", observed=True)["fish_key"].nunique()
        rows.append(
            {
                "block_10_name": block,
                "raw_control_minus_test": raw_estimates[block],
                "pretraining_raw_control_minus_test": pre_estimate,
                "learning_contrast": estimate,
                "standard_error": standard_error,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_value,
                "control_fish_count": int(
                    fish_counts.get(config.control_condition, 0)
                ),
                "test_fish_count": int(fish_counts.get(config.test_condition, 0)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_value_holm"] = _holm_adjust(frame["p_value"].to_numpy(dtype=float))
    frame["supported"] = (
        (frame["ci_lower"] > config.delta_min)
        & (frame["p_value_holm"] < 1.0 - config.confidence_level)
    )
    return frame


def block_global_interaction_test(result: Any) -> pd.DataFrame:
    """Joint Wald test for every condition-by-block interaction coefficient."""
    from scipy.stats import chi2

    names = list(result.fe_params.index)
    indices = [
        index
        for index, name in enumerate(names)
        if ":" in name and "condition_id" in name and "block_10_name" in name
    ]
    if not indices:
        return pd.DataFrame(
            [
                {
                    "test": "condition_by_block",
                    "degrees_of_freedom": 0,
                    "wald_chi_square": np.nan,
                    "p_value": np.nan,
                    "diagnostic_status": "failed",
                    "error": "No condition-by-block interaction coefficients.",
                }
            ]
        )
    beta = result.fe_params.to_numpy(dtype=float)[indices]
    covariance = _fixed_covariance(result)[np.ix_(indices, indices)]
    try:
        statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
        p_value = float(chi2.sf(statistic, len(indices)))
        status = "ok"
        error = None
    except (ValueError, np.linalg.LinAlgError) as exception:
        statistic = np.nan
        p_value = np.nan
        status = "failed"
        error = str(exception)
    return pd.DataFrame(
        [
            {
                "test": "condition_by_block",
                "degrees_of_freedom": len(indices),
                "wald_chi_square": statistic,
                "p_value": p_value,
                "diagnostic_status": status,
                "error": error,
            }
        ]
    )


def trial_contrasts(
    result: Any,
    model_input: pd.DataFrame,
    *,
    config: LearningOnsetConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Calculate model-adjusted learning contrasts at every scheduled trial."""
    beta = result.fe_params.to_numpy(dtype=float)
    covariance = _fixed_covariance(result)
    baseline_reference = float(model_input["log_baseline"].mean())
    trial_center = float(model_input["trial_center"].iloc[0])
    trial_scale = float(model_input["trial_scale"].iloc[0])
    trials = np.sort(model_input["trial_number"].astype(int).unique())
    pre_trials = np.sort(
        model_input.loc[
            model_input["block_10_name"].astype(str) == config.pretraining_block,
            "trial_number",
        ].astype(int).unique()
    )
    if not pre_trials.size:
        raise ConfigurationError("No pre-training trials are available.")

    def condition_vector(trial_number: int) -> np.ndarray:
        # Express the test-minus-control contrast at one trial in the model's
        # own encoded design space, not through hand-built coefficients.
        scaled = (float(trial_number) - trial_center) / trial_scale
        common = {
            "log_baseline": [baseline_reference],
            "trial_number": [int(trial_number)],
            "trial_scaled": [scaled],
        }
        control = _fixed_design_row(
            result,
            pd.DataFrame({**common, "condition_id": [config.control_condition]}),
        )
        test = _fixed_design_row(
            result,
            pd.DataFrame({**common, "condition_id": [config.test_condition]}),
        )
        return control - test

    raw_vectors = {int(trial): condition_vector(int(trial)) for trial in trials}
    pre_vector = np.mean([raw_vectors[int(trial)] for trial in pre_trials], axis=0)
    contrast_matrix = np.vstack(
        [raw_vectors[int(trial)] - pre_vector for trial in trials]
    )
    estimates = contrast_matrix @ beta
    variances = np.einsum(
        "ij,jk,ik->i", contrast_matrix, covariance, contrast_matrix
    )
    standard_errors = np.sqrt(np.maximum(variances, 0.0))
    quantile = NormalDist().inv_cdf(0.5 + config.confidence_level / 2.0)
    rows = []
    for index, trial in enumerate(trials):
        counts = model_input.loc[
            model_input["trial_number"].astype(int) == int(trial)
        ].groupby("condition_id", observed=True)["fish_key"].nunique()
        rows.append(
            {
                "trial_number": int(trial),
                "learning_contrast": float(estimates[index]),
                "standard_error": float(standard_errors[index]),
                "pointwise_lower": float(
                    estimates[index] - quantile * standard_errors[index]
                ),
                "pointwise_upper": float(
                    estimates[index] + quantile * standard_errors[index]
                ),
                "simultaneous_lower": np.nan,
                "simultaneous_upper": np.nan,
                "control_fish_count": int(
                    counts.get(config.control_condition, 0)
                ),
                "test_fish_count": int(counts.get(config.test_condition, 0)),
                "estimable": bool(np.isfinite(estimates[index])),
            }
        )
    return pd.DataFrame(rows), contrast_matrix


def localize_learning_onset(
    trial_contrast_table: pd.DataFrame,
    *,
    delta_min: float,
    persistence_trials: int,
    lower_column: str = "simultaneous_lower",
) -> dict[str, Any]:
    """Return the earliest consecutive scheduled-trial threshold crossing."""
    required = {"trial_number", lower_column, "estimable"}
    missing = required.difference(trial_contrast_table.columns)
    if missing:
        raise SchemaValidationError(
            f"Trial contrasts are missing onset columns: {sorted(missing)}"
        )
    ordered = trial_contrast_table.sort_values("trial_number").reset_index(drop=True)
    if ordered.empty:
        return {
            "localized": False,
            "onset_trial": None,
            "run_start_trial": None,
            "run_end_trial": None,
            "failure_reason": "no_trial_contrasts",
        }
    supported = (
        ordered["estimable"].astype(bool)
        & np.isfinite(ordered[lower_column].to_numpy(dtype=float))
        & (ordered[lower_column].to_numpy(dtype=float) > delta_min)
    )
    trials = ordered["trial_number"].to_numpy(dtype=int)
    for start in range(0, len(ordered) - persistence_trials + 1):
        end = start + persistence_trials
        window_trials = trials[start:end]
        consecutive = np.all(np.diff(window_trials) == 1)
        if consecutive and bool(np.all(supported.iloc[start:end])):
            return {
                "localized": True,
                "onset_trial": int(window_trials[0]),
                "run_start_trial": int(window_trials[0]),
                "run_end_trial": int(window_trials[-1]),
                "failure_reason": None,
            }
    reason = (
        "threshold_never_exceeded"
        if not supported.any()
        else "threshold_not_persistent"
    )
    return {
        "localized": False,
        "onset_trial": None,
        "run_start_trial": None,
        "run_end_trial": None,
        "failure_reason": reason,
    }


def _resample_fish_within_condition(
    data: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    # Bootstrap complete fish trajectories independently within each condition
    # so repeated trials remain paired and group membership is preserved.
    frames = []
    for condition, condition_data in data.groupby("condition_id", observed=True):
        fish = condition_data["fish_key"].astype(str).unique()
        sampled = rng.choice(fish, size=len(fish), replace=True)
        for copy_index, fish_key in enumerate(sampled):
            fish_rows = condition_data.loc[
                condition_data["fish_key"].astype(str) == str(fish_key)
            ].copy()
            fish_rows["bootstrap_fish_id"] = (
                f"{condition}:{copy_index:04d}:{fish_key}"
            )
            frames.append(fish_rows)
    return pd.concat(frames, ignore_index=True)


def bootstrap_trial_contrasts(
    model_input: pd.DataFrame,
    original: pd.DataFrame,
    *,
    formula: str,
    config: LearningOnsetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fish-bootstrap simultaneous band and onset distribution."""
    if config.n_bootstrap == 0:
        return (
            pd.DataFrame(
                columns=["replicate", "trial_number", "learning_contrast"]
            ),
            pd.DataFrame(
                columns=[
                    "replicate",
                    "localized",
                    "onset_trial",
                    "failure_reason",
                ]
            ),
            np.nan,
        )
    rng = np.random.default_rng(config.seed)
    trial_numbers = original["trial_number"].to_numpy(dtype=int)
    original_estimates = original["learning_contrast"].to_numpy(dtype=float)
    replicates: list[dict[str, Any]] = []
    onset_rows: list[dict[str, Any]] = []
    maximum_deviations: list[float] = []
    for replicate in range(config.n_bootstrap):
        sampled = _resample_fish_within_condition(model_input, rng)
        result, diagnostic = _fit_mixed_model(
            sampled,
            formula=formula,
            config=config,
            group_column="bootstrap_fish_id",
            collect_extended_diagnostics=False,
        )
        if result is None:
            onset_rows.append(
                {
                    "replicate": replicate,
                    "localized": False,
                    "onset_trial": np.nan,
                    "failure_reason": "model_fit_failed",
                }
            )
            continue
        try:
            contrast_table, _ = trial_contrasts(result, sampled, config=config)
        except Exception as error:  # noqa: BLE001 - bootstrap failure record
            onset_rows.append(
                {
                    "replicate": replicate,
                    "localized": False,
                    "onset_trial": np.nan,
                    "failure_reason": f"contrast_failed:{error}",
                }
            )
            continue
        aligned = contrast_table.set_index("trial_number").reindex(trial_numbers)
        values = aligned["learning_contrast"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            onset_rows.append(
                {
                    "replicate": replicate,
                    "localized": False,
                    "onset_trial": np.nan,
                    "failure_reason": "nonfinite_trial_contrast",
                }
            )
            continue
        maximum_deviations.append(float(np.max(np.abs(values - original_estimates))))
        for trial, value in zip(trial_numbers, values, strict=True):
            replicates.append(
                {
                    "replicate": replicate,
                    "trial_number": int(trial),
                    "learning_contrast": float(value),
                }
            )
    alpha = 1.0 - config.confidence_level
    critical = (
        float(np.quantile(maximum_deviations, 1.0 - alpha))
        if maximum_deviations
        else np.nan
    )
    replicate_frame = pd.DataFrame(
        replicates,
        columns=["replicate", "trial_number", "learning_contrast"],
    )
    if np.isfinite(critical) and not replicate_frame.empty:
        for replicate, frame in replicate_frame.groupby("replicate", sort=True):
            frame = frame.sort_values("trial_number")
            bootstrap_for_onset = pd.DataFrame(
                {
                    "trial_number": frame["trial_number"].to_numpy(dtype=int),
                    "bootstrap_lower": (
                        frame["learning_contrast"].to_numpy(dtype=float) - critical
                    ),
                    "estimable": True,
                }
            )
            onset = localize_learning_onset(
                bootstrap_for_onset,
                delta_min=config.delta_min,
                persistence_trials=config.persistence_trials,
                lower_column="bootstrap_lower",
            )
            onset_rows.append({"replicate": int(replicate), **onset})
    onset_frame = pd.DataFrame(
        onset_rows,
        columns=[
            "replicate",
            "localized",
            "onset_trial",
            "failure_reason",
            "run_start_trial",
            "run_end_trial",
        ],
    )
    if not onset_frame.empty:
        onset_frame = onset_frame.sort_values("replicate").reset_index(drop=True)
    return replicate_frame, onset_frame, critical


def fish_level_robustness(
    model_input: pd.DataFrame,
    *,
    config: LearningOnsetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Condition-aware late-minus-pre fish effects, permutation, and bootstrap."""
    rows = []
    for (fish_key, condition), frame in model_input.groupby(
        ["fish_key", "condition_id"], observed=True, sort=True
    ):
        pre = frame.loc[
            frame["block_10_name"].astype(str) == config.pretraining_block,
            "cr_score",
        ]
        late = frame.loc[
            frame["block_10_name"].astype(str).isin(config.late_blocks),
            "cr_score",
        ]
        if pre.empty or late.empty:
            continue
        rows.append(
            {
                "fish_key": str(fish_key),
                "experiment_id": str(frame["experiment_id"].iloc[0]),
                "fish_id": str(frame["fish_id"].iloc[0]),
                "condition_id": str(condition),
                "pretraining_cr_score": float(pre.median()),
                "late_cr_score": float(late.median()),
                "learning_change": float(late.median() - pre.median()),
                "pretraining_trial_count": len(pre),
                "late_trial_count": len(late),
            }
        )
    fish = pd.DataFrame(
        rows,
        columns=[
            "fish_key",
            "experiment_id",
            "fish_id",
            "condition_id",
            "pretraining_cr_score",
            "late_cr_score",
            "learning_change",
            "pretraining_trial_count",
            "late_trial_count",
        ],
    )
    if fish.empty:
        return fish, pd.DataFrame(
            [
                {
                    "test_minus_control": np.nan,
                    "permutation_p_value": np.nan,
                    "bootstrap_lower": np.nan,
                    "bootstrap_upper": np.nan,
                    "control_fish_count": 0,
                    "test_fish_count": 0,
                    "diagnostic_status": "failed",
                    "error": "No fish have both pre-training and late observations.",
                }
            ]
        )
    control = fish.loc[
        fish["condition_id"] == config.control_condition, "learning_change"
    ].to_numpy(dtype=float)
    test = fish.loc[
        fish["condition_id"] == config.test_condition, "learning_change"
    ].to_numpy(dtype=float)
    if len(control) < 2 or len(test) < 2:
        result = pd.DataFrame(
            [
                {
                    "test_minus_control": np.nan,
                    "permutation_p_value": np.nan,
                    "bootstrap_lower": np.nan,
                    "bootstrap_upper": np.nan,
                    "diagnostic_status": "failed",
                    "error": "At least two fish per condition are required.",
                }
            ]
        )
        return fish, result
    observed = float(np.mean(test) - np.mean(control))
    values = np.concatenate([control, test])
    n_control = len(control)
    rng = np.random.default_rng(config.seed)
    permuted = np.empty(config.n_permutations, dtype=float)
    for index in range(config.n_permutations):
        shuffled = rng.permutation(values)
        permuted[index] = float(
            np.mean(shuffled[n_control:]) - np.mean(shuffled[:n_control])
        )
    p_value = float(
        (1 + np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-15))
        / (config.n_permutations + 1)
    )
    bootstrap_count = max(config.n_bootstrap, 999)
    bootstrap = np.empty(bootstrap_count, dtype=float)
    for index in range(bootstrap_count):
        bootstrap[index] = float(
            np.mean(rng.choice(test, size=len(test), replace=True))
            - np.mean(rng.choice(control, size=len(control), replace=True))
        )
    alpha = 1.0 - config.confidence_level
    result = pd.DataFrame(
        [
            {
                "test_minus_control": observed,
                "permutation_p_value": p_value,
                "bootstrap_lower": float(np.quantile(bootstrap, alpha / 2.0)),
                "bootstrap_upper": float(
                    np.quantile(bootstrap, 1.0 - alpha / 2.0)
                ),
                "control_fish_count": len(control),
                "test_fish_count": len(test),
                "diagnostic_status": "ok",
                "error": None,
            }
        ]
    )
    return fish, result


def leave_one_fish_out(
    model_input: pd.DataFrame,
    *,
    block_formula: str,
    trial_formula: str,
    simultaneous_critical_distance: float,
    config: LearningOnsetConfig,
) -> pd.DataFrame:
    """Refit after omitting each fish and track the primary conclusions."""
    rows: list[dict[str, Any]] = []
    primary_late_block = config.late_blocks[-1]
    for fish_key in sorted(model_input["fish_key"].astype(str).unique()):
        subset = model_input.loc[
            model_input["fish_key"].astype(str) != fish_key
        ].copy()
        block_result, block_diagnostic = _fit_mixed_model(
            subset,
            formula=block_formula,
            config=config,
            collect_extended_diagnostics=False,
        )
        trial_result, trial_diagnostic = _fit_mixed_model(
            subset,
            formula=trial_formula,
            config=config,
            collect_extended_diagnostics=False,
        )
        late_contrast = np.nan
        onset_trial = np.nan
        localized = False
        error_parts: list[str] = []
        if block_result is not None:
            try:
                block_table = block_contrasts(
                    block_result, subset, config=config
                ).set_index("block_10_name")
                if primary_late_block in block_table.index:
                    late_contrast = float(
                        block_table.loc[primary_late_block, "learning_contrast"]
                    )
            except Exception as error:  # noqa: BLE001 - influence diagnostic
                error_parts.append(f"block:{error}")
        else:
            error_parts.append(f"block:{block_diagnostic.get('error')}")
        if trial_result is not None and np.isfinite(simultaneous_critical_distance):
            try:
                trial_table, _ = trial_contrasts(
                    trial_result, subset, config=config
                )
                trial_table["loo_lower"] = (
                    trial_table["learning_contrast"]
                    - simultaneous_critical_distance
                )
                onset = localize_learning_onset(
                    trial_table,
                    delta_min=config.delta_min,
                    persistence_trials=config.persistence_trials,
                    lower_column="loo_lower",
                )
                localized = bool(onset["localized"])
                onset_trial = (
                    float(onset["onset_trial"]) if localized else np.nan
                )
            except Exception as error:  # noqa: BLE001 - influence diagnostic
                error_parts.append(f"trial:{error}")
        else:
            error_parts.append(f"trial:{trial_diagnostic.get('error')}")
        omitted_condition = str(
            model_input.loc[
                model_input["fish_key"].astype(str) == fish_key, "condition_id"
            ].iloc[0]
        )
        omitted_fish_id = str(
            model_input.loc[
                model_input["fish_key"].astype(str) == fish_key, "fish_id"
            ].iloc[0]
        )
        omitted_experiment_id = str(
            model_input.loc[
                model_input["fish_key"].astype(str) == fish_key,
                "experiment_id",
            ].iloc[0]
        )
        rows.append(
            {
                "omitted_fish_key": fish_key,
                "omitted_experiment_id": omitted_experiment_id,
                "omitted_fish_id": omitted_fish_id,
                "omitted_condition": omitted_condition,
                "primary_late_block": primary_late_block,
                "late_block_learning_contrast": late_contrast,
                "onset_localized": localized,
                "onset_trial": onset_trial,
                "block_model_status": block_diagnostic["diagnostic_status"],
                "trial_model_status": trial_diagnostic["diagnostic_status"],
                "diagnostic_status": "ok" if not error_parts else "failed",
                "error": "; ".join(error_parts) if error_parts else None,
            }
        )
    return pd.DataFrame(rows)


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    if not len(frame.columns):
        raise SchemaValidationError(
            f"Refusing to publish a schema-less Parquet table: {path.name}"
        )
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


def _coverage_table(eligibility: pd.DataFrame) -> pd.DataFrame:
    """Summarize contributing and excluded fish/rows by trial and block."""
    prepared = eligibility.copy()
    prepared["eligible_fish_key"] = prepared["fish_key"].where(
        prepared["eligible"].astype(bool)
    )
    by_trial = (
        prepared.groupby(
            ["condition_id", "trial_number", "block_10_name"],
            observed=True,
            dropna=False,
        )
        .agg(
            cohort_fish_count=("fish_key", "nunique"),
            eligible_fish_count=("eligible_fish_key", "nunique"),
            trial_row_count=("eligible", "size"),
            eligible_row_count=("eligible", "sum"),
        )
        .reset_index()
    )
    by_trial.insert(0, "coverage_level", "condition_trial")
    by_fish_block = (
        prepared.groupby(
            [
                "condition_id",
                "experiment_id",
                "fish_id",
                "fish_key",
                "block_10_name",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            scheduled_trial_count=("trial_number", "nunique"),
            eligible_trial_count=("eligible", "sum"),
            ineligible_trial_count=("eligible", lambda values: int((~values).sum())),
        )
        .reset_index()
    )
    by_fish_block.insert(0, "coverage_level", "fish_block")
    return pd.concat([by_trial, by_fish_block], ignore_index=True, sort=False)


def _descriptive_trajectory_tables(
    model_input: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Produce transparent fish- and condition-level summaries separate from the
    # mixed-model estimates used for formal onset inference.
    fish = (
        model_input.assign(activity_ratio=np.exp(-model_input["cr_score"]))
        .groupby(
            [
                "fish_key",
                "experiment_id",
                "fish_id",
                "condition_id",
                "trial_number",
            ],
            observed=True,
            sort=False,
        )["activity_ratio"]
        .median()
        .reset_index()
    )
    group = (
        fish.groupby(
            ["condition_id", "trial_number"], observed=True, sort=False
        )["activity_ratio"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
            fish_count="size",
        )
        .reset_index()
    )
    return fish, group


def build_learning_onset_analysis(
    project_dir: Path,
    *,
    cohort_id: str,
    analysis_id: str,
    config: LearningOnsetConfig,
    overwrite: bool = False,
) -> LearningOnsetResult:
    """Publish the complete cohort-authenticated learning-onset analysis."""
    _validate_identifier(analysis_id, "Analysis ID")
    project_dir = project_dir.resolve()
    outcomes, cohort_summary = load_cohort_trial_outcomes(project_dir, cohort_id)
    cohort_hash = str(cohort_summary["cohort_hash"])
    eligibility = build_analysis_eligibility(
        outcomes,
        metric_id=config.metric_id,
        outcome_id=config.outcome_id,
        alignment=config.alignment,
        min_baseline_samples=config.min_baseline_samples,
        min_response_samples=config.min_response_samples,
    )
    model_input = build_learning_model_input(
        outcomes, eligibility, config=config
    )
    reference = repr(config.control_condition)
    block_formula = (
        "log_response ~ log_baseline + "
        f"C(condition_id, Treatment(reference={reference})) * C(block_10_name)"
    )
    trial_formula = (
        "log_response ~ log_baseline + "
        f"C(condition_id, Treatment(reference={reference})) * "
        f"bs(trial_scaled, df={config.spline_df}, degree=3, include_intercept=False)"
    )
    categorical_formula = (
        "log_response ~ log_baseline + "
        f"C(condition_id, Treatment(reference={reference})) * C(trial_number)"
    )
    block_result, block_diagnostic = _fit_mixed_model(
        model_input, formula=block_formula, config=config
    )
    trial_result, trial_diagnostic = _fit_mixed_model(
        model_input, formula=trial_formula, config=config
    )
    if config.run_categorical_sensitivity:
        categorical_result, categorical_diagnostic = _fit_mixed_model(
            model_input,
            formula=categorical_formula,
            config=config,
        )
    else:
        categorical_result = None
        categorical_diagnostic = {
            "formula": categorical_formula,
            "diagnostic_status": "not_run",
            "error": "Categorical-trial sensitivity disabled by configuration.",
            "observation_count": len(model_input),
            "fish_count": int(model_input["fish_key"].nunique()),
        }
    block_coefficients = model_coefficients(
        block_result,
        model_name="block",
        confidence_level=config.confidence_level,
    )
    longitudinal_coefficients = model_coefficients(
        trial_result,
        model_name="longitudinal",
        confidence_level=config.confidence_level,
    )
    block_table = (
        block_contrasts(block_result, model_input, config=config)
        if block_result is not None
        else pd.DataFrame(
            columns=[
                "block_10_name",
                "raw_control_minus_test",
                "pretraining_raw_control_minus_test",
                "learning_contrast",
                "standard_error",
                "ci_lower",
                "ci_upper",
                "p_value",
                "control_fish_count",
                "test_fish_count",
                "p_value_holm",
                "supported",
            ]
        )
    )
    block_global = (
        block_global_interaction_test(block_result)
        if block_result is not None
        else pd.DataFrame(
            columns=[
                "test",
                "degrees_of_freedom",
                "wald_chi_square",
                "p_value",
                "diagnostic_status",
                "error",
            ]
        )
    )
    if trial_result is not None:
        trial_table, _ = trial_contrasts(trial_result, model_input, config=config)
        bootstrap_trials, bootstrap_onsets, critical = bootstrap_trial_contrasts(
            model_input,
            trial_table,
            formula=trial_formula,
            config=config,
        )
        if np.isfinite(critical):
            trial_table["simultaneous_lower"] = (
                trial_table["learning_contrast"] - critical
            )
            trial_table["simultaneous_upper"] = (
                trial_table["learning_contrast"] + critical
            )
        onset = localize_learning_onset(
            trial_table,
            delta_min=config.delta_min,
            persistence_trials=config.persistence_trials,
        )
    else:
        trial_table = pd.DataFrame(
            columns=[
                "trial_number",
                "learning_contrast",
                "standard_error",
                "pointwise_lower",
                "pointwise_upper",
                "simultaneous_lower",
                "simultaneous_upper",
                "control_fish_count",
                "test_fish_count",
                "estimable",
            ]
        )
        bootstrap_trials = pd.DataFrame(
            columns=["replicate", "trial_number", "learning_contrast"]
        )
        bootstrap_onsets = pd.DataFrame(
            columns=[
                "replicate",
                "localized",
                "onset_trial",
                "failure_reason",
            ]
        )
        critical = np.nan
        onset = {
            "localized": False,
            "onset_trial": None,
            "run_start_trial": None,
            "run_end_trial": None,
            "failure_reason": "longitudinal_model_failed",
        }
    if categorical_result is not None:
        categorical_trial_table, _ = trial_contrasts(
            categorical_result,
            model_input,
            config=config,
        )
    else:
        categorical_trial_table = pd.DataFrame(
            columns=[
                "trial_number",
                "learning_contrast",
                "standard_error",
                "pointwise_lower",
                "pointwise_upper",
                "simultaneous_lower",
                "simultaneous_upper",
                "control_fish_count",
                "test_fish_count",
                "estimable",
            ]
        )
    successful_bootstrap = int(
        bootstrap_trials["replicate"].nunique()
        if not bootstrap_trials.empty
        else 0
    )
    bootstrap_success_fraction = (
        successful_bootstrap / config.n_bootstrap
        if config.n_bootstrap > 0
        else 0.0
    )
    bootstrap_band_accepted = bool(
        np.isfinite(critical)
        and successful_bootstrap >= config.min_successful_bootstrap
        and bootstrap_success_fraction >= config.min_bootstrap_success_fraction
    )
    if not bootstrap_band_accepted and not trial_table.empty:
        trial_table["simultaneous_lower"] = np.nan
        trial_table["simultaneous_upper"] = np.nan
        onset = {
            "localized": False,
            "onset_trial": None,
            "run_start_trial": None,
            "run_end_trial": None,
            "failure_reason": "insufficient_successful_bootstrap_refits",
        }
    onset.update(
        {
            "delta_min": config.delta_min,
            "persistence_trials": config.persistence_trials,
            "confidence_level": config.confidence_level,
            "simultaneous_critical_distance": (
                float(critical) if bootstrap_band_accepted else None
            ),
            "successful_bootstrap_replicates": successful_bootstrap,
            "bootstrap_success_fraction": bootstrap_success_fraction,
        }
    )
    if not bootstrap_onsets.empty:
        localized = bootstrap_onsets["localized"].fillna(False).astype(bool)
        onset_trials = bootstrap_onsets.loc[localized, "onset_trial"].to_numpy(
            dtype=float
        )
        alpha = 1.0 - config.confidence_level
        onset.update(
            {
                "bootstrap_replicates": len(bootstrap_onsets),
                "bootstrap_localized_proportion": float(localized.mean()),
                "bootstrap_not_localized_proportion": float((~localized).mean()),
                "bootstrap_onset_lower": (
                    float(np.quantile(onset_trials, alpha / 2.0))
                    if onset_trials.size
                    else None
                ),
                "bootstrap_onset_upper": (
                    float(np.quantile(onset_trials, 1.0 - alpha / 2.0))
                    if onset_trials.size
                    else None
                ),
            }
        )
    else:
        onset.update(
            {
                "bootstrap_replicates": 0,
                "bootstrap_localized_proportion": None,
                "bootstrap_not_localized_proportion": None,
                "bootstrap_onset_lower": None,
                "bootstrap_onset_upper": None,
            }
        )
    supported_blocks = (
        block_table.loc[
            block_table.get("supported", pd.Series(dtype=bool)).astype(bool),
            "block_10_name",
        ].astype(str).tolist()
        if not block_table.empty
        else []
    )
    onset["first_supported_block"] = (
        next(
            (
                block
                for block in BLOCK_ORDER
                if block != config.pretraining_block and block in supported_blocks
            ),
            None,
        )
    )
    onset_table = pd.DataFrame([onset])
    fish_effects, robustness = fish_level_robustness(model_input, config=config)
    adjusted_trajectories = adjusted_condition_trajectories(
        trial_result,
        model_input,
        config=config,
    )
    residual_parts = [
        model_residuals(block_result, model_input, model_name="block"),
        model_residuals(trial_result, model_input, model_name="longitudinal"),
        model_residuals(
            categorical_result, model_input, model_name="categorical_trial"
        ),
    ]
    nonempty_residual_parts = [frame for frame in residual_parts if not frame.empty]
    residuals = (
        pd.concat(nonempty_residual_parts, ignore_index=True)
        if nonempty_residual_parts
        else residual_parts[0]
    )
    block_sensitivity, block_sensitivity_diagnostics = model_sensitivity_checks(
        block_result,
        model_input,
        model_name="block",
        formula=block_formula,
        config=config,
    )
    trial_sensitivity, trial_sensitivity_diagnostics = model_sensitivity_checks(
        trial_result,
        model_input,
        model_name="longitudinal",
        formula=trial_formula,
        config=config,
    )
    model_sensitivity = pd.concat(
        [block_sensitivity, trial_sensitivity], ignore_index=True
    )
    coverage = _coverage_table(eligibility)
    fish_trajectory, group_trajectory = _descriptive_trajectory_tables(model_input)
    influence = leave_one_fish_out(
        model_input,
        block_formula=block_formula,
        trial_formula=trial_formula,
        simultaneous_critical_distance=(
            critical if bootstrap_band_accepted else np.nan
        ),
        config=config,
    )
    influence_reasons: list[str] = []
    if influence.empty or not influence["diagnostic_status"].eq("ok").all():
        influence_reasons.append("one_or_more_leave_one_fish_out_fits_failed")
    if bool(onset["localized"]) and not influence.empty:
        if not influence["onset_localized"].astype(bool).all():
            influence_reasons.append("onset_disappears_when_one_fish_is_omitted")
        finite_loo_onsets = influence["onset_trial"].dropna().to_numpy(dtype=float)
        if (
            finite_loo_onsets.size
            and np.max(np.abs(finite_loo_onsets - float(onset["onset_trial"])))
            > config.persistence_trials
        ):
            influence_reasons.append("onset_shifts_beyond_persistence_window")
    primary_block_effect = np.nan
    if not block_table.empty and config.late_blocks[-1] in set(
        block_table["block_10_name"].astype(str)
    ):
        primary_block_effect = float(
            block_table.loc[
                block_table["block_10_name"].astype(str) == config.late_blocks[-1],
                "learning_contrast",
            ].iloc[0]
        )
    loo_block_effects = influence["late_block_learning_contrast"].dropna().to_numpy(
        dtype=float
    )
    if (
        np.isfinite(primary_block_effect)
        and primary_block_effect != 0
        and loo_block_effects.size
        and np.any(np.sign(loo_block_effects) != np.sign(primary_block_effect))
    ):
        influence_reasons.append("primary_block_effect_changes_sign")
    diagnostics = pd.DataFrame(
        [
            {
                "model": "block",
                "required_for_publication": True,
                **block_diagnostic,
            },
            {
                "model": "longitudinal",
                "required_for_publication": True,
                **trial_diagnostic,
            },
            {
                "model": "categorical_trial_sensitivity",
                "required_for_publication": False,
                **categorical_diagnostic,
            },
            {
                "model": "simultaneous_band",
                "required_for_publication": True,
                "diagnostic_status": (
                    "ok" if bootstrap_band_accepted else "failed"
                ),
                "error": (
                    None
                    if bootstrap_band_accepted
                    else (
                        "Fish-bootstrap simultaneous band did not meet the "
                        "configured successful-refit count/fraction."
                    )
                ),
                "observation_count": len(model_input),
                "fish_count": int(model_input["fish_key"].nunique()),
            },
            {
                "model": "leave_one_fish_out",
                "required_for_publication": True,
                "diagnostic_status": "failed" if influence_reasons else "ok",
                "error": "; ".join(influence_reasons) if influence_reasons else None,
                "observation_count": len(model_input),
                "fish_count": int(model_input["fish_key"].nunique()),
            },
            {
                "model": "fish_level_robustness",
                "required_for_publication": True,
                "diagnostic_status": str(
                    robustness.iloc[0]["diagnostic_status"]
                ),
                "error": robustness.iloc[0]["error"],
                "observation_count": len(fish_effects),
                "fish_count": len(fish_effects),
            },
            *block_sensitivity_diagnostics,
            *trial_sensitivity_diagnostics,
        ]
    )

    paths, summary_path, marker_path = _analysis_paths(project_dir, analysis_id)
    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    all_paths = (*paths.values(), summary_path, marker_path)
    existing = [path for path in all_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "model_input": model_input,
        "eligibility": eligibility,
        "block_global_test": block_global,
        "block_coefficients": block_coefficients,
        "block_contrasts": block_table,
        "longitudinal_coefficients": longitudinal_coefficients,
        "trial_contrasts": trial_table,
        "adjusted_trajectories": adjusted_trajectories,
        "categorical_trial_contrasts": categorical_trial_table,
        "onset": onset_table,
        "fish_effects": fish_effects,
        "robustness": robustness,
        "diagnostics": diagnostics,
        "model_sensitivity": model_sensitivity,
        "residuals": residuals,
        "coverage": coverage,
        "influence": influence,
        "fish_trajectory": fish_trajectory,
        "group_trajectory": group_trajectory,
        "bootstrap_trials": bootstrap_trials,
        "bootstrap_onsets": bootstrap_onsets,
    }
    with artifact_staging(
        project_dir,
        prefix=f".{analysis_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged = {name: staging_root / path.name for name, path in paths.items()}
        records = {
            name: _write_parquet(staged[name], frames[name]) for name in paths
        }
        for name, path in paths.items():
            records[name]["path"] = str(path)
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "implemented_not_paper_approved",
            "paper_approved": False,
            "analysis_id": analysis_id,
            "cohort_id": cohort_id,
            "cohort_hash": cohort_hash,
            "config": asdict(config),
            "config_sha256": _config_hash(config),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_formulas": {
                "block": block_formula,
                "longitudinal": trial_formula,
                "categorical_trial_sensitivity": categorical_formula,
            },
            "onset": onset,
            "cohort_outcomes_sha256": cohort_summary["artifacts"]["outcomes"][
                "sha256"
            ],
            "artifacts": records,
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "analysis_id": analysis_id,
                "cohort_id": cohort_id,
                "cohort_hash": cohort_hash,
                "config_sha256": summary["config_sha256"],
                "artifact_sha256": {
                    name: record["sha256"] for name, record in records.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        publish_transaction(
            (
                *((staged[name], path) for name, path in paths.items()),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )
    return LearningOnsetResult(
        analysis_id=analysis_id,
        cohort_id=cohort_id,
        cohort_hash=cohort_hash,
        model_input_path=paths["model_input"],
        eligibility_path=paths["eligibility"],
        block_global_test_path=paths["block_global_test"],
        block_coefficients_path=paths["block_coefficients"],
        block_contrasts_path=paths["block_contrasts"],
        longitudinal_coefficients_path=paths["longitudinal_coefficients"],
        trial_contrasts_path=paths["trial_contrasts"],
        adjusted_trajectories_path=paths["adjusted_trajectories"],
        categorical_trial_contrasts_path=paths[
            "categorical_trial_contrasts"
        ],
        onset_path=paths["onset"],
        fish_effects_path=paths["fish_effects"],
        robustness_path=paths["robustness"],
        diagnostics_path=paths["diagnostics"],
        model_sensitivity_path=paths["model_sensitivity"],
        residuals_path=paths["residuals"],
        coverage_path=paths["coverage"],
        influence_path=paths["influence"],
        fish_trajectory_path=paths["fish_trajectory"],
        group_trajectory_path=paths["group_trajectory"],
        bootstrap_onsets_path=paths["bootstrap_onsets"],
        summary_path=summary_path,
        completion_marker_path=marker_path,
    )
