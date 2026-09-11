"""Frozen standard-main stage-5 statistical routes."""

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
from scipy.stats import mannwhitneyu, wilcoxon

from classical_conditioning.analysis.legacy_normalized_vigor import (
    BASELINE_COLUMN,
    NORMALIZED_COLUMN,
    RECIPE_ID as NORMALIZED_RECIPE_ID,
    RESPONSE_COLUMN,
)
from classical_conditioning.analysis.legacy_standard_main import ALIGNMENTS
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    verify_completed_parquet_set,
    write_json_atomic,
)
from classical_conditioning.config import get_experiment_spec
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    SchemaValidationError,
)

RECIPE_ID = "legacy-statistics-v1"
CONFIG_SHA256 = "55771849b6635888bc1ae50434ca41707161f010e0917385d362a4cad0e914c2"
SOURCE_COMMIT = "b0dfcb345fc185343802f6072ff7b3740498345b"
SOURCE_BLOB = "f14095d53dec169478b410151dddeddf6ce6f326"

_FIVE_TRIAL_BLOCK_NAMES = (
    "Early Pre-Train",
    "Late Pre-Train",
    "Early Train",
    "Train 2",
    "Train 3",
    "Train 4",
    "Train 5",
    "Train 6",
    "Train 7",
    "Train 8",
    "Train 9",
    "Late Train",
    "Early Test",
    "Test 2",
    "Test 3",
    "Test 4",
    "Test 5",
    "Late Test",
)
_RESULT_TABLES = (
    "model_input",
    "block_medians",
    "nonparametric",
    "global_model",
    "block_models",
    "trial_models",
)


@dataclass(frozen=True)
class LegacyStatisticsConfig:
    minimum_trials_per_fish_per_block: int = 6
    apply_fish_discard: bool = False
    age_filter: tuple[str, ...] = ("all",)
    setup_color_filter: tuple[str, ...] = ("all",)
    selected_cs_blocks: tuple[str, ...] = (
        "Early Pre-Train",
        "Early Test",
        "Late Test",
    )
    alpha: float = 0.05
    nonparametric_correction: str = "holm"
    model_correction: str = "fdr_bh"
    random_effects_formula: str = "~Log_Baseline"
    global_optimizer: str = "lbfgs"
    local_optimizer: str = "powell"
    jitter_scale: float = 0.0
    jitter_seed: int = 10
    trajectory_bootstrap_resamples: int = 100
    trajectory_bootstrap_seed: int = 10

    def __post_init__(self) -> None:
        if self.minimum_trials_per_fish_per_block < 1:
            raise ConfigurationError("Minimum trial count must be positive.")
        if self.apply_fish_discard:
            raise ConfigurationError(
                f"{RECIPE_ID} must preserve disabled stage-5 fish exclusion."
            )
        if self.age_filter != ("all",) or self.setup_color_filter != ("all",):
            raise ConfigurationError(
                f"{RECIPE_ID} must preserve disabled age and rig filters."
            )
        if not 0 < self.alpha < 1:
            raise ConfigurationError("Statistical alpha must be between zero and one.")
        if self.nonparametric_correction != "holm":
            raise ConfigurationError("Frozen nonparametric correction is Holm.")
        if self.model_correction != "fdr_bh":
            raise ConfigurationError("Frozen model correction is Benjamini-Hochberg.")
        if self.random_effects_formula != "~Log_Baseline":
            raise ConfigurationError("Frozen random-effects formula changed.")
        if self.global_optimizer != "lbfgs" or self.local_optimizer != "powell":
            raise ConfigurationError("Frozen mixed-model optimizers changed.")
        if self.jitter_scale != 0.0 or self.jitter_seed != 10:
            raise ConfigurationError("Frozen mixed-model jitter settings changed.")
        if (
            self.trajectory_bootstrap_resamples != 100
            or self.trajectory_bootstrap_seed != 10
        ):
            raise ConfigurationError("Frozen trajectory bootstrap settings changed.")


@dataclass(frozen=True)
class LegacyStatisticsResult:
    analysis_id: str
    artifact_paths: dict[str, Path]
    summary_path: Path
    completion_marker_path: Path
    model_errors: tuple[dict[str, Any], ...]


def _config_hash(config: LegacyStatisticsConfig) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )


def prepare_legacy_model_input(
    outcomes: pd.DataFrame,
    *,
    config: LegacyStatisticsConfig = LegacyStatisticsConfig(),
) -> pd.DataFrame:
    required = {
        "Exp.",
        "Fish",
        "Block name",
        "Trial number",
        BASELINE_COLUMN,
        RESPONSE_COLUMN,
        NORMALIZED_COLUMN,
    }
    missing = required.difference(outcomes.columns)
    if missing:
        raise SchemaValidationError(
            f"Stage-5 outcomes are missing columns: {sorted(missing)}"
        )
    frame = outcomes.rename(
        columns={
            "Exp.": "Condition",
            "Fish": "Fish_ID",
            "Block name": "Block_name",
        }
    ).copy()
    frame.dropna(
        subset=[
            NORMALIZED_COLUMN,
            "Trial number",
            "Block_name",
            BASELINE_COLUMN,
            RESPONSE_COLUMN,
        ],
        inplace=True,
    )
    counts = frame.groupby(["Fish_ID", "Block_name"], observed=True).size()
    valid_fish = counts[
        counts >= config.minimum_trials_per_fish_per_block
    ].reset_index()["Fish_ID"].unique()
    frame = frame.loc[frame["Fish_ID"].isin(valid_fish)].copy()
    frame["Normalized vigor plot"] = frame[NORMALIZED_COLUMN]
    frame["Log_Baseline"] = np.log(frame[BASELINE_COLUMN] + 1)
    frame["Log_Response"] = np.log(frame[RESPONSE_COLUMN] + 1)
    if config.jitter_scale > 0:
        np.random.seed(config.jitter_seed)
        frame["Log_Response"] += np.random.normal(
            0,
            config.jitter_scale,
            size=len(frame),
        )
    frame["Trial number"] = frame["Trial number"].astype("int")
    if not frame.empty:
        frame["Trial number"] = (
            frame["Trial number"] - frame["Trial number"].min() + 1
        )
    return frame.reset_index(drop=True)


def build_legacy_block_medians(
    outcomes: pd.DataFrame,
    *,
    alignment: str,
    config: LegacyStatisticsConfig = LegacyStatisticsConfig(),
) -> pd.DataFrame:
    columns = ["Fish", "Block name", "Exp.", NORMALIZED_COLUMN]
    if alignment != "CS":
        return pd.DataFrame(columns=columns)
    frame = outcomes.copy()
    trial_numbers = pd.to_numeric(frame["Trial number"], errors="coerce")
    indices = ((trial_numbers - 5) // 5).astype("Int64")
    valid = trial_numbers.between(5, 94) & indices.between(
        0,
        len(_FIVE_TRIAL_BLOCK_NAMES) - 1,
    )
    frame = frame.loc[valid].copy()
    indices = indices.loc[valid].astype(int)
    frame["Block name"] = [
        _FIVE_TRIAL_BLOCK_NAMES[index] for index in indices
    ]
    frame = frame.loc[
        frame["Block name"].isin(config.selected_cs_blocks)
    ].copy()
    return (
        frame.dropna()
        .groupby(["Fish", "Block name", "Exp."], observed=True)[NORMALIZED_COLUMN]
        .median()
        .reset_index()
    )


def _apply_correction(
    rows: list[dict[str, Any]],
    *,
    alpha: float,
    method: str,
) -> None:
    if not rows:
        return
    multipletests = _require_multipletests()

    reject, adjusted, _, _ = multipletests(
        [row["p_raw"] for row in rows],
        alpha=alpha,
        method=method,
    )
    for row, rejected, corrected in zip(rows, reject, adjusted, strict=True):
        row["p_adjusted"] = float(corrected)
        row["reject"] = bool(rejected)


def _require_multipletests():
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as error:
        raise RuntimeError(
            f"{RECIPE_ID} requires the project's 'legacy' optional dependencies."
        ) from error
    return multipletests


def run_legacy_nonparametric_tests(
    block_medians: pd.DataFrame,
    *,
    condition_order: tuple[str, ...],
    config: LegacyStatisticsConfig = LegacyStatisticsConfig(),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = list(config.selected_cs_blocks)

    for condition in condition_order:
        family: list[dict[str, Any]] = []
        condition_data = block_medians.loc[
            block_medians["Exp."].astype(str) == condition
        ]
        present = [
            block
            for block in selected
            if block in condition_data["Block name"].dropna().unique()
        ]
        for left, right in zip(present[:-1], present[1:], strict=True):
            left_values = condition_data.loc[
                condition_data["Block name"] == left,
                NORMALIZED_COLUMN,
            ].dropna()
            right_values = condition_data.loc[
                condition_data["Block name"] == right,
                NORMALIZED_COLUMN,
            ].dropna()
            if left_values.empty or right_values.empty:
                continue
            statistic, p_value = mannwhitneyu(
                left_values,
                right_values,
                alternative="two-sided",
            )
            family.append(
                {
                    "route": "block-line-within-mann-whitney",
                    "condition_left": condition,
                    "condition_right": condition,
                    "block_left": left,
                    "block_right": right,
                    "statistic": float(statistic),
                    "p_raw": float(p_value),
                    "correction_family": f"within-mann-whitney:{condition}",
                }
            )
        _apply_correction(
            family,
            alpha=config.alpha,
            method=config.nonparametric_correction,
        )
        rows.extend(family)

    between: list[dict[str, Any]] = []
    condition_pairs: list[tuple[str, str]] = []
    if len(condition_order) == 2:
        condition_pairs = [(condition_order[0], condition_order[1])]
    elif len(condition_order) == 3:
        condition_pairs = [
            (condition_order[0], condition_order[1]),
            (condition_order[0], condition_order[2]),
        ]
    elif len(condition_order) >= 4:
        condition_pairs = list(
            zip(condition_order[::2], condition_order[1::2], strict=True)
        )
    for block in selected:
        for left_condition, right_condition in condition_pairs:
            left_values = block_medians.loc[
                (block_medians["Block name"] == block)
                & (block_medians["Exp."].astype(str) == left_condition),
                NORMALIZED_COLUMN,
            ].dropna()
            right_values = block_medians.loc[
                (block_medians["Block name"] == block)
                & (block_medians["Exp."].astype(str) == right_condition),
                NORMALIZED_COLUMN,
            ].dropna()
            if left_values.empty or right_values.empty:
                continue
            statistic, p_value = mannwhitneyu(
                left_values,
                right_values,
                alternative="two-sided",
            )
            between.append(
                {
                    "route": "block-box-between-mann-whitney",
                    "condition_left": left_condition,
                    "condition_right": right_condition,
                    "block_left": block,
                    "block_right": block,
                    "statistic": float(statistic),
                    "p_raw": float(p_value),
                    "correction_family": "between-mann-whitney",
                }
            )
    _apply_correction(
        between,
        alpha=config.alpha,
        method=config.nonparametric_correction,
    )
    rows.extend(between)

    paired: list[dict[str, Any]] = []
    complete_counts = block_medians.groupby(
        ["Fish", "Exp."],
        observed=True,
    )["Block name"].nunique()
    complete = {
        (fish, condition)
        for (fish, condition), count in complete_counts.items()
        if count == len(selected)
    }
    complete_data = block_medians.loc[
        [
            (fish, condition) in complete
            for fish, condition in zip(
                block_medians["Fish"],
                block_medians["Exp."].astype(str),
                strict=True,
            )
        ]
    ]
    for condition in condition_order:
        condition_data = complete_data.loc[
            complete_data["Exp."].astype(str) == condition
        ]
        pivot = condition_data.pivot(
            index="Fish",
            columns="Block name",
            values=NORMALIZED_COLUMN,
        )
        for left, right in zip(selected[:-1], selected[1:], strict=True):
            if left not in pivot or right not in pivot:
                continue
            pairs = pivot[[left, right]].dropna()
            if pairs.empty:
                continue
            try:
                statistic, p_value = wilcoxon(
                    pairs[left],
                    pairs[right],
                )
            except ValueError as error:
                statistic = np.nan
                p_value = np.nan
                error_text = str(error)
            else:
                error_text = None
            paired.append(
                {
                    "route": "block-box-within-wilcoxon",
                    "condition_left": condition,
                    "condition_right": condition,
                    "block_left": left,
                    "block_right": right,
                    "statistic": float(statistic),
                    "p_raw": float(p_value),
                    "correction_family": "within-wilcoxon",
                    "error": error_text,
                }
            )
    finite_paired = [row for row in paired if np.isfinite(row["p_raw"])]
    _apply_correction(
        finite_paired,
        alpha=config.alpha,
        method=config.nonparametric_correction,
    )
    rows.extend(paired)
    columns = [
        "route",
        "condition_left",
        "condition_right",
        "block_left",
        "block_right",
        "statistic",
        "p_raw",
        "p_adjusted",
        "reject",
        "correction_family",
        "error",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def fit_legacy_mixed_model(
    frame: pd.DataFrame,
    formula: str,
    groups_column: str,
    *,
    re_formula: str | None = None,
    method: str = "powell",
    reml: bool = False,
):
    try:
        import statsmodels.formula.api as smf

        model = smf.mixedlm(
            formula,
            frame,
            groups=frame[groups_column],
            re_formula=re_formula,
        )
        return model.fit(reml=reml, method=method), None
    except Exception as error:
        return None, str(error)


def _model_terms(
    result: Any,
    *,
    scope: str,
    unit: str,
) -> list[dict[str, Any]]:
    return [
        {
            "scope": scope,
            "unit": unit,
            "term": str(term),
            "coefficient": float(result.params[term]),
            "p_raw": float(result.pvalues[term]),
        }
        for term in result.params.index
    ]


def run_legacy_mixed_models(
    model_input: pd.DataFrame,
    *,
    condition_order: tuple[str, ...],
    config: LegacyStatisticsConfig = LegacyStatisticsConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    result_columns = ["scope", "unit", "term", "coefficient", "p_raw"]
    block_columns = [
        "Block",
        "Term_Main",
        "CondLevel_Main",
        "Coef_Mean",
        "P_Mean",
        "Term_Slope",
        "CondLevel_Slope",
        "Coef_Slope",
        "P_Slope",
        "P_Mean_FDR",
        "P_Slope_FDR",
        "Sig_Mean",
        "Sig_Slope_Raw",
        "Sig_Slope",
    ]
    trial_columns = ["Trial", "Term", "P_raw", "P_FDR", "Sig"]
    if model_input.empty:
        return (
            pd.DataFrame(columns=result_columns),
            pd.DataFrame(columns=block_columns),
            pd.DataFrame(columns=trial_columns),
            [{"Type": "Model-Input", "Unit": "All", "Error": "No retained rows"}],
        )

    reference = condition_order[0]
    errors: list[dict[str, Any]] = []
    global_formula = (
        "Log_Response ~ Log_Baseline + "
        f"C(Condition, Treatment('{reference}')) * C(Block_name)"
    )
    global_result, error = fit_legacy_mixed_model(
        model_input,
        global_formula,
        "Fish_ID",
        re_formula=config.random_effects_formula,
        method=config.global_optimizer,
    )
    if global_result is None:
        global_rows: list[dict[str, Any]] = []
        errors.append({"Type": "Global", "Unit": "All", "Error": error})
    else:
        global_rows = _model_terms(global_result, scope="global", unit="All")

    block_order = (
        model_input.groupby("Block_name", observed=True)["Trial number"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    block_rows: list[dict[str, Any]] = []
    for block in block_order:
        block_frame = model_input.loc[
            model_input["Block_name"] == block
        ].copy()
        if block_frame["Condition"].nunique() < 2:
            continue
        block_frame["Trial_Centered"] = (
            block_frame["Trial number"] - block_frame["Trial number"].mean()
        )
        formula = (
            "Log_Response ~ Log_Baseline + "
            f"C(Condition, Treatment('{reference}')) * Trial_Centered"
        )
        result, error = fit_legacy_mixed_model(
            block_frame,
            formula,
            "Fish_ID",
            re_formula=config.random_effects_formula,
            method=config.local_optimizer,
        )
        if result is None:
            errors.append({"Type": "Block-Fit", "Unit": block, "Error": error})
            continue
        main_terms = [
            term
            for term in result.params.index
            if "Condition" in term and ":" not in term
        ]
        slope_terms = [
            term
            for term in result.params.index
            if "Condition" in term and ":" in term
        ]
        if not main_terms or not slope_terms:
            errors.append(
                {
                    "Type": "Block-TermsMissing",
                    "Unit": block,
                    "Error": "Missing condition mean/slope terms",
                }
            )
            continue
        best_main = min(
            sorted(main_terms),
            key=lambda term: float(result.pvalues.get(term, 1.0)),
        )
        best_slope = min(
            sorted(slope_terms),
            key=lambda term: float(result.pvalues.get(term, 1.0)),
        )

        def level(term: str) -> str:
            match = re.search(r"\[T\.(.+?)\]", term)
            return match.group(1) if match else term

        block_rows.append(
            {
                "Block": block,
                "Term_Main": best_main,
                "CondLevel_Main": level(best_main),
                "Coef_Mean": float(result.params[best_main]),
                "P_Mean": float(result.pvalues[best_main]),
                "Term_Slope": best_slope,
                "CondLevel_Slope": level(best_slope),
                "Coef_Slope": float(result.params[best_slope]),
                "P_Slope": float(result.pvalues[best_slope]),
            }
        )
    block_results = pd.DataFrame(block_rows)
    if not block_results.empty:
        multipletests = _require_multipletests()

        _, block_results["P_Mean_FDR"], _, _ = multipletests(
            block_results["P_Mean"],
            alpha=config.alpha,
            method=config.model_correction,
        )
        _, block_results["P_Slope_FDR"], _, _ = multipletests(
            block_results["P_Slope"],
            alpha=config.alpha,
            method=config.model_correction,
        )
        block_results["Sig_Mean"] = (
            block_results["P_Mean_FDR"] < config.alpha
        )
        block_results["Sig_Slope_Raw"] = (
            block_results["P_Slope"] < config.alpha
        )
        block_results["Sig_Slope"] = (
            block_results["P_Slope_FDR"] < config.alpha
        )
    block_results = block_results.reindex(columns=block_columns)

    trial_rows: list[dict[str, Any]] = []
    for trial in sorted(model_input["Trial number"].unique()):
        trial_frame = model_input.loc[
            model_input["Trial number"] == trial
        ].copy()
        if trial_frame["Condition"].nunique() < 2:
            continue
        formula = (
            "Log_Response ~ Log_Baseline + "
            f"C(Condition, Treatment('{reference}'))"
        )
        result, error = fit_legacy_mixed_model(
            trial_frame,
            formula,
            "Fish_ID",
            method=config.local_optimizer,
        )
        if result is None:
            errors.append({"Type": "Trial-Fit", "Unit": int(trial), "Error": error})
            continue
        terms = [term for term in result.params.index if "Condition" in term]
        if not terms:
            errors.append(
                {
                    "Type": "Trial-TermMissing",
                    "Unit": int(trial),
                    "Error": "Missing condition term",
                }
            )
            continue
        term = terms[0]
        trial_rows.append(
            {
                "Trial": int(trial),
                "Term": term,
                "P_raw": float(result.pvalues[term]),
            }
        )
    trial_results = pd.DataFrame(trial_rows)
    if not trial_results.empty:
        multipletests = _require_multipletests()

        reject, adjusted, _, _ = multipletests(
            trial_results["P_raw"],
            alpha=config.alpha,
            method=config.model_correction,
        )
        trial_results["P_FDR"] = adjusted
        trial_results["Sig"] = reject
    trial_results = trial_results.reindex(columns=trial_columns)
    return (
        pd.DataFrame(global_rows).reindex(columns=result_columns),
        block_results,
        trial_results,
        errors,
    )


def _verify_normalized_input(
    project_dir: Path,
    recording_id: str,
):
    source_dir = project_dir / "Processed data" / recording_id
    return verify_completed_parquet_set(
        {
            alignment: source_dir / f"{NORMALIZED_RECIPE_ID}_{alignment}.parquet"
            for alignment in ALIGNMENTS
        },
        project_dir
        / "Quality checks"
        / recording_id
        / f"{NORMALIZED_RECIPE_ID}_summary.json",
        project_dir
        / "Metadata"
        / f"{recording_id}_{NORMALIZED_RECIPE_ID}_complete.json",
        recipe=NORMALIZED_RECIPE_ID,
        recording_id=recording_id,
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return {
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": len(table.column_names),
        "size_bytes": path.stat().st_size,
        "compression": "zstd",
        "compression_lossless": True,
    }


def build_legacy_statistics(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    alignment: str = "CS",
    experiment_name: str = "allDelay",
    config: LegacyStatisticsConfig = LegacyStatisticsConfig(),
    overwrite: bool = False,
) -> LegacyStatisticsResult:
    _validate_analysis_id(analysis_id)
    recording_ids = tuple(dict.fromkeys(recording_ids))
    if not recording_ids:
        raise ConfigurationError("At least one recording ID is required.")
    if alignment not in ALIGNMENTS:
        raise ConfigurationError(f"Unknown stage-5 alignment: {alignment!r}")
    if config != LegacyStatisticsConfig():
        raise ConfigurationError(
            f"{RECIPE_ID} uses a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    _require_multipletests()
    recipe_hash = _config_hash(config)
    if recipe_hash != CONFIG_SHA256:
        raise ConfigurationError(
            f"The frozen {RECIPE_ID} configuration hash changed. "
            "Use a new recipe identity for changed behavior."
        )
    project_dir = project_dir.resolve()
    sources = {
        recording_id: _verify_normalized_input(project_dir, recording_id)
        for recording_id in recording_ids
    }
    missing_alignment = [
        recording_id
        for recording_id, source in sources.items()
        if alignment not in source.data_paths
    ]
    if missing_alignment:
        raise SchemaValidationError(
            f"Requested {alignment} outcomes are absent for recordings: "
            f"{missing_alignment}"
        )
    frames = []
    for recording_id, source in sources.items():
        frame = pq.read_table(source.data_paths[alignment]).to_pandas()
        frame.insert(0, "Recording ID", recording_id)
        frames.append(frame)
    if not frames:
        raise SchemaValidationError(
            f"No authenticated {alignment} outcomes exist for the requested recordings."
        )
    outcomes = pd.concat(frames, ignore_index=True)
    model_input = prepare_legacy_model_input(outcomes, config=config)
    block_medians = build_legacy_block_medians(
        outcomes,
        alignment=alignment,
        config=config,
    )
    condition_order = tuple(
        condition.condition_id
        for condition in get_experiment_spec(experiment_name).conditions
    )
    nonparametric = run_legacy_nonparametric_tests(
        block_medians,
        condition_order=condition_order,
        config=config,
    )
    global_model, block_models, trial_models, model_errors = (
        run_legacy_mixed_models(
            model_input,
            condition_order=condition_order,
            config=config,
        )
    )
    tables = {
        "model_input": model_input,
        "block_medians": block_medians,
        "nonparametric": nonparametric,
        "global_model": global_model,
        "block_models": block_models,
        "trial_models": trial_models,
    }

    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    artifact_paths = {
        name: output_dir / f"{RECIPE_ID}_{alignment}_{name}.parquet"
        for name in _RESULT_TABLES
    }
    summary_path = (
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{RECIPE_ID}_{alignment}_summary.json"
    )
    marker_path = (
        project_dir
        / "Metadata"
        / f"{analysis_id}_{RECIPE_ID}_{alignment}_complete.json"
    )
    outputs = (*artifact_paths.values(), summary_path, marker_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{analysis_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_paths: dict[str, Path] = {}
        records: dict[str, dict[str, Any]] = {}
        for name, table in tables.items():
            staged_path = staging_root / artifact_paths[name].name
            record = _write_parquet(staged_path, table)
            record["path"] = str(artifact_paths[name])
            staged_paths[name] = staged_path
            records[name] = record
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        source_lineage = {
            recording_id: {
                "summary_sha256": sha256_file(source.summary_path),
                "artifact_sha256": source.marker["artifact_sha256"].get(alignment),
            }
            for recording_id, source in sources.items()
        }
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "legacy_reproduction",
            "analysis_id": analysis_id,
            "alignment": alignment,
            "recording_ids": list(recording_ids),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "historical_source": {
                "commit": SOURCE_COMMIT,
                "blob": SOURCE_BLOB,
                "file": "5_NormalizedVigorPlotting.py",
                "functions": [
                    "prepare_main_df",
                    "run_block_summary_lines",
                    "run_block_summary_boxplot",
                    "run_trial_by_trial",
                ],
            },
            "config": asdict(config),
            "config_sha256": recipe_hash,
            "inputs": source_lineage,
            "artifacts": records,
            "model_errors": model_errors,
            "validation": {
                "recording_count": len(recording_ids),
                "fish_count": int(outcomes["Fish"].nunique()),
                "condition_count": int(outcomes["Exp."].nunique()),
                "cohort_scale_validated": False,
                "reason": (
                    "Cohort-scale validation remains blocked until multiple "
                    "recordings and both conditions are supplied."
                ),
            },
            "known_legacy_behavior": [
                "A fish is retained when any one block meets the six-trial threshold.",
                "Within-condition line-route comparisons use unpaired Mann-Whitney tests.",
                "Selected-block between-condition Mann-Whitney tests use Holm correction.",
                "Complete-fish selected-block paired comparisons use Wilcoxon with Holm correction.",
                "Global, block-local, and trial-local mixed models group by fish.",
                "Block mean and slope tests use separate Benjamini-Hochberg corrections.",
                "Trial-local tests use a separate Benjamini-Hochberg correction.",
                "Mixed-model fit failures are recorded as strings rather than raised.",
                "Trajectory bootstrap settings are 100 resamples with seed 10 and no fish cluster.",
            ],
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": RECIPE_ID,
                "analysis_id": analysis_id,
                "alignment": alignment,
                "recording_ids": list(recording_ids),
                "artifact_sha256": {
                    name: record["sha256"] for name, record in records.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        for recording_id, source in sources.items():
            input_path = source.data_paths[alignment]
            stat = input_path.stat()
            if source.data_states[alignment] != (
                stat.st_size,
                stat.st_mtime_ns,
            ) or sha256_file(input_path) != source.marker["artifact_sha256"][
                alignment
            ]:
                raise ArtifactIntegrityError(
                    f"Stage-5 input changed during statistics: {recording_id}"
                )
        publish_transaction(
            tuple(
                (staged_paths[name], artifact_paths[name])
                for name in _RESULT_TABLES
            )
            + (
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )
    return LegacyStatisticsResult(
        analysis_id=analysis_id,
        artifact_paths=artifact_paths,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        model_errors=tuple(model_errors),
    )
