"""Fish-level summary + sign-flip permutation (workshop family A).

This is intentionally not a mixed-effects variant. Each fish collapses to one
effect, then a randomization test asks whether the population mean effect is
systematically nonzero. See Plans/Notes/STATISTICS_METHODOLOGY_WORKSHOP.md.
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

RECIPE_ID = "candidate-fish-permutation-v1"
DEFAULT_EARLY_BLOCKS = ("Pre-train", "Train 1")
DEFAULT_LATE_BLOCKS = ("Train 5", "Test 1", "Test 2", "Test 3")


@dataclass(frozen=True)
class FishPermutationConfig:
    alignment: str = "CS"
    early_blocks: tuple[str, ...] = DEFAULT_EARLY_BLOCKS
    late_blocks: tuple[str, ...] = DEFAULT_LATE_BLOCKS
    n_permutations: int = 4_999
    seed: int = 10
    activity_offset: float = 1e-6

    def __post_init__(self) -> None:
        if self.alignment not in {"CS", "US"}:
            raise ConfigurationError("Permutation alignment must be CS or US.")
        if not self.early_blocks or not self.late_blocks:
            raise ConfigurationError("Early and late block sets must be non-empty.")
        if set(self.early_blocks).intersection(self.late_blocks):
            raise ConfigurationError("Early and late block sets must be disjoint.")
        if self.n_permutations < 99:
            raise ConfigurationError("Use at least 99 permutations.")
        if self.activity_offset <= 0:
            raise ConfigurationError("Activity offset must be positive.")


@dataclass(frozen=True)
class FishPermutationResult:
    analysis_id: str
    recording_ids: tuple[str, ...]
    fish_effects_path: Path
    population_path: Path
    summary_path: Path
    completion_marker_path: Path


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError(
            "Analysis ID must use only letters, numbers, dot, underscore, or hyphen."
        )


def _config_hash(config: FishPermutationConfig) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_fish_learning_effects(
    model_input: pd.DataFrame,
    *,
    config: FishPermutationConfig = FishPermutationConfig(),
) -> pd.DataFrame:
    """Collapse trials to one early-vs-late log-activity effect per fish."""
    required = {
        "fish_id",
        "metric_id",
        "outcome_id",
        "block_10_name",
        "log_response",
        "log_baseline",
        "alignment",
    }
    missing = required.difference(model_input.columns)
    if missing:
        raise ConfigurationError(
            f"Model input is missing columns: {sorted(missing)}"
        )
    frame = model_input.loc[
        model_input["alignment"].astype(str) == config.alignment
    ].copy()
    frame["adjusted"] = frame["log_response"] - frame["log_baseline"]
    rows: list[dict[str, Any]] = []
    for (fish_id, metric_id, outcome_id), subset in frame.groupby(
        ["fish_id", "metric_id", "outcome_id"],
        observed=True,
        sort=True,
    ):
        early = subset.loc[
            subset["block_10_name"].isin(config.early_blocks),
            "adjusted",
        ]
        late = subset.loc[
            subset["block_10_name"].isin(config.late_blocks),
            "adjusted",
        ]
        if early.empty or late.empty:
            continue
        early_mean = float(early.mean())
        late_mean = float(late.mean())
        rows.append(
            {
                "fish_id": str(fish_id),
                "metric_id": str(metric_id),
                "outcome_id": str(outcome_id),
                "early_mean_adjusted": early_mean,
                "late_mean_adjusted": late_mean,
                # Negative => late activity below early after baseline adjustment.
                "learning_effect": late_mean - early_mean,
                "early_trial_count": int(len(early)),
                "late_trial_count": int(len(late)),
            }
        )
    if not rows:
        raise ConfigurationError(
            "No fish had both early and late blocks for the permutation summary."
        )
    return pd.DataFrame(rows)


def permutation_test_mean_effect(
    fish_effects: pd.DataFrame,
    *,
    config: FishPermutationConfig = FishPermutationConfig(),
) -> pd.DataFrame:
    """Two-sided sign-flip test of the mean fish-level learning effect."""
    rows: list[dict[str, Any]] = []
    for (metric_id, outcome_id), subset in fish_effects.groupby(
        ["metric_id", "outcome_id"],
        observed=True,
        sort=True,
    ):
        effects = subset["learning_effect"].to_numpy(dtype=float)
        fish_count = int(effects.size)
        observed = float(np.mean(effects))
        if fish_count < 2:
            rows.append(
                {
                    "metric_id": str(metric_id),
                    "outcome_id": str(outcome_id),
                    "fish_count": fish_count,
                    "observed_mean_effect": observed,
                    "permutation_p_value": np.nan,
                    "n_permutations": config.n_permutations,
                    "null_mean": np.nan,
                    "null_sd": np.nan,
                    "diagnostic_status": "failed",
                    "error": "At least two fish are required.",
                }
            )
            continue
        seed_material = f"{config.seed}:{metric_id}:{outcome_id}".encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        # Include the observed sign pattern as permutation 0.
        null_means = np.empty(config.n_permutations + 1, dtype=float)
        null_means[0] = observed
        for index in range(1, config.n_permutations + 1):
            signs = rng.choice(np.array([-1.0, 1.0]), size=fish_count)
            null_means[index] = float(np.mean(signs * effects))
        extreme = np.abs(null_means) >= abs(observed) - 1e-15
        p_value = float(np.mean(extreme))
        rows.append(
            {
                "metric_id": str(metric_id),
                "outcome_id": str(outcome_id),
                "fish_count": fish_count,
                "observed_mean_effect": observed,
                "permutation_p_value": p_value,
                "n_permutations": int(config.n_permutations),
                "null_mean": float(np.mean(null_means)),
                "null_sd": float(np.std(null_means, ddof=1)),
                "diagnostic_status": "ok",
                "error": None,
            }
        )
    return pd.DataFrame(rows)


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


def build_candidate_fish_permutation(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    metric_recipe: str = "tail-candidate-corrected-v1",
    config: FishPermutationConfig = FishPermutationConfig(),
    overwrite: bool = False,
) -> FishPermutationResult:
    """Publish fish-level learning effects and sign-flip permutation tests."""
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
    model_input = build_candidate_model_input(
        outcomes,
        config=ModelInputConfig(
            alignment=config.alignment,
            activity_offset=config.activity_offset,
            require_block_label=True,
        ),
    )
    coverage = model_input_coverage(model_input)
    fish_effects = summarize_fish_learning_effects(model_input, config=config)
    population = permutation_test_mean_effect(fish_effects, config=config)

    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    fish_effects_path = output_dir / f"{RECIPE_ID}_fish_effects.parquet"
    population_path = output_dir / f"{RECIPE_ID}_population.parquet"
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
        for path in (fish_effects_path, population_path, summary_path, marker_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"{RECIPE_ID} outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{analysis_id}-{RECIPE_ID}-",
    ) as staging_root:
        staged_fish = staging_root / fish_effects_path.name
        staged_population = staging_root / population_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        records = {
            "fish_effects": _write_parquet(staged_fish, fish_effects),
            "population": _write_parquet(staged_population, population),
        }
        records["fish_effects"]["path"] = str(fish_effects_path)
        records["population"]["path"] = str(population_path)
        summary = {
            "recipe": RECIPE_ID,
            "scientific_status": "exploratory_fish_permutation",
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
            "workshop_family": "A_fish_level_permutation",
            "workshop_note": "Plans/Notes/STATISTICS_METHODOLOGY_WORKSHOP.md",
            "estimand": (
                "Population mean of fish-level (late - early) baseline-adjusted "
                "log activity; two-sided sign-flip permutation p-value."
            ),
            "outcome_ids": list(OUTCOME_SPECS),
            "inputs": inputs,
            "artifacts": records,
            "inference": {
                "performed": True,
                "confirmatory": False,
                "reason": (
                    "Alternative to mixed-effects for workshop comparison. "
                    "Not Gate-S approved. Fixture N makes p-values coarse."
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
                (staged_fish, fish_effects_path),
                (staged_population, population_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return FishPermutationResult(
        analysis_id=analysis_id,
        recording_ids=recording_ids,
        fish_effects_path=fish_effects_path,
        population_path=population_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
    )
