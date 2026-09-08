"""Fish-level bootstrap intervals for early-vs-late learning effects (Step 10.4).

Resamples fish (never trials or frames) to estimate uncertainty of the mean
fish-level effect. Companion to candidate-fish-permutation-v1; not Gate S.
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

from classical_conditioning.analysis.fish_permutation import (
    DEFAULT_EARLY_BLOCKS,
    DEFAULT_LATE_BLOCKS,
    summarize_fish_learning_effects,
)
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

RECIPE_ID = "candidate-fish-bootstrap-v1"


@dataclass(frozen=True)
class FishBootstrapConfig:
    alignment: str = "CS"
    early_blocks: tuple[str, ...] = DEFAULT_EARLY_BLOCKS
    late_blocks: tuple[str, ...] = DEFAULT_LATE_BLOCKS
    n_bootstrap: int = 1_999
    seed: int = 10
    confidence_level: float = 0.95
    activity_offset: float = 1e-6

    def __post_init__(self) -> None:
        if self.alignment not in {"CS", "US"}:
            raise ConfigurationError("Bootstrap alignment must be CS or US.")
        if not self.early_blocks or not self.late_blocks:
            raise ConfigurationError("Early and late block sets must be non-empty.")
        if set(self.early_blocks).intersection(self.late_blocks):
            raise ConfigurationError("Early and late block sets must be disjoint.")
        if self.n_bootstrap < 99:
            raise ConfigurationError("Use at least 99 bootstrap replicates.")
        if not 0.5 < self.confidence_level < 1.0:
            raise ConfigurationError("Confidence level must be in (0.5, 1).")
        if self.activity_offset <= 0:
            raise ConfigurationError("Activity offset must be positive.")


@dataclass(frozen=True)
class FishBootstrapResult:
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


def _config_hash(config: FishBootstrapConfig) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bootstrap_mean_effect(
    fish_effects: pd.DataFrame,
    *,
    config: FishBootstrapConfig = FishBootstrapConfig(),
) -> pd.DataFrame:
    """Percentile bootstrap of the mean fish-level learning effect."""
    alpha = 1.0 - config.confidence_level
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    rows: list[dict[str, Any]] = []
    for (metric_id, outcome_id), subset in fish_effects.groupby(
        ["metric_id", "outcome_id"],
        observed=True,
        sort=True,
    ):
        effects = subset["learning_effect"].to_numpy(dtype=float)
        fish_count = int(effects.size)
        observed = float(np.mean(effects)) if fish_count else float("nan")
        if fish_count < 2:
            rows.append(
                {
                    "metric_id": str(metric_id),
                    "outcome_id": str(outcome_id),
                    "fish_count": fish_count,
                    "observed_mean_effect": observed,
                    "bootstrap_mean": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "n_bootstrap": config.n_bootstrap,
                    "confidence_level": config.confidence_level,
                    "resample_unit": "fish_id",
                    "diagnostic_status": "failed",
                    "error": "At least two fish are required.",
                }
            )
            continue
        seed_material = f"{config.seed}:{metric_id}:{outcome_id}".encode("utf-8")
        seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        replicates = np.empty(config.n_bootstrap, dtype=float)
        for index in range(config.n_bootstrap):
            sample = rng.choice(effects, size=fish_count, replace=True)
            replicates[index] = float(np.mean(sample))
        rows.append(
            {
                "metric_id": str(metric_id),
                "outcome_id": str(outcome_id),
                "fish_count": fish_count,
                "observed_mean_effect": observed,
                "bootstrap_mean": float(np.mean(replicates)),
                "ci_lower": float(np.percentile(replicates, lower_q)),
                "ci_upper": float(np.percentile(replicates, upper_q)),
                "n_bootstrap": int(config.n_bootstrap),
                "confidence_level": float(config.confidence_level),
                "resample_unit": "fish_id",
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


def build_candidate_fish_bootstrap(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    metric_recipe: str = "tail-candidate-corrected-v1",
    config: FishBootstrapConfig = FishBootstrapConfig(),
    overwrite: bool = False,
) -> FishBootstrapResult:
    """Publish fish-level effects and fish-resampled percentile intervals."""
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
    # Reuse the same early/late fish collapse as the permutation route.
    from classical_conditioning.analysis.fish_permutation import FishPermutationConfig

    fish_effects = summarize_fish_learning_effects(
        model_input,
        config=FishPermutationConfig(
            alignment=config.alignment,
            early_blocks=config.early_blocks,
            late_blocks=config.late_blocks,
            activity_offset=config.activity_offset,
        ),
    )
    population = bootstrap_mean_effect(fish_effects, config=config)

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
            "scientific_status": "exploratory_fish_bootstrap",
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
            "workshop_note": "Plans/STATISTICS_METHODOLOGY_WORKSHOP.md",
            "estimand": (
                "Percentile CI for the mean fish-level (late - early) "
                "baseline-adjusted log activity; fish are the resample unit."
            ),
            "outcome_ids": list(OUTCOME_SPECS),
            "inputs": inputs,
            "artifacts": records,
            "inference": {
                "performed": True,
                "confirmatory": False,
                "reason": (
                    "Fish-level bootstrap for Step 10.4 plumbing and sensitivity. "
                    "Not Gate-S approved. Fixture N yields wide intervals."
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

    return FishBootstrapResult(
        analysis_id=analysis_id,
        recording_ids=recording_ids,
        fish_effects_path=fish_effects_path,
        population_path=population_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
    )
