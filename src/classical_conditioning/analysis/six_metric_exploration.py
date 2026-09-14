"""Descriptive allDelay six-metric cohort exploration.

This module deliberately does not select a paper metric or emit p-values.  It
creates a reproducible, fish-level comparison of two distinct log-suppression
scores from the corrected six-metric candidate route.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.model_input import load_authenticated_trial_outcomes
from classical_conditioning.analysis.movement_state import METRIC_IDS
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.exceptions import ConfigurationError, SchemaValidationError

RECIPE_ID = "six-metric-allDelay-exploration"
METRIC_RECIPE = "tail-candidate-corrected-v1"
EXPECTED_METRICS = tuple(METRIC_IDS.values())
SCORE_SPECS = {
    "total_activity_log_suppression": (
        "baseline_total_activity",
        "response_total_activity",
        "All valid frames; rest is retained in both window means.",
    ),
    "bout_conditional_log_suppression": (
        "baseline_conditional_intensity",
        "conditional_intensity",
        "Only detector-moving frames contribute; this is not an immobility outcome.",
    ),
}


@dataclass(frozen=True)
class SixMetricExplorationConfig:
    experiment_name: str = "allDelay"
    alignment: str = "CS"
    early_blocks: tuple[str, ...] = ("Pre-train", "Train 1")
    late_blocks: tuple[str, ...] = ("Train 5", "Test 1", "Test 2", "Test 3")
    n_bootstrap: int = 4_999
    seed: int = 10

    def __post_init__(self) -> None:
        if self.experiment_name != "allDelay":
            raise ConfigurationError("This exploratory recipe is frozen to allDelay.")
        if self.alignment != "CS":
            raise ConfigurationError("This exploratory recipe is frozen to CS alignment.")
        if not self.early_blocks or not self.late_blocks:
            raise ConfigurationError("Early and late block sets must be non-empty.")
        if set(self.early_blocks) & set(self.late_blocks):
            raise ConfigurationError("Early and late block sets must be disjoint.")
        if self.n_bootstrap < 99:
            raise ConfigurationError("Use at least 99 fish-level bootstrap replicates.")


@dataclass(frozen=True)
class SixMetricExplorationResult:
    analysis_id: str
    recording_ids: tuple[str, ...]
    trial_scores_path: Path
    fish_effects_path: Path
    condition_summary_path: Path
    rank_path: Path
    correlation_path: Path
    summary_path: Path
    completion_marker_path: Path
    figure_paths: tuple[Path, ...]


def _validate_analysis_id(analysis_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
        raise ConfigurationError("Analysis ID must use letters, numbers, dot, underscore, or hyphen.")


def discover_completed_six_metric_recordings(project_dir: Path) -> tuple[str, ...]:
    """Return records whose canonical corrected trial marker is complete.

    Exact six-metric identity and the superseding trial-outcome schema are
    checked when authenticated artifacts are loaded below.
    """
    metadata = project_dir / "Metadata"
    suffix = "_candidate-trial-outcomes-corrected-v1_complete.json"
    recording_ids: list[str] = []
    for path in sorted(metadata.glob(f"*{suffix}")):
        recording_id = path.name[: -len(suffix)]
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ConfigurationError(f"Invalid corrected trial marker: {path}") from error
        if (
            marker.get("status") == "complete"
            and marker.get("recipe") == "candidate-trial-outcomes-corrected-v1"
            and marker.get("recording_id") == recording_id
        ):
            recording_ids.append(recording_id)
    if not recording_ids:
        raise FileNotFoundError("No completed corrected trial outcomes were found.")
    return tuple(recording_ids)


def build_trial_log_suppression_scores(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Create both no-offset log scores and retain invalid rows for audit."""
    required = {
        "recording_id", "fish_id", "condition_id", "alignment", "trial_id",
        "trial_number", "block_10_name", "metric_id", "baseline_valid_sample_count",
        "response_valid_sample_count", *(value for spec in SCORE_SPECS.values() for value in spec[:2]),
    }
    missing = required.difference(outcomes.columns)
    if missing:
        raise SchemaValidationError(f"Six-metric trial outcomes missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for row in outcomes.itertuples(index=False):
        base = {
            "recording_id": str(row.recording_id), "fish_id": str(row.fish_id),
            "condition_id": str(row.condition_id), "alignment": str(row.alignment),
            "trial_id": str(row.trial_id), "trial_number": int(row.trial_number),
            "block_10_name": None if pd.isna(row.block_10_name) else str(row.block_10_name),
            "metric_id": str(row.metric_id),
        }
        for score_type, (baseline_column, response_column, semantics) in SCORE_SPECS.items():
            baseline = float(getattr(row, baseline_column))
            response = float(getattr(row, response_column))
            reason: str | None = None
            if int(row.baseline_valid_sample_count) < 1:
                reason = "no_baseline_valid_samples"
            elif int(row.response_valid_sample_count) < 1:
                reason = "no_response_valid_samples"
            elif not np.isfinite(baseline) or not np.isfinite(response):
                reason = "nonfinite_window_value"
            elif baseline <= 0:
                reason = "nonpositive_baseline"
            elif response <= 0:
                reason = "nonpositive_response"
            rows.append({
                **base, "score_type": score_type, "semantics": semantics,
                "baseline_value": baseline, "response_value": response,
                "log_suppression": (float(np.log(baseline) - np.log(response)) if reason is None else np.nan),
                "included": reason is None, "exclusion_reason": reason,
            })
    return pd.DataFrame(rows)


def summarize_fish_learning_effects(scores: pd.DataFrame, config: SixMetricExplorationConfig) -> pd.DataFrame:
    """Summarize each fish's late-minus-early suppression for every metric."""
    frame = scores.loc[
        (scores["alignment"] == config.alignment) & scores["included"]
    ].copy()
    rows: list[dict[str, object]] = []
    keys = ["recording_id", "fish_id", "condition_id", "metric_id", "score_type"]
    for identity, subset in frame.groupby(keys, observed=True, sort=True):
        early = subset.loc[subset["block_10_name"].isin(config.early_blocks), "log_suppression"]
        late = subset.loc[subset["block_10_name"].isin(config.late_blocks), "log_suppression"]
        early = early[np.isfinite(early)]
        late = late[np.isfinite(late)]
        recording_id, fish_id, condition_id, metric_id, score_type = identity
        reason = None
        if early.empty:
            reason = "no_included_early_trials"
        elif late.empty:
            reason = "no_included_late_trials"
        rows.append({
            "recording_id": recording_id, "fish_id": fish_id, "condition_id": condition_id,
            "metric_id": metric_id, "score_type": score_type,
            "early_mean_log_suppression": float(early.mean()) if not early.empty else np.nan,
            "late_mean_log_suppression": float(late.mean()) if not late.empty else np.nan,
            "learning_effect": float(late.mean() - early.mean()) if reason is None else np.nan,
            "early_trial_count": int(len(early)), "late_trial_count": int(len(late)),
            "included": reason is None, "exclusion_reason": reason,
        })
    return pd.DataFrame(rows)


def _bootstrap_summary(values: np.ndarray, rng: np.random.Generator, n: int) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize_conditions(fish_effects: pd.DataFrame, config: SixMetricExplorationConfig) -> pd.DataFrame:
    """Fish-level bootstrap summaries and delay-minus-control contrast."""
    rows: list[dict[str, object]] = []
    frame = fish_effects.loc[fish_effects["included"]].copy()
    for (metric_id, score_type), subset in frame.groupby(["metric_id", "score_type"], observed=True, sort=True):
        groups = {name: value["learning_effect"].to_numpy(dtype=float) for name, value in subset.groupby("condition_id", observed=True)}
        if set(groups) != {"control", "delay"}:
            raise ConfigurationError(f"{metric_id}/{score_type} does not contain exactly control and delay fish.")
        seed = int(hashlib.sha256(f"{config.seed}:{metric_id}:{score_type}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        control, delay = groups["control"], groups["delay"]
        for condition, values in (("control", control), ("delay", delay)):
            mean, lower, upper = _bootstrap_summary(values, rng, config.n_bootstrap)
            rows.append({"metric_id": metric_id, "score_type": score_type, "summary_type": "condition_mean", "condition_id": condition, "fish_count": int(np.isfinite(values).sum()), "estimate": mean, "ci_lower": lower, "ci_upper": upper})
        if not len(control) or not len(delay):
            estimate = lower = upper = np.nan
        else:
            estimate = float(np.mean(delay) - np.mean(control))
            draws = (rng.choice(delay, size=(config.n_bootstrap, len(delay)), replace=True).mean(axis=1) - rng.choice(control, size=(config.n_bootstrap, len(control)), replace=True).mean(axis=1))
            lower, upper = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        rows.append({"metric_id": metric_id, "score_type": score_type, "summary_type": "delay_minus_control", "condition_id": "delay-minus-control", "fish_count": int(np.isfinite(delay).sum() + np.isfinite(control).sum()), "estimate": estimate, "ci_lower": lower, "ci_upper": upper})
    return pd.DataFrame(rows)


def _correlations(fish_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for score_type, subset in fish_effects.loc[fish_effects["included"]].groupby("score_type", observed=True):
        wide = subset.pivot(index="fish_id", columns="metric_id", values="learning_effect")
        corr = wide.corr(min_periods=3)
        for metric_a in corr.index:
            for metric_b in corr.columns:
                rows.append({"score_type": score_type, "metric_a": metric_a, "metric_b": metric_b, "pearson_r": corr.loc[metric_a, metric_b], "pair_count": int(wide[[metric_a, metric_b]].dropna().shape[0])})
    return pd.DataFrame(rows)


def _write_figures(condition_summary: pd.DataFrame, trial_scores: pd.DataFrame, output_dir: Path) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    figures: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    contrasts = condition_summary.loc[condition_summary["summary_type"] == "delay_minus_control"].copy()
    for score_type, subset in contrasts.groupby("score_type", observed=True):
        subset = subset.sort_values("estimate")
        fig, axis = plt.subplots(figsize=(9, 4.8))
        axis.errorbar(subset["estimate"], range(len(subset)), xerr=[subset["estimate"] - subset["ci_lower"], subset["ci_upper"] - subset["estimate"]], fmt="o", color="#1f6feb")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(range(len(subset)), subset["metric_id"])
        axis.set_xlabel("Delay − control late-minus-early log suppression")
        axis.set_title(score_type.replace("_", " "))
        fig.tight_layout()
        path = output_dir / f"{RECIPE_ID}_{score_type}_contrast.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        figures.append(path)
    audit = trial_scores.groupby(["score_type", "included"], observed=True).size().unstack(fill_value=0)
    fig, axis = plt.subplots(figsize=(7, 4))
    audit.plot.bar(stacked=True, ax=axis, color=["#d1242f", "#2da44e"])
    axis.set_ylabel("Trial-metric rows")
    axis.set_title("Log-suppression inclusion audit")
    fig.tight_layout()
    path = output_dir / f"{RECIPE_ID}_inclusion_audit.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    figures.append(path)
    return tuple(figures)


def build_six_metric_exploration(project_dir: Path, *, analysis_id: str, config: SixMetricExplorationConfig = SixMetricExplorationConfig(), overwrite: bool = False) -> SixMetricExplorationResult:
    """Build authenticated descriptive outputs for the completed six-metric cohort."""
    _validate_analysis_id(analysis_id)
    project_dir = project_dir.resolve()
    recording_ids = discover_completed_six_metric_recordings(project_dir)
    outcomes, inputs, resolved_recipe = load_authenticated_trial_outcomes(project_dir, recording_ids, metric_recipe=METRIC_RECIPE)
    if resolved_recipe != METRIC_RECIPE:
        raise ConfigurationError("Resolved metric recipe differs from six-metric exploration route.")
    observed_metrics = outcomes.assign(metric_id=outcomes["metric_id"].astype(str)).groupby(
        "recording_id", observed=True
    )["metric_id"].agg(lambda values: set(values))
    invalid_records = observed_metrics.loc[
        observed_metrics.apply(lambda values: values != set(EXPECTED_METRICS))
    ]
    if not invalid_records.empty:
        detail = {str(recording_id): sorted(values) for recording_id, values in invalid_records.items()}
        raise SchemaValidationError(
            "Every recording must contain exactly the six expected metric IDs; "
            f"invalid recordings: {detail}"
        )
    identity = outcomes[["recording_id", "experiment_id", "condition_id"]].drop_duplicates()
    if not identity["experiment_id"].astype(str).eq(config.experiment_name).all():
        raise ConfigurationError("Completed recordings are not all allDelay artifacts.")
    if set(identity["condition_id"].astype(str)) != {"control", "delay"}:
        raise ConfigurationError("Completed recordings must contain exactly control and delay conditions.")

    trial_scores = build_trial_log_suppression_scores(outcomes)
    fish_effects = summarize_fish_learning_effects(trial_scores, config)
    condition_summary = summarize_conditions(fish_effects, config)
    ranks = condition_summary.loc[condition_summary["summary_type"] == "delay_minus_control"].copy()
    ranks["exploratory_rank"] = ranks.groupby("score_type", observed=True)["estimate"].rank(ascending=False, method="min")
    correlations = _correlations(fish_effects)

    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    qc_dir = project_dir / "Quality checks" / "Analyses" / analysis_id
    figure_dir = project_dir / "Figures" / "PNG" / "Analyses" / analysis_id
    paths = {
        "trial_scores": output_dir / f"{RECIPE_ID}_trial_scores.parquet",
        "fish_effects": output_dir / f"{RECIPE_ID}_fish_effects.parquet",
        "condition_summary": output_dir / f"{RECIPE_ID}_condition_summary.parquet",
        "ranks": output_dir / f"{RECIPE_ID}_metric_ranks.parquet",
        "correlations": output_dir / f"{RECIPE_ID}_metric_correlations.parquet",
    }
    summary_path = qc_dir / f"{RECIPE_ID}_summary.json"
    marker_path = project_dir / "Metadata" / f"{analysis_id}_{RECIPE_ID}_complete.json"
    if any(path.exists() for path in (*paths.values(), summary_path, marker_path)) and not overwrite:
        raise FileExistsError("Six-metric exploration outputs already exist; choose a new analysis ID.")

    with artifact_staging(project_dir, prefix=f".{analysis_id}-{RECIPE_ID}-") as staging:
        staged = {name: staging / path.name for name, path in paths.items()}
        frames = {"trial_scores": trial_scores, "fish_effects": fish_effects, "condition_summary": condition_summary, "ranks": ranks, "correlations": correlations}
        for name, frame in frames.items():
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), staged[name], compression="zstd", write_statistics=True)
        figure_stage = staging / "figures"
        figure_paths = _write_figures(condition_summary, trial_scores, figure_stage)
        hashes = {name: sha256_file(path) for name, path in staged.items()}
        summary = {
            "recipe": RECIPE_ID, "scientific_status": "exploratory_descriptive", "paper_approved": False,
            "analysis_id": analysis_id, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "recording_ids": list(recording_ids), "metric_recipe": METRIC_RECIPE,
            "metric_ids": list(EXPECTED_METRICS), "inputs": inputs,
            "semantics": {name: spec[2] for name, spec in SCORE_SPECS.items()},
            "no_p_values": True, "artifacts": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
        }
        staged_summary = staging / summary_path.name
        staged_marker = staging / marker_path.name
        write_json_atomic(staged_summary, summary)
        write_json_atomic(staged_marker, {"status": "complete", "recipe": RECIPE_ID, "analysis_id": analysis_id, "recording_ids": list(recording_ids), "summary_sha256": sha256_file(staged_summary), "artifact_sha256": hashes})
        published_figures: list[Path] = []
        transactions = [(staged[name], path) for name, path in paths.items()] + [(staged_summary, summary_path), (staged_marker, marker_path)]
        for staged_figure in figure_paths:
            destination = figure_dir / staged_figure.name
            transactions.append((staged_figure, destination))
            published_figures.append(destination)
        publish_transaction(tuple(transactions), staging, overwrite=overwrite)

    return SixMetricExplorationResult(analysis_id, recording_ids, paths["trial_scores"], paths["fish_effects"], paths["condition_summary"], paths["ranks"], paths["correlations"], summary_path, marker_path, tuple(published_figures))
