"""Outcome-identical descriptive comparison of candidate activity metrics."""

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
    CandidateMetricSource,
    resolve_candidate_metric_source,
)
from classical_conditioning.analysis.temporal_profiles import METRIC_IDS
from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    SchemaValidationError,
)
from classical_conditioning.paths import condition_from_recording_name

RECIPE_ID = "candidate-metric-comparison-v1"
SOURCE_RECIPE_ID = "candidate-temporal-outcomes-v2"
CORRECTED_RECIPE_ID = "candidate-metric-comparison-corrected-v1"
OUTCOME_COLUMNS = {
    "total-activity": "Total activity mean",
    "movement-probability": "Movement probability",
    "fraction-time-moving": "Fraction time moving",
    "conditional-intensity": "Conditional intensity mean",
    "bout-rate": "Bout rate per minute",
}


@dataclass(frozen=True)
class MetricComparisonConfig:
    baseline_window_s: tuple[float, float] = (-15.0, 0.0)
    response_window_s: tuple[float, float] = (0.0, 9.0)
    interval_closure: str = "left"
    recording_aggregation: str = "equal_trial_weight"
    cohort_aggregation: str = "equal_recording_weight"

    def __post_init__(self) -> None:
        if self.baseline_window_s[0] >= self.baseline_window_s[1]:
            raise ConfigurationError("Baseline comparison window must be increasing.")
        if self.response_window_s[0] >= self.response_window_s[1]:
            raise ConfigurationError("Response comparison window must be increasing.")
        if self.baseline_window_s[1] > self.response_window_s[0]:
            raise ConfigurationError("Comparison windows cannot overlap.")
        if self.interval_closure != "left":
            raise ConfigurationError(
                f"{RECIPE_ID} uses left-closed, right-open windows."
            )
        if self.recording_aggregation != "equal_trial_weight":
            raise ConfigurationError(
                f"{RECIPE_ID} gives every trial equal weight within recording."
            )
        if self.cohort_aggregation != "equal_recording_weight":
            raise ConfigurationError(
                f"{RECIPE_ID} gives every recording equal cohort weight."
            )


@dataclass(frozen=True)
class MetricComparisonResult:
    analysis_id: str
    recording_ids: tuple[str, ...]
    artifact_paths: dict[str, Path]
    summary_path: Path
    completion_marker_path: Path


@dataclass(frozen=True)
class _VerifiedTemporalProfiles:
    path: Path
    summary_path: Path
    marker_path: Path
    digest: str
    state: tuple[int, int]


def _config_hash(config: MetricComparisonConfig) -> str:
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


def _verify_temporal_profiles(
    project_dir: Path,
    recording_id: str,
    source: CandidateMetricSource,
) -> _VerifiedTemporalProfiles:
    path = (
        project_dir
        / "Processed data"
        / recording_id
        / source.temporal_artifact_name
    )
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.temporal_summary_name
    )
    marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{source.temporal_marker_suffix}"
    )
    missing = [
        candidate
        for candidate in (path, summary_path, marker_path)
        if not candidate.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing candidate temporal artifacts: {missing}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError(
            f"Candidate temporal metadata is invalid JSON: {error}"
        ) from error
    digest = sha256_file(path)
    if (
        summary.get("recipe") != source.temporal_recipe
        or summary.get("recording_id") != recording_id
        or summary.get("artifact", {}).get("sha256") != digest
        or marker.get("status") != "complete"
        or marker.get("recipe") != source.temporal_recipe
        or marker.get("recording_id") != recording_id
        or marker.get("profiles_sha256") != digest
        or marker.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ArtifactIntegrityError(
            f"Candidate temporal lineage is invalid for {recording_id}."
        )
    stat = path.stat()
    return _VerifiedTemporalProfiles(
        path=path,
        summary_path=summary_path,
        marker_path=marker_path,
        digest=digest,
        state=(stat.st_size, stat.st_mtime_ns),
    )


def summarize_candidate_metric_windows(
    profiles: pd.DataFrame,
    *,
    config: MetricComparisonConfig = MetricComparisonConfig(),
) -> pd.DataFrame:
    """Build equal-trial recording summaries for every metric and outcome."""
    required = {
        "Recording ID",
        "Trial type",
        "Trial number",
        "Time bin center (s)",
        "Metric ID",
        "Valid fraction",
        "Detector valid fraction",
        "Acquisition coverage",
        *OUTCOME_COLUMNS.values(),
    }
    missing = required.difference(profiles.columns)
    if missing:
        raise SchemaValidationError(
            f"Candidate temporal profiles are missing columns: {sorted(missing)}"
        )
    observed_metrics = set(profiles["Metric ID"].astype(str).unique())
    expected_metrics = set(METRIC_IDS.values())
    if observed_metrics != expected_metrics:
        raise SchemaValidationError(
            "Candidate temporal profiles do not contain exactly the five "
            f"candidate metrics: {sorted(observed_metrics)}"
        )
    for recording_id, recording in profiles.groupby(
        "Recording ID",
        observed=True,
        sort=False,
    ):
        recording_metrics = set(recording["Metric ID"].astype(str).unique())
        if recording_metrics != expected_metrics:
            raise SchemaValidationError(
                f"Recording {recording_id!r} does not contain exactly the five "
                f"candidate metrics: {sorted(recording_metrics)}"
            )

    rows: list[dict[str, Any]] = []
    group_columns = ["Recording ID", "Trial type", "Metric ID"]
    for identity, metric_frame in profiles.groupby(
        group_columns,
        observed=True,
        sort=True,
    ):
        recording_id, alignment, metric_id = map(str, identity)
        metric_time = metric_frame["Time bin center (s)"].to_numpy(dtype=float)
        masks = {
            "baseline": (
                (metric_time >= config.baseline_window_s[0])
                & (metric_time < config.baseline_window_s[1])
            ),
            "response": (
                (metric_time >= config.response_window_s[0])
                & (metric_time < config.response_window_s[1])
            ),
        }
        for outcome_id, column in OUTCOME_COLUMNS.items():
            trial_rows = []
            for trial_number, trial in metric_frame.groupby(
                "Trial number",
                observed=True,
                sort=True,
            ):
                trial_time = trial["Time bin center (s)"].to_numpy(dtype=float)
                trial_masks = {
                    "baseline": (
                        (trial_time >= config.baseline_window_s[0])
                        & (trial_time < config.baseline_window_s[1])
                    ),
                    "response": (
                        (trial_time >= config.response_window_s[0])
                        & (trial_time < config.response_window_s[1])
                    ),
                }
                values = trial[column].to_numpy(dtype=float)
                baseline = values[trial_masks["baseline"]]
                response = values[trial_masks["response"]]
                trial_rows.append(
                    {
                        "trial_number": int(trial_number),
                        "baseline": (
                            float(np.nanmean(baseline))
                            if np.isfinite(baseline).any()
                            else np.nan
                        ),
                        "response": (
                            float(np.nanmean(response))
                            if np.isfinite(response).any()
                            else np.nan
                        ),
                    }
                )
            trials = pd.DataFrame(trial_rows)
            complete = trials["baseline"].notna() & trials["response"].notna()
            paired_trials = trials.loc[complete]
            baseline_mean = float(paired_trials["baseline"].mean())
            response_mean = float(paired_trials["response"].mean())
            baseline_sd = float(paired_trials["baseline"].std(ddof=1))
            raw_difference = response_mean - baseline_mean
            standardized_difference = (
                raw_difference / baseline_sd
                if np.isfinite(baseline_sd) and baseline_sd > 0
                else np.nan
            )
            coverage = metric_frame.loc[
                masks["baseline"] | masks["response"],
                [
                    "Valid fraction",
                    "Detector valid fraction",
                    "Acquisition coverage",
                ],
            ]
            rows.append(
                {
                    "Recording ID": recording_id,
                    "Trial type": alignment,
                    "Metric ID": metric_id,
                    "Outcome ID": outcome_id,
                    "Baseline mean": baseline_mean,
                    "Response mean": response_mean,
                    "Response minus baseline": raw_difference,
                    "Baseline trial SD": baseline_sd,
                    "Standardized difference": standardized_difference,
                    "Trial count": int(len(trials)),
                    "Baseline trial count": int(trials["baseline"].notna().sum()),
                    "Response trial count": int(trials["response"].notna().sum()),
                    "Complete trial count": int(complete.sum()),
                    "Mean valid fraction": float(coverage["Valid fraction"].mean()),
                    "Mean detector valid fraction": float(
                        coverage["Detector valid fraction"].mean()
                    ),
                    "Mean acquisition coverage": float(
                        coverage["Acquisition coverage"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_candidate_metric_cohort(
    recording_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Average recording-level summaries so each recording has equal weight.

    When ``Condition ID`` is present, one cohort row set is emitted for every
    condition and one pooled ``all`` group.
    """
    required = {
        "Recording ID",
        "Trial type",
        "Metric ID",
        "Outcome ID",
        "Baseline mean",
        "Response mean",
        "Response minus baseline",
        "Standardized difference",
        "Mean valid fraction",
        "Mean detector valid fraction",
        "Mean acquisition coverage",
    }
    missing = required.difference(recording_summary.columns)
    if missing:
        raise SchemaValidationError(
            f"Recording metric summary is missing columns: {sorted(missing)}"
        )
    value_columns = [
        "Baseline mean",
        "Response mean",
        "Response minus baseline",
        "Standardized difference",
        "Mean valid fraction",
        "Mean detector valid fraction",
        "Mean acquisition coverage",
    ]
    frames: list[tuple[str, pd.DataFrame]] = [("all", recording_summary)]
    if "Condition ID" in recording_summary.columns:
        for condition, frame in recording_summary.groupby(
            "Condition ID",
            observed=True,
            sort=True,
        ):
            if str(condition).strip():
                frames.append((str(condition), frame))

    rows = []
    for cohort_group, grouped in frames:
        for identity, frame in grouped.groupby(
            ["Trial type", "Metric ID", "Outcome ID"],
            observed=True,
            sort=True,
        ):
            alignment, metric_id, outcome_id = map(str, identity)
            row: dict[str, Any] = {
                "Cohort group": cohort_group,
                "Trial type": alignment,
                "Metric ID": metric_id,
                "Outcome ID": outcome_id,
                "Recording count": int(frame["Recording ID"].nunique()),
            }
            for column in value_columns:
                values = frame[column].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                row[f"Contributing recordings {column}"] = int(finite.size)
                row[f"Mean {column}"] = (
                    float(np.mean(finite)) if finite.size else np.nan
                )
                row[f"SD {column}"] = (
                    float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return {
        "path": "",
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": len(table.column_names),
        "size_bytes": path.stat().st_size,
        "compression": "zstd",
        "compression_lossless": True,
    }


def comparison_config_for_experiment(
    experiment_name: str,
    config: MetricComparisonConfig | None = None,
) -> MetricComparisonConfig:
    spec = get_experiment_spec(experiment_name)
    base = config or MetricComparisonConfig()
    return MetricComparisonConfig(
        baseline_window_s=base.baseline_window_s,
        response_window_s=(
            spec.conditioned_response_window.start_s,
            spec.conditioned_response_window.end_s,
        ),
        interval_closure=base.interval_closure,
        recording_aggregation=base.recording_aggregation,
        cohort_aggregation=base.cohort_aggregation,
    )


def _condition_for_recording(project_dir: Path, recording_id: str) -> str:
    manifest_path = project_dir / "Metadata" / f"{recording_id}_source_manifest.json"
    if not manifest_path.is_file():
        return ""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("condition_id"):
        return str(payload["condition_id"])
    name = payload.get("recording_name")
    if name:
        return condition_from_recording_name(str(name))
    return ""


def build_candidate_metric_comparison(
    project_dir: Path,
    recording_ids: Iterable[str],
    *,
    analysis_id: str,
    config: MetricComparisonConfig | None = None,
    experiment_name: str | None = None,
    metric_recipe: str | None = None,
    comparison_recipe: str | None = None,
    overwrite: bool = False,
) -> MetricComparisonResult:
    """Publish a non-inferential five-metric comparison over explicit recordings."""
    _validate_analysis_id(analysis_id)
    recording_ids = tuple(dict.fromkeys(recording_ids))
    if not recording_ids:
        raise ConfigurationError("At least one recording ID is required.")
    if experiment_name:
        config = comparison_config_for_experiment(experiment_name, config)
    elif config is None:
        config = MetricComparisonConfig()
    route = resolve_candidate_metric_source(
        metric_recipe=metric_recipe,
        comparison_recipe=comparison_recipe,
    )
    recipe_id = route.comparison_recipe
    project_dir = project_dir.resolve()
    sources = {
        recording_id: _verify_temporal_profiles(project_dir, recording_id, route)
        for recording_id in recording_ids
    }
    profiles = pd.concat(
        [pq.read_table(source.path).to_pandas() for source in sources.values()],
        ignore_index=True,
    )
    recording_summary = summarize_candidate_metric_windows(
        profiles,
        config=config,
    )
    recording_summary["Condition ID"] = recording_summary["Recording ID"].map(
        lambda recording_id: _condition_for_recording(project_dir, str(recording_id))
    )
    cohort_summary = summarize_candidate_metric_cohort(recording_summary)

    output_dir = project_dir / "Processed data" / "Analyses" / analysis_id
    artifact_paths = {
        "recording_summary": output_dir / f"{recipe_id}_recording_summary.parquet",
        "cohort_summary": output_dir / f"{recipe_id}_cohort_summary.parquet",
    }
    summary_path = (
        project_dir
        / "Quality checks"
        / "Analyses"
        / analysis_id
        / f"{recipe_id}_summary.json"
    )
    marker_path = (
        project_dir / "Metadata" / f"{analysis_id}_{recipe_id}_complete.json"
    )
    outputs = (*artifact_paths.values(), summary_path, marker_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{recipe_id} outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with artifact_staging(
        project_dir,
        prefix=f".{analysis_id}-{recipe_id}-",
    ) as staging_root:
        staged_paths = {
            name: staging_root / final_path.name
            for name, final_path in artifact_paths.items()
        }
        records = {
            "recording_summary": _write_parquet(
                staged_paths["recording_summary"],
                recording_summary,
            ),
            "cohort_summary": _write_parquet(
                staged_paths["cohort_summary"],
                cohort_summary,
            ),
        }
        for name, final_path in artifact_paths.items():
            records[name]["path"] = str(final_path)

        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        summary = {
            "recipe": recipe_id,
            "scientific_status": route.scientific_status,
            "metric_recipe": route.metric_recipe,
            "temporal_recipe": route.temporal_recipe,
            "analysis_id": analysis_id,
            "experiment_name": experiment_name,
            "recording_ids": list(recording_ids),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "config_sha256": _config_hash(config),
            "metric_ids": list(METRIC_IDS.values()),
            "outcome_ids": list(OUTCOME_COLUMNS),
            "inputs": {
                recording_id: {
                    "recipe": route.temporal_recipe,
                    "path": str(source.path),
                    "sha256": source.digest,
                    "summary_sha256": sha256_file(source.summary_path),
                }
                for recording_id, source in sources.items()
            },
            "artifacts": records,
            "selection": {
                "performed": False,
                "paper_approved": False,
                "reason": (
                    "This artifact applies identical descriptive summaries to all "
                    "five metrics. Metric selection requires prespecified weights, "
                    "reviewed trace annotations, independent validation, and a "
                    "multi-recording cohort."
                ),
            },
            "inference": {
                "performed": False,
                "reason": (
                    "No inferential comparison is valid until the primary cohort, "
                    "outcome, metric-selection partition, and statistical model are "
                    "approved."
                ),
            },
        }
        write_json_atomic(staged_summary, summary)
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": recipe_id,
                "analysis_id": analysis_id,
                "recording_ids": list(recording_ids),
                "artifact_sha256": {
                    name: record["sha256"] for name, record in records.items()
                },
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        for recording_id, source in sources.items():
            stat = source.path.stat()
            if (
                source.state != (stat.st_size, stat.st_mtime_ns)
                or sha256_file(source.path) != source.digest
            ):
                raise ArtifactIntegrityError(
                    f"Candidate temporal input changed during comparison: {recording_id}"
                )
        publish_transaction(
            tuple(
                (staged_paths[name], artifact_paths[name])
                for name in artifact_paths
            )
            + (
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return MetricComparisonResult(
        analysis_id=analysis_id,
        recording_ids=recording_ids,
        artifact_paths=artifact_paths,
        summary_path=summary_path,
        completion_marker_path=marker_path,
    )
