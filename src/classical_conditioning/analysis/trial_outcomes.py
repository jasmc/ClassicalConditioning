"""Exact per-trial candidate outcomes from authenticated measured-time artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.movement_state import (
    CandidateMetricSource,
    DETECTOR_COLUMNS,
    METRIC_IDS,
    resolve_candidate_metric_source,
)
from classical_conditioning.artifacts import (
    VerifiedArtifactSet,
    artifact_staging,
    load_and_verify_source_manifest,
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
from classical_conditioning.preprocessing.candidates_v1 import CANDIDATE_COLUMNS

DEFAULT_TRIAL_RECIPE_ID = "candidate-trial-outcomes-v1"
RECIPE_ID = DEFAULT_TRIAL_RECIPE_ID  # retained for development-route callers


@dataclass(frozen=True)
class TrialOutcomeConfig:
    baseline_window_s: tuple[float, float] = (-15.0, 0.0)
    response_window_s: tuple[float, float] = (0.0, 9.0)
    interval_closure: str = "left"

    def __post_init__(self) -> None:
        if self.baseline_window_s[0] >= self.baseline_window_s[1]:
            raise ConfigurationError("Baseline trial window must be increasing.")
        if self.response_window_s[0] >= self.response_window_s[1]:
            raise ConfigurationError("Response trial window must be increasing.")
        if self.baseline_window_s[1] > self.response_window_s[0]:
            raise ConfigurationError("Trial outcome windows cannot overlap.")
        if self.interval_closure != "left":
            raise ConfigurationError(
                f"{RECIPE_ID} uses left-closed, right-open windows."
            )


@dataclass(frozen=True)
class TrialOutcomeResult:
    recording_id: str
    outcomes_path: Path
    coverage_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int


def parse_recording_identity(
    recording_name: str,
    *,
    experiment_name: str,
) -> dict[str, str]:
    """Parse the acquisition identity under the validated legacy naming contract."""
    parts = recording_name.split("_")
    if len(parts) < 6:
        raise SchemaValidationError(
            f"Cannot parse recording identity: {recording_name!r}"
        )
    condition_source = parts[2].lower()
    experiment = get_experiment_spec(experiment_name)
    condition_by_source = {
        condition.source_name.lower(): condition.condition_id
        for condition in experiment.conditions
    }
    if condition_source not in condition_by_source:
        raise SchemaValidationError(
            f"Recording condition is not configured for {experiment_name}: "
            f"{condition_source!r}"
        )
    return {
        "experiment_id": experiment.experiment_id,
        "recording_id": f"{parts[0]}_{parts[1]}",
        "fish_id": f"{parts[0]}_{parts[1]}",
        "condition_id": condition_by_source[condition_source],
        "day": parts[0],
        "fish_number": parts[1],
    }


def _window_mask(
    absolute_time: np.ndarray,
    event_start_ms: int,
    window_s: tuple[float, float],
) -> np.ndarray:
    relative_ms = absolute_time - event_start_ms
    return (
        (relative_ms >= int(round(window_s[0] * 1_000)))
        & (relative_ms < int(round(window_s[1] * 1_000)))
    )


def _finite_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask & np.isfinite(values)]
    return float(np.mean(selected)) if selected.size else np.nan


def _weighted_fraction(
    numerator: np.ndarray,
    denominator: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float:
    eligible = mask & denominator & np.isfinite(weights) & (weights > 0)
    denominator_weight = float(np.sum(weights[eligible]))
    if denominator_weight <= 0:
        return np.nan
    numerator_weight = float(np.sum(weights[eligible & numerator]))
    return numerator_weight / denominator_weight


def aggregate_trial_outcomes(
    frames: pd.DataFrame,
    movement: pd.DataFrame,
    protocol: pd.DataFrame,
    *,
    identity: dict[str, str],
    experiment_name: str,
    config: TrialOutcomeConfig = TrialOutcomeConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate exact trial windows without population aggregation."""
    required_frames = {
        "FrameID",
        "AbsoluteTime",
        "FrameStep",
        "DeltaTimeMs",
        *CANDIDATE_COLUMNS,
    }
    missing_frames = required_frames.difference(frames.columns)
    if missing_frames:
        raise SchemaValidationError(
            f"Candidate frames are missing columns: {sorted(missing_frames)}"
        )
    required_movement = {"FrameID", "AbsoluteTime", *DETECTOR_COLUMNS}
    missing_movement = required_movement.difference(movement.columns)
    if missing_movement:
        raise SchemaValidationError(
            f"Movement state is missing columns: {sorted(missing_movement)}"
        )
    if not np.array_equal(
        frames["FrameID"].to_numpy(dtype=np.int64),
        movement["FrameID"].to_numpy(dtype=np.int64),
    ) or not np.array_equal(
        frames["AbsoluteTime"].to_numpy(dtype=np.int64),
        movement["AbsoluteTime"].to_numpy(dtype=np.int64),
    ):
        raise SchemaValidationError(
            "Candidate frame and movement-state identities do not align."
        )
    required_protocol = {"Type", "Beg", "End"}
    missing_protocol = required_protocol.difference(protocol.columns)
    if missing_protocol:
        raise SchemaValidationError(
            f"Protocol is missing columns: {sorted(missing_protocol)}"
        )

    experiment = get_experiment_spec(experiment_name)
    trial_specs = {
        (trial.alignment.value, trial.trial_number): trial
        for trial in experiment.analysis_trials
    }
    absolute_time = frames["AbsoluteTime"].to_numpy(dtype=np.int64)
    delta_time = frames["DeltaTimeMs"].to_numpy(dtype=float)
    adjacent = frames["FrameStep"].to_numpy(dtype=np.int64) == 1
    event_types = {"Cycle": "CS", "Reinforcer": "US"}
    counters = {"CS": 0, "US": 0}
    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    relevant = protocol.loc[
        protocol["Type"].astype(str).isin(event_types)
    ].sort_values("Beg", kind="stable")
    for event in relevant.itertuples(index=False):
        alignment = event_types[str(event.Type)]
        counters[alignment] += 1
        trial_number = counters[alignment]
        event_start = int(event.Beg)
        baseline_mask = _window_mask(
            absolute_time,
            event_start,
            config.baseline_window_s,
        )
        response_mask = _window_mask(
            absolute_time,
            event_start,
            config.response_window_s,
        )
        trial_spec = trial_specs.get((alignment, trial_number))
        event_id = f"{alignment}-{trial_number:03d}-{event_start}"
        trial_id = f"{identity['recording_id']}:{alignment}:{trial_number:03d}"

        # The detector is shared, so validity, movement, and bout identity are
        # read once and reused for every metric.
        valid = movement["valid"].to_numpy(dtype=bool)
        moving = movement["moving"].to_numpy(dtype=bool)
        bout_ids = movement["bout_id"].to_numpy(dtype=np.int64)
        for source_column, metric_id in METRIC_IDS.items():
            values = frames[source_column].to_numpy(dtype=float)
            bout_start = (bout_ids > 0) & np.concatenate(
                ([True], bout_ids[1:] != bout_ids[:-1])
            )
            response_valid = response_mask & valid & adjacent
            response_moving = response_valid & moving
            baseline_valid = baseline_mask & valid & adjacent
            response_duration_ms = float(
                np.sum(delta_time[response_valid & np.isfinite(delta_time)])
            )
            bout_count = int(np.count_nonzero(response_mask & bout_start))
            response_bout_ids = np.unique(bout_ids[response_moving & (bout_ids > 0)])
            bout_durations = [
                float(
                    np.sum(
                        delta_time[
                            (bout_ids == bout_id)
                            & valid
                            & adjacent
                            & np.isfinite(delta_time)
                        ]
                    )
                )
                for bout_id in response_bout_ids
            ]
            valid_response_values = response_valid & np.isfinite(values)
            conditional_values = response_moving & np.isfinite(values)
            rows.append(
                {
                    **identity,
                    "trial_id": trial_id,
                    "alignment": alignment,
                    "event_id": event_id,
                    "event_start_absolute_time_ms": event_start,
                    "trial_number": trial_number,
                    "phase": (
                        trial_spec.phase.value if trial_spec is not None else None
                    ),
                    "block_10_id": (
                        trial_spec.block_10_id if trial_spec is not None else None
                    ),
                    "block_10_name": (
                        trial_spec.block_10_name if trial_spec is not None else None
                    ),
                    "is_catch": (
                        trial_spec.catch if trial_spec is not None else None
                    ),
                    "metric_id": metric_id,
                    "detector_id": "quiet-window-hysteresis-v1",
                    "baseline_total_activity": _finite_mean(
                        values,
                        baseline_valid,
                    ),
                    "response_total_activity": _finite_mean(
                        values,
                        response_valid,
                    ),
                    "movement_probability": (
                        float(np.mean(moving[response_valid]))
                        if np.any(response_valid)
                        else np.nan
                    ),
                    "fraction_time_moving": _weighted_fraction(
                        moving,
                        valid,
                        delta_time,
                        response_mask & adjacent,
                    ),
                    "conditional_intensity": (
                        float(np.mean(values[conditional_values]))
                        if np.any(conditional_values)
                        else np.nan
                    ),
                    "bout_count": bout_count,
                    "bout_rate_per_minute": (
                        bout_count * 60_000 / response_duration_ms
                        if response_duration_ms > 0
                        else np.nan
                    ),
                    "mean_bout_duration_ms": (
                        float(np.mean(bout_durations))
                        if bout_durations
                        else np.nan
                    ),
                    "baseline_valid_sample_count": int(
                        np.count_nonzero(baseline_valid)
                    ),
                    "response_valid_sample_count": int(
                        np.count_nonzero(response_valid)
                    ),
                    "response_moving_sample_count": int(
                        np.count_nonzero(response_moving)
                    ),
                }
            )
            coverage_rows.append(
                {
                    **identity,
                    "trial_id": trial_id,
                    "alignment": alignment,
                    "trial_number": trial_number,
                    "metric_id": metric_id,
                    "baseline_sample_count": int(
                        np.count_nonzero(baseline_mask)
                    ),
                    "baseline_valid_sample_count": int(
                        np.count_nonzero(baseline_valid)
                    ),
                    "baseline_valid_fraction": (
                        float(np.mean(valid[baseline_mask]))
                        if np.any(baseline_mask)
                        else 0.0
                    ),
                    "response_sample_count": int(
                        np.count_nonzero(response_mask)
                    ),
                    "response_valid_sample_count": int(
                        np.count_nonzero(response_valid)
                    ),
                    "response_valid_fraction": (
                        float(np.mean(valid[response_mask]))
                        if np.any(response_mask)
                        else 0.0
                    ),
                    "response_valid_duration_ms": response_duration_ms,
                    "missing_reason": (
                        None
                        if np.any(valid_response_values)
                        else (
                            "no_acquisition_samples"
                            if not np.any(response_mask)
                            else "no_detector_valid_samples"
                        )
                    ),
                }
            )
    outcomes = pd.DataFrame(rows)
    coverage = pd.DataFrame(coverage_rows)
    key = ["recording_id", "trial_id", "alignment", "metric_id"]
    if outcomes.duplicated(key).any() or coverage.duplicated(key).any():
        raise SchemaValidationError("Trial outcome identities are not unique.")
    return outcomes, coverage


def _verify_inputs(
    project_dir: Path,
    recording_id: str,
    *,
    metric_recipe: str = "tail-candidate-development-v1",
) -> tuple[str, Path, Path, Path, dict[str, str], CandidateMetricSource]:
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    source_dir = project_dir / "Processed data" / recording_id
    frame_path = source_dir / source.metrics_name
    movement_path = source_dir / source.movement_artifact_name
    protocol_path = source_dir / "stimulus_events.parquet"
    candidate_marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{source.metric_marker_suffix}"
    )
    movement_marker_path = (
        project_dir
        / "Metadata"
        / f"{recording_id}_{source.movement_marker_suffix}"
    )
    candidate_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.metric_summary_name
    )
    movement_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.movement_summary_name
    )
    recording_name, source_artifacts, _ = load_and_verify_source_manifest(
        project_dir,
        recording_id,
    )
    candidate_marker = json.loads(
        candidate_marker_path.read_text(encoding="utf-8")
    )
    movement_marker = json.loads(
        movement_marker_path.read_text(encoding="utf-8")
    )
    candidate_summary = json.loads(
        candidate_summary_path.read_text(encoding="utf-8")
    )
    movement_summary = json.loads(
        movement_summary_path.read_text(encoding="utf-8")
    )
    frame_hash = sha256_file(frame_path)
    movement_hash = sha256_file(movement_path)
    if (
        candidate_marker.get("recipe") != source.metric_recipe
        or movement_marker.get("recipe") != source.movement_recipe
        or candidate_marker.get("metrics_sha256") != frame_hash
        or candidate_marker.get("summary_sha256")
        != sha256_file(candidate_summary_path)
        or candidate_summary.get("artifact", {}).get("sha256") != frame_hash
        or movement_marker.get("movement_sha256") != movement_hash
        or movement_marker.get("summary_sha256")
        != sha256_file(movement_summary_path)
        or movement_summary.get("artifact", {}).get("sha256") != movement_hash
        or movement_summary.get("inputs", {})
        .get("candidate_metrics", {})
        .get("sha256")
        != frame_hash
        or sha256_file(protocol_path) != source_artifacts["protocol"]["sha256"]
    ):
        raise ArtifactIntegrityError(
            f"Candidate trial inputs have invalid lineage for {recording_id}."
        )
    return (
        recording_name,
        frame_path,
        movement_path,
        protocol_path,
        {
            "candidate_frames": frame_hash,
            "movement_state": movement_hash,
            "protocol": source_artifacts["protocol"]["sha256"],
            "metric_recipe": source.metric_recipe,
            "movement_recipe": source.movement_recipe,
            "trial_recipe": source.trial_recipe,
        },
        source,
    )


def verify_candidate_trial_outcomes(
    project_dir: Path,
    recording_id: str,
    *,
    metric_recipe: str = "tail-candidate-development-v1",
) -> VerifiedArtifactSet:
    """Authenticate trial outputs and bind them to current upstream artifacts."""
    project_dir = project_dir.resolve()
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    source_dir = project_dir / "Processed data" / recording_id
    verified = verify_completed_parquet_set(
        {
            "outcomes": source_dir / source.trial_outcomes_name,
            "coverage": source_dir / source.trial_coverage_name,
        },
        project_dir
        / "Quality checks"
        / recording_id
        / source.trial_summary_name,
        project_dir / "Metadata" / f"{recording_id}_{source.trial_marker_suffix}",
        recipe=source.trial_recipe,
        recording_id=recording_id,
    )
    _, _, _, _, current_inputs, _ = _verify_inputs(
        project_dir,
        recording_id,
        metric_recipe=metric_recipe,
    )
    if verified.summary.get("inputs") != current_inputs:
        raise ArtifactIntegrityError(
            f"{source.trial_recipe} uses stale upstream artifacts for {recording_id}."
        )
    return verified


def build_candidate_trial_outcomes(
    project_dir: Path,
    recording_id: str,
    *,
    experiment_name: str = "allDelay",
    config: TrialOutcomeConfig = TrialOutcomeConfig(),
    metric_recipe: str = "tail-candidate-development-v1",
    overwrite: bool = False,
) -> TrialOutcomeResult:
    """Publish exact candidate trial outcomes and coverage."""
    if config != TrialOutcomeConfig():
        raise ConfigurationError(
            "Candidate trial outcomes use a frozen configuration. "
            "Changed windows require a new recipe identity."
        )
    project_dir = project_dir.resolve()
    (
        recording_name,
        frame_path,
        movement_path,
        protocol_path,
        input_hashes,
        source,
    ) = _verify_inputs(
        project_dir,
        recording_id,
        metric_recipe=metric_recipe,
    )
    identity = parse_recording_identity(
        recording_name,
        experiment_name=experiment_name,
    )
    if identity["recording_id"] != recording_id:
        raise ArtifactIntegrityError(
            "Parsed recording identity differs from the authenticated source."
        )
    frames = pq.read_table(frame_path).to_pandas()
    movement = pq.read_table(movement_path).to_pandas()
    protocol = pq.read_table(protocol_path).to_pandas()
    outcomes, coverage = aggregate_trial_outcomes(
        frames,
        movement,
        protocol,
        identity=identity,
        experiment_name=experiment_name,
        config=config,
    )

    output_dir = project_dir / "Processed data" / recording_id
    outcomes_path = output_dir / source.trial_outcomes_name
    coverage_path = output_dir / source.trial_coverage_name
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.trial_summary_name
    )
    marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{source.trial_marker_suffix}"
    )
    outputs = (outcomes_path, coverage_path, summary_path, marker_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{source.trial_recipe} outputs already exist: {existing}"
        )

    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-{source.trial_recipe}-",
    ) as staging_root:
        staged_outcomes = staging_root / outcomes_path.name
        staged_coverage = staging_root / coverage_path.name
        staged_summary = staging_root / summary_path.name
        staged_marker = staging_root / marker_path.name
        pq.write_table(
            pa.Table.from_pandas(outcomes, preserve_index=False, safe=True),
            staged_outcomes,
            compression="zstd",
            write_statistics=True,
        )
        pq.write_table(
            pa.Table.from_pandas(coverage, preserve_index=False, safe=True),
            staged_coverage,
            compression="zstd",
            write_statistics=True,
        )
        artifact_hashes = {
            "outcomes": sha256_file(staged_outcomes),
            "coverage": sha256_file(staged_coverage),
        }
        write_json_atomic(
            staged_summary,
            {
                "recipe": source.trial_recipe,
                "scientific_status": "candidate_development",
                "paper_approved": False,
                "recording_id": recording_id,
                "recording_name": recording_name,
                "experiment": experiment_name,
                "identity": identity,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(config),
                "metric_ids": list(METRIC_IDS.values()),
                "row_count": len(outcomes),
                "trial_counts": {
                    alignment: int(
                        outcomes.loc[
                            outcomes["alignment"] == alignment,
                            "trial_id",
                        ].nunique()
                    )
                    for alignment in ("CS", "US")
                },
                "inputs": input_hashes,
                "artifacts": {
                    "outcomes": {
                        "path": str(outcomes_path),
                        "sha256": artifact_hashes["outcomes"],
                        "compression": "zstd",
                        "compression_lossless": True,
                    },
                    "coverage": {
                        "path": str(coverage_path),
                        "sha256": artifact_hashes["coverage"],
                        "compression": "zstd",
                        "compression_lossless": True,
                    },
                },
                "limits": [
                    "No candidate metric is selected or paper-approved.",
                    "block_5 and expected-US identities are not emitted because the current approved configuration does not define them.",
                    "Population aggregation and inference are not performed.",
                ],
            },
        )
        write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": source.trial_recipe,
                "recording_id": recording_id,
                "artifact_sha256": artifact_hashes,
                "summary_sha256": sha256_file(staged_summary),
            },
        )
        if (
            sha256_file(frame_path) != input_hashes["candidate_frames"]
            or sha256_file(movement_path) != input_hashes["movement_state"]
            or sha256_file(protocol_path) != input_hashes["protocol"]
        ):
            raise ArtifactIntegrityError(
                "Candidate trial inputs changed during outcome generation."
            )
        publish_transaction(
            (
                (staged_outcomes, outcomes_path),
                (staged_coverage, coverage_path),
                (staged_summary, summary_path),
                (staged_marker, marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )
    return TrialOutcomeResult(
        recording_id=recording_id,
        outcomes_path=outcomes_path,
        coverage_path=coverage_path,
        summary_path=summary_path,
        completion_marker_path=marker_path,
        row_count=len(outcomes),
    )
