"""Candidate movement-state calibration and bout detection."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import ndimage

from classical_conditioning.artifacts import (
    artifact_staging,
    load_and_verify_source_manifest,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.preprocessing.candidates_v1 import (
    CANDIDATE_COLUMNS,
)

METRIC_IDS = {
    CANDIDATE_COLUMNS[0]: "segment_absolute_angular_speed_sum",
    CANDIDATE_COLUMNS[1]: "all_segment_angular_rms",
    CANDIDATE_COLUMNS[2]: "whole_tail_xy_rms_speed",
    CANDIDATE_COLUMNS[3]: "whole_tail_xy_mean_speed",
    CANDIDATE_COLUMNS[4]: "curvature_change_rms",
}


@dataclass(frozen=True)
class CandidateMetricSource:
    """Frozen path pairing across metric, movement, temporal, and trial recipes."""

    metric_recipe: str
    metrics_name: str
    metric_summary_name: str
    metric_marker_suffix: str
    movement_recipe: str
    movement_artifact_name: str
    movement_summary_name: str
    movement_marker_suffix: str
    temporal_recipe: str
    temporal_artifact_name: str
    temporal_summary_name: str
    temporal_marker_suffix: str
    trial_recipe: str
    trial_outcomes_name: str
    trial_coverage_name: str
    trial_summary_name: str
    trial_marker_suffix: str
    comparison_recipe: str
    runner_recipe: str
    scientific_status: str
    requires_corrected_preprocess: bool = False


CANDIDATE_METRIC_SOURCES: dict[str, CandidateMetricSource] = {
    "tail-candidate-development-v1": CandidateMetricSource(
        metric_recipe="tail-candidate-development-v1",
        metrics_name="frame_activity_candidates-v1.parquet",
        metric_summary_name="candidate-v1_activity_summary.json",
        metric_marker_suffix="candidate-v1_complete.json",
        movement_recipe="movement-candidate-v1",
        movement_artifact_name="movement_state_candidates-v1.parquet",
        movement_summary_name="movement-candidate-v1_summary.json",
        movement_marker_suffix="movement-candidate-v1_complete.json",
        temporal_recipe="candidate-temporal-outcomes-v2",
        temporal_artifact_name="candidate_temporal_outcomes-v2.parquet",
        temporal_summary_name="candidate-v2_temporal_outcomes_summary.json",
        temporal_marker_suffix="candidate-temporal-outcomes-v2_complete.json",
        trial_recipe="candidate-trial-outcomes-v1",
        trial_outcomes_name="candidate-trial-outcomes-v1.parquet",
        trial_coverage_name="candidate-trial-outcomes-v1_coverage.parquet",
        trial_summary_name="candidate-trial-outcomes-v1_summary.json",
        trial_marker_suffix="candidate-trial-outcomes-v1_complete.json",
        comparison_recipe="candidate-metric-comparison-v1",
        runner_recipe="candidate-development-runner-v1",
        scientific_status="candidate_development",
    ),
    "tail-candidate-corrected-v1": CandidateMetricSource(
        metric_recipe="tail-candidate-corrected-v1",
        metrics_name="frame_activity_candidates-corrected-v1.parquet",
        metric_summary_name="candidate-corrected-v1_activity_summary.json",
        metric_marker_suffix="candidate-corrected-v1_complete.json",
        movement_recipe="movement-candidate-corrected-v1",
        movement_artifact_name="movement_state_candidates-corrected-v1.parquet",
        movement_summary_name="movement-candidate-corrected-v1_summary.json",
        movement_marker_suffix="movement-candidate-corrected-v1_complete.json",
        temporal_recipe="candidate-temporal-outcomes-corrected-v2",
        temporal_artifact_name="candidate_temporal_outcomes-corrected-v2.parquet",
        temporal_summary_name="candidate-corrected-v2_temporal_outcomes_summary.json",
        temporal_marker_suffix="candidate-temporal-outcomes-corrected-v2_complete.json",
        trial_recipe="candidate-trial-outcomes-corrected-v1",
        trial_outcomes_name="candidate-trial-outcomes-corrected-v1.parquet",
        trial_coverage_name="candidate-trial-outcomes-corrected-v1_coverage.parquet",
        trial_summary_name="candidate-trial-outcomes-corrected-v1_summary.json",
        trial_marker_suffix="candidate-trial-outcomes-corrected-v1_complete.json",
        comparison_recipe="candidate-metric-comparison-corrected-v1",
        runner_recipe="candidate-corrected-runner-v1",
        scientific_status="candidate_corrected",
        requires_corrected_preprocess=True,
    ),
}

MOVEMENT_RECIPE_TO_METRIC_SOURCE = {
    source.movement_recipe: source.metric_recipe
    for source in CANDIDATE_METRIC_SOURCES.values()
}
TEMPORAL_RECIPE_TO_METRIC_SOURCE = {
    source.temporal_recipe: source.metric_recipe
    for source in CANDIDATE_METRIC_SOURCES.values()
}
TRIAL_RECIPE_TO_METRIC_SOURCE = {
    source.trial_recipe: source.metric_recipe
    for source in CANDIDATE_METRIC_SOURCES.values()
}
COMPARISON_RECIPE_TO_METRIC_SOURCE = {
    source.comparison_recipe: source.metric_recipe
    for source in CANDIDATE_METRIC_SOURCES.values()
}
RUNNER_RECIPE_TO_METRIC_SOURCE = {
    source.runner_recipe: source.metric_recipe
    for source in CANDIDATE_METRIC_SOURCES.values()
}


def resolve_candidate_metric_source(
    *,
    metric_recipe: str | None = None,
    movement_recipe: str | None = None,
    temporal_recipe: str | None = None,
    trial_recipe: str | None = None,
    comparison_recipe: str | None = None,
    runner_recipe: str | None = None,
) -> CandidateMetricSource:
    """Resolve the frozen candidate-route path pairing."""
    selectors = [
        value
        for value in (
            metric_recipe,
            movement_recipe,
            temporal_recipe,
            trial_recipe,
            comparison_recipe,
            runner_recipe,
        )
        if value is not None
    ]
    if not selectors:
        return CANDIDATE_METRIC_SOURCES["tail-candidate-development-v1"]

    source: CandidateMetricSource | None = None
    if metric_recipe is not None:
        source = CANDIDATE_METRIC_SOURCES.get(metric_recipe)
        if source is None:
            raise ValueError(f"Unsupported metric recipe: {metric_recipe}")
    if comparison_recipe is not None:
        metric_key = COMPARISON_RECIPE_TO_METRIC_SOURCE.get(comparison_recipe)
        if metric_key is None:
            raise ValueError(f"Unsupported comparison recipe: {comparison_recipe}")
        resolved = CANDIDATE_METRIC_SOURCES[metric_key]
        if source is not None and source.metric_recipe != resolved.metric_recipe:
            raise ValueError(
                "Incompatible candidate-route selectors: "
                f"{metric_recipe=} {comparison_recipe=}"
            )
        source = resolved
    if runner_recipe is not None:
        metric_key = RUNNER_RECIPE_TO_METRIC_SOURCE.get(runner_recipe)
        if metric_key is None:
            raise ValueError(f"Unsupported runner recipe: {runner_recipe}")
        resolved = CANDIDATE_METRIC_SOURCES[metric_key]
        if source is not None and source.metric_recipe != resolved.metric_recipe:
            raise ValueError(
                "Incompatible candidate-route selectors: "
                f"{metric_recipe=} {runner_recipe=}"
            )
        source = resolved
    if movement_recipe is not None:
        metric_key = MOVEMENT_RECIPE_TO_METRIC_SOURCE.get(movement_recipe)
        if metric_key is None:
            raise ValueError(f"Unsupported movement recipe: {movement_recipe}")
        resolved = CANDIDATE_METRIC_SOURCES[metric_key]
        if source is not None and source.metric_recipe != resolved.metric_recipe:
            raise ValueError("Recipes do not form a frozen pairing.")
        source = resolved
    if temporal_recipe is not None:
        metric_key = TEMPORAL_RECIPE_TO_METRIC_SOURCE.get(temporal_recipe)
        if metric_key is None:
            raise ValueError(f"Unsupported temporal recipe: {temporal_recipe}")
        resolved = CANDIDATE_METRIC_SOURCES[metric_key]
        if source is not None and source.metric_recipe != resolved.metric_recipe:
            raise ValueError("Recipes do not form a frozen pairing.")
        source = resolved
    if trial_recipe is not None:
        metric_key = TRIAL_RECIPE_TO_METRIC_SOURCE.get(trial_recipe)
        if metric_key is None:
            raise ValueError(f"Unsupported trial recipe: {trial_recipe}")
        resolved = CANDIDATE_METRIC_SOURCES[metric_key]
        if source is not None and source.metric_recipe != resolved.metric_recipe:
            raise ValueError("Recipes do not form a frozen pairing.")
        source = resolved
    assert source is not None
    if metric_recipe is not None and source.metric_recipe != metric_recipe:
        raise ValueError("Recipes do not form a frozen pairing.")
    if movement_recipe is not None and source.movement_recipe != movement_recipe:
        raise ValueError("Recipes do not form a frozen pairing.")
    if temporal_recipe is not None and source.temporal_recipe != temporal_recipe:
        raise ValueError("Recipes do not form a frozen pairing.")
    if trial_recipe is not None and source.trial_recipe != trial_recipe:
        raise ValueError("Recipes do not form a frozen pairing.")
    if (
        comparison_recipe is not None
        and source.comparison_recipe != comparison_recipe
    ):
        raise ValueError("Recipes do not form a frozen pairing.")
    if runner_recipe is not None and source.runner_recipe != runner_recipe:
        raise ValueError("Recipes do not form a frozen pairing.")
    return source


@dataclass(frozen=True)
class MovementCalibrationConfig:
    smoothing_window_ms: float = 10.0
    quiet_window_ms: float = 100.0
    quiet_window_fraction: float = 0.20
    low_threshold_quantile: float = 0.99
    high_threshold_quantile: float = 0.999
    minimum_valid_tail_fraction: float = 0.80
    minimum_bout_duration_ms: float = 40 / 700 * 1_000
    maximum_interbout_gap_ms: float = 10 / 700 * 1_000
    positive_control_window_ms: float = 500.0
    control_baseline_start_ms: float = -5_000.0
    control_baseline_end_ms: float = -1_000.0


@dataclass(frozen=True)
class MovementStateResult:
    recording_id: str
    movement_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int


@dataclass(frozen=True)
class MovementSensitivityResult:
    recording_id: str
    summary_path: Path
    completion_marker_path: Path


def _odd_window_samples(window_ms: float, median_interval_ms: float) -> int:
    samples = max(1, int(round(window_ms / median_interval_ms)))
    if samples % 2 == 0:
        samples += 1
    return samples


def smooth_contiguous_median(
    values: np.ndarray,
    frame_steps: np.ndarray,
    *,
    window_samples: int,
) -> np.ndarray:
    """Centered median without smoothing across frame discontinuities."""
    values = np.asarray(values, dtype=np.float64)
    frame_steps = np.asarray(frame_steps, dtype=np.int64)
    if values.shape != frame_steps.shape:
        raise ValueError("Values and frame steps must have identical shapes.")
    if window_samples < 1 or window_samples % 2 == 0:
        raise ValueError("Median window must be a positive odd number.")
    boundaries = (frame_steps != 1) & (frame_steps != 0)
    segment_id = np.cumsum(boundaries)
    frame = pd.DataFrame(
        {"value": values, "segment": segment_id}
    )
    smoothed = frame.groupby("segment", sort=False)["value"].transform(
        lambda group: group.rolling(
            window=window_samples,
            center=True,
            min_periods=window_samples,
        ).median()
    )
    return smoothed.to_numpy(dtype=np.float64)


def calibrate_quiet_window_thresholds(
    values: np.ndarray,
    elapsed_time_ms: np.ndarray,
    valid: np.ndarray,
    *,
    config: MovementCalibrationConfig,
) -> dict[str, float | int]:
    """Estimate candidate thresholds from the quietest fixed-duration windows."""
    values = np.asarray(values, dtype=np.float64)
    elapsed_time_ms = np.asarray(elapsed_time_ms, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not np.any(valid):
        raise ValueError("No valid samples are available for threshold calibration.")
    window_id = np.floor(
        (elapsed_time_ms - elapsed_time_ms[0]) / config.quiet_window_ms
    ).astype(np.int64)
    calibration = pd.DataFrame(
        {
            "window_id": window_id[valid],
            "value": values[valid],
        }
    )
    window_medians = calibration.groupby("window_id", sort=False)["value"].median()
    quiet_window_count = max(
        1,
        int(np.ceil(len(window_medians) * config.quiet_window_fraction)),
    )
    ranked_windows = (
        window_medians.rename("median")
        .reset_index()
        .sort_values(["median", "window_id"], kind="stable")
    )
    quiet_selection = ranked_windows.iloc[:quiet_window_count]
    quiet_cutoff = float(quiet_selection["median"].max())
    quiet_ids = set(
        quiet_selection["window_id"].to_numpy(dtype=np.int64)
    )
    quiet = valid & np.fromiter(
        (identifier in quiet_ids for identifier in window_id),
        dtype=bool,
        count=len(window_id),
    )
    quiet_values = values[quiet]
    if quiet_values.size < 100:
        raise ValueError("Too few putative quiet samples for calibration.")
    low = float(np.quantile(quiet_values, config.low_threshold_quantile))
    high = float(np.quantile(quiet_values, config.high_threshold_quantile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(
            f"Invalid candidate thresholds: low={low}, high={high}."
        )
    return {
        "quiet_window_count": quiet_window_count,
        "total_window_count": int(len(window_medians)),
        "quiet_sample_count": int(quiet_values.size),
        "quiet_window_median_cutoff": quiet_cutoff,
        "low_threshold": low,
        "high_threshold": high,
    }


def detect_hysteresis_bouts(
    values: np.ndarray,
    elapsed_time_ms: np.ndarray,
    delta_time_ms: np.ndarray,
    frame_steps: np.ndarray,
    valid: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
    minimum_bout_duration_ms: float,
    maximum_interbout_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect bouts from high-threshold seeds and low-threshold support."""
    values = np.asarray(values, dtype=np.float64)
    elapsed_time_ms = np.asarray(elapsed_time_ms, dtype=np.float64)
    delta_time_ms = np.asarray(delta_time_ms, dtype=np.float64)
    frame_steps = np.asarray(frame_steps, dtype=np.int64)
    if not (
        values.shape
        == elapsed_time_ms.shape
        == delta_time_ms.shape
        == frame_steps.shape
        == np.asarray(valid).shape
    ):
        raise ValueError("Detector arrays must have identical shapes.")
    valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(values)
        & np.isfinite(delta_time_ms)
        & (delta_time_ms > 0)
    )
    if not (low_threshold < high_threshold):
        raise ValueError("Low threshold must be below high threshold.")
    barriers = ~valid | ((frame_steps != 1) & (frame_steps != 0))
    weak = valid & (values > low_threshold)
    strong = valid & (values > high_threshold)
    weak_labels, _ = ndimage.label(weak)
    seeded_labels = np.unique(weak_labels[strong])
    seeded_labels = seeded_labels[seeded_labels != 0]
    active = np.isin(weak_labels, seeded_labels)

    inactive_labels, inactive_count = ndimage.label(~active)
    if inactive_count:
        indices = np.arange(len(active), dtype=np.int64)
        first = np.full(inactive_count + 1, len(active), dtype=np.int64)
        last = np.full(inactive_count + 1, -1, dtype=np.int64)
        np.minimum.at(first, inactive_labels, indices)
        np.maximum.at(last, inactive_labels, indices)
        safe_delta = np.where(np.isfinite(delta_time_ms), delta_time_ms, 0.0)
        gap_duration = np.bincount(
            inactive_labels,
            weights=safe_delta,
            minlength=inactive_count + 1,
        )
        barrier_count = np.bincount(
            inactive_labels,
            weights=barriers.astype(np.int8),
            minlength=inactive_count + 1,
        )
        fill = (
            (np.arange(inactive_count + 1) != 0)
            & (first > 0)
            & (last < len(active) - 1)
            & (barrier_count == 0)
            & (
                (gap_duration <= maximum_interbout_gap_ms)
                | np.isclose(
                    gap_duration,
                    maximum_interbout_gap_ms,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        )
        active |= fill[inactive_labels]

    active_labels, active_count = ndimage.label(active)
    if active_count:
        safe_delta = np.where(np.isfinite(delta_time_ms), delta_time_ms, 0.0)
        bout_duration = np.bincount(
            active_labels,
            weights=safe_delta,
            minlength=active_count + 1,
        )
        keep = (bout_duration >= minimum_bout_duration_ms) | np.isclose(
            bout_duration,
            minimum_bout_duration_ms,
            rtol=1e-12,
            atol=1e-12,
        )
        keep[0] = False
        active = keep[active_labels]

    bout_ids, _ = ndimage.label(active)
    return active, bout_ids.astype(np.int32)


def _positive_control(
    frame_ids: np.ndarray,
    absolute_time: np.ndarray,
    moving: np.ndarray,
    valid: np.ndarray,
    protocol: pd.DataFrame,
    config: MovementCalibrationConfig,
) -> dict[str, float | int | None]:
    us_events = protocol.loc[protocol["Type"].astype(str) == "Reinforcer", "Beg"]
    post_values: list[np.ndarray] = []
    baseline_values: list[np.ndarray] = []
    post_total = 0
    baseline_total = 0
    post_evaluated_events = 0
    baseline_evaluated_events = 0
    acquisition_start = int(absolute_time[0])
    acquisition_end = int(absolute_time[-1])
    for start in us_events.to_numpy(dtype=np.int64):
        post_window_end = start + config.positive_control_window_ms
        baseline_window_start = start + config.control_baseline_start_ms
        baseline_window_end = start + config.control_baseline_end_ms
        if acquisition_start <= start and post_window_end <= acquisition_end:
            post_evaluated_events += 1
            post_start = np.searchsorted(absolute_time, start, side="left")
            post_end = np.searchsorted(
                absolute_time,
                post_window_end,
                side="left",
            )
            post_total += int(
                frame_ids[post_end - 1] - frame_ids[post_start] + 1
            )
            post_valid = valid[post_start:post_end]
            post_values.append(moving[post_start:post_end][post_valid])
        if (
            acquisition_start <= baseline_window_start
            and baseline_window_end <= acquisition_end
        ):
            baseline_evaluated_events += 1
            baseline_start = np.searchsorted(
                absolute_time,
                baseline_window_start,
                side="left",
            )
            baseline_end = np.searchsorted(
                absolute_time,
                baseline_window_end,
                side="left",
            )
            baseline_total += int(
                frame_ids[baseline_end - 1]
                - frame_ids[baseline_start]
                + 1
            )
            baseline_valid = valid[baseline_start:baseline_end]
            baseline_values.append(
                moving[baseline_start:baseline_end][baseline_valid]
            )
    post = np.concatenate(post_values) if post_values else np.array([], dtype=bool)
    baseline = (
        np.concatenate(baseline_values)
        if baseline_values
        else np.array([], dtype=bool)
    )
    post_fraction = float(np.mean(post)) if post.size else None
    baseline_fraction = float(np.mean(baseline)) if baseline.size else None
    return {
        "event_count": int(len(us_events)),
        "post_us_evaluated_event_count": post_evaluated_events,
        "pre_us_baseline_evaluated_event_count": baseline_evaluated_events,
        "post_us_moving_fraction": post_fraction,
        "pre_us_baseline_moving_fraction": baseline_fraction,
        "post_us_valid_count": int(post.size),
        "pre_us_baseline_valid_count": int(baseline.size),
        "post_us_valid_fraction": (
            float(post.size / post_total) if post_total else None
        ),
        "pre_us_baseline_valid_fraction": (
            float(baseline.size / baseline_total) if baseline_total else None
        ),
        "difference": (
            post_fraction - baseline_fraction
            if post_fraction is not None and baseline_fraction is not None
            else None
        ),
    }


def build_candidate_movement_state(
    project_dir: Path,
    recording_id: str,
    *,
    config: MovementCalibrationConfig | None = None,
    metric_recipe: str = "tail-candidate-development-v1",
    overwrite: bool = False,
) -> MovementStateResult:
    """Calibrate and apply exploratory movement detectors to all candidates."""
    config = config or MovementCalibrationConfig()
    if config != MovementCalibrationConfig():
        raise ValueError(
            "Movement calibration uses a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    metric_path = source_dir / source.metrics_name
    protocol_path = source_dir / "stimulus_events.parquet"
    candidate_marker_path = (
        project_dir / "Metadata" / f"{recording_id}_{source.metric_marker_suffix}"
    )
    candidate_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.metric_summary_name
    )
    marker = json.loads(candidate_marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "complete"
        or marker.get("recipe") != source.metric_recipe
        or marker.get("recording_id") != recording_id
    ):
        raise ValueError(
            f"Candidate metric marker is invalid for {source.metric_recipe}."
        )
    if sha256_file(metric_path) != marker.get("metrics_sha256"):
        raise ValueError("Candidate metric artifact hash differs from its marker.")
    if sha256_file(candidate_summary_path) != marker.get("summary_sha256"):
        raise ValueError("Candidate metric summary hash differs from its marker.")
    candidate_summary = json.loads(
        candidate_summary_path.read_text(encoding="utf-8")
    )
    _, intake_artifacts, intake_state = load_and_verify_source_manifest(
        project_dir,
        recording_id,
    )
    for kind, record in intake_artifacts.items():
        summary_inputs = candidate_summary["input_artifacts"]
        if kind not in summary_inputs:
            raise ValueError(
                f"Candidate metrics summary is missing intake {kind} provenance."
            )
        if summary_inputs[kind]["sha256"] != record["sha256"]:
            raise ValueError(
                f"Candidate metrics were built from a different {kind} artifact."
            )

    output_path = source_dir / source.movement_artifact_name
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.movement_summary_name
    )
    completion_path = (
        project_dir
        / "Metadata"
        / f"{recording_id}_{source.movement_marker_suffix}"
    )
    existing = [
        path for path in (output_path, summary_path, completion_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Movement candidate outputs already exist: {existing}")

    columns = [
        "FrameID",
        "ElapsedTime",
        "AbsoluteTime",
        "FrameStep",
        "DeltaTimeMs",
        "valid_derivative",
        "xy_valid_tail_fraction",
        "angular_valid_tail_fraction",
        "curvature_valid_tail_fraction",
        *CANDIDATE_COLUMNS,
    ]
    frames = pq.read_table(metric_path, columns=columns).to_pandas()
    frame_ids = frames["FrameID"].to_numpy(dtype=np.int64)
    elapsed = frames["ElapsedTime"].to_numpy(dtype=np.float64)
    absolute = frames["AbsoluteTime"].to_numpy(dtype=np.int64)
    frame_steps = frames["FrameStep"].to_numpy(dtype=np.int64)
    delta_time = frames["DeltaTimeMs"].to_numpy(dtype=np.float64)
    if (
        not np.all(np.isfinite(elapsed))
        or np.any(np.diff(elapsed) <= 0)
        or np.any(np.diff(absolute) < 0)
    ):
        raise ValueError(
            "Movement calibration requires finite increasing elapsed time and "
            "nondecreasing absolute time."
        )
    adjacent_interval = frames.loc[
        frames["FrameStep"].eq(1) & frames["DeltaTimeMs"].gt(0),
        "DeltaTimeMs",
    ].to_numpy(dtype=float)
    median_interval_ms = float(np.median(adjacent_interval))
    smoothing_samples = _odd_window_samples(
        config.smoothing_window_ms,
        median_interval_ms,
    )

    movement = pd.DataFrame(
        {
            "FrameID": frame_ids,
            "ElapsedTime": elapsed,
            "AbsoluteTime": absolute,
            "FrameStep": frame_steps,
        }
    )
    protocol = pq.read_table(protocol_path).to_pandas()
    calibration_results: dict[str, dict[str, Any]] = {}
    coverage_columns = {
        CANDIDATE_COLUMNS[0]: "angular_valid_tail_fraction",
        CANDIDATE_COLUMNS[1]: "angular_valid_tail_fraction",
        CANDIDATE_COLUMNS[2]: "xy_valid_tail_fraction",
        CANDIDATE_COLUMNS[3]: "xy_valid_tail_fraction",
        CANDIDATE_COLUMNS[4]: "curvature_valid_tail_fraction",
    }
    base_valid = frames["valid_derivative"].to_numpy(dtype=bool)

    for column, metric_id in METRIC_IDS.items():
        values = frames[column].to_numpy(dtype=np.float64)
        smoothed = smooth_contiguous_median(
            values,
            frame_steps,
            window_samples=smoothing_samples,
        )
        coverage = frames[coverage_columns[column]].to_numpy(dtype=float)
        valid = (
            base_valid
            & np.isfinite(smoothed)
            & (coverage >= config.minimum_valid_tail_fraction)
        )
        thresholds = calibrate_quiet_window_thresholds(
            smoothed,
            elapsed,
            valid,
            config=config,
        )
        moving, bout_ids = detect_hysteresis_bouts(
            smoothed,
            elapsed,
            delta_time,
            frame_steps,
            valid,
            low_threshold=float(thresholds["low_threshold"]),
            high_threshold=float(thresholds["high_threshold"]),
            minimum_bout_duration_ms=config.minimum_bout_duration_ms,
            maximum_interbout_gap_ms=config.maximum_interbout_gap_ms,
        )
        movement[f"{metric_id}__moving"] = moving
        movement[f"{metric_id}__valid"] = valid
        movement[f"{metric_id}__bout_id"] = bout_ids
        bout_count = int(bout_ids.max())
        calibration_results[metric_id] = {
            **thresholds,
            "source_column": column,
            "coverage_column": coverage_columns[column],
            "movement_fraction": float(np.mean(moving[valid])),
            "valid_fraction": float(np.mean(valid)),
            "bout_count": bout_count,
            "positive_control": _positive_control(
                frame_ids,
                absolute,
                moving,
                valid,
                protocol,
                config,
            ),
        }

    metric_state = (metric_path.stat().st_size, metric_path.stat().st_mtime_ns)
    if (
        sha256_file(metric_path) != marker["metrics_sha256"]
        or metric_state
        != (metric_path.stat().st_size, metric_path.stat().st_mtime_ns)
    ):
        raise RuntimeError("Candidate metric artifact changed during detector build.")
    protocol_stat = protocol_path.stat()
    if intake_state["protocol"] != (
        protocol_stat.st_size,
        protocol_stat.st_mtime_ns,
    ):
        raise RuntimeError("Protocol artifact changed during detector build.")

    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-{source.movement_recipe}-",
    ) as staging_root:
        staged_output = staging_root / output_path.name
        staged_summary = staging_root / summary_path.name
        staged_completion = staging_root / completion_path.name
        table = pa.Table.from_pandas(movement, preserve_index=False)
        pq.write_table(
            table,
            staged_output,
            compression="zstd",
            row_group_size=250_000,
            write_statistics=True,
        )
        output_hash = sha256_file(staged_output)
        summary = {
            "recipe": source.movement_recipe,
            "scientific_status": "candidate_development",
            "paper_approved": False,
            "recording_id": recording_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "metric_source_recipe": source.metric_recipe,
            "resolved_smoothing_window_samples": smoothing_samples,
            "median_adjacent_frame_interval_ms": median_interval_ms,
            "calibration": calibration_results,
            "known_limitations": [
                "Thresholds are calibrated from putative quiet windows, not video labels.",
                "This single-fish calibration is exploratory and not transferable by default.",
                "The detector has not been selected for paper inference.",
                "Threshold sensitivity and blinded/manual validation remain required.",
            ],
            "artifact": {
                "path": str(output_path.resolve()),
                "sha256": output_hash,
                "rows": len(movement),
                "compression": "zstd",
                "compression_lossless": True,
            },
            "inputs": {
                "candidate_metrics": {
                    "path": str(metric_path),
                    "sha256": marker["metrics_sha256"],
                    "recipe": source.metric_recipe,
                },
                "protocol": intake_artifacts["protocol"],
            },
        }
        write_json_atomic(staged_summary, summary)
        summary_hash = sha256_file(staged_summary)
        write_json_atomic(
            staged_completion,
            {
                "status": "complete",
                "recipe": source.movement_recipe,
                "recording_id": recording_id,
                "movement_sha256": output_hash,
                "summary_sha256": summary_hash,
            },
        )
        publish_transaction(
            (
                (staged_output, output_path),
                (staged_summary, summary_path),
                (staged_completion, completion_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return MovementStateResult(
        recording_id=recording_id,
        movement_path=output_path,
        summary_path=summary_path,
        completion_marker_path=completion_path,
        row_count=len(movement),
    )

def evaluate_smoothing_sensitivity(
    frames: pd.DataFrame,
    protocol: pd.DataFrame,
    *,
    smoothing_windows_ms: tuple[float, ...] = (0.0, 10.0, 20.0),
    base_config: MovementCalibrationConfig | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate compact detector summaries without saving duplicate frame states."""
    base_config = base_config or MovementCalibrationConfig()
    frame_ids = frames["FrameID"].to_numpy(dtype=np.int64)
    elapsed = frames["ElapsedTime"].to_numpy(dtype=np.float64)
    absolute = frames["AbsoluteTime"].to_numpy(dtype=np.int64)
    frame_steps = frames["FrameStep"].to_numpy(dtype=np.int64)
    delta_time = frames["DeltaTimeMs"].to_numpy(dtype=np.float64)
    if (
        not np.all(np.isfinite(elapsed))
        or np.any(np.diff(elapsed) <= 0)
        or np.any(np.diff(absolute) < 0)
    ):
        raise ValueError(
            "Sensitivity analysis requires finite increasing elapsed time and "
            "nondecreasing absolute time."
        )
    adjacent = delta_time[(frame_steps == 1) & np.isfinite(delta_time) & (delta_time > 0)]
    if adjacent.size == 0:
        raise ValueError("No adjacent frame intervals are available.")
    median_interval_ms = float(np.median(adjacent))
    base_valid = frames["valid_derivative"].to_numpy(dtype=bool)
    coverage_columns = {
        CANDIDATE_COLUMNS[0]: "angular_valid_tail_fraction",
        CANDIDATE_COLUMNS[1]: "angular_valid_tail_fraction",
        CANDIDATE_COLUMNS[2]: "xy_valid_tail_fraction",
        CANDIDATE_COLUMNS[3]: "xy_valid_tail_fraction",
        CANDIDATE_COLUMNS[4]: "curvature_valid_tail_fraction",
    }
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for smoothing_ms in smoothing_windows_ms:
        if smoothing_ms < 0:
            raise ValueError("Smoothing windows must be non-negative.")
        smoothing_samples = (
            1
            if smoothing_ms == 0
            else _odd_window_samples(smoothing_ms, median_interval_ms)
        )
        config = replace(base_config, smoothing_window_ms=smoothing_ms)
        window_result: dict[str, dict[str, Any]] = {}
        for column, metric_id in METRIC_IDS.items():
            values = frames[column].to_numpy(dtype=np.float64)
            smoothed = smooth_contiguous_median(
                values,
                frame_steps,
                window_samples=smoothing_samples,
            )
            coverage = frames[coverage_columns[column]].to_numpy(dtype=float)
            valid = (
                base_valid
                & np.isfinite(smoothed)
                & (coverage >= config.minimum_valid_tail_fraction)
            )
            thresholds = calibrate_quiet_window_thresholds(
                smoothed,
                elapsed,
                valid,
                config=config,
            )
            moving, bout_ids = detect_hysteresis_bouts(
                smoothed,
                elapsed,
                delta_time,
                frame_steps,
                valid,
                low_threshold=float(thresholds["low_threshold"]),
                high_threshold=float(thresholds["high_threshold"]),
                minimum_bout_duration_ms=config.minimum_bout_duration_ms,
                maximum_interbout_gap_ms=config.maximum_interbout_gap_ms,
            )
            window_result[metric_id] = {
                **thresholds,
                "smoothing_window_samples": smoothing_samples,
                "valid_fraction": float(np.mean(valid)),
                "movement_fraction": float(np.mean(moving[valid])),
                "bout_count": int(bout_ids.max()),
                "positive_control": _positive_control(
                    frame_ids,
                    absolute,
                    moving,
                    valid,
                    protocol,
                    config,
                ),
            }
        results[f"{smoothing_ms:g}ms"] = window_result
    return results


def build_movement_sensitivity_report(
    project_dir: Path,
    recording_id: str,
    *,
    smoothing_windows_ms: tuple[float, ...] = (0.0, 10.0, 20.0),
    overwrite: bool = False,
) -> MovementSensitivityResult:
    """Write a compact smoothing sensitivity report for candidate detectors."""
    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    metric_path = source_dir / "frame_activity_candidates-v1.parquet"
    protocol_path = source_dir / "stimulus_events.parquet"
    marker_path = project_dir / "Metadata" / f"{recording_id}_candidate-v1_complete.json"
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "movement-smoothing-sensitivity-v1.json"
    )
    completion_path = (
        project_dir
        / "Metadata"
        / f"{recording_id}_movement-smoothing-sensitivity-v1_complete.json"
    )
    if any(path.exists() for path in (summary_path, completion_path)) and not overwrite:
        raise FileExistsError("Movement smoothing sensitivity output already exists.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "complete"
        or marker.get("recipe") != "tail-candidate-development-v1"
        or marker.get("recording_id") != recording_id
        or sha256_file(metric_path) != marker.get("metrics_sha256")
    ):
        raise ValueError("Candidate metric artifact or marker is invalid.")
    candidate_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "candidate-v1_activity_summary.json"
    )
    if (
        not candidate_summary_path.is_file()
        or sha256_file(candidate_summary_path) != marker.get("summary_sha256")
    ):
        raise ValueError("Candidate metric summary is invalid.")
    candidate_summary = json.loads(
        candidate_summary_path.read_text(encoding="utf-8")
    )
    _, intake_artifacts, intake_state = load_and_verify_source_manifest(
        project_dir, recording_id
    )
    for kind, record in intake_artifacts.items():
        if candidate_summary["input_artifacts"][kind]["sha256"] != record["sha256"]:
            raise ValueError(
                f"Candidate metrics were built from a different {kind} artifact."
            )
    if sha256_file(protocol_path) != intake_artifacts["protocol"]["sha256"]:
        raise ValueError("Protocol artifact differs from the intake manifest.")

    columns = [
        "FrameID",
        "ElapsedTime",
        "AbsoluteTime",
        "FrameStep",
        "DeltaTimeMs",
        "valid_derivative",
        "xy_valid_tail_fraction",
        "angular_valid_tail_fraction",
        "curvature_valid_tail_fraction",
        *CANDIDATE_COLUMNS,
    ]
    frames = pq.read_table(metric_path, columns=columns).to_pandas()
    protocol = pq.read_table(protocol_path).to_pandas()
    results = evaluate_smoothing_sensitivity(
        frames,
        protocol,
        smoothing_windows_ms=smoothing_windows_ms,
    )
    if sha256_file(metric_path) != marker["metrics_sha256"]:
        raise RuntimeError("Candidate metric artifact changed during sensitivity analysis.")
    protocol_stat = protocol_path.stat()
    if (
        intake_state["protocol"]
        != (protocol_stat.st_size, protocol_stat.st_mtime_ns)
        or sha256_file(protocol_path) != intake_artifacts["protocol"]["sha256"]
    ):
        raise RuntimeError("Protocol artifact changed during sensitivity analysis.")

    payload = {
        "recipe": "movement-smoothing-sensitivity-v1",
        "scientific_status": "candidate_development",
        "recording_id": recording_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoothing_windows_ms": list(smoothing_windows_ms),
        "results": results,
        "interpretation_limits": [
            "Thresholds are recalibrated independently for each smoothing variant.",
            "This report measures sensitivity, not metric accuracy.",
            "One fish cannot determine the approved smoothing or detector.",
        ],
        "inputs": {
            "candidate_metrics": {
                "path": str(metric_path),
                "sha256": marker["metrics_sha256"],
            },
            "protocol": intake_artifacts["protocol"],
        },
    }
    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-movement-sensitivity-v1-",
    ) as staging_root:
        staged_summary = staging_root / summary_path.name
        staged_completion = staging_root / completion_path.name
        write_json_atomic(staged_summary, payload)
        summary_hash = sha256_file(staged_summary)
        write_json_atomic(
            staged_completion,
            {
                "status": "complete",
                "recipe": "movement-smoothing-sensitivity-v1",
                "recording_id": recording_id,
                "summary_sha256": summary_hash,
            },
        )
        publish_transaction(
            (
                (staged_summary, summary_path),
                (staged_completion, completion_path),
            ),
            staging_root,
            overwrite=overwrite,
        )
    return MovementSensitivityResult(
        recording_id=recording_id,
        summary_path=summary_path,
        completion_marker_path=completion_path,
    )
