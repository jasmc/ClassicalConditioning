"""Shared, metric-independent bout detection for the candidate route.

One detector runs per recording, on the distal cumulative-angle speed, using
the historical envelope rule and thresholds. Every candidate metric inherits
that segmentation, so bout-derived outcomes describe the animal's behavior
rather than the metric used to measure it.
"""

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
    CANDIDATE_COLUMNS[5]: "legacy_distal_angular_speed",
}

# Bout detection is a property of the animal's behavior, not of the metric used
# to describe it. One detector runs on the distal cumulative-angle speed, which
# is the signal the historical pipeline used, and every metric shares its
# segmentation.
DETECTOR_SOURCE_COLUMN = CANDIDATE_COLUMNS[5]
DETECTOR_COVERAGE_COLUMN = "angular_valid_tail_fraction"
DETECTOR_COLUMNS = ("valid", "moving", "bout_id")

_DEGREES_TO_RADIANS = np.pi / 180.0


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
        movement_recipe="movement-candidate-v2",
        movement_artifact_name="movement_state_candidates-v2.parquet",
        movement_summary_name="movement-candidate-v2_summary.json",
        movement_marker_suffix="movement-candidate-v2_complete.json",
        temporal_recipe="candidate-temporal-outcomes-v3",
        temporal_artifact_name="candidate_temporal_outcomes-v3.parquet",
        temporal_summary_name="candidate-v3_temporal_outcomes_summary.json",
        temporal_marker_suffix="candidate-temporal-outcomes-v3_complete.json",
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
        movement_recipe="movement-candidate-corrected-v2",
        movement_artifact_name="movement_state_candidates-corrected-v2.parquet",
        movement_summary_name="movement-candidate-corrected-v2_summary.json",
        movement_marker_suffix="movement-candidate-corrected-v2_complete.json",
        temporal_recipe="candidate-temporal-outcomes-corrected-v3",
        temporal_artifact_name="candidate_temporal_outcomes-corrected-v3.parquet",
        temporal_summary_name="candidate-corrected-v3_temporal_outcomes_summary.json",
        temporal_marker_suffix="candidate-temporal-outcomes-corrected-v3_complete.json",
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
    """Frozen shared-detector parameters ported from the historical pipeline.

    Historical values were expressed in frames at the interpolated 700 FPS rate
    (max window 20, min window 400, minimum duration 40, interbout gap 10) and
    in deg/ms. They are stored here in milliseconds and degrees so the detector
    can run on measured timestamps rather than assuming a fixed frame rate.
    """

    smoothing_window_ms: float = 10.0
    envelope_max_window_ms: float = 20 / 700 * 1_000
    envelope_min_window_ms: float = 400 / 700 * 1_000
    envelope_threshold_deg_per_ms: float = 4.0
    bout_amplitude_threshold_deg_per_ms: float = 1.0
    minimum_valid_tail_fraction: float = 0.80
    minimum_bout_duration_ms: float = 40 / 700 * 1_000
    maximum_interbout_gap_ms: float = 10 / 700 * 1_000
    positive_control_window_ms: float = 500.0
    control_baseline_start_ms: float = -5_000.0
    control_baseline_end_ms: float = -1_000.0

    @property
    def envelope_threshold_rad_per_ms(self) -> float:
        return self.envelope_threshold_deg_per_ms * _DEGREES_TO_RADIANS

    @property
    def bout_amplitude_threshold_rad_per_ms(self) -> float:
        return self.bout_amplitude_threshold_deg_per_ms * _DEGREES_TO_RADIANS


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


def rolling_extreme_envelope(
    values: np.ndarray,
    frame_steps: np.ndarray,
    *,
    max_window_samples: int,
    min_window_samples: int,
) -> np.ndarray:
    """Historical bout-detection envelope: rolling max minus rolling min.

    Both windows are centered and require full occupancy, matching the legacy
    ``rolling(...).max() - rolling(...).min()`` followed by ``dropna``. Windows
    never span a frame discontinuity, so a tracking gap yields NaN instead of
    an envelope computed across missing time.
    """
    values = np.asarray(values, dtype=np.float64)
    frame_steps = np.asarray(frame_steps, dtype=np.int64)
    if values.shape != frame_steps.shape:
        raise ValueError("Values and frame steps must have identical shapes.")
    if max_window_samples < 1 or min_window_samples < 1:
        raise ValueError("Envelope windows must be positive.")
    boundaries = (frame_steps != 1) & (frame_steps != 0)
    segment_id = np.cumsum(boundaries)
    frame = pd.DataFrame({"value": values, "segment": segment_id})
    grouped = frame.groupby("segment", sort=False)["value"]
    rolling_max = grouped.transform(
        lambda group: group.rolling(
            window=max_window_samples,
            center=True,
            min_periods=max_window_samples,
        ).max()
    )
    rolling_min = grouped.transform(
        lambda group: group.rolling(
            window=min_window_samples,
            center=True,
            min_periods=min_window_samples,
        ).min()
    )
    return (rolling_max - rolling_min).to_numpy(dtype=np.float64)


def _merge_short_gaps(
    active: np.ndarray,
    delta_time_ms: np.ndarray,
    barriers: np.ndarray,
    maximum_interbout_gap_ms: float,
) -> np.ndarray:
    inactive_labels, inactive_count = ndimage.label(~active)
    if not inactive_count:
        return active
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
    return active | fill[inactive_labels]


def _drop_short_bouts(
    active: np.ndarray,
    delta_time_ms: np.ndarray,
    minimum_bout_duration_ms: float,
) -> np.ndarray:
    labels, count = ndimage.label(active)
    if not count:
        return active
    safe_delta = np.where(np.isfinite(delta_time_ms), delta_time_ms, 0.0)
    duration = np.bincount(labels, weights=safe_delta, minlength=count + 1)
    keep = (duration >= minimum_bout_duration_ms) | np.isclose(
        duration,
        minimum_bout_duration_ms,
        rtol=1e-12,
        atol=1e-12,
    )
    keep[0] = False
    return keep[labels]


def _drop_weak_bouts(
    active: np.ndarray,
    peak_values: np.ndarray,
    amplitude_threshold: float,
) -> np.ndarray:
    labels, count = ndimage.label(active)
    if not count:
        return active
    safe_peak = np.where(np.isfinite(peak_values), peak_values, -np.inf)
    peak = np.full(count + 1, -np.inf, dtype=np.float64)
    np.maximum.at(peak, labels, safe_peak)
    keep = peak >= amplitude_threshold
    keep[0] = False
    return keep[labels]


def detect_legacy_envelope_bouts(
    envelope: np.ndarray,
    peak_values: np.ndarray,
    delta_time_ms: np.ndarray,
    frame_steps: np.ndarray,
    valid: np.ndarray,
    *,
    envelope_threshold: float,
    amplitude_threshold: float,
    minimum_bout_duration_ms: float,
    maximum_interbout_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect bouts with the historical four-step envelope rule.

    1. Threshold the max-minus-min envelope.
    2. Merge bouts separated by a gap shorter than the interbout minimum.
    3. Drop bouts shorter than the minimum duration.
    4. Drop bouts whose peak instantaneous angular speed is too weak.

    Steps 2 and 3 are measured in milliseconds of elapsed time rather than in
    frame counts, and gaps spanning a tracking discontinuity are never merged.
    """
    envelope = np.asarray(envelope, dtype=np.float64)
    peak_values = np.asarray(peak_values, dtype=np.float64)
    delta_time_ms = np.asarray(delta_time_ms, dtype=np.float64)
    frame_steps = np.asarray(frame_steps, dtype=np.int64)
    if not (
        envelope.shape
        == peak_values.shape
        == delta_time_ms.shape
        == frame_steps.shape
        == np.asarray(valid).shape
    ):
        raise ValueError("Detector arrays must have identical shapes.")
    if not np.isfinite(envelope_threshold) or envelope_threshold <= 0:
        raise ValueError("Envelope threshold must be positive and finite.")
    if not np.isfinite(amplitude_threshold) or amplitude_threshold <= 0:
        raise ValueError("Amplitude threshold must be positive and finite.")
    valid = (
        np.asarray(valid, dtype=bool)
        & np.isfinite(envelope)
        & np.isfinite(delta_time_ms)
        & (delta_time_ms > 0)
    )
    barriers = ~valid | ((frame_steps != 1) & (frame_steps != 0))

    active = valid & (envelope >= envelope_threshold)
    active = _merge_short_gaps(
        active,
        delta_time_ms,
        barriers,
        maximum_interbout_gap_ms,
    )
    active = _drop_short_bouts(active, delta_time_ms, minimum_bout_duration_ms)
    active = _drop_weak_bouts(active, peak_values, amplitude_threshold)

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
    envelope_max_samples = _odd_window_samples(
        config.envelope_max_window_ms,
        median_interval_ms,
    )
    envelope_min_samples = _odd_window_samples(
        config.envelope_min_window_ms,
        median_interval_ms,
    )
    base_valid = frames["valid_derivative"].to_numpy(dtype=bool)

    detector_values = frames[DETECTOR_SOURCE_COLUMN].to_numpy(dtype=np.float64)
    smoothed = smooth_contiguous_median(
        detector_values,
        frame_steps,
        window_samples=smoothing_samples,
    )
    envelope = rolling_extreme_envelope(
        smoothed,
        frame_steps,
        max_window_samples=envelope_max_samples,
        min_window_samples=envelope_min_samples,
    )
    coverage = frames[DETECTOR_COVERAGE_COLUMN].to_numpy(dtype=float)
    valid = (
        base_valid
        & np.isfinite(envelope)
        & (coverage >= config.minimum_valid_tail_fraction)
    )
    moving, bout_ids = detect_legacy_envelope_bouts(
        envelope,
        detector_values,
        delta_time,
        frame_steps,
        valid,
        envelope_threshold=config.envelope_threshold_rad_per_ms,
        amplitude_threshold=config.bout_amplitude_threshold_rad_per_ms,
        minimum_bout_duration_ms=config.minimum_bout_duration_ms,
        maximum_interbout_gap_ms=config.maximum_interbout_gap_ms,
    )
    movement["valid"] = valid
    movement["moving"] = moving
    movement["bout_id"] = bout_ids
    detector_report: dict[str, Any] = {
        "source_column": DETECTOR_SOURCE_COLUMN,
        "coverage_column": DETECTOR_COVERAGE_COLUMN,
        "shared_across_metrics": True,
        "applies_to_metrics": sorted(METRIC_IDS.values()),
        "envelope_max_window_samples": envelope_max_samples,
        "envelope_min_window_samples": envelope_min_samples,
        "envelope_threshold_rad_per_ms": config.envelope_threshold_rad_per_ms,
        "bout_amplitude_threshold_rad_per_ms": (
            config.bout_amplitude_threshold_rad_per_ms
        ),
        "movement_fraction": (
            float(np.mean(moving[valid])) if bool(np.any(valid)) else None
        ),
        "valid_fraction": float(np.mean(valid)),
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
            "detector": detector_report,
            "known_limitations": [
                "One shared detector segments bouts for every metric; bout "
                "outcomes are metric-independent by construction.",
                "Thresholds are the historical constants, not values validated "
                "against video labels on this data.",
                "Historical windows and durations were frame counts at an "
                "assumed 700 FPS; here they are applied as milliseconds of "
                "measured time.",
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
) -> dict[str, dict[str, Any]]:
    """Evaluate compact detector summaries without saving duplicate frame states.

    The detector is shared across metrics, so sensitivity is reported once per
    smoothing window rather than once per metric.
    """
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
    detector_values = frames[DETECTOR_SOURCE_COLUMN].to_numpy(dtype=np.float64)
    coverage = frames[DETECTOR_COVERAGE_COLUMN].to_numpy(dtype=float)
    results: dict[str, dict[str, Any]] = {}
    for smoothing_ms in smoothing_windows_ms:
        if smoothing_ms < 0:
            raise ValueError("Smoothing windows must be non-negative.")
        smoothing_samples = (
            1
            if smoothing_ms == 0
            else _odd_window_samples(smoothing_ms, median_interval_ms)
        )
        config = replace(base_config, smoothing_window_ms=smoothing_ms)
        smoothed = smooth_contiguous_median(
            detector_values,
            frame_steps,
            window_samples=smoothing_samples,
        )
        envelope = rolling_extreme_envelope(
            smoothed,
            frame_steps,
            max_window_samples=_odd_window_samples(
                config.envelope_max_window_ms,
                median_interval_ms,
            ),
            min_window_samples=_odd_window_samples(
                config.envelope_min_window_ms,
                median_interval_ms,
            ),
        )
        valid = (
            base_valid
            & np.isfinite(envelope)
            & (coverage >= config.minimum_valid_tail_fraction)
        )
        moving, bout_ids = detect_legacy_envelope_bouts(
            envelope,
            detector_values,
            delta_time,
            frame_steps,
            valid,
            envelope_threshold=config.envelope_threshold_rad_per_ms,
            amplitude_threshold=config.bout_amplitude_threshold_rad_per_ms,
            minimum_bout_duration_ms=config.minimum_bout_duration_ms,
            maximum_interbout_gap_ms=config.maximum_interbout_gap_ms,
        )
        results[f"{smoothing_ms:g}ms"] = {
            "smoothing_window_samples": smoothing_samples,
            "detector_source_column": DETECTOR_SOURCE_COLUMN,
            "valid_fraction": float(np.mean(valid)),
            "movement_fraction": (
                float(np.mean(moving[valid])) if bool(np.any(valid)) else None
            ),
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
