"""Measured-time trial alignment and compact temporal profiles."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.artifacts import (
    artifact_staging,
    load_and_verify_source_manifest,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.analysis.movement_state import (
    DETECTOR_COLUMNS,
    METRIC_IDS,
    resolve_candidate_metric_source,
)
from classical_conditioning.config import get_trial_block_lookup
from classical_conditioning.preprocessing.candidates_v1 import CANDIDATE_COLUMNS

# Identity of the one metric-independent detector whose segmentation every
# bout-derived outcome in this table uses.
SHARED_DETECTOR_ID = "legacy-envelope-shared-v1"


@dataclass(frozen=True)
class TemporalProfileConfig:
    window_start_s: float = -45.0
    window_end_s: float = 45.0
    bin_width_s: float = 0.5
    interval_closure: str = "left"
    aggregation: str = "mean_total_activity"
    # Historical two-layer scaling constants: layer 1 uses frames earlier than
    # -15 s, layer 2 uses every pre-onset bin, and the result is clipped to
    # the unit interval.
    scaling_baseline_end_s: float = -15.0
    scaling_onset_s: float = 0.0
    scaling_lower_quantile: float = 0.1
    scaling_upper_quantile: float = 0.9
    scaling_clip: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class TemporalProfileResult:
    recording_id: str
    profiles_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int
    cs_trial_count: int
    us_trial_count: int


def _block_lookup(experiment_name: str) -> dict[tuple[str, int], str]:
    return get_trial_block_lookup(experiment_name)


def _quantile_range_scale(
    values: np.ndarray,
    reference: np.ndarray,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> np.ndarray:
    """Map values onto a reference window's quantile range.

    Returns all-NaN when the reference window is empty or degenerate, so a
    trial with no usable baseline is never silently rescaled against itself.
    """
    finite_reference = reference[np.isfinite(reference)]
    if finite_reference.size == 0:
        return np.full(values.shape, np.nan)
    low = float(np.quantile(finite_reference, lower_quantile))
    high = float(np.quantile(finite_reference, upper_quantile))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.full(values.shape, np.nan)
    return (values - low) / (high - low)


def _two_layer_scaled_activity(
    values: np.ndarray,
    valid: np.ndarray,
    bin_indices: np.ndarray,
    trial_seconds: np.ndarray,
    bin_centers: np.ndarray,
    *,
    bin_count: int,
    config: TemporalProfileConfig,
) -> np.ndarray:
    """Reproduce the historical two-layer per-trial scaling.

    Layer 1 runs on frames, before binning, against the P10-P90 range of the
    samples earlier than ``scaling_baseline_end_s`` in this trial. Layer 2 runs
    on the binned result against the P10-P90 range of every pre-onset bin, then
    clips to the unit interval. Both layers are per trial and per metric, which
    is what the pre-refactor pipeline did.
    """
    frame_reference = values[
        valid & (trial_seconds < config.scaling_baseline_end_s)
    ]
    scaled_frames = _quantile_range_scale(
        values,
        frame_reference,
        lower_quantile=config.scaling_lower_quantile,
        upper_quantile=config.scaling_upper_quantile,
    )
    usable = valid & np.isfinite(scaled_frames)
    if not np.any(usable):
        return np.full(bin_count, np.nan)
    scaled_count = np.bincount(bin_indices[usable], minlength=bin_count)
    scaled_sum = np.bincount(
        bin_indices[usable],
        weights=scaled_frames[usable],
        minlength=bin_count,
    )
    binned = np.divide(
        scaled_sum,
        scaled_count,
        out=np.full(bin_count, np.nan),
        where=scaled_count > 0,
    )
    normalized = _quantile_range_scale(
        binned,
        binned[bin_centers < config.scaling_onset_s],
        lower_quantile=config.scaling_lower_quantile,
        upper_quantile=config.scaling_upper_quantile,
    )
    return np.clip(normalized, *config.scaling_clip)


def aggregate_event_profiles(
    frames: pd.DataFrame,
    protocol: pd.DataFrame,
    *,
    config: TemporalProfileConfig,
    block_lookup: dict[tuple[str, int], str] | None = None,
    movement_state: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aggregate continuous candidate activity into trial-relative time bins."""
    if config.window_start_s >= config.window_end_s:
        raise ValueError("Profile window start must precede its end.")
    if config.bin_width_s <= 0:
        raise ValueError("Profile bin width must be positive.")
    if config.interval_closure != "left":
        raise ValueError(
            "candidate-temporal-outcomes-v3 uses left-closed, right-open bins."
        )
    required_frame_columns = {
        "AbsoluteTime",
        "FrameStep",
        "DeltaTimeMs",
        *CANDIDATE_COLUMNS,
    }
    missing = required_frame_columns.difference(frames.columns)
    if missing:
        raise KeyError(f"Candidate frames are missing columns: {sorted(missing)}")
    required_protocol_columns = {"Type", "Beg", "End"}
    missing_protocol = required_protocol_columns.difference(protocol.columns)
    if missing_protocol:
        raise KeyError(
            f"Protocol is missing columns: {sorted(missing_protocol)}"
        )

    absolute_time = frames["AbsoluteTime"].to_numpy(dtype=np.int64)
    if np.any(np.diff(absolute_time) < 0):
        raise ValueError("Candidate frame AbsoluteTime must be monotonic.")
    metric_values = {
        column: frames[column].to_numpy(dtype=np.float64)
        for column in CANDIDATE_COLUMNS
    }
    # One shared segmentation serves every metric, so bout-derived outcomes are
    # metric-independent: they differ across metric rows only in that the raw
    # intensity averaged inside those bouts differs.
    shared_movement: dict[str, np.ndarray] | None = None
    if movement_state is not None:
        if not np.array_equal(
            frames["AbsoluteTime"].to_numpy(dtype=np.int64),
            movement_state["AbsoluteTime"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("Movement-state rows do not align with candidate frames.")
        delta_time_all = frames["DeltaTimeMs"].to_numpy(dtype=np.float64)
        missing_detector = [
            column
            for column in DETECTOR_COLUMNS
            if column not in movement_state.columns
        ]
        if missing_detector:
            raise KeyError(
                "Movement state is missing shared detector columns: "
                f"{missing_detector}"
            )
        valid = movement_state["valid"].to_numpy(dtype=bool)
        moving = movement_state["moving"].to_numpy(dtype=bool)
        bout_id = movement_state["bout_id"].to_numpy(dtype=np.int32)
        bout_start = (bout_id > 0) & np.concatenate(
            [[True], bout_id[1:] != bout_id[:-1]]
        )
        bout_duration = np.bincount(
            bout_id,
            weights=np.where(
                valid & np.isfinite(delta_time_all),
                delta_time_all,
                0.0,
            ),
            minlength=int(bout_id.max()) + 1,
        )
        shared_movement = {
            "valid": valid,
            "moving": moving,
            "bout_id": bout_id,
            "bout_start": bout_start,
            "bout_duration": bout_duration,
        }
    delta_time = frames["DeltaTimeMs"].to_numpy(dtype=float)
    frame_step = frames["FrameStep"].to_numpy(dtype=np.int64)
    valid_delta_time = delta_time[
        np.isfinite(delta_time) & (delta_time > 0) & (frame_step == 1)
    ]
    if valid_delta_time.size == 0:
        raise ValueError("No positive measured frame intervals are available.")
    median_delta_time_ms = float(np.median(valid_delta_time))
    expected_sample_count = max(
        1,
        int(round(config.bin_width_s * 1_000 / median_delta_time_ms)),
    )
    bin_count = int(
        np.ceil(
            (config.window_end_s - config.window_start_s) / config.bin_width_s
        )
    )
    rows: list[dict[str, Any]] = []
    event_mapping = {"Cycle": "CS", "Reinforcer": "US"}
    counters = {"CS": 0, "US": 0}
    block_lookup = block_lookup or {}

    relevant_protocol = protocol[
        protocol["Type"].astype(str).isin(event_mapping)
    ].sort_values("Beg", kind="stable")
    for event in relevant_protocol.itertuples(index=False):
        event_type = str(event.Type)
        if event_type not in event_mapping:
            continue
        trial_type = event_mapping[event_type]
        counters[trial_type] += 1
        trial_number = counters[trial_type]
        event_start_ms = int(event.Beg)
        window_start_ms = event_start_ms + int(config.window_start_s * 1_000)
        window_end_ms = event_start_ms + int(config.window_end_s * 1_000)
        start_index = int(np.searchsorted(absolute_time, window_start_ms, side="left"))
        end_index = int(np.searchsorted(absolute_time, window_end_ms, side="left"))
        relative_seconds = (
            absolute_time[start_index:end_index] - event_start_ms
        ) / 1_000
        bin_indices = np.floor(
            (relative_seconds - config.window_start_s) / config.bin_width_s
        ).astype(np.int64)
        in_range = (bin_indices >= 0) & (bin_indices < bin_count)
        bin_indices = bin_indices[in_range]
        sample_count = np.bincount(bin_indices, minlength=bin_count)
        trial_seconds = relative_seconds[in_range]
        bin_centers = (
            config.window_start_s
            + (np.arange(bin_count) + 0.5) * config.bin_width_s
        )

        for column, metric_id in METRIC_IDS.items():
            values = metric_values[column][start_index:end_index][in_range]
            valid = np.isfinite(values)
            valid_count = np.bincount(
                bin_indices[valid],
                minlength=bin_count,
            )
            sums = np.bincount(
                bin_indices[valid],
                weights=values[valid],
                minlength=bin_count,
            )
            means = np.divide(
                sums,
                valid_count,
                out=np.full(bin_count, np.nan),
                where=valid_count > 0,
            )
            valid_fraction = np.divide(
                valid_count,
                sample_count,
                out=np.zeros(bin_count, dtype=float),
                where=sample_count > 0,
            )
            scaled_means = _two_layer_scaled_activity(
                values,
                valid,
                bin_indices,
                trial_seconds,
                bin_centers,
                bin_count=bin_count,
                config=config,
            )
            movement_probability = np.full(bin_count, np.nan)
            fraction_time_moving = np.full(bin_count, np.nan)
            conditional_intensity = np.full(bin_count, np.nan)
            bout_count = np.full(bin_count, np.nan)
            bout_rate = np.full(bin_count, np.nan)
            mean_bout_duration = np.full(bin_count, np.nan)
            detector_valid_fraction = np.full(bin_count, np.nan)
            if shared_movement is not None:
                movement_metric = shared_movement
                detector_valid = movement_metric["valid"][
                    start_index:end_index
                ][in_range]
                moving = movement_metric["moving"][start_index:end_index][
                    in_range
                ]
                delta_time = frames["DeltaTimeMs"].to_numpy(dtype=float)[
                    start_index:end_index
                ][in_range]
                valid_detector_count = np.bincount(
                    bin_indices[detector_valid],
                    minlength=bin_count,
                )
                moving_count = np.bincount(
                    bin_indices[detector_valid & moving],
                    minlength=bin_count,
                )
                movement_probability = np.divide(
                    moving_count,
                    valid_detector_count,
                    out=np.full(bin_count, np.nan),
                    where=valid_detector_count > 0,
                )
                valid_time = np.bincount(
                    bin_indices[detector_valid],
                    weights=delta_time[detector_valid],
                    minlength=bin_count,
                )
                moving_time = np.bincount(
                    bin_indices[detector_valid & moving],
                    weights=delta_time[detector_valid & moving],
                    minlength=bin_count,
                )
                fraction_time_moving = np.divide(
                    moving_time,
                    valid_time,
                    out=np.full(bin_count, np.nan),
                    where=valid_time > 0,
                )
                moving_values = valid & detector_valid & moving
                conditional_sum = np.bincount(
                    bin_indices[moving_values],
                    weights=values[moving_values],
                    minlength=bin_count,
                )
                conditional_count = np.bincount(
                    bin_indices[moving_values],
                    minlength=bin_count,
                )
                conditional_intensity = np.divide(
                    conditional_sum,
                    conditional_count,
                    out=np.full(bin_count, np.nan),
                    where=conditional_count > 0,
                )
                onset = movement_metric["bout_start"][start_index:end_index][
                    in_range
                ]
                onset_ids = movement_metric["bout_id"][start_index:end_index][
                    in_range
                ][onset]
                bout_count = np.bincount(
                    bin_indices[onset],
                    minlength=bin_count,
                ).astype(float)
                bout_count[valid_detector_count == 0] = np.nan
                duration_sum = np.bincount(
                    bin_indices[onset],
                    weights=movement_metric["bout_duration"][onset_ids],
                    minlength=bin_count,
                )
                mean_bout_duration = np.divide(
                    duration_sum,
                    bout_count,
                    out=np.full(bin_count, np.nan),
                    where=bout_count > 0,
                )
                bout_rate = np.divide(
                    bout_count * 60_000,
                    valid_time,
                    out=np.full(bin_count, np.nan),
                    where=valid_time > 0,
                )
                detector_valid_fraction = np.divide(
                    valid_detector_count,
                    expected_sample_count,
                    out=np.zeros(bin_count, dtype=float),
                    where=expected_sample_count > 0,
                )
                detector_valid_fraction = np.clip(
                    detector_valid_fraction,
                    0.0,
                    1.0,
                )
            for bin_index in range(bin_count):
                rows.append(
                    {
                        "Trial type": trial_type,
                        "Trial number": trial_number,
                        "Block name": block_lookup.get(
                            (trial_type, trial_number),
                            "",
                        ),
                        "Event start absolute time (ms)": event_start_ms,
                        "Time bin start (s)": (
                            config.window_start_s
                            + bin_index * config.bin_width_s
                        ),
                        "Time bin center (s)": (
                            config.window_start_s
                            + (bin_index + 0.5) * config.bin_width_s
                        ),
                        "Metric ID": metric_id,
                        "Movement detector ID": (
                            SHARED_DETECTOR_ID
                            if shared_movement is not None
                            else None
                        ),
                        "Total activity mean": means[bin_index],
                        "Scaled total activity": scaled_means[bin_index],
                        "Movement probability": movement_probability[bin_index],
                        "Fraction time moving": fraction_time_moving[bin_index],
                        "Conditional intensity mean": conditional_intensity[bin_index],
                        "Bout count": (
                            int(bout_count[bin_index])
                            if np.isfinite(bout_count[bin_index])
                            else None
                        ),
                        "Bout rate per minute": bout_rate[bin_index],
                        "Mean bout duration (ms)": mean_bout_duration[bin_index],
                        "Detector valid fraction": detector_valid_fraction[bin_index],
                        "Valid fraction": valid_fraction[bin_index],
                        "Expected sample count": expected_sample_count,
                        "Acquisition coverage": min(
                            1.0,
                            sample_count[bin_index] / expected_sample_count,
                        ),
                        "Valid expected fraction": min(
                            1.0,
                            valid_count[bin_index] / expected_sample_count,
                        ),
                        "Sample count": int(sample_count[bin_index]),
                        "Valid sample count": int(valid_count[bin_index]),
                    }
                )
    columns = [
        "Trial type",
        "Trial number",
        "Block name",
        "Event start absolute time (ms)",
        "Time bin start (s)",
        "Time bin center (s)",
        "Metric ID",
        "Movement detector ID",
        "Total activity mean",
        "Scaled total activity",
        "Movement probability",
        "Fraction time moving",
        "Conditional intensity mean",
        "Bout count",
        "Bout rate per minute",
        "Mean bout duration (ms)",
        "Detector valid fraction",
        "Valid fraction",
        "Expected sample count",
        "Acquisition coverage",
        "Valid expected fraction",
        "Sample count",
        "Valid sample count",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_candidate_temporal_profiles(
    project_dir: Path,
    recording_id: str,
    *,
    experiment_name: str = "allDelay",
    config: TemporalProfileConfig | None = None,
    metric_recipe: str = "tail-candidate-development-v1",
    overwrite: bool = False,
) -> TemporalProfileResult:
    """Build candidate temporal profiles from the candidate frame artifact."""
    config = config or TemporalProfileConfig()
    if config != TemporalProfileConfig():
        raise ValueError(
            "Temporal profile recipes use a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    source = resolve_candidate_metric_source(metric_recipe=metric_recipe)
    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    frame_path = source_dir / source.metrics_name
    protocol_path = source_dir / "stimulus_events.parquet"
    candidate_marker = (
        project_dir / "Metadata" / f"{recording_id}_{source.metric_marker_suffix}"
    )
    movement_path = source_dir / source.movement_artifact_name
    movement_marker_path = (
        project_dir
        / "Metadata"
        / f"{recording_id}_{source.movement_marker_suffix}"
    )
    movement_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.movement_summary_name
    )
    if not frame_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("Candidate frame or protocol artifact is missing.")
    if not candidate_marker.is_file():
        raise FileNotFoundError(f"Missing candidate completion marker: {candidate_marker}")

    marker = json.loads(candidate_marker.read_text(encoding="utf-8"))
    if (
        marker.get("status") != "complete"
        or marker.get("recording_id") != recording_id
        or marker.get("recipe") != source.metric_recipe
    ):
        raise ValueError("Candidate completion marker is invalid.")
    if sha256_file(frame_path) != marker.get("metrics_sha256"):
        raise ValueError("Candidate frame artifact hash differs from its marker.")

    candidate_summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.metric_summary_name
    )
    if not candidate_summary_path.is_file():
        raise FileNotFoundError(f"Missing candidate summary: {candidate_summary_path}")
    if sha256_file(candidate_summary_path) != marker.get("summary_sha256"):
        raise ValueError("Candidate summary hash differs from its marker.")
    candidate_summary = json.loads(
        candidate_summary_path.read_text(encoding="utf-8")
    )
    if not movement_path.is_file() or not movement_marker_path.is_file():
        raise FileNotFoundError("Candidate movement-state artifact is missing.")
    movement_marker = json.loads(
        movement_marker_path.read_text(encoding="utf-8")
    )
    if (
        movement_marker.get("status") != "complete"
        or movement_marker.get("recipe") != source.movement_recipe
        or movement_marker.get("recording_id") != recording_id
        or sha256_file(movement_path) != movement_marker.get("movement_sha256")
    ):
        raise ValueError("Candidate movement-state marker or artifact is invalid.")
    if (
        not movement_summary_path.is_file()
        or sha256_file(movement_summary_path)
        != movement_marker.get("summary_sha256")
    ):
        raise ValueError("Candidate movement-state summary is invalid.")
    movement_summary = json.loads(
        movement_summary_path.read_text(encoding="utf-8")
    )
    if (
        movement_summary.get("recording_id") != recording_id
        or movement_summary.get("recipe") != source.movement_recipe
        or movement_summary["inputs"]["candidate_metrics"]["sha256"]
        != marker["metrics_sha256"]
    ):
        raise ValueError(
            "Candidate movement state was built from different candidate metrics."
        )

    _, input_artifacts, input_state = load_and_verify_source_manifest(
        project_dir,
        recording_id,
    )
    for kind, current in input_artifacts.items():
        previous = candidate_summary["input_artifacts"][kind]
        if previous["sha256"] != current["sha256"]:
            raise ValueError(
                f"Candidate metrics were built from a different {kind} artifact."
            )
    protocol_record = input_artifacts["protocol"]
    if sha256_file(protocol_path) != protocol_record["sha256"]:
        raise ValueError("Protocol artifact hash differs from the intake manifest.")

    output_path = source_dir / source.temporal_artifact_name
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / source.temporal_summary_name
    )
    completion_path = (
        project_dir
        / "Metadata"
        / f"{recording_id}_{source.temporal_marker_suffix}"
    )
    existing = [
        path for path in (output_path, summary_path, completion_path) if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Candidate temporal outputs already exist: {existing}")

    frame_columns = [
        "FrameID",
        "AbsoluteTime",
        "FrameStep",
        "DeltaTimeMs",
        *CANDIDATE_COLUMNS,
    ]
    frames = pq.read_table(frame_path, columns=frame_columns).to_pandas()
    movement_columns = ["FrameID", "AbsoluteTime", *DETECTOR_COLUMNS]
    movement_state = pq.read_table(
        movement_path,
        columns=movement_columns,
    ).to_pandas()
    movement_state_file_state = (
        movement_path.stat().st_size,
        movement_path.stat().st_mtime_ns,
    )
    if not np.array_equal(
        frames["FrameID"].to_numpy(dtype=np.int64),
        movement_state["FrameID"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Candidate movement-state FrameID values do not align.")
    protocol = pq.read_table(protocol_path).to_pandas()
    profiles = aggregate_event_profiles(
        frames,
        protocol,
        config=config,
        block_lookup=_block_lookup(experiment_name),
        movement_state=movement_state,
    )
    profiles.insert(0, "Recording ID", recording_id)
    profiles["Trial type"] = profiles["Trial type"].astype("category")
    profiles["Block name"] = profiles["Block name"].astype("category")
    profiles["Metric ID"] = profiles["Metric ID"].astype("category")

    if sha256_file(frame_path) != marker["metrics_sha256"]:
        raise RuntimeError("Candidate frame artifact changed during profile build.")
    if (
        movement_state_file_state
        != (movement_path.stat().st_size, movement_path.stat().st_mtime_ns)
        or sha256_file(movement_path) != movement_marker["movement_sha256"]
    ):
        raise RuntimeError("Candidate movement state changed during profile build.")
    stat = protocol_path.stat()
    if input_state["protocol"] != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError("Protocol artifact changed during profile build.")

    cs_trials = int(
        profiles.loc[profiles["Trial type"] == "CS", "Trial number"].nunique()
    )
    us_trials = int(
        profiles.loc[profiles["Trial type"] == "US", "Trial number"].nunique()
    )
    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-{source.temporal_recipe}-",
    ) as staging_root:
        staged_profiles = staging_root / output_path.name
        staged_summary = staging_root / summary_path.name
        staged_completion = staging_root / completion_path.name
        table = pa.Table.from_pandas(profiles, preserve_index=False)
        pq.write_table(
            table,
            staged_profiles,
            compression="zstd",
            row_group_size=100_000,
            write_statistics=True,
        )
        profile_hash = sha256_file(staged_profiles)
        summary = {
            "recipe": source.temporal_recipe,
            "scientific_status": "candidate_development",
            "paper_approved": False,
            "metric_source_recipe": source.metric_recipe,
            "movement_source_recipe": source.movement_recipe,
            "recording_id": recording_id,
            "experiment": experiment_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config),
            "row_count": len(profiles),
            "trial_counts": {"CS": cs_trials, "US": us_trials},
            "metric_ids": list(METRIC_IDS.values()),
            "output_semantics": (
                "Continuous total activity includes valid zeros. Movement "
                "probability, time moving, conditional intensity, and bout "
                f"outcomes use {source.movement_recipe}."
            ),
            "artifact": {
                "path": str(output_path.resolve()),
                "sha256": profile_hash,
                "compression": "zstd",
                "compression_lossless": True,
            },
            "inputs": {
                "candidate_frames": {
                    "path": str(frame_path),
                    "sha256": marker["metrics_sha256"],
                    "recipe": source.metric_recipe,
                },
                "protocol": protocol_record,
                "movement_state": {
                    "path": str(movement_path),
                    "sha256": movement_marker["movement_sha256"],
                    "recipe": source.movement_recipe,
                },
            },
        }
        write_json_atomic(staged_summary, summary)
        summary_hash = sha256_file(staged_summary)
        write_json_atomic(
            staged_completion,
            {
                "status": "complete",
                "recipe": source.temporal_recipe,
                "recording_id": recording_id,
                "profiles_sha256": profile_hash,
                "summary_sha256": summary_hash,
            },
        )
        publish_transaction(
            (
                (staged_profiles, output_path),
                (staged_summary, summary_path),
                (staged_completion, completion_path),
            ),
            staging_root,
            overwrite=overwrite,
        )

    return TemporalProfileResult(
        recording_id=recording_id,
        profiles_path=output_path,
        summary_path=summary_path,
        completion_marker_path=completion_path,
        row_count=len(profiles),
        cs_trial_count=cs_trials,
        us_trial_count=us_trials,
    )
