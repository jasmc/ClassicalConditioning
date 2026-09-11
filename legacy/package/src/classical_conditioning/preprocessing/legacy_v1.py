"""Legacy-equivalent stage-1 preprocessing.

This module intentionally preserves the executable behavior of the original
stage-1 script. Known scientific limitations are recorded in the output
summary and are addressed only by separate corrected recipes.
"""

from __future__ import annotations

import json
import os
import gc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import CategoricalDtype
from scipy import interpolate

from classical_conditioning.artifacts import (
    artifact_staging,
    load_and_verify_source_manifest as _load_and_verify_source_manifest,
    publish_transaction as _publish_transaction,
    sha256_file as _sha256_file,
    write_json_atomic as _write_json_atomic,
)
from classical_conditioning.config import (
    get_experiment_spec,
    get_trial_block_lookup,
)

TIME_COLUMN = "Trial time (frame) [700 FPS]"
ANGLE_TEMPLATE = "Angle of point {index} (deg)"
VIGOR_COLUMN = "Vigor (deg/ms)"
BOUT_METRIC_COLUMN = "Vigor for bout detection (deg/ms)"
SCALED_VIGOR_COLUMN = "Scaled vigor (AU)"

KNOWN_LIMITATIONS = (
    "Camera ingestion discards the first 13,999 source rows.",
    "Tracking ingestion discards the final source row.",
    "Measured x/y coordinates are not used.",
    "Frame-loss detection uses accumulated timestamp drift rather than direct FrameID gaps.",
    "Tracking is interpolated and extrapolated onto a uniform nominal 700 FPS grid.",
    "Spatial filtering is not applied although a spatial window is configured.",
    "Vigor is the absolute derivative of the distal cumulative angle.",
    "The configured secondary bout threshold is not used.",
    "Scaled vigor uses P10/P90 from samples earlier than -15 seconds.",
)


@dataclass(frozen=True)
class LegacyPreprocessingConfig:
    expected_framerate_hz: float = 700.0
    camera_rows_discarded_at_start: int = 13_999
    maximum_interval_deviation_ms: float = 0.005
    frame_loss_buffer_frames: int = 700
    tracking_error_threshold_deg: float = 2 * 180 / np.pi
    angle_point_count: int = 16
    temporal_filter_frames: int = 10
    bout_max_window_frames: int = 20
    bout_min_window_frames: int = 400
    bout_threshold_primary_deg_per_ms: float = 4.0
    bout_threshold_secondary_deg_per_ms: float = 1.0
    minimum_bout_duration_frames: int = 40
    minimum_interbout_frames: int = 10
    trial_start_frames: int = -45 * 700
    trial_end_frames: int = 45 * 700
    baseline_window_frames: int = 15 * 700


@dataclass(frozen=True)
class LegacyPreprocessingResult:
    recording_id: str
    samples_path: Path
    summary_path: Path
    completion_marker_path: Path
    row_count: int
    cs_trial_count: int
    us_trial_count: int


def _parse_legacy_metadata(recording_name: str) -> dict[str, str]:
    parts = recording_name.lower().split("_")
    if len(parts) < 6:
        raise ValueError(f"Cannot parse legacy fish metadata: {recording_name}")
    return {
        "Day": parts[0],
        "Fish no.": parts[1],
        "Exp.": parts[2],
        "ProtocolRig": parts[3],
        "Strain": parts[4],
        "Age (dpf)": parts[5].replace("dpf", ""),
    }


def prepare_legacy_camera(
    camera: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    """Apply the current stage-1 camera-row discard."""
    required = {"FrameID", "ElapsedTime", "AbsoluteTime"}
    missing = required.difference(camera.columns)
    if missing:
        raise KeyError(f"Camera table is missing columns: {sorted(missing)}")
    result = camera.iloc[config.camera_rows_discarded_at_start :].copy()
    if result.empty:
        raise ValueError("Camera table is empty after the legacy initial-row discard.")
    return result


def prepare_legacy_tracking(
    tracking: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    """Select legacy angle columns, drop the final row, and convert to degrees."""
    source_angle_columns = [
        f"angle{index}" for index in range(config.angle_point_count)
    ]
    required = {"FrameID", *source_angle_columns}
    missing = required.difference(tracking.columns)
    if missing:
        raise KeyError(f"Tracking table is missing columns: {sorted(missing)}")
    if len(tracking) < 2:
        raise ValueError("Tracking table needs at least two rows.")

    result = tracking.loc[:, ["FrameID", *source_angle_columns]].iloc[:-1].copy()
    result["Original frame number"] = result["FrameID"].astype("int32")
    result[source_angle_columns] = (
        result[source_angle_columns].astype("float32") * np.float32(180 / np.pi)
    )
    result.rename(
        columns={
            source: ANGLE_TEMPLATE.format(index=index)
            for index, source in enumerate(source_angle_columns)
        },
        inplace=True,
    )
    return result


def legacy_framerate_and_reference(
    camera: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> tuple[float, int, bool]:
    """Reproduce the current timestamp-based reference and loss flag."""
    camera_diff = camera["ElapsedTime"].astype(float).diff()
    interframe_interval = float(camera_diff.median())
    valid = np.where(
        np.abs(camera_diff - interframe_interval)
        <= config.maximum_interval_deviation_ms
    )[0]
    valid_steps = np.diff(valid)

    reference_frame_id = 0
    last_frame_id = 0
    for index in range(1, len(valid_steps)):
        if valid_steps[index - 1] == 1 and valid_steps[index] == 1:
            reference_frame_id = int(camera["FrameID"].iloc[valid[index] - 1])
            break
    for index in range(len(valid_steps) - 1, 0, -1):
        if valid_steps[index - 1] == 1 and valid_steps[index] == 1:
            last_frame_id = int(camera["FrameID"].iloc[valid[index] - 1])
            break

    try:
        start = reference_frame_id - int(camera["FrameID"].iloc[0])
        stop = last_frame_id - int(camera["FrameID"].iloc[0])
        interframe_interval = float(camera_diff.iloc[start:stop].mean())
    except (IndexError, TypeError, ValueError):
        pass

    predicted_framerate = 1_000 / interframe_interval
    delay = (camera_diff - interframe_interval).cumsum().to_numpy()
    frames_lost = np.floor(
        delay / (interframe_interval * config.frame_loss_buffer_frames)
    )
    frames_lost = np.where(frames_lost >= 0, frames_lost, 0)
    frame_loss_steps = np.floor(np.diff(frames_lost, prepend=0))
    frame_loss_steps = np.where(frame_loss_steps >= 0, frame_loss_steps, 0)
    has_lost_frames = bool(np.any(frame_loss_steps > 0))
    return predicted_framerate, reference_frame_id, has_lost_frames


def has_legacy_tracking_errors(
    tracking: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> bool:
    angle_columns = [column for column in tracking if column.startswith("Angle")]
    return bool(
        tracking[angle_columns].isna().to_numpy().any()
        or (tracking[angle_columns].abs().max() > config.tracking_error_threshold_deg).any()
    )


def synchronize_legacy(
    tracking: pd.DataFrame,
    camera: pd.DataFrame,
) -> pd.DataFrame:
    result = pd.merge_ordered(tracking, camera, on="FrameID", how="inner")
    if result.empty:
        raise ValueError("Camera and tracking have no overlapping FrameID values.")
    result["FrameID"] -= result["FrameID"].iat[0]
    return result


def interpolate_legacy(
    data: pd.DataFrame,
    expected_framerate: float,
    predicted_framerate: float,
) -> pd.DataFrame:
    source = data.copy(deep=True)
    source["FrameID"] *= expected_framerate / predicted_framerate
    source.rename(columns={"FrameID": TIME_COLUMN}, inplace=True)
    values = source.drop(columns=TIME_COLUMN)
    function = interpolate.interp1d(
        source[TIME_COLUMN],
        values,
        kind="slinear",
        axis=0,
        assume_sorted=True,
        bounds_error=False,
        fill_value="extrapolate",
    )
    new_time = np.arange(source[TIME_COLUMN].iat[0], source[TIME_COLUMN].iat[-1])
    result = pd.DataFrame({TIME_COLUMN: new_time})
    result[values.columns] = function(result[TIME_COLUMN])
    return result


def _legacy_protocol(protocol: pd.DataFrame) -> pd.DataFrame:
    required = {"Type", "Beg", "End"}
    missing = required.difference(protocol.columns)
    if missing:
        raise KeyError(f"Protocol table is missing columns: {sorted(missing)}")
    if protocol.empty or protocol["Beg"].iloc[0] == 0:
        raise ValueError("Legacy protocol rule rejects an empty or zero-start protocol.")
    result = protocol.rename(
        columns={"Type": "Experiment type", "Beg": "beg (ms)", "End": "end (ms)"}
    ).set_index("Experiment type")
    return result.sort_values("beg (ms)")


def annotate_stimuli_legacy(
    data: pd.DataFrame,
    protocol: pd.DataFrame,
) -> pd.DataFrame:
    """Reproduce strict-start/inclusive-end stimulus markers."""
    result = data.copy(deep=True)
    absolute_time = result["AbsoluteTime"].to_numpy(dtype=float)
    for column in ("CS beg", "CS end", "US beg", "US end"):
        result[column] = np.zeros(len(result), dtype=np.int32)

    event_mapping = {"CS": "Cycle", "US": "Reinforcer"}
    for label, event_type in event_mapping.items():
        if event_type not in protocol.index:
            continue
        selected = protocol.loc[
            event_type, ["beg (ms)", "end (ms)"]
        ].to_numpy()
        for event_index, event in enumerate(selected, start=1):
            try:
                begin, end = float(event[0]), float(event[1])
            except (IndexError, TypeError):
                continue
            begin_index = int(np.searchsorted(absolute_time, begin, side="right"))
            end_index = int(np.searchsorted(absolute_time, end, side="right") - 1)
            if begin_index >= len(result) or end_index < 0:
                continue
            result.iat[
                begin_index, result.columns.get_loc(f"{label} beg")
            ] = event_index
            result.iat[end_index, result.columns.get_loc(f"{label} end")] = event_index

    result["AbsoluteTime"] = (
        result["AbsoluteTime"] - result["AbsoluteTime"].iat[0]
    ).astype("int32")
    return result


def cumulative_angles_legacy(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    angle_columns = sorted(
        [column for column in result if column.startswith("Angle of point")],
        key=lambda value: int(value.split("Angle of point ")[1].split(" ")[0]),
    )
    result[angle_columns] = result[angle_columns].cumsum(axis=1)
    return result


def filter_angles_legacy(
    data: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    result = data.copy()
    angle_columns = [column for column in result if "Angle" in column]
    result[angle_columns] = result[angle_columns].rolling(
        window=config.temporal_filter_frames,
        center=True,
    ).mean()
    result = result.dropna().copy()
    result[angle_columns] = result[angle_columns].astype("float32")
    return result


def calculate_vigor_legacy(
    data: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    result = data.copy()
    tail_column = ANGLE_TEMPLATE.format(index=config.angle_point_count - 1)
    angle = result[tail_column].to_numpy()
    differences = np.abs(np.diff(angle, prepend=angle[0]))
    result[VIGOR_COLUMN] = (
        differences * (config.expected_framerate_hz / 1_000)
    ).astype("float32")
    return result


def calculate_bout_metric_legacy(
    data: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    result = data.copy()
    result[BOUT_METRIC_COLUMN] = (
        result[VIGOR_COLUMN]
        .rolling(window=config.bout_max_window_frames, center=True)
        .max()
        - result[VIGOR_COLUMN]
        .rolling(window=config.bout_min_window_frames, center=True)
        .min()
    )
    return result.dropna().copy()


def detect_bouts_legacy(
    data: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    result = data.copy()
    metric = result[BOUT_METRIC_COLUMN]
    bouts = np.zeros(len(metric))
    bouts[1:-1][
        metric.iloc[1:-1] >= config.bout_threshold_primary_deg_per_ms
    ] = 1

    def bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        begins = np.where(np.diff(values) > 0)[0] + 1
        ends = np.where(np.diff(values) < 0)[0]
        return begins, ends

    begins, ends = bounds(bouts)
    if len(begins) == len(ends):
        intervals = begins[1:] - ends[:-1]
        for index in reversed(
            np.where(intervals < config.minimum_interbout_frames)[0]
        ):
            bouts[ends[index] + 1 : begins[index + 1]] = 1
    begins, ends = bounds(bouts)
    durations = ends - begins
    for index in np.where(durations < config.minimum_bout_duration_frames)[0]:
        bouts[begins[index] : ends[index] + 1] = 0

    result["Bout"] = bouts.astype(bool)
    result["Bout beg"] = result["Bout"].diff() > 0
    result["Bout end"] = result["Bout"].diff() < 0
    return result


def segment_trials_legacy(
    data: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    trials: list[pd.DataFrame] = []
    for trial_type in ("CS", "US"):
        marker = f"{trial_type} beg"
        trial_ids = pd.to_numeric(
            data.loc[data[marker] != 0, marker],
            errors="coerce",
        ).dropna().unique()
        for trial_id in trial_ids:
            reference = data.loc[
                pd.to_numeric(data[marker], errors="coerce") == trial_id,
                TIME_COLUMN,
            ].to_numpy()[0]
            trial = data.loc[
                data[TIME_COLUMN].between(
                    reference + config.trial_start_frames,
                    reference + config.trial_end_frames,
                )
            ].copy()
            trial["Trial type"] = trial_type
            trial["Trial number"] = int(trial_id)
            trial[TIME_COLUMN] = np.arange(
                config.trial_start_frames,
                len(trial) + config.trial_start_frames,
            )
            trials.append(trial)
    if not trials:
        return data.iloc[0:0].copy()
    result = pd.concat(trials)
    result["Trial type"] = result["Trial type"].astype("category")
    result["Trial number"] = result["Trial number"].astype("int32")
    return result


def assign_blocks_legacy(
    data: pd.DataFrame,
    experiment_name: str,
) -> pd.DataFrame:
    experiment = get_experiment_spec(experiment_name)
    block_lookup = get_trial_block_lookup(experiment_name)
    result = data.copy()
    result["Block name"] = ""
    for trial_type in ("CS", "US"):
        alignment_mask = result["Trial type"].eq(trial_type)
        mapped_blocks = result.loc[
            alignment_mask,
            "Trial number",
        ].astype(int).map(
            {
                trial_number: block_name
                for (alignment, trial_number), block_name in block_lookup.items()
                if alignment == trial_type
            }
        )
        result.loc[alignment_mask, "Block name"] = mapped_blocks.fillna("")
    categories = list(
        dict.fromkeys(
            trial.block_10_name
            for trial in experiment.analysis_trials
            if trial.alignment.value == "CS"
        )
    )
    result["Block name"] = result["Block name"].astype(
        CategoricalDtype(categories=categories, ordered=True)
    )
    return result


def scale_vigor_legacy(
    data: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    result = data.copy()
    result[SCALED_VIGOR_COLUMN] = result[VIGOR_COLUMN]
    for trial_id in result["Trial number"].unique():
        trial_mask = result["Trial number"] == trial_id
        baseline = result.loc[
            trial_mask & (result[TIME_COLUMN] < -config.baseline_window_frames),
            VIGOR_COLUMN,
        ].dropna().to_numpy()
        if baseline.size == 0:
            result.loc[trial_mask, SCALED_VIGOR_COLUMN] = np.nan
            continue
        minimum, maximum = np.quantile(baseline, [0.1, 0.9])
        if np.isnan(maximum) or minimum == maximum:
            result.loc[trial_mask, SCALED_VIGOR_COLUMN] = np.nan
            continue
        result.loc[trial_mask, SCALED_VIGOR_COLUMN] = (
            result.loc[trial_mask, VIGOR_COLUMN] - minimum
        ) / (maximum - minimum)
    return result


def finalize_legacy_samples(
    data: pd.DataFrame,
    metadata: dict[str, str],
    config: LegacyPreprocessingConfig,
) -> pd.DataFrame:
    result = data.drop(columns=BOUT_METRIC_COLUMN).copy()
    angle_columns = [
        ANGLE_TEMPLATE.format(index=index)
        for index in range(config.angle_point_count)
    ]
    ordered = [
        "Original frame number",
        TIME_COLUMN,
        "CS beg",
        "CS end",
        "US beg",
        "US end",
        "Trial type",
        "Trial number",
        "Block name",
        VIGOR_COLUMN,
        SCALED_VIGOR_COLUMN,
        "Bout beg",
        "Bout end",
        "Bout",
        *angle_columns,
    ]
    result = result.loc[:, [column for column in ordered if column in result]]
    result[angle_columns] = result[angle_columns].astype("float32")
    result[VIGOR_COLUMN] = result[VIGOR_COLUMN].astype("float32")
    result[SCALED_VIGOR_COLUMN] = result[SCALED_VIGOR_COLUMN].astype("float32")
    result[TIME_COLUMN] = result[TIME_COLUMN].astype("int32")
    result[["Bout beg", "Bout end", "Bout"]] = result[
        ["Bout beg", "Bout end", "Bout"]
    ].astype(bool)
    for column in ("CS beg", "CS end", "US beg", "US end", "Trial number"):
        values = pd.to_numeric(result[column], errors="raise").astype("int32")
        result[column] = pd.Categorical(
            values,
            categories=np.sort(values.unique()),
            ordered=True,
        )
    for column, value in reversed(tuple(metadata.items())):
        result.insert(
            0,
            column,
            pd.Categorical([value] * len(result), categories=[value]),
        )
    return result.reset_index(drop=True)


def _write_parquet_atomic(
    frame: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Derived artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        row_group_size=250_000,
        write_statistics=True,
    )
    parquet = pq.ParquetFile(temporary)
    try:
        if parquet.metadata.num_rows != len(frame):
            raise RuntimeError("Legacy sample Parquet row-count verification failed.")
        row_groups = parquet.num_row_groups
    finally:
        parquet.close()
    size_bytes = temporary.stat().st_size
    digest = _sha256_file(temporary)
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "rows": len(frame),
        "columns": len(frame.columns),
        "size_bytes": size_bytes,
        "sha256": digest,
        "compression": "zstd",
        "compression_lossless": True,
        "row_groups": row_groups,
    }


def preprocess_legacy_recording(
    project_dir: Path,
    recording_id: str,
    *,
    experiment_name: str = "allDelay",
    config: LegacyPreprocessingConfig | None = None,
    overwrite: bool = False,
) -> LegacyPreprocessingResult:
    """Run the frozen current stage-1 behavior from canonical source mirrors."""
    config = config or LegacyPreprocessingConfig()
    if config != LegacyPreprocessingConfig():
        raise ValueError(
            "legacy-paper-v1 uses a frozen configuration. "
            "Parameter changes require a different recipe identity."
        )
    project_dir = project_dir.resolve()
    source_dir = project_dir / "Processed data" / recording_id
    camera_path = source_dir / "camera.parquet"
    tracking_path = source_dir / "tracking.parquet"
    protocol_path = source_dir / "stimulus_events.parquet"
    for path in (camera_path, tracking_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing intake artifact: {path}")

    samples_path = source_dir / "samples_legacy-v1.parquet"
    summary_path = (
        project_dir
        / "Quality checks"
        / recording_id
        / "legacy-v1_preprocessing_summary.json"
    )
    completion_marker_path = (
        project_dir
        / "Metadata"
        / f"{recording_id}_legacy-v1_complete.json"
    )
    if not overwrite:
        existing = [
            path
            for path in (samples_path, summary_path, completion_marker_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(f"Legacy preprocessing outputs exist: {existing}")

    recording_name, input_artifacts, input_file_state = (
        _load_and_verify_source_manifest(project_dir, recording_id)
    )
    metadata = _parse_legacy_metadata(recording_name)
    stage_counts: dict[str, int] = {}

    camera = prepare_legacy_camera(pq.read_table(camera_path).to_pandas(), config)
    tracking_columns = ["FrameID"] + [
        f"angle{index}" for index in range(config.angle_point_count)
    ]
    tracking = prepare_legacy_tracking(
        pq.read_table(tracking_path, columns=tracking_columns).to_pandas(),
        config,
    )
    protocol = _legacy_protocol(pq.read_table(protocol_path).to_pandas())
    stage_counts["camera_after_initial_discard"] = len(camera)
    stage_counts["tracking_after_final_row_discard"] = len(tracking)

    predicted_framerate, reference_frame_id, has_lost_frames = (
        legacy_framerate_and_reference(camera, config)
    )
    if has_lost_frames:
        raise ValueError("Legacy frame-loss rule rejects this recording.")

    camera = camera.loc[camera["FrameID"] >= reference_frame_id].copy()
    tracking = tracking.loc[tracking["FrameID"] >= reference_frame_id].copy()
    tracking_errors = has_legacy_tracking_errors(tracking, config)
    if tracking_errors:
        raise ValueError("Legacy tracking-error rule rejects this recording.")
    merged = synchronize_legacy(tracking, camera)
    del camera, tracking
    gc.collect()
    stage_counts["merged"] = len(merged)
    data = interpolate_legacy(
        merged,
        config.expected_framerate_hz,
        predicted_framerate,
    )
    del merged
    gc.collect()
    stage_counts["interpolated"] = len(data)
    if data["AbsoluteTime"].max() - protocol["beg (ms)"].max() < 0:
        raise ValueError("Legacy rule reports incomplete acquisition coverage.")

    data = annotate_stimuli_legacy(data, protocol)
    time_step = 1_000 / predicted_framerate
    data["AbsoluteTime"] = (
        data["AbsoluteTime"].iat[0]
        + np.arange(len(data), dtype="float64") * time_step
    )
    data = cumulative_angles_legacy(data)
    data = filter_angles_legacy(data, config)
    stage_counts["filtered"] = len(data)
    data = calculate_vigor_legacy(data, config)
    data = calculate_bout_metric_legacy(data, config)
    stage_counts["bout_metric"] = len(data)
    data[TIME_COLUMN] -= data[TIME_COLUMN].iat[0]
    data = detect_bouts_legacy(data, config)
    data = segment_trials_legacy(data, config)
    stage_counts["trial_samples"] = len(data)
    data = assign_blocks_legacy(data, experiment_name)
    data = scale_vigor_legacy(data, config)
    data = finalize_legacy_samples(data, metadata, config)
    stage_counts["final"] = len(data)

    cs_trials = int(data.loc[data["Trial type"] == "CS", "Trial number"].nunique())
    us_trials = int(data.loc[data["Trial type"] == "US", "Trial number"].nunique())
    summary: dict[str, Any] = {
        "recipe": "legacy-paper-v1",
        "scientific_status": "legacy_reproduction",
        "recording_id": recording_id,
        "recording_name": recording_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_name,
        "config": asdict(config),
        "predicted_framerate_hz": predicted_framerate,
        "reference_frame_id": reference_frame_id,
        "legacy_frame_loss_flag": has_lost_frames,
        "legacy_tracking_error_flag": tracking_errors,
        "stage_row_counts": stage_counts,
        "trial_counts": {"CS": cs_trials, "US": us_trials},
        "bout_fraction": float(data["Bout"].mean()),
        "known_limitations": list(KNOWN_LIMITATIONS),
        "input_artifacts": input_artifacts,
    }

    input_paths = {
        "camera": camera_path,
        "tracking": tracking_path,
        "protocol": protocol_path,
    }
    for kind, path in input_paths.items():
        stat = path.stat()
        if input_file_state[kind] != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"{kind} intake artifact changed during preprocessing.")
        if _sha256_file(path) != input_artifacts[kind]["sha256"]:
            raise RuntimeError(f"{kind} intake artifact hash changed during preprocessing.")

    source_dir.mkdir(parents=True, exist_ok=True)
    with artifact_staging(
        project_dir,
        prefix=f".{recording_id}-legacy-v1-",
    ) as staging_root:
        staged_samples = staging_root / "samples_legacy-v1.parquet"
        staged_summary = staging_root / "legacy-v1_preprocessing_summary.json"
        staged_marker = staging_root / "legacy-v1_complete.json"
        artifact = _write_parquet_atomic(data, staged_samples, overwrite=True)
        artifact["path"] = str(samples_path.resolve())
        summary["artifact"] = artifact
        _write_json_atomic(staged_summary, summary)
        summary_hash = _sha256_file(staged_summary)
        _write_json_atomic(
            staged_marker,
            {
                "status": "complete",
                "recipe": "legacy-paper-v1",
                "recording_id": recording_id,
                "samples_sha256": artifact["sha256"],
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": summary_hash,
            },
        )
        _publish_transaction(
            (
                (staged_samples, samples_path),
                (staged_summary, summary_path),
                (staged_marker, completion_marker_path),
            ),
            staging_root,
            overwrite=overwrite,
        )
    return LegacyPreprocessingResult(
        recording_id=recording_id,
        samples_path=samples_path,
        summary_path=summary_path,
        completion_marker_path=completion_marker_path,
        row_count=len(data),
        cs_trial_count=cs_trials,
        us_trial_count=us_trials,
    )
