"""Behavior-preserving equivalence helpers for legacy preprocessing.

These helpers compare migrated operations without changing scientific
behavior. They never authorize corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from classical_conditioning.ingestion.readers import read_tracking
from classical_conditioning.preprocessing.legacy_v1 import (
    BOUT_METRIC_COLUMN,
    TIME_COLUMN,
    VIGOR_COLUMN,
    ANGLE_TEMPLATE,
    SCALED_VIGOR_COLUMN,
    LegacyPreprocessingConfig,
    annotate_stimuli_legacy,
    assign_blocks_legacy,
    calculate_bout_metric_legacy,
    calculate_vigor_legacy,
    cumulative_angles_legacy,
    detect_bouts_legacy,
    filter_angles_legacy,
    interpolate_legacy,
    prepare_legacy_tracking,
    scale_vigor_legacy,
    segment_trials_legacy,
    synchronize_legacy,
)


@dataclass(frozen=True)
class OperationComparison:
    operation: str
    equal: bool
    detail: str
    left_shape: tuple[int, int] | None
    right_shape: tuple[int, int] | None


def first_dataframe_divergence(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    absolute_tolerance: float = 0.0,
) -> str | None:
    """Return a short description of the first mismatch, or None if equal."""
    if list(left.columns) != list(right.columns):
        return (
            f"columns differ: left={list(left.columns)!r} "
            f"right={list(right.columns)!r}"
        )
    if len(left) != len(right):
        return f"row counts differ: left={len(left)} right={len(right)}"

    left_reset = left.reset_index(drop=True)
    right_reset = right.reset_index(drop=True)
    for column in left_reset.columns:
        left_values = left_reset[column]
        right_values = right_reset[column]
        if pd.api.types.is_bool_dtype(left_values) or pd.api.types.is_bool_dtype(
            right_values
        ):
            left_bool = left_values.to_numpy()
            right_bool = right_values.to_numpy()
            if not np.array_equal(left_bool, right_bool):
                index = int(np.flatnonzero(left_bool != right_bool)[0])
                return (
                    f"column {column!r} differs at row {index}: "
                    f"{left_values.iloc[index]!r} vs {right_values.iloc[index]!r}"
                )
            continue

        if pd.api.types.is_numeric_dtype(left_values) and pd.api.types.is_numeric_dtype(
            right_values
        ):
            left_num = pd.to_numeric(left_values, errors="coerce").to_numpy(dtype=float)
            right_num = pd.to_numeric(right_values, errors="coerce").to_numpy(
                dtype=float
            )
            both_nan = np.isnan(left_num) & np.isnan(right_num)
            close = np.isclose(
                left_num,
                right_num,
                rtol=0.0,
                atol=absolute_tolerance,
                equal_nan=False,
            )
            mismatched = ~(close | both_nan)
            if np.any(mismatched):
                index = int(np.flatnonzero(mismatched)[0])
                return (
                    f"column {column!r} differs at row {index}: "
                    f"{left_num[index]!r} vs {right_num[index]!r}"
                )
            continue

        left_obj = left_values.astype("string")
        right_obj = right_values.astype("string")
        unequal = (left_obj != right_obj) & ~(left_obj.isna() & right_obj.isna())
        if bool(unequal.any()):
            index = int(np.flatnonzero(unequal.to_numpy())[0])
            return (
                f"column {column!r} differs at row {index}: "
                f"{left_values.iloc[index]!r} vs {right_values.iloc[index]!r}"
            )
    return None


def prepare_legacy_tracking_from_raw_txt(
    tracking_path: Path,
    config: LegacyPreprocessingConfig | None = None,
) -> pd.DataFrame:
    """Read raw tracking TXT then apply the frozen legacy preparation.

    The raw reader keeps the trailing acquisition summary row so
    ``prepare_legacy_tracking`` remains the only place that drops it. This
    prevents a double-drop when chaining reader and legacy prep.
    """
    settings = config or LegacyPreprocessingConfig()
    raw = read_tracking(
        tracking_path,
        mode="full",
        drop_trailing_summary_row=False,
        convert_angles_to_degrees=False,
    )
    return prepare_legacy_tracking(raw.frame, settings)


def compare_legacy_tracking_preparation_paths(
    tracking_path: Path,
    config: LegacyPreprocessingConfig | None = None,
) -> OperationComparison:
    """Compare helper prep against direct prepare on the same raw table."""
    settings = config or LegacyPreprocessingConfig()
    via_helper = prepare_legacy_tracking_from_raw_txt(tracking_path, settings)
    raw = read_tracking(
        tracking_path,
        mode="full",
        drop_trailing_summary_row=False,
    )
    direct = prepare_legacy_tracking(raw.frame, settings)
    divergence = first_dataframe_divergence(via_helper, direct)
    return OperationComparison(
        operation="prepare_legacy_tracking_from_raw_txt",
        equal=divergence is None,
        detail=divergence or "exact match",
        left_shape=via_helper.shape,
        right_shape=direct.shape,
    )


def compare_legacy_angles_reader_to_prepare(
    tracking_path: Path,
    config: LegacyPreprocessingConfig | None = None,
) -> OperationComparison:
    """Compare legacy-angles reader conversion with prepare_legacy_tracking."""
    settings = config or LegacyPreprocessingConfig()
    prepared = prepare_legacy_tracking_from_raw_txt(tracking_path, settings)
    legacy_reader = read_tracking(
        tracking_path,
        mode="legacy_angles",
        drop_trailing_summary_row=True,
        convert_angles_to_degrees=True,
        legacy_angle_point_count=settings.angle_point_count,
    ).frame.copy()
    legacy_reader["Original frame number"] = legacy_reader["FrameID"].astype("int32")
    angle_columns = [
        f"Angle of point {index} (deg)"
        for index in range(settings.angle_point_count)
    ]
    legacy_reader = legacy_reader.loc[
        :,
        ["FrameID", *angle_columns, "Original frame number"],
    ]
    prepared = prepared.loc[:, list(legacy_reader.columns)]
    divergence = first_dataframe_divergence(
        prepared,
        legacy_reader,
        absolute_tolerance=1e-5,
    )
    return OperationComparison(
        operation="legacy_angles_reader_vs_prepare_legacy_tracking",
        equal=divergence is None,
        detail=divergence or "exact match within absolute_tolerance=1e-5",
        left_shape=prepared.shape,
        right_shape=legacy_reader.shape,
    )


def compare_operation_pair(
    operation: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    absolute_tolerance: float = 0.0,
) -> OperationComparison:
    divergence = first_dataframe_divergence(
        left,
        right,
        absolute_tolerance=absolute_tolerance,
    )
    return OperationComparison(
        operation=operation,
        equal=divergence is None,
        detail=divergence or "exact match",
        left_shape=left.shape,
        right_shape=right.shape,
    )


def analysis_utils_reference_steps(
    config: LegacyPreprocessingConfig,
) -> dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
    """Build reference callables from the current analysis_utils contracts."""
    import analysis_utils

    tail_column = ANGLE_TEMPLATE.format(index=config.angle_point_count - 1)

    def cumulative(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        angle_columns = [
            column
            for column in result.columns
            if column.startswith("Angle of point")
        ]
        result[angle_columns] = result[angle_columns].cumsum(axis=1)
        return result

    def filter_angles(data: pd.DataFrame) -> pd.DataFrame:
        return analysis_utils.filter_data(
            data.copy(),
            space_window=3,
            time_window=config.temporal_filter_frames,
        )

    def vigor(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result[VIGOR_COLUMN] = analysis_utils.calculate_vigor_fast_pure_numpy(
            result[tail_column].to_numpy(),
            config.expected_framerate_hz,
        ).astype("float32")
        return result

    def bout_metric(data: pd.DataFrame) -> pd.DataFrame:
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

    def detect_bouts(data: pd.DataFrame) -> pd.DataFrame:
        return analysis_utils.find_beg_and_end_of_bouts(
            data.copy(),
            thr1=config.bout_threshold_primary_deg_per_ms,
            min_dur=config.minimum_bout_duration_frames,
            min_gap=config.minimum_interbout_frames,
            thr2=config.bout_threshold_secondary_deg_per_ms,
        )

    return {
        "cumulative_angles_legacy": cumulative,
        "filter_angles_legacy": filter_angles,
        "calculate_vigor_legacy": vigor,
        "calculate_bout_metric_legacy": bout_metric,
        "detect_bouts_legacy": detect_bouts,
    }


def compare_synchronize_and_interpolate(
    tracking: pd.DataFrame,
    camera: pd.DataFrame,
    *,
    expected_framerate: float,
    predicted_framerate: float,
) -> list[OperationComparison]:
    """Compare synchronization and interpolation against analysis_utils."""
    import analysis_utils

    sync_left = synchronize_legacy(tracking, camera)
    sync_right = analysis_utils.merge_camera_with_data(tracking.copy(), camera.copy())
    comparisons = [
        compare_operation_pair("synchronize_legacy", sync_left, sync_right),
    ]
    if not comparisons[0].equal:
        return comparisons

    interp_left = interpolate_legacy(
        sync_left,
        expected_framerate,
        predicted_framerate,
    )
    interp_right = analysis_utils.interpolate_data(
        sync_left.copy(),
        expected_framerate,
        predicted_framerate,
    )
    comparisons.append(
        compare_operation_pair(
            "interpolate_legacy",
            interp_left,
            interp_right,
            absolute_tolerance=1e-9,
        )
    )
    return comparisons


def compare_stimulus_annotation(
    data: pd.DataFrame,
    protocol: pd.DataFrame,
) -> OperationComparison:
    """Compare migrated stimulus markers with analysis_utils.stim_in_data."""
    import analysis_utils

    left = annotate_stimuli_legacy(data, protocol)
    right = analysis_utils.stim_in_data(data.copy(), protocol)
    columns = ["CS beg", "CS end", "US beg", "US end"]
    return compare_operation_pair(
        "annotate_stimuli_legacy",
        left.loc[:, columns],
        right.loc[:, columns],
    )


def compare_trial_segmentation(
    data: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> OperationComparison:
    """Compare migrated trial windows with analysis_utils.identify_trials."""
    import analysis_utils

    left = segment_trials_legacy(data, config)
    right = analysis_utils.identify_trials(
        data.copy(),
        config.trial_start_frames,
        config.trial_end_frames,
    )
    # Categories can differ in ordering metadata; compare as strings/ints.
    left_cmp = left.copy()
    right_cmp = right.copy()
    left_cmp["Trial type"] = left_cmp["Trial type"].astype(str)
    right_cmp["Trial type"] = right_cmp["Trial type"].astype(str)
    shared = [
        column
        for column in left_cmp.columns
        if column in right_cmp.columns
    ]
    return compare_operation_pair(
        "segment_trials_legacy",
        left_cmp.loc[:, shared].reset_index(drop=True),
        right_cmp.loc[:, shared].reset_index(drop=True),
    )


def compare_block_assignment(
    segmented: pd.DataFrame,
    *,
    experiment_name: str = "allDelay",
) -> OperationComparison:
    """Compare migrated block names against the experiment trial map."""
    from classical_conditioning.config import get_trial_block_lookup

    assigned = assign_blocks_legacy(segmented, experiment_name)
    lookup = get_trial_block_lookup(experiment_name)
    expected = []
    for trial_type, trial_number in zip(
        assigned["Trial type"].astype(str),
        assigned["Trial number"].astype(int),
        strict=True,
    ):
        expected.append(lookup.get((trial_type, int(trial_number)), ""))
    actual = assigned["Block name"].astype(str).tolist()
    equal = actual == expected
    detail = "exact match"
    if not equal:
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            if left != right:
                detail = (
                    f"Block name differs at row {index}: {left!r} vs {right!r}"
                )
                break
    return OperationComparison(
        operation="assign_blocks_legacy",
        equal=equal,
        detail=detail,
        left_shape=assigned.shape,
        right_shape=assigned.shape,
    )


def compare_scaled_vigor_roundtrip(
    segmented: pd.DataFrame,
    config: LegacyPreprocessingConfig,
) -> OperationComparison:
    """Check scaled vigor is finite on trials with usable baselines."""
    scaled = scale_vigor_legacy(segmented, config)
    if SCALED_VIGOR_COLUMN not in scaled.columns:
        return OperationComparison(
            operation="scale_vigor_legacy",
            equal=False,
            detail="missing scaled vigor column",
            left_shape=scaled.shape,
            right_shape=None,
        )
    trial_ids = scaled["Trial number"].unique()
    usable = 0
    for trial_id in trial_ids:
        trial = scaled.loc[scaled["Trial number"] == trial_id]
        baseline = trial.loc[
            trial[TIME_COLUMN] < -config.baseline_window_frames,
            VIGOR_COLUMN,
        ]
        if baseline.notna().any() and float(baseline.quantile(0.1)) != float(
            baseline.quantile(0.9)
        ):
            usable += 1
            if not np.isfinite(trial[SCALED_VIGOR_COLUMN]).any():
                return OperationComparison(
                    operation="scale_vigor_legacy",
                    equal=False,
                    detail=f"no finite scaled vigor for trial {trial_id}",
                    left_shape=scaled.shape,
                    right_shape=scaled.shape,
                )
    return OperationComparison(
        operation="scale_vigor_legacy",
        equal=usable > 0,
        detail=(
            f"finite scaled vigor on {usable} trial(s)"
            if usable
            else "no trial had a usable baseline"
        ),
        left_shape=scaled.shape,
        right_shape=scaled.shape,
    )


def run_legacy_angle_operation_chain(
    frame: pd.DataFrame,
    config: LegacyPreprocessingConfig,
    reference_steps: dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
) -> list[OperationComparison]:
    """Compare migrated angle operations against reference callables in order."""
    comparisons: list[OperationComparison] = []
    current = frame
    pipeline: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
        ("cumulative_angles_legacy", cumulative_angles_legacy),
        (
            "filter_angles_legacy",
            lambda data: filter_angles_legacy(data, config),
        ),
        (
            "calculate_vigor_legacy",
            lambda data: calculate_vigor_legacy(data, config),
        ),
        (
            "calculate_bout_metric_legacy",
            lambda data: calculate_bout_metric_legacy(data, config),
        ),
        (
            "detect_bouts_legacy",
            lambda data: detect_bouts_legacy(data, config),
        ),
    ]
    for name, function in pipeline:
        migrated = function(current)
        if name not in reference_steps:
            comparisons.append(
                OperationComparison(
                    operation=name,
                    equal=False,
                    detail="missing reference step",
                    left_shape=migrated.shape,
                    right_shape=None,
                )
            )
            break
        reference = reference_steps[name](current)
        comparison = compare_operation_pair(name, migrated, reference)
        comparisons.append(comparison)
        if not comparison.equal:
            break
        current = migrated
    return comparisons


def operation_comparisons_to_dict(
    comparisons: list[OperationComparison],
) -> dict[str, Any]:
    first_failure = next(
        (item.operation for item in comparisons if not item.equal),
        None,
    )
    return {
        "artifact_kind": "legacy-operation-equivalence-v1",
        "all_equal": all(item.equal for item in comparisons),
        "first_divergence_operation": first_failure,
        "operations": [
            {
                "operation": item.operation,
                "equal": item.equal,
                "detail": item.detail,
                "left_shape": list(item.left_shape) if item.left_shape else None,
                "right_shape": list(item.right_shape) if item.right_shape else None,
            }
            for item in comparisons
        ],
    }
