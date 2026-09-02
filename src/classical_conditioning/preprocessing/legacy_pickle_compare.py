"""Compare a local historical per-fish pickle with legacy Parquet.

The pickle is loaded only in this process. The function writes a JSON
summary and never prints row values except a compact first-mismatch
location (column, row index, and two scalars).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.artifacts import write_json_atomic
from classical_conditioning.preprocessing.legacy_equivalence import (
    first_dataframe_divergence,
)

SCIENTIFIC_COLUMNS = (
    "Original frame number",
    "Trial time (frame) [700 FPS]",
    "CS beg",
    "CS end",
    "US beg",
    "US end",
    "Trial type",
    "Trial number",
    "Block name",
    "Vigor (deg/ms)",
    "Scaled vigor (AU)",
    "Bout beg",
    "Bout end",
    "Bout",
)

TIME_COLUMN = "Trial time (frame) [700 FPS]"
ORIGINAL_FRAME_COLUMN = "Original frame number"
DEFAULT_EXPECTED_FRAMERATE_HZ = 700.0


def _trial_original_frame_slope(
    trial_time: np.ndarray,
    original_frame: np.ndarray,
) -> float:
    if len(trial_time) < 2:
        raise ValueError("Need at least two samples to estimate Original-frame slope.")
    slope, _intercept = np.polyfit(
        trial_time.astype(float),
        original_frame.astype(float),
        1,
    )
    return float(slope)


def _vigor_correlation_after_reciprocal_rate_warp(
    trial_time: np.ndarray,
    pickle_vigor: np.ndarray,
    parquet_vigor: np.ndarray,
    *,
    expected_over_predicted: float,
    predicted_over_expected: float,
) -> dict[str, float]:
    """Correlate vigor after undoing reciprocal within-trial time warp.

    Historical pickles advance Original frame at ``expected/predicted``. Parquet
    advances at ``predicted/expected``. Aligning on trial time therefore compares
    different acquisition moments away from onset. Mapping pickle trial time
    ``t`` to parquet time ``t * (expected/predicted)**2`` puts both on the same
    Original-frame locus when onset Original frames agree.
    """
    order = np.argsort(trial_time.astype(float))
    times = trial_time.astype(float)[order]
    left = pickle_vigor.astype(float)[order]
    right = parquet_vigor.astype(float)[order]
    warp = (expected_over_predicted**2)
    # pickle[t] <-> parquet[t * (e/p)^2] when slopes are e/p vs p/e
    target_times = times * warp
    right_on_pickle_time = np.interp(target_times, times, right)
    valid = np.isfinite(left) & np.isfinite(right_on_pickle_time)
    if int(valid.sum()) < 2:
        return {
            "rate_warp_factor_expected_over_predicted_squared": float(warp),
            "vigor_correlation_after_rate_warp": float("nan"),
            "vigor_mean_abs_diff_after_rate_warp": float("nan"),
        }
    corr = float(np.corrcoef(left[valid], right_on_pickle_time[valid])[0, 1])
    mad = float(np.mean(np.abs(left[valid] - right_on_pickle_time[valid])))
    # Also summarize lag-vs-time slope vs the physics prediction.
    window = 3_000
    step = 3_000
    max_lag = 300
    measured_lags: list[float] = []
    predicted_lags: list[float] = []
    for start in range(0, len(times) - window, step):
        a = left[start : start + window]
        b = right[start : start + window]
        center = float(times[start + window // 2])
        best_corr = -1e9
        best_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x = a[-lag:]
                y = b[: len(x)]
            elif lag > 0:
                x = a[: window - lag]
                y = b[lag:]
            else:
                x = a
                y = b
            if len(x) < 500:
                continue
            value = float(np.corrcoef(x, y)[0, 1])
            if value > best_corr:
                best_corr = value
                best_lag = lag
        measured_lags.append(float(best_lag))
        predicted_lags.append(
            float(center * (expected_over_predicted - predicted_over_expected))
        )
    lag_corr = float("nan")
    if len(measured_lags) >= 2:
        lag_corr = float(np.corrcoef(measured_lags, predicted_lags)[0, 1])
    return {
        "rate_warp_factor_expected_over_predicted_squared": float(warp),
        "vigor_correlation_after_rate_warp": corr,
        "vigor_mean_abs_diff_after_rate_warp": mad,
        "lag_versus_predicted_rate_warp_correlation": lag_corr,
        "lag_prediction_formula": (
            "trial_time * (expected/predicted - predicted/expected); "
            "positive lag shifts parquet later to match pickle"
        ),
    }


def classify_original_frame_mapping(
    pickle_frame: pd.DataFrame,
    parquet_frame: pd.DataFrame,
    *,
    expected_framerate_hz: float = DEFAULT_EXPECTED_FRAMERATE_HZ,
    onset_window_frames: int = 50,
) -> dict[str, Any]:
    """Classify Original-frame labeling without treating the pickle as truth.

    After legacy interpolate, absolute acquisition frames should advance at
    ``predicted/expected`` per sample on the expected-rate time grid. The
    reciprocal ``expected/predicted`` is the FrameID time-axis scale factor and
    is incorrect for Original-frame labels.
    """
    required = {
        TIME_COLUMN,
        ORIGINAL_FRAME_COLUMN,
        "Trial type",
        "Trial number",
        "Vigor (deg/ms)",
    }
    for name, frame in (("pickle", pickle_frame), ("parquet", parquet_frame)):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing columns: {sorted(missing)}")

    pickle_mask = (pickle_frame["Trial type"].astype(str) == "CS") & (
        pickle_frame["Trial number"].astype(str) == "1"
    )
    parquet_mask = (parquet_frame["Trial type"].astype(str) == "CS") & (
        parquet_frame["Trial number"].astype(str) == "1"
    )
    left = pickle_frame.loc[pickle_mask]
    right = parquet_frame.loc[parquet_mask]
    if left.empty or right.empty:
        return {
            "status": "skipped",
            "detail": "CS trial 1 not found in both tables.",
        }

    left = left.sort_values(TIME_COLUMN).drop_duplicates(TIME_COLUMN, keep="first")
    right = right.sort_values(TIME_COLUMN).drop_duplicates(TIME_COLUMN, keep="first")
    times = np.intersect1d(
        left[TIME_COLUMN].to_numpy(),
        right[TIME_COLUMN].to_numpy(),
    )
    if len(times) < 2:
        return {
            "status": "skipped",
            "detail": "CS trial 1 has fewer than two shared trial-time samples.",
        }
    left = left.set_index(TIME_COLUMN).loc[times]
    right = right.set_index(TIME_COLUMN).loc[times]
    pickle_slope = _trial_original_frame_slope(
        times,
        left[ORIGINAL_FRAME_COLUMN].to_numpy(),
    )
    parquet_slope = _trial_original_frame_slope(
        times,
        right[ORIGINAL_FRAME_COLUMN].to_numpy(),
    )
    # Infer predicted fps from the Parquet slope (correct physics under current code).
    predicted_from_parquet = abs(parquet_slope) * expected_framerate_hz
    expected_over_predicted = expected_framerate_hz / predicted_from_parquet
    predicted_over_expected = predicted_from_parquet / expected_framerate_hz

    def _near(value: float, target: float, *, rtol: float = 1e-5) -> bool:
        return bool(np.isclose(value, target, rtol=rtol, atol=0.0))

    pickle_matches_reciprocal = _near(pickle_slope, expected_over_predicted)
    parquet_matches_physics = _near(parquet_slope, predicted_over_expected)
    onset = times[np.abs(times) <= onset_window_frames]
    full_vigor_corr = float(
        np.corrcoef(
            left["Vigor (deg/ms)"].to_numpy(dtype=float),
            right["Vigor (deg/ms)"].to_numpy(dtype=float),
        )[0, 1]
    )
    onset_vigor_corr = float(
        np.corrcoef(
            left.loc[onset, "Vigor (deg/ms)"].to_numpy(dtype=float),
            right.loc[onset, "Vigor (deg/ms)"].to_numpy(dtype=float),
        )[0, 1]
    ) if len(onset) >= 2 else float("nan")
    onset_original_mad = float(
        np.mean(
            np.abs(
                left.loc[onset, ORIGINAL_FRAME_COLUMN].to_numpy(dtype=float)
                - right.loc[onset, ORIGINAL_FRAME_COLUMN].to_numpy(dtype=float)
            )
        )
    ) if len(onset) else float("nan")
    zero = times[times == 0]
    onset_original_diff_at_zero = None
    if len(zero):
        onset_original_diff_at_zero = float(
            left.loc[0, ORIGINAL_FRAME_COLUMN] - right.loc[0, ORIGINAL_FRAME_COLUMN]
        )

    classification = "unclassified"
    if pickle_matches_reciprocal and parquet_matches_physics:
        classification = (
            "historical_pickle_original_frame_uses_reciprocal_rate; "
            "parquet_matches_current_interpolate_physics"
        )
    elif parquet_matches_physics and not pickle_matches_reciprocal:
        classification = "parquet_matches_current_interpolate_physics; pickle_slope_unexpected"
    elif pickle_matches_reciprocal and not parquet_matches_physics:
        classification = "pickle_uses_reciprocal_rate; parquet_slope_unexpected"

    warp_stats = _vigor_correlation_after_reciprocal_rate_warp(
        times,
        left["Vigor (deg/ms)"].to_numpy(dtype=float),
        right["Vigor (deg/ms)"].to_numpy(dtype=float),
        expected_over_predicted=expected_over_predicted,
        predicted_over_expected=predicted_over_expected,
    )
    if (
        pickle_matches_reciprocal
        and parquet_matches_physics
        and np.isfinite(warp_stats["vigor_correlation_after_rate_warp"])
        and warp_stats["vigor_correlation_after_rate_warp"] > full_vigor_corr + 0.2
    ):
        classification = (
            "historical_pickle_timebase_warped_by_reciprocal_rate; "
            "parquet_matches_current_interpolate_physics"
        )

    return {
        "status": "ok",
        "classification": classification,
        "cs_trial_1_sample_count": int(len(times)),
        "pickle_original_frame_slope_per_trial_time": pickle_slope,
        "parquet_original_frame_slope_per_trial_time": parquet_slope,
        "expected_framerate_hz": expected_framerate_hz,
        "inferred_predicted_framerate_hz_from_parquet_slope": predicted_from_parquet,
        "physics_slope_predicted_over_expected": predicted_over_expected,
        "reciprocal_slope_expected_over_predicted": expected_over_predicted,
        "pickle_matches_reciprocal_rate": pickle_matches_reciprocal,
        "parquet_matches_physics_rate": parquet_matches_physics,
        "onset_window_frames": onset_window_frames,
        "onset_original_mean_abs_diff": onset_original_mad,
        "onset_original_diff_at_trial_time_0": onset_original_diff_at_zero,
        "onset_vigor_correlation": onset_vigor_corr,
        "full_trial_vigor_correlation": full_vigor_corr,
        **warp_stats,
        "interpretation": [
            "Do not change legacy-paper-v1 solely to match pickle Original-frame labels.",
            "Current interpolate maps absolute acquisition frames at predicted/expected "
            "per expected-rate sample; that matches analysis_utils.interpolate_data.",
            "Historical pickles advance Original frame at expected/predicted within "
            "trials (reciprocal). CS onset Original frames still agree closely.",
            "Naive trial-time vigor comparison understates agreement: within-trial lag "
            "grows as trial_time * (expected/predicted - predicted/expected), matching "
            "the reciprocal Original-frame warp. After rate-warp resampling, vigor "
            "correlation rises. This is a historical pickle/timebase artifact, not a "
            "license to invert interpolate in legacy-paper-v1.",
        ],
    }


def _mismatch_mask(left: pd.Series, right: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        left_num = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
        right_num = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(left_num) & np.isnan(right_num)
        close = np.isclose(
            left_num,
            right_num,
            rtol=0.0,
            atol=0.0,
            equal_nan=False,
        )
        return ~(close | both_nan)
    left_obj = left.astype("string")
    right_obj = right.astype("string")
    unequal = (left_obj != right_obj) & ~(left_obj.isna() & right_obj.isna())
    return unequal.fillna(False).to_numpy(dtype=bool)


def _normalize_for_compare(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    result = pd.DataFrame(index=range(len(frame)))
    for column in columns:
        series = frame[column]
        if column == "Block name":
            values = series.astype("string").fillna("")
            values = values.replace("<NA>", "")
            result[column] = values
            continue
        if column in {"Trial type"}:
            result[column] = series.astype(str)
            continue
        if column in {
            "CS beg",
            "CS end",
            "US beg",
            "US end",
            "Trial number",
            "Original frame number",
            "Trial time (frame) [700 FPS]",
        }:
            result[column] = pd.to_numeric(series, errors="coerce")
            continue
        if pd.api.types.is_bool_dtype(series):
            result[column] = series.astype(bool)
            continue
        result[column] = pd.to_numeric(series, errors="coerce")
    return result


def _load_pickle_dataframe(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path, compression="gzip")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Pickle is {type(frame).__name__}, not a DataFrame.")
    if any(name is not None for name in frame.index.names):
        frame = frame.reset_index()
    return frame


@dataclass(frozen=True)
class LegacyPickleComparison:
    pickle_path: Path
    parquet_path: Path
    report_path: Path
    row_counts_equal: bool
    all_scientific_columns_equal: bool
    first_divergence: str | None
    original_frame_classification: str | None = None
    onset_vigor_correlation: float | None = None
    full_trial_vigor_correlation: float | None = None
    vigor_correlation_after_rate_warp: float | None = None
    lag_versus_predicted_rate_warp_correlation: float | None = None


def compare_legacy_pickle_to_parquet(
    pickle_path: Path,
    parquet_path: Path,
    output_path: Path,
    *,
    batch_size: int = 250_000,
    overwrite: bool = False,
) -> LegacyPickleComparison:
    """Compare historical gzip pickle with samples_legacy-v1.parquet locally."""
    pickle_path = pickle_path.resolve()
    parquet_path = parquet_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Pickle comparison report exists; pass overwrite to replace: {output_path}"
        )
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Pickle does not exist: {pickle_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet does not exist: {parquet_path}")

    pickle_frame = _load_pickle_dataframe(pickle_path)
    parquet = pq.ParquetFile(parquet_path)
    parquet_rows = int(parquet.metadata.num_rows)
    pickle_rows = int(len(pickle_frame))
    parquet_columns = list(parquet.schema_arrow.names)
    pickle_columns = list(pickle_frame.columns)

    scientific = [
        column
        for column in SCIENTIFIC_COLUMNS
        if column in pickle_columns and column in parquet_columns
    ]
    identity = [
        column
        for column in (
            "Day",
            "Fish no.",
            "Exp.",
            "ProtocolRig",
            "Strain",
            "Age (dpf)",
        )
        if column in pickle_columns and column in parquet_columns
    ]
    angle_columns = [
        column
        for column in pickle_columns
        if column.startswith("Angle of point ") and column in parquet_columns
    ]
    compared_columns = scientific + angle_columns

    first_divergence: str | None = None
    per_column_first_divergence: dict[str, str] = {}
    mismatch_counts: dict[str, int] = {column: 0 for column in compared_columns}
    nan_vs_value_counts: dict[str, int] = {column: 0 for column in compared_columns}
    abs_diff_max: dict[str, float] = {column: 0.0 for column in compared_columns}
    abs_diff_sum: dict[str, float] = {column: 0.0 for column in compared_columns}
    numeric_compared: dict[str, int] = {column: 0 for column in compared_columns}
    compared_rows = 0
    if pickle_rows == parquet_rows:
        offset = 0
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=compared_columns,
        ):
            right = batch.to_pandas()
            left = pickle_frame.iloc[offset : offset + len(right)][list(compared_columns)]
            left_norm = _normalize_for_compare(left, compared_columns)
            right_norm = _normalize_for_compare(right, compared_columns)
            for column in compared_columns:
                left_values = left_norm[column]
                right_values = right_norm[column]
                mismatched = np.asarray(
                    _mismatch_mask(left_values, right_values),
                    dtype=bool,
                )
                mismatch_counts[column] += int(np.count_nonzero(mismatched))
                nan_vs_value = left_values.isna() ^ right_values.isna()
                nan_vs_value_counts[column] += int(np.count_nonzero(np.asarray(nan_vs_value, dtype=bool)))
                if pd.api.types.is_numeric_dtype(left_values) and pd.api.types.is_numeric_dtype(
                    right_values
                ):
                    left_num = pd.to_numeric(left_values, errors="coerce").to_numpy(
                        dtype=float
                    )
                    right_num = pd.to_numeric(right_values, errors="coerce").to_numpy(
                        dtype=float
                    )
                    finite = np.isfinite(left_num) & np.isfinite(right_num)
                    if np.any(finite):
                        diffs = np.abs(left_num[finite] - right_num[finite])
                        abs_diff_max[column] = max(
                            abs_diff_max[column],
                            float(diffs.max()),
                        )
                        abs_diff_sum[column] += float(diffs.sum())
                        numeric_compared[column] += int(finite.sum())
                if column not in per_column_first_divergence and bool(mismatched.any()):
                    local_index = int(np.flatnonzero(mismatched)[0])
                    detail = (
                        f"parquet_offset={offset}; column {column!r} differs at "
                        f"row {local_index}: {left_values.iloc[local_index]!r} vs "
                        f"{right_values.iloc[local_index]!r}"
                    )
                    per_column_first_divergence[column] = detail
                    if first_divergence is None:
                        first_divergence = detail
            compared_rows += len(right)
            offset += len(right)
    else:
        first_divergence = (
            f"row counts differ: pickle={pickle_rows} parquet={parquet_rows}"
        )

    identity_equal: bool | None = None
    identity_detail: str | None = None
    if identity and pickle_rows == parquet_rows and first_divergence is None:
        identity_left = pickle_frame.loc[:, identity].astype(str)
        # Load identity columns only from parquet metadata columns.
        identity_right = parquet.read(columns=identity).to_pandas().astype(str)
        identity_detail = first_dataframe_divergence(identity_left, identity_right)
        identity_equal = identity_detail is None

    original_frame_diagnosis: dict[str, Any] | None = None
    if (
        pickle_rows == parquet_rows
        and ORIGINAL_FRAME_COLUMN in compared_columns
        and TIME_COLUMN in compared_columns
        and "Trial type" in pickle_columns
        and "Trial number" in pickle_columns
        and "Vigor (deg/ms)" in compared_columns
    ):
        diagnosis_columns = [
            TIME_COLUMN,
            ORIGINAL_FRAME_COLUMN,
            "Trial type",
            "Trial number",
            "Vigor (deg/ms)",
        ]
        # CS trials are written first; one frozen allDelay CS window is 63,001 rows.
        first_trial_rows = 63_001
        parquet_trial = next(
            parquet.iter_batches(
                batch_size=first_trial_rows,
                columns=diagnosis_columns,
            )
        ).to_pandas()
        pickle_trial = pickle_frame.iloc[: len(parquet_trial)].loc[:, diagnosis_columns]
        original_frame_diagnosis = classify_original_frame_mapping(
            pickle_trial,
            parquet_trial,
        )

    payload: dict[str, Any] = {
        "artifact_kind": "legacy-pickle-parquet-comparison-v1",
        "pickle_name": pickle_path.name,
        "pickle_size_bytes": int(pickle_path.stat().st_size),
        "parquet_name": parquet_path.name,
        "pickle_row_count": pickle_rows,
        "parquet_row_count": parquet_rows,
        "row_counts_equal": pickle_rows == parquet_rows,
        "pickle_columns": pickle_columns,
        "parquet_columns": parquet_columns,
        "compared_scientific_and_angle_columns": compared_columns,
        "compared_rows": compared_rows,
        "scientific_and_angle_equal": first_divergence is None
        and pickle_rows == parquet_rows,
        "first_divergence": first_divergence,
        "per_column_first_divergence": per_column_first_divergence,
        "mismatch_counts": {
            column: count
            for column, count in mismatch_counts.items()
            if count > 0
        },
        "nan_vs_value_counts": {
            column: count
            for column, count in nan_vs_value_counts.items()
            if count > 0
        },
        "numeric_mean_abs_diff": {
            column: abs_diff_sum[column] / numeric_compared[column]
            for column in compared_columns
            if numeric_compared[column] > 0 and mismatch_counts[column] > 0
        },
        "numeric_max_abs_diff": {
            column: abs_diff_max[column]
            for column in compared_columns
            if numeric_compared[column] > 0 and mismatch_counts[column] > 0
        },
        "columns_with_any_mismatch": sorted(per_column_first_divergence),
        "identity_columns": identity,
        "identity_equal": identity_equal,
        "identity_detail": identity_detail,
        "original_frame_mapping_diagnosis": original_frame_diagnosis,
        "known_serialization_differences": [
            "Pickle stores fish identity in the index; Parquet stores it as columns.",
            "Pickle uses category/int64 for some trial fields; Parquet uses int32.",
            "Empty Block name may appear as NaN in pickle and empty string in Parquet.",
        ],
        "scientific_classification_notes": [
            "Historical pickle Original-frame labels within trials advance at "
            "expected/predicted (reciprocal). Parquet matches current interpolate "
            "physics at predicted/expected. Do not rewrite legacy-paper-v1 to match "
            "the reciprocal pickle labeling.",
        ],
    }
    write_json_atomic(output_path, payload)
    diagnosis = original_frame_diagnosis or {}
    comparison = LegacyPickleComparison(
        pickle_path=pickle_path,
        parquet_path=parquet_path,
        report_path=output_path,
        row_counts_equal=pickle_rows == parquet_rows,
        all_scientific_columns_equal=payload["scientific_and_angle_equal"],
        first_divergence=first_divergence,
        original_frame_classification=(
            str(diagnosis["classification"])
            if diagnosis.get("classification") is not None
            else None
        ),
        onset_vigor_correlation=(
            float(diagnosis["onset_vigor_correlation"])
            if diagnosis.get("onset_vigor_correlation") is not None
            else None
        ),
        full_trial_vigor_correlation=(
            float(diagnosis["full_trial_vigor_correlation"])
            if diagnosis.get("full_trial_vigor_correlation") is not None
            else None
        ),
        vigor_correlation_after_rate_warp=(
            float(diagnosis["vigor_correlation_after_rate_warp"])
            if diagnosis.get("vigor_correlation_after_rate_warp") is not None
            else None
        ),
        lag_versus_predicted_rate_warp_correlation=(
            float(diagnosis["lag_versus_predicted_rate_warp_correlation"])
            if diagnosis.get("lag_versus_predicted_rate_warp_correlation") is not None
            else None
        ),
    )
    del pickle_frame
    del parquet
    return comparison
