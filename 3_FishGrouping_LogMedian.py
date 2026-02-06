"""
Fish Grouping Pipeline — Log-Median Variant
=============================================

Pipeline context — Step 3 of 6 (log-median variant)
-----------------------------------------------------
This script is a standalone variant of ``3_FishGrouping.py``.  It bridges
individual-animal preprocessing (Step 1) and group-level visualization /
statistics (Steps 4–6) by aggregating all per-fish DataFrames into pooled,
per-condition datasets that are activity-filtered, smoothed, downsampled,
log-transformed, and baseline-normalised — ready for the scaled-vigor
heatmaps (Step 4), normalized-vigor analysis (Step 5), and learner
classification (Step 6).

Differences from the standard pipeline
---------------------------------------
1. **Rolling median** instead of rolling mean for vigor smoothing.
2. **Downsampling immediately after smoothing** (before any scaling).
3. **No within-trial P10–P90 baseline scaling** — removed entirely.
4. **Log transform** of vigor values after downsampling.
5. **Baseline median subtraction** — per (Fish, Trial), the median of
   log-vigor in the pre-stimulus baseline window is subtracted from all
   time points, yielding a baseline-centred log-vigor metric stored in
   the ``Scaled vigor (AU)`` column.

Workflow
--------
1. **Load and validate** — Scan the processed-data directory for per-fish
   pickle files (``*.pkl.gz``), skip any in the ``Excluded/`` subdirectory or
   with missing required columns.

2. **Activity filtering (bout masking)** — For every trial, set vigor to NaN
   outside detected swim bouts so that quiescent baseline periods do not
   inflate group averages.

3. **Vigor smoothing** — Apply a right-aligned rolling median (window = 10
   frames, with optional Numba acceleration) to suppress high-frequency
   noise in the vigor signal.

4. **Downsampling** — Keep every *N*-th row (default *N* = 10) to reduce file
   size while retaining sufficient temporal resolution for downstream
   time-binned analyses.

5. **Log transform** — Compute ``log(vigor)`` for all positive vigor values;
   zeros and negatives become NaN.

6. **Baseline median subtraction** — For each (Fish, Trial), compute the
   median of log-vigor in the pre-stimulus baseline window and subtract it
   from all time points.  The result is stored in ``Scaled vigor (AU)``.

7. **Concatenation and harmonization** — Merge all fish DataFrames within each
   experimental condition, standardize categorical columns (block names, trial
   types), and ensure consistent dtypes across fish.

8. **Saving** — Write one compressed pickle per condition × trial type
   (e.g., ``Paired_CS_new_logmedian.pkl``, ``Paired_US_new_logmedian.pkl``),
   verifying column consistency before saving.

Inputs
------
- Per-fish compressed pickles (``{fish_id}.pkl``) from Step 1.
- Experiment configuration (condition types, block structure, timing
  parameters) from ``experiment_configuration``.

Outputs
-------
- Pooled per-condition pickles in ``path_all_fish``:
  ``{condition}_CS_new_logmedian.pkl`` and ``{condition}_US_new_logmedian.pkl``.
  Each contains a single DataFrame with harmonized, activity-filtered,
  median-smoothed, downsampled, log-transformed, and baseline-subtracted data
  for all included fish, suitable for group-level plotting and statistical
  analyses in Steps 4–6.
"""


# %%
# region Imports
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from tqdm import tqdm

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

import analysis_utils
import file_utils
import plotting_style
from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as gen_config

pd.set_option("mode.copy_on_write", True)
plotting_style.set_plot_style()
# endregion Imports

config = get_experiment_config(ExperimentType.ALL_DELAY.value)

# region Parameters
# ------------------------------------------------------------------------------
# Processing Parameters
# ------------------------------------------------------------------------------
WINDOW_SIZE = 10
DOWNSAMPLE_FACTOR = 10
TIME_COL = gen_config.time_trial_frame_label
TAIL_ANGLE_LABEL = gen_config.tail_angle_label
FISH_SUBSET: list[str] | None = None

# ------------------------------------------------------------------------------
# Column Labels
# ------------------------------------------------------------------------------
DAY_COL = "Day"
FISH_COL = "Fish"
FISH_NO_COL = "Fish no."
EXP_COL = "Exp."
TRIAL_TYPE_COL = "Trial type"
TRIAL_NUMBER_COL = "Trial number"
BLOCK_NAME_COL = "Block name"
BOUT_COL = "Bout"
BOUT_BEG_COL = "Bout beg"
BOUT_END_COL = "Bout end"
VIGOR_COL = "Vigor (deg/ms)"
SCALED_VIGOR_COL = "Scaled vigor (AU)"
CS_BEG_COL = "CS beg"
CS_END_COL = "CS end"
US_BEG_COL = "US beg"
US_END_COL = "US end"
TRIAL_TIME_SECONDS_COL = "Trial time (s)"
CS_LABEL = "CS"

GROUPING_COLS = [FISH_COL, TRIAL_NUMBER_COL]

NEEDED_BASE_COLS = [
    "Strain",
    "Age (dpf)",
    EXP_COL,
    "ProtocolRig",
    DAY_COL,
    FISH_NO_COL,
    FISH_COL,
    TIME_COL,
    CS_BEG_COL,
    CS_END_COL,
    US_BEG_COL,
    US_END_COL,
    TRIAL_TYPE_COL,
    TRIAL_NUMBER_COL,
    BLOCK_NAME_COL,
    TAIL_ANGLE_LABEL,
    VIGOR_COL,
    SCALED_VIGOR_COL,
    BOUT_BEG_COL,
    BOUT_END_COL,
    BOUT_COL,
]
# endregion Parameters


# region Helper Functions
def _angle_sort_key(name: str):
    """Extract numeric index from angle column name for sorting."""
    try:
        return int(name.split('Angle of point ')[1].split(' ')[0])
    except Exception:
        return name


def get_tail_angle_col(data: pd.DataFrame) -> str | None:
    """Find the canonical tail angle column in a DataFrame."""
    if TAIL_ANGLE_LABEL in data.columns:
        return TAIL_ANGLE_LABEL

    angle_cols = [c for c in data.columns if c.startswith('Angle of point ')]
    if not angle_cols:
        return None

    return sorted(angle_cols, key=_angle_sort_key)[-1]


# ---------------------------------------------------------------------------
# Rolling median (right-aligned / trailing window)
# ---------------------------------------------------------------------------
if HAS_NUMBA:
    @njit
    def rolling_median_numba(arr, window):
        n = len(arr)
        result = np.empty(n, dtype=arr.dtype)
        buf = np.empty(window, dtype=arr.dtype)
        for i in range(n):
            start = max(0, i - window + 1)
            count = 0
            for j in range(start, i + 1):
                if not np.isnan(arr[j]):
                    buf[count] = arr[j]
                    count += 1
            if count > 0:
                sub = buf[:count].copy()
                sub.sort()
                if count % 2 == 0:
                    result[i] = (sub[count // 2 - 1] + sub[count // 2]) / 2.0
                else:
                    result[i] = sub[count // 2]
            else:
                result[i] = np.nan
        return result
else:
    def rolling_median_numba(arr, window):
        n = len(arr)
        result = np.empty(n, dtype=arr.dtype)
        for i in range(n):
            start = max(0, i - window + 1)
            result[i] = np.nanmedian(arr[start:i + 1])
        return result


def apply_rolling_numba(x, window_size):
    return rolling_median_numba(x.to_numpy(), window_size)


def arrange_data(data: pd.DataFrame, time_col: str, tail_angle_col: str | None, skip_reset=False) -> pd.DataFrame:
    if not skip_reset and DAY_COL not in data.columns:
        data = data.reset_index(drop=False)

    if DAY_COL in data.columns and FISH_NO_COL in data.columns:
        data[FISH_COL] = data[DAY_COL].astype('string') + '_' + data[FISH_NO_COL].astype('string')
        data[[DAY_COL, FISH_NO_COL]] = data[[DAY_COL, FISH_NO_COL]].astype('string')
    elif FISH_COL in data.columns:
        data[FISH_COL] = data[FISH_COL].astype('string')
    else:
        raise ValueError("Missing fish identifier (expected 'Fish' or 'Day' + 'Fish no.').")

    dtype_dict = {
        time_col: 'int32',
        CS_BEG_COL: pd.SparseDtype('int32', 0),
        CS_END_COL: pd.SparseDtype('int32', 0),
        US_BEG_COL: pd.SparseDtype('int32', 0),
        US_END_COL: pd.SparseDtype('int32', 0),
        TRIAL_NUMBER_COL: 'int32',
        tail_angle_col: 'float32' if tail_angle_col else None,
        VIGOR_COL: 'float32',
        SCALED_VIGOR_COL: 'float32',
        BOUT_BEG_COL: pd.SparseDtype('bool'),
        BOUT_END_COL: pd.SparseDtype('bool'),
        BOUT_COL: pd.SparseDtype('bool'),
    }

    existing_cols = {k: v for k, v in dtype_dict.items() if k and k in data.columns}
    if existing_cols:
        data = data.astype(existing_cols, copy=False)

    return data


def process_data(
    df: pd.DataFrame,
    window_size: int,
    downsample_factor: int,
    grouping_cols: list[str],
    time_col: str,
    baseline_window_frames: int,
) -> pd.DataFrame:
    """Apply rolling-median smoothing, downsampling, log-transform, and baseline-median subtraction.

    Pipeline order:
    1. Bout masking — set vigor to NaN outside bouts.
    2. Sort by grouping columns + time.
    3. Rolling median (right-aligned, *window_size* frames) on VIGOR_COL.
    4. Downsample (every *downsample_factor*-th row).
    5. Log-transform positive vigor values (zeros/negatives → NaN).
    6. Baseline median subtraction — per (Fish, Trial), subtract the median
       of log-vigor in the pre-stimulus baseline window.  Result stored in
       SCALED_VIGOR_COL.
    """
    # 1. Bout masking
    df.loc[~df[BOUT_COL], VIGOR_COL] = np.nan

    # 2. Sort
    df = df.sort_values(grouping_cols + [time_col])

    # 3. Rolling median on VIGOR_COL only
    df[VIGOR_COL] = df.groupby(grouping_cols, observed=True, sort=False)[VIGOR_COL].transform(
        apply_rolling_numba, window_size=window_size
    )

    # 4. Downsample
    df = df.groupby(grouping_cols, observed=True, group_keys=False, sort=False).nth(
        slice(None, None, downsample_factor)
    )

    # 5. Log transform (NaN-safe; zeros/negatives → NaN)
    vigor = df[VIGOR_COL].to_numpy(dtype='float64')
    df[VIGOR_COL] = np.where(vigor > 0, np.log(vigor), np.nan)

    # 6. Baseline median subtraction per (Fish, Trial)
    baseline_mask = df[time_col] < -baseline_window_frames
    baseline_medians = (
        df.loc[baseline_mask]
        .groupby(grouping_cols, observed=True)[VIGOR_COL]
        .median()
    )
    df = df.merge(baseline_medians.rename('_bl_median'),
                  on=grouping_cols, how='left')
    df[SCALED_VIGOR_COL] = df[VIGOR_COL] - df['_bl_median']
    df.drop(columns=['_bl_median'], inplace=True)

    df.drop(columns=['index', 'level_0'], inplace=True, errors='ignore')
    return df


# region block_categories_from_config
def block_categories_from_config() -> list[str]:
    """Extract ordered block category names from experiment configuration."""
    cs_names = list(config.blocks_dict['blocks 10 trials']['CS']['names of blocks'])
    us_names = list(config.blocks_dict['blocks 10 trials']['US']['names of blocks'])
    return list(dict.fromkeys(cs_names + us_names))


# endregion block_categories_from_config


# region harmonize_and_concat


def harmonize_and_concat(data_list: list[pd.DataFrame], block_categories: list[str]) -> pd.DataFrame | None:
    """Concatenate DataFrames with consistent dtypes and block categories."""
    if not data_list:
        return None

    try:
        return pd.concat(data_list, ignore_index=False, copy=False)


    # endregion harmonize_and_concat


    # region clean_and_save
    except (ValueError, TypeError):
        int_cols = [CS_BEG_COL, CS_END_COL, US_BEG_COL, US_END_COL, TRIAL_NUMBER_COL]

        for df in data_list:
            existing_int_cols = [c for c in int_cols if c in df.columns]
            if existing_int_cols:
                df[existing_int_cols] = df[existing_int_cols].astype('int32')

            if BLOCK_NAME_COL in df.columns and df[BLOCK_NAME_COL].dtype.name != 'category':
                df[BLOCK_NAME_COL] = df[BLOCK_NAME_COL].astype(
                    CategoricalDtype(categories=block_categories, ordered=True)
                )

        return pd.concat(data_list, ignore_index=False, copy=False)


def clean_and_save(
    data: pd.DataFrame | None,
    path: Path,
    time_col: str,
    tail_angle_col: str | None,
) -> pd.DataFrame | None:
    """Clean DataFrame columns and save to pickle."""
    if data is None or len(data) == 0:
        return data

    data.loc[~data[BOUT_COL], [VIGOR_COL, SCALED_VIGOR_COL]] = np.nan

    if DAY_COL not in data.columns:
        data = arrange_data(data, time_col, tail_angle_col, skip_reset=True)

    data.drop(columns=['index', 'level_0', BOUT_BEG_COL, BOUT_END_COL], inplace=True, errors='ignore')
    data.to_pickle(path, compression='gzip')
    return data


# endregion clean_and_save


# region load_fish_data


def load_fish_data(
    fish_path: Path,
    condition: str,
    block_categories: list[str],
    fish_subset: list[str] | None = None,
) -> pd.DataFrame | None:
    """Load and preprocess a single fish pickle file."""
    try:
        data = pd.read_pickle(str(fish_path), compression='gzip')

        if TIME_COL not in data.columns and TRIAL_TIME_SECONDS_COL in data.columns:
            data = analysis_utils.convert_time_from_s_to_frame(data)

        if EXP_COL not in data.columns:
            if EXP_COL in data.index.names:
                data = data.reset_index(EXP_COL, drop=False)
            else:
                data[EXP_COL] = condition
        data[EXP_COL] = data[EXP_COL].astype(str).str.split("-", n=1).str[0].str.lower()

        tail_angle_col = get_tail_angle_col(data)
        if tail_angle_col is None:
            print(f"Skipping {fish_path.name}: missing tail angle column.")
            return None
        if tail_angle_col != TAIL_ANGLE_LABEL and TAIL_ANGLE_LABEL not in data.columns:
            data = data.rename(columns={tail_angle_col: TAIL_ANGLE_LABEL})
            tail_angle_col = TAIL_ANGLE_LABEL

        needed_cols = [c for c in NEEDED_BASE_COLS if c in data.columns]
        if tail_angle_col not in needed_cols and tail_angle_col in data.columns:
            needed_cols.append(tail_angle_col)
        data = data.loc[:, needed_cols].copy()

        if TRIAL_TYPE_COL not in data.columns or TRIAL_NUMBER_COL not in data.columns:
            print(f"Skipping {fish_path.name}: missing trial identifiers.")
            return None

        if (
            BLOCK_NAME_COL not in data.columns
            or data[BLOCK_NAME_COL].isna().any()
            or (data[BLOCK_NAME_COL] == '').any()
        ):
            data = analysis_utils.identify_blocks_trials(data, config.blocks_dict)

        data = data[(data[BLOCK_NAME_COL] != '') & (data[BLOCK_NAME_COL].notna())]
        if data.empty:
            return None

        if block_categories and BLOCK_NAME_COL in data.columns:
            data[BLOCK_NAME_COL] = data[BLOCK_NAME_COL].astype(
                CategoricalDtype(categories=block_categories, ordered=True)
            )

        data = arrange_data(data, TIME_COL, tail_angle_col)

        if fish_subset:
            normalized = [token.lower() for token in fish_subset]
            fish_ids = data[FISH_COL].astype(str).str.lower()
            mask = fish_ids.apply(lambda value: matches_subset(value, normalized))
            data = data.loc[mask]
            if data.empty:
                return None

        return data

    except Exception as exc:
        print(f"{fish_path} cannot be read: {exc}")
        return None


    # endregion load_fish_data


    # region process_subset


def process_subset(
    data: pd.DataFrame,
    mask: pd.Series,
    window_size: int,
    downsample_factor: int,
    grouping_cols: list[str],
    time_col: str,
    baseline_window_frames: int,
) -> pd.DataFrame | None:
    """Process a subset of trials (CS or US) through the smoothing/log-median pipeline."""
    if not mask.any():
        return None

    subset = data.loc[mask].copy(deep=True)
    subset.drop(columns=TRIAL_TYPE_COL, inplace=True)
    return process_data(
        subset,
        window_size,
        downsample_factor,
        grouping_cols,
        time_col,
        baseline_window_frames,
    )


# endregion process_subset
# endregion Helper Functions


def matches_subset(value: str, tokens: list[str]) -> bool:
    """Check if value contains any of the given tokens."""
    return any(token in value for token in tokens)


# region filter_paths_by_subset
def filter_paths_by_subset(paths: list[Path], fish_subset: list[str] | None) -> list[Path]:
    """Filter file paths to those matching the fish subset tokens."""
    if not fish_subset:
        return paths

    normalized = [token.lower() for token in fish_subset]
    selected = [path for path in paths if matches_subset(path.stem.lower(), normalized)]

    matched_tokens = {
        token for token in normalized if any(token in path.stem.lower() for path in selected)
    }
    missing_tokens = [token for token in normalized if token not in matched_tokens]
    if missing_tokens:
        print(f"No files matched subset entries: {', '.join(missing_tokens)}")

    if not selected:
        print("No file-name matches for FISH_SUBSET; falling back to per-row fish ID filtering.")
        return paths

    return selected


# endregion filter_paths_by_subset


# region Main
# region main
def main():
    """Process all fish files and create pooled per-condition datasets."""
    (
        _,
        _,
        _,
        _,
        _,
    paths = file_utils.create_folders(config.path_save)

    all_fish_data_paths = list(paths.orig_pkl.glob('**/*.pkl'))
    all_fish_data_paths_lower = {path: path.stem.lower() for path in all_fish_data_paths}

    print(config)

    baseline_window_frames = int(round(gen_config.baseline_window * gen_config.expected_framerate))
    block_categories = block_categories_from_config()

    for condition in config.cond_types:
        print(condition)

        condition_lower = condition.lower() + '_'
        condition_paths = [
            path for path, stem_lower in all_fish_data_paths_lower.items()
            if condition_lower in stem_lower
        ]
        condition_paths = filter_paths_by_subset(condition_paths, FISH_SUBSET)

        path_all_fish_condition_cs = paths.all_fish / f'{condition}_CS_new_logmedian.pkl'
        path_all_fish_condition_us = paths.all_fish / f'{condition}_US_new_logmedian.pkl'

        if not condition_paths:
            continue

        condition_all_data_cs = []
        condition_all_data_us = []

        for fish_i, fish in tqdm(
            enumerate(condition_paths),
            total=len(condition_paths),
            desc=f'Processing {condition}'
        ):
            data = load_fish_data(fish, condition, block_categories, FISH_SUBSET)
            if data is None:
                continue

            mask_cs = data[TRIAL_TYPE_COL] == CS_LABEL

            data_cs = process_subset(
                data, mask_cs, WINDOW_SIZE, DOWNSAMPLE_FACTOR, GROUPING_COLS, TIME_COL, baseline_window_frames
            )
            if data_cs is not None:
                condition_all_data_cs.append(data_cs)
                del data_cs

            data_us = process_subset(
                data, ~mask_cs, WINDOW_SIZE, DOWNSAMPLE_FACTOR, GROUPING_COLS, TIME_COL, baseline_window_frames
            )
            if data_us is not None:
                condition_all_data_us.append(data_us)
                del data_us

            del data, mask_cs

            if fish_i % 10 == 0:
                gc.collect()

        if condition_all_data_cs:
            condition_all_data_cs = harmonize_and_concat(condition_all_data_cs, block_categories)
            condition_all_data_cs = clean_and_save(
                condition_all_data_cs, path_all_fish_condition_cs, TIME_COL, TAIL_ANGLE_LABEL
            )

        if condition_all_data_us:
            condition_all_data_us = harmonize_and_concat(condition_all_data_us, block_categories)
            condition_all_data_us = clean_and_save(
                condition_all_data_us, path_all_fish_condition_us, TIME_COL, TAIL_ANGLE_LABEL
            )

        num_fish = 0
        if condition_all_data_cs is not None and len(condition_all_data_cs) > 0:
            num_fish = condition_all_data_cs[FISH_COL].nunique()
            print(f'Number fish in {condition} condition: {num_fish}')
            print(f"CS max trial number: {condition_all_data_cs[TRIAL_NUMBER_COL].max()}")
        elif condition_all_data_us is not None and len(condition_all_data_us) > 0:
            num_fish = condition_all_data_us[FISH_COL].nunique()
            print(f'Number fish in {condition} condition: {num_fish}')

        gc.collect()

    print('\n\ndone')

# endregion main


if __name__ == '__main__':
    main()
# endregion Main
