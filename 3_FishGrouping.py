"""Join per-fish datasets into pooled per-condition CS/US files with smoothing and scaling."""
# %%
# region Imports & Configuration
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
# pd.options.mode.chained_assignment = None

# Set plotting style (shared across analysis scripts)
plotting_style.set_plot_style()
# endregion

config = get_experiment_config(ExperimentType.ALL_DELAY.value)

# region Parameters
WINDOW_SIZE = 10
DOWNSAMPLE_FACTOR = 10
TIME_COL = gen_config.time_trial_frame_label
TAIL_ANGLE_LABEL = gen_config.tail_angle_label
# Optional subset of fish to process. Entries are matched as case-insensitive substrings
# against file stems and per-row fish IDs (Day_Fish no.); None processes all fish.
FISH_SUBSET: list[str] | None = None

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
# endregion


# region Helpers
def _angle_sort_key(name: str):
    try:
        return int(name.split('Angle of point ')[1].split(' ')[0])
    except Exception:
        return name


def get_tail_angle_col(data: pd.DataFrame) -> str | None:
    if TAIL_ANGLE_LABEL in data.columns:
        return TAIL_ANGLE_LABEL

    angle_cols = [c for c in data.columns if c.startswith('Angle of point ')]
    if not angle_cols:
        return None

    return sorted(angle_cols, key=_angle_sort_key)[-1]


if HAS_NUMBA:
    @njit
    def rolling_mean_numba(arr, window):
        result = np.empty_like(arr)
        for i in range(len(arr)):
            start = 0
            if i - window + 1 > 0:
                start = i - window + 1
            total = 0.0
            count = 0
            for j in range(start, i + 1):
                val = arr[j]
                if not np.isnan(val):
                    total += val
                    count += 1
            if count > 0:
                result[i] = total / count
            else:
                result[i] = np.nan
        return result
else:
    def rolling_mean_numba(arr, window):
        result = np.empty_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result[i] = np.nanmean(arr[start:i + 1])
        return result


def apply_rolling_numba(x, window_size):
    return rolling_mean_numba(x.to_numpy(), window_size)


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
    # Remove non-bout frames so baseline/scaling/smoothing only reflect active movement.
    df.loc[~df[BOUT_COL], VIGOR_COL] = np.nan

    # Compute per-trial/fish baseline from pre-CS frames.
    # The 10th/90th percentiles give a robust low/high reference to rescale vigor.
    baseline_mask = df[time_col] < -baseline_window_frames
    baseline_stats = (
        df.loc[baseline_mask]
        .groupby(grouping_cols, observed=True)[VIGOR_COL]
        .quantile([0.1, 0.9])
        .unstack()
    )

    if baseline_stats.shape[1] == 2:
        baseline_stats.columns = ['min_pre', 'max_pre']
        df = df.merge(baseline_stats, on=grouping_cols, how='left')

        # Scale each sample with the baseline quantiles; guard against flat baselines.
        numerator = df[VIGOR_COL] - df['min_pre']
        denominator = df['max_pre'] - df['min_pre']

        df[SCALED_VIGOR_COL] = np.nan
        valid_mask = (denominator > 0) & (denominator.notna())
        df.loc[valid_mask, SCALED_VIGOR_COL] = numerator[valid_mask] / denominator[valid_mask]

        df.drop(columns=['min_pre', 'max_pre'], inplace=True)
    else:
        df[SCALED_VIGOR_COL] = np.nan

    # Apply rolling mean per trial/fish to smooth high-frequency fluctuations.
    df = df.sort_values(grouping_cols + [time_col])

    for col in [VIGOR_COL, SCALED_VIGOR_COL]:
        if col in df.columns:
            df[col] = df.groupby(grouping_cols, observed=True, sort=False)[col].transform(
                apply_rolling_numba, window_size=window_size
            )

    # Downsample each trial by keeping every Nth frame.
    df_downsampled = df.groupby(grouping_cols, observed=True, group_keys=False, sort=False).nth(
        slice(None, None, downsample_factor)
    )

    df_downsampled.drop(columns=['index', 'level_0'], inplace=True, errors='ignore')
    return df_downsampled


def block_categories_from_config() -> list[str]:
    cs_names = list(config.blocks_dict['blocks 10 trials']['CS']['names of blocks'])
    us_names = list(config.blocks_dict['blocks 10 trials']['US']['names of blocks'])
    return list(dict.fromkeys(cs_names + us_names))


def harmonize_and_concat(data_list: list[pd.DataFrame], block_categories: list[str]) -> pd.DataFrame | None:
    if not data_list:
        return None

    try:
        return pd.concat(data_list, ignore_index=False, copy=False)
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
    if data is None or len(data) == 0:
        return data

    data.loc[~data[BOUT_COL], [VIGOR_COL, SCALED_VIGOR_COL]] = np.nan

    if DAY_COL not in data.columns:
        data = arrange_data(data, time_col, tail_angle_col, skip_reset=True)

    data.drop(columns=['index', 'level_0', BOUT_BEG_COL, BOUT_END_COL], inplace=True, errors='ignore')
    data.to_pickle(path, compression='gzip')
    return data


def load_fish_data(
    fish_path: Path,
    condition: str,
    block_categories: list[str],
    fish_subset: list[str] | None = None,
) -> pd.DataFrame | None:
    try:
        data = pd.read_pickle(str(fish_path), compression='gzip')

        # Harmonize time base if older files stored trial time in seconds.
        if TIME_COL not in data.columns and TRIAL_TIME_SECONDS_COL in data.columns:
            data = analysis_utils.convert_time_from_s_to_frame(data)

        # Ensure experiment label exists and is normalized for condition matching.
        if EXP_COL not in data.columns:
            if EXP_COL in data.index.names:
                data = data.reset_index(EXP_COL, drop=False)
            else:
                data[EXP_COL] = condition
        data[EXP_COL] = data[EXP_COL].astype(str).str.split("-", n=1).str[0].str.lower()

        # Resolve the tail angle column to a single canonical label.
        tail_angle_col = get_tail_angle_col(data)
        if tail_angle_col is None:
            print(f"Skipping {fish_path.name}: missing tail angle column.")
            return None
        if tail_angle_col != TAIL_ANGLE_LABEL and TAIL_ANGLE_LABEL not in data.columns:
            data = data.rename(columns={tail_angle_col: TAIL_ANGLE_LABEL})
            tail_angle_col = TAIL_ANGLE_LABEL

        # Keep only the columns used downstream for pooling/analysis.
        needed_cols = [c for c in NEEDED_BASE_COLS if c in data.columns]
        if tail_angle_col not in needed_cols and tail_angle_col in data.columns:
            needed_cols.append(tail_angle_col)
        data = data.loc[:, needed_cols].copy()

        # Require trial labels to split CS/US and group per trial.
        if TRIAL_TYPE_COL not in data.columns or TRIAL_NUMBER_COL not in data.columns:
            print(f"Skipping {fish_path.name}: missing trial identifiers.")
            return None

        # Identify blocks/trials if missing or malformed.
        if (
            BLOCK_NAME_COL not in data.columns
            or data[BLOCK_NAME_COL].isna().any()
            or (data[BLOCK_NAME_COL] == '').any()
        ):
            data = analysis_utils.identify_blocks_trials(data, config.blocks_dict)

        # Remove rows without a valid block label.
        data = data[(data[BLOCK_NAME_COL] != '') & (data[BLOCK_NAME_COL].notna())]
        if data.empty:
            return None

        # Enforce a consistent block order for downstream grouping/plots.
        if block_categories and BLOCK_NAME_COL in data.columns:
            data[BLOCK_NAME_COL] = data[BLOCK_NAME_COL].astype(
                CategoricalDtype(categories=block_categories, ordered=True)
            )

        # Standardize dtypes and fish identifiers.
        data = arrange_data(data, TIME_COL, tail_angle_col)

        # Optionally filter the loaded data by fish identifiers.
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


def process_subset(
    data: pd.DataFrame,
    mask: pd.Series,
    window_size: int,
    downsample_factor: int,
    grouping_cols: list[str],
    time_col: str,
    baseline_window_frames: int,
) -> pd.DataFrame | None:
    if not mask.any():
        return None

    # Separate CS/US trials and run the core smoothing/scaling/downsample pipeline.
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
# endregion


def matches_subset(value: str, tokens: list[str]) -> bool:
    return any(token in value for token in tokens)


def filter_paths_by_subset(paths: list[Path], fish_subset: list[str] | None) -> list[Path]:
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


# region Main
def main():
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        path_orig_pkl,
        path_all_fish,
        _,
    ) = file_utils.create_folders(config.path_save)

    all_fish_data_paths = list(path_orig_pkl.glob('*.pkl'))
    all_fish_data_paths_lower = {path: path.stem.lower() for path in all_fish_data_paths}

    print(config)

    baseline_window_frames = int(round(gen_config.baseline_window * gen_config.expected_framerate))
    block_categories = block_categories_from_config()

    for condition in config.cond_types:
        print(condition)

        # Identify all per-fish files for this condition.
        condition_lower = condition.lower() + '_'
        condition_paths = [
            path for path, stem_lower in all_fish_data_paths_lower.items()
            if condition_lower in stem_lower
        ]
        condition_paths = filter_paths_by_subset(condition_paths, FISH_SUBSET)

        path_all_fish_condition_cs = path_all_fish / f'{condition}_CS.pkl'
        path_all_fish_condition_us = path_all_fish / f'{condition}_US.pkl'

        if not condition_paths:
            continue

        condition_all_data_cs = []
        condition_all_data_us = []

        for fish_i, fish in tqdm(
            enumerate(condition_paths),
            total=len(condition_paths),
            desc=f'Processing {condition}'
        ):
            # Load and harmonize a single fish file.
            data = load_fish_data(fish, condition, block_categories, FISH_SUBSET)
            if data is None:
                continue

            # Split into CS and non-CS trials, then smooth/scale/downsample each subset.
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

        # Concatenate all fish for this condition and persist to disk.
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


if __name__ == '__main__':
    main()
# endregion
