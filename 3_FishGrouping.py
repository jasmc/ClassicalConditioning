"""
Fish Grouping Pipeline
======================

Join per-fish datasets into pooled per-condition CS/US files:
- Smooth vigor signals with rolling mean
- Scale vigor relative to baseline
- Downsample for efficient storage
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

import matplotlib.pyplot as plt

import analysis_utils
import figure_saving
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

# Apply per-experiment discarded fish list if present under "Processed data".
APPLY_FISH_DISCARD = True

# DPI for heatmap summary grids
FIG_DPI_SUMMARY_GRID: int = 200

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
    """Apply smoothing, scaling, and downsampling to vigor data."""
    df.loc[~df[BOUT_COL], VIGOR_COL] = np.nan

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

        numerator = df[VIGOR_COL] - df['min_pre']
        denominator = df['max_pre'] - df['min_pre']

        df[SCALED_VIGOR_COL] = np.nan
        valid_mask = (denominator > 0) & (denominator.notna())
        df.loc[valid_mask, SCALED_VIGOR_COL] = numerator[valid_mask] / denominator[valid_mask]

        df.drop(columns=['min_pre', 'max_pre'], inplace=True)
    else:
        df[SCALED_VIGOR_COL] = np.nan

    df = df.sort_values(grouping_cols + [time_col])

    for col in [VIGOR_COL, SCALED_VIGOR_COL]:
        if col in df.columns:
            df[col] = df.groupby(grouping_cols, observed=True, sort=False)[col].transform(
                apply_rolling_numba, window_size=window_size
            )

    df_downsampled = df.groupby(grouping_cols, observed=True, group_keys=False, sort=False).nth(
        slice(None, None, downsample_factor)
    )

    df_downsampled.drop(columns=['index', 'level_0'], inplace=True, errors='ignore')
    return df_downsampled


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
    """Process a subset of trials (CS or US) through the smoothing/scaling pipeline."""
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


# region load_discard_reasons
def load_discard_reasons(path_processed_data: Path) -> dict[str, str]:
    """Load fish-ID -> discard-reason mapping from ``Fish to discard.txt``.

    The file written by the preprocessing script has lines of the form::

        fish_name  reason text here

    where the fish name and reason are separated by two or more spaces.
    Header / comment / indented lines are skipped.
    """
    reasons_file = path_processed_data / "Fish to discard.txt"
    if not reasons_file.exists():
        return {}

    reasons: dict[str, str] = {}
    for raw_line in reasons_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Skip indented lines (parameter sub-entries) and header-like lines
        if raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        if line.lower().startswith("analysis parameters"):
            continue
        if line.lower().startswith("missing excluded"):
            continue

        import re as _re
        parts = _re.split(r"\s{2,}", line, maxsplit=1)
        if len(parts) == 2:
            fish_name, reason = parts
            reasons[fish_name.strip()] = reason.strip()
        elif len(parts) == 1:
            # Line with only a fish name and no reason
            reasons[parts[0].strip()] = ""
    return reasons


# endregion load_discard_reasons


# region save_heatmap_grid
def save_heatmap_grid(
    fish_ids: list[str],
    path_heatmap_fig_cs: Path,
    output_dir: Path,
    title: str = "Scaled Vigor Heatmaps",
    filename: str = "Heatmap_Grid.png",
    border_color: str = "#333333",
    discard_reasons: dict[str, str] | None = None,
) -> None:
    """Grid figure with pre-saved heatmaps (SVG/PNG) for each fish.

    Parameters
    ----------
    discard_reasons : dict mapping fish ID -> reason string, optional.
        When provided, a semi-transparent text box with the reason is placed
        in the top-right corner of each subplot.
    """
    import io

    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    n_fish = len(fish_ids)
    if n_fish == 0:
        print(f"  [WARN] No fish to plot for: {title}")
        return

    # Build lookup: fish_id (lower) -> heatmap path
    heatmap_files = (
        list(Path(path_heatmap_fig_cs).glob("*.svg"))
        + list(Path(path_heatmap_fig_cs).glob("*.png"))
    )
    fish_to_heatmap: dict[str, Path] = {}
    for fpath in heatmap_files:
        parts = fpath.stem.split("_")
        if len(parts) >= 2:
            fid = "_".join(parts[:2]).lower()
            if fid not in fish_to_heatmap or fpath.suffix.lower() == ".png":
                fish_to_heatmap[fid] = fpath

    def _crop_border(img: np.ndarray, frac: float = 0.02) -> np.ndarray:
        if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
            return img
        h, w = int(img.shape[0]), int(img.shape[1])
        dy, dx = int(round(h * frac)), int(round(w * frac))
        if (h - 2 * dy) < 2 or (w - 2 * dx) < 2:
            return img
        return img[dy : h - dy, dx : w - dx, ...]

    def _load_image(img_path: Path) -> np.ndarray | None:
        try:
            if img_path.suffix.lower() == ".svg":
                try:
                    import importlib

                    cairosvg = importlib.import_module("cairosvg")
                    png_data = cairosvg.svg2png(url=str(img_path))
                    img = Image.open(io.BytesIO(png_data))
                    return _crop_border(np.array(img), frac=0.02)
                except ImportError:
                    try:
                        import importlib

                        renderPM = importlib.import_module("reportlab.graphics.renderPM")
                        svglib = importlib.import_module("svglib.svglib")
                        drawing = svglib.svg2rlg(str(img_path))
                        png_data = renderPM.drawToString(drawing, fmt="PNG")
                        img = Image.open(io.BytesIO(png_data))
                        return _crop_border(np.array(img), frac=0.02)
                    except ImportError:
                        print(f"    [WARN] Cannot load SVG (install cairosvg or svglib): {img_path.name}")
                        return None
            img = Image.open(img_path)
            return _crop_border(np.array(img), frac=0.02)
        except Exception as e:
            print(f"    [WARN] Failed to load image {img_path.name}: {e}")
            return None

    n_cols = 6
    n_rows = max(1, (n_fish + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows + 1.5),
        facecolor="white",
        squeeze=False,
    )
    fig.suptitle(title, fontsize=16, y=0.99)

    missing_fish: list[str] = []
    loaded_count = 0

    for plot_idx, fish in enumerate(fish_ids):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row, col]
        fish_lower = fish.lower()

        heatmap_path = fish_to_heatmap.get(fish_lower)
        if heatmap_path is not None:
            img_array = _load_image(heatmap_path)
            if img_array is not None:
                ax.imshow(img_array)
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                loaded_count += 1
            else:
                ax.text(0.5, 0.5, f"Load Error\n{fish}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=13, color="red")
                ax.axis("off")
        else:
            missing_fish.append(fish)
            ax.text(0.5, 0.5, f"No heatmap\n{fish}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="gray")
            ax.axis("off")

        ax.set_title(fish, fontsize=14, color=border_color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(1.5)

        # Show discard reason as a text box in the top-right corner
        if discard_reasons is not None:
            reason = discard_reasons.get(fish, discard_reasons.get(fish_lower, ""))
            if reason:
                ax.text(
                    0.98, 0.96, reason,
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor=border_color),
                    color="#333333",
                    wrap=True,
                )

    # Hide unused axes
    for plot_idx in range(n_fish, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.subplots_adjust(wspace=0.08, hspace=0.14)

    grid_path = output_dir / filename
    figure_saving.save_figure(fig, grid_path, frmt="png", dpi=FIG_DPI_SUMMARY_GRID, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved heatmap grid: {grid_path.name}")
    print(f"    Loaded: {loaded_count}/{n_fish} heatmaps")
    if missing_fish:
        print(f"    Missing heatmaps for: {missing_fish[:10]}{'...' if len(missing_fish) > 10 else ''}")


# endregion save_heatmap_grid


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
        path_processed_data,
        _,
        _,
        _,
        _,
        _,
        path_scaled_vigor_fig_cs,
        _,
        _,
        _,
        path_pooled_vigor_fig,
        _,
        path_orig_pkl,
        path_all_fish,
        _,
    ) = file_utils.create_folders(config.path_save)

    all_fish_data_paths = list(path_orig_pkl.glob('*.pkl'))
    all_fish_data_paths_lower = {path: path.stem.lower() for path in all_fish_data_paths}

    # Load discarded fish IDs (if configured)
    discard_file = path_processed_data / "Discarded_fish_IDs.txt"
    fish_ids_to_discard: list[str] = []
    if APPLY_FISH_DISCARD and discard_file.exists():
        fish_ids_to_discard = file_utils.load_discarded_fish_ids(discard_file)
        print(f"  Loaded {len(fish_ids_to_discard)} discarded fish IDs from: {discard_file}")

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

        path_all_fish_condition_cs = path_all_fish / f'{condition}_CS_new.pkl'
        path_all_fish_condition_us = path_all_fish / f'{condition}_US_new.pkl'

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

        # --- Heatmap grids: discarded vs included fish ---
        all_condition_fish: set[str] = set()
        if condition_all_data_cs is not None and len(condition_all_data_cs) > 0:
            all_condition_fish.update(condition_all_data_cs[FISH_COL].unique())
        if condition_all_data_us is not None and len(condition_all_data_us) > 0:
            all_condition_fish.update(condition_all_data_us[FISH_COL].unique())

        if all_condition_fish:
            heatmap_output_dir = path_pooled_vigor_fig / "Fish grouping heatmaps"

            if APPLY_FISH_DISCARD and fish_ids_to_discard:
                discarded_in_condition = sorted(
                    f for f in all_condition_fish if f in fish_ids_to_discard
                )
                included_in_condition = sorted(
                    f for f in all_condition_fish if f not in fish_ids_to_discard
                )

                if discarded_in_condition:
                    save_heatmap_grid(
                        discarded_in_condition,
                        path_scaled_vigor_fig_cs,
                        heatmap_output_dir,
                        title=f"Discarded Fish - Scaled Vigor Heatmaps ({condition})",
                        filename=f"Heatmap_Grid_Discarded_{condition}.png",
                        border_color="#CC3333",
                    )
                else:
                    print(f"  No discarded fish in {condition} condition.")

                if included_in_condition:
                    save_heatmap_grid(
                        included_in_condition,
                        path_scaled_vigor_fig_cs,
                        heatmap_output_dir,
                        title=f"Included Fish - Scaled Vigor Heatmaps ({condition})",
                        filename=f"Heatmap_Grid_Included_{condition}.png",
                        border_color="#336699",
                    )
                else:
                    print(f"  No included fish in {condition} condition.")
            else:
                # No discard list: save a single grid with all fish
                save_heatmap_grid(
                    sorted(all_condition_fish),
                    path_scaled_vigor_fig_cs,
                    heatmap_output_dir,
                    title=f"All Fish - Scaled Vigor Heatmaps ({condition})",
                    filename=f"Heatmap_Grid_All_{condition}.png",
                )

        gc.collect()

    print('\n\ndone')

# endregion main


if __name__ == '__main__':
    main()
# endregion Main
