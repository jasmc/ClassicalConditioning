"""
Pipeline for Example Fish Figure Generation
============================================

This script consolidates multiple plotting scripts for generating publication-quality
figures of individual fish behavior:

1. **Fig1 Traces**: Tail angle and vigor traces for selected trials
2. **Tail Trajectory Heatmaps**: 2D density plots of tail positions before/after stimuli
3. **Bout Zoom**: Detailed view of individual bouts around US onset
4. **Individual Trials**: Tail angle traces, vigor heatmaps, and normalized vigor plots

**Usage:**
    - Set the boolean flags to enable/disable specific figure types
    - Configure the experiment types and fish IDs for each stage
    - Output directories are derived from each experiment's configured path_save
    - Run: `python Pipeline_Example_Fish.py`
"""

import sys
# %%
# region Imports & Configuration
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the repository root containing shared modules to the Python path
if "__file__" in globals():
    module_root = Path(__file__).resolve().parent
else:
    module_root = Path.cwd()
if str(module_root) not in sys.path:
    sys.path.insert(0, str(module_root))

import importlib

import analysis_utils

analysis_utils = importlib.reload(analysis_utils)
import file_utils
import plotting_style
from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as gen_config
from plotting_style import (configure_axes_for_pipeline,
                                configure_trace_axes, get_plot_config)

# Apply shared plotting aesthetics (fonts, sizes, etc.)
plotting_style.set_plot_style(use_constrained_layout=False)

# endregion


# region Parameters
# ==============================================================================
# PIPELINE CONTROL FLAGS
# ==============================================================================
RUN_TRACES = True             # Generate tail angle + vigor traces
RUN_INDIVIDUAL_TRIALS = False  # Generate individual-trial plots for selected fish
RUN_BOUT_ZOOM = False           # Generate zoomed bout plots
RUN_TRAJECTORY = False         # Generate tail trajectory heatmaps

# ==============================================================================
# GENERAL PARAMETERS
# ==============================================================================
OVERWRITE_FIGURES = True            # Regenerate existing figures
FIG_SIZE_IN = (5 / 2.54, 6 / 2.54)
FIG_DPI = 600

# ==============================================================================
# TRACES PARAMETERS
# ==============================================================================
# Reference: Plot\Single fish\Example fish.py and Example fish_MK.py
# TRACES_EXPERIMENT_TYPE = ExperimentType.ALL_DELAY.value
# TRACES_FISH_ID = [
#     '20221115_07', # delay
#     '20221115_09', # control
# ]
# '20221123_09'
TRACES_EXPERIMENT_TYPE = [
    # ExperimentType.ALL_DELAY.value,
    # ExperimentType.ALL_3S_TRACE.value,
    # ExperimentType.ALL_10S_TRACE.value,
    ExperimentType.ALL_DELAY.value,
]
TRACES_FISH_ID = [
    # '20221123_09',
    # '20221115_07', # delay
    # '20230307_12', # 3sTrace
    # '20230307_04', # 10sTrace
    '20221115_09', # control
]
TRACES_TRIAL_NUMBER_OFFSET = 4      # Subtract from trial numbers for display
TRACES_TRIALS = [
    # [5, 13, 59, 62, 89], # delay
    # [],
    # [],
    # []
    [5, 21, 59, 62, 89] # control
]
TRACES_TRIAL_NAMES = ["Pre-Train trial", "Early Train trial", "Late Train trial", "Early Test trial", "Late Test trial"]

# Plot limits and styling
TRACES_Y_TICKS_RAW = [-150, 0, 150]
TRACES_Y_TICKS_VIGOR = [-15, 15]
TRACES_Y_CLIP_RAW = (-180, 180)
TRACES_Y_CLIP_VIGOR = (0, 18)
TRACES_X_LIM = (-20, 20)
TRACES_X_TICKS = [-20, -10, 0, 10, 20]
TRACES_Y_TICKS_RAW = [-150, 0, 150]
TRACES_Y_TICKS_VIGOR = [0, 10]
TRACES_SUBPLOT_HSPACE = 0.9

# ==============================================================================
# INDIVIDUAL TRIALS PARAMETERS
# ==============================================================================
# Reference: Pipeline_Analysis.py -> run_plot_individual_trials()

INDIVIDUAL_TRIALS_OVERWRITE = True
INDIVIDUAL_TRIALS_RAW_TAIL_ANGLE = False
INDIVIDUAL_TRIALS_RAW_VIGOR = False
INDIVIDUAL_TRIALS_SCALED_VIGOR = True
INDIVIDUAL_TRIALS_NORMALIZED_VIGOR_TRIAL = False
INDIVIDUAL_TRIALS_METRIC = gen_config.tail_angle_label
INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S = 40
INDIVIDUAL_TRIALS_INTERVAL_BETWEEN_XTICKS_S = 20
INDIVIDUAL_TRIALS_FIG_FORMAT = 'svg'
INDIVIDUAL_TRIALS_STIMULI = ["CS", "US"]  # "US"

# ==============================================================================
# BOUT ZOOM PARAMETERS
# ==============================================================================
# Reference: Plot\Example bouts before and after US.py
BOUT_ZOOM_EXPERIMENT_TYPE = ExperimentType.RESP_TO_US.value
BOUT_ZOOM_FISH_ID = '20220809_24'

BOUT_ZOOM_CSUS = 'US'
BOUT_ZOOM_TRIAL_NUMBER = 2
BOUT_ZOOM_TIME_BEF = -5
BOUT_ZOOM_TIME_AFT = 5
BOUT_ZOOM_Y_LIM = (-150, 150)
BOUT_ZOOM_FORMAT = 'svg'

# ==============================================================================
# TAIL TRAJECTORY PARAMETERS
# ==============================================================================
# Reference: Fig1_sign of potential learning.py
TRAJ_EXPERIMENT_TYPE = ExperimentType.FIRST_DELAY.value
TRAJ_FISH_ID = '20221115_09'  # Prefix of the .pkl file

TRAJ_CSUS = "CS"
TRAJ_TRIAL_NUMBER_OFFSET = 4
TRAJ_TRIALS_SINGLE = (59, 59)
TRAJ_TRIALS_POOLED = (65, 65)
TRAJ_TIME_INTERVAL_S = 9
TRAJ_BEFORE_WINDOW_S = (-9, 0)
TRAJ_AFTER_WINDOW_S = (0, 9)
TRAJ_POOLED_BEFORE_WINDOW_S = (-10.5, -7.5)
TRAJ_POOLED_AFTER_WINDOW_S = (7.5, 10.5)
TRAJ_TAIL_SCALE = -1.2
TRAJ_SEGMENT_LEN = 0.05
TRAJ_NUM_FINE = 10000
TRAJ_HIST_BINS = 1000
TRAJ_SINGLE_XLIM = (0.2, 0.7)
TRAJ_SINGLE_YLIM = (-0.23, 0.073)
TRAJ_POOLED_XLIM = (-0.1, 0.8)
TRAJ_POOLED_YLIM = (-0.4, 0.35)
TRAJ_SINGLE_VMIN_DIV = 10_000_000
TRAJ_SINGLE_VMAX_DIV = 100_000
TRAJ_POOLED_VMIN_DIV = 50_000_000
TRAJ_POOLED_VMAX_DIV = 100_000
# endregion
# region Helper Functions
def resolve_experiment_paths(experiment_type: str) -> tuple:
    """
    Resolve the experiment configuration along with standard output and input paths.
    """
    # Load experiment-specific configuration
    config = get_experiment_config(experiment_type)
    # Validate that the output path is configured
    if not config.path_save:
        raise ValueError("config.path_save is empty; set a valid experiment path before running.")

    # Create folder structure and collect paths
    (
        _, _, _, path_processed_data, _, _, _, _, _, _, _, _, _, _, _,
        path_orig_pkl, _, _
    ) = file_utils.create_folders(config.path_save)
    # Return config and key paths used by the pipeline
    return config, path_processed_data, path_orig_pkl


def resolve_fish_path(path_orig_pkl: Path, fish_id: str | Path) -> Path:
    """
    Find the processed .pkl file for a fish based on its ID prefix.
    """
    # Normalize and validate fish ID
    fish_id_str = str(fish_id).strip()
    if not fish_id_str:
        raise ValueError("fish_id is empty; set a valid fish ID.")

    # Support direct absolute paths
    candidate = Path(fish_id_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # If a filename was provided, check within the expected folder
    if candidate.suffix == ".pkl":
        candidate = path_orig_pkl / candidate.name
        if candidate.exists():
            return candidate

    # Fallback: prefix match inside the folder
    matches = sorted(path_orig_pkl.glob(f"{fish_id_str}*.pkl"))
    if not matches:
        raise FileNotFoundError(f"No fish file matching '{fish_id_str}' in {path_orig_pkl}")
    if len(matches) > 1:
        print(f"Multiple fish files matched '{fish_id_str}'; using {matches[0].name}")
    # Return the first match (deterministic due to sorting)
    return matches[0]


def resolve_fish_paths(path_orig_pkl: Path, fish_ids: Sequence[str | Path]) -> list[Path]:
    """
    Resolve multiple fish IDs into a list of .pkl paths, preserving input order.
    """
    # Resolve each fish ID and deduplicate while preserving order
    fish_paths = []
    seen = set()
    for fish_id in fish_ids:
        try:
            fish_path = resolve_fish_path(path_orig_pkl, fish_id)
        except FileNotFoundError as exc:
            print(str(exc))
            continue
        if fish_path in seen:
            continue
        seen.add(fish_path)
        fish_paths.append(fish_path)
    # Return resolved paths for downstream processing
    return fish_paths


def flatten_fish_ids(fish_ids: object) -> list[str]:
    """
    Flatten nested fish ID lists while preserving order.
    """
    # Handle a single string/path input
    if isinstance(fish_ids, (str, Path)):
        return [str(fish_ids)]
    # Normalize non-list inputs to a single-item list
    if not isinstance(fish_ids, (list, tuple)):
        return [str(fish_ids)]
    # Flatten nested lists
    if fish_ids and all(isinstance(item, (list, tuple)) for item in fish_ids):
        return [str(fish_id) for sublist in fish_ids for fish_id in sublist]
    # Return a normalized list of strings
    return [str(fish_id) for fish_id in fish_ids]


def normalize_fish_groups(fish_ids: object) -> list[list[str]]:
    """
    Normalize fish ID input into a list of fish-id groups.
    """
    # Wrap single items into a single group
    if isinstance(fish_ids, (str, Path)):
        return [[str(fish_ids)]]
    if isinstance(fish_ids, np.ndarray):
        fish_ids = fish_ids.tolist()
    # Ensure input is list-like
    if not isinstance(fish_ids, (list, tuple)):
        return [[str(fish_ids)]]

    # If nested groups are provided, normalize each group
    if fish_ids and all(isinstance(item, (list, tuple, np.ndarray)) for item in fish_ids):
        groups = []
        for item in fish_ids:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            groups.append([str(fish_id) for fish_id in item])
        return groups

    # Otherwise, treat each item as its own group
    return [[str(fish_id)] for fish_id in fish_ids]


def get_axes_padding(fig, axes) -> dict:
    """
    Compute padding around a set of axes in figure-relative units.
    """
    # Render once to ensure layout positions are available
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Flatten possible nested axes structures into a list
    def flatten_axes(items):
        if items is None:
            return []
        if isinstance(items, np.ndarray):
            return [ax for ax in items.ravel().tolist() if ax is not None]
        if isinstance(items, (list, tuple)):
            flattened = []
            for item in items:
                flattened.extend(flatten_axes(item))
            return flattened
        return [items]

    axes_list = flatten_axes(axes)
    # Collect tight bounding boxes for visible axes
    bboxes = [
        ax.get_tightbbox(renderer)
        for ax in axes_list
        if hasattr(ax, "get_visible") and ax.get_visible()
    ]
    if not bboxes:
        return {"left": 0, "right": 0, "bottom": 0, "top": 0}

    # Compute global bounds across axes
    x0 = min(bbox.x0 for bbox in bboxes)
    y0 = min(bbox.y0 for bbox in bboxes)
    x1 = max(bbox.x1 for bbox in bboxes)
    y1 = max(bbox.y1 for bbox in bboxes)

    # Convert pixel bounds to figure-relative padding
    fig_bbox = fig.bbox
    fw, fh = fig.get_size_inches() * fig.dpi

    return {
        "left": (x0 - fig_bbox.x0) / fw,
        "right": (fig_bbox.x1 - x1) / fw,
        "bottom": (y0 - fig_bbox.y0) / fh,
        "top": (fig_bbox.y1 - y1) / fh,
    }


def get_axes_bounds(fig, axes) -> dict:
    """
    Compute the tight bounding box around a set of axes in figure-relative units.
    """
    # Render once to ensure layout positions are available
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Flatten possible nested axes structures into a list
    def flatten_axes(items):
        if items is None:
            return []
        if isinstance(items, np.ndarray):
            return [ax for ax in items.ravel().tolist() if ax is not None]
        if isinstance(items, (list, tuple)):
            flattened = []
            for item in items:
                flattened.extend(flatten_axes(item))
            return flattened
        return [items]

    axes_list = flatten_axes(axes)
    # Collect tight bounding boxes for visible axes
    bboxes = [
        ax.get_tightbbox(renderer)
        for ax in axes_list
        if hasattr(ax, "get_visible") and ax.get_visible()
    ]
    if not bboxes:
        return {"left": 0, "right": 1, "bottom": 0, "top": 1}

    # Compute global bounds across axes
    x0 = min(bbox.x0 for bbox in bboxes)
    y0 = min(bbox.y0 for bbox in bboxes)
    x1 = max(bbox.x1 for bbox in bboxes)
    y1 = max(bbox.y1 for bbox in bboxes)

    # Convert pixel bounds to figure-relative coordinates
    fig_bbox = fig.bbox
    fw, fh = fig.get_size_inches() * fig.dpi

    return {
        "left": (x0 - fig_bbox.x0) / fw,
        "right": (x1 - fig_bbox.x0) / fw,
        "bottom": (y0 - fig_bbox.y0) / fh,
        "top": (y1 - fig_bbox.y0) / fh,
    }


def pts_to_fig_frac(fig, pts, axis="x"):
    """
    Convert points to figure fraction along the given axis.
    """
    # Convert points to inches
    inches = float(pts) / 72.0
    # Pick figure dimension along desired axis
    size_in = fig.get_size_inches()[0] if axis == "x" else fig.get_size_inches()[1]
    # Convert inches to fraction of figure
    return inches / size_in


def resolve_trials_list(trials_spec: list, fish_id=None, fish_id_order=None) -> list:
    """
    Normalize trial selections to a flat list for a single fish.
    """
    # Normalize numpy input
    if isinstance(trials_spec, np.ndarray):
        trials_spec = trials_spec.tolist()
    # Guard against invalid input
    if not isinstance(trials_spec, (list, tuple)):
        return []
    if not trials_spec:
        return []

    # Detect nested trial lists
    is_nested = all(isinstance(item, (list, tuple, np.ndarray)) for item in trials_spec)
    if is_nested:
        nested_trials = []
        for item in trials_spec:
            if isinstance(item, np.ndarray):
                nested_trials.append(item.tolist())
            else:
                nested_trials.append(list(item))

        # If fish order is known, match trials by fish index
        if fish_id is not None and fish_id_order:
            fish_id_str = str(fish_id)
            try:
                idx = fish_id_order.index(fish_id_str)
            except ValueError:
                idx = None
            if idx is not None and idx < len(nested_trials):
                return list(nested_trials[idx])

        # Fallback: return the first non-empty group
        for item in nested_trials:
            if item:
                return list(item)
        return list(nested_trials[0]) if nested_trials else []

    # Non-nested case: return as list
    return list(trials_spec)
# endregion


# region Pipeline functions

# region Tail angle and vigor Traces
def run_traces(fish_id=None, trials_list=None, experiment_type=None, _emit_status=True, _fish_id_order=None) -> None:
    """
    Generate tail angle and vigor trace plots for selected trials.
    
    Creates two figures:
    - Raw tail angle traces
    - Vigor traces
    
    Each figure shows multiple trials as subplots with stimulus markers.
    """
    # Entry banner for the pipeline stage
    if _emit_status:
        print("\n" + "="*80)
        print("RUNNING  TRACES")
        print("="*80)

    # Resolve default inputs
    if experiment_type is None:
        experiment_type = TRACES_EXPERIMENT_TYPE
    if fish_id is None:
        fish_id = TRACES_FISH_ID

    # Normalize numpy inputs to lists
    if isinstance(experiment_type, np.ndarray):
        experiment_type = experiment_type.tolist()
    if isinstance(fish_id, np.ndarray):
        fish_id = fish_id.tolist()

    # Track fish order to map trials if needed
    if _fish_id_order is None:
        try:
            _fish_id_order = flatten_fish_ids(TRACES_FISH_ID)
        except Exception:
            _fish_id_order = None

    # Handle multi-experiment inputs by delegating per experiment
    if isinstance(experiment_type, (list, tuple)):
        experiment_types = list(experiment_type)
        if not experiment_types:
            raise ValueError("TRACES_EXPERIMENT_TYPE must be a non-empty list.")

        # Resolve fish IDs relative to experiment list shape
        if isinstance(fish_id, (str, Path)) or not isinstance(fish_id, (list, tuple)):
            if len(experiment_types) > 1:
                raise ValueError(
                    "TRACES_EXPERIMENT_TYPE has multiple entries; TRACES_FISH_ID must match their length."
                )
            experiment_type = experiment_types[0]
        else:
            if len(experiment_types) == 1:
                fish_ids_by_exp = {experiment_types[0]: list(fish_id)}
            elif all(isinstance(item, (list, tuple, np.ndarray)) for item in fish_id):
                if len(fish_id) != len(experiment_types):
                    raise ValueError(
                        "When providing nested fish ID lists, TRACES_EXPERIMENT_TYPE must match their length."
                    )
                fish_ids_by_exp = {
                    exp_type: (item.tolist() if isinstance(item, np.ndarray) else list(item))
                    for exp_type, item in zip(experiment_types, fish_id)
                }
            elif len(fish_id) == len(experiment_types):
                fish_ids_by_exp = {}
                for exp_type, fish_item in zip(experiment_types, fish_id):
                    fish_ids_by_exp.setdefault(exp_type, []).append(fish_item)
            else:
                raise ValueError(
                    "TRACES_EXPERIMENT_TYPE must have length 1 or match TRACES_FISH_ID."
                )

            # Delegate each experiment run
            for exp_type in experiment_types:
                exp_fish_ids = fish_ids_by_exp.get(exp_type, [])
                if not exp_fish_ids:
                    continue
                run_traces(
                    exp_fish_ids,
                    trials_list=trials_list,
                    experiment_type=exp_type,
                    _emit_status=False,
                    _fish_id_order=_fish_id_order,
                )

            # Emit completion for multi-experiment call
            if _emit_status:
                print(" TRACES FINISHED")
            return

    # Resolve experiment paths for the current experiment
    config, output_dir, path_orig_pkl = resolve_experiment_paths(experiment_type)

    # Handle grouped fish IDs by delegating per fish
    if isinstance(fish_id, (list, tuple)):
        fish_groups = normalize_fish_groups(fish_id)
        if not fish_groups:
            print("No fish IDs configured for traces; skipping.")
            return

        # Resolve trials per group
        trials_spec = TRACES_TRIALS
        if isinstance(trials_spec, np.ndarray):
            trials_spec = trials_spec.tolist()
        if not isinstance(trials_spec, (list, tuple)):
            trials_groups = []
        else:
            is_nested_trials = trials_spec and all(
                isinstance(item, (list, tuple, np.ndarray)) for item in trials_spec
            )
            if is_nested_trials:
                trials_groups = [
                    item.tolist() if isinstance(item, np.ndarray) else list(item)
                    for item in trials_spec
                ]
            else:
                trials_groups = [list(trials_spec)]

        if not trials_groups or all(len(group) == 0 for group in trials_groups):
            print("No trials configured for traces; skipping.")
            return

        # Use fish order to map nested trial selections
        fish_id_order = _fish_id_order or flatten_fish_ids(fish_groups)
        for group_idx, fish_group in enumerate(fish_groups):
            if not fish_group:
                continue

            if len(trials_groups) == 1:
                group_trials = list(trials_groups[0])
            elif group_idx < len(trials_groups):
                group_trials = list(trials_groups[group_idx])
            else:
                group_trials = resolve_trials_list(TRACES_TRIALS, fish_group[0], fish_id_order)

            if not group_trials:
                print("No trials configured for traces; skipping.")
                continue

            # Delegate per fish in the group
            for group_fish_id in fish_group:
                run_traces(
                    group_fish_id,
                    group_trials,
                    experiment_type=experiment_type,
                    _emit_status=False,
                    _fish_id_order=fish_id_order,
                )

        # Emit completion after grouped processing
        if _emit_status:
            print(" TRACES FINISHED")
        return

    # Resolve trials if not provided
    if trials_list is None:
        fish_id_order = _fish_id_order or flatten_fish_ids(TRACES_FISH_ID)
        trials_list = resolve_trials_list(TRACES_TRIALS, fish_id, fish_id_order)
        if not trials_list:
            print("No trials configured for traces; skipping.")
            return

    # Resolve the fish data path
    fish_path = resolve_fish_path(path_orig_pkl, fish_id)

    # Infer condition from the filename convention: <date>_<fish>_<condition>_... .pkl
    name_parts = fish_path.name.split("_")
    cond = name_parts[2].capitalize() if len(name_parts) > 2 else fish_path.stem

    # Normalize trial names to match trial count
    trials_list_names = list(TRACES_TRIAL_NAMES)
    if len(trials_list_names) < len(trials_list):
        trials_list_names.extend([f"Trial {t}" for t in trials_list[len(trials_list_names):]])
    trials_list_names = trials_list_names[:len(trials_list)]
    

    # Load fish data, normalize stimulus column names, and keep only CS trials.
    data = pd.read_pickle(str(fish_path), compression="gzip")
    data = analysis_utils.standardize_stim_cols(data)
    data = data.loc[data["Trial type"] == "CS"].copy()
    data.drop(columns=["Trial type"], inplace=True, errors="ignore")

    # Identify the tail angle column
    tail_angle_col = analysis_utils.get_tail_angle_col(data)
    if tail_angle_col is None:
        raise ValueError("Tail angle column not found in the data.")

    # Convert frame-based time to seconds if needed.
    time_col = gen_config.time_trial_frame_label
    if time_col in data.columns:
        data = analysis_utils.convert_time_from_frame_to_s(data)
    if "Trial time (s)" not in data.columns:
        raise ValueError("Trial time column is missing after conversion.")

    time_col = "Trial time (s)"

    # Adjust trial numbering to match figure conventions.
    trial_numbers = pd.to_numeric(data["Trial number"], errors="coerce")
    data = data.loc[trial_numbers.notna()].copy()
    data["Trial number"] = trial_numbers.loc[trial_numbers.notna()].astype(int) - TRACES_TRIAL_NUMBER_OFFSET
    data = data.loc[data["Trial number"] > 0]

    # Validate required columns early so errors are explicit.
    required_cols = [
        time_col,
        tail_angle_col,
        "Vigor (deg/ms)",
        "CS beg",
        "CS end",
        "US beg",
        "US end",
        "Trial number",
    ]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create two vertically stacked figures: tail angle (raw) and vigor.
    fig_size = FIG_SIZE_IN[0]*1.5, FIG_SIZE_IN[1]
    fig_raw, axs_raw = plt.subplots(
        len(trials_list),
        1,
        facecolor="white",
        sharex=False,
        sharey=True,
        figsize=fig_size,
        constrained_layout=False,
    )
    fig_vigor, axs_vigor = plt.subplots(
        len(trials_list),
        1,
        facecolor="white",
        sharex=False,
        sharey=True,
        figsize=fig_size,
        constrained_layout=False,
    )

    # Plot each requested trial in its own row.
    for index, trial in enumerate(trials_list):
        data_trial = data.loc[
            (data["Trial number"] == trial) & (data[time_col].between(TRACES_X_LIM[0], TRACES_X_LIM[1])),
            required_cols,
        ].copy(deep=True)

        if data_trial.empty:
            print(f"No data for trial {trial}")
            continue

        # Find the US onset time (if present) so we can annotate training rows.
        us_beg, _ = analysis_utils.find_events(data_trial, "US beg", "US end", time_col)
        stim_onset = us_beg[0] if len(us_beg) else None

        # Baseline subtract (median pre-stim) and clip for display stability.
        data_trial[tail_angle_col] = (
            data_trial[tail_angle_col] - data_trial.loc[data_trial[time_col] < 0, tail_angle_col].median()
        )
        data_trial[tail_angle_col] = data_trial[tail_angle_col].clip(
            TRACES_Y_CLIP_RAW[0], TRACES_Y_CLIP_RAW[1]
        )

        # Tail angle trace.
        axs_raw[index].plot(data_trial[time_col], data_trial[tail_angle_col], "k", clip_on=False, lw=0.4)

        # Vigor trace (clipped so rare spikes don't dominate the y-axis).
        data_trial["Vigor (deg/ms)"] = data_trial["Vigor (deg/ms)"].clip(TRACES_Y_CLIP_VIGOR[0], TRACES_Y_CLIP_VIGOR[1])
        axs_vigor[index].plot(data_trial[time_col], data_trial["Vigor (deg/ms)"], "k", clip_on=False, lw=0.4)

        # Add US marker for training trials (here, rows 2–3 correspond to training).
        if index in [1, 2] and stim_onset is not None:
            axs_raw[index].axvline(
                x=stim_onset,
                color=gen_config.plotting.us_color,
                clip_on=False,
                alpha=0.75,
                lw=2,
                linestyle="-",
                ymin=-0.15,
                ymax=1.4,
                zorder=10,
            )
            axs_vigor[index].axvline(
                x=stim_onset,
                color=gen_config.plotting.us_color,
                clip_on=False,
                alpha=0.75,
                lw=2,
                linestyle="-",
                ymin=-0.15,
                ymax=1.4,
                zorder=10,
            )

    # Style all axes
    # Style all axes
    for i, axs in enumerate([axs_raw, axs_vigor]):
        for ax_i, ax in enumerate(axs):
            yticks = TRACES_Y_TICKS_RAW if i == 0 else TRACES_Y_TICKS_VIGOR
            is_last = ax_i == len(trials_list) - 1

            if is_last:
                configure_axes_for_pipeline(
                    ax,
                    xlim=TRACES_X_LIM,
                    ylim=(min(yticks), max(yticks)),
                    x_ticks=TRACES_X_TICKS,
                    y_ticks=yticks,
                    show_xticks=True,
                    show_yticks=True,
                )
            else:
                configure_axes_for_pipeline(
                    ax,
                    xlim=TRACES_X_LIM,
                    ylim=(min(yticks), max(yticks)),
                    show_xticks=False,
                    show_yticks=True,
                    y_ticks=[],
                    hide_spines=("top", "right", "bottom"),
                )

            # Add CS markers
            ax.axvline(
                x=0,
                color=gen_config.plotting.cs_color,
                clip_on=False,
                alpha=0.75,
                lw=2,
                linestyle="-",
                ymin=-0.15,
                ymax=1.4,
                zorder=10,
            )
            ax.axvline(
                x=config.cs_duration,
                color=gen_config.plotting.cs_color,
                clip_on=False,
                alpha=0.75,
                lw=2,
                linestyle="-",
                ymin=-0.15,
                ymax=1.4,
                zorder=10,
            )

            # Per-row label: human-readable trial name.
            ax.set_title(
                trials_list_names[ax_i],
                loc="left",
                va="bottom",
                ha="left",
                x=0,   
                y=0.7,           
                backgroundcolor="none",
                fontsize=10,
            )

            # if ax_i == 0:
            #     ax.text(
            #         1,
            #         2,
            #         f"Example {cond.capitalize()} fish",
            #         transform=ax.transAxes,
            #         va="bottom",
            #         ha="right",
            #         backgroundcolor="none",
            #         fontsize=11,
            #     )

        # Only the last row shows x ticks.
        axs[-1].tick_params(axis="x", top=False, bottom=True, direction="out")

    # Layout adjustments
    fig_raw.subplots_adjust(hspace=TRACES_SUBPLOT_HSPACE)
    fig_vigor.subplots_adjust(hspace=TRACES_SUBPLOT_HSPACE)

    # fig_raw.canvas.draw()
    # fig_vigor.canvas.draw()

    # region supylabel
    # Add shared labels and titles
    plot_config = get_plot_config()
    analysis_utils.add_component(
        fig_raw,
        analysis_utils.AddTextSpec(
            component="supylabel",
            text="Tail angle (deg)",
            anchor_h="left",
            anchor_v="center",
            pad_pt=plot_config.pad_supylabel,
            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "fontweight": "bold",  "rotation": 90},
        ),
    )

    # NOTE: Keep vigor label units explicit (deg/ms) to match upstream processing.
    analysis_utils.add_component(
        fig_vigor,
        analysis_utils.AddTextSpec(
            component="supylabel",
            text="Vigor (deg/ms)",
            anchor_h="left",
            anchor_v="center",
            pad_pt=plot_config.pad_supylabel,
            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "fontweight": "bold",  "rotation": 90},
        ),
    )

    fig_raw.canvas.draw()
    fig_vigor.canvas.draw()

    analysis_utils.add_component(
        fig_raw,
        analysis_utils.AddTextSpec(
            component="supxlabel",
            text="Time relative to CS onset (s)",
            anchor_h="center",
            anchor_v="bottom",
            pad_pt=plot_config.pad_supxlabel,
            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "fontweight": "bold"},
        ),
    )
    analysis_utils.add_component(
        fig_vigor,
        analysis_utils.AddTextSpec(
            component="supxlabel",
            text="Time relative to CS onset (s)",
            anchor_h="center",
            anchor_v="bottom",
            pad_pt=plot_config.pad_supxlabel,
            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "fontweight": "bold"},
        ),
    )

    fig_raw.canvas.draw()
    fig_vigor.canvas.draw()

    # Add figure titles
    analysis_utils.add_component(
        fig_raw,
        analysis_utils.AddTextSpec(
            component="fig_title",
            text=f"Example {cond.capitalize()} fish",
            anchor_h="right",
            anchor_v="top",
            pad_pt=plot_config.pad_fig_title,
            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "backgroundcolor": "none", "va": "bottom"},
        ),
    )
    analysis_utils.add_component(
        fig_vigor,
        analysis_utils.AddTextSpec(
            component="fig_title",
            text=f"Example {cond.capitalize()} fish",
            anchor_h="right",
            anchor_v="top",
            pad_pt=plot_config.pad_fig_title,
            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "backgroundcolor": "none", "va": "bottom"},
        ),
    )

    # Render once so text extents are available for consistent panel-label placement.
    fig_raw.canvas.draw()
    fig_vigor.canvas.draw()

    # Panel labels
    analysis_utils.add_component(
        fig_raw,
        analysis_utils.AddTextSpec(
            component="text",
            text="C",
            anchor_h="left",
            anchor_v="top",
            pad_pt=plot_config.pad_panel_label,
            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "fontweight": "bold", "backgroundcolor": "none"},
        ),
    )

    analysis_utils.add_component(
        fig_vigor,
        analysis_utils.AddTextSpec(
            component="text",
            text="D",
            anchor_h="left",
            anchor_v="top",
            pad_pt=plot_config.pad_panel_label,
            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "fontweight": "bold", "backgroundcolor": "none"},
        ),
    )

    # Save figures
    fig_raw.savefig(
        str(output_dir / f"raw_{cond}.svg"),
        # bbox_inches="tight",
        dpi=FIG_DPI,
        format="svg",
        transparent=False,
    )
    fig_vigor.savefig(
        str(output_dir / f"vigor_{cond}.svg"),
        # bbox_inches="tight",
        dpi=FIG_DPI,
        format="svg",
        transparent=False,
    )
    # Cleanup
    # plt.close("all")

    # Emit completion
    if _emit_status:
        print(" TRACES FINISHED")
# endregion Tail angle and vigor Traces

# region Heatmaps
def run_individual_trials() -> None:
    """
    Generate the same individual-trial plots as Pipeline_Analysis.run_plot_individual_trials
    for a selected list of fish.
    """
    # Entry banner for the pipeline stage
    print("\n" + "=" * 80)
    print("RUNNING INDIVIDUAL TRIALS")
    print("=" * 80)

    plot_config = get_plot_config()

    # Resolve experiment types and fish IDs
    experiment_types = TRACES_EXPERIMENT_TYPE
    if isinstance(experiment_types, (str, Path)):
        experiment_types = [experiment_types]
    if not isinstance(experiment_types, (list, tuple)) or not experiment_types:
        raise ValueError("INDIVIDUAL_TRIALS_EXPERIMENT_TYPE must be a non-empty list.")

    # Normalize experiment identifiers to strings (safe for dict keys and path resolution).
    experiment_types = [str(e) for e in experiment_types]

    fish_ids = TRACES_FISH_ID
    if isinstance(fish_ids, (str, Path)):
        fish_ids = [fish_ids]
    if not fish_ids:
        print("No fish IDs provided for individual trials.")
        return

    # Map fish IDs to experiment types
    if len(experiment_types) == 1:
        fish_ids_by_exp = {experiment_types[0]: list(fish_ids)}
    elif all(isinstance(item, (list, tuple)) for item in fish_ids):
        if len(fish_ids) != len(experiment_types):
            raise ValueError(
                "When providing nested fish ID lists, INDIVIDUAL_TRIALS_EXPERIMENT_TYPE must match their length."
            )
        
        fish_ids_by_exp = {
            exp_type: list(exp_fish_ids)
            for exp_type, exp_fish_ids in zip(experiment_types, fish_ids)
        }
    elif len(fish_ids) == len(experiment_types):
        fish_ids_by_exp = {}
        for exp_type, fish_id in zip(experiment_types, fish_ids):
            fish_ids_by_exp.setdefault(exp_type, []).append(fish_id)
    else:
        raise ValueError(
            "INDIVIDUAL_TRIALS_EXPERIMENT_TYPE must have length 1 or match INDIVIDUAL_TRIALS_FISH_IDS."
        )

    # Process each experiment separately. This mirrors the structure in Pipeline_Analysis.py,
    # but restricts outputs to a curated set of "example fish".

    # Process each experiment separately
    for experiment_type in experiment_types:
        _, traces_output_dir, _ = resolve_experiment_paths(experiment_type)
        config, _, path_orig_pkl = resolve_experiment_paths(experiment_type)
        file_utils.create_folders(config.path_save)

        # Resolve fish paths for this experiment
        exp_fish_ids = fish_ids_by_exp.get(experiment_type, [])
        if not exp_fish_ids:
            print(f"No fish IDs provided for individual trials in {experiment_type}.")
            continue

        fish_paths = resolve_fish_paths(path_orig_pkl, exp_fish_ids)
        if not fish_paths:
            print(f"No matching fish files found for individual trials in {experiment_type}.")
            continue

        # Normalize CR window for downstream logic
        cr_window = config.cr_window
        if isinstance(cr_window, (int, float, np.integer, np.floating)):
            cr_window = [0, cr_window]

        # Compute tick spacing in frame units
        interval_between_xticks_frames = int(
            INDIVIDUAL_TRIALS_INTERVAL_BETWEEN_XTICKS_S * gen_config.expected_framerate
        )
        xtick_step_raw = max(1, int(interval_between_xticks_frames / gen_config.plotting.downsampling_step))
        xtick_step_scaled = max(1, int(interval_between_xticks_frames))

        # Process CS and/or US aligned trials
        for csus in INDIVIDUAL_TRIALS_STIMULI:
            print(f"Processing {csus} trials for {experiment_type}...")
            try:
                trials_blocks = config.blocks_dict["blocks 10 trials"][csus]["trials in each block"]
                if not trials_blocks:
                    print(f"No trial blocks defined for {csus} in config.")
                    continue
            except Exception as exc:
                print(f"Failed to extract trial block info for {csus}: {exc}")
                continue

            # Select stimulus duration based on alignment
            if csus == "CS":
                stim_duration = config.cs_duration
            else:
                stim_duration = gen_config.us_duration

            # Keep a stable fish order for trial selection
            fish_id_order = flatten_fish_ids(TRACES_FISH_ID)

            # Process each fish file
            for fish_path in fish_paths:
                cond = fish_path.name.split("_")[2].capitalize()
                stem_fish_path_orig = fish_path.stem.lower()
                stem_split = stem_fish_path_orig.split("_")
                fish_id = "_".join(stem_split[:2]) if len(stem_split) >= 2 else stem_fish_path_orig
                # Resolve which trials to mark
                trials_list = resolve_trials_list(TRACES_TRIALS, fish_id, fish_id_order)
                trials_to_mark = [t + TRACES_TRIAL_NUMBER_OFFSET for t in trials_list]

                fig_path_tail = Path(
                    str(traces_output_dir / stem_fish_path_orig)
                    + f"_tail angle aligned to {csus}.{INDIVIDUAL_TRIALS_FIG_FORMAT}"
                )
                fig_path_raw = Path(
                    str(traces_output_dir / stem_fish_path_orig)
                    + f"_raw vigor heatmap aligned to {csus}.{INDIVIDUAL_TRIALS_FIG_FORMAT}"
                )
                fig_path_sc = Path(
                    str(traces_output_dir / stem_fish_path_orig)
                    + f"_scaled vigor heatmap aligned to {csus}.{INDIVIDUAL_TRIALS_FIG_FORMAT}"
                )
                fig_path_norm = Path(
                    str(traces_output_dir / stem_fish_path_orig)
                    + f"_normalized vigor trial aligned to {csus}.{INDIVIDUAL_TRIALS_FIG_FORMAT}"
                )

                # Decide which plots need to be generated
                do_tail = INDIVIDUAL_TRIALS_RAW_TAIL_ANGLE and (
                    INDIVIDUAL_TRIALS_OVERWRITE or not fig_path_tail.exists()
                )
                do_raw = INDIVIDUAL_TRIALS_RAW_VIGOR and (
                    INDIVIDUAL_TRIALS_OVERWRITE or not fig_path_raw.exists()
                )
                do_sc = INDIVIDUAL_TRIALS_SCALED_VIGOR and (
                    INDIVIDUAL_TRIALS_OVERWRITE or not fig_path_sc.exists()
                )
                do_norm = INDIVIDUAL_TRIALS_NORMALIZED_VIGOR_TRIAL and (
                    INDIVIDUAL_TRIALS_OVERWRITE or not fig_path_norm.exists()
                )

                # Skip if no outputs are required
                if not any([do_tail, do_raw, do_sc, do_norm]):
                    print(f"Skip {csus} {fish_id}: all figures exist")
                    continue

                # Load data for this fish
                try:
                    data = pd.read_pickle(str(fish_path), compression="gzip")
                except Exception:
                    print(f"Skip {csus} {fish_id}: read failed")
                    continue
                data.reset_index(drop=True, inplace=True)

                # Identify tail angle column
                tail_col = analysis_utils.get_tail_angle_col(data)
                if tail_col is None:
                    print(f"Skip {csus} {fish_id}: missing tail angle col")
                    continue

                # Ensure time column is in frames
                time_col = gen_config.time_trial_frame_label
                if time_col not in data.columns and "Trial time (s)" in data.columns:
                    data = analysis_utils.convert_time_from_s_to_frame(data)
                if time_col not in data.columns:
                    print(f"Skip {csus} {fish_id}: missing time col")
                    continue

                # Resolve metric to plot
                metric = INDIVIDUAL_TRIALS_METRIC
                if metric == gen_config.tail_angle_label:
                    metric = tail_col

                # Build required column list
                needed_cols = [
                    time_col,
                    "CS beg",
                    "CS end",
                    "US beg",
                    "US end",
                    "Trial type",
                    "Trial number",
                    "Block name",
                    tail_col,
                    "Vigor (deg/ms)",
                    "Bout beg",
                    "Bout end",
                    "Bout",
                ]
                if metric not in needed_cols:
                    needed_cols.append(metric)

                # Filter to required columns
                try:
                    data = data.loc[:, needed_cols].copy()
                except Exception:
                    print(f"Skip {csus} {fish_id}: missing required columns")
                    continue

                # Filter to selected stimulus type
                data = data.loc[data["Trial type"] == csus, :]
                if data.empty:
                    print(f"Skip {csus} {fish_id}: no trials for csus")
                    continue

                print(f"Analyze {csus} {fish_id}")

                # Convert time to seconds and window the data
                data = analysis_utils.convert_time_from_frame_to_s(data)
                data = data.loc[
                    data["Trial time (s)"].between(-INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S, INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S)
                ]

                # Densify sparse columns if present
                for col in ["Vigor (deg/ms)", "Bout beg", "Bout end", "Bout"]:
                    if col in data.columns:
                        try:
                            data[col] = data[col].sparse.to_dense()
                        except:
                            pass

                # Tail-angle and/or metric traces
                if do_tail:
                    data_plot = data[["Trial time (s)", "Trial number", metric]].copy()
                    trials = data_plot["Trial number"].unique().astype("int")

                    fig, axs = plt.subplots(len(trials), 1, sharex=False, sharey=False, facecolor="white")
                    if len(trials) == 1:
                        axs = [axs]

                    # Render each trial as a row
                    for t_i, t in enumerate(trials):
                        data_trial = data[data["Trial number"] == t]

                        try:
                            data_trial_events = data_trial[["Trial time (s)", "CS beg", "CS end"]].copy()
                            cs_b, cs_e = analysis_utils.find_events(data_trial_events, "CS beg", "CS end", "Trial time (s)")
                        except Exception as exc:
                            print(f"Failed to find CS events for trial {t} in {fish_id}: {exc}")
                            cs_b, cs_e = [], []

                        try:
                            data_trial_events = data_trial[["Trial time (s)", "US beg", "US end"]].copy()
                            us_b, us_e = analysis_utils.find_events(data_trial_events, "US beg", "US end", "Trial time (s)")
                        except Exception as exc:
                            print(f"Failed to find US events for trial {t} in {fish_id}: {exc}")
                            us_b, us_e = [], []

                        # Plot either a continuous trace or an event raster
                        if metric in [tail_col, "Vigor (deg/ms)"]:
                            axs[t_i].plot(
                                data_trial["Trial time (s)"],
                                data_trial[metric],
                                "k",
                                alpha=1,
                                lw=0.3,
                                clip_on=False,
                            )
                        else:
                            axs[t_i].eventplot(
                                data_trial.loc[data_trial[metric], "Trial time (s)"].to_list(),
                                color="k",
                                alpha=1,
                                lineoffsets=1,
                                linelengths=1,
                            )

                        # Add CS markers
                        for cs_beg_, cs_end_ in zip(cs_b, cs_e):
                            axs[t_i].axvline(
                                cs_beg_,
                                color=gen_config.plotting.cs_color,
                                alpha=0.75,
                                lw=2,
                                linestyle="-",
                                zorder=10,
                            )
                            axs[t_i].axvline(
                                cs_end_,
                                color=gen_config.plotting.cs_color,
                                alpha=0.75,
                                lw=2,
                                linestyle="--",
                                zorder=10,
                            )

                        # Add US markers
                        for us_beg_, us_end_ in zip(us_b, us_e):
                            axs[t_i].axvline(
                                us_beg_,
                                color=gen_config.plotting.us_color,
                                alpha=0.75,
                                lw=2,
                                linestyle="-",
                                zorder=10,
                            )
                            axs[t_i].axvline(
                                us_end_,
                                color=gen_config.plotting.us_color,
                                alpha=0.75,
                                lw=2,
                                linestyle="--",
                                zorder=10,
                            )

                        # Per-axis styling
                        axs[t_i].spines[:].set_visible(False)
                        axs[t_i].set_title(t, fontsize="small", loc="left", backgroundcolor="none")
                        axs[t_i].tick_params(axis="both", which="both", bottom=True, top=False, right=False, direction="out")
                        axs[t_i].set_xlim((gen_config.time_bef_s, gen_config.time_aft_s))
                        axs[t_i].set_xticks([])
                        axs[t_i].set_yticks([])
                        if metric == tail_col:
                            axs[t_i].set_ylim((-50, 50))

                    axs[-1].tick_params(axis="both", which="both", bottom=True, top=False, right=False, direction="out")
                    axs[-1].set_xlim((gen_config.time_bef_s, gen_config.time_aft_s))
                    axs[-1].set_xticks([gen_config.time_bef_s, 0, 10, gen_config.time_aft_s])

                    # Final axis limits based on metric
                    if metric == tail_col:
                        axs[-1].set_ylim((-50, 50))
                        axs[-1].set_yticks([-50, 50])
                    elif metric == "Vigor (deg/ms)":
                        # Use the pooled per-fish windowed data for baseline scaling.
                        # (At this point, `data_trial` refers to a loop-local variable and may not exist
                        # if trials were empty/skipped.)
                        baseline_mask = data_plot["Trial time (s)"].between(-gen_config.baseline_window, 0)
                        baseline_q95 = data_plot.loc[baseline_mask, metric].quantile(0.95)
                        axs[-1].set_ylim((0, baseline_q95))
                    elif metric in ["Bout", "Bout beg", "Bout end"]:
                        axs[-1].set_ylim((0.5, 1))

                    # Add shared labels/titles
                    fig.set_size_inches(fig.get_size_inches()[0], 20)
                    analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="supxlabel",
                            text="Time relative to CS onset (s)",
                            anchor_h="center",
                            anchor_v="bottom",
                            pad_pt=plot_config.pad_supxlabel,
                            text_kwargs={"fontweight": "bold"},
                        ),
                    )
                    supylabel = analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="supylabel",
                            text="Tail angle (deg)",
                            anchor_h="left",
                            anchor_v="center",
                            pad_pt=plot_config.pad_supylabel,
                            text_kwargs={"fontweight": "bold"},
                        ),
                    )

                    suptitle = analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="fig_title",
                            text=f"Example {cond.capitalize()} fish",
                            anchor_h="right",
                            anchor_v="top",
                            pad_pt=plot_config.pad_fig_title,
                            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "backgroundcolor": "none"},
                        ),
                    )

                    if supylabel is not None and suptitle is not None:
                        analysis_utils.add_component(
                            fig,
                            analysis_utils.AddTextSpec(
                                component="text",
                                text="B",
                                anchor_h="left",
                                anchor_v="top",
                                pad_pt=plot_config.pad_panel_label,
                                text_kwargs={"fontsize": plot_config.fontsize_axis_label, "fontweight": "bold", "backgroundcolor": "none"},
                            ),
                        )

                    # Save tail angle figure
                    fig.savefig(
                        str(fig_path_tail),
                        format=INDIVIDUAL_TRIALS_FIG_FORMAT,
                        dpi=300,
                        transparent=False,
                        # bbox_inches="tight",
                    )
                

                # Raw vigor heatmap
                if do_raw:
                    data_plot = data.copy(deep=True)
                    data_plot = data_plot[
                        data_plot["Trial time (s)"].between(-INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S, INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S)
                    ]

                    v_min, v_max = np.quantile(data_plot["Vigor (deg/ms)"], [0.1, 0.9])

                    data_plot = (
                        data_plot[["Trial time (s)", "Trial number", "Vigor (deg/ms)"]]
                        .pivot(index="Trial time (s)", columns="Trial number")
                        .reset_index()
                        .iloc[::gen_config.plotting.downsampling_step]
                        .set_index("Trial time (s)")
                        .droplevel(0, axis=1)
                        .T
                    )

                    phases_trials = config.blocks_dict["blocks phases"][csus]["trials in each block"]
                    phases_names = config.blocks_dict["blocks phases"][csus]["names of blocks"]

                    fig, axs = plt.subplots(
                        len(phases_trials),
                        1,
                        facecolor="white",
                        gridspec_kw={"height_ratios": [len(x) for x in phases_trials], "hspace": 0},
                        sharex=True,
                        squeeze=False,
                    )

                    # Render heatmap blocks
                    for b_i, b in enumerate(phases_names):
                        sns.heatmap(
                            data_plot[data_plot.index.isin(phases_trials[b_i])],
                            cbar=False,
                            robust=False,
                            xticklabels=xtick_step_raw,
                            yticklabels=False,
                            ax=axs[b_i][0],
                            clip_on=False,
                            vmin=v_min,
                            vmax=v_max,
                        )

                        trials_in_this_block = data_plot[data_plot.index.isin(phases_trials[b_i])].index
                        for tr_i, tr in enumerate(trials_in_this_block):
                            if tr in trials_to_mark:
                                axs[b_i][0].plot(
                                    1,
                                    tr_i + 0.5,
                                    marker="<",
                                    color="k",
                                    transform=axs[b_i][0].get_yaxis_transform(),
                                    clip_on=False,
                                    zorder=20,
                                    markersize=3,
                                )

                        xlims = axs[b_i][0].get_xlim()
                        middle = np.mean(xlims)
                        factor = (xlims[-1] - xlims[0]) / (2 * INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S)

                        axs[b_i][0].set_xlabel("")
                        axs[b_i][0].set_ylabel("")

                        analysis_utils.add_component(
                            axs[b_i][0],
                            analysis_utils.AddTextSpec(
                                component="axis_title",
                                text=b,
                                anchor_h="left",
                                anchor_v="center",
                                pad_pt=plot_config.pad_supylabel,
                                text_kwargs={"color": "k", "rotation": 90, "fontweight": "bold"},
                            ),
                        )
                        axs[b_i][0].axvline(
                            middle, color="white", alpha=0.75, lw=2, linestyle="-", zorder=10
                        )

                        if csus == "CS":
                            axs[b_i][0].axvline(
                                middle + stim_duration * factor,
                                color="white",
                                alpha=0.75,
                                lw=2,
                                linestyle="-",
                                zorder=10,
                            )

                        axs[b_i][0].axhline(axs[b_i][0].get_ylim()[0], color="white", alpha=0.95, lw=2, linestyle="-")

                    axs[-1][0].tick_params(axis="both", which="both", bottom=True, top=False, right=False, direction="out")

                    # Add shared label and title
                    fig.set_size_inches(fig.get_size_inches()[0], len(phases_trials) * 4)
                    analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="supxlabel",
                            text=f"Time relative to \n{csus} onset (s)",
                            anchor_h="center",
                            anchor_v="bottom",
                            pad_pt=plot_config.pad_supxlabel,
                            text_kwargs={"fontweight": "bold"},
                        ),
                    )
                    
                    # Note: We rely on the last added supylabel mostly being aligned or needing a stable reference.
                    # Creating a dummy supylabel for Title/A-Label alignment if one doesn't exist globally?
                    # But we added per-block labels. Let's try to capture one.
                    # Ideally we should refrain from adding "C" if logic is fragile, but instruction says "make like run_traces".
                    # For consistency, we'll assume the top block's label is a good anchor.
                    
                    suptitle = analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="fig_title",
                            text=f"Example {cond.capitalize()} fish",
                            anchor_h="right",
                            anchor_v="top",
                            pad_pt=plot_config.pad_fig_title,
                            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "backgroundcolor": "none"},
                        ),
                    )
                    # No global supylabel stored easily. Skipping dynamic "C" placement to avoid overlap errors if not standard.
                    # Or we can insert just the suptitle logic as requested? 
                    # User asked for ".text and suptitle".
                    # I will simply place C at (0, 1) like typical if robust alignment isn't easy, 
                    # OR try to get the first axes ylabel.
                    
                    # Save raw heatmap
                    fig.savefig(
                        str(fig_path_raw),
                        format=INDIVIDUAL_TRIALS_FIG_FORMAT,
                        dpi=300,
                        transparent=False,
                        # bbox_inches="tight",
                    )
                
# region Scaled vigor
                # Scaled vigor heatmap
                if do_sc:
                    data_plot = data.copy(deep=True)
                    data_plot = data_plot[data_plot["Block name"] != ""]
                    data_plot = data_plot[
                        data_plot["Trial time (s)"].between(-INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S, INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S)
                    ]
                    data_plot.loc[~data_plot["Bout"], "Vigor (deg/ms)"] = np.nan

                    # Normalize vigor within trials
                    for t in data_plot["Trial number"].unique():
                        mask_trial = data_plot["Trial number"] == t
                        data_trial = data_plot.loc[mask_trial].copy(deep=True)

                        beg_bouts_trial, end_bouts_trial = analysis_utils.find_events(
                            data_trial, "Bout beg", "Bout end", "Trial time (s)"
                        )

                        for bout_b, bout_e in zip(beg_bouts_trial, end_bouts_trial):
                            mask_bout = data_trial["Trial time (s)"].between(bout_b, bout_e)
                            mean_vigor = data_trial.loc[mask_bout, "Vigor (deg/ms)"].mean()
                            data_trial.loc[mask_bout, "Vigor (deg/ms)"] = mean_vigor

                        mask_baseline = data_trial["Trial time (s)"] < -gen_config.baseline_window
                        baseline_vigor = data_trial.loc[mask_baseline, "Vigor (deg/ms)"].dropna().values

                        if baseline_vigor.size == 0:
                            continue

                        min_vigor_pre_stim, max_vigor_pre_stim = np.quantile(baseline_vigor, [0.1, 0.9])

                        if np.isnan(max_vigor_pre_stim) or min_vigor_pre_stim == max_vigor_pre_stim:
                            continue

                        data_trial.loc[mask_trial, "Vigor (deg/ms)"] = (
                            (data_trial.loc[mask_trial, "Vigor (deg/ms)"] - min_vigor_pre_stim)
                            / (max_vigor_pre_stim - min_vigor_pre_stim)
                        )

                        data_trial["Vigor (deg/ms)"] = data_trial["Vigor (deg/ms)"].clip(0, 1)
                        data_plot.loc[mask_trial] = data_trial

                    data_plot.loc[~data_plot["Bout"], "Vigor (deg/ms)"] = np.nan
                    data_plot = (
                        data_plot[["Trial time (s)", "Trial number", "Vigor (deg/ms)"]]
                        .pivot(index="Trial time (s)", columns="Trial number")
                        .reset_index()
                        .set_index("Trial time (s)")
                        .droplevel(0, axis=1)
                        .T
                    )

                    data_plot.columns = data_plot.columns.astype("int")

                    # Configure phase blocks based on stimulus type
                    if csus == "CS":
                        phases_trial_numbers = config.trials_cs_blocks_phases
                        phases_block_names = config.names_cs_blocks_phases
                    else:
                        phases_trial_numbers = config.trials_us_blocks_phases
                        phases_block_names = config.names_us_blocks_phases or ["Train"]

                    fig, axs = plt.subplots(
                        len(phases_trial_numbers),
                        1,
                        facecolor="white",
                        gridspec_kw={
                            "height_ratios": [len(b) for b in phases_trial_numbers],
                            "hspace": 0.025,
                        },
                        squeeze=False,
                        constrained_layout=False,
                        figsize=(5 / 2.54, 6 / 2.54),
                    )

                    supylabel = None
                    title_ax_idx = 0

                    # Render heatmap blocks with markers
                    if csus == "CS":
                        for b_i, b in enumerate(phases_block_names):
                            show_xticks = b_i == len(phases_block_names) - 1
                            sns.heatmap(
                                data_plot[data_plot.index.isin(phases_trial_numbers[b_i])],
                                cbar=False,
                                robust=False,
                                xticklabels=xtick_step_scaled if show_xticks else False,
                                yticklabels=False,
                                ax=axs[b_i][0],
                                clip_on=False,
                                vmin=0,
                                vmax=1,
                                rasterized=True,
                            )

                            trials_in_this_block = data_plot[data_plot.index.isin(phases_trial_numbers[b_i])].index
                            for tr_i, tr in enumerate(trials_in_this_block):
                                if tr in trials_to_mark:
                                    axs[b_i][0].plot(
                                        1.025,
                                        tr_i + 0.5,
                                        marker="<",
                                        color="k",
                                        transform=axs[b_i][0].get_yaxis_transform(),
                                        clip_on=False,
                                        zorder=20,
                                        markersize=3,
                                    )

                            axs[b_i][0].set_xlabel("")
                            axs[b_i][0].set_ylabel("")

                            xlims = axs[b_i][0].get_xlim()
                            middle = np.mean(xlims)
                            factor = (xlims[-1] - xlims[0]) / (INDIVIDUAL_TRIALS_WINDOW_DATA_PLOT_S * 2)

                            axs[b_i][0].axvline(
                                middle,
                                color=gen_config.plotting.cs_color,
                                alpha=0.75,
                                lw=2,
                                linestyle="-",
                                zorder=10,
                            )
                            axs[b_i][0].axvline(
                                middle + config.cs_duration * factor,
                                color=gen_config.plotting.cs_color,
                                alpha=0.75,
                                lw=2,
                                linestyle="-",
                                zorder=10,
                            )

                            axs[b_i][0].set_facecolor("k")
                            axs[b_i][0].set_rasterization_zorder(0)

                        axs[-1][0].tick_params(axis="both", which="both", bottom=True, top=False, right=False, direction="out")
                        analysis_utils.add_component(
                            fig,
                            analysis_utils.AddTextSpec(
                                component="supxlabel",
                                text=f"Time relative to \n{csus} onset (s)",
                                anchor_h="center",
                                anchor_v="bottom",
                                pad_pt=plot_config.pad_supxlabel,
                                text_kwargs={"fontweight": "bold"},
                            ),
                        )

                    else:
                        train_idx = 1 if len(phases_trial_numbers) > 1 else 0
                        sns.heatmap(
                            data_plot[data_plot.index.isin(phases_trial_numbers[train_idx])],
                            cbar=False,
                            robust=False,
                            xticklabels=xtick_step_scaled,
                            yticklabels=False,
                            ax=axs[train_idx][0],
                            clip_on=False,
                            vmin=0,
                            vmax=1,
                            rasterized=True,
                        )

                        trials_in_this_block = data_plot[data_plot.index.isin(phases_trial_numbers[train_idx])].index
                        # TODO: trial numbers need to be adjusted as the indicated refer to CS-aligned data
                        for tr_i, tr in enumerate(trials_in_this_block):
                            if tr in trials_to_mark:
                                axs[train_idx][0].plot(
                                    1.025,
                                    tr_i + 0.5,
                                    marker="<",
                                    color="k",
                                    transform=axs[train_idx][0].get_yaxis_transform(),
                                    clip_on=False,
                                    zorder=20,
                                    markersize=3,
                                )

                        xlims = axs[train_idx][0].get_xlim()
                        middle = np.mean(xlims)

                        axs[train_idx][0].set_facecolor("k")
                        axs[train_idx][0].set_rasterization_zorder(0)
                        
                        axs[train_idx][0].set_xlabel("")
                        axs[train_idx][0].set_ylabel("")

                        title_ax_idx = train_idx

                        for i in range(len(phases_trial_numbers)):
                            if i != train_idx:
                                axs[i][0].set_visible(False)

                        axs[train_idx][0].axvline(
                            middle,
                            color=gen_config.plotting.us_color,
                            alpha=0.75,
                            lw=2,
                            linestyle="-",
                            zorder=10,
                        )

                        cs_rows_total = sum(len(block) for block in config.trials_cs_blocks_phases)
                        us_rows_total = sum(len(block) for block in phases_trial_numbers)
                        train_rows = len(trials_in_this_block)
                        train_block_len = len(phases_trial_numbers[train_idx])
                        if cs_rows_total > 0 and us_rows_total > 0 and train_rows > 0 and train_block_len > 0:
                            # Match CS row height by resizing the US train axis.
                            pos = axs[train_idx][0].get_position()
                            total_height = pos.height * (us_rows_total / train_block_len)
                            target_height = total_height * (train_rows / cs_rows_total)
                            new_y0 = pos.y0 + (pos.height - target_height) / 2
                            new_y0 = max(0, min(new_y0, 1 - target_height))
                            axs[train_idx][0].set_position([pos.x0, new_y0, pos.width, target_height])

                        axs[train_idx][0].tick_params(axis="both", which="both", bottom=True, top=False, right=False, direction="out")
                        analysis_utils.add_component(
                            fig,
                            analysis_utils.AddTextSpec(
                                component="supxlabel",
                                text=f"Time relative to \n{csus} onset (s)",
                                anchor_h="center",
                                anchor_v="bottom",
                                pad_pt=plot_config.pad_supxlabel,
                                text_kwargs={"fontweight": "bold"},
                            ),
                        )


                    for b_i, b in enumerate(phases_block_names):
                        analysis_utils.add_component(
                            axs[b_i][0],
                            analysis_utils.AddTextSpec(
                                component="supylabel",
                                text=b,
                                anchor_h="left",
                                anchor_v="center",
                                pad_pt=(0, 0),
                                text_kwargs={"rotation": 90, "fontweight": "bold", "color": "k"},
                            ),
                        )

                    # Add shared title and panel label
                    analysis_utils.add_component(
                        axs[title_ax_idx][0],
                        analysis_utils.AddTextSpec(
                            component="axis_title",
                            text=f"Example {cond.capitalize()} fish",
                            anchor_h="right",
                            anchor_v="top",
                            pad_pt=(0, 0),
                            text_kwargs={"fontsize": plot_config.figure_titlesize, "backgroundcolor": "none", "color": "k"},
                        ),
                    )

                    analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="text",
                            text="E",
                            anchor_h="left",
                            anchor_v="top",
                            pad_pt=plot_config.pad_panel_label,
                            text_kwargs={"fontsize": plot_config.fontsize_axis_label, "fontweight": "bold", "backgroundcolor": "none"},
                        ),
                    )

                    # Save scaled heatmap
                    fig.savefig(
                        str(fig_path_sc),
                        format=INDIVIDUAL_TRIALS_FIG_FORMAT,
                        dpi=1000,
                        transparent=False,
                        # bbox_inches="tight",
                    )
    # endregion                

                # Normalized vigor summary plot
                if do_norm:
                    data_plot = data.copy(deep=True)
                    data_plot = data_plot[data_plot["Block name"] != ""]
                    data_plot.loc[~data_plot["Bout"], "Vigor (deg/ms)"] = np.nan

                    trials_blocks = config.blocks_dict["blocks 10 trials"][csus]["trials in each block"]
                    trial_numbers = trials_blocks[-1][-1] - trials_blocks[0][0] + 1

                    data_trial_bef = np.ones(trial_numbers)
                    data_trial_dur = np.ones(trial_numbers)
                    data_trial_nv = np.ones(trial_numbers)

                    trials_range = np.arange(trials_blocks[0][0], trials_blocks[-1][-1] + 1)

                    for t_i, t in enumerate(trials_range):
                        mask_trial = data_plot["Trial number"] == t
                        data_trial = data_plot.loc[mask_trial]

                        if csus == "CS":
                            data_trial_bef[t_i] = data_trial.loc[
                                data_trial["Trial time (s)"].between(-gen_config.baseline_window, 0),
                                "Vigor (deg/ms)",
                            ].mean()
                            data_trial_dur[t_i] = data_trial.loc[
                                data_trial["Trial time (s)"].between(cr_window[0], cr_window[1]),
                                "Vigor (deg/ms)",
                            ].mean()
                        else:
                            data_trial_bef[t_i] = data_trial.loc[
                                data_trial["Trial time (s)"].between(-gen_config.baseline_window - cr_window[1], -cr_window[1]),
                                "Vigor (deg/ms)",
                            ].mean()
                            data_trial_dur[t_i] = data_trial.loc[
                                data_trial["Trial time (s)"].between(cr_window[0] - cr_window[1], 0),
                                "Vigor (deg/ms)",
                            ].mean()

                        data_trial_nv[t_i] = data_trial_dur[t_i] / data_trial_bef[t_i]

                    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, facecolor="white")

                    for t_i in range(trial_numbers):
                        axs[1].plot(
                            t_i + 1, data_trial_nv[t_i], ".", color="k", alpha=0.8, lw=0, markersize=10, clip_on=False
                        )

                    axs[0].legend(["Before CS onset", "After CS onset"], loc="upper right", frameon=False)

                    for ax in axs:
                        ax.spines[["top", "right"]].set_visible(False)
                        ax.locator_params(axis="y", tight=False, nbins=4)
                        ax.tick_params(axis="both", which="both", bottom=True, top=False, right=False, direction="out")

                    max_ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
                    axs[0].set_ylim((None, max_ylim))

                    axs[1].set_yticks([0.75, 1, 1.25])
                    axs[1].set_yticklabels(["0.75", "1.0", "1.25"])
                    axs[1].set_ylim(0.75, 1.25)

                    axs[1].axhline(1, color="k", alpha=0.8, lw=1)
                    axs[1].axhline(0, color="k", alpha=0.8, lw=1)

                    # Add shared labels and save
                    fig.set_size_inches(fig.get_size_inches()[0], 7)
                    analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="supylabel",
                            text="Average vigor (deg/ms)",
                            anchor_h="left",
                            anchor_v="center",
                            pad_pt=plot_config.pad_supylabel,
                            text_kwargs={"fontweight": "bold"},
                        ),
                    )
                    analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="supylabel",
                            text="Normalized vigor (AU)",
                            anchor_h="left",
                            anchor_v="center",
                            pad_pt=plot_config.pad_supylabel,
                            text_kwargs={"fontweight": "bold"},
                        ),
                    )
                    analysis_utils.add_component(
                        fig,
                        analysis_utils.AddTextSpec(
                            component="supxlabel",
                            text="Trial number",
                            anchor_h="center",
                            anchor_v="bottom",
                            pad_pt=plot_config.pad_supxlabel,
                            text_kwargs={"fontweight": "bold"},
                        ),
                    )
                    fig.savefig(str(fig_path_norm), format=INDIVIDUAL_TRIALS_FIG_FORMAT, dpi=300, transparent=False)
    # Cleanup
    plt.close("all")
    print("INDIVIDUAL TRIALS FINISHED")
# endregion Heatmaps

# region Bouts before and after US
def run_bout_zoom() -> None:
    """
    Generate a zoomed-in plot of tail angle around a specific bout.
    
    Useful for detailed examination of individual behavioral responses.
    """
    # Entry banner for the pipeline stage
    print("\n" + "="*80)
    print("RUNNING BOUT ZOOM")
    print("="*80)

    plot_config = get_plot_config()

    # Resolve experiment paths and fish file
    config, _, path_orig_pkl = resolve_experiment_paths(BOUT_ZOOM_EXPERIMENT_TYPE)
    # Use the first traces experiment to resolve the output directory when multiple are configured.
    traces_exp = TRACES_EXPERIMENT_TYPE
    if isinstance(traces_exp, (list, tuple, np.ndarray)):
        traces_exp = traces_exp[0] if traces_exp else BOUT_ZOOM_EXPERIMENT_TYPE
    _, traces_output_dir, _ = resolve_experiment_paths(str(traces_exp))
    try:
        fish_path = resolve_fish_path(path_orig_pkl, BOUT_ZOOM_FISH_ID)
    except FileNotFoundError as exc:
        print(str(exc))
        return

    # Build output path
    stem_fish_path_orig = fish_path.stem
    save_path = traces_output_dir / f'{stem_fish_path_orig}.{BOUT_ZOOM_FORMAT}'

    # Load data for selected trial
    data = pd.read_pickle(fish_path, compression='infer')
    data = data[data['Trial type'] == BOUT_ZOOM_CSUS]
    data = data[data['Trial number'] == BOUT_ZOOM_TRIAL_NUMBER]
    
    if data.empty:
        print(f"No data for Trial {BOUT_ZOOM_TRIAL_NUMBER} in {fish_path}")
        return

    data.reset_index(drop=True, inplace=True)
    
    # Time conversion
    time_col = 'Trial time (s)'
    if gen_config.time_trial_frame_label in data.columns:
        data[time_col] = data[gen_config.time_trial_frame_label] / gen_config.expected_framerate
    
    # Filter time window
    data = data.loc[data[time_col].between(BOUT_ZOOM_TIME_BEF, BOUT_ZOOM_TIME_AFT)]
    
    # Baseline subtraction
    tail_angle = gen_config.tail_angle_label
    if tail_angle in data.columns:
        data[tail_angle] -= data[tail_angle].mean()

    # Determine stimulus color for marker
    stim_color = (
        gen_config.plotting.us_color if BOUT_ZOOM_CSUS == 'US' 
        else gen_config.plotting.cs_color
    )

    # Plotting
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, facecolor='white', figsize=(6.5/2.54, 3.7/2.54))
    
    ax.plot(data[time_col], data[tail_angle], 'k', alpha=1, lw=1, zorder=0, clip_on=False)

    # Highlight bouts
    if 'Bout' in data.columns:
        ax.fill_between(
            data[time_col],
            BOUT_ZOOM_Y_LIM[0],
            BOUT_ZOOM_Y_LIM[1],
            where=data['Bout'],
            color='gray',
            alpha=0.2,
            lw=0,
            transform=ax.get_xaxis_transform(),
            zorder=-1
        )
    
    # Add stimulus onset marker at t=0
    ax.axvline(x=0, color=stim_color, alpha=0.75, lw=2, linestyle='--', zorder=10)
    
    # Axis Styling
    ax.tick_params(axis='both', which='both', bottom=True, top=False, right=False, direction='out')
    ax.set_xlim((BOUT_ZOOM_TIME_BEF, BOUT_ZOOM_TIME_AFT))
    ax.set_ylim(BOUT_ZOOM_Y_LIM)
    ax.set_xticks([BOUT_ZOOM_TIME_BEF, -2.5, 0, 2.5, BOUT_ZOOM_TIME_AFT])
    ax.set_yticks([BOUT_ZOOM_Y_LIM[0], 0, BOUT_ZOOM_Y_LIM[1]])
    # Add shared labels
    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supxlabel",
            text=f"Time relative to {BOUT_ZOOM_CSUS} onset (s)",
            anchor_h="center",
            anchor_v="bottom",
            pad_pt=plot_config.pad_supxlabel,
            text_kwargs={"fontweight": "bold"},
        ),
    )
    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supylabel",
            text="Tail angle (deg)",
            anchor_h="left",
            anchor_v="center",
            pad_pt=plot_config.pad_supylabel,
            text_kwargs={"fontweight": "bold"},
        ),
    )

    # Save output
    fig.savefig(str(save_path), format=BOUT_ZOOM_FORMAT, dpi=100, transparent=False, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: {save_path}")
# endregion Bouts before and after US

# region Tail trajectory
def run_trajectory() -> None:
    """
    Generate tail trajectory heatmaps showing spatial distribution of tail positions.
    
    Creates before/after heatmaps for single fish and pooled data.
    """
    # Entry banner for the pipeline stage
    print("\n" + "="*80)
    print("RUNNING  TRAJECTORY")
    print("="*80)

    # Resolve paths and output directory
    config, _, path_orig_pkl = resolve_experiment_paths(TRAJ_EXPERIMENT_TYPE)
    # Use the first traces experiment to resolve the output directory when multiple are configured.
    traces_exp = TRACES_EXPERIMENT_TYPE
    if isinstance(traces_exp, (list, tuple, np.ndarray)):
        traces_exp = traces_exp[0] if traces_exp else TRAJ_EXPERIMENT_TYPE
    _, traces_output_dir, _ = resolve_experiment_paths(str(traces_exp))
    fish_path = resolve_fish_path(path_orig_pkl, TRAJ_FISH_ID)
    trajectory_dir = traces_output_dir / "tail_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    data = pd.read_pickle(str(fish_path), compression="infer")
    data = analysis_utils.standardize_stim_cols(data)
    # Filter by trial type if present
    if "Trial type" in data.columns:
        data = data[data["Trial type"] == TRAJ_CSUS].copy()
    
    # Ensure time is in seconds
    if "Trial time (s)" not in data.columns:
        data = analysis_utils.convert_time_from_frame_to_s(data)
    
    # Identify tail angle columns
    angle_cols = analysis_utils.sorted_angle_cols(data)
    if not angle_cols:
        raise ValueError("No angle columns found for tail trajectory plot.")

    # Optional: keep only bout frames
    if "Bout" in data.columns:
        data = data.loc[data["Bout"]].copy()
        data.drop(columns=["Bout"], inplace=True, errors="ignore")

    # Adjust trial numbers
    trial_numbers = pd.to_numeric(data["Trial number"], errors="coerce")
    data = data.loc[trial_numbers.notna()].copy()
    data["Trial number"] = trial_numbers.loc[trial_numbers.notna()].astype(int) - TRAJ_TRIAL_NUMBER_OFFSET
    data = data.loc[data["Trial number"] > 0]

    # Filter to target trials and time window
    data = data.loc[
        data["Trial number"].between(TRAJ_TRIALS_SINGLE[0], TRAJ_TRIALS_SINGLE[1])
        & data["Trial time (s)"].between(-TRAJ_TIME_INTERVAL_S, TRAJ_TIME_INTERVAL_S),
        angle_cols + ["Trial time (s)"],
    ].reset_index(drop=True)

    # Split before/after
    data_bef = data.loc[data["Trial time (s)"].between(*TRAJ_BEFORE_WINDOW_S), angle_cols]
    data_aft = data.loc[data["Trial time (s)"].between(*TRAJ_AFTER_WINDOW_S), angle_cols]

    data_bef = np.deg2rad(data_bef.to_numpy(dtype=np.float32))
    data_aft = np.deg2rad(data_aft.to_numpy(dtype=np.float32))

    # Generate heatmaps
    for label, angle_data in zip(("before", "after"), (data_bef, data_aft)):
        if angle_data.size == 0:
            continue
        tail_xy = analysis_utils.tail_angles_to_xy(angle_data, TRAJ_SEGMENT_LEN, TRAJ_TAIL_SCALE)
        x_fine, y_fine = analysis_utils.interpolate_tail_xy(tail_xy, TRAJ_NUM_FINE)
        output_path = trajectory_dir / f"{fish_path.stem}_{TRAJ_TRIALS_SINGLE}_{label}.png"
        analysis_utils.plot_tail_histogram(
            x_fine,
            y_fine,
            output_path,
            TRAJ_SINGLE_XLIM,
            TRAJ_SINGLE_YLIM,
            TRAJ_SINGLE_VMIN_DIV,
            TRAJ_SINGLE_VMAX_DIV,
            TRAJ_HIST_BINS,
        )
    # Cleanup
    plt.close("all")
    print(" TRAJECTORY FINISHED")
# endregion Tail trajectory

# endregion Pipeline functions


# region Main
def main() -> None:
    """Execute the selected pipeline stages."""
    # Dispatch selected pipeline stages. Each stage is wrapped so one failure doesn't
    # prevent running the remaining stages during exploratory work.

    # Fig 1C,D
    if RUN_TRACES:
        try:
            run_traces()
        except Exception as e:
            print(f"Error in  TRACES: {e}")

    # Fig 1E-H
    if RUN_INDIVIDUAL_TRIALS:
        try:
            run_individual_trials()
        except Exception as e:
            print(f"Error in INDIVIDUAL TRIALS: {e}")

    # Optional/extra figures
    if RUN_BOUT_ZOOM:
        try:
            run_bout_zoom()
        except Exception as e:
            print(f"Error in BOUT ZOOM: {e}")
    if RUN_TRAJECTORY:
        try:
            run_trajectory()
        except Exception as e:
            print(f"Error in  TRAJECTORY: {e}")

if __name__ == "__main__":
    main()
# endregion

