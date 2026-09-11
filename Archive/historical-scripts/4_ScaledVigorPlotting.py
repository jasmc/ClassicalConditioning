"""
Scaled Vigor Plotting Pipeline
===============================

Build and plot scaled-vigor heatmaps and lineplots from pooled all-fish data.
"""

# %%
# region Imports
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

# Add the repository root containing shared modules to the Python path.
if "__file__" in globals():
    module_root = Path(__file__).resolve()
else:
    module_root = Path.cwd()

import analysis_utils
import file_utils
import plotting_style
from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as gen_config
from plotting_style import get_plot_config

pd.set_option("mode.copy_on_write", True)

# Set plotting style (shared across analysis scripts).
plotting_style.set_plot_style(use_constrained_layout=False)
# endregion Imports


# %%
# region Parameters
# ------------------------------------------------------------------------------
# Pipeline Control Flags
# ------------------------------------------------------------------------------
RUN_BUILD_POOLED_OUTPUTS = False
RUN_COUNT_HEATMAP = False
RUN_SV_HEATMAP_RENDERING = False
RUN_SV_LINEPLOTS_INDIVIDUAL = True
RUN_SV_LINEPLOTS_CATCH_TRIALS = False
RUN_SV_LINEPLOTS_BLOCKS = False

# ------------------------------------------------------------------------------
# Shared Parameters
# ------------------------------------------------------------------------------
EXPERIMENT = ExperimentType.ALL_DELAY.value

# Apply per-experiment discarded fish list if present under "Processed data".
APPLY_FISH_DISCARD = True

csus = "CS"  # Stimulus alignment: "CS" or "US".
x_lim = (-20, 20)
window_data_plot = [-20, 20]

y_lim = (-0.1, 0.1)
y_clip = [-0.14, 0.14]
n_boot = 10

# ------------------------------------------------------------------------------
# Build Pooled Outputs Parameters
# ------------------------------------------------------------------------------
binning_windows = [0.5, 1.0]
default_binning_window = 1.0

# ------------------------------------------------------------------------------
# Heatmap Parameters
# ------------------------------------------------------------------------------
interval_between_xticks_heatmap = 20

# ------------------------------------------------------------------------------
# Count Heatmap Parameters
# ------------------------------------------------------------------------------
count_heatmap_binning_window = 0.5

# ------------------------------------------------------------------------------
# SV Heatmap Rendering Parameters
# ------------------------------------------------------------------------------
binning_window_heatmap = 0.5
frmt = "svg"

# ------------------------------------------------------------------------------
# SV Lineplot Parameters
# ------------------------------------------------------------------------------
DO_TIME_BIN = True
DO_BASELINE_SUBTRACT = True

trial_names = ["Train trial 11", "Train trial 25", "Train trial 39", "Train trial 45", "Test trial 1"]
catch_trials_training = [25, 39, 53, 59]
catch_trials_test = 65
catch_trials_retraining = [11 + 30 + 64, 25 + 30 + 64]
retraining_trial_names = ["Re-Train trial 11", "Re-Train trial 25"]
# endregion Parameters


# %%
# region Plot Formatting
from plotting_style import (HEATMAP_FIGSIZE, HEATMAP_GRIDSPACE,
                            HEATMAP_STYLE_KW, HEATMAP_TICK_KW, LEGEND_KW,
                            LINEPLOT_FIG_KW, LINEPLOT_STYLE_KW, SAVEFIG_KW)

# endregion Plot Formatting


# %%
# region Context Setup
config = get_experiment_config(EXPERIMENT)

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
    _,
    _,
    _,
    _,
    path_pooled_vigor_fig,
    _,
    path_orig_pkl,
    path_all_fish,
    path_pooled_data,
) = file_utils.create_folders(config.path_save)

# All figures from this script should be saved under a "Scaled vigor" subfolder
# inside the experiment's pooled-figure output directory.
path_scaled_vigor_fig = path_pooled_vigor_fig / "Scaled vigor"
path_scaled_vigor_fig.mkdir(parents=True, exist_ok=True)

stim_duration = config.cs_duration if csus == "CS" else gen_config.us_duration
stim_color = gen_config.plotting.cs_color if csus == "CS" else gen_config.plotting.us_color
time_frame_col = gen_config.time_trial_frame_label

# Load discarded fish IDs (if configured)
discard_file = path_processed_data / "Discarded_fish_IDs.txt"

fish_ids_to_discard: list[str] = []
discard_source: Path | None = None

if APPLY_FISH_DISCARD and discard_file.exists():
    fish_ids_to_discard = file_utils.load_discarded_fish_ids(discard_file)
    discard_source = discard_file

if discard_source is not None:
    try:
        src_str = str(discard_source)
    except Exception:
        src_str = "<unknown>"
    print(f"  Loaded {len(fish_ids_to_discard)} discarded fish IDs from: {src_str}")


def filter_discarded_fish_ids(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Drop rows whose Fish ID is in the discarded list (if present).
    
    Prints unique fish count before and after discarding.
    """
    if df is None or df.empty:
        return df
    
    print(df.columns)
    
    fish_col = "Fish"
    
    before = df[fish_col].nunique()
    prefix = f"  [{source}] " if source else "  "
    print(f"{prefix}Fish unique before discard: {before}")
    
    if not APPLY_FISH_DISCARD or not fish_ids_to_discard:
        return df
    
    df_filtered = df[~df[fish_col].isin(fish_ids_to_discard)].copy()
    after = df_filtered[fish_col].nunique()
    print(f"{prefix}Fish unique after discard: {after}")
    return df_filtered
# endregion Context Setup


# %%
# region Helper Functions
def densify_sparse(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any SparseDtype columns to dense; robust across pandas versions."""
    # Some pooled outputs are stored with pandas SparseDtype to reduce disk size.
    # Downstream plotting (seaborn/matplotlib) and some groupby operations can behave
    # inconsistently with sparse columns, so we densify defensively.
    try:
        return df.sparse.to_dense()
    except Exception:
        pass

    for col in df.columns:
        ser = df[col]
        dtype = ser.dtype
        try:
            if isinstance(dtype, pd.SparseDtype) or hasattr(ser, "sparse"):
                try:
                    df[col] = ser.sparse.to_dense()
                except Exception:
                    # Some pandas versions don't expose .subtype in type stubs; use getattr for robustness.
                    subtype = getattr(dtype, "subtype", None) if isinstance(dtype, pd.SparseDtype) else None
                    if subtype is not None:
                        df[col] = ser.astype(subtype)
                    else:
                        df[col] = pd.Series(ser.to_numpy(), index=df.index)
        except Exception:
            df[col] = pd.Series(np.asarray(ser), index=df.index)
    return df


def compute_time_bins(x_lim, binning_window):
    """Compute time bins once and reuse."""
    # We extend the requested window slightly so binning covers edges cleanly.
    # Bins are built separately for negative and positive time to guarantee a bin edge at 0.
    bin_start_target = x_lim[0] - 1
    bin_end_target = x_lim[1] + 2
    start_neg = np.floor(bin_start_target / binning_window) * binning_window
    neg_bins = np.arange(start_neg, 0 + binning_window, binning_window)
    end_pos = np.ceil(bin_end_target / binning_window) * binning_window
    pos_bins = np.arange(0, end_pos + binning_window / 2, binning_window)
    return np.unique(np.concatenate((neg_bins, pos_bins))).tolist()


def plot_cond_line(ax, data, x_col, y_col, cond, color_rgb_list, n_boot, ci=95):
    """Centralized seaborn line plot for consistency."""
    return sns.lineplot(
        data=data,
        x=x_col,
        y=y_col,
        color=color_rgb_list,
        markerfacecolor=color_rgb_list,
        label=cond,
        errorbar=("ci", ci),
        n_boot=n_boot,
        **LINEPLOT_STYLE_KW,
        ax=ax,
    )


def style_axes(ax, stim_color, stim_duration, x_lim, y_lim, show_xlabel=False, outward=0):
    """Apply consistent axis styling."""
    # Reference lines: baseline and stimulus on/off.
    ax.axhline(0, color="k", alpha=0.5, lw=0.5)
    ax.axvline(0, color=stim_color, alpha=0.7, lw=1, linestyle="-")
    ax.axvline(stim_duration, color=stim_color, alpha=0.7, lw=1, linestyle="-")

    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_yticks([y_lim[0], 0, y_lim[1]])

    ax.set_xlim(x_lim[0] - 0.5, x_lim[1] + 0.5)
    ax.set_xticks(ticks=np.arange(x_lim[0], x_lim[1] + 1, 20))
    # Minor ticks improve temporal readability without adding label clutter.
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(which="minor", length=1, width=0.3)
    ax.tick_params(axis="x", which="both", bottom=True, top=False)

    ax.spines["bottom"].set_position(("outward", outward))
    ax.set_xlabel("")
    ax.set_ylabel("")

    if not show_xlabel:
        ax.set_xticklabels([])
        ax.spines["bottom"].set_visible(False)


def apply_baseline_and_clip(df, y_clip):
    """Apply optional baseline subtraction and clip scaled vigor for display."""
    # For line plots, subtract the pre-stimulus baseline median to focus on modulation,
    # and clip to keep axes comparable across conditions.
    if DO_BASELINE_SUBTRACT:
        baseline_mask = df["Trial time (s)"].between(-gen_config.baseline_window, 0)
        if baseline_mask.any():
            df["Scaled vigor (AU)"] -= df.loc[baseline_mask, "Scaled vigor (AU)"].median()
    df["Scaled vigor (AU)"] = df["Scaled vigor (AU)"].clip(y_clip[0], y_clip[1])
    return df


def make_lineplot_axes(nrows, figsize, sharex=False, sharey=True):
    """Create lineplot figures with shared defaults."""
    return plt.subplots(
        nrows,
        1,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        **LINEPLOT_FIG_KW,
    )


def add_condition_legend(fig, handles, cond_types):
    """Add a shared legend for condition lines outside the axes."""
    # If nothing was drawn (e.g., missing data), don't add an empty legend.
    if not handles:
        return
    
    kwargs = LEGEND_KW.copy()
    kwargs.update({
        "bbox_to_anchor": (1, 1),
        "loc": "upper left",
        "bbox_transform": fig.transFigure,
    })
    
    fig.legend(handles, [cond_title(c) for c in cond_types], **kwargs)


def remove_axis_legend(ax):
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def make_heatmap_axes(
    phases_trial_numbers,
    ncols=1,
    sharey: bool | Literal["none", "all", "row", "col"] = False,
    sharex: bool | Literal["none", "all", "row", "col"] = False,
):
    """Create stacked heatmap axes with block-sized height ratios."""
    height_ratios = [len(b) for b in phases_trial_numbers]
    fig_width = HEATMAP_FIGSIZE[0] * max(ncols, 1)
    fig, axs = plt.subplots(
        len(phases_trial_numbers),
        ncols,
        facecolor="white",
        gridspec_kw={"height_ratios": height_ratios, "hspace": HEATMAP_GRIDSPACE, "wspace": 0.25},
        squeeze=False,
        constrained_layout=False,
        figsize=(fig_width, HEATMAP_FIGSIZE[1]),
        sharey=sharey,
        sharex=sharex,
    )
    return fig, axs


def add_heatmap_stimulus_lines(ax, stim_color, stim_duration, window_data_plot):
    """Draw stimulus onset/offset lines onto heatmap axes."""
    xlims = ax.get_xlim()
    middle = np.mean(xlims)
    factor = (xlims[-1] - xlims[0]) / (window_data_plot[1] - window_data_plot[0])
    ax.axvline(middle, color=stim_color, alpha=0.7, lw=1.5, linestyle="-")
    ax.axvline(middle + stim_duration * factor, color=stim_color, alpha=0.7, lw=1.5, linestyle="-")


def render_heatmap_blocks(
    axs,
    data_cond,
    phases_trial_numbers,
    phases_block_names,
    ticks,
    stim_color,
    stim_duration,
    window_data_plot,
    show_ylabel=True,
    show_yticklabels=False,
    **heatmap_kwargs,
):
    """Render stacked heatmap blocks with consistent formatting."""
    # Each block corresponds to a training phase (e.g. Pre-train / Train / Test).
    # Only the bottom block shows x tick labels to reduce clutter.

    kwargs = {k: v for k, v in HEATMAP_STYLE_KW.items() if k != "yticklabels"}
    kwargs.update(heatmap_kwargs)

    # If norm is provided, remove vmin/vmax to avoid conflicts/warnings
    if "norm" in kwargs:
        kwargs.pop("vmin", None)
        kwargs.pop("vmax", None)

    for b_i, trials in enumerate(phases_trial_numbers):
        show_xticks = b_i == len(phases_trial_numbers) - 1
        sns.heatmap(
            select_trials(data_cond, trials),
            xticklabels=ticks if show_xticks else False,
            yticklabels=show_yticklabels,
            ax=axs[b_i][0],
            **kwargs,
        )
        if show_xticks:
            # Normalize tick label formatting (avoid long float strings).
            labels = []
            for lbl in axs[b_i][0].get_xticklabels():
                text = lbl.get_text()
                if text == "":
                    labels.append("")
                    continue
                try:
                    labels.append(f"{float(text):g}")
                except ValueError:
                    labels.append(text)
            axs[b_i][0].set_xticklabels(labels)


        axs[b_i][0].set_xlabel("")
        axs[b_i][0].set_ylabel("")
        add_heatmap_stimulus_lines(axs[b_i][0], stim_color, stim_duration, window_data_plot)
        axs[b_i][0].set_facecolor("k")









def cond_color(cond):
    color = config.cond_dict.get(cond, {}).get("color")
    if color is None:
        return None
    return [x / 256 for x in color]


def cond_title(cond):
    return config.cond_dict.get(cond, {}).get("name", cond)


def subset_columns(df, columns):
    existing = [c for c in columns if c in df.columns]
    return df[existing].copy()


def ensure_time_seconds(df):
    if "Trial time (s)" not in df.columns:
        df = analysis_utils.convert_time_from_frame_to_s(df)
    return df


def select_trials(df, trials):
    if isinstance(df.index, pd.MultiIndex) and "Trial number" in df.index.names:
        mask = df.index.get_level_values("Trial number").isin(trials)
        return df.loc[mask]
    return df[df.index.isin(trials)]


def resolve_catch_trials():
    trials_to_use = catch_trials_training + [catch_trials_test]
    trial_names_here = list(trial_names)

    if "long" in EXPERIMENT.lower():
        trials_to_use = trials_to_use + catch_trials_retraining
        trial_names_here = trial_names_here + retraining_trial_names

    if len(trial_names_here) != len(trials_to_use):
        trial_names_here = [f"Trial {t}" for t in trials_to_use]

    return trials_to_use, trial_names_here


def numeric_time_columns(columns):
    time_vals = pd.to_numeric(columns, errors="coerce")
    return [col for col, val in zip(columns, time_vals) if pd.notna(val)]


def infer_binning_window_from_columns(columns, fallback):
    time_cols = numeric_time_columns(columns)
    if len(time_cols) >= 2:
        time_vals = np.array(time_cols, dtype=float)
        diffs = np.diff(np.sort(time_vals))
        diffs = diffs[diffs > 0]
        if diffs.size > 0:
            return float(diffs[0])
    return fallback


def find_heatmap_paths(patterns):
    paths = [*Path(path_pooled_data).glob("*.pkl")]
    for pattern in patterns:
        matched = [
            path for path in paths
            if pattern in path.stem and _stem_matches_csus(path.stem, csus)
        ]
        if matched:
            return matched
    return []


def _stringify_for_filename(value) -> str:
    """Convert common objects (lists/arrays) into filename-friendly strings."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return "-".join(str(v) for v in value)
    return str(value)


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename component for Windows filesystems."""
    # Windows disallowed characters: <>:"/\|?*
    invalid = '<>:"/\\|?*'
    out = "".join("_" if ch in invalid else ch for ch in str(name))
    # Avoid trailing spaces/dots which Windows strips/blocks.
    out = out.strip().rstrip(".")
    # Keep filenames reasonably compact.
    out = " ".join(out.split())
    return out if out else "figure"


SELECTED_FISH_SUFFIX = "_selectedFish"


def _maybe_append_selected_fish_stem(stem: str) -> str:
    """Append `_selectedFish` to a filename stem when discard is enabled."""
    if not APPLY_FISH_DISCARD:
        return str(stem)
    stem = str(stem)
    return stem if stem.endswith(SELECTED_FISH_SUFFIX) else f"{stem}{SELECTED_FISH_SUFFIX}"


def _stem_matches_csus(stem: str, csus_value: str) -> bool:
    """Match stems ending with _{CSUS}[_{selected/allFish}] without fragile splits."""
    stem = str(stem)
    csus_value = str(csus_value)
    return (
        stem.endswith(f"_{csus_value}")
        or stem.endswith(f"_{csus_value}{SELECTED_FISH_SUFFIX}")
        or stem.endswith(f"_{csus_value}_allFish")
    )


def save_fig(fig: Figure, stem: str, frmt: str) -> Path:
    """Save a figure under path_scaled_vigor_fig with consistent naming."""
    safe_stem = _maybe_append_selected_fish_stem(_sanitize_filename(stem))
    safe_frmt = str(frmt).lstrip(".")
    save_path = path_scaled_vigor_fig / f"{safe_stem}.{safe_frmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), format=safe_frmt, **SAVEFIG_KW)
    return save_path
# endregion Helper Functions


# %%
# region Pipeline Functions

# region build_pooled_outputs
def run_build_pooled_outputs():
    """Load per-fish data and aggregate into pooled heatmap/lineplot outputs."""
    selected_suffix = SELECTED_FISH_SUFFIX if APPLY_FISH_DISCARD else ""
    all_data_csus_paths = sorted(
        [path for path in Path(path_all_fish).glob("*.pkl") if _stem_matches_csus(path.stem, csus)]
    )
    if not all_data_csus_paths:
        print(f"No data found in {path_all_fish} for {csus}.")

    cond_types_here = sorted({path.stem.split("_")[-2] for path in all_data_csus_paths})

    cols_to_use = [
        "Exp.",
        time_frame_col,
        "CS beg",
        "CS end",
        "Trial number",
        "Block name",
        "Scaled vigor (AU)",
        "Fish",
        "Bout",
    ]

    print(csus)
    datasets = []
    for path in tqdm(all_data_csus_paths, desc="Loading Data"):

        print(path)
        try:
            df = pd.read_pickle(path, compression="gzip")
            print(df)
            print(df.columns)
            print(df["Fish"].nunique())
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue

        df = filter_discarded_fish_ids(df, source=path.stem)
        print(df["Fish"].nunique())

        # return
    
        if "Block name" in df.columns and df["Block name"].isna().all():
            df["Trial type"] = csus
            df = analysis_utils.identify_blocks_trials(df, config.blocks_dict)

        if "Bout" in df.columns:
            df.loc[~df["Bout"], ["Vigor (deg/ms)", "Scaled vigor (AU)"]] = np.nan

        df = subset_columns(df, cols_to_use)
        if "Exp." not in df.columns:
            df["Exp."] = path.stem.split("_")[-2]

        cond = df["Exp."].unique()[0]
        num_fish = df["Fish"].nunique() if "Fish" in df.columns else np.nan
        print(f"Processing condition: {cond}, Fish: {num_fish}")

        with open(path_scaled_vigor_fig / f"1_Heatmap notes{selected_suffix}.txt", "a") as file:
            file.write(f"Number fish in {cond}: {num_fish}\n\n")

        df = ensure_time_seconds(df)
        df = densify_sparse(df)

        if "Trial time (s)" not in df.columns:
            print(f"Skipping {path.name}: missing Trial time (s).")
            continue

        datasets.append({"data": df, "cond": cond, "num_fish": num_fish})

    data_plot_line = None
    for binning_window in binning_windows:
        print(f"Processing {binning_window}s bins...")
        bin_edges = compute_time_bins(x_lim, binning_window)
        list_heatmap_count, list_heatmap_sv, list_line = [], [], []

        for ds in datasets:
            df = ds["data"].copy(deep=True)
            cond = ds["cond"]
            num_fish = int(ds["num_fish"]) if pd.notna(ds["num_fish"]) else 1
            num_fish = max(num_fish, 1)

            if "Trial time (s)" not in df.columns or "Trial number" not in df.columns:
                continue
            if "Block name" not in df.columns:
                df["Block name"] = ""

            df_agg = (
                df.groupby(["Trial time (s)", "Trial number", "Block name"], observed=True)
                .agg(Scaled_vigor=("Scaled vigor (AU)", "median"), Count=("Scaled vigor (AU)", "count"))
                .reset_index()
                .rename(columns={"Scaled_vigor": "Scaled vigor (AU)"})
            )

            df_agg["bin"] = pd.cut(df_agg["Trial time (s)"], bin_edges, include_lowest=True)

            df_binned = (
                df_agg.groupby(["bin", "Trial number", "Block name"], observed=True)
                .agg(Scaled_vigor=("Scaled vigor (AU)", "mean"), Count=("Count", "sum"))
                .reset_index()
                .rename(columns={"Scaled_vigor": "Scaled vigor (AU)"})
            )
            df_binned["Trial time (s)"] = df_binned["bin"].apply(lambda x: x.mid).astype(float)
            df_binned = df_binned.drop(columns=["bin"]).dropna(subset=["Trial time (s)"])
            df_binned["Count"] = df_binned["Count"] / num_fish
            df_binned["Exp."] = cond
            list_line.append(df_binned.copy(deep=True))

            pivot_count = df_binned.pivot_table(
                index="Trial number", columns="Trial time (s)", values="Count"
            )
            pivot_count["Exp."] = cond
            list_heatmap_count.append(pivot_count)

            baseline_data = df_binned[df_binned["Trial time (s)"] < 0]
            baseline_stats = (
                baseline_data.groupby("Trial number")["Scaled vigor (AU)"].quantile([0.1, 0.9]).unstack()
            )
            if baseline_stats.empty:
                df_norm = df_binned.copy()
                df_norm["Scaled vigor (AU)"] = np.nan
            else:
                baseline_stats.columns = ["min_pre", "max_pre"]
                df_norm = df_binned.merge(baseline_stats, on="Trial number", how="left")
                numerator = df_norm["Scaled vigor (AU)"] - df_norm["min_pre"]
                denominator = df_norm["max_pre"] - df_norm["min_pre"]
                valid = (denominator > 0) & (denominator.notna())
                df_norm["Scaled vigor (AU)"] = np.nan
                df_norm.loc[valid, "Scaled vigor (AU)"] = (numerator[valid] / denominator[valid]).clip(0, 1)
            pivot_sv = df_norm.pivot_table(
                index="Trial number", columns="Trial time (s)", values="Scaled vigor (AU)"
            )
            pivot_sv["Exp."] = cond
            list_heatmap_sv.append(pivot_sv)

        if list_heatmap_count:
            pd.concat(list_heatmap_count).to_pickle(
                path_pooled_data
                / f"Count heatmap {binning_window}s bins all fish_{cond_types_here}_{csus}{selected_suffix}.pkl",
                compression="gzip",
            )
        if list_heatmap_sv:
            pd.concat(list_heatmap_sv).to_pickle(
                path_pooled_data
                / f"SV heatmap {binning_window}s bins all fish_{cond_types_here}_{csus}{selected_suffix}.pkl",
                compression="gzip",
            )
        if list_line:
            data_plot_line = pd.concat(list_line)
            data_plot_line.to_pickle(
                path_pooled_data
                / f"SV lineplot {binning_window}s bins all fish_{cond_types_here}_{csus}{selected_suffix}.pkl",
                compression="gzip",
            )

    if data_plot_line is not None:
        print(data_plot_line.max())

# endregion run_build_pooled_outputs


# %%
# region count_heatmap
def run_count_heatmap():
    """Plot stacked per-block count heatmaps.

    Format mirrors SV heatmap rendering: stacked blocks, stimulus lines, black background,
    and bottom-row x-ticks with a shared time label.
    """
    # These heatmaps visualize the fraction of fish showing behavior per trial/time-bin.
    # The layout is stacked by experimental phase, with one column per condition.
    window_data_plot_heatmap = np.array(window_data_plot) + [
        -count_heatmap_binning_window / 2,
        count_heatmap_binning_window / 2,
    ]

    all_data_csus_paths = [*Path(path_pooled_data).glob("*.pkl")]
    all_data_csus_paths = [
        path
        for path in all_data_csus_paths
        if f"Count heatmap {count_heatmap_binning_window}s bins all fish_" in path.stem
    ]
    all_data_csus_paths = [path for path in all_data_csus_paths if _stem_matches_csus(path.stem, csus)]

    if not all_data_csus_paths:
        print("Skipping Count heatmap (no pooled count files found).")
        return

    data_plots = []
    for path in all_data_csus_paths:
        data_plot = pd.read_pickle(Path(path), compression="gzip")

        if csus == "US" and config.trials_us_blocks_phases:
            data_plot = select_trials(data_plot, config.trials_us_blocks_phases[0])
            data_plot["Block name"] = "Train"

        data_plots.append(data_plot)

    if not data_plots:
        print("Skipping Count heatmap (no pooled count files found).")
        return

    data_plot = pd.concat(data_plots) if len(data_plots) > 1 else data_plots[0]
    binning_window = np.diff(data_plot.columns[1:3])[0]
    ticks = int(np.round(interval_between_xticks_heatmap / binning_window))

    if csus == "CS":
        phases_trial_numbers = config.trials_cs_blocks_phases
        phases_block_names = config.names_cs_blocks_phases
    else:
        if config.trials_us_blocks_phases:
            phases_trial_numbers = config.trials_us_blocks_phases
            phases_block_names = config.names_us_blocks_phases or ["Train"]
        else:
            phases_trial_numbers = None
            phases_block_names = ["Train"]
            if not data_plot.empty:
                phases_trial_numbers = [data_plot.index.unique()]
            if phases_trial_numbers is None:
                print("Skipping Count heatmap (no trials found).")
                return

    cond_types = data_plot["Exp."].unique()
    cond_titles = [cond_title(e) for e in cond_types]

    # Compute actual data range for better color scaling
    numeric_cols = [c for c in data_plot.columns if c != "Exp."]
    data_values = data_plot[numeric_cols].values.flatten()
    data_values = data_values[~np.isnan(data_values)]
    data_min = np.nanmin(data_values) if len(data_values) > 0 else 0
    data_max = np.nanmax(data_values) if len(data_values) > 0 else 1
    data_p95 = np.nanpercentile(data_values, 95) if len(data_values) > 0 else 1
    print(f"  Count heatmap data range: min={data_min:.4f}, max={data_max:.4f}, p95={data_p95:.4f}")

    # Use LogNorm to improve contrast at low counts.
    # Note: LogNorm requires strictly positive values, so we compute a small
    # positive floor and clip zeros up to that floor.
    cmap = "hot"

    positive = data_values[data_values > 0]
    if positive.size == 0:
        vmin_count = 1e-3
        vmax_count = 1.0
    else:
        # Use robust percentiles so a few outliers don't dominate.
        vmin_count = float(np.nanpercentile(positive, 20))
        vmax_count = float(np.nanpercentile(positive, 80))
        vmin_count = max(vmin_count, float(np.nanmin(positive)), 1e-4)
        # Ensure vmax > vmin for LogNorm.
        vmax_count = max(vmax_count, vmin_count * 10)

    count_norm = LogNorm(vmin=vmin_count, vmax=vmax_count)

    fig, axs = make_heatmap_axes(phases_trial_numbers, ncols=len(cond_types), sharey="row", sharex=True)
    xlabels = []
    for col_i, cond_type in enumerate(cond_types):
        data_cond = data_plot[data_plot["Exp."] == cond_type].copy()
        if data_cond.empty:
            for ax in axs[:, col_i]:
                ax.set_visible(False)
            continue

        data_cond.drop(columns="Exp.", inplace=True)

        time_vals = pd.to_numeric(data_cond.columns, errors="coerce")
        mask_time = (time_vals >= window_data_plot_heatmap[0]) & (time_vals <= window_data_plot_heatmap[1])
        data_cond = data_cond.loc[:, mask_time]
        data_cond.columns = [float(s) + binning_window / 2 for s in data_cond.columns]

        # Clip to positive floor for LogNorm (keeps zeros visible as the lowest color).
        data_cond = data_cond.clip(lower=vmin_count)

        render_heatmap_blocks(
            axs[:, [col_i]],
            data_cond,
            phases_trial_numbers,
            phases_block_names,
            ticks,
            stim_color,
            stim_duration,
            window_data_plot,
            show_ylabel=(col_i == 0),
            show_yticklabels=(col_i == 0),
            norm=count_norm,
            cmap=cmap,
        )

        if col_i > 0:
            for ax in axs[:, col_i]:
                ax.yaxis.label.set_visible(False)
                ax.set_yticklabels([])
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Capture tick labels as strings before clearing (Text objects become stale)
        if col_i == 0:
            fig.canvas.draw()  # Need to draw first to populate the tick labels
            xlabels = [t.get_text() for t in axs[-1][col_i].get_xticklabels()]
        axs[-1][col_i].xaxis.label.set_visible(False)
        axs[-1][col_i].set_xticklabels([])
        axs[-1][col_i].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    
    fig.canvas.draw()

    for b_i in range(len(phases_trial_numbers)):
        plot_cfg = get_plot_config()
        analysis_utils.add_component(
            axs[b_i][0],
            analysis_utils.AddTextSpec(
                component="supylabel",
                text=phases_block_names[b_i],
                anchor_h="left",
                anchor_v="center",
                pad_pt=(0,0),
                text_kwargs={"rotation": 90, "fontweight": "bold", "color": "k"},
            ),
        )
    
    # Restore x-axis labels on bottom row
    for col_i, cond_type in enumerate(cond_types):
        axs[-1][col_i].set_xticklabels(xlabels)
        axs[-1][col_i].tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    fig.canvas.draw()

    plot_cfg = get_plot_config()
    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supxlabel",
            text=f"Time relative to {csus} onset (s)",
            anchor_h="center",
            anchor_v="bottom",
            pad_pt=plot_cfg.pad_supxlabel,
            text_kwargs={"fontweight": plot_cfg.axes_labelweight, "fontsize": plot_cfg.axes_labelsize, "color": "k"},
        ),
    )

    fig.canvas.draw()

    for col_i, cond_type in enumerate(cond_types):

        analysis_utils.add_component(
            axs[0, col_i],
            analysis_utils.AddTextSpec(
                component="axis_title",
                text=cond_titles[col_i],
                anchor_h="right",
                anchor_v="top",
                pad_pt=(0, 0),
                text_kwargs={"fontsize": plot_cfg.figure_titlesize, "backgroundcolor": "none", "color": "k"},
            ),
        )

    fig.canvas.draw()

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="text",
            text="A",
            anchor_h="left",
            anchor_v="top",
            pad_pt=(-5,10),
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "fontweight": "bold", "backgroundcolor": "none", "color": "k"},
        ),
    )

    cond_label = _stringify_for_filename(cond_types.tolist() if hasattr(cond_types, "tolist") else list(cond_types))
    save_fig(fig, f"Percentage all fish_{cond_label}_{csus}", frmt)

# endregion run_count_heatmap


# %%
# region sv_heatmap_rendering
def run_sv_heatmap_rendering():
    """Render normalized SV heatmaps in stacked block layout.

    Format mirrors count heatmaps: stacked blocks, stimulus lines, black background,
    and bottom-row x-ticks with a shared time label.
    """
    # This renders the precomputed pooled SV heatmap files. Values are normalized within
    # each trial using pre-stimulus quantiles (see build step), then shown as a stacked
    # heatmap per phase with stimulus markers.
    window_data_plot_heatmap = np.array(window_data_plot) + [
        -binning_window_heatmap / 2,
        binning_window_heatmap / 2,
    ]

    all_data_csus_paths = [*Path(path_pooled_data).glob("*.pkl")]
    all_data_csus_paths = [
        path for path in all_data_csus_paths if f"SV heatmap {binning_window_heatmap}s bins all fish_" in path.stem
    ]
    all_data_csus_paths = [path for path in all_data_csus_paths if _stem_matches_csus(path.stem, csus)]

    if not all_data_csus_paths:
        print("Skipping SV heatmap rendering (no pooled heatmap files found).")
        return

    data_plots = []
    for path in all_data_csus_paths:
        data_plot = pd.read_pickle(Path(path), compression="gzip")

        if csus == "US" and config.trials_us_blocks_phases:
            data_plot = data_plot[data_plot.index.isin(config.trials_us_blocks_phases[0])]
            data_plot["Block name"] = "Train"

        data_plots.append(data_plot)

    if not data_plots:
        print("Skipping SV heatmap rendering (no pooled heatmap files found).")
        return

    data_plot = pd.concat(data_plots) if len(data_plots) > 1 else data_plots[0]
    binning_window = np.diff(data_plot.columns[1:3])[0]
    ticks = int(np.round(interval_between_xticks_heatmap / binning_window))

    phases_trial_numbers = None
    phases_block_names = None

    if csus == "CS":
        phases_trial_numbers = config.trials_cs_blocks_phases
        phases_block_names = config.names_cs_blocks_phases
    else:
        # US-aligned plots: prefer the configured phase blocks; otherwise fall back to all trials.
        if config.trials_us_blocks_phases:
            phases_trial_numbers = config.trials_us_blocks_phases
            phases_block_names = config.names_us_blocks_phases or ["Train"]
        else:
            if not data_plot.empty:
                phases_trial_numbers = [data_plot.index.unique()]
                phases_block_names = ["Train"]

        if phases_trial_numbers is None or phases_block_names is None:
            print("Skipping SV heatmap rendering (no trials found).")
            return
    cond_types = data_plot["Exp."].unique()
    cond_titles = [cond_title(e) for e in cond_types]

    fig, axs = make_heatmap_axes(phases_trial_numbers, ncols=len(cond_types), sharey="row", sharex=True)
    xlabels = []
    for col_i, cond_type in enumerate(cond_types):
        data_cond = data_plot[data_plot["Exp."] == cond_type].copy()
        if data_cond.empty:
            for ax in axs[:, col_i]:
                ax.set_visible(False)
            continue

        data_cond.drop(columns="Exp.", inplace=True)

        time = data_cond.columns.to_numpy().astype("float")
        mask_time = (time >= window_data_plot_heatmap[0]) & (time <= window_data_plot_heatmap[1])
        data_cond = data_cond.loc[:, mask_time]
        data_cond.columns = [float(s) + binning_window / 2 for s in data_cond.columns]

        render_heatmap_blocks(
            axs[:, [col_i]],
            data_cond,
            phases_trial_numbers,
            phases_block_names,
            ticks,
            stim_color,
            stim_duration,
            window_data_plot,
            show_ylabel=(col_i == 0),
            show_yticklabels=(col_i == 0),
        )

        if col_i > 0:
            for ax in axs[:, col_i]:
                ax.yaxis.label.set_visible(False)
                ax.set_yticklabels([])
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Capture tick labels as strings before clearing (Text objects become stale)
        if col_i == 0:
            fig.canvas.draw()  # Need to draw first to populate the tick labels
            xlabels = [t.get_text() for t in axs[-1][col_i].get_xticklabels()]
        axs[-1][col_i].xaxis.label.set_visible(False)
        axs[-1][col_i].set_xticklabels([])
        axs[-1][col_i].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    
    fig.canvas.draw()

    for b_i in range(len(phases_trial_numbers)):
        plot_cfg = get_plot_config()
        analysis_utils.add_component(
            axs[b_i][0],
            analysis_utils.AddTextSpec(
                component="supylabel",
                text=phases_block_names[b_i],
                anchor_h="left",
                anchor_v="center",
                pad_pt=(0,0),
                text_kwargs={"rotation": 90, "fontweight": "bold", "color": "k"},
            ),
        )
    
    # Restore x-axis labels on bottom row
    for col_i, cond_type in enumerate(cond_types):
        axs[-1][col_i].set_xticklabels(xlabels)
        axs[-1][col_i].tick_params(axis="x", which="both", bottom=True, labelbottom=True)

    fig.canvas.draw()

    plot_cfg = get_plot_config()
    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supxlabel",
            text=f"Time relative to {csus} onset (s)",
            anchor_h="center",
            anchor_v="bottom",
            pad_pt=plot_cfg.pad_supxlabel,
            text_kwargs={"fontweight": plot_cfg.axes_labelweight, "fontsize": plot_cfg.axes_labelsize, "color": "k"},
        ),
    )

    fig.canvas.draw()

    for col_i, cond_type in enumerate(cond_types):

        analysis_utils.add_component(
            axs[0, col_i],
            analysis_utils.AddTextSpec(
                component="axis_title",
                text=cond_titles[col_i],
                anchor_h="right",
                anchor_v="top",
                pad_pt=(0,0),
                text_kwargs={"fontsize": plot_cfg.figure_titlesize, "backgroundcolor": "none", "color": "k"},
            ),
        )

        fig.canvas.draw()

    fig.canvas.draw()

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="text",
            text="A",
            anchor_h="left",
            anchor_v="top",
            pad_pt=(-5,10),
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "fontweight": "bold", "backgroundcolor": "none", "color": "k"},
        ),
    )

    cond_label = _stringify_for_filename(cond_types.tolist() if hasattr(cond_types, "tolist") else list(cond_types))
    save_fig(fig, f"SV all fish_{cond_label}_{csus}", frmt)

# endregion run_sv_heatmap_rendering


# %%
# region sv_lineplots_individual_catch_trials
def run_sv_lineplots_individual_catch_trials():
    """Plot one row per catch trial with condition lines and CI bands.

    Format: per-trial rows, median traces with CI bands, baseline-subtracted and
    clipped, stimulus lines on each axis, and a shared legend outside the panel.
    """
    all_data_csus_paths = [*Path(path_all_fish).glob("*.pkl")]
    all_data_csus_paths = [path for path in all_data_csus_paths if _stem_matches_csus(path.stem, csus)]

    if not all_data_csus_paths:
        print("Skipping individual catch trials (no per-condition pooled files found).")
        return

    data_original = []
    for p in all_data_csus_paths:
        df = pd.read_pickle(str(p), compression="gzip").reset_index()
        df = filter_discarded_fish_ids(df, source=p.stem)
        df = densify_sparse(df)
        df = ensure_time_seconds(df)
        cond = df["Exp."].unique()[0] if "Exp." in df.columns else p.stem.split("_")[-2]
        data_original.append({"cond": cond, "data": df})

    cond_types_here = [d["cond"] for d in data_original]
    trials_to_use, trial_names_here = resolve_catch_trials()
    time_bins_short = compute_time_bins(x_lim, default_binning_window) if DO_TIME_BIN else None

    fig, axs = make_lineplot_axes(len(trials_to_use), (2.5 / 2.54, 17 / 2.54))

    # Collect legend handles once (first axis) so we can place a shared legend.
    handles = []

    for item in data_original:
        cond = item["cond"]
        data_cond = item["data"].copy(deep=True)

        data_cond = data_cond[data_cond["Trial number"].isin(trials_to_use)]
        data_cond["Trial number"] = pd.Categorical(
            data_cond["Trial number"], categories=trials_to_use, ordered=True
        )
        data_cond = data_cond.loc[data_cond["Trial time (s)"].between(x_lim[0] - 0.5, x_lim[1] + 0.5), :]

        if "Bout" in data_cond.columns:
            data_cond.loc[~data_cond["Bout"], "Scaled vigor (AU)"] = np.nan

        data_cond = (
            data_cond.groupby(by=["Trial time (s)", "Trial number"], as_index=True, observed=True)[
                "Scaled vigor (AU)"
            ]
            .agg("median")
            .reset_index()
            .copy(deep=True)
        )

        if DO_TIME_BIN and time_bins_short is not None:
            bins = pd.cut(data_cond["Trial time (s)"], time_bins_short, include_lowest=True)
            data_cond = (
                data_cond.groupby(by=[bins, "Trial number"], as_index=True, observed=True)["Scaled vigor (AU)"]
                .agg("median")
                .reset_index()
            )
            data_cond["Trial time (s)"] = [interval.right for interval in data_cond["Trial time (s)"]]

        for ax_i, tnum in enumerate(trials_to_use):
            ax = axs[ax_i]
            data_cond_block = data_cond[data_cond["Trial number"] == tnum].copy(deep=True)

            data_cond_block = apply_baseline_and_clip(data_cond_block, y_clip)

            plot_cond_line(
                ax=ax,
                data=data_cond_block,
                x_col="Trial time (s)",
                y_col="Scaled vigor (AU)",
                cond=cond,
                color_rgb_list=cond_color(cond),
                n_boot=n_boot,
                ci=95,
            )

            style_axes(
                ax, stim_color, stim_duration, x_lim, y_lim, show_xlabel=(ax_i == len(trials_to_use) - 1), outward=0
            )
            # Use add_component for axis title instead of ax.set_title
            plot_cfg = get_plot_config()
            analysis_utils.add_component(
                ax,
                analysis_utils.AddTextSpec(
                    component="axis_title",
                    text=trial_names_here[ax_i],
                    anchor_h="left",
                    anchor_v="top",
                    pad_pt=(0.0, 0.0),
                    text_kwargs={"fontsize": plot_cfg.axes_titlesize},
                    use_tight_bbox=False,
                ),
            )

            if ax_i == 0 and not handles:
                handles, _ = ax.get_legend_handles_labels()
            remove_axis_legend(ax)

    add_condition_legend(fig, handles, cond_types_here)

    cond_label = _stringify_for_filename(cond_types_here)
    stem_bits = "_".join(all_data_csus_paths[0].stem.split("_")[:-2])
    path_part = f"SV lineplot all individual catch trials {trials_to_use}_{stem_bits}_{cond_label}"
    
    fig.canvas.draw()
    plot_cfg = get_plot_config()

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supylabel",
            text="Scaled vigor (AU)",
            anchor_h="left",
            anchor_v="center",
            pad_pt=plot_cfg.pad_supylabel,
            text_kwargs={"rotation": 90, "fontweight": plot_cfg.axes_labelweight, "fontsize": plot_cfg.axes_labelsize},
        ),
    )

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supxlabel",
            text=f"Time relative to\n{csus} onset (s)",
            anchor_h="center",
            anchor_v="bottom",
            pad_pt=plot_cfg.pad_supxlabel,
            text_kwargs={"fontweight": plot_cfg.axes_labelweight, "fontsize": plot_cfg.axes_labelsize},
        ),
    )

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="fig_title",
            text="Individual catch trials",
            anchor_h="left",
            anchor_v="top",
            pad_pt=plot_cfg.pad_fig_title,
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "backgroundcolor": "none"},
        ),
    )

    fig.canvas.draw()

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="text",
            text="A",
            anchor_h="left",
            anchor_v="top",
            pad_pt=(-5, 10),
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "fontweight": "bold", "backgroundcolor": "none", "color": "k"},
        ),
    )
    
    save_fig(fig, path_part, frmt)

# endregion run_sv_lineplots_individual_catch_trials


# %%
# region sv_lineplot_all_catch_trials
def run_sv_lineplot_all_catch_trials():
    """Plot a single panel with all catch trials per condition and CI bands.

    Format: one axis, median traces with CI bands, baseline-subtracted and clipped,
    stimulus lines, and a shared legend outside the panel.
    """
    pooled_paths = [*Path(path_pooled_data).glob("*.pkl")]
    pooled_paths = [
        p for p in pooled_paths if f"SV lineplot {default_binning_window}s bins all fish_" in p.stem
    ]
    pooled_paths = [p for p in pooled_paths if _stem_matches_csus(p.stem, csus)]

    if not pooled_paths:
        print("Skipping catch trials line plot (no pooled lineplot files found).")
        return

    data_original = pd.read_pickle(str(pooled_paths[0]), compression="gzip").reset_index()
    data_original = densify_sparse(data_original)

    cond_types_here = data_original["Exp."].unique()
    trials_to_use, _ = resolve_catch_trials()

    fig, ax = make_lineplot_axes(1, (4 / 2.54, 4 / 2.54), sharex=True, sharey=True)

    for cond in cond_types_here:
        data_cond = data_original[data_original["Exp."] == cond].copy(deep=True)
        data_cond = data_cond[data_cond["Trial number"].isin(trials_to_use)]
        data_cond["Trial number"] = pd.Categorical(
            data_cond["Trial number"], categories=trials_to_use, ordered=True
        )
        data_cond = data_cond.loc[data_cond["Trial time (s)"].between(x_lim[0], x_lim[1]), :]
        data_cond = data_cond.reset_index().copy(deep=True)

        data_cond = apply_baseline_and_clip(data_cond, y_clip)

        plot_cond_line(
            ax=ax,
            data=data_cond,
            x_col="Trial time (s)",
            y_col="Scaled vigor (AU)",
            cond=cond,
            color_rgb_list=cond_color(cond),
            n_boot=n_boot,
            ci=95,
        )

        # Use add_component for axis title instead of ax.set_title
        plot_cfg = get_plot_config()
        # analysis_utils.add_component(
        #     ax,
        #     analysis_utils.AddTextSpec(
        #         component="axis_title",
        #         text="All catch trials",
        #         anchor_h="left",
        #         anchor_v="center",
        #         pad_pt=(0.0, 0.0),
        #         text_kwargs={"fontsize": plot_cfg.axes_titlesize},
        #     ),
        # )
        remove_axis_legend(ax)

    style_axes(ax, stim_color, stim_duration, x_lim, y_lim, show_xlabel=True, outward=0)

    fig.canvas.draw()
    plot_cfg = get_plot_config()

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supylabel",
            text="Scaled vigor (AU)",
            anchor_h="left",
            anchor_v="center",
            pad_pt=plot_cfg.pad_supylabel,
            text_kwargs={"rotation": 90, "fontweight": plot_cfg.axes_labelweight, "fontsize": plot_cfg.axes_labelsize},
        ),
    )

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supxlabel",
            text=f"Time relative to\n{csus} onset (s)",
            anchor_h="center",
            anchor_v="bottom",
            pad_pt=plot_cfg.pad_supxlabel,
            text_kwargs={"fontweight": plot_cfg.axes_labelweight, "fontsize": plot_cfg.axes_labelsize},
        ),
    )

    # return

    handles, _ = ax.get_legend_handles_labels()
    add_condition_legend(fig, handles, cond_types_here)

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="fig_title",
            text="All catch trials",
            anchor_h="left",
            anchor_v="top",
            pad_pt=plot_cfg.pad_fig_title,
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "backgroundcolor": "none"},
        ),
    )

    fig.canvas.draw()

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="text",
            text="A",
            anchor_h="left",
            anchor_v="top",
            pad_pt=(-5, 10),
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "fontweight": "bold", "backgroundcolor": "none", "color": "k"},
        ),
    )

    cond_label = _stringify_for_filename(cond_types_here)
    stem_bits = "_".join(pooled_paths[0].stem.split("_")[:-2])
    save_fig(fig, f"SV lineplot all catch trials_{stem_bits}_{cond_label}", frmt)

# endregion run_sv_lineplot_all_catch_trials


# %%
# region sv_lineplots_per_block
def run_sv_lineplots_per_block():
    """Plot one row per block with condition-wise median SV traces.

    Format: stacked block rows, median traces with CI bands, baseline-subtracted and
    clipped, stimulus lines, and a shared legend outside the panel.
    """
    pooled_data_files = [*Path(path_pooled_data).glob("*.pkl")]
    pooled_paths = [
        p for p in pooled_data_files if f"SV lineplot {default_binning_window}s bins all fish_" in p.stem
    ]
    pooled_paths = [p for p in pooled_paths if _stem_matches_csus(p.stem, csus)]

    if not pooled_paths:
        print("Skipping block line plots (no pooled lineplot files found).")
        return

    pooled_data_path = pooled_paths[0]

    if csus == "CS":
        block_names = config.names_cs_blocks_10
    else:
        block_names = config.names_us_blocks_10

    if not block_names:
        print("Skipping block line plots (no block names for current csus).")
        return

    data_original = pd.read_pickle(str(pooled_data_path), compression="gzip").reset_index()
    data_original = densify_sparse(data_original)

    cond_types_here = data_original["Exp."].unique()

    fig, axs = make_lineplot_axes(len(block_names), (4 / 2.54, 16.5 / 2.54))

    # Collect legend handles once (first axis) so we can place a shared legend.
    handles = []

    for cond in cond_types_here:
        data_cond = data_original[data_original["Exp."] == cond].copy(deep=True)
        data_cond = data_cond.loc[data_cond["Trial time (s)"].between(x_lim[0], x_lim[1]), :]

        data_cond = (
            data_cond.groupby(
                by=["Trial time (s)", "Trial number", "Block name"],
                as_index=True,
                observed=True,
                dropna=True,
            )["Scaled vigor (AU)"]
            .agg("median")
            .reset_index()
            .copy(deep=True)
        )

        for ax_i, block_name in enumerate(block_names):
            ax = axs[ax_i]
            data_cond_block = data_cond[data_cond["Block name"] == block_name].copy(deep=True)

            data_cond_block = apply_baseline_and_clip(data_cond_block, y_clip)

            plot_cond_line(
                ax=ax,
                data=data_cond_block,
                x_col="Trial time (s)",
                y_col="Scaled vigor (AU)",
                cond=cond,
                color_rgb_list=cond_color(cond),
                n_boot=n_boot,
                ci=95,
            )

            style_axes(
                ax, stim_color, stim_duration, x_lim, y_lim, show_xlabel=(ax_i == len(block_names) - 1), outward=5
            )

            # return
        



            if ax_i == 0 and not handles:
                handles, _ = ax.get_legend_handles_labels()
            remove_axis_legend(ax)



    fig.subplots_adjust(hspace=1)

    # return

    for ax_i, block_name in enumerate(block_names):

        # Use add_component for axis title instead of ax.set_title
        plot_cfg = get_plot_config()
        analysis_utils.add_component(
            axs[ax_i],
            analysis_utils.AddTextSpec(
                component="axis_title",
                text=block_name,
                anchor_h="left",
                anchor_v="top",
                pad_pt=(0,0),
                text_kwargs={"fontsize": plot_cfg.axes_titlesize},
                use_tight_bbox=False,
            ),
        )


        # axs[ax_i].set_title(
        #     block_name,
        #     loc="left",
        #     va="bottom",
        #     ha="left",
        #     x=0,   
        #     y=0.7,           
        #     backgroundcolor="none",
        #     fontsize=10,
        # )



    # return

    fig.canvas.draw()
    plot_cfg = get_plot_config()


    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supxlabel",
            text=f"Time relative to\n{csus} onset (s)",
            anchor_h="center",
            anchor_v="bottom",
            pad_pt=plot_cfg.pad_supxlabel,
            text_kwargs={"fontweight": plot_cfg.axes_labelweight, "fontsize": plot_cfg.axes_labelsize},
        ),
    )
    
    fig.canvas.draw()
    plot_cfg = get_plot_config()


    fig.canvas.draw()
    plot_cfg = get_plot_config()

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="supylabel",
            text="Scaled vigor (AU)",
            anchor_h="left",
            anchor_v="center",
            pad_pt=plot_cfg.pad_supylabel,
            text_kwargs={"rotation": 90, "fontweight": plot_cfg.axes_labelweight, "fontsize": plot_cfg.axes_labelsize},
        ),
    )


    # add_condition_legend(fig, handles, cond_types_here)

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="fig_title",
            text="Per 10-trial blocks",
            anchor_h="left",
            anchor_v="top",
            pad_pt=plot_cfg.pad_fig_title,
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "backgroundcolor": "none"},
        ),
    )

    fig.canvas.draw()

    analysis_utils.add_component(
        fig,
        analysis_utils.AddTextSpec(
            component="text",
            text="A",
            anchor_h="left",
            anchor_v="top",
            pad_pt=(-5, 10),
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "fontweight": "bold", "backgroundcolor": "none", "color": "k"},
        ),
    )

    cond_label = _stringify_for_filename(cond_types_here)
    save_fig(fig, f"SV per 10-trial blocks_{pooled_data_path.stem}_{cond_label}", frmt)

# endregion run_sv_lineplots_per_block
# endregion Pipeline Functions


# %%
# region Main
# region main
def main():
    if RUN_BUILD_POOLED_OUTPUTS:
        try:
            run_build_pooled_outputs()
        except Exception as exc:
            print(f"Error in Build pooled outputs: {exc}")

    if RUN_COUNT_HEATMAP:
        try:
            run_count_heatmap()
        except Exception as exc:
            print(f"Error in Count heatmap: {exc}")

    if RUN_SV_HEATMAP_RENDERING:
        try:
            run_sv_heatmap_rendering()
        except Exception as exc:
            print(f"Error in SV heatmap rendering: {exc}")

    if RUN_SV_LINEPLOTS_INDIVIDUAL:
        try:
            run_sv_lineplots_individual_catch_trials()
        except Exception as exc:
            print(f"Error in SV lineplots (individual catch trials): {exc}")

    if RUN_SV_LINEPLOTS_CATCH_TRIALS:
        try:
            run_sv_lineplot_all_catch_trials()
        except Exception as exc:
            print(f"Error in SV lineplot (all catch trials): {exc}")

    if RUN_SV_LINEPLOTS_BLOCKS:
        try:
            run_sv_lineplots_per_block()
        except Exception as exc:
            print(f"Error in SV lineplots (per block): {exc}")

# endregion main


if __name__ == "__main__":
    main()
# endregion Main
