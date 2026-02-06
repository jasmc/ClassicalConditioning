"""
Scaled Vigor Heatmap Plotting — Log-Median Variant
====================================================

Pipeline context — Step 4 of 6 (log-median variant)
-----------------------------------------------------
This script is a standalone variant of ``4_ScaledVigorPlotting.py`` designed to
work with the output of ``3_FishGrouping_LogMedian.py``.  It generates
heatmap-only visualizations (count heatmaps and SV heatmaps) of the
log-median-transformed vigor signal.

Key differences from the standard Step 4 pipeline
---------------------------------------------------
- **No within-trial 0–1 scaling** — the SV heatmap displays raw
  baseline-subtracted log-median vigor values, preserving the natural scale
  produced by Step 3.
- **Fixed color scale 0–1.5** — the SV heatmap uses ``vmin=0, vmax=1.5``
  instead of a normalized 0–1 range.
- **Heatmaps only** — lineplot rendering is omitted; use the standard
  ``4_ScaledVigorPlotting.py`` for line-plot analyses.

Steps
-----
1. **Build pooled outputs** (``RUN_BUILD_POOLED_OUTPUTS``):
   - Load pooled CS (or US) pickle files for each condition.
   - Optionally discard fish using the experiment's discard list.
   - Time-bin the data at one or more bin widths.
   - Aggregate scaled vigor by trial x time-bin (median across fish) and count
     the number of contributing observations per bin.
   - Save two pooled outputs per bin width as compressed pickles:
     a. *Count heatmap* — fraction of fish with behavior per trial / time-bin.
     b. *SV heatmap* — unscaled median vigor per trial / time-bin.

2. **Count heatmap** (``RUN_COUNT_HEATMAP``):
   - Load the pre-built count-heatmap pickle.
   - Render a stacked-block heatmap with LogNorm color scaling.

3. **SV heatmap** (``RUN_SV_HEATMAP_RENDERING``):
   - Same stacked-block layout, showing baseline-subtracted log-median vigor
     with a fixed 0–1.5 color scale and black background.

Inputs
------
- Pooled per-condition pickles from Step 3 (log-median variant):
  ``{condition}_{CS|US}_new_logmedian.pkl``.

Outputs
-------
- Intermediate compressed pickles (count heatmap, SV heatmap) stored in
  ``path_pooled_data``, one set per bin width.
- Publication-quality SVG figures saved to a ``Scaled vigor`` subfolder.
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
RUN_SV_HEATMAP_RENDERING = True

# ------------------------------------------------------------------------------
# Shared Parameters
# ------------------------------------------------------------------------------
EXPERIMENT = ExperimentType.ALL_DELAY.value

# Apply per-experiment discarded fish list if present under "Processed data".
APPLY_FISH_DISCARD = True

csus = "CS"  # Stimulus alignment: "CS" or "US".
INPUT_PKL_SUFFIX = "_new_logmedian"  # suffix of grouped pkl files from Step 3
x_lim = (-20, 20)
window_data_plot = [-20, 20]

# ------------------------------------------------------------------------------
# Build Pooled Outputs Parameters
# ------------------------------------------------------------------------------
binning_windows = [0.5, 1.0]

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
SV_HEATMAP_VMIN = 0
SV_HEATMAP_VMAX = 1.5
frmt = "svg"
# endregion Parameters


# %%
# region Plot Formatting
from plotting_style import (HEATMAP_FIGSIZE, HEATMAP_GRIDSPACE,
                            HEATMAP_STYLE_KW, HEATMAP_TICK_KW,
                            SAVEFIG_KW)

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
discard_file = path_orig_pkl / "Excluded" / "excluded_fish_ids.txt"

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
    bin_start_target = x_lim[0] - 1
    bin_end_target = x_lim[1] + 2
    start_neg = np.floor(bin_start_target / binning_window) * binning_window
    neg_bins = np.arange(start_neg, 0 + binning_window, binning_window)
    end_pos = np.ceil(bin_end_target / binning_window) * binning_window
    pos_bins = np.arange(0, end_pos + binning_window / 2, binning_window)
    return np.unique(np.concatenate((neg_bins, pos_bins))).tolist()


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


def _stringify_for_filename(value) -> str:
    """Convert common objects (lists/arrays) into filename-friendly strings."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return "-".join(str(v) for v in value)
    return str(value)


def _sanitize_filename(name: str) -> str:
    """Sanitize a filename component for Windows filesystems."""
    invalid = '<>:"/\\|?*'
    out = "".join("_" if ch in invalid else ch for ch in str(name))
    out = out.strip().rstrip(".")
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
    """Load per-fish data and aggregate into pooled heatmap outputs.

    Unlike the standard pipeline, this version does NOT apply within-trial
    0–1 normalization to the SV heatmap.  The raw baseline-subtracted
    log-median vigor values are preserved.
    """
    selected_suffix = SELECTED_FISH_SUFFIX if APPLY_FISH_DISCARD else ""
    all_data_csus_paths = sorted(
        [path for path in Path(path_all_fish).glob(f"*_{csus}{INPUT_PKL_SUFFIX}.pkl")]
    )
    if not all_data_csus_paths:
        print(f"No data found in {path_all_fish} for {csus}.")

    cond_types_here = sorted({path.stem.split("_")[0] for path in all_data_csus_paths})

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

        if "Block name" in df.columns and df["Block name"].isna().all():
            df["Trial type"] = csus
            df = analysis_utils.identify_blocks_trials(df, config.blocks_dict)

        if "Bout" in df.columns:
            df.loc[~df["Bout"], ["Vigor (deg/ms)", "Scaled vigor (AU)"]] = np.nan

        df = subset_columns(df, cols_to_use)
        if "Exp." not in df.columns:
            df["Exp."] = path.stem.split("_")[0]

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

    for binning_window in binning_windows:
        print(f"Processing {binning_window}s bins...")
        bin_edges = compute_time_bins(x_lim, binning_window)
        list_heatmap_count, list_heatmap_sv = [], []

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

            # --- Count heatmap pivot ---
            pivot_count = df_binned.pivot_table(
                index="Trial number", columns="Trial time (s)", values="Count"
            )
            pivot_count["Exp."] = cond
            list_heatmap_count.append(pivot_count)

            # --- SV heatmap pivot (NO scaling — keep raw values) ---
            pivot_sv = df_binned.pivot_table(
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

# endregion run_build_pooled_outputs


# %%
# region count_heatmap
def run_count_heatmap():
    """Plot stacked per-block count heatmaps.

    Format mirrors SV heatmap rendering: stacked blocks, stimulus lines, black background,
    and bottom-row x-ticks with a shared time label.
    """
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
    cmap = "hot"

    positive = data_values[data_values > 0]
    if positive.size == 0:
        vmin_count = 1e-3
        vmax_count = 1.0
    else:
        vmin_count = float(np.nanpercentile(positive, 20))
        vmax_count = float(np.nanpercentile(positive, 80))
        vmin_count = max(vmin_count, float(np.nanmin(positive)), 1e-4)
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

        # Clip to positive floor for LogNorm
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

        if col_i == 0:
            fig.canvas.draw()
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
                pad_pt=(0, 0),
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
            pad_pt=(-5, 10),
            text_kwargs={"fontsize": plot_cfg.figure_titlesize, "fontweight": "bold", "backgroundcolor": "none", "color": "k"},
        ),
    )

    cond_label = _stringify_for_filename(cond_types.tolist() if hasattr(cond_types, "tolist") else list(cond_types))
    save_fig(fig, f"Percentage all fish_{cond_label}_{csus}", frmt)

# endregion run_count_heatmap


# %%
# region sv_heatmap_rendering
def run_sv_heatmap_rendering():
    """Render unscaled SV heatmaps in stacked block layout.

    Unlike the standard pipeline, vigor values are NOT normalized to 0–1.
    The color scale is fixed to 0–1.5 to display the raw baseline-subtracted
    log-median vigor from Step 3.
    """
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
            vmin=SV_HEATMAP_VMIN,
            vmax=SV_HEATMAP_VMAX,
        )

        if col_i > 0:
            for ax in axs[:, col_i]:
                ax.yaxis.label.set_visible(False)
                ax.set_yticklabels([])
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        if col_i == 0:
            fig.canvas.draw()
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
                pad_pt=(0, 0),
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

    cond_label = _stringify_for_filename(cond_types.tolist() if hasattr(cond_types, "tolist") else list(cond_types))
    save_fig(fig, f"SV all fish_{cond_label}_{csus}", frmt)

# endregion run_sv_heatmap_rendering
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

# endregion main


if __name__ == "__main__":
    main()
# endregion Main
