"""Render pooled NV plots (block summaries, boxplots, phase medians, trial-by-trial LME)."""
# %%
# region Imports & Configuration
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from pandas.api.types import CategoricalDtype
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

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

pd.set_option("mode.copy_on_write", True)
# pd.options.mode.chained_assignment = None

# Apply shared plotting style with script-specific overrides.
PLOT_STYLE_OVERRIDES = {"figure.constrained_layout.use": False}
plotting_style.set_plot_style(rc_overrides=PLOT_STYLE_OVERRIDES)
# endregion

# region Parameters
# ==============================================================================
# PIPELINE CONTROL FLAGS
# ==============================================================================
RUN_PROCESS = False
RUN_BLOCK_SUMMARY_LINES = True
RUN_BLOCK_SUMMARY_BOXPLOT = True
RUN_PHASE_SUMMARY = True
RUN_TRIAL_BY_TRIAL = True

# ==============================================================================
# GLOBAL SETTINGS
# ==============================================================================
EXPERIMENT = ExperimentType.ALL_10S_TRACE.value
# ExperimentType.ALL_3S_TRACE.value
# ExperimentType.ALL_DELAY.value

csus = "CS"  # Stimulus alignment: "CS" or "US".
STATS = True  # Enable statistical tests.
RUN_LME = True  # Enable LME analysis.
EXPORT_TEXT = False  # Export trial data to text.

minimum_trials_per_fish_per_block = 6
age_filter = ["all"]
setup_color_filter = ["all"]

# ==============================================================================
# DATA AGGREGATION PARAMETERS
# ==============================================================================
TRIAL_WINDOW_S = (-21, 21)

# ==============================================================================
# SHARED PLOT PARAMETERS
# ==============================================================================
frmt = "svg"
Hide_non_significant = True
y_lim = (0.8, 1.2)

# ==============================================================================
# BLOCK SUMMARY (LINES) PARAMETERS
# ==============================================================================
# (uses shared plot parameters)

# ==============================================================================
# BLOCK SUMMARY (BOXPLOT) PARAMETERS
# ==============================================================================
DISCARD_FISH_IDS_BOXPLOT = []  # e.g., ["20221115_05"]

# ==============================================================================
# PHASE SUMMARY PARAMETERS
# ==============================================================================
HIGHLIGHT_FISH_ID = None  # e.g., "20221115_05"

# ==============================================================================
# TRIAL-BY-TRIAL PARAMETERS
# ==============================================================================
n_boot = 1000
y_lim_plot = (0.7, 1.4)
lme_frmt = "png"
APPLY_FISH_DISCARD_LME = False  # Apply excluded-fish filter to trial-by-trial plot + LME.
# endregion

# region Plot Formatting Defaults
SAVEFIG_KW = {"dpi": 600, "transparent": False, "bbox_inches": "tight"}
FIGURE_KW = {"facecolor": "white", "clip_on": False, "constrained_layout": True}

BASELINE_LINE_KW = {"color": "k", "alpha": 0.5, "lw": 0.5}
BLOCK_DIVIDER_KW = {"color": "gray", "alpha": 0.95, "linestyle": "-", "linewidth": 0.5}

BLOCK_FIG_HEIGHT = 4 / 2.54
BLOCK_WIDTH_PER_COND = 2
BOXPLOT_FIGSIZE = (6 / 2.54, 6 / 2.54)
TRIAL_BY_TRIAL_FIGSIZE = (14 / 2.54, 9 / 2.54)

LEGEND_OUTSIDE_KW = {"frameon": False, "bbox_to_anchor": (1.02, 1.0), "loc": "upper left", "borderaxespad": 0}

TRIAL_LINE_KW = {
    "alpha": 0.9,
    "linewidth": 0.5,
    "estimator": "median",
    "errorbar": ("ci", 95),
    "err_style": "band",
}
# endregion

# region Context Setup
config = None
cond_types = []
cond_titles = []
color_palette = []
cr_window = None
blocks_dict = {}
path_pooled_vigor_fig = None
path_orig_pkl = None
path_all_fish = None
path_pooled_data = None
fish_ids_to_discard = []
skip_block_stats = False


def initialize_context():
    global config, cond_types, cond_titles, color_palette, cr_window, blocks_dict
    global path_pooled_vigor_fig, path_orig_pkl, path_all_fish, path_pooled_data
    global fish_ids_to_discard, skip_block_stats

    config = get_experiment_config(EXPERIMENT)
    cond_types = list(config.cond_types)
    cond_titles = [config.cond_dict[c]["name"] for c in cond_types]
    color_palette = config.color_palette

    cr_window = config.cr_window
    if isinstance(cr_window, (int, float, np.integer, np.floating)):
        cr_window = [0, cr_window]
    blocks_dict = config.blocks_dict

    (
        _path_lost_frames,
        _path_summary_exp,
        _path_summary_beh,
        _path_processed_data,
        _path_cropped_exp_with_bout_detection,
        _path_tail_angle_fig_cs,
        _path_tail_angle_fig_us,
        _path_raw_vigor_fig_cs,
        _path_raw_vigor_fig_us,
        _path_scaled_vigor_fig_cs,
        _path_scaled_vigor_fig_us,
        _path_normalized_fig_cs,
        _path_normalized_fig_us,
        path_pooled_vigor_fig,
        _path_analysis_protocols,
        path_orig_pkl,
        path_all_fish,
        path_pooled_data,
    ) = file_utils.create_folders(config.path_save)

    excluded_dir = path_orig_pkl / "Excluded new"
    fish_ids_to_discard = file_utils.load_excluded_fish_ids(excluded_dir)

    skip_block_stats = EXPERIMENT == ExperimentType.MOVING_CS_4COND.value


def ensure_context():
    if config is None:
        initialize_context()
# endregion


# %%
# region Helper Functions
def apply_panel_label(fig, label, x=0, y=1, ha="right"):
    fig.suptitle(label, fontsize=11, fontweight="bold", x=x, y=y, va="bottom", ha=ha)


def save_figure(fig, path_out, frmt, **overrides):
    save_kwargs = dict(SAVEFIG_KW)
    save_kwargs.update(overrides)
    fig.savefig(str(path_out), format=frmt, **save_kwargs)


def add_baseline_line(ax, y=1):
    ax.axhline(y, **BASELINE_LINE_KW)


def add_block_dividers(ax, blocks):
    if blocks and len(blocks) > 1:
        for li in range(len(blocks) - 1):
            ax.axvline(li + 0.5, **BLOCK_DIVIDER_KW)


def apply_y_limits(ax, y_limits, labels=None):
    ax.set_ylim(y_limits)
    ax.set_yticks([y_limits[0], 1, y_limits[1]])
    if labels is not None:
        ax.set_yticklabels(labels)


def add_outside_legend(fig, handles, labels, **overrides):
    legend_kwargs = dict(LEGEND_OUTSIDE_KW)
    legend_kwargs.update(overrides)
    fig.legend(handles, labels, **legend_kwargs)


def load_latest_pooled():
    ensure_context()
    paths = [*Path(path_pooled_data).glob("*.pkl")]
    paths = [p for p in paths if "NV per trial per fish" in p.stem]
    paths_csus = [
        p for p in paths
        if len(p.stem.split("_")) > 3 and p.stem.split("_")[3] == csus
    ]
    if not paths_csus:
        paths_csus = [p for p in paths if p.stem.split("_")[-1] == csus]
    paths = paths_csus
    if not paths:
        return pd.DataFrame()
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"Loading: {paths[0].name}")
    return pd.read_pickle(paths[0], compression="gzip")

def load_first_pooled():
    ensure_context()
    paths = [*Path(path_pooled_data).glob("*.pkl")]
    paths = [p for p in paths if "NV per trial per fish" in p.stem and p.stem.split("_")[-1] == csus]
    if not paths:
        return pd.DataFrame()
    return pd.read_pickle(paths[0], compression="gzip")


def filter_pooled_data(data, apply_fish_discard=True):
    ensure_context()
    # Apply global filters shared across plots and stats.
    if apply_fish_discard:
        data = data[~data["Fish"].isin(fish_ids_to_discard)]
    if age_filter != ["all"]:
        data = data[data["Age (dpf)"].isin(age_filter)]
    if setup_color_filter != ["all"]:
        color_rigs = data["ProtocolRig"].str.split("-").str[0]
        data = data[color_rigs.isin(setup_color_filter)]
    return data


def load_phase_plot_data():
    ensure_context()
    data = load_first_pooled()
    if data.empty:
        return data
    return filter_pooled_data(data)


def run_mixed_model(df, formula, groups_col, re_formula=None, method="powell"):
    try:
        model = smf.mixedlm(formula, df, groups=df[groups_col], re_formula=re_formula)
        res = model.fit(reml=False, method=method)
        return res, None
    except Exception as exc:
        return None, str(exc)


def select_baseline_response_columns(df):
    baseline_candidates = [c for c in df.columns if "s before" in c]
    if not baseline_candidates:
        raise ValueError("No baseline column found (expected '* s before').")
    baseline_col = baseline_candidates[0]

    if "Mean CR" in df.columns:
        response_col = "Mean CR"
    else:
        response_candidates = [c for c in df.columns if "Mean" in c and "CR" in c]
        if not response_candidates:
            raise ValueError("No response column found (expected 'Mean CR').")
        response_col = response_candidates[0]

    return baseline_col, response_col


def prepare_main_df(data, apply_fish_discard=True):
    data = filter_pooled_data(data, apply_fish_discard=apply_fish_discard)

    baseline_col, response_col = select_baseline_response_columns(data)

    df = data.rename(columns={"Exp.": "Condition", "Fish": "Fish_ID", "Block name": "Block_name"})
    df.dropna(subset=["Normalized vigor", "Trial number", "Block_name", baseline_col, response_col], inplace=True)

    counts = df.groupby(["Fish_ID", "Block_name"], observed=True).size()
    valid_fish = counts[counts >= minimum_trials_per_fish_per_block].reset_index()["Fish_ID"].unique()
    df = df[df["Fish_ID"].isin(valid_fish)]

    df["Normalized vigor plot"] = df["Normalized vigor"]
    df["Log_Baseline"] = np.log(df[baseline_col] + 1)
    df["Log_Response"] = np.log(df[response_col] + 1)
    df["Trial number"] = df["Trial number"].astype("int")
    df["Trial number"] = df["Trial number"] - df["Trial number"].min() + 1

    return df


def block_order_from_data(df):
    return (
        df.groupby("Block_name", observed=True)["Trial number"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )


def block_boundaries_from_data(df, block_order):
    max_trials = (
        df.groupby("Block_name", observed=True)["Trial number"]
        .max()
        .reindex(block_order)
        .dropna()
    )
    return [val + 0.5 for val in max_trials.iloc[:-1].tolist()]


def get_block_config(number_blocks_original):
    number_trials_block = 1
    blocks_chosen = ["Train"]
    blocks_chosen_labels = []
    block_names = []

    if number_blocks_original == 7:
        block_names = ["Pre-Train", "Early Train", "Train 2", "Train 3", "Train 4", "Late Train", "Test"]
        number_trials_block = 10
        blocks_chosen = ["Pre-Train", "Test"]
        blocks_chosen_labels = blocks_chosen
    elif number_blocks_original == 9:
        block_names = [
            "Early Pre-Train", "Late Pre-Train", "Early Train", "Train 2", "Train 3", "Train 4",
            "Train 5", "Train 6", "Train 7", "Train 8", "Train 9", "Late Train",
            "Early Test", "Test 2", "Test 3", "Test 4", "Test 5", "Late Test",
        ]
        number_trials_block = 5
        blocks_chosen = ["Early Pre-Train", "Early Test", "Late Test"]
        blocks_chosen_labels = ["PTr", "ETe", "LTe"]
    elif number_blocks_original == 12:
        block_names = [
            "Early Pre-Train", "Late Pre-Train", "Early Train", "Train 2", "Train 3", "Train 4",
            "Train 5", "Train 6", "Train 7", "Train 8", "Train 9", "Late Train",
            "Early Test", "Test 2", "Test 3", "Test 4", "Test 5", "Late Test",
            "Early Re-Train", "Re-Train 2", "Re-Train 3", "Re-Train 4", "Re-Train 5", "Late Re-Train",
        ]
        number_trials_block = 5
        blocks_chosen = ["Late Pre-Train", "Early Test", "Late Test"]
        blocks_chosen_labels = ["PT", "ET", "LT"]

    if not blocks_chosen_labels:
        blocks_chosen_labels = blocks_chosen

    return block_names, blocks_chosen, blocks_chosen_labels, number_trials_block


def prepare_block_data(data, block_names, number_trials_block):
    ensure_context()
    if csus == "CS" and config.trials_cs_blocks_10 and block_names:
        blocks = [
            range(x, x + number_trials_block)
            for x in range(config.trials_cs_blocks_10[0][0], config.trials_cs_blocks_10[-1][-1] + 1, number_trials_block)
        ]
        data = data[data["Block name"].isin(config.names_cs_blocks_10)]
        data["Block name"] = data["Block name"].astype(
            CategoricalDtype(categories=config.names_cs_blocks_10, ordered=True)
        )
        data = analysis_utils.change_block_names(data, blocks, block_names)
    return data


def load_block_plot_data():
    ensure_context()
    data = load_first_pooled()
    if data.empty:
        return data, None

    data = filter_pooled_data(data)
    if data.empty:
        return data, None

    number_blocks_original = data["Block name"].nunique()
    block_names, blocks_chosen, blocks_chosen_labels, number_trials_block = get_block_config(number_blocks_original)
    data = prepare_block_data(data, block_names, number_trials_block)
    data = pd.concat([data.loc[data["Exp."] == e] for e in cond_types])

    block_cfg = {
        "blocks_chosen": blocks_chosen,
        "blocks_chosen_labels": blocks_chosen_labels,
        "number_trials_block": number_trials_block,
    }
    return data, block_cfg


def build_between_condition_pairs(blocks, conditions, experiment):
    pairs = []
    if experiment == ExperimentType.MOVING_CS_4COND.value and len(conditions) > 1:
        for bl in blocks:
            pairs += [[(bl, conditions[0]), (bl, conditions[i + 1])] for i in range(len(conditions) - 1)]
        return pairs

    if experiment == ExperimentType.CA8_ABLATION.value and len(conditions) >= 4:
        for bl in blocks:
            pairs += [[(bl, conditions[0]), (bl, conditions[2])], [(bl, conditions[1]), (bl, conditions[3])]]
        return pairs

    for bl in blocks:
        if len(conditions) == 2:
            pairs += [[(bl, conditions[0]), (bl, conditions[1])]]
        elif len(conditions) == 3:
            pairs += [[(bl, conditions[0]), (bl, conditions[1])], [(bl, conditions[0]), (bl, conditions[2])]]
        elif len(conditions) >= 4:
            for idx in range(0, len(conditions) - 1, 2):
                pairs += [[(bl, conditions[idx]), (bl, conditions[idx + 1])]]
    return pairs


def build_within_condition_pairs(blocks, conditions):
    pairs = []
    for cond in conditions:
        for b_i in range(len(blocks) - 1):
            pairs += [[(blocks[b_i], cond), (blocks[b_i + 1], cond)]]
    return pairs
# endregion


# region Pipeline Functions

# %%
def run_data_aggregation():
    # region Data Aggregation & Processing
    ensure_context()
    if not RUN_PROCESS:
        return pd.DataFrame()

    data_pooled = pd.DataFrame()

    columns_groupby = [
        "Strain", "Age (dpf)", "Exp.", "ProtocolRig", "Day", "Fish no.",
        "Fish", "Block name", "Trial number",
    ]
    column_names = [
        f"Mean {gen_config.baseline_window} s before",
        "Mean CR",
        "Normalized vigor",
    ]

    all_data_csus_paths = [*Path(path_all_fish).glob("*.pkl")]
    all_data_csus_paths = [
        p for p in all_data_csus_paths
        if p.stem.split("_")[1] == csus
    ]
    all_data_csus_paths = [
        p for p in all_data_csus_paths
        if p.stem.split("_")[0].lower() in [c.lower() for c in cond_types]
    ]

    data_plot_list = []
    print("Processing data for aggregation...")
    print("CR window:", cr_window)

    for e_i, cond in enumerate(cond_types):
        current_cond_paths = [
            p for p in all_data_csus_paths if p.stem.split("_")[0].lower() == cond.lower()
        ]
        if not current_cond_paths:
            print(f"No file found for condition: {cond}")
            continue

        path = current_cond_paths[0]
        try:
            data = pd.read_pickle(str(path), compression="gzip")
        except Exception as exc:
            print(f"Error reading {path}: {exc}")
            continue

        data.drop(columns=["Angle of point 15 (deg)", "Bout beg", "Bout end"], inplace=True, errors="ignore")
        cond_actual = data["Exp."].unique()[0]
        print(f"Processing {cond_actual}, Fish count: {len(data['Fish'].unique())}")

        with open(path_pooled_data / "NV per trial per fish.txt", "a") as file:
            file.write(f"Number fish in {cond_actual}: {len(data['Fish'].unique())}\n\n")

        if isinstance(data["Scaled vigor (AU)"].dtype, pd.SparseDtype):
            data["Scaled vigor (AU)"] = data["Scaled vigor (AU)"].sparse.to_dense()

        data = analysis_utils.convert_time_from_frame_to_s(data)
        data = data.loc[data["Trial time (s)"].between(TRIAL_WINDOW_S[0], TRIAL_WINDOW_S[1]), :]

        data["Trial type"] = csus
        data = analysis_utils.identify_blocks_trials(data, blocks_dict)

        mask_us_beg = (data["Trial time (s)"] < 0) & (data["US beg"] > 0)
        if mask_us_beg.any():
            print(f"Discarding {mask_us_beg.sum()} trials with US before CS.")
            data.loc[mask_us_beg, "Vigor (deg/ms)"] = np.nan

        if csus == "CS":
            trials_bef_onset = data.loc[
                data["Trial time (s)"].between(-gen_config.baseline_window, 0), :
            ].groupby(columns_groupby, observed=True)["Vigor (deg/ms)"].agg("mean")
            trials_aft_onset = data.loc[
                data["Trial time (s)"].between(cr_window[0], cr_window[1]), :
            ].groupby(columns_groupby, observed=True)["Vigor (deg/ms)"].agg("mean")
        elif csus == "US":
            trials_bef_onset = data.loc[
                data["Trial time (s)"].between(-gen_config.baseline_window - cr_window[1], -cr_window[1]), :
            ].groupby(columns_groupby, observed=True)["Vigor (deg/ms)"].agg("mean")
            trials_aft_onset = data.loc[
                data["Trial time (s)"].between(cr_window[0] - cr_window[1], 0), :
            ].groupby(columns_groupby, observed=True)["Vigor (deg/ms)"].agg("mean")

        data_agg = pd.concat(
            [trials_bef_onset, trials_aft_onset, trials_aft_onset / trials_bef_onset],
            axis=1,
            keys=column_names,
        ).reset_index()

        data_agg["Trial type"] = csus
        data_agg = analysis_utils.identify_blocks_trials(data_agg, blocks_dict)

        cat_cols = ["Exp.", "ProtocolRig", "Age (dpf)", "Day", "Fish no.", "Strain", "Fish"]
        existing_cat_cols = [c for c in cat_cols if c in data_agg.columns]
        data_agg[existing_cat_cols] = data_agg[existing_cat_cols].astype("category")

        if data_agg.empty:
            print(f"Empty aggregated data for condition {cond}, skipping.")
            continue

        data_plot_list.append(data_agg)

    if data_plot_list:
        data_pooled = pd.concat(data_plot_list)
        output_filename = (
            f"NV per trial per fish_CR window {cr_window} s all fish, clean_{cond_types}_{csus}.pkl"
        )
        data_pooled.to_pickle(path_pooled_data / output_filename, compression="gzip")
        print(f"Saved pooled data to {output_filename}")
    else:
        print("No data processed.")
    # endregion
    return data_pooled


# %%
def run_block_summary_lines():
    """Plot per-fish block medians with a median overlay per condition.

    Format: one column per condition, scatter+line per fish, bold median line,
    baseline at 1.0, optional block dividers for CS, and shared y-limits.
    """
    # region Block Summary: Individual Fish Lines
    ensure_context()
    data, block_cfg = load_block_plot_data()

    if data.empty:
        print("Skipping block summary lines (No pooled data file found).")
    else:
        data = data.copy()
        blocks_chosen = block_cfg["blocks_chosen"]
        blocks_chosen_labels = block_cfg["blocks_chosen_labels"]
        data["Block name"] = data["Block name"].astype(CategoricalDtype(categories=blocks_chosen, ordered=True))

        data_agg_block = (
            data.dropna()
            .groupby(["Fish", "Block name", "Exp."], observed=True)["Normalized vigor"]
            .agg("median")
            .reset_index()
        )

        n_cols = max(1, len(cond_types))
        fig_width = n_cols * BLOCK_WIDTH_PER_COND
        fig_b, ax_b = plt.subplots(
            1, n_cols, figsize=(fig_width, BLOCK_FIG_HEIGHT), sharex=True, sharey=True, **FIGURE_KW
        )
        if not isinstance(ax_b, np.ndarray):
            ax_b = [ax_b]

        for e_i, cond in enumerate(cond_types):
            add_baseline_line(ax_b[e_i])
            if csus == "CS" and len(blocks_chosen) > 1:
                add_block_dividers(ax_b[e_i], blocks_chosen)
            data_cond = data_agg_block[data_agg_block["Exp."] == cond]

            for fish in data_cond["Fish"].unique():
                data_fish = data_cond.loc[data_cond["Fish"] == fish, :].copy()
                data_fish["Normalized vigor"] = data_fish["Normalized vigor"].clip(y_lim[0], y_lim[1])

                sns.scatterplot(
                    data=data_fish,
                    x="Block name",
                    y="Normalized vigor",
                    color=color_palette[e_i],
                    edgecolor="k",
                    linewidth=0.5,
                    s=4,
                    alpha=0.4,
                    ax=ax_b[e_i],
                    clip_on=False,
                    zorder=2,
                    legend=False,
                )
                sns.lineplot(
                    data=data_fish,
                    x="Block name",
                    y="Normalized vigor",
                    color=color_palette[e_i],
                    linewidth=0.5,
                    alpha=0.5,
                    sort=False,
                    estimator=None,
                    clip_on=False,
                    ax=ax_b[e_i],
                    zorder=1,
                )

            median_data = (
                data_cond.groupby("Block name", observed=True)["Normalized vigor"]
                .median()
                .reindex(blocks_chosen)
                .reset_index()
            )
            sns.lineplot(
                data=median_data,
                x="Block name",
                y="Normalized vigor",
                color="k",
                linewidth=1,
                marker="o",
                markersize=4,
                markeredgecolor="k",
                markeredgewidth=0.2,
                alpha=0.75,
                estimator=None,
                sort=False,
                ax=ax_b[e_i],
                zorder=3,
                label="_median",
            )

            if HIGHLIGHT_FISH_ID:
                data_fish = data_agg_block[data_agg_block["Fish"] == HIGHLIGHT_FISH_ID]
                data_fish = data_fish[data_fish["Exp."] == cond]
                if not data_fish.empty:
                    sns.lineplot(
                        data=data_fish,
                        x="Block name",
                        y="Normalized vigor",
                        color="k",
                        linewidth=1.5,
                        marker="o",
                        markersize=4,
                        markeredgecolor="k",
                        markeredgewidth=0.5,
                        estimator=None,
                        sort=False,
                        ax=ax_b[e_i],
                        zorder=4,
                    )

            if STATS and len(blocks_chosen) > 1 and not skip_block_stats:
                pairs_within_cond = [
                    (blocks_chosen[b_i], blocks_chosen[b_i + 1]) for b_i in range(len(blocks_chosen) - 1)
                ]
                plot_params_b = {"data": data_cond, "x": "Block name", "y": "Normalized vigor", "order": blocks_chosen}
                annotator_b = Annotator(ax_b[e_i], pairs_within_cond, **plot_params_b)
                annotator_b.configure(
                    test="Mann-Whitney",
                    text_format="star",
                    comparisons_correction="holm-bonferroni",
                    loc="inside",
                    fontsize="8",
                    line_width=0.5,
                    line_height=0.01,
                    hide_non_significant=Hide_non_significant,
                    verbose=False,
                )
                annotator_b.apply_test().annotate()

            ax_b[e_i].set_xticks(list(np.arange(len(blocks_chosen))))
            ax_b[e_i].set_xticklabels(blocks_chosen_labels, rotation=0, ha="center", fontweight="bold", fontsize=10)
            ax_b[e_i].tick_params(axis="x", bottom=False)
            ax_b[e_i].grid(False)
            ax_b[e_i].spines["bottom"].set_visible(False)
            if e_i != 0:
                ax_b[e_i].set_ylabel(None, fontweight="bold")
                ax_b[e_i].spines["left"].set_visible(False)
                ax_b[e_i].set_yticks([])
            else:
                ax_b[0].set_ylabel("Normalized vigor (AU)", fontweight="bold")
                ax_b[0].spines["left"].set_visible(True)
                ax_b[0].spines["left"].set_position(("outward", 3))
                apply_y_limits(ax_b[0], y_lim)
            if e_i != 0:
                ax_b[e_i].set_ylim(y_lim)
            ax_b[e_i].set_title(cond_titles[e_i], loc="left", fontsize=6)
            ax_b[e_i].set_xlabel("")
        for i in range(len(cond_types), len(ax_b)):
            fig_b.delaxes(ax_b[i])

        apply_panel_label(fig_b, "A")
        path_part_b = (
            f"SV lineplot single catch trials_{block_cfg['number_trials_block']}_{age_filter}_{setup_color_filter}_{cond_types}.{frmt}"
        )
        save_figure(fig_b, path_pooled_vigor_fig / path_part_b, frmt)
    # endregion



# %%
def run_block_summary_boxplot():
    """Plot block-wise boxplots of per-fish medians with stats overlays.

    Format: single axis, grouped by block and condition (hue), baseline at 1.0,
    custom outliers per box, annotated between/within-condition stats, and an
    outside legend with fish counts.
    """
    # region Block Summary: Boxplot (Good Exp)
    ensure_context()
    data_box, block_cfg = load_block_plot_data()

    if data_box.empty:
        print("Skipping block summary boxplot (No pooled data file found).")
        return

    blocks_chosen = block_cfg["blocks_chosen"]
    blocks_chosen_labels = block_cfg["blocks_chosen_labels"]
    number_trials_block = block_cfg["number_trials_block"]

    data_box = data_box.copy()
    if DISCARD_FISH_IDS_BOXPLOT:
        data_box = data_box[~data_box["Fish"].isin(DISCARD_FISH_IDS_BOXPLOT)]
    data_box = data_box[data_box["Block name"].isin(blocks_chosen)]
    data_box["Block name"] = data_box["Block name"].astype(
        CategoricalDtype(categories=blocks_chosen, ordered=True)
    )
    data_box = pd.concat([data_box[data_box["Exp."] == cond] for cond in cond_types])
    data_box["Trial number"] = data_box["Trial number"].astype("int")
    data_box["Trial number"] = data_box["Trial number"] - data_box["Trial number"].min() + 1

    data_box = (
        data_box.dropna()
        .groupby(["Fish", "Block name", "Exp."], observed=True)["Normalized vigor"]
        .agg("median")
        .reset_index()
    )

    cond_types_box = [cond for cond in cond_types if cond in data_box["Exp."].unique()]
    if not cond_types_box:
        print("Skipping block summary boxplot (No matching conditions found).")
        return

    data_box = pd.concat([data_box[data_box["Exp."] == cond] for cond in cond_types_box])

    cond_number_fish = [
        len(data_box[data_box["Exp."] == cond]["Fish"].unique()) for cond in cond_types_box
    ]
    cond_title_map = {
        cond: config.cond_dict.get(cond, {}).get("name", cond) for cond in cond_types_box
    }
    cond_titles_box = [
        f"{cond_title_map[cond]} (n={cond_number_fish[idx]})"
        for idx, cond in enumerate(cond_types_box)
    ]

    fig_a, ax_a = plt.subplots(1, 1, figsize=BOXPLOT_FIGSIZE, **FIGURE_KW)
    add_baseline_line(ax_a)

    sns.boxplot(
        x="Block name",
        y="Normalized vigor",
        hue="Exp.",
        hue_order=cond_types_box,
        fill=True,
        showfliers=False,
        dodge=True,
        palette=color_palette,
        data=data_box,
        notch=False,
        showmeans=False,
        meanline=False,
        medianprops=dict(color="k", linestyle="--", linewidth=0.5),
        boxprops=dict(edgecolor="k", lw=0.5),
        whiskerprops=dict(color="k", ls="-", lw=0.5),
        capprops=dict(color="k", lw=0.5),
        saturation=1,
        ax=ax_a,
    )

    data_box["Normalized vigor"] = data_box["Normalized vigor"].clip(y_lim[0], y_lim[1])

    # Draw outliers manually so they match each box color.
    positions = np.arange(len(blocks_chosen))
    width = 0.8
    offset = width / len(cond_types_box)
    for i, exp in enumerate(cond_types_box):
        for j, block in enumerate(blocks_chosen):
            exp_data = data_box[
                (data_box["Exp."] == exp) & (data_box["Block name"] == block)
            ]["Normalized vigor"].to_numpy()
            if exp_data.size == 0:
                continue
            q1, q3 = np.percentile(exp_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = exp_data[(exp_data < lower_bound) | (exp_data > upper_bound)]
            if outliers.size == 0:
                continue
            jitter = np.random.uniform(-0.1, 0.1, size=outliers.size)
            x_vals = positions[j] + offset * (i - (len(cond_types_box) - 1) / 2) + jitter
            ax_a.scatter(
                x_vals,
                outliers,
                color=color_palette[i],
                marker=".",
                alpha=0.8,
                edgecolor="k",
                linewidth=0.5,
                clip_on=False,
                s=8,
            )

    if STATS:
        print("Mann-Whitney U test")
        hue_plot_params = {
            "data": data_box,
            "x": "Block name",
            "y": "Normalized vigor",
            "order": blocks_chosen,
            "hue": "Exp.",
            "hue_order": list(cond_types_box),
            "palette": color_palette,
        }

        pairs_between_exp = []

        def add_pairs(cond_pairs):
            for bl in blocks_chosen:
                for left_idx, right_idx in cond_pairs:
                    if left_idx < len(cond_types_box) and right_idx < len(cond_types_box):
                        pairs_between_exp.append(
                            [(bl, cond_types_box[left_idx]), (bl, cond_types_box[right_idx])]
                        )

        exp_name = EXPERIMENT
        if exp_name == ExperimentType.MOVING_CS_4COND.value:
            add_pairs([(0, i + 1) for i in range(len(cond_types_box) - 1)])
        elif exp_name in {
            ExperimentType.FIRST_DELAY.value,
            ExperimentType.ALL_DELAY.value,
            ExperimentType.ALL_3S_TRACE.value,
            ExperimentType.TRACE_10S.value,
            ExperimentType.DELAY_LONG_TERM_SPACED_PUROMYCIN_5MG_ALL.value,
            "increasingTraceMerged",
        }:
            if len(cond_types_box) == 2:
                add_pairs([(0, 1)])
            elif len(cond_types_box) == 3:
                add_pairs([(0, 1), (0, 2)])
            elif len(cond_types_box) == 4:
                add_pairs([(0, 1), (2, 3)])
        elif exp_name == ExperimentType.DELAY_MK801.value:
            add_pairs([(0, 1), (2, 3), (4, 5)])
        elif exp_name in {
            ExperimentType.TRACE_MK801.value,
            ExperimentType.TRACE_PUROMYCIN_SHORT.value,
            "tracepuromycin",
        }:
            add_pairs([(0, 1), (2, 3)])
        elif exp_name == ExperimentType.CA8_ABLATION.value:
            add_pairs([(0, 2), (1, 3)])

        if pairs_between_exp:
            ax_a.set_ylim((y_lim[0], y_lim[1]))
            annotator = Annotator(ax_a, pairs_between_exp, **hue_plot_params)
            annotator.configure(
                test="Mann-Whitney",
                text_format="star",
                comparisons_correction="holm-bonferroni",
                verbose=2,
                loc="outside",
                fontsize="9",
                line_width=0.5,
                line_height=0.02,
                hide_non_significant=Hide_non_significant,
                pvalue_thresholds=[[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]],
            ).apply_test().annotate()

        if EXPERIMENT != ExperimentType.MOVING_CS_4COND.value:
            print("Wilcoxon test")
            pairs_within_exp = build_within_condition_pairs(blocks_chosen, cond_types_box)
            if pairs_within_exp:
                ax_a.set_ylim((y_lim[0], y_lim[1] + 0.1))
                annotator = Annotator(ax_a, pairs_within_exp, **hue_plot_params)
                annotator.configure(
                    test="Wilcoxon",
                    text_format="star",
                    comparisons_correction="holm-bonferroni",
                    verbose=2,
                    loc="outside",
                    fontsize="9",
                    line_width=0.5,
                    line_height=0.02,
                    hide_non_significant=Hide_non_significant,
                    pvalue_thresholds=[
                        [1e-4, "****"],
                        [1e-3, "***"],
                        [1e-2, "**"],
                        [0.05, "*"],
                        [1, "ns"],
                    ],
                ).apply_test().annotate()

    ax_a.spines["bottom"].set_visible(False)
    ax_a.grid(False)
    ax_a.set_xticks(list(np.arange(len(blocks_chosen))))
    ax_a.set_xticklabels(blocks_chosen_labels, rotation=0, ha="center", fontweight="bold", fontsize=10)
    ax_a.set_ylabel("Normalized vigor (AU)")
    apply_y_limits(ax_a, y_lim, labels=[f"<={y_lim[0]}", "1.0", f">={y_lim[1]}"])
    ax_a.locator_params(axis="y", tight=False, nbins=4)
    ax_a.tick_params(axis="both", which="both", bottom=False, top=False, right=False)
    ax_a.set_xlabel("")

    handles, _ = ax_a.get_legend_handles_labels()
    ax_a.legend().remove()
    add_outside_legend(fig_a, handles[:len(cond_types_box)], cond_titles_box)
    apply_panel_label(fig_a, "D")

    if EXPERIMENT == ExperimentType.MOVING_CS_4COND.value:
        fig_a.set_size_inches(5/2.54, 5/2.54)

    path_part = f"3.1_{number_trials_block}_{age_filter}_{setup_color_filter}_{cond_types_box}.{frmt}"
    save_figure(fig_a, path_pooled_vigor_fig / path_part, frmt)
    # endregion
    
def run_phase_summary():
    """Plot phase-level medians per condition with per-fish trajectories.

    Format: one column per condition, per-fish scatter+lines across blocks,
    optional highlighted fish, baseline at 1.0, and phase divider lines.
    """
    # region Phase Summary: Median per Block
    ensure_context()
    data = load_phase_plot_data()

    if data.empty:
        print("Skipping phase summary (No pooled data file found).")
    else:
        data = pd.concat([data[data["Exp."] == cond] for cond in cond_types])
        data["Trial number"] = data["Trial number"].astype("int")
        data["Trial number"] = data["Trial number"] - data["Trial number"].min() + 1

        if csus == "CS":
            blocks_order = config.names_cs_blocks_10
            trials_blocks_10 = config.trials_cs_blocks_10
            trials_blocks_phases = config.trials_cs_blocks_phases
            names_blocks_phases = config.names_cs_blocks_phases
        else:
            blocks_order = config.names_us_blocks_10
            trials_blocks_10 = config.trials_us_blocks_10
            trials_blocks_phases = config.trials_us_blocks_phases
            names_blocks_phases = config.names_us_blocks_phases

        if blocks_order:
            data = data[data["Block name"].isin(blocks_order)]
            data["Block name"] = data["Block name"].astype(CategoricalDtype(categories=blocks_order, ordered=True))

        def _phase_markers(blocks_10, blocks_phases):
            if not blocks_10 or not blocks_phases:
                return [], []
            trial_to_block = {}
            for block_index, trials in enumerate(blocks_10):
                for t in trials:
                    trial_to_block[int(t)] = block_index
            phase_block_indices = []
            for phase_trials in blocks_phases:
                indices = sorted({trial_to_block.get(int(t)) for t in phase_trials if int(t) in trial_to_block})
                indices = [i for i in indices if i is not None]
                if indices:
                    phase_block_indices.append(indices)
            tick_positions = [indices[0] for indices in phase_block_indices]
            vline_positions = [indices[-1] + 0.5 for indices in phase_block_indices[:-1]]
            return tick_positions, vline_positions

        tick_positions, vline_positions = _phase_markers(trials_blocks_10, trials_blocks_phases)

        data_median = [0 for _ in cond_types]
        n_cols = max(1, len(cond_types))
        fig_width = n_cols * BLOCK_WIDTH_PER_COND
        fig, ax = plt.subplots(
            1, n_cols, figsize=(fig_width, BLOCK_FIG_HEIGHT), sharex=True, sharey=True, **FIGURE_KW
        )
        if not isinstance(ax, np.ndarray):
            ax = [ax]

        for cond_i, data_cond in enumerate([data[data["Exp."] == cond_type] for cond_type in cond_types]):
            add_baseline_line(ax[cond_i])

            if csus == "CS":
                for li in vline_positions:
                    ax[cond_i].axvline(li, color="gray", alpha=0.95, linestyle="-", linewidth=0.5)

                if tick_positions and len(names_blocks_phases) == len(tick_positions):
                    ax[cond_i].set_xticks(tick_positions)
                    ax[cond_i].set_xticklabels(names_blocks_phases, fontweight="bold")
                else:
                    ax[cond_i].set_xticks(np.arange(len(blocks_order)))
                    ax[cond_i].set_xticklabels(blocks_order, fontweight="bold")
            else:
                if tick_positions and len(names_blocks_phases) == len(tick_positions):
                    ax[cond_i].set_xticks(tick_positions)
                    ax[cond_i].set_xticklabels(names_blocks_phases, fontweight="bold")
                else:
                    ax[cond_i].set_xticks([0])
                    ax[cond_i].set_xticklabels(["Train"], fontweight="bold")

            data_median[cond_i] = (
                data_cond.groupby(["Fish", "Block name", "Exp."], observed=True)["Normalized vigor"]
                .agg([np.nanmedian])
                .reset_index()
            )
            data_median[cond_i]["nanmedian"] = np.clip(
                data_median[cond_i]["nanmedian"], y_lim[0], y_lim[1]
            )

            sns.scatterplot(
                data=data_median[cond_i],
                x="Block name",
                y="nanmedian",
                color=color_palette[cond_i],
                ax=ax[cond_i],
                alpha=0.4,
                linewidth=0.5,
                markers=False,
                marker=".",
                size=4,
                edgecolor="k",
                facecolor=color_palette[cond_i],
                legend=False,
                clip_on=False,
                zorder=2,
            )
            sns.lineplot(
                data=data_median[cond_i],
                x="Block name",
                y="nanmedian",
                color=color_palette[cond_i],
                ax=ax[cond_i],
                alpha=0.5,
                linewidth=0.5,
                markers=False,
                legend=False,
                clip_on=False,
                units="Fish",
                estimator=None,
                zorder=1,
            )

            if HIGHLIGHT_FISH_ID:
                data_highlight = data_median[cond_i][data_median[cond_i]["Fish"] == HIGHLIGHT_FISH_ID]
                if not data_highlight.empty:
                    sns.lineplot(
                        data=data_highlight,
                        x="Block name",
                        y="nanmedian",
                        color="k",
                        linewidth=1.5,
                        marker="o",
                        markersize=4,
                        markeredgecolor="k",
                        markeredgewidth=0.5,
                        estimator=None,
                        sort=False,
                        ax=ax[cond_i],
                        zorder=4,
                    )

            ax[cond_i].tick_params(axis="x", bottom=False)
            ax[cond_i].set_xlabel("")
            ax[cond_i].spines["bottom"].set_visible(False)
            ax[cond_i].spines["left"].set_position(("outward", 3))

            if csus == "CS":
                ax[cond_i].set_ylabel("Normalized vigor (AU)")
            else:
                ax[cond_i].set_ylabel("Normalized vigor (AU)\nrelative to US")
            apply_y_limits(ax[cond_i], y_lim, labels=[f"<={y_lim[0]}", "1", f">={y_lim[1]}"])

            ax[cond_i].set_title(cond_titles[cond_i], loc="left")
        for i in range(len(cond_types), len(ax)):
            fig.delaxes(ax[i])

        apply_panel_label(fig, "B")
        path_part = f"NV single fish median of trials_{cond_types}_{csus}.{frmt}"
        save_figure(fig, path_pooled_vigor_fig / path_part, frmt)
    # endregion



# %%
def run_trial_by_trial(data_pooled=None):
    """Plot trial-by-trial median trajectories with LME significance annotations.

    Format: single axis, median line with CI per condition, optional annotation
    lanes for global/block/trial significance, and block boundary markers.
    """
    # region Trial-by-Trial: Lineplot + LME
    ensure_context()
    if data_pooled is None or data_pooled.empty:
        data_plot = load_latest_pooled()
    else:
        data_plot = data_pooled.copy()

    if data_plot.empty:
        print("No matching pooled data file found.")
    else:
        if EXPORT_TEXT:
            output_path = path_pooled_data / "data_output.txt"
            data_plot[["Exp.", "Fish", "Block name", "Trial number", "Normalized vigor"]].to_csv(
                output_path, sep="\t", index=False
            )

        try:
            df_main = prepare_main_df(data_plot, apply_fish_discard=APPLY_FISH_DISCARD_LME)
        except ValueError as exc:
            print(f"Skipping Trial-by-Trial plot ({exc})")
            df_main = pd.DataFrame()

        if df_main.empty:
            print("Dataframe is empty.")
        else:
            print(f"Data Prepared: {len(df_main)} rows, {df_main['Fish_ID'].nunique()} subjects.")
            ref_cond = cond_types[0]
            block_order = block_order_from_data(df_main)
            block_centers = {
                b: df_main[df_main["Block_name"] == b]["Trial number"].mean()
                for b in block_order
            }
            block_boundaries = block_boundaries_from_data(df_main, block_order)

            global_interactions = None
            ph_df = pd.DataFrame()
            sig_trials = []
            model_errors = []

            # LME analysis matches Multivariate LME analysis NEW.py output.
            if RUN_LME and STATS:
                print("\n--- 4.1 Global ANCOVA ---")
                f_global = f"Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}')) * C(Block_name)"
                res_global, err = run_mixed_model(df_main, f_global, "Fish_ID", re_formula="~Log_Baseline")

                if res_global:
                    print(res_global.summary())
                    summary_table = res_global.summary().tables[1]
                    global_interactions = summary_table.loc[lambda x: x.index.str.contains(":")]
                    global_interactions["P>|z|"] = pd.to_numeric(global_interactions["P>|z|"], errors="coerce")
                else:
                    model_errors.append({"Type": "Global", "Unit": "All", "Error": err})

                print("\n--- 4.2 Post-Hoc Block Analysis ---")
                posthoc_res = []
                for block in block_order:
                    df_blk = df_main[df_main["Block_name"] == block].copy()
                    if df_blk["Condition"].nunique() < 2:
                        continue

                    df_blk["Trial_Centered"] = df_blk["Trial number"] - df_blk["Trial number"].mean()
                    f_local = f"Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}')) * Trial_Centered"
                    res_local, err = run_mixed_model(df_blk, f_local, "Fish_ID", re_formula="~Log_Baseline")

                    if res_local:
                        params = res_local.params
                        pvals = res_local.pvalues
                        main_terms = [t for t in params.index if "Condition" in t and ":" not in t]
                        slope_terms = [t for t in params.index if "Condition" in t and ":" in t]
                        if main_terms and slope_terms:
                            term_main = main_terms[0]
                            term_slope = slope_terms[0]
                            posthoc_res.append({
                                "Block": block,
                                "Coef_Mean": params[term_main],
                                "P_Mean": pvals[term_main],
                                "Coef_Slope": params[term_slope],
                                "P_Slope": pvals[term_slope],
                            })
                    else:
                        model_errors.append({"Type": "Block", "Unit": block, "Error": err})

                if posthoc_res:
                    ph_df = pd.DataFrame(posthoc_res)
                    _, ph_df["P_Mean_FDR"], _, _ = multipletests(ph_df["P_Mean"], alpha=0.05, method="fdr_bh")
                    _, ph_df["P_Slope_FDR"], _, _ = multipletests(ph_df["P_Slope"], alpha=0.05, method="fdr_bh")
                    ph_df["Sig_Mean"] = ph_df["P_Mean_FDR"] < 0.05
                    ph_df["Sig_Slope"] = ph_df["P_Slope_FDR"] < 0.05

                print("\n--- 4.3 Trial-by-Trial Analysis ---")
                trial_res = []
                for trial in sorted(df_main["Trial number"].unique()):
                    df_t = df_main[df_main["Trial number"] == trial].copy()
                    if df_t["Condition"].nunique() < 2:
                        continue

                    f_trial = f"Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}'))"
                    res_t, err = run_mixed_model(df_t, f_trial, "Fish_ID")

                    if res_t:
                        term = [x for x in res_t.params.index if "Condition" in x][0]
                        trial_res.append({"Trial": trial, "P_raw": res_t.pvalues[term]})

                if trial_res:
                    t_df = pd.DataFrame(trial_res)
                    reject, _, _, _ = multipletests(t_df["P_raw"], alpha=0.05, method="fdr_bh")
                    sig_trials = t_df[reject]["Trial"].tolist()

                if model_errors:
                    print("\n--- LME Fit Warnings ---")
                    for err in model_errors:
                        print(f"{err['Type']} ({err['Unit']}): {err['Error']}")

            fig_c, ax_c = plt.subplots(1, 1, facecolor="white", figsize=TRIAL_BY_TRIAL_FIGSIZE, layout="tight")

            for i, cond in enumerate(cond_types):
                sns.lineplot(
                    data=df_main[df_main["Condition"] == cond],
                    x="Trial number",
                    y="Normalized vigor plot",
                    color=color_palette[i],
                    label=cond_titles[i],
                    n_boot=n_boot,
                    **TRIAL_LINE_KW,
                    ax=ax_c,
                )

            # Annotation lanes for global, block, and trial-level significance markers.
            y_gold = y_lim_plot[1] + 0.25
            y_silver = y_lim_plot[1] + 0.15
            y_black = y_lim_plot[1] + 0.05

            if global_interactions is not None:
                for term, row in global_interactions.iterrows():
                    if row["P>|z|"] < 0.05:
                        match = re.search(r"C\(Block_name\)\[T\.(.+?)\]", term)
                        if match and match.group(1) in block_centers:
                            ax_c.text(
                                block_centers[match.group(1)],
                                y_gold,
                                match.group(1),
                                color="gold",
                                ha="center",
                                fontsize=7,
                                fontweight="bold",
                            )

            if not ph_df.empty:
                for _, row in ph_df.iterrows():
                    blk = row["Block"]
                    if blk in block_centers:
                        if row["Sig_Mean"]:
                            ax_c.text(
                                block_centers[blk], y_silver, "Mean", color="gray", ha="center", fontsize=6
                            )
                        if row["Sig_Slope"]:
                            ax_c.text(
                                block_centers[blk],
                                y_silver - 0.05,
                                "Rate",
                                color="firebrick",
                                ha="center",
                                fontsize=6,
                                fontweight="bold",
                            )

            if sig_trials:
                ax_c.scatter(
                    sig_trials,
                    [y_black] * len(sig_trials),
                    color="black",
                    marker="*",
                    s=8,
                    zorder=10,
                    linewidths=0,
                )

            vline_positions = []
            spec_var = None
            if csus == "CS":
                try:
                    import my_experiment_specific_variables as spec_var
                    vline_positions = list(spec_var.where_v_lines_1)
                except Exception:
                    vline_positions = block_boundaries

            for boundary in vline_positions:
                ax_c.axvline(boundary, color="gray", alpha=0.5, linewidth=0.5)

            ax_c.set_xlim(0, df_main["Trial number"].max() + 1)
            ax_c.set_ylim(y_lim_plot[0], y_lim_plot[1] + 0.3)
            ax_c.set_ylabel("Normalized vigor (ratio)")
            ax_c.legend().remove()

            save_root = path_pooled_vigor_fig
            if spec_var is not None and hasattr(spec_var, "path_pooled_vigor_fig"):
                save_root = spec_var.path_pooled_vigor_fig
            save_path = save_root / f"NV_LME_Visualized_Raw_{cond_types}.{lme_frmt}"
            save_figure(fig_c, save_path, lme_frmt)
            print(f"Figure saved: {save_path.name}")
    # endregion

# endregion


# region Main
def main():
    initialize_context()
    print(f"Experiment: {EXPERIMENT}")
    print(f"Conditions: {cond_types}")

    data_pooled = pd.DataFrame()
    if RUN_PROCESS:
        data_pooled = run_data_aggregation()

    if RUN_BLOCK_SUMMARY_LINES:
        run_block_summary_lines()

    if RUN_BLOCK_SUMMARY_BOXPLOT:
        run_block_summary_boxplot()

    if RUN_PHASE_SUMMARY:
        run_phase_summary()

    if RUN_TRIAL_BY_TRIAL:
        run_trial_by_trial(data_pooled)


if __name__ == "__main__":
    main()
# endregion

