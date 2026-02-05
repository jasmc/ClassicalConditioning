"""
Normalized Vigor Plotting Pipeline
===================================

Render pooled NV plots including:
- Block summaries (lines and boxplots)
- Phase medians
- Trial-by-trial LME analysis
"""

# %%
# region Imports
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib.patches import ConnectionPatch
from pandas.api.types import CategoricalDtype
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests

# Add the repository root containing shared modules to the Python path.
if "__file__" in globals():
    module_root = Path(__file__).resolve()
else:
    module_root = Path.cwd()

import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning

import analysis_utils
import figure_saving
import file_utils
import plotting_style
from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as gen_config

pd.set_option("mode.copy_on_write", True)

# Apply shared plotting style with script-specific overrides.
plotting_style.set_plot_style(rc_overrides={"figure.constrained_layout.use": False})
# endregion Imports


# region Parameters
# ------------------------------------------------------------------------------
# Pipeline Control Flags
# ------------------------------------------------------------------------------
RUN_PROCESS = True
RUN_BLOCK_SUMMARY_LINES = True
RUN_BLOCK_SUMMARY_BOXPLOT = True
RUN_PHASE_SUMMARY = True
RUN_TRIAL_BY_TRIAL = True

# ------------------------------------------------------------------------------
# Global Settings
# ------------------------------------------------------------------------------
EXPERIMENT = ExperimentType.ALL_DELAY.value

# Apply per-experiment discarded fish list if present under "Processed data".
APPLY_FISH_DISCARD = False

csus = "CS"  # Stimulus alignment: "CS" or "US".
STATS = True  # Enable statistical tests.
RUN_LME = True  # Enable LME analysis.
EXPORT_TEXT = False  # Export trial data to text.

minimum_trials_per_fish_per_block = 6
age_filter = ["all"]
setup_color_filter = ["all"]

# ------------------------------------------------------------------------------
# Data Aggregation Parameters
# ------------------------------------------------------------------------------
TRIAL_WINDOW_S = (-21, 21)
# Invalidate per-trial metrics if NaN fraction exceeds this in either window.
# (I.e., if more than this fraction of time has missing vigor in baseline and/or CR window.)
MAX_NAN_FRAC_PER_WINDOW = 0.90
APPLY_MAX_NAN_FRAC_PER_WINDOW = False  # If False, skip NaN-fraction invalidation; paths get no _nanFracFilt suffix.

# ------------------------------------------------------------------------------
# Shared Plot Parameters
# ------------------------------------------------------------------------------
frmt = "svg"
Hide_non_significant = True
y_lim = (0.8, 1.2)

# ------------------------------------------------------------------------------
# Phase Summary Parameters
# ------------------------------------------------------------------------------
HIGHLIGHT_FISH_ID = [
    # '20221115_07',  # delay
    # '20230307_12',  # 3s trace
    # '20230307_04',  # 10s trace
    # '20221115_09',  # control
]

# ------------------------------------------------------------------------------
# Trial-by-Trial Parameters
# ------------------------------------------------------------------------------
n_boot = 100
y_lim_plot = (0.7, 1.4)

# %% Suppress statsmodels MixedLM convergence warnings (boundary / singular fits)

# warnings.filterwarnings(
#     # "ignore",
#     # message=r".*MLE may be on the boundary of the parameter space.*",
#     category=ConvergenceWarning,
# )

# LME Model Configuration
# -----------------------
# Random effects formula options:
#   - "~Log_Baseline": Random slope by baseline (default, ANCOVA-style adjustment)
#   - "~Condition": Random slope by condition (use if conditions have systematically
#                   different variances; matches old implementation)
#   - None: Random intercept only (simplest model)
LME_RE_FORMULA = "~Log_Baseline"
# LME_RE_FORMULA = None
# "~Condition"

# Jitter for singular-fit-prone datasets:
#   Adds microscopic noise to break ties when identical values cause singular fits.
#   Set to 0 to disable. Typical value: 1e-4 (0.0001)
LME_JITTER_SCALE = 0.0
LME_JITTER_SEED = 10  # For reproducibility

# Optimizer for LME models:
#   - "lbfgs": Faster, better for large global models (old code used this for global)
#   - "powell": More robust for singular/small datasets (block-level, trial-by-trial)
LME_GLOBAL_METHOD = "lbfgs"  # Optimizer for global ANCOVA model
LME_LOCAL_METHOD = "powell"  # Optimizer for block-level and trial-by-trial models

# Debug printing
#   If True, prints a compact post-hoc table including raw and FDR-corrected p-values.
#   This is useful to verify that plotted "Rate" markers match the FDR thresholding.
LME_DEBUG_POSTHOC_TABLE = False
# endregion Parameters


# region Plot Formatting
SAVEFIG_KW = {"dpi": 600, "transparent": False, "bbox_inches": "tight"}
FIGURE_KW = {"facecolor": "white", "clip_on": False, "constrained_layout": False}

BASELINE_LINE_KW = {"color": "k", "alpha": 1, "lw": plt.rcParams["ytick.major.width"], "zorder": 0}
BLOCK_DIVIDER_KW = {"color": "gray", "alpha": 0.95, "linestyle": "-", "linewidth": 0.5}

BLOCK_FIG_HEIGHT = 4 / 2.54
BLOCK_WIDTH_PER_COND = 2
BOXPLOT_FIGSIZE = (6 / 2.54, 6 / 2.54)
TRIAL_BY_TRIAL_FIGSIZE = (6 / 2.54, 6 / 2.54)

LEGEND_OUTSIDE_KW = {"frameon": False, "bbox_to_anchor": (1.02, 1.0), "loc": "upper left", "borderaxespad": 0}

TRIAL_LINE_KW = {
    "alpha": 0.9,
    "linewidth": 0.5,
    "estimator": "median",
    "errorbar": ("ci", 95),
    "err_style": "band",
    "err_kws": {"alpha": 0.3},
    "seed": 10,
}
# endregion Plot Formatting


# region Context Setup
config = None
cond_types = []
cond_titles = []
color_palette = []
cr_window = None
blocks_dict = {}
path_pooled_vigor_fig = None
path_scaled_vigor_fig = None
path_orig_pkl = None
path_all_fish = None
path_pooled_data = None
fish_ids_to_discard = []
skip_block_stats = False


def initialize_context():
    global config, cond_types, cond_titles, color_palette, cr_window, blocks_dict
    global path_pooled_vigor_fig, path_orig_pkl, path_all_fish, path_pooled_data, path_scaled_vigor_fig
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
        path_processed_data,
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

    # All figures from this script should be saved under a "Normalized vigor" subfolder
    # inside the experiment's pooled-figure output directory.
    nan_suffix = "_nanFracFilt" if APPLY_MAX_NAN_FRAC_PER_WINDOW else ""
    path_scaled_vigor_fig = path_pooled_vigor_fig / (f"Normalized vigor{nan_suffix}")
    path_scaled_vigor_fig.mkdir(parents=True, exist_ok=True)

    # Single discard list source: Processed data/Discarded_fish_IDs.txt
    discard_file = path_processed_data / "Discarded_fish_IDs.txt"

    fish_ids_to_discard = []
    discard_source = None
    if APPLY_FISH_DISCARD and discard_file.exists():
        fish_ids_to_discard = file_utils.load_discarded_fish_ids(discard_file)
        discard_source = discard_file

    if discard_source is not None:
        try:
            src_str = str(discard_source)
        except Exception:
            src_str = "<unknown>"
        print(f"  Loaded {len(fish_ids_to_discard)} discarded fish IDs from: {src_str}")

    skip_block_stats = EXPERIMENT == ExperimentType.MOVING_CS_4COND.value


def ensure_context():
    if config is None:
        initialize_context()


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


# region Helper Functions
def apply_panel_label(fig, label, x=0, y=1, ha="right"):
    fig.suptitle(label, fontsize=11, fontweight="bold", x=x, y=y, va="bottom", ha=ha)


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


def save_fig(fig, stem: str, frmt: str) -> Path:
    """Save a figure under path_scaled_vigor_fig with consistent naming."""
    ensure_context()
    if path_scaled_vigor_fig is None:
        raise RuntimeError("Context not initialized: path_scaled_vigor_fig is None")
    safe_stem = _sanitize_filename(stem)
    safe_frmt = str(frmt).lstrip(".")
    save_path = path_scaled_vigor_fig / f"{safe_stem}.{safe_frmt}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), format=safe_frmt, **SAVEFIG_KW)
    return save_path


def save_figure(fig, path_out, frmt, **overrides):
    # Centralized save: mkdir + Windows-safe names + enforce suffix to match frmt.
    figure_saving.save_figure(fig, path_out, frmt=frmt, savefig_kw=SAVEFIG_KW, **overrides)


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
    fish_suffix = "_selectedFish" if APPLY_FISH_DISCARD else "_allFish"
    paths = [*Path(path_pooled_data).glob("*.pkl")]
    paths = [p for p in paths if "NV per trial per fish" in p.stem and fish_suffix in p.stem]
    paths = [p for p in paths if ("_nanFracFilt" in p.stem) == APPLY_MAX_NAN_FRAC_PER_WINDOW]
    paths_csus = [
        p for p in paths
        if len(p.stem.split("_")) > 3 and p.stem.split("_")[3] == csus
    ]
    if not paths_csus:
        paths_csus = [p for p in paths if p.stem.split("_")[-1].replace(fish_suffix, "") == csus]
    paths = paths_csus
    if not paths:
        return pd.DataFrame()
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"  Loading: {paths[0].name}")
    return pd.read_pickle(paths[0], compression="gzip")

def load_first_pooled():
    ensure_context()
    fish_suffix = "_selectedFish" if APPLY_FISH_DISCARD else "_allFish"
    nan_suffix = "_nanFracFilt" if APPLY_MAX_NAN_FRAC_PER_WINDOW else ""
    paths = [*Path(path_pooled_data).glob("*.pkl")]
    paths = [p for p in paths if "NV per trial per fish" in p.stem and fish_suffix in p.stem]
    paths = [p for p in paths if ("_nanFracFilt" in p.stem) == APPLY_MAX_NAN_FRAC_PER_WINDOW]
    # Filter by csus: the suffix pattern is ..._{csus}{fish_suffix}[_nanFracFilt].pkl
    paths = [p for p in paths if p.stem.endswith(f"_{csus}{fish_suffix}{nan_suffix}")]
    print(paths)
    if not paths:
        return pd.DataFrame()
    return pd.read_pickle(paths[0], compression="gzip")


def filter_pooled_data(data, apply_fish_discard=True, source: str = ""):
    ensure_context()
    # Apply global filters shared across plots and stats.
    
    fish_col = "Fish"
    
    prefix = f"  [{source}] " if source else "  "
    
    if fish_col is not None:
        before = data[fish_col].nunique()
        print(f"{prefix}Fish unique before discard: {before}")
    
    if apply_fish_discard and APPLY_FISH_DISCARD and fish_ids_to_discard and fish_col is not None:
        data = data[~data[fish_col].isin(fish_ids_to_discard)]
        after = data[fish_col].nunique()
        print(f"{prefix}Fish unique after discard: {after}")
    
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
    return filter_pooled_data(data, source="load_phase_plot_data")


def run_mixed_model(
    df,
    formula,
    groups_col,
    re_formula=None,
    method="powell",
    reml=False,
    rmel=None,
):
    try:
        # Backwards/typo compatibility: allow callers to pass `rmel=` (common typo)
        # while the actual statsmodels argument is `reml=`.
        if rmel is not None:
            reml = rmel
        model = smf.mixedlm(formula, df, groups=df[groups_col], re_formula=re_formula)
        res = model.fit(reml=reml, method=method)
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


def prepare_main_df(data, apply_fish_discard=True, jitter_scale=0.0, jitter_seed=42):
    data = filter_pooled_data(data, apply_fish_discard=apply_fish_discard, source="prepare_main_df")

    baseline_col, response_col = select_baseline_response_columns(data)

    df = data.rename(columns={"Exp.": "Condition", "Fish": "Fish_ID", "Block name": "Block_name"})
    df.dropna(subset=["Normalized vigor", "Trial number", "Block_name", baseline_col, response_col], inplace=True)

    counts = df.groupby(["Fish_ID", "Block_name"], observed=True).size()
    valid_fish = counts[counts >= minimum_trials_per_fish_per_block].reset_index()["Fish_ID"].unique()
    df = df[df["Fish_ID"].isin(valid_fish)]

    df["Normalized vigor plot"] = df["Normalized vigor"]
    df["Log_Baseline"] = np.log(df[baseline_col] + 1)
    df["Log_Response"] = np.log(df[response_col] + 1)

    # Optional jitter for singular-fit-prone datasets:
    # Adds microscopic noise to break ties if you have identical values.
    if jitter_scale > 0:
        np.random.seed(jitter_seed)
        noise = np.random.normal(0, jitter_scale, size=len(df))
        df["Log_Response"] = df["Log_Response"] + noise
        print(f"  Applied jitter (scale={jitter_scale}, seed={jitter_seed}) to Log_Response")

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

    data = filter_pooled_data(data, source="load_block_plot_data")
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


def filter_pairs_in_data_no_hue(pairs, data: pd.DataFrame, x_col: str):
    """Filter (x1, x2) pairs to only those present in `data[x_col]`."""
    if not pairs or data is None or data.empty or x_col not in data.columns:
        return []

    x_values = set(data[x_col].dropna().unique().tolist())
    kept = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = pair
        if a in x_values and b in x_values:
            kept.append(pair)
    return kept


def filter_pairs_in_data_with_hue(pairs, data: pd.DataFrame, x_col: str, hue_col: str):
    """Filter ((x, hue), (x, hue)) pairs to only those present in `data`."""
    if not pairs or data is None or data.empty or x_col not in data.columns or hue_col not in data.columns:
        return []

    existing = set(
        data[[x_col, hue_col]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    kept = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        left, right = pair
        try:
            left_t = tuple(left)
            right_t = tuple(right)
        except TypeError:
            continue
        if left_t in existing and right_t in existing:
            kept.append(pair)
    return kept
# endregion Helper Functions


# region Pipeline Functions

# %%
# region data_aggregation
def run_data_aggregation():
    """Aggregate trial-level data into per-fish, per-block normalized vigor."""
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
    print(f"  CR window: {cr_window}")

    for e_i, cond in enumerate(cond_types):
        current_cond_paths = [
            p for p in all_data_csus_paths if p.stem.split("_")[0].lower() == cond.lower()
        ]
        if not current_cond_paths:
            print(f"  [WARN] No file found for condition: {cond}")
            continue

        path = current_cond_paths[0]
        try:
            data = pd.read_pickle(str(path), compression="gzip")
        except Exception as exc:
            print(f"  [ERROR] Reading {path.name}: {exc}")
            continue

        data.drop(columns=["Angle of point 15 (deg)", "Bout beg", "Bout end"], inplace=True, errors="ignore")
        


        print(data)
        print(data.columns)
        print(data["Fish"].nunique())



        # Apply fish discarding
        data = filter_discarded_fish_ids(data, source=path.stem)


        print(data["Fish"].nunique())



        
        cond_actual = data["Exp."].unique()[0]
        print(f"  Processing {cond_actual}: {len(data['Fish'].unique())} fish")

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
            print(f"  [WARN] Discarding {mask_us_beg.sum()} trials with US before CS")
            data.loc[mask_us_beg, "Vigor (deg/ms)"] = np.nan

        time_col = "Trial time (s)"
        vigor_col = "Vigor (deg/ms)"

        if csus == "CS":
            baseline_mask = data[time_col].between(-gen_config.baseline_window, 0)
            cr_mask = data[time_col].between(cr_window[0], cr_window[1])
        elif csus == "US":
            baseline_mask = data[time_col].between(-gen_config.baseline_window - cr_window[1], -cr_window[1])
            cr_mask = data[time_col].between(cr_window[0] - cr_window[1], 0)
        else:
            raise ValueError(f"Unknown csus value: {csus!r}")

        # Window means (baseline and CR)
        trials_bef_onset = data.loc[baseline_mask, :].groupby(columns_groupby, observed=True)[vigor_col].agg("mean")
        trials_aft_onset = data.loc[cr_mask, :].groupby(columns_groupby, observed=True)[vigor_col].agg("mean")

        # Per-trial non-NaN fractions in each window (used to invalidate trial metrics)
        baseline_non_nan_frac = (
            data.loc[baseline_mask, columns_groupby + [vigor_col]]
            .assign(__notna=lambda d: d[vigor_col].notna())
            .groupby(columns_groupby, observed=True)["__notna"]
            .mean()
            .rename("__baseline_non_nan_frac")
        )
        cr_non_nan_frac = (
            data.loc[cr_mask, columns_groupby + [vigor_col]]
            .assign(__notna=lambda d: d[vigor_col].notna())
            .groupby(columns_groupby, observed=True)["__notna"]
            .mean()
            .rename("__cr_non_nan_frac")
        )

        data_agg = pd.concat(
            [trials_bef_onset, trials_aft_onset, trials_aft_onset / trials_bef_onset],
            axis=1,
            keys=column_names,
        ).reset_index()

        # Invalidate trial metrics if either window has too many NaNs (nan_frac > max allowed).
        # Missing fractions (no samples in the window) are treated as 1.0 (i.e., invalidated by this rule).
        data_agg = data_agg.merge(baseline_non_nan_frac.reset_index(), on=columns_groupby, how="left")
        data_agg = data_agg.merge(cr_non_nan_frac.reset_index(), on=columns_groupby, how="left")
        if APPLY_MAX_NAN_FRAC_PER_WINDOW:
            baseline_nan_frac = 1.0 - data_agg["__baseline_non_nan_frac"].fillna(0.0)
            cr_nan_frac = 1.0 - data_agg["__cr_non_nan_frac"].fillna(0.0)
            invalid = (
                baseline_nan_frac.gt(MAX_NAN_FRAC_PER_WINDOW)
                | cr_nan_frac.gt(MAX_NAN_FRAC_PER_WINDOW)
            )
            if invalid.any():
                data_agg.loc[invalid, column_names] = np.nan
                print(
                    f"  Invalidated {int(invalid.sum())}/{len(invalid)} trials "
                    f"(NaN frac > {MAX_NAN_FRAC_PER_WINDOW:.2f})"
                )
        data_agg.drop(columns=["__baseline_non_nan_frac", "__cr_non_nan_frac"], inplace=True, errors="ignore")

        data_agg["Trial type"] = csus
        data_agg = analysis_utils.identify_blocks_trials(data_agg, blocks_dict)

        cat_cols = ["Exp.", "ProtocolRig", "Age (dpf)", "Day", "Fish no.", "Strain", "Fish"]
        existing_cat_cols = [c for c in cat_cols if c in data_agg.columns]
        data_agg[existing_cat_cols] = data_agg[existing_cat_cols].astype("category")

        if data_agg.empty:
            print(f"  [WARN] Empty aggregated data for {cond}, skipping")
            continue

        data_plot_list.append(data_agg)

    if data_plot_list:
        data_pooled = pd.concat(data_plot_list)
        fish_suffix = "_selectedFish" if APPLY_FISH_DISCARD else "_allFish"
        nan_suffix = "_nanFracFilt" if APPLY_MAX_NAN_FRAC_PER_WINDOW else ""
        output_filename = (
            f"NV per trial per fish_CR window {cr_window} s, clean_{cond_types}_{csus}{fish_suffix}{nan_suffix}.pkl"
        )
        data_pooled.to_pickle(path_pooled_data / output_filename, compression="gzip")
        print(f"  Saved: {output_filename}")
    else:
        print("  [WARN] No data processed")

    return data_pooled

# endregion run_data_aggregation


# %%
# region block_summary_lines


#todo why some fish do not have data across all blocks?




def run_block_summary_lines():
    """Plot per-fish block medians with a median overlay per condition.

    Format: one column per condition, scatter+line per fish, bold median line,
    baseline at 1.0, optional block dividers for CS, and shared y-limits.
    """
    ensure_context()
    data, block_cfg = load_block_plot_data()

    if data.empty:
        print("  [SKIP] No pooled data file found")
        return

    if data.empty:
        print("  [SKIP] All fish discarded")
        return

    plot_cfg = plotting_style.get_plot_config()
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
        1, n_cols, figsize=(5 / 2.54, 4 / 2.54), sharex=True, sharey=True, **FIGURE_KW
    )
    # (fig_width, BLOCK_FIG_HEIGHT)
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
            highlight_ids = HIGHLIGHT_FISH_ID if isinstance(HIGHLIGHT_FISH_ID, list) else [HIGHLIGHT_FISH_ID]
            data_fish = data_agg_block[data_agg_block["Fish"].isin(highlight_ids)]
            data_fish = data_fish[data_fish["Exp."] == cond]
            if not data_fish.empty:
                sns.lineplot(
                    data=data_fish,
                    x="Block name",
                    y="Normalized vigor",
                    units="Fish",
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
            blocks_present = [b for b in blocks_chosen if b in data_cond["Block name"].dropna().unique().tolist()]
            if len(blocks_present) < 2:
                print(f"  [SKIP] Not enough blocks for within-condition stats in {cond}.")
            else:
                pairs_within_cond = [
                    (blocks_present[b_i], blocks_present[b_i + 1]) for b_i in range(len(blocks_present) - 1)
                ]
                pairs_within_cond = filter_pairs_in_data_no_hue(pairs_within_cond, data_cond, "Block name")
                if not pairs_within_cond:
                    print(f"  [SKIP] No valid within-condition pairs for stats in {cond}.")
                else:
                    plot_params_b = {"data": data_cond, "x": "Block name", "y": "Normalized vigor", "order": blocks_present}
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


    # Cross-condition statistical tests (Mann-Whitney U since fish are different across conditions)
    if STATS and EXPERIMENT != ExperimentType.MOVING_CS_4COND.value and len(cond_types) > 1:
        print("  --- Cross-condition Mann-Whitney U tests ---")
        
        # Build pairs of conditions to compare
        cross_cond_pairs = []
        if len(cond_types) == 2:
            cross_cond_pairs = [(cond_types[0], cond_types[1])]
        elif len(cond_types) == 3:
            cross_cond_pairs = [(cond_types[0], cond_types[1]), (cond_types[0], cond_types[2])]
        elif len(cond_types) >= 4:
            cross_cond_pairs = [(cond_types[i], cond_types[i + 1]) for i in range(0, len(cond_types) - 1, 2)]
        
        # Collect p-values for multiple comparison correction
        all_pvals = []
        all_comparisons = []
        
        for block in blocks_chosen:
            for cond1, cond2 in cross_cond_pairs:
                data_cond1 = data_agg_block[
                    (data_agg_block["Exp."] == cond1) & (data_agg_block["Block name"] == block)
                ]["Normalized vigor"].dropna()
                data_cond2 = data_agg_block[
                    (data_agg_block["Exp."] == cond2) & (data_agg_block["Block name"] == block)
                ]["Normalized vigor"].dropna()
                
                if len(data_cond1) > 0 and len(data_cond2) > 0:
                    stat, pval = mannwhitneyu(data_cond1, data_cond2, alternative="two-sided")
                    all_pvals.append(pval)
                    all_comparisons.append((block, cond1, cond2, stat, pval))
        
        # Apply Holm-Bonferroni correction
        if all_pvals:
            reject, pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.05, method="holm")
            
            print(f"  {'Block':<20} {'Comparison':<25} {'Stat':>8} {'p-raw':>10} {'p-corr':>10} {'Sig':>5}")
            print(f"  {'-' * 85}")
            for i, (block, cond1, cond2, stat, pval) in enumerate(all_comparisons):
                if pvals_corrected[i] < 0.0001:
                    sig_marker = "****"
                elif pvals_corrected[i] < 0.001:
                    sig_marker = "***"
                elif pvals_corrected[i] < 0.01:
                    sig_marker = "**"
                elif pvals_corrected[i] < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = "ns" if not Hide_non_significant else ""
                
                print(f"  {block:<20} {cond1} vs {cond2:<10} {stat:>8.1f} {pval:>10.4e} {pvals_corrected[i]:>10.4e} {sig_marker:>5}")
            
            # Add visual annotation for significant cross-condition comparisons
            block_annotation_offset = {}
            annotation_height_step = 0.04
            
            for i, (block, cond1, cond2, stat, pval) in enumerate(all_comparisons):
                if not reject[i] and Hide_non_significant:
                    continue
                
                if pvals_corrected[i] < 0.0001:
                    sig_text = "****"
                elif pvals_corrected[i] < 0.001:
                    sig_text = "***"
                elif pvals_corrected[i] < 0.01:
                    sig_text = "**"
                elif pvals_corrected[i] < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"
                
                try:
                    ax_idx1 = cond_types.index(cond1)
                    ax_idx2 = cond_types.index(cond2)
                except ValueError:
                    continue
                
                block_idx = blocks_chosen.index(block)
                y_range = y_lim[1] - y_lim[0]
                base_y = y_lim[1] + 0.02 * y_range
                
                block_key = block_idx
                if block_key not in block_annotation_offset:
                    block_annotation_offset[block_key] = 0
                
                y_pos = base_y + block_annotation_offset[block_key] * annotation_height_step * y_range
                block_annotation_offset[block_key] += 1
                
                ax_b[ax_idx1].plot([block_idx, block_idx], [y_lim[1], y_pos], 
                                   color='k', lw=0.5, clip_on=False)
                ax_b[ax_idx2].plot([block_idx, block_idx], [y_lim[1], y_pos], 
                                   color='k', lw=0.5, clip_on=False)
                
                con = ConnectionPatch(
                    xyA=(block_idx, y_pos), coordsA=ax_b[ax_idx1].transData,
                    xyB=(block_idx, y_pos), coordsB=ax_b[ax_idx2].transData,
                    color='k', lw=0.5, clip_on=False
                )
                fig_b.add_artist(con)
                
                fig_b.canvas.draw()
                pos1 = ax_b[ax_idx1].transData.transform((block_idx, y_pos))
                pos2 = ax_b[ax_idx2].transData.transform((block_idx, y_pos))
                mid_fig_x = (pos1[0] + pos2[0]) / 2
                mid_fig_y = pos1[1]
                
                inv_transform = fig_b.transFigure.inverted()
                mid_x, mid_y = inv_transform.transform((mid_fig_x, mid_fig_y))
                
                fig_b.text(mid_x, mid_y + 0.005, sig_text, ha='center', va='bottom', fontsize=8, fontweight='normal', transform=fig_b.transFigure)


    for e_i, cond in enumerate(cond_types):
        ax_b[e_i].set_xticks(list(np.arange(len(blocks_chosen))))
        ax_b[e_i].set_xticklabels(blocks_chosen_labels, rotation=0, ha="center", fontweight="bold", fontsize=10)
        ax_b[e_i].tick_params(axis="x", bottom=False)
        ax_b[e_i].grid(False)
        ax_b[e_i].spines["bottom"].set_visible(False)
        if e_i != 0:
            ax_b[e_i].set_ylabel(None, fontweight="bold")
            ax_b[e_i].spines["left"].set_visible(False)
            ax_b[e_i].set_yticks([])
        ax_b[0].set_ylabel("Normalized vigor (AU)", fontweight="bold")
        ax_b[0].spines["left"].set_visible(True)
        ax_b[0].spines["left"].set_position(("outward", 3))
        apply_y_limits(ax_b[0], y_lim)
        if e_i != 0:
            ax_b[e_i].set_ylim(y_lim)
        analysis_utils.add_component(
            ax_b[e_i],
            analysis_utils.AddTextSpec(
                component="axis_title",
                text=cond_titles[e_i],
                anchor_h="right",
                anchor_v="top",
                pad_pt=(0, 0),
                text_kwargs={
                    "fontsize": plot_cfg.figure_titlesize,
                    "backgroundcolor": "none",
                    "color": "k",
                },
            ),
        )
        ax_b[e_i].set_xlabel("")
    for i in range(len(cond_types), len(ax_b)):
        fig_b.delaxes(ax_b[i])

    apply_panel_label(fig_b, "A")
    cond_label = _stringify_for_filename(cond_types)
    age_label = _stringify_for_filename(age_filter)
    setup_label = _stringify_for_filename(setup_color_filter)
    stem = f"SV lineplot single catch trials_{block_cfg['number_trials_block']}_{age_label}_{setup_label}_{cond_label}"
    save_path = save_fig(fig_b, stem, frmt)
    print(f"  Saved: {save_path.name}")

# endregion run_block_summary_lines


# %%
# region block_summary_boxplot
def run_block_summary_boxplot():
    """Plot block-wise boxplots of per-fish medians with stats overlays.

    Format: single axis, grouped by block and condition (hue), baseline at 1.0,
    custom outliers per box, annotated between/within-condition stats, and an
    outside legend with fish counts.
    """
    ensure_context()
    data_box, block_cfg = load_block_plot_data()

    if data_box.empty:
        print("  [SKIP] No pooled data file found")
        return

    blocks_chosen = block_cfg["blocks_chosen"]
    blocks_chosen_labels = block_cfg["blocks_chosen_labels"]
    number_trials_block = block_cfg["number_trials_block"]

    data_box = data_box.copy()
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
        print("  [SKIP] No matching conditions found")
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
    palette_box = [color_palette[cond_types.index(cond)] for cond in cond_types_box]

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
        palette=palette_box,
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
                color=palette_box[i],
                marker=".",
                alpha=0.8,
                edgecolor="k",
                linewidth=0.5,
                clip_on=False,
                s=8,
            )

    if STATS:
        print("  --- Mann-Whitney U test (boxplot) ---")
        blocks_for_stats = [b for b in blocks_chosen if b in data_box["Block name"].dropna().unique().tolist()]
        if len(blocks_for_stats) < 1:
            print("  [SKIP] No blocks available for stats after filtering.")
            blocks_for_stats = []
        hue_plot_params = {
            "data": data_box,
            "x": "Block name",
            "y": "Normalized vigor",
            "order": blocks_for_stats if blocks_for_stats else blocks_chosen,
            "hue": "Exp.",
            "hue_order": list(cond_types_box),
            "palette": palette_box,
        }

        pairs_between_exp = []

        def add_pairs(cond_pairs):
            for bl in (blocks_for_stats if blocks_for_stats else blocks_chosen):
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

        pairs_between_exp = filter_pairs_in_data_with_hue(pairs_between_exp, data_box, "Block name", "Exp.")
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
        else:
            print("  [SKIP] No valid between-condition pairs found in data.")

        if EXPERIMENT != ExperimentType.MOVING_CS_4COND.value:
            print("  --- Wilcoxon test (boxplot) ---")
            if blocks_for_stats and len(blocks_for_stats) != len(blocks_chosen):
                print("  [SKIP] Wilcoxon requires complete block coverage; missing blocks after filtering.")
            else:
                # Filter data to only include fish with complete data in all blocks
                # This is required for paired Wilcoxon test
                fish_block_counts = data_box.groupby(['Fish', 'Exp.'], observed=True)['Block name'].nunique()
                complete_fish = fish_block_counts[fish_block_counts == len(blocks_chosen)].reset_index()[['Fish', 'Exp.']]
                data_box_complete = data_box.merge(complete_fish, on=['Fish', 'Exp.'], how='inner')

                pairs_within_exp = []
                cond_types_complete = [cond for cond in cond_types_box if cond in data_box_complete["Exp."].unique()]
                if not cond_types_complete:
                    print("  [SKIP] No conditions with complete data for Wilcoxon.")
                else:
                    pairs_within_exp = build_within_condition_pairs(blocks_chosen, cond_types_complete)
                    pairs_within_exp = filter_pairs_in_data_with_hue(
                        pairs_within_exp, data_box_complete, "Block name", "Exp."
                    )
                
                hue_plot_params_complete = hue_plot_params.copy()
                hue_plot_params_complete['data'] = data_box_complete
                hue_plot_params_complete["hue_order"] = list(cond_types_complete)
                hue_plot_params_complete["palette"] = [color_palette[cond_types.index(c)] for c in cond_types_complete]
                hue_plot_params_complete["order"] = blocks_chosen
                
                if pairs_within_exp:
                    ax_a.set_ylim((y_lim[0], y_lim[1] + 0.1))
                    annotator = Annotator(ax_a, pairs_within_exp, **hue_plot_params_complete)
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
                else:
                    print("  [SKIP] No valid within-condition (Wilcoxon) pairs found in data.")

    ax_a.spines["bottom"].set_visible(False)
    ax_a.grid(False)
    ax_a.set_xticks(list(np.arange(len(blocks_chosen))))
    ax_a.set_xticklabels(blocks_chosen_labels, rotation=0, ha="center", fontweight="bold", fontsize=10)
    ax_a.set_ylabel("Normalized vigor (AU)")
    apply_y_limits(ax_a, y_lim, labels=[f"{y_lim[0]}", "1.0", f"{y_lim[1]}"])
    ax_a.locator_params(axis="y", tight=False, nbins=4)
    ax_a.tick_params(axis="both", which="both", bottom=False, top=False, right=False)
    ax_a.set_xlabel("")

    handles, _ = ax_a.get_legend_handles_labels()
    ax_a.legend().remove()
    add_outside_legend(fig_a, handles[:len(cond_types_box)], cond_titles_box)
    apply_panel_label(fig_a, "D")

    if EXPERIMENT == ExperimentType.MOVING_CS_4COND.value:
        fig_a.set_size_inches(5/2.54, 5/2.54)

    cond_label = _stringify_for_filename(cond_types_box)
    age_label = _stringify_for_filename(age_filter)
    setup_label = _stringify_for_filename(setup_color_filter)
    stem = f"3.1_{number_trials_block}_{age_label}_{setup_label}_{cond_label}"
    save_path = save_fig(fig_a, stem, frmt)
    print(f"  Saved: {save_path.name}")

# endregion run_block_summary_boxplot


# %%
# region phase_summary
def run_phase_summary():
    """Plot phase-level medians per condition with per-fish trajectories.

    Format: one column per condition, per-fish scatter+lines across blocks,
    optional highlighted fish, baseline at 1.0, and phase divider lines.
    """
    ensure_context()
    data = load_phase_plot_data()

    if data.empty:
        print("  [SKIP] No pooled data file found")
        return

    if data.empty:
        print("  [SKIP] All fish discarded")
        return

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
            highlight_ids = HIGHLIGHT_FISH_ID if isinstance(HIGHLIGHT_FISH_ID, list) else [HIGHLIGHT_FISH_ID]
            data_highlight = data_median[cond_i][data_median[cond_i]["Fish"].isin(highlight_ids)]
            if not data_highlight.empty:
                sns.lineplot(
                    data=data_highlight,
                    x="Block name",
                    y="nanmedian",
                    units="Fish",
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
        apply_y_limits(ax[cond_i], y_lim, labels=[f"{y_lim[0]}", "1", f"{y_lim[1]}"])

        plot_cfg = plotting_style.get_plot_config()
        analysis_utils.add_component(
            ax[cond_i],
            analysis_utils.AddTextSpec(
                component="axis_title",
                text=cond_titles[cond_i],
                anchor_h="right",
                anchor_v="top",
                pad_pt=(0, 0),
                text_kwargs={
                    "fontsize": plot_cfg.figure_titlesize,
                    "backgroundcolor": "none",
                    "color": "k",
                },
            ),
        )
    for i in range(len(cond_types), len(ax)):
        fig.delaxes(ax[i])

    apply_panel_label(fig, "B")
    cond_label = _stringify_for_filename(cond_types)
    stem = f"NV single fish median of trials_{cond_label}_{csus}"
    save_path = save_fig(fig, stem, frmt)
    print(f"  Saved: {save_path.name}")

# endregion run_phase_summary


# %%
# region LME
# region trial_by_trial
def run_trial_by_trial(data_pooled=None):
    """Plot trial-by-trial median trajectories with LME significance annotations.

    Format: single axis, median line with CI per condition, optional annotation
    lanes for global/block/trial significance, and block boundary markers.
    """
    ensure_context()
    if data_pooled is None or data_pooled.empty:
        data_plot = load_latest_pooled()
    else:
        data_plot = data_pooled.copy()

    if data_plot.empty:
        print("  [SKIP] No pooled data file found")
        return

    # Print initial fish count
    before_any_discard = data_plot["Fish"].nunique()
    print(f"  [run_trial_by_trial] Fish unique before discard: {before_any_discard}")

    # Apply shared discard list (from Discarded_fish_IDs.txt)
    data_plot = filter_discarded_fish_ids(data_plot, source="run_trial_by_trial")

    print(f"  Fish remaining after discard: {data_plot['Fish'].nunique()}")

    if EXPORT_TEXT:
        output_path = path_pooled_data / "data_output.txt"
        data_plot[["Exp.", "Fish", "Block name", "Trial number", "Normalized vigor"]].to_csv(
            output_path, sep="\t", index=False
        )

    try:
        df_main = prepare_main_df(
            data_plot,
            apply_fish_discard=APPLY_FISH_DISCARD,
            jitter_scale=LME_JITTER_SCALE,
            jitter_seed=LME_JITTER_SEED,
        )
    except ValueError as exc:
        print(f"  [SKIP] {exc}")
        return

    if df_main.empty:
        print("  [SKIP] Dataframe is empty")
        return

    # Setup & Metadata
    # Define reference condition and extract block structure.
    print(f"  Data prepared: {len(df_main)} rows, {df_main['Fish_ID'].nunique()} subjects")
    ref_cond = cond_types[0]
    block_order = block_order_from_data(df_main)
    block_centers = {
        b: df_main[df_main["Block_name"] == b]["Trial number"].mean()
        for b in block_order
    }
    block_boundaries = block_boundaries_from_data(df_main, block_order)


    print(df_main['Fish_ID'].nunique())

    # return

    # Containers for statistical results
    global_interactions = None
    ph_df = pd.DataFrame()
    sig_trials = []
    model_errors = []

    if RUN_LME and STATS:
        # ---------------------------------------------------------------------
        # Statistical Analysis (LME)
        # ---------------------------------------------------------------------
        # This section performs three levels of inference using Linear Mixed
        # Effects (LME) models (statsmodels MixedLM):
        #
        #  1) Global ANCOVA-style model across ALL blocks:
        #       Tests whether the relationship between Condition and response
        #       differs across blocks (Condition × Block interaction), while
        #       controlling for baseline vigor (Log_Baseline).
        #
        #  2) Post-hoc per-block models:
        #       For each block separately, tests:
        #         (a) Mean difference between conditions at the block "center"
        #         (b) Learning-rate (slope) difference within that block
        #
        #  3) Trial-by-trial models:
        #       Fits a separate model for each trial number to localize
        #       condition differences at specific trials (with FDR correction).
        #
        # Notes on variables:
        #   - Log_Response = log(response + 1)  (stabilizes variance / reduces skew)
        #   - Log_Baseline = log(baseline + 1)  (covariate; ANCOVA adjustment)
        #
        # Notes on mixed effects:
        #   - Fish_ID is used as the grouping variable to account for repeated
        #     measurements within fish (correlated observations).
        #   - re_formula controls random effects structure (here trying to allow
        #     fish-specific baseline-related deviations).
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # 1) Global ANCOVA: Condition × Block interaction
        # ---------------------------------------------------------------------
        # Goal:
        #   Test whether the effect of Condition depends on Block, after adjusting
        #   for baseline vigor. This is a global "is there any interaction?"
        #   question rather than a block-localized question.
        #
        # Model interpretation (fixed effects):
        #   Log_Response ~ Log_Baseline + Condition * Block
        #
        #   - Log_Baseline: covariate adjustment (ANCOVA)
        #   - Condition: differences vs reference condition (ref_cond)
        #   - Block_name: differences vs reference block level (statsmodels chooses)
        #   - Condition:Block_name: interaction terms indicating that the condition
        #     effect differs by block (what we primarily want here)
        print("  --- LME: Global ANCOVA ---")
        print(f"  Using re_formula: {LME_RE_FORMULA}, method: {LME_GLOBAL_METHOD}")


#! why not? f_local = "Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}')) * C(Trial_Number)"
        f_global = f"Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}')) * C(Block_name)"
        res_global, err = run_mixed_model(df_main, f_global, "Fish_ID", re_formula=LME_RE_FORMULA, method=LME_GLOBAL_METHOD)

        if res_global:
            # Print full summary for transparency (useful for debugging/reporting).
            print(res_global.summary())

            # Extract interaction terms from the fitted model directly.
            # Using the summary tables here is brittle (they are often SimpleTable, not pandas).
            pvals = res_global.pvalues
            interaction_terms = [t for t in pvals.index if ":" in t]
            if interaction_terms:
                global_interactions = pd.DataFrame(
                    {"P>|z|": pd.to_numeric(pvals.loc[interaction_terms], errors="coerce")},
                    index=interaction_terms,
                )
            else:
                global_interactions = pd.DataFrame(columns=["P>|z|"])
        else:
            # Record model fit failures rather than crashing the pipeline.
            model_errors.append({"Type": "Global", "Unit": "All", "Error": err})

        # ---------------------------------------------------------------------
        # 2) Post-Hoc Block Analysis: per-block mean + slope differences
        # ---------------------------------------------------------------------
        # Goal:
        #   Localize effects within blocks. Even if the global interaction suggests
        #   differences somewhere, this step estimates:
        #     - Mean offset at block center (Condition main effect)
        #     - Difference in within-block learning rate (Condition × Trial slope)
        #
        # Why "Trial_Centered"?
        #   We subtract the mean trial number within each block:
        #       Trial_Centered = Trial - mean(Trial in block)
        #   This makes the Condition main-effect interpretable as the difference
        #   at the middle of the block, and reduces collinearity between main and
        #   interaction terms.
        print("  --- LME: Post-Hoc Block Analysis ---")
        posthoc_res = []

        for block in block_order:
            # Subset to a single block. This isolates within-block trajectories.
            df_blk = df_main[df_main["Block_name"] == block].copy()

            # If there is only one condition represented in this block subset,
            # there is no between-condition comparison to be made.
            if df_blk["Condition"].nunique() < 2:
                continue

            # Center trial number within the block for interpretability and stability.
            df_blk["Trial_Centered"] = df_blk["Trial number"] - df_blk["Trial number"].mean()

            # Local (per-block) model:
            #   - Condition term: mean difference at Trial_Centered == 0 (block center)
            #   - Trial_Centered term: slope in the reference condition
            #   - Condition:Trial_Centered: slope difference vs reference condition
            f_local = f"Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}')) * Trial_Centered"

            # Fit mixed model within the block.
            # If convergence/singularity occurs, run_mixed_model returns (None, error).
            res_local, err = run_mixed_model(
                df_blk,
                f_local,
                "Fish_ID",
                re_formula=LME_RE_FORMULA,
                method=LME_LOCAL_METHOD,
                reml=False,
            )

            if res_local:
                params = res_local.params
                pvals = res_local.pvalues

                # Identify fixed-effect terms for condition mean differences:
                #   e.g. "C(Condition, Treatment('ref'))[T.OTHER]"
                # and for slope differences:
                #   e.g. "C(Condition,...)[T.OTHER]:Trial_Centered"
                main_terms = [t for t in params.index if "Condition" in t and ":" not in t]
                slope_terms = [t for t in params.index if "Condition" in t and ":" in t]

                # Print slope output results (per block, per term) with RAW p-values
                if slope_terms:
                    print(f"    [Block {block}] Slope (Condition × Trial_Centered) terms (raw p):")
                    for t in slope_terms:
                        print(f"      {t}: coef={params[t]: .6f}, p_slope_raw={pvals[t]: .4e}")
                else:
                    print(f"    [Block {block}] No slope terms found.")

                if main_terms and slope_terms:

                    def _extract_level(term: str) -> str:
                        m = re.search(r"\[T\.(.+?)\]", term)
                        return m.group(1) if m else term

                    main_terms_sorted = sorted(main_terms)
                    slope_terms_sorted = sorted(slope_terms)

                    best_main = min(main_terms_sorted, key=lambda t: float(pvals.get(t, 1.0)))
                    best_slope = min(slope_terms_sorted, key=lambda t: float(pvals.get(t, 1.0)))

                    # Explicitly print the RAW p-value used for P_Slope in posthoc_res
                    print(
                        f"    [Block {block}] Selected P_Slope raw: "
                        f"{best_slope} -> {float(pvals[best_slope]):.4e}"
                    )

                    posthoc_res.append(
                        {
                            "Block": block,
                            "Term_Main": best_main,
                            "CondLevel_Main": _extract_level(best_main),
                            "Coef_Mean": params[best_main],
                            "P_Mean": pvals[best_main],
                            "Term_Slope": best_slope,
                            "CondLevel_Slope": _extract_level(best_slope),
                            "Coef_Slope": params[best_slope],
                            "P_Slope": pvals[best_slope],  # raw p-value
                        }
                    )
                else:
                    model_errors.append(
                        {
                            "Type": "Block-TermsMissing",
                            "Unit": block,
                            "Error": "Missing condition mean/slope terms",
                        }
                    )
            else:
                model_errors.append({"Type": "Block-Fit", "Unit": block, "Error": err})

        # Multiple-comparisons correction across blocks:
        #   We correct mean tests and slope tests separately using Benjamini-Hochberg FDR.
        if posthoc_res:
            ph_df = pd.DataFrame(posthoc_res)

            _, ph_df["P_Mean_FDR"], _, _ = multipletests(ph_df["P_Mean"], alpha=0.05, method="fdr_bh")
            _, ph_df["P_Slope_FDR"], _, _ = multipletests(ph_df["P_Slope"], alpha=0.05, method="fdr_bh")

            # These boolean flags drive plot markers.
            ph_df["Sig_Mean"] = ph_df["P_Mean_FDR"] < 0.05
            # Slope significance shown both before and after FDR correction
            ph_df["Sig_Slope_Raw"] = ph_df["P_Slope"] < 0.05
            ph_df["Sig_Slope"] = ph_df["P_Slope_FDR"] < 0.05

            if LME_DEBUG_POSTHOC_TABLE:
                cols = [
                    "Block",
                    "CondLevel_Main",
                    "Coef_Mean",
                    "P_Mean",
                    "P_Mean_FDR",
                    "Sig_Mean",
                    "CondLevel_Slope",
                    "Coef_Slope",
                    "P_Slope",
                    "P_Slope_FDR",
                    "Sig_Slope",
                ]
                cols = [c for c in cols if c in ph_df.columns]
                print("  --- LME: Post-Hoc Block Summary (raw + FDR) ---")
                print(ph_df[cols].to_string(index=False))

        # ---------------------------------------------------------------------
        # 3) Trial-by-Trial Analysis: per-trial condition differences
        # ---------------------------------------------------------------------
        # Goal:
        #   Identify specific trials where conditions differ (after baseline adjustment).
        #
        # Approach:
        #   For each trial number, fit:
        #       Log_Response ~ Log_Baseline + Condition
        #   and extract the p-value of the Condition term (vs reference).
        #
        # Then apply FDR correction across all trials (multiple testing problem).
        print("  --- LME: Trial-by-Trial Analysis ---")
        trial_res = []

        for trial in sorted(df_main["Trial number"].unique()):
            df_t = df_main[df_main["Trial number"] == trial].copy()

            # If only one condition appears at this trial, cannot compare conditions.
            if df_t["Condition"].nunique() < 2:
                continue

            f_trial = f"Log_Response ~ Log_Baseline + C(Condition, Treatment('{ref_cond}'))"
            res_t, err = run_mixed_model(df_t, f_trial, "Fish_ID", method=LME_LOCAL_METHOD)

            if res_t:
                # Pick the condition coefficient term (first non-reference).
                # With >2 conditions, there may be multiple terms; current logic uses the first.
                term = [x for x in res_t.params.index if "Condition" in x][0]
                trial_res.append({"Trial": trial, "P_raw": res_t.pvalues[term]})
            else:
                # Optional: record failures per trial (not required for plotting).
                model_errors.append({"Type": "Trial-Fit", "Unit": trial, "Error": err})

        if trial_res:
            t_df = pd.DataFrame(trial_res)

            # FDR correction across all tested trials.
            reject, _, _, _ = multipletests(t_df["P_raw"], alpha=0.05, method="fdr_bh")

            # Trials marked True in `reject` are significant after correction.
            sig_trials = t_df[reject]["Trial"].tolist()

        # Report any warnings/errors encountered during model fitting.
        if model_errors:
            print("  [WARN] LME fit warnings:")
            for err in model_errors:
                print(f"    {err['Type']} ({err['Unit']}): {err['Error']}")

    # Plotting
    fig_c, ax_c = plt.subplots(1, 1, facecolor="white", figsize=TRIAL_BY_TRIAL_FIGSIZE, layout="tight")

    # Main trajectories: median normalized vigor with bootstrap CI.
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

    # Statistical annotations: lanes of markers above the plot.
    # Gold = global interaction, Silver = block effects, Black = trial significance.
    y_top = y_lim[1]
    y_gold = y_top + 0.25
    y_silver = y_top + 0.15
    y_silver_raw = y_silver - 0.05
    y_silver_fdr = y_silver - 0.10
    # y_black = y_lim_plot[1] + 0.05

    # Global interactions (gold markers).
    if global_interactions is not None:
        for term, row in global_interactions.iterrows():
            if row["P>|z|"] < 0.05:
                match = re.search(r"C\(Block_name\)\[T\.(.+?)\]", term)
                if match and match.group(1) in block_centers:
                    ax_c.text(
                        block_centers[match.group(1)],
                        y_gold,
                        "D",
                        # match.group(1),
                        color="gold",
                        ha="center",
                        fontsize=7,
                        fontweight="bold",
                        clip_on=False,
                    )

    # Post-hoc block effects (silver/firebrick markers).
    if not ph_df.empty:
        for _, row in ph_df.iterrows():
            blk = row["Block"]
            if blk in block_centers:
                if row["Sig_Mean"]:
                    ax_c.text(
                        block_centers[blk],
                        y_silver,
                        "M",
                        color="gray",
                        ha="center",
                        fontsize=6,
                        clip_on=False,
                    )
                if row.get("Sig_Slope_Raw", False):
                    ax_c.text(
                        block_centers[blk],
                        y_silver_raw,
                        "R (raw)",
                        color="firebrick",
                        ha="center",
                        fontsize=6,
                        clip_on=False,
                    )
                if row["Sig_Slope"]:
                    ax_c.text(
                        block_centers[blk],
                        y_silver_fdr,
                        "R (FDR)",
                        color="firebrick",
                        ha="center",
                        fontsize=6,
                        fontweight="bold",
                        clip_on=False,
                    )

    # Trial-by-trial significance (black stars).
    y_black = y_top + 0.01
    if sig_trials:
        ax_c.scatter(
            sig_trials,
            [y_black] * len(sig_trials),
            color="black",
            marker="*",
            s=10,
            zorder=10,
            linewidths=0,
            label="p < 0.05 (FDR)",
            clip_on=False,
        )

    # Add vertical lines for block boundaries.
    for boundary in block_boundaries:
        ax_c.axvline(boundary, color="gray", alpha=0.5, linewidth=0.5)
        # ax_c.spines["bottom"].set_visible(False)
        # ax_c.tick_params(axis="x", bottom=False)
    ax_c.set_xlim(0, df_main["Trial number"].max() + 1)
    ax_c.set_ylim(y_lim[0], y_lim[1])
    ax_c.set_ylabel("Normalized vigor (AU)")
    ax_c.legend().remove()

    cond_label = _stringify_for_filename(cond_types)
    fish_suffix = "_selectedFish" if APPLY_FISH_DISCARD else "_allFish"
    save_path = save_fig(fig_c, f"NV_LME_Visualized_Raw_{cond_label}" + f"_{csus}_{fish_suffix}", frmt)
    print(f"  Saved: {save_path.name}")
# endregion run_trial_by_trial
# endregion LME
# endregion Pipeline Functions

# region Main
# region main
def main():
    initialize_context()
    print(f"{'='*60}")
    print(f"Experiment: {EXPERIMENT}")
    print(f"Conditions: {cond_types}")
    print(f"{'='*60}")

    data_pooled = None
    if RUN_PROCESS:
        print("\n--- RUN: Data Aggregation ---")
        data_pooled = run_data_aggregation()

    if RUN_BLOCK_SUMMARY_LINES:
        print("\n--- RUN: Block Summary (Lines) ---")
        run_block_summary_lines()

    if RUN_BLOCK_SUMMARY_BOXPLOT:
        print("\n--- RUN: Block Summary (Boxplot) ---")
        run_block_summary_boxplot()

    if RUN_PHASE_SUMMARY:
        print("\n--- RUN: Phase Summary ---")
        run_phase_summary()

    if RUN_TRIAL_BY_TRIAL:
        print("\n--- RUN: Trial-by-Trial (LME) ---")
        run_trial_by_trial(data_pooled=data_pooled if RUN_PROCESS else None)

    print("\n--- Done ---")

# endregion main


if __name__ == "__main__":
    main()
# endregion Main

# %%
