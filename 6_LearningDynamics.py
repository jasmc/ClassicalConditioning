# %%
"""
Refactored multivariate LME analysis using shared modules and unified config.
"""

# region Imports & Configuration
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Add the directory containing shared modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

import analysis_utils  # noqa: F401
import file_utils
from experiment_configuration import ExperimentType, get_experiment_config
from general_configuration import config as gen_config  # noqa: F401
import plotting_style

pd.set_option("mode.copy_on_write", True)
# pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

# Set plotting style (shared across analysis scripts)
plotting_style.set_plot_style()
sns.color_palette("colorblind")

EXPERIMENT = ExperimentType.ALL_DELAY.value
config = get_experiment_config(EXPERIMENT)

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
    _path_orig_pkl,
    _path_all_fish,
    path_pooled_data,
) = file_utils.create_folders(config.path_save)

CSUS = "CS"
STATS = True
FRMT = "png"
N_BOOT = 1000
MIN_TRIALS_PER_BLOCK = 6
APPLY_FISH_DISCARD = False

age_filter = ["all"]
setup_color_filter = ["all"]

cond_types = config.cond_types
cond_titles = [config.cond_dict[c]["name"] for c in cond_types]
color_palette = config.color_palette

excluded_dir = _path_orig_pkl / "Excluded new"
fish_ids_to_discard = file_utils.load_excluded_fish_ids(excluded_dir)

y_lim_plot = (0.7, 1.4)

print(f"Experiment: {EXPERIMENT}")
print(f"Conditions: {cond_types}")
# endregion


# %%
# region Helper Functions
def run_mixed_model(df, formula, groups_col, re_formula=None, method="powell"):
    try:
        model = smf.mixedlm(formula, df, groups=df[groups_col], re_formula=re_formula)
        res = model.fit(reml=False, method=method)
        return res, None
    except Exception as exc:
        return None, str(exc)


def load_latest_pooled():
    paths = [*Path(path_pooled_data).glob("*.pkl")]
    paths = [p for p in paths if "NV per trial per fish" in p.stem]
    paths_csus = [
        p for p in paths
        if len(p.stem.split("_")) > 3 and p.stem.split("_")[3] == CSUS
    ]
    if not paths_csus:
        paths_csus = [p for p in paths if p.stem.split("_")[-1] == CSUS]
    paths = paths_csus
    if not paths:
        return pd.DataFrame()
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"Loading: {paths[0].name}")
    return pd.read_pickle(paths[0], compression="gzip")


def filter_pooled_data(data):
    if APPLY_FISH_DISCARD and fish_ids_to_discard:
        data = data[~data["Fish"].isin(fish_ids_to_discard)]
    if age_filter != ["all"]:
        data = data[data["Age (dpf)"].isin(age_filter)]
    if setup_color_filter != ["all"]:
        color_rigs = data["ProtocolRig"].str.split("-").str[0]
        data = data[color_rigs.isin(setup_color_filter)]
    return data


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


def prepare_main_df(data):
    data = filter_pooled_data(data)

    baseline_col, response_col = select_baseline_response_columns(data)

    df = data.rename(columns={"Exp.": "Condition", "Fish": "Fish_ID", "Block name": "Block_name"})
    df.dropna(subset=["Normalized vigor", "Trial number", "Block_name", baseline_col, response_col], inplace=True)

    counts = df.groupby(["Fish_ID", "Block_name"], observed=True).size()
    valid_fish = counts[counts >= MIN_TRIALS_PER_BLOCK].reset_index()["Fish_ID"].unique()
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
# endregion


# %%
# region Data Loading & Preparation
data_pooled = load_latest_pooled()
if data_pooled.empty:
    raise SystemExit("No matching pooled data file found.")

df_main = prepare_main_df(data_pooled)
if df_main.empty:
    raise SystemExit("Dataframe is empty.")

ref_cond = cond_types[0]
block_order = df_main["Block_name"].unique().tolist()
block_centers = {
    b: df_main[df_main["Block_name"] == b]["Trial number"].mean()
    for b in block_order
}
block_boundaries = block_boundaries_from_data(df_main, block_order)

print(f"Data Prepared: {len(df_main)} rows, {df_main['Fish_ID'].nunique()} subjects.")
# endregion


# %%
# region Statistical Analysis (LME)
global_interactions = None
ph_df = pd.DataFrame()
sig_trials = []
model_errors = []

if STATS:
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
# endregion


# %%
# region Visualization
fig_c, ax_c = plt.subplots(1, 1, facecolor="white", figsize=(14 / 2.54, 9 / 2.54), layout="tight")

for i, cond in enumerate(cond_types):
    sns.lineplot(
        data=df_main[df_main["Condition"] == cond],
        x="Trial number",
        y="Normalized vigor plot",
        color=color_palette[i],
        alpha=0.9,
        linewidth=0.5,
        label=cond_titles[i],
        estimator="median",
        errorbar=("ci", 95),
        err_style="band",
        n_boot=N_BOOT,
        ax=ax_c,
    )

y_gold, y_silver, y_black = y_lim_plot[1] + 0.25, y_lim_plot[1] + 0.15, y_lim_plot[1] + 0.05

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
                ax_c.text(block_centers[blk], y_silver, "Mean", color="gray", ha="center", fontsize=6)
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
    ax_c.scatter(sig_trials, [y_black] * len(sig_trials), color="black", marker="*", s=8, zorder=10, linewidths=0)

vline_positions = []
spec_var = None
if CSUS == "CS":
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
save_path = save_root / f"NV_LME_Visualized_Raw_{cond_types}.{FRMT}"
fig_c.savefig(str(save_path), format=FRMT, dpi=600, bbox_inches="tight")
print(f"Figure saved: {save_path.name}")
# endregion
