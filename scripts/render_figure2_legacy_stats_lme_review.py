"""Figure 2D legacy-style stars and Figure 2G authenticated LME review.

D repeats the legacy test families on corrected, approved-cohort fish ratios:
Holm-corrected between-condition Mann-Whitney and paired within-condition
Wilcoxon tests. G displays the existing total-activity LME only for its fitted
metric, including its global and block results and the simultaneous trial band.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from scipy.stats import norm

from classical_conditioning.analysis.inference.learning_onset import load_learning_onset_analysis
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import METRIC_DISPLAY_NAMES
from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
from classical_conditioning.figures.theme import apply_theme, mm_to_in, style_axes


BLOCK_LABELS = ("Pre-train", "Early Test", "Late Test")
COLORS = {"control": "#00AEEF", "delay": "#EC008C"}


def stars(p: float) -> str:
    if not np.isfinite(p) or p >= .05:
        return "ns"
    return "****" if p < 1e-4 else "***" if p < .001 else "**" if p < .01 else "*"


def legacy_block_tests(fish: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = fish.loc[fish["Eligible"]].pivot(index=["fish_id", "condition_id"],
        columns="Selected block order", values="Fish median response / baseline").reindex(columns=[0, 1, 2])
    if wide.isna().any().any() or len(wide) != 57:
        raise ValueError("Complete three-block values are required for all 57 fish")
    records = []
    for block in range(3):
        control = wide.xs("control", level="condition_id")[block]
        delay = wide.xs("delay", level="condition_id")[block]
        records.append({"family": "between conditions", "condition": "delay vs control",
            "contrast": BLOCK_LABELS[block], "block_left": block, "block_right": block,
            "test": "Mann-Whitney U", "p_raw": float(mannwhitneyu(control, delay).pvalue),
            "n_control": len(control), "n_delay": len(delay)})
    for condition in ("control", "delay"):
        selected = wide.xs(condition, level="condition_id")
        for left, right in ((0, 1), (1, 2)):
            records.append({"family": "within condition", "condition": condition,
                "contrast": f"{BLOCK_LABELS[left]} vs {BLOCK_LABELS[right]}",
                "block_left": left, "block_right": right,
                "test": "paired Wilcoxon", "p_raw": float(wilcoxon(selected[left], selected[right]).pvalue),
                "n_control": len(selected) if condition == "control" else np.nan,
                "n_delay": len(selected) if condition == "delay" else np.nan})
    table = pd.DataFrame.from_records(records)
    for family, indices in table.groupby("family", sort=False).groups.items():
        table.loc[indices, "p_holm"] = multipletests(table.loc[indices, "p_raw"], method="holm")[1]
    table["stars"] = table["p_holm"].map(stars)
    return wide, table


def render_d(wide: pd.DataFrame, table: pd.DataFrame, metric_id: str):
    theme = apply_theme()
    figure, axes = plt.subplots(1, 2, figsize=mm_to_in(183, 112), sharey=True,
                                constrained_layout=True)
    values = wide.to_numpy(dtype=float)
    lower = min(.55, float(np.nanmin(values)) - .05)
    upper_data = float(np.nanmax(values))
    upper = max(1.48, upper_data + .15)
    for axis, condition in zip(axes, ("control", "delay")):
        selected = wide.xs(condition, level="condition_id")
        color = COLORS[condition]
        for _, row in selected.iterrows():
            axis.plot(range(3), row.to_numpy(dtype=float), color=color, alpha=.23,
                      linewidth=.65, marker="o", markersize=2.4)
        median = selected.median(axis=0)
        axis.plot(range(3), median, color="0.1", linewidth=1.6, marker="o",
                  markersize=4.5, zorder=5)
        axis.axhline(1, color="0.65", linewidth=.7)
        axis.set_xticks(range(3), BLOCK_LABELS)
        axis.set_xlim(-.2, 2.2)
        axis.set_ylim(lower, upper)
        axis.set_title(f"{condition.capitalize()} (n={len(selected)})", color=color)
        style_axes(axis, theme=theme, show_xticks=True, show_yticks=condition == "control")
        for j, (left, right) in enumerate(((0, 1), (1, 2))):
            result = table.loc[(table["family"] == "within condition") &
                               (table["condition"] == condition) &
                               (table["block_left"] == left)].iloc[0]
            if result["stars"] != "ns":
                y = upper_data + .04 + j * .06
                axis.plot([left, left, right, right], [y-.012, y, y, y-.012],
                          color="0.2", linewidth=.7)
                axis.text((left+right)/2, y+.005, result["stars"],
                          ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("Response / pre-CS baseline")
    between = table.loc[table["family"].eq("between conditions")]
    label = "Between conditions (Holm): " + "   ".join(
        f"{BLOCK_LABELS[i]} {between.iloc[i]['stars']}" for i in range(3))
    figure.suptitle(f"Figure 2D · {METRIC_DISPLAY_NAMES[metric_id]}\n{label}", fontsize=9)
    return figure


def legacy_local_lme(model_input: pd.DataFrame) -> pd.DataFrame:
    """Legacy-style within-block mean and rate tests on current model rows."""
    rows = []
    order = (model_input[["block_10_name", "trial_number"]].drop_duplicates()
             .groupby("block_10_name", observed=True)["trial_number"].min().sort_values().index)
    formula = ("log_response ~ log_baseline + "
               "C(condition_id, Treatment(reference='control')) * within_block_trial")
    for name in order:
        data = model_input.loc[model_input["block_10_name"].eq(name)].copy()
        data["within_block_trial"] = data["trial_number"] - data["trial_number"].mean()
        record = {"block_10_name": name, "n_fish": data["fish_key"].nunique(),
                  "n_rows": len(data), "converged": False, "warning": "", "error": ""}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = smf.mixedlm(formula, data, groups=data["fish_key"],
                                  re_formula="1").fit(reml=False, method="powell",
                                                       maxiter=200, disp=False)
            record["converged"] = bool(fit.converged)
            record["warning"] = " | ".join(str(item.message) for item in caught)
            terms = list(fit.params.index)
            mean_term = next(t for t in terms if "C(condition_id" in t and ":" not in t)
            slope_term = next(t for t in terms if "C(condition_id" in t and ":within_block_trial" in t)
            for label, term in (("mean", mean_term), ("slope", slope_term)):
                record[f"{label}_estimate"] = float(fit.params[term])
                record[f"{label}_p_raw"] = float(fit.pvalues[term])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
    result = pd.DataFrame(rows)
    for label in ("mean", "slope"):
        column = f"{label}_p_raw"
        mask = result["converged"] & np.isfinite(result[column])
        result[f"{label}_p_fdr"] = np.nan
        if mask.any():
            result.loc[mask, f"{label}_p_fdr"] = multipletests(
                result.loc[mask, column], method="fdr_bh")[1]
    return result


def trial_fdr_table(trials: pd.DataFrame) -> pd.DataFrame:
    result = trials[["trial_number", "learning_contrast", "standard_error",
                     "pointwise_lower", "pointwise_upper", "simultaneous_lower",
                     "simultaneous_upper", "estimable"]].copy()
    result["p_pointwise"] = 2 * norm.sf(np.abs(result["learning_contrast"] /
                                                   result["standard_error"]))
    mask = result["estimable"] & np.isfinite(result["p_pointwise"])
    result["p_fdr_bh"] = np.nan
    result.loc[mask, "p_fdr_bh"] = multipletests(result.loc[mask, "p_pointwise"],
                                                 method="fdr_bh")[1]
    result["simultaneous_supported"] = result["simultaneous_lower"] > 0
    return result


def render_g(frames: dict[str, pd.DataFrame], summary: dict,
             local: pd.DataFrame, trial_tests: pd.DataFrame):
    apply_theme()
    model = frames["model_input"]
    metric = model["metric_id"].unique()
    outcome = model["outcome_id"].unique()
    if len(metric) != 1 or len(outcome) != 1 or outcome[0] != "total-activity":
        raise ValueError("This review expects one total-activity LME metric")
    group = frames["group_trajectory"]
    blocks = frames["block_contrasts"]
    global_test = frames["block_global_test"].iloc[0]
    trial = frames["trial_contrasts"].sort_values("trial_number")
    onset = frames["onset"].iloc[0]
    fig, (marks_axis, axis, contrast_axis) = plt.subplots(3, 1,
        figsize=mm_to_in(183, 150), sharex=True, constrained_layout=True,
        height_ratios=(.72, 1.55, 1))
    for condition in ("control", "delay"):
        d = group.loc[group["condition_id"].eq(condition)].sort_values("trial_number")
        x = d["trial_number"].to_numpy(dtype=float)
        color = COLORS[condition]
        axis.fill_between(x, d["q25"].to_numpy(dtype=float), d["q75"].to_numpy(dtype=float),
                          color=color, alpha=.16)
        axis.plot(x, d["median"].to_numpy(dtype=float), color=color, linewidth=1.25,
                  label=f"{condition.capitalize()} (n={int(d['fish_count'].min())})")
    axis.axhline(1, color=".6", linewidth=.65)
    axis.set_xlim(5, 94)
    axis.set_ylim(.65, 1.35)
    axis.set_ylabel("Response / baseline\nmedian [fish IQR]")
    axis.legend(loc="lower right", ncol=2, fontsize=8)
    # Block positions are derived from the authenticated model input.
    block_positions = model.groupby("block_10_name", observed=True)["trial_number"].agg(["min", "max"])
    block_positions["center"] = (block_positions["min"] + block_positions["max"]) / 2
    for _, row in block_positions.iterrows():
        if row["min"] > 5:
            for ax in (marks_axis, axis, contrast_axis):
                ax.axvline(row["min"]-.5, color=".83", linewidth=.55, zorder=0)
    marks_axis.set_ylim(-.5, 4.5)
    marks_axis.set_yticks((0, 1, 2, 3, 4),
        ("Trial FDR", "Slope raw", "Slope FDR", "Mean FDR", "Block Holm"))
    marks_axis.tick_params(axis="y", length=0, labelsize=7)
    marks_axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    marks_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    for _, row in blocks.loc[blocks["supported"]].iterrows():
        center = float(block_positions.loc[row["block_10_name"], "center"])
        marks_axis.text(center, 4, stars(float(row["p_value_holm"])),
                  ha="center", va="center", fontsize=10, color=".2")
    for _, row in local.iterrows():
        center = float(block_positions.loc[row["block_10_name"], "center"])
        for label, y, color in (("mean", 3, ".45"), ("slope", 2, "#B64700")):
            marker = stars(float(row[f"{label}_p_fdr"]))
            if marker != "ns":
                marks_axis.text(center, y, marker, ha="center", va="center",
                                fontsize=9, color=color)
        raw_slope = stars(float(row["slope_p_raw"]))
        if raw_slope != "ns":
            marks_axis.text(center, 1, raw_slope, ha="center", va="center",
                            fontsize=8, color="#B64700")
    significant = trial_tests.loc[trial_tests["p_fdr_bh"] < .05, "trial_number"]
    if len(significant):
        marks_axis.scatter(significant, np.zeros(len(significant)), marker="*",
                           s=15, color=".1")
    x = trial["trial_number"].to_numpy(dtype=float)
    contrast_axis.fill_between(x, trial["pointwise_lower"].to_numpy(dtype=float),
        trial["pointwise_upper"].to_numpy(dtype=float), color=".65", alpha=.22,
        label="Pointwise 95% CI")
    contrast_axis.fill_between(x, trial["simultaneous_lower"].to_numpy(dtype=float),
        trial["simultaneous_upper"].to_numpy(dtype=float), color="#674189", alpha=.13,
        label="Simultaneous 95% band")
    contrast_axis.plot(x, trial["learning_contrast"].to_numpy(dtype=float),
                       color="#674189", linewidth=1.15)
    contrast_axis.axhline(0, color=".25", linewidth=.7)
    contrast_axis.set_ylabel("LME learning\ncontrast (log activity)")
    contrast_axis.set_xlabel("Global CS trial")
    contrast_axis.legend(loc="lower left", ncol=2, fontsize=7)
    fig.suptitle(f"Figure 2G exploratory LME · {METRIC_DISPLAY_NAMES[metric[0]]}\n"
        f"Global condition × block p={float(global_test['p_value']):.2g}; "
        f"block Holm, local FDR, trial FDR={len(significant)}/90; "
        f"simultaneous onset {'localized' if bool(onset['localized']) else 'not localized'}",
        fontsize=9)
    return fig, metric[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path,
        default=Path("/Volumes/JOAQUIM/Digested Data/allDelay-full-v1"))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/figure2-delay"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("outputs/figure2-delay/legacy-stats-lme-review"))
    parser.add_argument("--analysis-id", default="allDelay-full-learning-onset-v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    snippet = f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src .venv/bin/python scripts/render_figure2_legacy_stats_lme_review.py --project-dir '{args.project_dir}' --input-dir '{args.input_dir}' --output-dir '{args.output_dir}' --analysis-id {args.analysis_id} --overwrite"
    for metric in METRIC_DISPLAY_NAMES:
        fish_path = args.input_dir / f"figure-2D_{metric}_fish-data.parquet"
        fish = pd.read_parquet(fish_path)
        wide, stats = legacy_block_tests(fish)
        stats_path = args.output_dir / f"figure-2D_{metric}_legacy-tests.csv"
        if stats_path.exists() and not args.overwrite:
            raise FileExistsError(stats_path)
        stats.to_csv(stats_path, index=False)
        fig = render_d(wide, stats, metric)
        base = args.output_dir / f"figure-2D_delay-control_{metric}_legacy-stars"
        try:
            result = export_matplotlib_figure(fig, base,
                FigureProvenance(figure_id="figure-2D-delay-control-legacy-tests-review",
                    analysis_recipe="corrected-ratio-legacy-nonparametric-tests",
                    source_file=str(source), source_symbol="render_d", source_hash=sha256_file(source),
                    reproduction_snippet=snippet,
                    input_artifacts=tuple({"path": str(p.resolve()), "sha256": sha256_file(p)}
                                          for p in (fish_path, stats_path))),
                mode=FigureMode.STATIC, panel_ids=["control", "delay"], overwrite=args.overwrite)
        finally:
            plt.close(fig)
        print(*result.outputs, stats_path)
    frames, summary = load_learning_onset_analysis(args.project_dir, args.analysis_id)
    local = legacy_local_lme(frames["model_input"])
    trial_tests = trial_fdr_table(frames["trial_contrasts"])
    local_path = args.output_dir / "figure-2G_local-block-LME.csv"
    trial_path = args.output_dir / "figure-2G_trial-LME-FDR.csv"
    for path in (local_path, trial_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    local.to_csv(local_path, index=False)
    trial_tests.to_csv(trial_path, index=False)
    fig, metric = render_g(frames, summary, local, trial_tests)
    analysis_root = args.project_dir / "Processed data" / "Analyses" / args.analysis_id
    quality_root = args.project_dir / "Quality checks" / "Analyses" / args.analysis_id
    paths = [analysis_root / f"{name}.parquet" for name in
             ("learning-model-input", "figure-group-trajectories", "block-contrasts",
              "trial-contrasts", "learning-onset")]
    paths.append(analysis_root / "block-global-test.parquet")
    paths.extend((quality_root / "learning-onset_summary.json", local_path, trial_path))
    base = args.output_dir / f"figure-2G_delay-control_{metric}_lme-review"
    try:
        result = export_matplotlib_figure(fig, base,
            FigureProvenance(figure_id="figure-2G-delay-control-LME-review",
                analysis_recipe="authenticated-learning-onset-global-block-trial-LME",
                source_file=str(source), source_symbol="render_g", source_hash=sha256_file(source),
                reproduction_snippet=snippet,
                input_artifacts=tuple({"path": str(p.resolve()), "sha256": sha256_file(p)}
                                      for p in paths), cohort_hash=summary["cohort_hash"]),
            mode=FigureMode.STATIC, panel_ids=["significance_lanes", "trajectories", "learning_contrast"],
            overwrite=args.overwrite)
    finally:
        plt.close(fig)
    print(*result.outputs)


if __name__ == "__main__":
    main()
