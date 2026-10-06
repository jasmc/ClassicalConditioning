"""Compare Figure 2E/H and Figure 4 layouts using saved exploratory data.

Figure 2 variants adapt fig2-dg-descriptive and fig2-dg-legacy-stars-and-lme
from review-variants.json. The latter's block-test family is recomputed here;
its Delay LME results are not transferred to 3sTrace. Figure 4 layout variants
reuse authenticated signed group bins without rerunning the classifier.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests
import seaborn as sns

from classical_conditioning.analysis.cohort_outcomes import load_cohort_trial_outcomes
from classical_conditioning.analysis.figure4 import load_figure4_analysis, STRATA
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
from classical_conditioning.figures.figure4 import COLORS, STYLES
from render_figure2_3strace_legacy_layout import _prepare, _panel_e, _panel_h, BLOCKS, METRIC, PALETTE, LABELS


def block_tests(blocks: pd.DataFrame) -> pd.DataFrame:
    wide = blocks.pivot(index=["fish_id", "condition_id"], columns="Selected block order",
                        values="Fish median response / baseline").reindex(columns=[0, 1, 2])
    records = []
    for block in range(3):
        c = wide.xs("control", level="condition_id")[block].dropna()
        t = wide.xs("trace", level="condition_id")[block].dropna()
        records.append(dict(family="between", condition="trace vs control", left=block, right=block,
                            test="Mann-Whitney U", n=len(c)+len(t), n_control=len(c), n_trace=len(t),
                            p_raw=float(mannwhitneyu(c, t, alternative="two-sided").pvalue)))
    for condition in ("control", "trace"):
        part = wide.xs(condition, level="condition_id")
        for left, right in ((0, 1), (1, 2)):
            pairs = part[[left, right]].dropna()
            difference = pairs[left] - pairs[right]
            p = 1.0 if difference.eq(0).all() else float(wilcoxon(pairs[left], pairs[right]).pvalue)
            records.append(dict(family="within", condition=condition, left=left, right=right,
                                test="paired Wilcoxon", n=len(pairs), p_raw=p))
    results = pd.DataFrame(records)
    for _, indices in results.groupby("family").groups.items():
        results.loc[indices, "p_holm"] = multipletests(results.loc[indices, "p_raw"], method="holm")[1]
    return results


def figure2_e_summary(blocks: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.1, 3.8), layout="constrained")
    rng = np.random.default_rng(10)
    for offset, condition in ((-.12, "control"), (.12, "trace")):
        d = blocks.loc[blocks.condition_id.eq(condition)]
        for order in range(3):
            values = d.loc[d["Selected block order"].eq(order), "Fish median response / baseline"].dropna()
            x = order + offset
            ax.scatter(x + rng.uniform(-.035, .035, len(values)), values, s=12,
                       alpha=.4, color=PALETTE[condition], linewidths=0)
            q = values.quantile([.25, .5, .75]).to_numpy()
            ax.errorbar(x, q[1], yerr=[[q[1]-q[0]], [q[2]-q[1]]], fmt="o",
                        color=PALETTE[condition], capsize=4, markersize=5, linewidth=1.7)
    ax.axhline(1, color=".4", lw=.6)
    ax.set_xticks(range(3), [b.label for b in BLOCKS])
    ax.set_ylabel("Response / pre-CS baseline")
    ax.set_title("2E · fish points and median [IQR]", loc="left")
    ax.legend([Line2D([], [], marker="o", color=PALETTE[c], ls="") for c in ("control", "trace")],
              [f"{LABELS[c]} (n={blocks.loc[blocks.condition_id.eq(c), 'fish_id'].nunique()})" for c in ("control", "trace")],
              frameon=False)
    return fig


def figure2_e_paired(blocks: pd.DataFrame, tests: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), sharey=True, layout="constrained")
    for ax, condition in zip(axes, ("control", "trace")):
        d = blocks.loc[blocks.condition_id.eq(condition)]
        wide = d.pivot(index="fish_id", columns="Selected block order", values="Fish median response / baseline").reindex(columns=[0,1,2])
        for _, row in wide.iterrows():
            ax.plot(range(3), row, "o-", color=PALETTE[condition], alpha=.25, lw=.65, ms=2)
        ax.plot(range(3), wide.median(), "o-", color="black", lw=1.6, ms=4)
        ax.axhline(1, color=".5", lw=.6)
        ax.set_xticks(range(3), [b.label for b in BLOCKS])
        ax.set_title(f"{LABELS[condition]} (n={len(wide)})")
        lines = []
        for r in tests.loc[tests.condition.eq(condition)].itertuples():
            lines.append(f"{BLOCKS[r.left].label}–{BLOCKS[r.right].label}: p={r.p_holm:.3g}")
        ax.text(.03, .03, "Paired Wilcoxon, Holm\n" + "\n".join(lines), transform=ax.transAxes,
                fontsize=7, va="bottom", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    axes[0].set_ylabel("Response / pre-CS baseline")
    fig.suptitle("2E · paired fish trajectories; black = median", fontsize=10)
    return fig


def figure2_h(trials: pd.DataFrame, *, split: bool):
    fig, axes = plt.subplots(2 if split else 1, 1, figsize=(7.1, 4.8 if split else 3.6),
                             sharex=True, sharey=True, layout="constrained", squeeze=False)
    axes = axes.ravel()
    for i, condition in enumerate(("control", "trace")):
        ax = axes[i] if split else axes[0]
        d = trials.loc[trials.condition_id.eq(condition)]
        label = f"{LABELS[condition]} (n={d.fish_id.nunique()})"
        if split:
            sns.lineplot(d, x="trial_number", y="Fish median response / baseline",
                         estimator="median", errorbar=("ci", 95), n_boot=100, seed=10,
                         color=PALETTE[condition], lw=1, label=label, ax=ax,
                         err_kws={"alpha": .2})
        else:
            grouped = d.groupby("trial_number")["Fish median response / baseline"]
            median, lower, upper = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
            ax.plot(median.index, median, color=PALETTE[condition], lw=1, label=label)
            ax.fill_between(median.index, lower, upper, color=PALETTE[condition], alpha=.2, lw=0)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes:
        ax.axhline(1, color=".4", lw=.6)
        for boundary in (14.5, 64.5):
            ax.axvline(boundary, color=".4", ls=":", lw=.7)
        ax.set_xlim(4, 95)
        ax.set_ylim(.8, 1.2)
        ax.set_ylabel("Response / baseline")
    axes[-1].set_xlabel("CS trial number")
    axes[0].set_title("2H · " + ("separate conditions, median [95% bootstrap CI]" if split else "condition median [fish IQR]"), loc="left", fontsize=10)
    return fig


def figure4_grid(groups: pd.DataFrame, flow: pd.DataFrame, us: float, *, condition: str | None = None):
    rows = groups.loc[groups.group_type.isin(("block", "pooled_catch")), ["group_name", "group_order"]].drop_duplicates().sort_values("group_order")
    fig, axes = plt.subplots(5, 2, figsize=(9.2, 9.4), sharex=True, sharey=True, layout="constrained")
    strata = STRATA[:2] if condition == "trace" else STRATA[2:] if condition == "control" else STRATA
    counts = flow.plot_stratum.value_counts()
    signed = groups[["signed_q25", "signed_q75"]].to_numpy(dtype=float)
    finite = np.abs(signed[np.isfinite(signed)])
    limit = max(.25, float(finite.max()) * 1.05)
    for ax, r in zip(axes.ravel(), rows.itertuples()):
        for stratum in strata:
            d = groups.loc[groups.group_name.eq(r.group_name) & groups.plot_stratum.eq(stratum)].sort_values("time_s")
            ax.plot(d.time_s, d.signed_median, color=COLORS[stratum], ls=STYLES[stratum], lw=1)
            ax.fill_between(d.time_s, d.signed_q25, d.signed_q75,
                            where=d.signed_fish.ge(2), color=COLORS[stratum], alpha=.12, lw=0)
        ax.axhline(0, color=".5", lw=.45)
        ax.axvspan(0, 10, color="#009E73", alpha=.07, lw=0)
        ax.axvline(us, color=".4", ls=":", lw=.7)
        ax.set_title(r.group_name, loc="left", fontsize=9)
        ax.set_xlim(-20, 20)
        ax.set_ylim(-limit, limit)
        ax.tick_params(labelsize=7)
    fig.supxlabel("Time from CS onset (s)", fontsize=9)
    fig.supylabel("Signed log vigor · median [fish IQR]", fontsize=9)
    title = "All classifier groups" if condition is None else LABELS[condition]
    handles = [Line2D([], [], color=COLORS[s], ls=STYLES[s], label=f"{s} (n={int(counts.get(s, 0))})") for s in strata]
    handles.extend([Patch(facecolor="#009E73", alpha=.1, label="CS 0–10 s"),
                    Line2D([], [], color=".4", ls=":", label=f"Expected US {us:.3f} s")])
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(.5,.965), frameon=False)
    fig.suptitle("Figure 4 · " + title, fontsize=11)
    fig.get_layout_engine().set(rect=(0,0,1,.89))
    return fig


def save(fig, path: Path, description: str, inputs: tuple[dict, ...], identity: dict) -> None:
    source = Path(__file__).resolve()
    try:
        export_matplotlib_figure(fig, path, FigureProvenance(
            figure_id=path.name, analysis_recipe="3strace-plot-versions/1.0",
            source_file=str(source), source_symbol="main", source_hash=sha256_file(source),
            reproduction_snippet=identity["reproduction_snippet"],
            input_artifacts=inputs, analysis_identity={**identity, "plot_description": description}),
            mode=FigureMode.STATIC, overwrite=True)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--cohort-id", default="all3sTrace-full-exploratory")
    parser.add_argument("--figure4-analysis-id", default="figure4-3strace-window13-legacy-59fish")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outcomes, cohort_summary = load_cohort_trial_outcomes(args.project_dir, args.cohort_id)
    blocks, trials = _prepare(outcomes)
    f4_path = args.project_dir / "Processed data" / "Analyses" / args.figure4_analysis_id / "figure4" / "analysis.json"
    summary4, tables = load_figure4_analysis(f4_path)
    if summary4["metric_id"] != METRIC or summary4["cohort_hashes"]["all3sTrace"] != cohort_summary["cohort_hash"]:
        raise ValueError("Figure 2 and 4 must use the same metric and cohort")
    input_paths = [args.project_dir / "Processed data" / "Cohorts" / args.cohort_id / "cohort-trial-outcomes.parquet",
                   f4_path, f4_path.with_name("group-bins.parquet"), f4_path.with_name("sample-flow.parquet"),
                   Path(__file__).with_name("render_figure2_3strace_legacy_layout.py")]
    inputs = tuple({"path": str(p.resolve()), "sha256": sha256_file(p)} for p in input_paths)
    identity = dict(metric_id=METRIC, cohort_id=args.cohort_id, cohort_hash=cohort_summary["cohort_hash"],
                    classifier_execution_id=summary4["classifier_execution_id"], response_window_s=[0,13],
                    block_trials={b.label: [b.start_trial,b.end_trial] for b in BLOCKS},
                    scientific_status="descriptive_provisional_legacy_rule",
                    reproduction_snippet=(f'python scripts/render_3strace_plot_versions.py --project-dir "{args.project_dir.resolve()}" '
                                          f'--cohort-id {args.cohort_id} --figure4-analysis-id {args.figure4_analysis_id} --output-dir "{out}"'))
    tests = block_tests(blocks)
    tests.to_csv(out / "figure-2E_legacy-holm-tests.csv", index=False)
    blocks.to_csv(out / "figure-2E_fish-data.csv", index=False)
    trials.to_csv(out / "figure-2H_fish-data.csv", index=False)
    versions = []
    def render(name, label, fig, desc):
        save(fig, out / name, desc, inputs, identity)
        versions.append(dict(file=name+".png", label=label, description=desc))
    _panel_e(blocks, out / "figure-2E_v1-legacy-boxplot.png")
    _panel_h(trials, out / "figure-2H_v1-legacy-bootstrap.png")
    versions.extend([dict(file="figure-2E_v1-legacy-boxplot.png", label="2E v1 · legacy grouped boxes", description="Per-fish median in PTr 5–9, ETe 65–69, LTe 90–94; median and box IQR."),
                     dict(file="figure-2H_v1-legacy-bootstrap.png", label="2H v1 · legacy bootstrap", description="Condition median, 95% bootstrap CI, 100 resamples, seed 10.")])
    render("figure-2E_v2-descriptive-iqr", "2E v2 · descriptive median/IQR", figure2_e_summary(blocks), "Adapted fig2-dg-descriptive: fish points and median/IQR.")
    render("figure-2E_v3-paired-legacy-tests", "2E v3 · paired fish and legacy tests", figure2_e_paired(blocks, tests), "Adapted fig2-dg-legacy-stars-and-lme block layout; paired Wilcoxon and Mann–Whitney tables recomputed, Holm within test families. Finite pairs used, counts exported.")
    render("figure-2H_v2-descriptive-iqr", "2H v2 · median/IQR", figure2_h(trials, split=False), "Adapted fig2-dg-descriptive: condition median and fish IQR, no smoothing.")
    render("figure-2H_v3-separate-conditions", "2H v3 · separate conditions", figure2_h(trials, split=True), "Same legacy bootstrap estimator and interval, conditions in separate axes; no new LME fitted.")
    groups, flow = tables["group-bins"], tables["sample-flow"]
    us = summary4["expected_us"]["all3sTrace"]["expected_us_s"]
    original = args.project_dir / "Figures" / "PNG" / "Analyses" / args.figure4_analysis_id / "all3sTrace" / f"figure-4_{METRIC}.png"
    shutil.copy2(original, out / "figure-4_v1-stacked.png")
    versions.append(dict(file="figure-4_v1-stacked.png", label="4 v1 · existing stacked profiles", description="Saved ten-row median/IQR profiles, same classifier labels."))
    render("figure-4_v2-compact-grid", "4 v2 · compact grid", figure4_grid(groups, flow, us), "Same saved group median/IQR in a 5×2 grid, common y scale across all groups.")
    for condition in ("trace", "control"):
        render(f"figure-4_v3-{condition}-grid", f"4 v3 · {LABELS[condition]} separately",
               figure4_grid(groups, flow, us, condition=condition), "Same saved group median/IQR, split conditions; common y scale retained.")
    manifest = dict(**identity, versions=versions, inputs=inputs,
                    outputs={v["file"]: sha256_file(out / v["file"]) for v in versions},
                    notes="Styling adaptations of registered Figure 2 versions. Figure 4 grid layouts are new display-only alternatives. No archived Delay LME marks transferred.")
    (out / "plot-versions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cards = "\n".join(f'<section><h2>{html.escape(v["label"])}</h2><p>{html.escape(v["description"])}</p><a href="{v["file"]}"><img src="{v["file"]}"></a></section>' for v in versions)
    counts = flow.plot_stratum.value_counts()
    count_text = "; ".join(f"{s}: {int(counts.get(s, 0))}" for s in STRATA)
    unclassified = int(flow.classifier_label.eq("Unclassified").sum())
    (out / "comparison.html").write_text('<!doctype html><meta charset="utf-8"><title>3sTrace plotting versions</title><style>body{font:16px system-ui;background:#eef1f4;margin:24px;color:#17212b}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:24px}section{background:white;padding:20px;border-radius:8px}img{width:100%;height:auto}h2{font-size:20px}p{line-height:1.5}header{margin-bottom:24px}</style><header><h1>3sTrace · Figure 2E/H and Figure 4 plotting versions</h1>'
        +f'<p>{flow.fish_id.nunique()} cohort fish · legacy distal angular speed · 0–13 s response · {html.escape(count_text)}; unclassified: {unclassified}. Same data and labels throughout.</p>'
        +'<p>IQR shows spread across fish; bootstrap CI shows uncertainty in the median. Figure 2E tests are exploratory; no archived Delay LME marks are transferred. Figure 4 uses the existing signed profiles. PTr = trials 5–9; ETe = 65–69; LTe = 90–94.</p></header><main>'+cards+'</main>', encoding="utf-8")
    print(out / "comparison.html")


if __name__ == "__main__":
    main()
