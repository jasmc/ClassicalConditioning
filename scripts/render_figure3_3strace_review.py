"""Render an exploratory 3sTrace learner review from the archived WIP rule.

This is a descriptive Figure 3 companion, not the paper's validated Gate L.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cohort import load_cohort_manifest, logical_cohort_hash
from classical_conditioning.figures.export import (
    FigureMode, FigureProvenance, export_matplotlib_figure,
)


METRIC = "tail_length_weighted_angular_l1"
METRICS = (METRIC, "whole_tail_xy_mean_speed_normalized", "legacy_distal_angular_speed")
VARIANT = "legacy-wip"


def load_review(project: Path, cohort_id: str, comparison_dir: Path,
                metric_id: str = METRIC) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if metric_id not in METRICS:
        raise ValueError(f"Unsupported Figure 3 metric: {metric_id}")
    cohort = load_cohort_manifest(project, cohort_id)
    selected = cohort.loc[cohort["primary_included"].astype(bool)].copy()
    cohort_hash = logical_cohort_hash(cohort)
    cohort_path = project / "Processed data" / "Cohorts" / cohort_id / "cohort-manifest.parquet"
    outcomes_path = project / "Processed data" / "Cohorts" / cohort_id / "cohort-trial-outcomes.parquet"
    comparison_path = comparison_dir / "comparison.json"
    report = json.loads(comparison_path.read_text(encoding="utf-8"))
    variant = next((item for item in report["variants"] if item["variant"] == VARIANT), None)
    if (report.get("experiment_id") != "all3sTrace" or report.get("metric_id") != metric_id
            or report.get("cohort_hash") != cohort_hash
            or report.get("cohort_sha256") != sha256_file(cohort_path)
            or report.get("outcomes_sha256") != sha256_file(outcomes_path)
            or variant is None or variant.get("status") != "completed"
            or variant.get("result_sha256") != sha256_file(comparison_dir / f"{VARIANT}.parquet")):
        raise ValueError("Figure 3 review input identities or hashes disagree")
    scores = pd.read_parquet(comparison_dir / f"{VARIANT}.parquet")
    scores["condition_id"] = scores["Condition"].replace({"delay": "trace"})
    scores["fish_id"] = scores["Fish_ID"].astype(str)
    scores = scores[["condition_id", "fish_id", "T_joint", "decision_threshold", "Is_Learner"]]
    if scores.duplicated(["condition_id", "fish_id"]).any():
        raise ValueError("WIP result contains duplicate fish")
    fish = selected[["condition_id", "fish_id"]].merge(
        scores, on=["condition_id", "fish_id"], how="left", validate="one_to_one",
    )
    if (fish["T_joint"].isna() != fish["Is_Learner"].isna()).any() or fish["Is_Learner"].notna().sum() == 0:
        raise ValueError("WIP score and classification eligibility disagree")
    fish["Is_Learner"] = fish["Is_Learner"].astype("boolean")
    outcomes = pd.read_parquet(outcomes_path)
    outcomes = outcomes.loc[outcomes["alignment"].eq("CS") & outcomes["metric_id"].eq(metric_id)
                            & outcomes["trial_number"].between(5, 94)].copy()
    if set(outcomes["fish_id"].astype(str)) != set(fish["fish_id"].astype(str)):
        raise ValueError("Cohort trial outcomes and WIP fish differ")
    baseline = pd.to_numeric(outcomes["baseline_total_activity"], errors="coerce")
    response = pd.to_numeric(outcomes["response_total_activity"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        outcomes["response_baseline_ratio"] = response / baseline
    outcomes.loc[~np.isfinite(outcomes["response_baseline_ratio"])
                 | outcomes["response_baseline_ratio"].le(0), "response_baseline_ratio"] = np.nan
    return fish, outcomes, report


def render(fish: pd.DataFrame, outcomes: pd.DataFrame):
    colors = {"trace": "#0072B2", "control": "#777777"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5), constrained_layout=True)
    eligible = fish.loc[fish["Is_Learner"].notna()]
    if set(eligible["condition_id"]) != {"control", "trace"}:
        raise ValueError("WIP review requires classified 3sTrace and control fish")
    threshold = float(pd.to_numeric(eligible["decision_threshold"]).iloc[0])
    if not np.allclose(pd.to_numeric(eligible["decision_threshold"]), threshold):
        raise ValueError("WIP threshold is inconsistent across fish")
    for x, condition in enumerate(("control", "trace")):
        group = eligible.loc[eligible["condition_id"].eq(condition)]
        jit = np.linspace(-.12, .12, len(group)) if len(group) > 1 else [0.]
        axes[0, 0].scatter(np.full(len(group), x) + jit, group["T_joint"],
                           c=["#D55E00" if value else colors[condition]
                              for value in group["Is_Learner"]], s=25, alpha=.8)
    axes[0, 0].axhline(threshold, color="#D55E00", linestyle="--", lw=1,
                       label=f"WIP threshold {threshold:.2f}")
    axes[0, 0].set_xticks([0, 1], ["Control", "3sTrace"])
    axes[0, 0].set_ylabel("Archived WIP joint score")
    axes[0, 0].legend(handles=[
        Line2D([0], [0], color="#D55E00", linestyle="--", lw=1,
               label=f"WIP score threshold {threshold:.2f}"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#D55E00",
               label="Flagged learner"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#0072B2",
               label="3sTrace not flagged"),
    ], frameon=False, fontsize=7)
    axes[0, 0].set_title("A  Score distribution and classifier flags", loc="left")

    counts = eligible.groupby("condition_id")["Is_Learner"].agg(["sum", "count"])
    for x, condition in enumerate(("control", "trace")):
        row = counts.loc[condition]
        rate = float(row["sum"] / row["count"])
        axes[0, 1].bar(x, rate, color=colors[condition], width=.55)
        axes[0, 1].text(x, min(1.03, rate + .04), f"{int(row['sum'])}/{int(row['count'])}",
                        ha="center", fontsize=9)
    axes[0, 1].set_xticks([0, 1], ["Control flagged", "3sTrace learners"])
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].set_ylabel("Fraction of classified fish")
    axes[0, 1].set_title("B  Same-data descriptive fractions", loc="left")

    ratio = outcomes[["condition_id", "fish_id", "trial_number", "response_baseline_ratio"]].copy()
    ratio["log2_ratio"] = np.log2(ratio["response_baseline_ratio"])
    early = ratio.loc[ratio["trial_number"].between(5, 14)].groupby(
        ["condition_id", "fish_id"])["log2_ratio"].median().rename("early")
    late = ratio.loc[ratio["trial_number"].between(85, 94)].groupby(
        ["condition_id", "fish_id"])["log2_ratio"].median().rename("late")
    change = pd.concat([early, late], axis=1).reset_index()
    change["late_minus_early"] = change["late"] - change["early"]
    change = change.merge(fish[["condition_id", "fish_id", "Is_Learner"]],
                          on=["condition_id", "fish_id"], validate="one_to_one")
    for x, condition in enumerate(("control", "trace")):
        group = change.loc[change["condition_id"].eq(condition)].sort_values("fish_id")
        jit = np.linspace(-.12, .12, len(group)) if len(group) > 1 else [0.]
        axes[1, 0].scatter(np.full(len(group), x) + jit, group["late_minus_early"],
                           c=["#999999" if pd.isna(value) else ("#D55E00" if value else colors[condition])
                              for value in group["Is_Learner"]], s=25, alpha=.8)
        axes[1, 0].plot([x - .18, x + .18], [group["late_minus_early"].median()] * 2,
                        color="black", lw=2)
    axes[1, 0].axhline(0, color="black", lw=.7)
    axes[1, 0].set_xticks([0, 1], ["Control", "3sTrace"])
    axes[1, 0].set_ylabel("Late Test − Pre-train median log₂(response / baseline)")
    axes[1, 0].set_title("C  All-fish paired change", loc="left")

    trace = eligible.loc[eligible["condition_id"].eq("trace")].sort_values("T_joint")
    if trace.empty:
        raise ValueError("No classified 3sTrace fish for the score-selected trajectories")
    representatives = pd.concat([trace.head(1), trace.iloc[[max(0, len(trace)//2 - 1)]],
                                  trace.tail(2)]).drop_duplicates("fish_id")
    trial = ratio.loc[ratio["condition_id"].eq("trace")]
    for row in representatives.itertuples(index=False):
        fish_trials = trial.loc[trial["fish_id"].eq(row.fish_id)].sort_values("trial_number")
        values = fish_trials["log2_ratio"].rolling(5, min_periods=2, center=True).median()
        axes[1, 1].plot(fish_trials["trial_number"], values,
                        lw=1.3, label=f"{row.fish_id} ({'flagged' if row.Is_Learner else 'not flagged'})")
    for boundary in (14.5, 64.5):
        axes[1, 1].axvline(boundary, color="0.6", lw=.7)
    axes[1, 1].axhline(0, color="black", lw=.7)
    axes[1, 1].set_xlim(5, 94)
    axes[1, 1].set_xlabel("Global CS trial")
    axes[1, 1].set_ylabel("5-trial median log₂(response / baseline)")
    axes[1, 1].set_title("D  Score-selected example trajectories", loc="left")
    axes[1, 1].legend(frameon=False, fontsize=7)
    fig.suptitle("3sTrace Figure 3 exploratory review · archived legacy-wip rule", fontsize=12)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--metric", choices=METRICS, default=METRIC)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project, comparison_dir = args.project_dir.resolve(), args.comparison_dir.resolve()
    fish, outcomes, report = load_review(project, args.cohort_id, comparison_dir,
                                         metric_id=args.metric)
    fig = render(fish, outcomes)
    output = (args.output_dir or project / "Figures" / "PNG" / "Analyses"
              / "figure3-3strace-exploratory").resolve()
    cohort_hash = report["cohort_hash"]
    comparison_path = comparison_dir / "comparison.json"
    variant_path = comparison_dir / f"{VARIANT}.parquet"
    outcomes_path = project / "Processed data" / "Cohorts" / args.cohort_id / "cohort-trial-outcomes.parquet"
    source = Path(__file__).resolve()
    try:
        result = export_matplotlib_figure(
            fig, output / f"figure-3-3strace-review_{args.metric}",
            FigureProvenance(
                figure_id="figure-3-3strace-exploratory-review",
                analysis_recipe="legacy-wip-descriptive-classification-review/1.0",
                source_file=str(source), source_symbol="main", source_hash=sha256_file(source),
                reproduction_snippet=(
                    "python scripts/render_figure3_3strace_review.py "
                    f"--project-dir '{project}' --cohort-id {args.cohort_id} "
                    f"--comparison-dir '{comparison_dir}' --metric {args.metric}"
                ),
                input_artifacts=tuple({"path": str(path), "sha256": sha256_file(path)}
                                      for path in (comparison_path, variant_path, outcomes_path)),
                cohort_hash=cohort_hash,
                artist_mappings={"axes__score__main": {"metric_id": args.metric, "variant": VARIANT},
                                 "axes__paired_change__main": {"metric_id": args.metric, "trial_blocks": "Pre-train versus Test 3"}},
                analysis_identity={"experiment_id": "all3sTrace", "cohort_id": args.cohort_id,
                                   "metric_id": args.metric, "classifier_version": VARIANT,
                                   "scientific_status": "descriptive_provisional_legacy_rule"},
            ),
            mode=FigureMode.STATIC, panel_ids=("score", "fractions", "paired_change", "examples"),
            overwrite=args.overwrite,
        )
    finally:
        plt.close(fig)
    for path in (*result.outputs, result.sidecar):
        print(path)


if __name__ == "__main__":
    main()
