"""Render exploratory Figure 2E/H using the archived normalized-vigor layout.

The input is the frozen, corrected 0–13 s cohort trial table. Its
``legacy_distal_angular_speed`` candidate is the current approximation to
the archived Vigor (deg/ms) column; the archived fish exclusions and LME
statistics are deliberately not inferred from this table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from classical_conditioning.analysis.cohort_outcomes import load_cohort_trial_outcomes
from classical_conditioning.figures.cohort_response import (
    SelectedBlock,
    summarize_selected_block_ratios,
    summarize_trial_ratios,
)
from classical_conditioning.artifacts import sha256_file


METRIC = "legacy_distal_angular_speed"
BLOCKS = (
    SelectedBlock("PTr", 5, 9),
    SelectedBlock("ETe", 65, 69),
    SelectedBlock("LTe", 90, 94),
)
PALETTE = {"control": (0 / 255, 174 / 255, 239 / 255), "trace": (241 / 255, 90 / 255, 41 / 255)}
LABELS = {"control": "Control", "trace": "3sTrace"}
YLIM = (0.8, 1.2)


def _prepare(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    blocks, _ = summarize_selected_block_ratios(
        outcomes, metric_id=METRIC, selected_blocks=BLOCKS,
        min_trials_per_fish_block=1,
    )
    blocks = blocks.loc[blocks["Eligible"]].copy()
    trials, _ = summarize_trial_ratios(outcomes, metric_id=METRIC)
    trials = trials.loc[trials["trial_number"].between(5, 94)].copy()
    if blocks.empty or trials.empty:
        raise ValueError("No valid 3sTrace CS response/baseline ratios")
    return blocks, trials


def _style(axis: plt.Axes) -> None:
    axis.axhline(1.0, color="black", linewidth=0.55, zorder=0)
    axis.set_ylim(*YLIM)
    axis.set_yticks([0.8, 1.0, 1.2])
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8, width=0.5, length=2)
    axis.set_ylabel("Normalized vigor (AU)", fontsize=8)
    axis.set_xlabel("")


def _panel_e(blocks: pd.DataFrame, target: Path) -> None:
    frame = blocks.rename(columns={
        "Selected block": "Block",
        "Fish median response / baseline": "Normalized vigor",
        "condition_id": "Condition",
    })
    fig, axis = plt.subplots(figsize=(3.25, 2.8), layout="constrained")
    sns.boxplot(
        frame, x="Block", y="Normalized vigor", hue="Condition",
        order=[b.label for b in BLOCKS], hue_order=["control", "trace"],
        palette=PALETTE, showfliers=False, dodge=True, width=0.72,
        saturation=1, linewidth=0.55, ax=axis,
        medianprops={"color": "black", "linestyle": "--", "linewidth": 0.6},
        whiskerprops={"linewidth": 0.55}, capprops={"linewidth": 0.55},
    )
    _style(axis)
    axis.set_xticks(range(len(BLOCKS)), [b.label for b in BLOCKS], fontweight="bold")
    handles, _ = axis.get_legend_handles_labels()
    axis.legend(handles, [f"{LABELS[c]} (n={frame.loc[frame.Condition == c, 'fish_id'].nunique()})"
                          for c in ("control", "trace")], frameon=False,
                loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    fig.savefig(target, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _panel_h(trials: pd.DataFrame, target: Path) -> None:
    frame = trials.rename(columns={
        "Fish median response / baseline": "Normalized vigor",
        "trial_number": "Trial number",
        "condition_id": "Condition",
    })
    fig, axis = plt.subplots(figsize=(3.25, 2.8), layout="constrained")
    for condition in ("control", "trace"):
        selected = frame.loc[frame.Condition == condition]
        sns.lineplot(
            selected, x="Trial number", y="Normalized vigor",
            estimator="median", errorbar=("ci", 95), n_boot=100, seed=10,
            err_style="band", err_kws={"alpha": 0.3},
            color=PALETTE[condition], alpha=0.9, linewidth=0.8,
            label=f"{LABELS[condition]} (n={selected.fish_id.nunique()})", ax=axis,
        )
    for boundary in (14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5, 84.5):
        axis.axvline(boundary, color="gray", alpha=0.5, linewidth=0.5, zorder=0)
    _style(axis)
    axis.set_xlim(4, 95)
    axis.set_xticks([5, 15, 35, 55, 65, 75, 94])
    axis.set_xlabel("CS trial number", fontsize=8)
    axis.legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(target, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outcomes, summary = load_cohort_trial_outcomes(args.project_dir, args.cohort_id)
    blocks, trials = _prepare(outcomes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e_path = args.output_dir / "figure-2E_legacy-distal-angular-speed.png"
    h_path = args.output_dir / "figure-2H_legacy-distal-angular-speed.png"
    _panel_e(blocks, e_path)
    _panel_h(trials, h_path)
    blocks.to_csv(args.output_dir / "figure-2E_fish-block-data.csv", index=False)
    trials.to_csv(args.output_dir / "figure-2H_fish-trial-data.csv", index=False)
    manifest = {
        "cohort_id": args.cohort_id,
        "cohort_hash": summary["cohort_hash"],
        "metric_id": METRIC,
        "response_window_s": [0, 13],
        "source": "legacy/scripts/5_NormalizedVigorPlotting.py",
        "block_trials": {b.label: [b.start_trial, b.end_trial] for b in BLOCKS},
        "plot_spec": "fish median per five-trial block; grouped boxes; per-trial condition median and 95% bootstrap CI (100 resamples, seed 10); y=0.8–1.2",
        "limits": "The current legacy_distal_angular_speed metric approximates archived Vigor (deg/ms); no archived fish exclusions or LME significance markers applied.",
        "inputs": [{"path": str(args.project_dir / "Processed data" / "Cohorts" / args.cohort_id / "cohort-trial-outcomes.parquet"),
                    "sha256": sha256_file(args.project_dir / "Processed data" / "Cohorts" / args.cohort_id / "cohort-trial-outcomes.parquet")}],
        "outputs": {p.name: sha256_file(p) for p in (e_path, h_path)},
    }
    (args.output_dir / "figure-2EH-legacy-layout.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure_2E": str(e_path), "figure_2H": str(h_path),
                      "fish": int(trials.fish_id.nunique())}, indent=2))


if __name__ == "__main__":
    main()
