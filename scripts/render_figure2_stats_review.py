"""Add exploratory fish-level condition statistics to saved Figure 2D/G panels.

Tests compare Delay and control changes from each fish's pre-train reference.
The maximal absolute difference across all tested blocks/trials controls the
family-wise error rate by permuting whole-fish condition labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.cohort_response import _plot_selected_blocks
from classical_conditioning.figures.example_traces import METRIC_DISPLAY_NAMES
from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
from render_legacy_ssd_figure2_delay import _plot_trial_summary


def differences(fish: pd.DataFrame, panel: str):
    field = "Fish median response / baseline"
    if panel == "D":
        wide = fish.loc[fish["Eligible"]].pivot(index=["fish_id", "condition_id"],
            columns="Selected block order", values=field).reindex(columns=[0, 1, 2])
        if wide.isna().any().any():
            raise ValueError("Incomplete Figure 2D fish blocks")
        values = wide[[1, 2]].to_numpy() - wide[[0]].to_numpy()
        labels = ["Early Test − Pre-train", "Late Test − Pre-train"]
    else:
        wide = fish.pivot(index=["fish_id", "condition_id"], columns="trial_number",
                          values=field).reindex(columns=range(1, 95))
        if wide.isna().any().any():
            raise ValueError("Incomplete Figure 2G fish trials")
        baseline = wide.loc[:, 10:14].median(axis=1).to_numpy()[:, None]
        values = wide.loc[:, 15:94].to_numpy() - baseline
        labels = [f"Trial {number}" for number in range(15, 95)]
    condition = np.array(wide.index.get_level_values("condition_id") == "delay")
    if condition.sum() != 29 or (~condition).sum() != 28:
        raise ValueError("Expected 29 Delay and 28 control fish")
    return values, condition, labels


def max_t_permutation(values, condition, labels, *, repetitions=4999, seed=20260924):
    rng = np.random.default_rng(seed)
    n_delay = int(condition.sum())
    effect = values[condition].mean(axis=0) - values[~condition].mean(axis=0)
    null_max = np.empty(repetitions)
    for i in range(repetitions):
        selected = rng.permutation(len(condition))[:n_delay]
        # Fixed group sizes and complete fish vectors preserve within-fish dependence.
        total = values.sum(axis=0)
        delayed = values[selected].sum(axis=0)
        permuted = delayed / n_delay - (total - delayed) / (len(condition) - n_delay)
        null_max[i] = np.max(np.abs(permuted))
    adjusted = (1 + (null_max[:, None] >= np.abs(effect)).sum(axis=0)) / (repetitions + 1)
    return pd.DataFrame({"contrast": labels, "delay_minus_control_change": effect,
                         "p_fwer_maxT": adjusted, "permutations": repetitions,
                         "seed": seed})


def render(panel, fish, group, metric_id, stats):
    if panel == "D":
        figure, ids, mappings = _plot_selected_blocks(fish, group, experiment_name="allDelay")
        axis = figure.axes[0]
        lines = [f"{label}: p={p:.3g}" for label, p in
                 zip(("Early Test", "Late Test"), stats["p_fwer_maxT"])]
        axis.text(.99, .02, "Exploratory Δ vs pre-train, max-T FWER\n" + "\n".join(lines),
                  transform=axis.transAxes, va="bottom", ha="right", fontsize=7,
                  bbox={"facecolor": "white", "alpha": .9, "edgecolor": "0.7"})
        axis.set_title(f"Figure 2D · {METRIC_DISPLAY_NAMES[metric_id]}")
    else:
        figure, ids, mappings = _plot_trial_summary(group, metric_id)
        axis = figure.axes[0]
        selected = stats.loc[stats["p_fwer_maxT"] < .05, "contrast"].str.removeprefix("Trial ").astype(int)
        if len(selected):
            axis.scatter(selected, np.full(len(selected), 1.31), marker="|", s=36,
                         color="black", label="max-T FWER p<0.05")
            axis.legend(loc="upper right")
        axis.text(.02, .03, f"Exploratory Δ vs trials 10–14; max-T FWER\n"
                  f"{len(selected)}/80 trials at p<0.05; see stats table",
                  transform=axis.transAxes, fontsize=7, va="bottom",
                  bbox={"facecolor": "white", "alpha": .9, "edgecolor": "0.7"})
    return figure, ids, mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/figure2-delay"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figure2-delay/stats-review"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    for metric_id in METRIC_DISPLAY_NAMES:
        for panel in ("D", "G"):
            stem = f"figure-2{panel}_{metric_id}"
            fish_path = args.input_dir / f"{stem}_fish-data.parquet"
            group_path = args.input_dir / f"{stem}_panel-data.parquet"
            fish, group = pd.read_parquet(fish_path), pd.read_parquet(group_path)
            values, condition, labels = differences(fish, panel)
            stats = max_t_permutation(values, condition, labels)
            stats_path = args.output_dir / f"{stem}_exploratory-stats.csv"
            if stats_path.exists() and not args.overwrite:
                raise FileExistsError(stats_path)
            stats.to_csv(stats_path, index=False)
            figure, ids, mappings = render(panel, fish, group, metric_id, stats)
            base = args.output_dir / f"figure-2{panel}_delay-control_{metric_id}_stats-review"
            try:
                result = export_matplotlib_figure(
                    figure, base,
                    FigureProvenance(
                        figure_id=f"figure-2{panel}-delay-control-exploratory-stats",
                        analysis_recipe="fish-change-from-pretrain-whole-fish-maxT-permutation",
                        source_file=str(source), source_symbol="main", source_hash=sha256_file(source),
                        reproduction_snippet=f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src .venv/bin/python scripts/render_figure2_stats_review.py --input-dir '{args.input_dir}' --output-dir '{args.output_dir}' --overwrite",
                        input_artifacts=tuple({"path": str(p.resolve()), "sha256": sha256_file(p)}
                                              for p in (fish_path, group_path, stats_path)),
                        artist_mappings=mappings,
                    ), mode=FigureMode.STATIC, panel_ids=ids, overwrite=args.overwrite,
                )
            finally:
                plt.close(figure)
            print(base, stats_path, f"significant={sum(stats['p_fwer_maxT'] < .05)}")


if __name__ == "__main__":
    main()
