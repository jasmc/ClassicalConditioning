"""Descriptive fish-level suppression comparison with paired-metric bootstrap."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from classical_conditioning.external_artifacts import external_output
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = external_output("outputs/pre-vs-late-train-early-test-review")
METRICS = ["tail_length_weighted_angular_l1", "whole_tail_xy_mean_speed_normalized", "legacy_distal_angular_speed"]
LABELS = ["Angular L1", "Whole-tail XY", "Legacy distal"]

def main():
    d = pd.read_csv(OUT / "fish_paired_change.csv")
    d["suppression"] = -d.late_minus_pre_log2_ratio
    wide = d.pivot(index=["fish_id", "condition_id"], columns="metric_id", values="suppression")[METRICS]
    assert len(wide) == 57 and not wide.isna().any().any()
    delay = wide.xs("delay", level="condition_id").to_numpy()
    control = wide.xs("control", level="condition_id").to_numpy()
    rng = np.random.default_rng(20261006)
    # Resample fish within each condition, preserving each fish's three metrics together.
    draws = delay[rng.integers(len(delay), size=(10000, len(delay)))].mean(axis=1) - control[rng.integers(len(control), size=(10000, len(control)))].mean(axis=1)
    effect = delay.mean(axis=0) - control.mean(axis=0)
    bounds = np.quantile(draws, [0.025, 0.975], axis=0)
    rows = []
    for i, metric in enumerate(METRICS):
        rows.append(dict(metric_id=metric, delay_mean=delay[:, i].mean(), control_mean=control[:, i].mean(),
                         mean_delay_minus_control=effect[i], ci_low=bounds[0, i], ci_high=bounds[1, i],
                         median_delay_minus_control=np.median(delay[:, i])-np.median(control[:, i]),
                         equivalent_reduction_percent=100*(1-2**(-effect[i]))))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "metric_response_magnitude.csv", index=False)
    differences = {}
    for a, b in [(1, 0), (2, 0), (1, 2)]:
        differences[f"{LABELS[a]} minus {LABELS[b]}"] = {"estimate": float(effect[a]-effect[b]),
            "ci": np.quantile(draws[:, a]-draws[:, b], [0.025, 0.975]).tolist()}
    (OUT / "metric_response_magnitude.json").write_text(json.dumps({"bootstrap_replicates": 10000,
        "seed": 20261006, "metric_differences": differences}, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    jitter = np.random.default_rng(17)
    for i in range(3):
        for vals, offset, color, label in [(delay[:, i], -0.16, "#1f77b4", "Delay (29 fish)"),
                                         (control[:, i], 0.16, "#d17c27", "Control (28 fish)")]:
            x = i + offset
            axes[0].scatter(x+jitter.uniform(-0.075, 0.075, len(vals)), vals, s=23, alpha=.65,
                            color=color, label=label if i == 0 else None)
            axes[0].plot([x-.10, x+.10], [vals.mean()]*2, color="black", lw=2)
    axes[0].axhline(0, color="0.6", ls="--", lw=1)
    axes[0].set_xticks(range(3), LABELS)
    axes[0].set_ylabel("Suppression after training (log₂ units)\nlog₂(Pre CS/baseline) − log₂(late CS/baseline)")
    axes[0].set_title("Each dot is one fish; black bars are means")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].errorbar(effect, range(3), xerr=np.vstack([effect-bounds[0], bounds[1]-effect]),
                     fmt="o", color="#333333", capsize=4, ms=7)
    axes[1].axvline(0, color="0.6", ls="--", lw=1)
    axes[1].set_yticks(range(3), LABELS)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean suppression: Delay minus control\n(log₂ units; 95% fish-bootstrap intervals)")
    axes[1].set_title("Larger values = larger group difference")
    fig.suptitle("Response magnitude: Pre-Train versus last 5 Train + first 5 Test", fontsize=13)
    fig.savefig(OUT / "metric_response_magnitude.png", dpi=180)
    fig.savefig(OUT / "metric_response_magnitude.pdf")
    print(summary.round(4).to_string(index=False))
    print(json.dumps(differences, indent=2))

if __name__ == "__main__":
    main()
