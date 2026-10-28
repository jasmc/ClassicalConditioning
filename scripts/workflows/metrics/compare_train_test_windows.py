"""Compare Pre-Train with pooled late-Train/early-Test CS trials for every fish."""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]


from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(_COMMAND_REPOSITORY / "src"))
from classical_conditioning.analysis.review.metric_comparison_storage import workspace, require_new_outputs, bind_inputs, RECEIPT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\Digested Data\allDelay-full-v1\Processed data")

METRICS = {
    "tail_length_weighted_angular_l1": "Weighted angular L1",
    "whole_tail_xy_mean_speed_normalized": "Whole-tail XY speed",
    "legacy_distal_angular_speed": "Legacy distal speed",
}
TRIALS = {"Pre": range(5, 15), "LateTrainEarlyTest": range(60, 70)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    out = workspace(args.output_dir)
    require_new_outputs(out, ["fish_window_summary.csv", "fish_paired_change.csv", "all_fish_pre_vs_late_train_early_test.png", "all_fish_pre_vs_late_train_early_test.pdf", RECEIPT])
    paths = sorted(args.processed_dir.glob("*/candidate-trial-outcomes-corrected-v1.parquet"))
    if not paths:
        raise RuntimeError(f"No processed trial outcomes found beneath {args.processed_dir}")
    cols = ["fish_id", "condition_id", "alignment", "phase", "trial_number", "metric_id",
            "baseline_total_activity", "response_total_activity", "baseline_conditional_intensity", "conditional_intensity"]
    data = pd.concat((pd.read_parquet(p, columns=cols) for p in paths), ignore_index=True)
    from classical_conditioning.analysis.bout_vigor import bout_only_trial_outcomes
    data = bout_only_trial_outcomes(data)
    data = data[data.alignment.eq("CS") & data.metric_id.isin(METRICS)].copy()
    data["window"] = np.where(data.phase.eq("Pre") & data.trial_number.between(5, 14), "Pre",
                              np.where((data.phase.eq("Train") & data.trial_number.between(60, 64))
                                       | (data.phase.eq("Test") & data.trial_number.between(65, 69)),
                                       "LateTrainEarlyTest", ""))
    data = data[data.window.ne("")]
    rows = []
    for (fish, condition, metric, phase), group in data.groupby(
        ["fish_id", "condition_id", "metric_id", "window"], sort=True
    ):
        valid = group[np.isfinite(group.baseline_total_activity)
                      & np.isfinite(group.response_total_activity)
                      & (group.baseline_total_activity > 0)
                      & (group.response_total_activity > 0)]
        if len(valid) != 10 or set(valid.trial_number) != set(TRIALS[phase]):
            continue
        base = valid.baseline_total_activity.mean()
        response = valid.response_total_activity.mean()
        rows.append({"fish_id": fish, "condition_id": condition, "metric_id": metric,
                     "phase": phase, "n_trials": len(valid), "baseline_mean": base,
                     "response_mean": response, "cs_baseline_ratio": response / base,
                     "log2_cs_baseline_ratio": np.log2(response / base)})
    summary = pd.DataFrame(rows)
    wide = summary.pivot(index=["fish_id", "condition_id", "metric_id"], columns="phase",
                         values="log2_cs_baseline_ratio").dropna().reset_index()
    wide["late_minus_pre_log2_ratio"] = wide.LateTrainEarlyTest - wide.Pre
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "fish_window_summary.csv", index=False)
    wide.to_csv(out / "fish_paired_change.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    colors = {"delay": "#1f77b4", "control": "#d17c27"}
    extent = max(0.5, np.ceil(np.nanmax(np.abs(wide[["Pre", "LateTrainEarlyTest"]].to_numpy())) * 2) / 2)
    for ax, (metric, label) in zip(axes, METRICS.items()):
        part = wide[wide.metric_id.eq(metric)]
        ax.plot([-extent, extent], [-extent, extent], color="0.55", lw=1, ls="--")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        for condition in ("delay", "control"):
            subset = part[part.condition_id.eq(condition)]
            ax.scatter(subset.Pre, subset.LateTrainEarlyTest, s=31, alpha=0.75,
                       color=colors[condition], label=f"{condition.title()} (n={len(subset)})")
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_xlabel("Pre-Train (10 trials): log₂(CS / baseline)")
    axes[0].set_ylabel("Late Train + early Test (5 + 5): log₂(CS / baseline)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Pre-Train versus pooled late Train + early Test", fontsize=14)
    fig.savefig(out / "all_fish_pre_vs_late_train_early_test.png")
    fig.savefig(out / "all_fish_pre_vs_late_train_early_test.pdf")
    print(f"Processed {len(paths)} fish; paired rows by metric: {wide.groupby('metric_id').size().to_dict()}")
    print(wide.groupby(["metric_id", "condition_id"])["late_minus_pre_log2_ratio"].median().to_string())
    bind_inputs(out)
    print(out)


if __name__ == "__main__":
    main()
