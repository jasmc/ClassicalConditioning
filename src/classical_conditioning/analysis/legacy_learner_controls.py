"""Descriptive control-flag audit for the four archived learner rules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from classical_conditioning.analysis.legacy_learners import _sha256
from classical_conditioning.exceptions import SchemaValidationError, ScientificValidationError


def control_flag_tables(results: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-condition rates and one auditable row per fish and variant."""
    rows: list[dict[str, object]] = []
    fish_rows: list[dict[str, object]] = []
    expected_keys: set[tuple[str, str]] | None = None
    for variant, frame in results.items():
        required = {"Fish_ID", "Condition", "learner_primary"}
        if required - set(frame.columns):
            raise SchemaValidationError(f"{variant} lacks {sorted(required - set(frame.columns))}")
        if frame.duplicated(["Condition", "Fish_ID"]).any():
            raise SchemaValidationError(f"{variant} has duplicate fish keys")
        keys = set(map(tuple, frame[["Condition", "Fish_ID"]].to_numpy()))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ScientificValidationError("Variant fish sets differ; rates are not directly comparable")
        for condition in ("control", "delay"):
            subset = frame.loc[frame["Condition"].eq(condition)]
            if subset.empty or subset["learner_primary"].isna().any():
                raise ScientificValidationError(f"{variant}: incomplete {condition} labels")
            flags = subset["learner_primary"].astype(bool)
            rows.append({"variant": variant, "condition": condition,
                         "fish_count": len(subset), "flagged_count": int(flags.sum()),
                         "flagged_fraction": float(flags.mean()),
                         "flagged_fish_ids": ";".join(sorted(subset.loc[flags, "Fish_ID"].astype(str)))})
            fish_rows.extend({"variant": variant, "condition": condition,
                              "fish_id": str(fish), "flagged": bool(flag)}
                             for fish, flag in zip(subset["Fish_ID"], flags, strict=True))
    return pd.DataFrame(rows), pd.DataFrame(fish_rows)


def summarize_control_flags(comparison_dir: Path) -> dict[str, object]:
    """Compare observed control flags with Delay flags and render the audit."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    comparison_dir = comparison_dir.resolve()
    report_path = comparison_dir / "comparison.json"
    report = json.loads(report_path.read_text())
    results = {}
    for item in report["variants"]:
        if item["status"] != "completed":
            raise ScientificValidationError("All variants must complete before control comparison")
        path = comparison_dir / f"{item['variant']}.parquet"
        if _sha256(path) != item["result_sha256"]:
            raise ScientificValidationError(f"{item['variant']} result changed since comparison")
        results[item["variant"]] = pd.read_parquet(path)
    rates, fish = control_flag_tables(results)
    rates_path = comparison_dir / "control-flag-rates.csv"
    fish_path = comparison_dir / "control-fish-flags.csv"
    figure_path = comparison_dir / "control-flag-comparison.png"
    pdf_path = comparison_dir / "control-flag-comparison.pdf"
    manifest_path = comparison_dir / "control-flag-comparison.json"
    if any(path.exists() for path in (rates_path, fish_path, figure_path, pdf_path, manifest_path)):
        raise FileExistsError("Control comparison outputs already exist")

    variants = list(results)
    control = rates.loc[rates["condition"].eq("control")].set_index("variant").loc[variants]
    delay = rates.loc[rates["condition"].eq("delay")].set_index("variant").loc[variants]
    flagged_controls = fish.loc[fish["condition"].eq("control") & fish["flagged"]]
    fish_ids = sorted(flagged_controls["fish_id"].unique())
    grid = np.array([
        [int(((flagged_controls["variant"] == variant) &
               (flagged_controls["fish_id"] == fish_id)).any()) for variant in variants]
        for fish_id in fish_ids
    ], dtype=int)

    fig, (ax, ax_grid) = plt.subplots(
        2, 1, figsize=(10.5, 7.0), gridspec_kw={"height_ratios": [2.2, 1.6]},
        constrained_layout=True,
    )
    x = np.arange(len(variants))
    width = 0.36
    control_bars = ax.bar(x - width / 2, control["flagged_fraction"] * 100,
                          width, color="#377eb8", label="Control")
    delay_bars = ax.bar(x + width / 2, delay["flagged_fraction"] * 100,
                        width, color="#d64a8c", label="Delay")
    for bars, counts in ((control_bars, control), (delay_bars, delay)):
        for bar, row in zip(bars, counts.itertuples(), strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.3,
                    f"{row.flagged_count}/{row.fish_count}", ha="center", fontsize=9)
    ax.set_ylim(0, max(76, float(delay["flagged_fraction"].max() * 100 + 12)))
    ax.set_ylabel("Fish flagged by learner rule (%)")
    ax.set_xticks(x, [v.removeprefix("legacy-").title() for v in variants])
    ax.legend(frameon=False, ncol=2)
    ax.set_title("In-sample legacy learner flags on the corrected Delay cohort", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax_grid.imshow(grid, aspect="auto", cmap=ListedColormap(["#f2f3f5", "#e05252"]), vmin=0, vmax=1)
    ax_grid.set_xticks(x, [v.removeprefix("legacy-").title() for v in variants])
    ax_grid.set_yticks(np.arange(len(fish_ids)), fish_ids)
    ax_grid.set_title(f"Controls flagged by at least one rule ({len(fish_ids)} of {int(control['fish_count'].iloc[0])})", loc="left")
    for i in range(len(fish_ids)):
        for j in range(len(variants)):
            if grid[i, j]:
                ax_grid.text(j, i, "●", ha="center", va="center", color="white", fontsize=15)
    ax_grid.set_xticks(np.arange(-0.5, len(variants), 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, len(fish_ids), 1), minor=True)
    ax_grid.grid(which="minor", color="white", linewidth=2)
    ax_grid.tick_params(which="minor", bottom=False, left=False)
    ax_grid.spines[:].set_visible(False)
    fig.savefig(figure_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    rates.to_csv(rates_path, index=False)
    fish.to_csv(fish_path, index=False)
    summary: dict[str, object] = {
        "schema": "legacy-control-flag-comparison/1.0",
        "analysis_mode": "descriptive_classifier_characterization",
        "comparison_sha256": _sha256(report_path),
        "metric_id": report["metric_id"], "cohort_hash": report["cohort_hash"],
        "interpretation": "Controls flagged by a rule are apparent in-sample positives, not independently estimated false-positive rates.",
        "control_fish_flagged_by_any": fish_ids,
        "control_fish_flagged_by_all": sorted(set.intersection(*[
            set(fish.loc[(fish["condition"] == "control") & (fish["variant"] == variant) & fish["flagged"], "fish_id"])
            for variant in variants
        ])),
        "artifacts": {"rates": str(rates_path), "fish": str(fish_path),
                      "png": str(figure_path), "pdf": str(pdf_path)},
    }
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
