"""Figure 4 and supplementary renderers from persisted signed panel data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from classical_conditioning.analysis.figure4 import EXPERIMENTS, STRATA, load_figure4_analysis
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config import get_experiment_spec
from classical_conditioning.figures.export import (
    FigureExportResult, FigureMode, FigureProvenance, export_matplotlib_figure,
)
from classical_conditioning.figures.theme import apply_theme, mm_to_in


COLORS = {
    STRATA[0]: "#C2185B", STRATA[1]: "#E67E22",
    STRATA[2]: "#0077BB", STRATA[3]: "#777777",
}
STYLES = {STRATA[0]: "-", STRATA[1]: "--", STRATA[2]: "-", STRATA[3]: "--"}
NAMES = {"allDelay": "Delay", "all3sTrace": "3sTrace", "all10sTrace": "10sTrace"}


def _draw_rows(
    groups: pd.DataFrame, flow: pd.DataFrame, *, experiment_id: str,
    expected_us_s: float, kind: str, value: str, y_limit: float,
) -> tuple[plt.Figure, list[str], dict[str, dict[str, Any]]]:
    """Draw saved group medians without recomputing a trial or fish result."""
    if kind == "main":
        rows = groups.loc[groups["group_type"].isin(("block", "pooled_catch")),
                          ["group_name", "group_order"]].drop_duplicates().sort_values("group_order")
    else:
        rows = groups.loc[groups["group_type"].eq("individual_catch"),
                          ["group_name", "group_order"]].drop_duplicates().sort_values("group_order")
    expected_rows = 10 if kind == "main" else 5
    if len(rows) != expected_rows:
        raise ValueError(f"{experiment_id}: expected {expected_rows} {kind} profile groups, found {len(rows)}.")
    apply_theme()
    height_mm = 238 if kind == "main" else 142
    figure, axes = plt.subplots(expected_rows, 1, figsize=mm_to_in(183, height_mm),
                               sharex=True, sharey=True, layout="constrained")
    panel_ids: list[str] = []
    mappings: dict[str, dict[str, Any]] = {}
    spec = get_experiment_spec(experiment_id)
    counts = flow.loc[flow["experiment_id"].eq(experiment_id), "plot_stratum"].value_counts()
    if value == "signed":
        limits = (-y_limit, y_limit)
        ylabel = "Signed log vigor"
    elif value == "movement":
        limits = (0, 1)
        ylabel = "Movement probability"
    else:
        limits = (0, 1)
        ylabel = "Contributing fish fraction"
    for axis, (_, item) in zip(axes, rows.iterrows(), strict=True):
        group_name = str(item["group_name"])
        panel_id = f"{experiment_id}_{kind}_{value}_{int(item['group_order'])}"
        panel_ids.append(panel_id)
        axis.axhline(0, color="0.35", linewidth=.45, zorder=0)
        axis.axvspan(0, spec.cs_duration_s, color="#009E73", alpha=.07, linewidth=0)
        axis.axvline(0, color="#008C56", linewidth=.7)
        axis.axvline(spec.cs_duration_s, color="#008C56", linewidth=.7)
        us_line = axis.axvline(expected_us_s, color="#4A4A4A", linestyle=":", linewidth=.8)
        us_line.set_gid(f"us__{panel_id}")
        mappings[f"us__{panel_id}"] = {"role": "expected-US", "time_s": expected_us_s,
                                        "source": "verified paired-training event protocol"}
        panel = groups.loc[groups["group_name"].eq(group_name)]
        for stratum in STRATA:
            subset = panel.loc[panel["plot_stratum"].eq(stratum)].sort_values("time_s")
            n = int(counts.get(stratum, 0))
            if n == 0 or subset.empty:
                continue
            x = subset["time_s"].to_numpy(dtype=float)
            if value == "coverage":
                y = subset["signed_fish"].to_numpy(dtype=float) / n
                low = high = None
            else:
                y = subset[f"{value}_median"].to_numpy(dtype=float)
                low = subset[f"{value}_q25"].to_numpy(dtype=float)
                high = subset[f"{value}_q75"].to_numpy(dtype=float)
            line, = axis.plot(x, y, color=COLORS[stratum], linestyle=STYLES[stratum],
                              linewidth=1.15, label=f"{stratum} (n={n})")
            line_id = f"line__{panel_id}__{STRATA.index(stratum)}"
            line.set_gid(line_id)
            mappings[line_id] = {"group_name": group_name, "stratum": stratum,
                                 "x_field": "time_s", "y_field": f"{value}_median" if value != "coverage" else "signed_fish/total_group_fish",
                                 "aggregation": "trial median within fish; equal-fish median"}
            if low is not None:
                supported = subset[f"{value}_fish"].to_numpy(dtype=int) >= 2
                if np.any(supported):
                    ribbon = axis.fill_between(x, low, high, where=supported,
                                               color=COLORS[stratum], alpha=.09, linewidth=0)
                    ribbon_id = f"iqr__{panel_id}__{STRATA.index(stratum)}"
                    ribbon.set_gid(ribbon_id)
                    mappings[ribbon_id] = {"group_name": group_name, "stratum": stratum,
                                           "role": "fish IQR", "minimum_contributing_fish": 2}
        axis.set_xlim(-20, 20)
        axis.set_ylim(*limits)
        axis.set_title(group_name, loc="left", fontsize=8, pad=2)
        axis.tick_params(labelsize=6)
        if expected_us_s >= 19.95:
            axis.annotate("US 20 s", (20, limits[1]), xytext=(-2, -2),
                          textcoords="offset points", ha="right", va="top", fontsize=5)
    axes[-1].set_xlabel("Time from CS onset (s)")
    figure.supylabel(ylabel, fontsize=8)
    figure.suptitle(f"{NAMES[experiment_id]}: {value.replace('_', ' ')} by classifier group",
                    fontsize=9, y=.995)
    handles = [Line2D([0], [0], color=COLORS[stratum], linestyle=STYLES[stratum],
                      linewidth=1.15) for stratum in STRATA]
    labels = [f"{stratum} (n={int(counts.get(stratum, 0))})" for stratum in STRATA]
    handles.extend((Patch(facecolor="#009E73", alpha=.12, edgecolor="none"),
                    Line2D([0], [0], color="#4A4A4A", linestyle=":", linewidth=.8)))
    labels.extend((f"CS 0–{spec.cs_duration_s:g} s",
                   f"Expected US (paired) {expected_us_s:g} s"))
    figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=6,
                  bbox_to_anchor=(.5, .978))
    figure.get_layout_engine().set(rect=(0, 0, 1, .925))
    return figure, panel_ids, mappings


def render_figure4(
    analysis_summary: Path, *, output_dir: Path, mode: FigureMode = FigureMode.STATIC,
    overwrite: bool = False,
) -> tuple[FigureExportResult, ...]:
    """Render main and supplementary figures only from authenticated saved tables."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Figure 4 supports static PNG or publication SVG/PDF.")
    summary, tables = load_figure4_analysis(analysis_summary)
    groups = tables["group-bins"]
    flow = tables["sample-flow"]
    signed_values = groups[["signed_q25", "signed_q75"]].to_numpy(dtype=float)
    finite = np.abs(signed_values[np.isfinite(signed_values)])
    y_limit = max(.5, np.ceil(float(finite.max()) * 4) / 4) if len(finite) else .5
    inputs = (
        {"path": str(analysis_summary.resolve()), "sha256": sha256_file(analysis_summary)},
        *({"path": record["path"], "sha256": record["sha256"]}
          for record in summary["tables"].values()),
        *summary["inputs"],
    )
    output_dir = output_dir.resolve()
    source = Path(__file__).resolve()
    results = []
    for experiment_id in EXPERIMENTS:
        panel = groups.loc[groups["experiment_id"].eq(experiment_id)]
        expected_us_s = float(summary["expected_us"][experiment_id]["expected_us_s"])
        for kind, value, label in (("main", "signed", "figure-4"),
                                   ("individual", "signed", "supplement-single-catches"),
                                   ("main", "movement", "supplement-movement"),
                                   ("main", "coverage", "supplement-coverage"),
                                   ("individual", "movement", "supplement-single-catch-movement"),
                                   ("individual", "coverage", "supplement-single-catch-coverage")):
            figure, panel_ids, mappings = _draw_rows(
                panel, flow, experiment_id=experiment_id, expected_us_s=expected_us_s,
                kind=kind, value=value, y_limit=y_limit,
            )
            provenance = FigureProvenance(
                figure_id=f"{label}-{experiment_id}-{summary['metric_id']}",
                analysis_recipe=summary["recipe"], source_file=str(source),
                source_symbol="render_figure4", source_hash=sha256_file(source),
                reproduction_snippet=(f"classical-conditioning figure4-render --analysis-summary "
                                      f"'{analysis_summary.resolve()}' --output-dir '{output_dir}' "
                                      f"--mode {mode.value}"),
                input_artifacts=inputs, cohort_hash=summary["cohort_hashes"][experiment_id],
                artist_mappings=mappings,
                analysis_identity={"metric_id": summary["metric_id"],
                                   "cohort_id": summary["cohort_ids"][experiment_id],
                                   "classifier_execution_id": summary["classifier_execution_id"],
                                   "classifier_manifest_sha256": summary["classifier_manifest_sha256"],
                                   "validation_mode": summary["validation_mode"],
                                   "expected_us": summary["expected_us"][experiment_id],
                                   "scientific_status": summary["scientific_status"]},
            )
            try:
                result = export_matplotlib_figure(
                    figure, output_dir / experiment_id / f"{label}_{summary['metric_id']}",
                    provenance, mode=mode, panel_ids=panel_ids, overwrite=overwrite,
                )
                results.append(result)
            finally:
                plt.close(figure)
    return tuple(results)
