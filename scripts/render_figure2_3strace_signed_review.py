"""Render the 3sTrace/control Figure 2B signed heatmap and fish coverage review."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.analysis.figure4 import _read_recording, verify_expected_us
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.cohort import load_cohort_manifest, logical_cohort_hash
from classical_conditioning.figures.export import (
    FigureMode, FigureProvenance, export_matplotlib_figure,
)
from classical_conditioning.figures.signed_bout_heatmap import summarize_equal_fish_signed_log_vigor
from classical_conditioning.figures.theme import apply_theme, heatmap_cmap, mm_to_in, style_axes


TRIALS = np.arange(5, 95)
TIMES = np.arange(-20.0, 20.0, 0.5) + 0.25


def _plot(data: pd.DataFrame, counts: dict[str, int], metric_id: str, *, coverage: bool):
    theme = apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=mm_to_in(183, 95), sharex=True,
                             sharey=True, constrained_layout=True)
    mappings = {}
    image = None
    field = "Fish coverage fraction" if coverage else "Mean signed log vigor"
    for axis, condition in zip(axes, ("control", "trace")):
        subset = data.loc[data["condition_id"].eq(condition)]
        matrix = subset.pivot(index="Trial number", columns="Time bin center (s)",
                              values=field).reindex(index=TRIALS, columns=TIMES)
        if matrix.isna().all().all():
            raise ValueError(f"No {field} values for {condition}")
        cmap = ("managua_r" if coverage else
                heatmap_cmap(theme.single_fish_scaled_vigor_cmap, theme))
        if not coverage:
            cmap.set_bad("black")
        image = axis.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            origin="upper", aspect="auto", interpolation="nearest",
            extent=(-20, 20, 94.5, 4.5), cmap=cmap,
            vmin=0 if coverage else theme.single_fish_scaled_vigor_vmin,
            vmax=1 if coverage else theme.single_fish_scaled_vigor_vmax,
            rasterized=True,
        )
        gid = f"heatmap__{condition}__{'coverage' if coverage else 'signed_log_vigor'}"
        image.set_gid(gid)
        mappings[gid] = {
            "condition_id": condition, "metric_id": metric_id,
            "value_field": field, "aggregation": "equal-fish mean per trial and bin",
            "total_fish": counts[condition],
        }
        axis.set_title(f"{'Control' if condition == 'control' else '3sTrace'} (n={counts[condition]})")
        for seconds in (0, 10):
            axis.axvline(seconds, color=theme.cs_color, linewidth=0.7)
        if condition == "trace":
            axis.axvline(13, color=theme.us_color, linewidth=0.7, linestyle="--")
        for boundary in (14.5, 64.5):
            axis.axhline(boundary, color="white", linewidth=0.7)
        axis.set_xlim(-20, 20)
        axis.set_ylim(94.5, 4.5)
        style_axes(axis, theme=theme, show_xticks=True,
                   show_yticks=condition == "control")
        axis.set_xlabel("Time relative to CS onset (s)")
    axes[0].set_ylabel("Global CS trial")
    assert image is not None
    bar = fig.colorbar(image, ax=axes.tolist(), fraction=0.025, pad=0.025)
    bar.set_label("Contributing fish fraction" if coverage else
                  "Mean signed log vigor relative to fish/trial pre-CS median")
    fig.suptitle("Figure 2B review · 3sTrace and matched controls")
    return fig, ["control", "trace", "colorbar"], mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--metric", default="tail_length_weighted_angular_l1",
                        choices=("tail_length_weighted_angular_l1",
                                 "whole_tail_xy_mean_speed_normalized",
                                 "legacy_distal_angular_speed"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = (args.output_dir or project / "Figures" / "PNG" / "Analyses"
              / "figure2-3strace-review").resolve()
    cohort = load_cohort_manifest(project, args.cohort_id)
    selected = cohort.loc[cohort["primary_included"].eq(True)].copy()
    if selected.empty or set(selected["experiment_id"]) != {"all3sTrace"} or set(selected["condition_id"]) != {"control", "trace"}:
        raise ValueError("A frozen 3sTrace/control primary cohort is required")
    counts = selected.groupby("condition_id")["fish_id"].nunique().to_dict()
    cohort_hash = logical_cohort_hash(cohort)
    manifest_path = project / "Processed data" / "Cohorts" / args.cohort_id / "cohort-manifest.parquet"
    inputs = [{"path": str(manifest_path), "sha256": sha256_file(manifest_path)}]
    fish_frames = []
    for row in selected.itertuples(index=False):
        bins, protocol, _, record_inputs = _read_recording(
            project, str(row.recording_id), args.metric, "tail-candidate-corrected",
            include_unmasked=True,
        )
        if row.condition_id == "trace":
            verify_expected_us(protocol, "all3sTrace")
        fish_frames.append(pd.DataFrame({
            "Recording ID": bins["recording_id"],
            "Metric ID": args.metric,
            "Trial number": bins["trial_number"],
            "Time bin center (s)": bins["time_s"],
            "Signed log vigor": bins["raw_signed_log_vigor"],
            "Baseline start (s)": -20.0,
            "Baseline end (s)": 0.0,
        }))
        inputs.extend(record_inputs)
    pooled = summarize_equal_fish_signed_log_vigor(
        pd.concat(fish_frames, ignore_index=True), metric_id=args.metric,
        condition_by_recording=selected.set_index("recording_id")["condition_id"].to_dict(),
    )
    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / f"figure-2B-3strace_{args.metric}_panel-data.parquet"
    if panel_path.exists() and not args.overwrite:
        raise FileExistsError(panel_path)
    pq.write_table(pa.Table.from_pandas(pooled, preserve_index=False), panel_path,
                   compression="zstd")
    inputs.append({"path": str(panel_path), "sha256": sha256_file(panel_path)})
    source = Path(__file__).resolve()
    for coverage, suffix in ((False, "signed"), (True, "coverage")):
        fig, panels, mappings = _plot(pooled, counts, args.metric, coverage=coverage)
        try:
            result = export_matplotlib_figure(
                fig, output / f"figure-2B-3strace_{suffix}_{args.metric}",
                FigureProvenance(
                    figure_id=f"figure-2B-3strace-{suffix}-review",
                    analysis_recipe="equal-fish-signed-bout-log-vigor-pre20-median-0p5s",
                    source_file=str(source), source_symbol="main",
                    source_hash=sha256_file(source),
                    reproduction_snippet=(
                        f"python scripts/render_figure2_3strace_signed_review.py "
                        f"--project-dir '{project}' --cohort-id {args.cohort_id} "
                        f"--metric {args.metric} --mode {args.mode}"
                    ),
                    input_artifacts=tuple(inputs), cohort_hash=cohort_hash,
                    artist_mappings=mappings,
                    analysis_identity={"experiment_id": "all3sTrace", "metric_id": args.metric,
                                       "cohort_id": args.cohort_id, "cohort_hash": cohort_hash,
                                       "scientific_status": "descriptive_review"},
                ),
                mode=FigureMode(args.mode), panel_ids=panels,
                overwrite=args.overwrite,
            )
        finally:
            plt.close(fig)
        for path in (*result.outputs, result.sidecar):
            print(path)
    print(panel_path)


if __name__ == "__main__":
    main()
