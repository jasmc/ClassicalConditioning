"""Render the historical 3sTrace example fish as a signed Figure 1F review heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_conditioning.analysis.figure4 import _read_recording, verify_expected_us
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.export import (
    FigureMode, FigureProvenance, export_matplotlib_figure,
)
from classical_conditioning.figures.theme import apply_theme, heatmap_cmap, mm_to_in


PHASES = (("Pre-Train", 5, 14), ("Train", 15, 64), ("Test", 65, 94))
TRIAL_MARKERS = (9, 17, 63, 66, 93)
TIME_CENTERS = np.arange(-20.0, 20.0, 0.5) + 0.25


def render(bins: pd.DataFrame, recording_id: str, metric_id: str, us_latency_s: float):
    theme = apply_theme()
    fig, axes = plt.subplots(
        3, 1, figsize=mm_to_in(90, 170),
        gridspec_kw={"height_ratios": [10, 50, 30]},
        sharex=True, constrained_layout=True,
    )
    cmap = heatmap_cmap(theme.single_fish_scaled_vigor_cmap, theme)
    cmap.set_bad("black")
    mappings = {}
    image = None
    for axis, (phase, first, last) in zip(axes, PHASES):
        subset = bins.loc[bins["trial_number"].between(first, last)]
        if subset.duplicated(["trial_number", "time_s"]).any():
            raise ValueError(f"Duplicate signed bins for {recording_id} {phase}")
        matrix = subset.pivot(
            index="trial_number", columns="time_s", values="raw_signed_log_vigor"
        ).reindex(index=range(first, last + 1), columns=TIME_CENTERS)
        if matrix.isna().all().all():
            raise ValueError(f"No signed bins for {recording_id} {phase}")
        panel = phase.lower().replace("-", "_")
        image = axis.imshow(
            np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
            aspect="auto", interpolation="nearest", origin="upper",
            extent=(-20, 20, last - first + 1, 0), cmap=cmap,
            vmin=theme.single_fish_scaled_vigor_vmin,
            vmax=theme.single_fish_scaled_vigor_vmax, rasterized=True,
        )
        image_id = f"heatmap__3strace_{panel}__{metric_id}"
        image.set_gid(image_id)
        mappings[image_id] = {
            "recording_id": recording_id, "phase": phase, "metric_id": metric_id,
            "signal": "signed_bout_log_vigor_pre20_median",
            "trial_start": first, "trial_end": last,
            "missing_bout_bins": "NaN, displayed black",
        }
        for seconds in (0, 10):
            axis.axvline(seconds, color=theme.cs_color, linewidth=0.75)
        if phase == "Train":
            axis.axvline(us_latency_s, color=theme.us_color, linewidth=0.75,
                         linestyle="--")
        for trial in TRIAL_MARKERS:
            if first <= trial <= last:
                axis.plot(20.35, trial - first + 0.5, marker="<", color="black",
                          markersize=3.5, clip_on=False)
        axis.set_xlim(-20, 20)
        axis.set_ylim(last - first + 1, 0)
        axis.set_yticks([])
        axis.set_ylabel(phase)
    axes[-1].set_xlabel("Time relative to CS onset (s)")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes.tolist(), fraction=0.04, pad=0.03)
    colorbar.set_label("Signed log vigor relative to pre-CS median")
    fig.suptitle(f"3sTrace example · {recording_id}")
    return fig, [name.lower().replace("-", "_") for name, _, _ in PHASES] + ["colorbar"], mappings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--recording-id", default="20230307_12")
    parser.add_argument("--metric", default="tail_length_weighted_angular_l1",
                        choices=("tail_length_weighted_angular_l1",
                                 "whole_tail_xy_mean_speed_normalized",
                                 "legacy_distal_angular_speed"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output = (args.output_dir or project / "Figures" / "PNG" / args.recording_id).resolve()
    bins, protocol, _, inputs = _read_recording(
        project, args.recording_id, args.metric, "tail-candidate-corrected",
        include_unmasked=True,
    )
    us_latency_s, paired_trials = verify_expected_us(protocol, "all3sTrace")
    fig, panels, mappings = render(bins, args.recording_id, args.metric, us_latency_s)
    source = Path(__file__).resolve()
    try:
        result = export_matplotlib_figure(
            fig, output / f"figure-1-F-3strace_{args.metric}",
            FigureProvenance(
                figure_id="figure-1-F-3strace-example-heatmap",
                analysis_recipe="signed-bout-log-vigor-pre20-median-0p5s",
                source_file=str(source), source_symbol="main",
                source_hash=sha256_file(source),
                reproduction_snippet=(
                    f"python scripts/render_figure1_3strace_example.py "
                    f"--project-dir '{project}' --recording-id {args.recording_id} "
                    f"--metric {args.metric} --mode {args.mode}"
                ),
                input_artifacts=tuple(inputs),
                artist_mappings=mappings,
                analysis_identity={
                    "recording_id": args.recording_id,
                    "metric_id": args.metric,
                    "experiment_id": "all3sTrace",
                    "verified_expected_us_s": us_latency_s,
                    "paired_training_trials": paired_trials,
                },
            ),
            mode=FigureMode(args.mode), panel_ids=panels,
            overwrite=args.overwrite,
        )
    finally:
        plt.close(fig)
    for path in (*result.outputs, result.sidecar):
        print(path)


if __name__ == "__main__":
    main()
