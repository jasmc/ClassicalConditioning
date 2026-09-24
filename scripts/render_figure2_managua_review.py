"""Render palette-only Figure 2A comparisons from saved, already scaled panel data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import METRIC_DISPLAY_NAMES
from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
from classical_conditioning.figures.theme import apply_theme, mm_to_in, style_axes


def render(data: pd.DataFrame, metric_id: str):
    theme = apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=mm_to_in(183, 95), sharex=True,
                             sharey=True, constrained_layout=True)
    trials = np.arange(5, 95)
    times = np.sort(data["Time bin center (s)"].unique())
    if len(times) != 80 or not np.allclose(np.diff(times), 0.5):
        raise ValueError("Expected the original 80 half-second bins")
    extent = (float(times[0] - .25), float(times[-1] + .25), 94.5, 4.5)
    mappings = {}
    for axis, condition, count in zip(axes, ("control", "delay"), (28, 29)):
        subset = data.loc[data["condition_id"].eq(condition)]
        matrix = subset.pivot(index="Trial number", columns="Time bin center (s)",
                              values="Mean per-trial scaled vigor").reindex(index=trials, columns=times)
        if matrix.shape != (90, 80):
            raise ValueError("Pooled heatmap matrix has unexpected shape")
        image = axis.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
                            origin="upper", aspect="auto", interpolation="nearest",
                            extent=extent, vmin=0, vmax=1, cmap="managua_r", rasterized=True)
        gid = f"heatmap__{condition}__managua_r"
        image.set_gid(gid)
        mappings[gid] = {"condition": condition, "metric_id": metric_id,
                         "value_field": "Mean per-trial scaled vigor", "cmap": "managua_r",
                         "vmin": "0", "vmax": "1"}
        axis.set_title(f"{condition.capitalize()} (n={count})")
        axis.set_xlabel("Time relative to CS onset (s)")
        for seconds in (0, 10):
            axis.axvline(seconds, color=theme.cs_color, linewidth=.7)
        for boundary in (14.5, 64.5):
            axis.axhline(boundary, color="white", linewidth=.7)
        axis.set_xlim(-20, 20)
        axis.set_ylim(94.5, 4.5)
        style_axes(axis, theme=theme, show_xticks=True, show_yticks=condition == "control")
    axes[0].set_ylabel("Global CS trial")
    fig.colorbar(image, ax=axes, fraction=.025, pad=.025,
                 label="Pooled per-trial scaled vigor (0–1)")
    fig.suptitle(f"Figure 2A managua_r review · {METRIC_DISPLAY_NAMES[metric_id]}")
    return fig, mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/figure2-delay"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figure2-delay/managua-r-review"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    for metric_id in METRIC_DISPLAY_NAMES:
        data_path = args.input_dir / f"figure-2A_{metric_id}_panel-data.parquet"
        data = pd.read_parquet(data_path)
        fig, mappings = render(data, metric_id)
        base = args.output_dir / f"figure-2A_delay-control_{metric_id}_managua-r"
        try:
            result = export_matplotlib_figure(
                fig, base,
                FigureProvenance(
                    figure_id="figure-2A-delay-control-managua-r-review",
                    analysis_recipe="palette-only-from-corrected-two-stage-pooled-scaling",
                    source_file=str(source), source_symbol="render", source_hash=sha256_file(source),
                    reproduction_snippet=f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src .venv/bin/python scripts/render_figure2_managua_review.py --input-dir '{args.input_dir}' --output-dir '{args.output_dir}' --overwrite",
                    input_artifacts=({"path": str(data_path.resolve()), "sha256": sha256_file(data_path)},),
                    artist_mappings=mappings,
                ), mode=FigureMode.STATIC, panel_ids=["control", "delay", "vigor_colorbar"],
                overwrite=args.overwrite,
            )
        finally:
            plt.close(fig)
        for path in (*result.outputs, result.sidecar):
            print(path)


if __name__ == "__main__":
    main()
