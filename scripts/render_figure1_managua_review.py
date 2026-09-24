"""Render Figure 1 scaled-vigor heatmap palette variants from saved panel data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import METRIC_COLUMNS
from classical_conditioning.figures.export import (
    FigureMode, FigureProvenance, export_matplotlib_figure,
)
from render_ssd_per_trial_example_heatmaps import _render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "publication"), default="static")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source_data = args.panel_data.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pq.read_table(source_data).to_pandas()
    required = {"Recording ID", "Trial number", "Time bin center (s)",
                "Metric ID", "Per-trial scaled vigor"}
    if required.difference(data.columns):
        raise ValueError(f"Missing panel fields: {required.difference(data.columns)}")
    source = Path(__file__).resolve()
    renderer = Path(_render.__code__.co_filename).resolve()
    snippet = (
        f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src .venv/bin/python "
        f"scripts/render_figure1_managua_review.py --panel-data '{source_data}' "
        f"--output-dir '{output}' --mode {args.mode} --overwrite"
    )
    for metric_id in METRIC_COLUMNS:
        figure, panels, mappings = _render(data, metric_id, cmap_name="managua_r")
        base = output / f"figure-1-E_delay-control_{metric_id}_managua-r"
        try:
            result = export_matplotlib_figure(
                figure, base,
                FigureProvenance(
                    figure_id="figure-1-E-delay-control-managua-r",
                    analysis_recipe="saved-per-trial-scaled-vigor-palette-review",
                    source_file=str(source), source_symbol="main",
                    source_hash=sha256_file(source),
                    reproduction_snippet=snippet,
                    input_artifacts=(
                        {"path": str(source_data), "sha256": sha256_file(source_data)},
                        {"path": str(renderer), "sha256": sha256_file(renderer)},
                    ),
                    artist_mappings=mappings,
                ),
                mode=FigureMode(args.mode), panel_ids=panels,
                overwrite=args.overwrite,
            )
        finally:
            plt.close(figure)
        for path in (*result.outputs, result.sidecar):
            print(path)


if __name__ == "__main__":
    main()
