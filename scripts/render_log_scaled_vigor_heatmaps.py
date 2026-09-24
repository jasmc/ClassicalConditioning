"""Render log-vigor, per-trial P10/P90-scaled heatmaps with managua_r."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import METRIC_COLUMNS, METRIC_DISPLAY_NAMES
from classical_conditioning.figures.export import (
    FigureMode, FigureProvenance, export_matplotlib_figure,
)
from classical_conditioning.figures.per_trial_scaled_vigor import (
    scale_per_trial_conditional_vigor, summarize_legacy_pooled_scaled_vigor,
)
from render_figure2_managua_review import render as render_pooled
from render_legacy_ssd_figure2_delay import _read_verified_recording, _verify_cohort
from render_legacy_ssd_example_heatmaps import _load_fish
from render_ssd_per_trial_example_heatmaps import _render


def _write(data: pd.DataFrame, path: Path) -> dict[str, str]:
    pq.write_table(pa.Table.from_pandas(data, preserve_index=False), path, compression="zstd")
    return {"path": str(path), "sha256": sha256_file(path)}


def _frame_scaled_example(metrics, movement, cycles, fish: str) -> pd.DataFrame:
    """Log bout means, scale each trial's frames, then bin for display."""
    import numpy as np

    absolute = metrics["AbsoluteTime"].to_numpy(dtype=np.int64)
    if np.any(np.diff(absolute) < 0):
        raise ValueError(f"Nonchronological frames: {fish}")
    detector_valid = movement["valid"].to_numpy(dtype=bool)
    moving = movement["moving"].to_numpy(dtype=bool)
    bout_ids = movement["bout_id"].to_numpy(dtype=np.int32)
    centers = np.arange(-20, 20, .5) + .25
    from classical_conditioning.figures.example_traces import METRIC_COLUMNS
    metric_arrays = {
        key: metrics[column].to_numpy(dtype=float)
        for key, column in METRIC_COLUMNS.items()
    }
    rows = []
    for trial in range(5, 95):
        onset = int(cycles.iloc[trial - 1]["Beg"])
        start = np.searchsorted(absolute, onset - 20000, side="left")
        stop = np.searchsorted(absolute, onset + 20000, side="left")
        times = (absolute[start:stop] - onset) / 1000.0
        indices = np.floor((times + 20) / .5).astype(np.int32)
        in_window = (indices >= 0) & (indices < 80)
        usable_detector = (detector_valid[start:stop]
                           & moving[start:stop]
                           & (bout_ids[start:stop] > 0))
        trial_bouts = bout_ids[start:stop]
        for metric_id, full in metric_arrays.items():
            raw = full[start:stop]
            usable = usable_detector & np.isfinite(raw) & (raw > 0)
            bout_mean = np.full(len(raw), np.nan)
            if usable.any():
                frame = pd.DataFrame({"bout": trial_bouts[usable], "value": raw[usable]})
                bout_mean[usable] = frame.groupby("bout", sort=False)["value"].transform("mean").to_numpy()
            log_vigor = np.log(bout_mean)
            baseline = log_vigor[(times >= -20) & (times < 0) & np.isfinite(log_vigor)]
            if len(baseline) >= 3:
                low, high = np.quantile(baseline, [.1, .9])
            else:
                low, high = np.nan, np.nan
            scaled = np.full(len(raw), np.nan)
            if np.isfinite(low) and np.isfinite(high) and high > low:
                scaled[usable] = np.clip((log_vigor[usable] - low) / (high - low), 0, 1)
            finite = in_window & np.isfinite(scaled)
            count = np.bincount(indices[finite], minlength=80)
            total = np.bincount(indices[finite], weights=scaled[finite], minlength=80)
            binned = np.divide(total, count, out=np.full(80, np.nan), where=count > 0)
            rows.extend({
                "Recording ID": fish, "Metric ID": metric_id,
                "Trial number": trial, "Time bin center (s)": float(center),
                "Per-trial scaled vigor": float(value),
                "Baseline log P10": float(low), "Baseline log P90": float(high),
                "Baseline moving frames": len(baseline),
                "Contributing frames": int(n),
                "Vigor transform": "log bout mean before frame scaling",
                "Signal semantics": "bout_mean_log_frame_scaled_per_trial_pre_minus20_to_0_then_binned",
            } for center, value, n in zip(centers, binned, count))
    return pd.DataFrame(rows)


def _export(fig, base, figure_id, panels, mappings, inputs, cohort_hash, snippet):
    source = Path(__file__).resolve()
    try:
        result = export_matplotlib_figure(
            fig, base,
            FigureProvenance(
                figure_id=figure_id,
                analysis_recipe="log-positive-conditional-vigor-fish-trial-preCS-P10-P90",
                source_file=str(source), source_symbol="main",
                source_hash=sha256_file(source),
                reproduction_snippet=snippet,
                input_artifacts=tuple(inputs), cohort_hash=cohort_hash,
                artist_mappings=mappings,
            ),
            mode=FigureMode.STATIC, panel_ids=panels, overwrite=True,
        )
    finally:
        plt.close(fig)
    for path in (*result.outputs, result.sidecar):
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--figure1-only", action="store_true")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    root = args.output_root.resolve()
    source = Path(__file__).resolve()
    scaling = Path(scale_per_trial_conditional_vigor.__code__.co_filename).resolve()
    snippet = (
        f"MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src .venv/bin/python "
        f"scripts/render_log_scaled_vigor_heatmaps.py --project-dir '{project}' "
        f"--output-root '{root}'"
        f"{' --figure1-only' if args.figure1_only else ''}"
    )

    example_output = root / "figure1-heatmaps" / "log-managua-r"
    example_output.mkdir(parents=True, exist_ok=True)
    example_profiles, example_inputs = [], []
    for fish in ("20221115_07", "20221115_09"):
        metrics, movement, cycles, verified = _load_fish(project, fish)
        example_profiles.append(_frame_scaled_example(metrics, movement, cycles, fish))
        example_inputs.extend(
            {"path": str(path), "sha256": sha256_file(path)} for path in verified
        )
    example = pd.concat(example_profiles, ignore_index=True)
    example_input = _write(example, example_output / "figure-1-log-scaled-vigor_panel-data.parquet")
    for metric_id in METRIC_COLUMNS:
        fig, panels, mappings = _render(example, metric_id, cmap_name="managua_r")
        fig.suptitle(f"Figure 1E log vigor scaled per trial · {METRIC_DISPLAY_NAMES[metric_id]}")
        fig.axes[-1].set_ylabel("Log-vigor scaled per trial (0–1)")
        base = example_output / f"figure-1-E_delay-control_{metric_id}_log-managua-r"
        _export(fig, base, "figure-1-E-log-scaled-vigor", panels, mappings,
                [*example_inputs, example_input,
                 {"path": str(scaling), "sha256": sha256_file(scaling)}], None, snippet)

    if args.figure1_only:
        return

    pooled_output = root / "figure2-delay" / "log-managua-r"
    pooled_output.mkdir(parents=True, exist_ok=True)
    cohort, cohort_hash, pooled_inputs = _verify_cohort(project)
    profiles = []
    for _, row in cohort.iterrows():
        profile, _, verified = _read_verified_recording(project, row)
        profiles.append(profile)
        pooled_inputs.extend(verified)
    scaled = scale_per_trial_conditional_vigor(
        pd.concat(profiles, ignore_index=True), transform="log", clip=False,
    )
    by_recording = cohort.set_index("recording_id")["condition_id"].to_dict()
    for metric_id in METRIC_COLUMNS:
        pooled = summarize_legacy_pooled_scaled_vigor(
            scaled, metric_id=metric_id, condition_by_recording=by_recording,
        )
        pooled["Vigor transform"] = "log before fish/trial scaling"
        pooled_input = _write(
            pooled, pooled_output / f"figure-2A_{metric_id}_log-panel-data.parquet",
        )
        fig, mappings = render_pooled(pooled, metric_id)
        fig.suptitle(f"Figure 2A pooled log vigor scaled per trial · {METRIC_DISPLAY_NAMES[metric_id]}")
        fig.axes[-1].set_ylabel("Pooled log-vigor scaled per trial (0–1)")
        base = pooled_output / f"figure-2A_delay-control_{metric_id}_log-managua-r"
        _export(fig, base, "figure-2A-log-scaled-vigor", ["control", "delay", "vigor_colorbar"],
                mappings, [*pooled_inputs, pooled_input,
                           {"path": str(scaling), "sha256": sha256_file(scaling)}],
                cohort_hash, snippet)


if __name__ == "__main__":
    main()
