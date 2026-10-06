"""Compare raw and signed bout vigor with exactly the same frame support."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_figure1_vigor_alignment import ROOT, OUT as AUDIT, TIME, digest

OUT = ROOT / "audit-vigor-same-frames-20261006"
TRIALS = (9, 17, 63, 66, 93)
STAGES = ("Pre-Train", "Early Train", "Late Train", "Early Test", "Late Test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=AUDIT/'joined_frames.parquet')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--bout-only', action='store_true',
                        help='Show only the bout signal actually binned, with matched signed axes')
    args = parser.parse_args()
    evidence = json.loads((AUDIT / "audit.json").read_text())
    for path, expected in evidence["verified_hashes"].items():
        assert digest(path) == expected, path
    frames = pd.read_parquet(args.source)
    heat_path = ROOT / "heatmaps/Fig1_PanelF_Delay_legacy-vigor_v2.parquet"
    heat = pd.read_parquet(heat_path)
    assert heat["Baseline start (s)"].eq(-15).all()
    assert heat["Baseline end (s)"].eq(0).all()
    frames["raw_on_bout_frames"] = frames.Vigor.where(frames.eligible)
    # Retain individual centred log values as well as the existing bout summary.
    frames["scaled_on_bout_frames"] = frames.centred_log.where(frames.eligible)
    frames["signed_bout_on_bout_frames"] = frames.bout_median.where(frames.eligible)
    raw_support = np.isfinite(frames.raw_on_bout_frames)
    np.testing.assert_array_equal(raw_support, np.isfinite(frames.scaled_on_bout_frames))
    np.testing.assert_array_equal(raw_support, np.isfinite(frames.signed_bout_on_bout_frames))
    np.testing.assert_array_equal(raw_support, frames.eligible)
    # Raw values on accepted frames are unaltered.
    np.testing.assert_array_equal(frames.loc[raw_support, "raw_on_bout_frames"],
                                  frames.loc[raw_support, "Vigor"])
    comparisons = []
    plt.rcParams.update({"svg.fonttype": "none", "path.simplify": False})
    fig, axes = plt.subplots(5, 3, figsize=(14, 9), sharex=True,
                             sharey="col", layout="constrained")
    for row, (trial, stage) in enumerate(zip(TRIALS, STAGES)):
        part = frames.loc[frames["Trial number"].eq(trial)]
        stored = heat.loc[heat["Trial number"].eq(trial)].sort_values("Time bin center (s)")
        rebuilt = part.groupby("bin_index").signed_bout_on_bout_frames.mean().reindex(range(80))
        np.testing.assert_allclose(rebuilt, stored["Signed log vigor"], atol=1e-12,
                                   rtol=0, equal_nan=True)
        axes[row, 0].plot(part[TIME], part.raw_on_bout_frames, color="black", lw=.45)
        if not args.bout_only:
            axes[row, 1].plot(part[TIME], part.scaled_on_bout_frames,
                              color="#c85a17", lw=.35, alpha=.5)
        axes[row, 1].plot(part[TIME], part.signed_bout_on_bout_frames,
                          color="#713600", lw=.8)
        for center, value in zip(stored["Time bin center (s)"], stored["Signed log vigor"]):
            if np.isfinite(value):
                axes[row, 2].bar(center-.25, value, width=.5, align="edge",
                                  color="#c85a17", alpha=.7)
            else:
                axes[row, 2].axvspan(center-.25, center+.25, color="#dddddd", lw=0)
        for col in range(3):
            axes[row, col].axvline(0, color="#0d7f3c", lw=.8)
            axes[row, col].axvline(10, color="#0d7f3c", lw=.8, ls="--")
            axes[row, col].set_xlim(-20, 20)
            axes[row, col].spines[["top", "right"]].set_visible(False)
            axes[row, col].tick_params(labelsize=8)
            if col:
                axes[row, col].axhline(0, color="grey", lw=.4)
            if row == 4:
                axes[row, col].set_xlabel("Seconds relative to CS onset")
        axes[row, 0].set_ylabel(f"{stage} · trial {trial}\nRaw vigor (rad/ms)", fontsize=9)
        comparisons.append({"trial": trial, "shared_frames": int(part.eligible.sum()),
                            "total_frames": len(part), "bin_max_error":
                            float(np.nanmax(np.abs(rebuilt.to_numpy()-stored["Signed log vigor"].to_numpy())))})
    axes[0, 0].set_title("Raw vigor on accepted bout frames", fontsize=11)
    axes[0, 1].set_title("Same frames: bout median\nExact signal used by column 3" if args.bout_only else
                          "Same frames: centred log (light)\nand bout median (dark)", fontsize=11)
    axes[0, 2].set_title("Exact stored 0.5 s bout-summary bins\nGrey = missing", fontsize=11)
    fig.suptitle("Figure 1E review · Delay fish 20221115_07\n"
                 "One shared bout-frame mask for raw and scaled vigor · baseline [-15,0) s",
                 fontsize=13)
    if args.bout_only:
        values = frames.signed_bout_on_bout_frames.dropna()
        low, high = min(-.25,float(values.min())),max(.25,float(values.max()))
        margin=(high-low)*.05
        for row in range(5):
            for col in (1,2):
                axes[row,col].set_ylim(low-margin,high+margin)
                axes[row,col].set_ylabel('Signed log vigor',fontsize=8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = args.output_dir / "same_frame_vigor.parquet"
    frames.to_parquet(data, index=False)
    svg = args.output_dir / ("direct_bout_to_bin_review.svg" if args.bout_only else "same_frame_vigor_review.svg")
    fig.savefig(svg)
    fig.savefig(svg.with_suffix(".png"), dpi=150)
    plt.close(fig)
    metadata = {"recording_id": "20221115_07", "metric_id": "legacy_distal_angular_speed",
                "baseline_s": [-15, 0], "support": "valid & moving & bout_id>0 & finite(Vigor) & Vigor>0",
                "raw_semantics": "Untransformed raw vigor; excluded frames NaN",
                "scaled_frame_semantics": "ln(raw vigor) minus trial baseline median log vigor",
                "scaled_bout_semantics": "Per-bout median of centred logs on the same eligible frames",
                "bin_semantics": "Existing F values; frame-weighted mean of supported bout medians",
                "y_clipping": "none", "checks": comparisons,
                "bout_only_display": args.bout_only,
                "source_frames": str(args.source), "source_frames_sha256": digest(args.source),
                "heatmap_sha256": digest(heat_path), "data_sha256": digest(data),
                "svg_sha256": digest(svg), "selection_status": "review"}
    svg.with_suffix(".svg.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"output": str(args.output_dir), "checks": comparisons}, indent=2))


if __name__ == "__main__":
    main()
