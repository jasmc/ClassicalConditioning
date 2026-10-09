# Delay SVG generated on 16 February 2026: comparison with current F/G/H

The referenced file is `F:/Results (paper)/2025_delay/Processed data/20221115_07_delay_blue-1_mitfaminusminus,elavl3gff,10uasgcamp6fef05_6dpf_scaled vigor heatmap aligned to CS_cmap_managua_r_vlim_auto.svg`.

**The processing in the matching historical Git branch differs from the current F/G/H display processing.** The exact generating worktree is not recoverable from the SVG alone. No original figures, data, or shared preprocessing were modified for this comparison.

## Direct evidence from the file

- Embedded SVG generation timestamp: **2026-02-16T16:45:53.111034**, without a timezone offset.
- Embedded creator: **Matplotlib v3.10.7**.
- Filesystem creation time: 16 February 2026 16:44:03; last write: 16 February 2026 16:45:55, as displayed by PowerShell. The embedded export date is the more useful generation evidence.
- Visible x-axis labels: **−20, 0, 20**; phase labels: Pre, Train, Test; title: Example Delay fish; historical panel label: E.
- Three embedded raster heatmap images, with native dimensions 763×163, 763×818 and 763×492. Their nonblack RGB colours match the `managua_r` lookup table within one RGB unit. These are rendered colours, not recoverable raw scalar matrices.
- The SVG contains three axes and no visible colourbar or numeric colour-limit metadata. The suffix `vlim_auto` does not establish automatic limits.

## Git chronology

The nearest preceding tracked example-fish revision is **2f63ef48361d6e80d1b8e1393894d7e5a3dedf55**, committed **14 February 2026, 17:29:05 UTC**. Its defaults are a ±40 s window, PNG output, `spring`, and limits [-0.5, 0.25]. Those defaults do not reproduce the dated SVG's configuration.

Revision **bf46bf7b02baaf6d8138d9772f255881f86c3c78**, committed **24 March 2026, 13:41:25 UTC**, changes those settings to ±20 s, SVG, `managua_r`, and [-0.25, 0.25], and includes both CS and US figures. These settings agree more closely with the asset, but that commit occurs after its embedded export date. The evidence is consistent with local settings changes made before they were committed; it does not prove that explanation or identify an exact generating SHA.

Between these two snapshots, the relevant heatmap transformation is unchanged. The preprocessing script, general configuration and analysis helper also have no changes between these commits. Exact snapshots and SHA256 hashes are saved alongside the evidence JSON.

## Matching historical branch

In both snapshots, the filename without `(P10-P90)` belongs to **`do_sc_new`**, the log-median branch. The separate old quantile branch writes a filename explicitly containing `(P10-P90)`. The branch matching this SVG performs:

1. Crop the sample table to the plotted trial window. Set non-bout samples to NaN.
2. Log-transform positive raw vigor; set non-positive values to NaN.
3. For each trial, select `Trial time (s) < -gen_config.baseline_window`. The configuration sets `baseline_window = 15`.
4. Subtract the median of those **sample-level log values** before bout replacement or plotting.
5. Replace samples within each detected bout interval with that bout's median log value.
6. Pivot the time-sample table directly into a trial-by-time heatmap. There is **no half-second scalar-bin aggregation or post-binning baseline median** in this branch. Casting time column labels to integers does not aggregate the underlying samples.

With the SVG's displayed ±20 s window, the matching code's baseline selection would be **[-20, -15) s**, not [-15, 0) s. The February default ±40 s would instead select [-40, -15) s. This baseline-window interpretation is an inference from the matching code and asset settings, not a baseline declaration embedded in the SVG.

The historical CS branch passes manual `INDIVIDUAL_TRIALS_SCALED_VIGOR_VMIN/VMAX` to `sns.heatmap` even though it also computes a symmetric automatic range and writes `vlim_auto` in the filename. The computed automatic limits are used in its US path. Thus neither the suffix nor the March defaults certify the exact limits used in this February asset. The absent colourbar is another presentation difference from the inspected snapshots, which pass `cbar=True`.

## Comparison with current F/G/H

| Step | Matching February/March historical branch | Current F/G/H review |
| --- | --- | --- |
| Time samples upstream | Interpolated 700 FPS grid | Presumed acquisition cadence; F approximately 702.57 FPS |
| Upstream angle filtering | 10-sample centred temporal mean in the tracked helper | Historical temporal mean not introduced into current figure rebuild |
| Heatmap cells | Direct time-sample columns carrying repeated bout values | One finite-frame mean of repeated bout log medians per 0.5 s bin |
| Baseline selection | Samples with t < -15 s, cropped to the plotted window | Finite scalar bins in [-15, 0) s |
| Display reference | Median of sample log vigor before bout replacement | Median of the already binned scalar values, equally weighted per finite bin |
| Range division | None in the filename-matching log branch | A/B none; C/D use baseline-bin quantile ranges |
| Colour map | managua_r in the asset | managua_r |
| Display limits | Not numerically recoverable from this SVG | Explicit symmetric limits, centred at zero |

The historical filter helper accepts a spatial window of 3 but does not actually apply a spatial rolling mean in these snapshots; its implemented smoothing is the temporal mean. Parameter names and module descriptions alone are not evidence that spatial filtering occurred.

The shared fish and colour map do not make the calculations identical. The current user-required **bin first → median of that trial's baseline scalar bins → subtract median** is a substantive change from the matching historical code. Current C/D quantile division is another substantive change. The earlier recreated frame-reference F/G/H version also differs from the historical baseline window and upstream processing.

Evidence: `historical_svg_20260216_evidence.json`; read-only audit: `audit_historical_svg.py`; source lines around 1520–1555 and 1620–1630 in `history/2026-02-14_2_ExampleFishPlotting.py`. Current reference verification is in `verify_postbin_display_reference.py` and [BASELINE_COLOUR_REVIEW.md](BASELINE_COLOUR_REVIEW.md).
