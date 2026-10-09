# Figure 2 D/E/F: legacy log-transformation audit

Inspected on 2026-10-09. This is a source comparison, not a rerender or freeze.

## Current selected panels

D is Delay/control; E is 3sTrace/control; F is 10sTrace/control and remains a placeholder/inconclusive. D/E use blocks 10-14, 65-69 and 90-94.

The newer scoped selection is `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row2-block-ratio-review/analysis-selection.json`, selecting `style-baseline-freeze/20261009T110531835455Z/Fig2_PanelD_reference-style.figure.json` and the corresponding E record. Their SHA-256 values matched the selection: D `2091331d00bafd6fad6287434119388ecd0cc77119a85fb7ad00c5da182857b4`, E `df88deaaa19af62adebcb11fc314b709f134fcbbe10f7f8be84392993e98fb7e`.

Both records specify response_total_activity / baseline_total_activity, then the median of eligible trial ratios within fish/block. Both explicitly record `log_transformation: false`. The inspected renderer plots those saved fish/block ratios directly on linear axes with reference y=1. These sources are newer than the sources in `configs/paper-figures/figure2-assembly.json`.

## Historical source evidence

| Snapshot | Plot calculation | No-change reference |
| --- | --- | --- |
| 2026-02-04, `55ce3b2` | Response-window mean / baseline-window mean, then fish/block median | 1 |
| 2026-02-14, `2f63ef48361d6e80d1b8e1393894d7e5a3dedf55` | Response-window median log vigor minus baseline-window median log vigor, then fish/block median | 0 |
| 2026-03-24, `bf46bf7b02baaf6d8138d9772f255881f86c3c78` | Same log-median implementation as February 14 | 0 |

The February 14 and March 24 copies of both Step 3 and Step 5 are byte-identical; see `source-manifest.json` for source revisions and SHA-256 values. Copies here preserve the Git blobs exactly.

In both dated `3_FishGrouping_LogMedian.py` files, line 293 overwrites `Vigor (deg/ms)` with `np.log(vigor)` for positive values, and uses NaN otherwise. The preceding operations mask non-bout frames, apply rolling-median smoothing and downsample.

In both dated `5_NormalizedVigorPlotting_LogMedian.py` files, line 175 selects `_new_logmedian` input. Lines 773-774 take the window medians of the already logged vigor column; line 794 subtracts baseline from response. The block renderer takes fish/block medians and plots `Normalized vigor` directly (line 900); the boxplot also plots this quantity (line 1212). The baseline helper defaults to zero (line 357), and the declared y limits are -0.2 to +0.2 (line 197). The block plotting paths contain no logarithmic-axis call. Their run flags enable processing and block line/boxplot rendering.

Thus the historical block-plot values themselves were log-derived. This is distinct from displaying unlogged ratios on logarithmic axes. It also differs from simply logging the current response/baseline ratio: the historical window aggregation and upstream masking/smoothing differ.

The February 4 standard implementation plotted raw mean ratios (line 708), but separately used `np.log(baseline + 1)` and `np.log(response + 1)` for its trial-by-trial mixed models (lines 459-460). The later LogMedian Step 5 retains those model transformations too (lines 494-495), applied to window summaries that are already in log units. Model transforms must be distinguished from the block plot's data transform.

This audit establishes what the code implements and enables; it does not establish which historical image was actually generated or accepted. No historical plots, selected panels, data, statistics or freeze records were changed.
