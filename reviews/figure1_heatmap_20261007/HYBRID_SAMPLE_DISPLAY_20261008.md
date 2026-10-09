# F/G/H trial: time samples displayed, scalar bins used for reference

This implements the user's table from 8 October 2026. Only panels F, G and H are rendered. Review outputs are in `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/fgh-sample-display-bin-baseline-20261008`. These are new trial artifacts, not frozen manuscript panels.

## Calculation, in order

1. Reconstruct the same acquired samples, raw angular-speed metric, bout IDs and eligible mask used by the existing F/G/H reconstruction. The camera clock, upstream filtering and bout detector are not changed for this trial.
2. For each trial, take the median log vigor of each bout's eligible samples within the displayed [-20, 20) s window. Repeat that scalar on the bout's eligible samples. Keep excluded samples as NaN and do not fill gaps between eligible samples.
3. **Temporarily bin this repeated-bout sample signal into 0.5 s bins.** Each bin is one scalar: the mean of its finite sample values. If multiple bouts contribute, their medians receive weight proportional to their contributing sample counts within that bin. A bin with no contribution is NaN.
4. Select that trial's finite scalar baseline bins in **[-15, 0) s**. Compute their **median m, P10 and P90**, giving each finite bin exactly one vote. Sample count does not weight the median or quantiles across bins.
5. Return to the **time samples**. Subtract m from each repeated bout-log sample and divide by the trial's bin-derived range:
   - C: `(sample - m) / ((P90 - P10) / 2)`.
   - D: `(sample - m) / max(m - P10, P90 - m)`.
6. Clip those transformed sample values to [-1, 1] and display them with `managua_r`. Explicit `CenteredNorm(vcenter=0, halfrange=1, clip=True)` maps the reference zero to palette midpoint 0.5, #582948.

**The final heatmap displays time samples, not the 0.5 s bin scalars.** Those bins determine the trial reference and scale only. Neither the baseline median nor the quantiles are estimated from frame-level observations. Clipping happens after reference/scale estimation.

Because displayed observations are now time samples, longer bouts occupy more horizontal space. The reference remains the median of the equally weighted baseline-bin scalars; the median of all individual displayed baseline samples is a different statistic and can differ from zero. Both are distinguished in the audit. Rebinning unclipped transformed samples reproduces the centred/scaled scalar-bin signal; rebinning after nonlinear clipping need not give the same means.

## Rendering and files

The presumed acquisition cadence is approximately 702.57 FPS (~1.423 ms per sample). For plotting, each acquired observation occupies one slot on a common grid at that cadence. There is no additional time averaging, interpolation or resampling of observations into new signal values. Assignment can place an observation within one acquisition interval of its exact timestamp; exact FrameIDs and timestamps remain in the sample Parquet. Empty grid positions and excluded observations are black. PNG/SVG/PDF pages render the dense sample grid as a raster image with nearest-neighbour rendering and vector labels, as appropriate for millions of observations.

The C/D grids are stored as float32 for display; the authoritative scalar sample values and unclipped transforms are retained as float64 in `PanelF_sample_values.parquet`, `PanelG_sample_values.parquet`, and `PanelH_sample_values.parquet`. The grids, slot times and trial IDs are also exported as compressed NPZ. `reference_bins.parquet` preserves the scalar bins used to estimate the references.

Each panel has a PNG, SVG and PDF in both C and D. `C_F-G-H_sample_display.png` and `D_F-G-H_sample_display.png` show the respective three panels. `trial_baseline_and_support_audit.csv` records references, quantile ranges, bin counts, acquired/eligible/excluded sample counts, sample-level medians after centring, and numerical comparison errors.

H trial 16 has one finite baseline bin and therefore no nonzero quantile range. C/D remain explicitly undefined for that trial; no range is borrowed from another trial or fish.

## Validation

All 21,600 scalar reference bins and contributing sample counts are checked against the preceding binned review. Its trial references and the C/D bin transformations are independently reproduced. Raw-bin comparisons use 1e-12 absolute tolerance; normalized comparisons use 1e-10, since division by narrow control ranges amplifies tiny summation-rounding differences. The actual maximum normalized error is recorded by trial.

Unit tests verify the defining hybrid distinction using unequal sample counts: a baseline with 100 observations in one bin and one observation in each of two others uses the **median of the three bin scalars**, rather than the median of all 102 samples. Tests also check sample gaps, empty bins, clipping, single-bin baseline handling, and the linear bin-to-sample correspondence. Eighteen tests pass across the review helpers.

Builder: `try_hybrid_sample_heatmaps.py`; pure reference/transform implementation: `hybrid_sample_display.py`. The build manifest records verified source hashes, output hashes, acquisition cadence and the exact formulas. Only the read-only acquisition/detector function of the existing builder is called; its main rendering path for other panels is not executed.
