# F, G and H: quantile review, 7 October 2026

The follow-up [baseline-colour review](BASELINE_COLOUR_REVIEW.md) corrects C and verifies every option's reference against the actual midpoint of `managua_r`. The original historical comparison below is retained as an audit trail.

Only panels F (Delay 20221115_07), G (3 s Trace 20230307_12) and H (Control 20221115_09) were rendered. `P190` is interpreted as `P90`, the percentile present in the historical code. All candidates use `managua_r`, half-second bins and each trial's own finite baseline bins in [-15, 0) seconds. No candidate is frozen as a manuscript asset.

## Historical evidence

The actual Git snapshots inspected were 14 February 2026, commit `2f63ef48361d6e80d1b8e1393894d7e5a3dedf55`, and 24 March 2026, commit `bf46bf7b02baaf6d8138d9772f255881f86c3c78`. Both root-level preprocessing and example-fish scripts are preserved under `history/` with hashes in the generated manifest. These snapshots differ from the currently relocated legacy scripts.

The historical example-fish script contains two methods. `do_sc_new` logs positive eligible vigor, subtracts a trial baseline median, then replaces eligible bout samples with a bout median. `do_sc_old` takes median **raw** vigor per bout and applies `(vigor - baseline_P10)/(baseline_P90 - baseline_P10)`, clipped to [0, 1]. Both heatmap branches select baseline times earlier than -15 s. The historical trace plotting uses another baseline window. Consequently this is not one internally consistent reference convention to copy wholesale.

The March settings enable the new method, disable the old method, use `managua_r`, and set the CS plot's manual colour limits to [-0.25, 0.25]. The old quantile method therefore does not explain the selected March log-vigor plot. A latent defect also exists in its colour-limit helper: taking the minimum of values clipped to [0, 1] and reflecting it about zero can produce [0, 0]. Executing only that pure helper on [0, 0.5, 1] reproduces the defect. No historical script was run as an analysis pipeline.

## Common input and order of operations

Source-angle reconstruction and eligibility are identical across candidates; details and checks against panel E are in [FGH_PROCESSING.md](FGH_PROCESSING.md). Briefly:

1. Sum the 16 local tail angles. Compute the absolute wrapped temporal change divided by the reconstructed acquisition interval, obtaining distal angular speed in rad/ms. Use inferred camera cadence, not variable frame-arrival intervals. Do not bridge missing FrameIDs.
2. Apply the existing moving/bout detector. Eligible samples must be valid, moving, assigned a positive bout ID, and have finite, positive vigor. Excluded samples remain missing rather than becoming zero.
3. Take natural logs of eligible vigor. Replace eligible samples of each bout with that bout's median log vigor in the displayed trial window; do not fill the gaps between eligible samples. The previous frame-baseline subtraction is undone to recover the uncentred bout-log value.
4. Average contributing eligible frame values within each half-second bin. Thus bouts are weighted by their contributing frame counts within each bin; an empty bin is NaN. One contributing frame is sufficient for a finite bin. There is no coverage threshold or interpolation.
5. From those displayed bins, calculate that trial's baseline median and subtract it. This makes the baseline median of the plotted log bins zero. Baseline quantiles are subsequently calculated from the same finite baseline bins, with equal weight per bin, not per frame or bout.

This final-bin reference differs from panel E's earlier frame reference; panel E was not altered. Baseline centring fixes a location statistic, not the distribution's width, skewness or mean.

## Rendered comparisons

Outputs are in `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/fgh-quantile-history-review-20261007`. Each B/C/D panel has PNG, SVG and PDF exports; `quantile_comparison.*` shows all four methods. Original log-bin values and transformed candidates are separately retained in `candidate_bins.parquet`.

| Candidate | Operation | Meaning of colours |
| --- | --- | --- |
| A | Existing centred log bins, display limits ±0.25 | Same log-vigor units across trials |
| B | Same stored values; symmetric pooled baseline P10/P90 display limits ±0.34162552 | Same log-vigor units across trials, less colour saturation |
| C | Per-trial `(x-P10)/(P90-P10)`, clipped to [0, 1] | Trial quantile range; baseline median generally not at colourbar midpoint |
| D | Per-trial `(x-P50)/max(P50-P10,P90-P50)`, clipped to [-1, 1] | Deviation relative to that trial's symmetric baseline quantile range; median stays at zero |

C adapts the historical formula to the current **log-bout bins and approved window**. It is not a reproduction of the historical raw-frame preprocessing. D is a new median-preserving adaptation, not a historical formula. D uses one denominator on both sides of zero. Using separate denominators moves the arithmetic sample median for even baseline sample counts, so that earlier exploratory variant was replaced and an even-sample regression check added.

D maps the wider of the two quantile tails to its colour endpoint. It deliberately does not force both P10 and P90 to opposite endpoints when they are asymmetrically spaced around the median. Stored transformed values are clipped; original log bins remain available separately. This preserves the zero median even after clipping.

## Findings and limits

Literal affine P10/P90 scaling does not satisfy the requested neutral baseline centre: individual trial medians in C span approximately 0.085–0.971. In G the pooled Test baseline median becomes 0.293 while Train becomes 0.677. This would create a phase-dependent colour shift despite the original trial-centred medians being zero.

B preserves quantitative log-vigor comparisons, but is not a complete visual solution: G Test baseline saturation falls from 48.1% to 33.1%. Its baseline spread and skewness are real properties of these binned values, so a median subtraction alone cannot make them mostly neutral-coloured. D addresses trial-to-trial range variation as well as location, but equal colours then represent different log-vigor amplitudes in different trials. Narrow, sparsely sampled baselines can amplify small changes; per-trial sample counts and ranges are exported for inspection.

For A/B, NaN bins remain F 754/7200, G 698/7200 and H 3272/7200. Quantile scaling in C/D is undefined for H trial 16: it has only **one** finite baseline bin, hence no nonzero quantile range can be estimated. Its 28 otherwise finite plotted bins become explicitly NaN, giving H 3300/7200. All supported bins and gaps are retained unchanged in every other trial. No missingness was added to F or G to make their plots look sparse.

Nine unit tests cover baseline boundaries, reference missingness, skewed and even-sized baselines, clipping, affine invariance, two-bin support, and degenerate ranges. The renderer additionally checks every valid D trial's baseline median is zero and its input missingness is retained. Checks validate the implementation, not the scientific suitability of adopting per-trial range division.

For the user's stated combination of trial-specific centring and quantile scaling, D is the coherent review version. For retaining comparable log-vigor amplitudes across trials/fish, B remains the appropriate alternative. Appearance alone cannot settle that scientific choice.
