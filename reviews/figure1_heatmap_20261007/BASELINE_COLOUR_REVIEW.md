# All F/G/H options: verified baseline colour

This review supersedes the A/B/C/D comparison in `fgh-quantile-history-review-20261007`. It changes only F, G and H and writes a new review folder: `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/fgh-all-options-baseline-colour-centred-20261007`.

## Required centre

Each trial's **median of finite displayed baseline bins in [-15, 0) seconds** is the reference. The bins are the same half-second means of eligible bout-median log-vigor frame values described in [FGH_PROCESSING.md](FGH_PROCESSING.md) and [QUANTILE_HISTORY_REVIEW.md](QUANTILE_HISTORY_REVIEW.md). The reference is neither pooled across trials nor inferred from the visible colours.

The mandatory display order is: **aggregate into 0.5 s bins → select finite baseline bins from that trial → take their median → subtract that median → apply any quantile range scaling and symmetric clipping → map zero to managua_r(0.5)**. The median gives every finite bin equal weight, irrespective of how many eligible frames contributed to it. The frame-stage reference in earlier data is removed/replaced; it does not define the display centre. `verify_postbin_display_reference.py` independently checks this order using the stored uncentred bin values and does not load frame observations.

With Matplotlib 3.10.9, the actual `managua_r` midpoint `cmap(0.5)` is RGBA corresponding to **#582948**, a dark purple. Its low endpoint is blue and its high endpoint is yellow. `CenteredNorm(vcenter=0, halfrange=L, clip=True)` explicitly maps zero to palette coordinate 0.5. The plotted cells and colourbar use the same normalization and copied colormap. NaN is black.

## Review and correction of the four options

Let x be a displayed bin before the final trial-baseline subtraction. Let m, p10 and p90 be that same trial's finite baseline-bin median, 10th percentile and 90th percentile.

| Option | Stored/display signal | Centre | Colour interval |
| --- | --- | --- | --- |
| A | x - m | Trial baseline median 0 | ±0.25 log units |
| B | x - m | Trial baseline median 0 | ±0.3416255204813127 log units |
| C, corrected | 2(x - m)/(p90 - p10), clipped to [-1, 1] | Trial baseline median 0 | ±1 relative units |
| D | (x - m)/max(m - p10, p90 - m), clipped to [-1, 1] | Trial baseline median 0 | ±1 relative units |

A and B already satisfied the centre requirement. B's pooled baseline quantiles set a common colour range only; every row retains its own median reference. A/B retain comparable log-vigor amplitudes across trials and fish.

The previous C applied `(x-p10)/(p90-p10)`, clipped to [0, 1], with a 0–1 colourbar. It failed to place the median at palette coordinate 0.5 in **all 269 trials** with a defined quantile range. C now subtracts m before range division and uses symmetric clipping. It keeps the idea of dividing by the P10–P90 width but is explicitly a median-centred adaptation rather than the literal historical formula. The failure audit is retained separately.

D was already correctly centred. It uses a single denominator for both sides of the median. Its scale is at least as large as C's half-width, so D's absolute normalized values and saturation cannot exceed C's for the same input. It is the gentler per-trial quantile option. It does not force both baseline quantile tails to the endpoints when their distances from the median differ.

C/D express values relative to each trial's baseline spread. Equal colours can therefore represent different log-vigor changes in different trials. The unclipped normalized signals, clipped signals, original log bins, individual quantile ranges and original input fields are retained in `all_options_bins.parquet` and `trial_quantile_ranges.csv`.

## Validation

The per-trial audit checks the median of finite baseline values **and** the median palette coordinate of those values after clipping, using the very same normalization used in the figures. It separately checks the reference zero has exactly the RGBA of `managua_r(0.5)`, and checks that all defined trials preserve their source NaN masks.

| Option | Trials with defined centre/scale | Maximum absolute baseline median | Maximum palette-median error from 0.5 | Total NaN bins |
| --- | ---: | ---: | ---: | ---: |
| A | 270 | 1.39e-17 | 0 | 4724 |
| B | 270 | 1.39e-17 | 0 | 4724 |
| C, corrected | 269 | 5.55e-17 | 0 | 4752 |
| D | 269 | 5.55e-17 | 0 | 4752 |

The extra 28 missing bins in C/D are all H trial 16. It has only one finite baseline bin, so its baseline median exists for A/B but a nonzero quantile scaling range does not. It is marked undefined, not assigned a borrowed trial/fish denominator. NaN bins in all other trials are unchanged. F and G have no added missingness.

The audit contains 1080 option/trial rows. Fifteen unit tests pass, including skewed odd/even baselines, clipping, baseline-window boundaries, single-bin and constant baselines, masked/missing inputs, and correspondence with the actual `managua_r` midpoint. Source hashes are verified before rendering; the output manifest records source, code, figure and audit hashes.

## What the colour centre does and does not establish

The **median reference** is purple for every defined trial. This does not require every individual baseline bin to look purple. Baseline bins can lie on either side of zero, and the mean, spread, skewness and visible area of each hue can differ even with exactly the same median. Averaging RGB colours is not a valid test of baseline centring. Missing bins are also excluded from the median rather than assigned the baseline colour.

The corrected comparison and every individual F/G/H panel are exported as PNG, SVG and PDF. `baseline_colour_key.png` shows the exact mapping; `all_options_trial_colour_audit.csv` contains the numeric checks. D remains the coherent choice for trial-specific quantile scaling; B preserves comparisons in log-vigor units.
