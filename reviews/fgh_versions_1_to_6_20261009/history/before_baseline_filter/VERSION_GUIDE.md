# F/G/H versions 1 to 6 - selection guide

Eight active options, organized under the historical version numbers. Version 3 has no established recipe. Four-colour Version 6 is discarded. Versions 5 and 6 use 1 s bins.

## Shared processing

- Use the same fish: F Delay 20221115_07, G 3 s Trace 20230310_08, H Control 20221115_09.
- Calculate raw framewise vigor before bout detection: sum 16 corrected tail-angle components, take the absolute wrapped consecutive-frame change, and divide by the inferred camera interval. Raw units are rad/ms.
- Use the same bout detector in every version: median smoothing 10 ms (7 samples); rolling maximum 28.57 ms (21 samples); minimum 571.43 ms (401 samples); envelope threshold 4 degrees/ms; peak threshold 1 degree/ms; minimum duration 57.14 ms; maximum gap 14.29 ms.
- Eligible samples require valid consecutive frames, finite detector envelope, at least 80% angular coverage, detected bout membership and finite positive raw vigor. Take the natural log. Non-bout and invalid/ineligible samples stay NaN.
- Binned versions average the finite eligible log values, not raw vigor followed by a log. Any single finite sample is sufficient; NaNs are ignored and all-NaN bins remain missing. There is no minimum coverage gate.
- Display trials 5-94 and [-20,20) s relative to CS. Pre = 5-14, Train = 15-64, Test = 65-94. Retain strong green CS boundaries at 0/10 s, phase names left of F, a shared colourbar and G arrows at trials 9/17/63/66/93.
- Missing cells are black. In the scaled Version 1/2 rows, black can also indicate an undefined trial scale; Control trial 16 is undefined. Versions 4/5/6 have all 270 references defined.

## Overview

| Version | Values and bins | Baseline | Zero | Colours |
|---|---|---|---|---|
| 1 | Complete-bout medians on samples; No bins | [-15,0) s; every eligible timepoint carrying its complete-bout median | Baseline median | Continuous managua_r; scaled range [-1,+1] |
| 2 | Direct log means in 0.5 s bins; 0.5 s; 80 bins/trial | [-15,0) s; one vote per finite unscaled baseline-bin mean | Median of baseline-bin means | Continuous managua_r; scaled range [-1,+1] |
| 3 | No established recipe; Not established | Not established | Not established | No figure |
| 4 | Direct log means in 0.25 s bins; 0.25 s; 160 bins/trial | [-20,0) s; mean of finite baseline-bin means, one vote per bin | Baseline mean; median may differ | Continuous managua_r; fixed limits +/-0.25 log units |
| 5 | Direct log means in 1 s bins; 1 s; 40 bins/trial | [-20,0) s; mean of finite new baseline-bin means, one vote per bin | Baseline mean; median may differ | Continuous managua_r; fixed limits +/-0.25 log units |
| 6 | 1 s means with five baseline percentile bands; 1 s; 40 bins/trial | [-20,0) s; P25/P45/P50/P55/P75 of finite bin means | Baseline median P50 | Five managua_r colours; narrow dark P45-P55 centre |

## Version 1: Complete-bout medians on samples

Summarize each bout, then preserve its eligible sample timing. Two percentile scales, C and D.

How it is made:

1. For each complete detected bout, calculate the median of all its eligible natural-log vigor samples, including support beyond the display crop.
2. Repeat that bout median on each of its eligible frame timepoints, preserving ineligible gaps; crop only afterwards.
3. For each trial, use eligible timepoints in [-15,0) s to calculate P10, m=P50 and P90. Every timepoint votes; longer eligible bout duration contributes more votes.
4. Subtract m, divide by the selected C or D denominator, then numerically clip to [-1,+1]. A nonpositive or missing denominator makes the trial undefined.

Difference: C and D use exactly the same bout summaries and baseline population. Only the scaling denominator differs.

When choosing: Retains fine sample timing but removes variation within each bout. Complete medians of bouts crossing CS can include post-CS samples, even for baseline timepoints. C/D are median-centred percentile scales, not ordinary min-max scaling.

### Version 1 C

Uses half the P10-P90 range. Often gives stronger contrast and more saturation than D.

`C = clip((x - m) / ((P90 - P10) / 2), -1, +1)`

### Version 1 D

Uses the larger distance from the median to either percentile. Absolute scaled values cannot exceed C for the same sample.

`D = clip((x - m) / max(m - P10, P90 - m), -1, +1)`

## Version 2: Direct log means in 0.5 s bins

Replace bout-median substitution with direct framewise log means, using a matching baseline-bin percentile scale.

How it is made:

1. Average direct eligible framewise log vigor in each CS-aligned 0.5 s bin. Do not substitute bout medians.
2. Within each trial, calculate P10, m=P50 and P90 from finite unscaled baseline-bin means in [-15,0) s. Each finite bin has one vote, regardless of its eligible frame count.
3. Subtract m, divide by (P90-P10)/2 and numerically clip to [-1,+1]. Missing or zero percentile range makes the trial undefined.

Difference: Compared with Version 1 C, changes the input value, temporal aggregation and baseline voting unit. Retains the same C formula and 15 s baseline interval.

When choosing: Describes direct eligible moving vigor, but the scale varies by trial. Equal colours across trials need not represent equal physical log differences. The older blue-offset row used a mismatched bout-median reference and is not this current Version 2.

### Version 2 C

Displayed baseline-bin median is zero for all 269 defined fish/trials; Control trial 16 remains undefined.

`C = clip((bin mean - m) / ((P90 - P10) / 2), -1, +1)`

## Version 3: No established recipe

The historical numbering contains a gap.

Difference: The recorded rows were Version 1 C, Version 1 D, Version 2 C and Version 4. Version 1 D is a second Version 1 option, not an approved Version 3.

When choosing: No formula or figure has been invented or renumbered to fill this gap.

## Version 4: Direct log means in 0.25 s bins

Keep physical log differences: subtract a matching baseline mean, without percentile scaling.

How it is made:

1. Average direct eligible framewise log vigor in each 0.25 s bin.
2. For each trial, take the arithmetic mean of its finite baseline-bin means in [-20,0) s: up to 80 baseline bins, one vote per finite bin.
3. Subtract this baseline mean from each bin mean. No P10/P90 scale, division or numerical clipping.
4. Render with fixed colour limits [-0.25,+0.25]. Values beyond the limits use endpoint colours while their full numeric values remain in exports.

Difference: Compared with Version 2, uses finer bins, a 20 s baseline and mean subtraction; removes percentile scaling and numeric clipping.

When choosing: Preserves physical log-unit differences across trials and fish. Fine bins give more temporal detail and more missing cells. Mean-zero does not guarantee median-zero or equal positive/negative counts. Earlier median and -infinity experiments are historical, not this recipe.

### Version 4 linear

Linear colour mapping: u = (clip(d,-0.25,+0.25) + 0.25) / 0.5.

`d = bin mean - mean(finite baseline-bin means)`

### Version 4 contrast

Same numeric data, baseline and masks as V4 linear. Only the colour mapping changes, increasing separation near zero. Colourbar remains in log units.

`Same d; colour coordinate u = 0.5 + 0.5 sign(d) sqrt(min(abs(d)/0.25,1))`

## Version 5: Direct log means in 1 s bins

The Version 4 recipe with coarser bins and its baseline recomputed from those new bins.

How it is made:

1. Recalculate each 1 s mean directly from eligible framewise log values. Do not simply average the old 0.25 s means: eligible sample counts differ.
2. For each trial, calculate the arithmetic mean of its finite new 1 s baseline-bin means in [-20,0) s, up to 20 bins. Each finite bin has one vote.
3. Subtract that baseline mean. Retain physical log units, no percentile scaling and no numeric clipping.
4. Use the same +/-0.25 colour limits as Version 4, with either linear or contrast rendering.

Difference: Compared with Version 4, only the bin width and the matching recomputed baseline change. Linear versus contrast is a presentation choice, not another scientific recipe.

When choosing: More eligible samples contribute to each wider bin and fewer bins are missing, but short temporal features are averaged together. Physical log differences remain comparable across trials and fish; visual endpoint saturation still hides exact magnitudes beyond +/-0.25.

### Version 5 linear

Linear continuous colours with the same physical scale as Version 4 linear.

`d = 1 s bin mean - mean(finite 1 s baseline-bin means)`

### Version 5 contrast

Exactly the same numeric values as V5 linear; stronger visual separation of small differences near zero.

`Same d; colour coordinate u = 0.5 + 0.5 sign(d) sqrt(min(abs(d)/0.25,1))`

## Version 6: 1 s means with five baseline percentile bands

Median-centre each trial, then show its position relative to its own baseline distribution.

How it is made:

1. Start from the same directly frame-recomputed 1 s bin means as Version 5.
2. For each trial separately, calculate P25/P45/P50/P55/P75 from its finite unscaled baseline-bin means in [-20,0) s, one vote per finite bin. Quantiles use linear interpolation.
3. Subtract that trial P50 from each bin mean. The median-centred baseline has P50 = 0.
4. Assign five bands using the uncentred boundaries P25/P45/P55/P75, or equivalently the centred boundaries P25-P50, P45-P50, P55-P50 and P75-P50.
5. The bands are: x < P25; P25 <= x < P45; P45 <= x < P55; P55 <= x < P75; x >= P75. Exact boundary ties enter the upper band. Missing values remain black.
6. Use managua_r positions 0/0.25/0.5/0.75/1: #81e7ff, #5775b3, #582948, #b26343, #ffcf67. The dark purple middle band contains zero and is distinct from missing black.
7. Retain original mean-centred values and new median-centred log values. Export ordinal display scores [-1,-0.5,0,+0.5,+1]; these are category labels, not a continuous amplitude normalization.

Difference: Compared with Version 5, retains the same 1 s means but changes mean centring to median centring and continuous physical colours to trial-specific percentile bands.

When choosing: Simplifies the display and identifies low/near-baseline/high activity relative to each trial. Equal colours across trials need not indicate equal physical log differences, and values within a band are merged visually. P45-P55 is a percentile interval, not necessarily exactly 10% of observed bins when there are few baseline bins or ties. The four-colour Version 6 is discarded; prior 0.25 s exports are historical.

### Version 6 five bands

All 270 trial references are defined; all have zero inside the dark central P45-P55 band.

`z = 1 s bin mean - baseline P50; display band = interval between trial baseline percentile boundaries`

## Choosing between recipes

- Version 1: bout summaries on eligible sample timing; trial-relative percentile scaling.
- Version 2: direct moving vigor in half-second means; trial-relative percentile scaling.
- Versions 4 and 5: physical log differences, with quarter-second or one-second detail respectively.
- Version 6: one-second means shown as five bands relative to each trial baseline; absolute amplitude detail is reduced.
- Linear versus contrast in Versions 4 and 5 changes colour mapping only. C versus D in Version 1 changes the numeric denominator.

All existing data, figures, scoped selections and frozen panels are preserved. No new recipe is assigned to the Version 3 gap.
