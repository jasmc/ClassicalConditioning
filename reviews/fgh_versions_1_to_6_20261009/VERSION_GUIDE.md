# F/G/H baseline-centred options - selection guide

Four available options meet baseline-median centring: V1 C, V1 D, V2 C and V6 five bands. Versions 4/5 are removed because mean centring does not ensure median-zero. Four-colour V6 remains discarded; Version 3 has no established recipe.

## Shared processing

- Use the same fish: F Delay 20221115_07, G 3 s Trace 20230310_08, H Control 20221115_09.
- Calculate raw framewise vigor before bout detection: sum 16 corrected tail-angle components, take the absolute wrapped consecutive-frame change, and divide by the inferred camera interval. Raw units are rad/ms.
- Use the same bout detector in every version: median smoothing 10 ms (7 samples); rolling maximum 28.57 ms (21 samples); minimum 571.43 ms (401 samples); envelope threshold 4 degrees/ms; peak threshold 1 degree/ms; minimum duration 57.14 ms; maximum gap 14.29 ms.
- Eligible samples require valid consecutive frames, finite detector envelope, at least 80% angular coverage, detected bout membership and finite positive raw vigor. Take the natural log. Non-bout and invalid/ineligible samples stay NaN.
- Binned versions average the finite eligible log values, not raw vigor followed by a log. Any single finite sample is sufficient; NaNs are ignored and all-NaN bins remain missing. There is no minimum coverage gate.
- Display trials 5-94 and [-20,20) s relative to CS. Pre = 5-14, Train = 15-64, Test = 65-94. Retain strong green CS boundaries at 0/10 s, phase names left of F, a shared colourbar and G arrows at trials 9/17/63/66/93.
- Missing cells are black. In V1 C/D and V2 C, Control trial 16 is black because its percentile scale is undefined. V6 has all 270 references defined.

## Overview

| Option | How it is made | Baseline / scale | Pros | Cons |
|---|---|---|---|---|
| Version 1 C | Complete-bout medians on samples; No bins | [-15,0) s; every eligible timepoint carrying its complete-bout median; C = clip((x - m) / ((P90 - P10) / 2), -1, +1) | Bout medians reduce the influence of extreme within-bout samples; continuous colours and precise eligible timing. | Suppresses within-bout variation; stronger clipping; baseline is duration weighted; complete bout medians can include post-CS values; one undefined trial. |
| Version 1 D | Complete-bout medians on samples; No bins | [-15,0) s; every eligible timepoint carrying its complete-bout median; D = clip((x - m) / max(m - P10, P90 - m), -1, +1) | Same bout summaries with a larger denominator: less or equal clipping than C and more retained graded colour variation. | Same loss of within-bout detail, duration-weighted baseline and crossing-CS support as C; lower contrast; one undefined trial. |
| Version 2 C | Direct log means in 0.5 s bins; 0.5 s; 80 bins/trial | [-15,0) s; one vote per finite unscaled baseline-bin mean; C = clip((bin mean - m) / ((P90 - P10) / 2), -1, +1) | Direct eligible log means; continuous detail; 0.5 s temporal resolution; median-centred reference matches displayed bin means. | Trial-specific scaling and numeric clipping hide absolute magnitudes; one undefined trial; sparse bins vote equally in the baseline. |
| Version 6 five bands | 1 s means with five baseline percentile bands; 1 s; 40 bins/trial | [-20,0) s; P25/P45/P50/P55/P75 of finite bin means; z = 1 s bin mean - baseline P50; display band = interval between trial baseline percentile boundaries | Meets P50=0 in all 270 trials; a distinct dark centre; direct 1 s means; five colours make baseline-relative changes easy to read. | Coarser temporal and amplitude detail; colours describe each trial baseline rather than common physical magnitudes; at most 20 baseline bins estimate the percentiles. |

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
- Versions 4 and 5: removed from active choices because baseline mean-zero does not ensure median-zero.
- Version 6: one-second means shown as five bands relative to each trial baseline; absolute amplitude detail is reduced.
- C versus D in Version 1 changes the numeric denominator; both preserve baseline median-zero when the scale is defined.

Recommend Version 6 for the main heatmap under your stated goal: all 270 medians are defined and centred, zero is inside the dark P45-P55 band, and the five colours provide a simple common baseline-relative interpretation. Version 2 is the strongest alternative if finer timing and continuous colour gradations matter more; it retains 0.5 s bins but has one undefined trial.

Centring puts the baseline median at the middle colour; it does not make all baseline cells dark. V1/V2 have 269 defined references; V6 has 270.

All existing data, figures, scoped selections and frozen panels are preserved. No new recipe is assigned to the Version 3 gap.
