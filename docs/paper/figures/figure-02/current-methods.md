# Figure 2: selected working-draft methods

This reference describes the Figure 2 draft selected by
[current state](current-state.json), reconciled on 25 October 2026. The
[matching caption](legend.md) describes the same draft. Accepted scientific
authority remains the [scoped selection index](../../../../records/frozen-analyses/selections.md):
A, D/E version B and G have scoped freezes; the later draft changes do not
replace them or create a whole-figure freeze.

The [detailed Methods preserved on 16 October](../shared/figure-specific-methods.md)
retain their original wording and hashes. They describe earlier rosters,
eligibility, correction families and pending panels. Use that document for its
recorded historical scope, rather than as the current H/I or whole-draft recipe.

| Scope | Preserved 16 October description | Selected working draft |
|---|---|---|
| B/E/H | 19 controls and 40 3sTrace fish | Expanded 90-control/89-3sTrace roster; B/E/H remain unfrozen |
| C/F/I | Pending authenticated inputs | Fixed 10sTrace draft with 32 controls and 29 10sTrace fish; common 61-fish roster |
| D/E/F block eligibility | D minimum 3; E minimum 1; F pending | Minimum 3 eligible trials in each five-trial block in all three panels |
| D/E/F multiplicity | Joint D/E Holm36 | Separate Holm18 within D, E and F; frozen D/E version B remains preserved |
| H/I trajectory | H total-activity ratio and fish IQR; I pending | Observed log-median differences and pointwise whole-fish bootstrap 95% intervals |
| G | Accepted Historical LogMedian with phase-aware LMM | Same scoped scientific selection and limitations; presentation is separately selected |

The 32 controls in the fixed 10sTrace roster share acquisition identities with
historical 3sTrace controls. They are not independent new acquisitions across
assays. A populated panel does not establish paper-wide dataset completeness
or an approved primary cohort.

## Outcome, aggregation and uncertainty

G/H/I use the selected tail-bend angular-speed metric. Within each fish/trial,
take the median natural logarithm of finite positive valid detected-bout frames
in the response window, then subtract the corresponding baseline log median.
Baseline is [−15,0) s relative to CS onset; response windows are [0,9), [0,13)
and [0,20) s for Delay, 3sTrace and fixed 10sTrace. Empty eligible windows remain
undefined. This is conditional bout-frame movement intensity; it does not measure
movement probability or zero-inclusive activity. There is one logarithm, with
no second log, amplitude scaling or temporal smoothing of these trajectories.

The plotted trajectory is the median across contributing fish at each scheduled
trial, 5–94. Pointwise percentile 95% intervals use 5,000 whole-fish trajectory
resamples within condition, with the recorded seed 10 and missing trials retained.
These describe uncertainty in observed medians. D/E/F black block summaries
instead show fish median/IQR after within-fish trial aggregation; IQR describes
fish spread and is not a confidence interval.

## Model annotations

The models fit the response-window log median with baseline log median as a
covariate. This baseline adjustment differs from the subtraction in the plotted
outcome. Fixed-effect population means and contrasts are distinct from plotted
fish medians and their bootstrap intervals.

| Row | Saved model/contrast | Correction |
|---|---|---|
| D | Baseline-adjusted condition-by-ten-trial-block interaction relative to Pre; fish random intercept and trial slope | Holm8 |
| M | Local baseline-adjusted condition difference at centred trial; fish random intercept | BH9 |
| R raw/FDR | Local condition-by-within-block-trial slope difference | Raw and BH9 |
| Trial change | Phase-aware baseline-adjusted condition trajectory; fish random intercept/slope; control-minus-experimental change relative to average Pre5–14 | BH90 |

The phase fixed design contains condition-by-phase terms, a Pre linear term and
separate schedule-defined four-column cubic spline bases for Train and Test.
Primary block/phase diagnostics record `lbfgs`, the requested `1 + trial_scaled`
random structure and no fallback. Local fits record `powell` and random intercepts.
Exact formulas, covariance, scaling, failures and warnings belong to the bound
run records, not current command defaults. The recovered H/I fitter explicitly
uses maximum likelihood; complete historical driver/runtime closure remains open.

Trial symbols mark BH90 p < .05, rather than a p-value tier. Other statistical
labels use the caption's thresholds; G's unavailable Test3 local results remain
`n/a`. Positive trial-change contrast and negative condition-by-block coefficients
can express the same experimental suppression direction because their contrasts
use opposite condition orders. Neither identifies learning onset or extinction.

## Audit and remaining limits

The [source-bound audit record](../../../../records/scientific-evidence/analyses/figure-02-ghi-statistical-audit.json)
links the exact evidence and SSD-owned Markdown/HTML report. It reconciles saved
outcomes, contributing counts, coefficients, correction families, phase contrasts,
bootstrap payloads and named SVG statistical annotations without refitting models.
It separately records missing historical code/runtime bindings and the original
producer-binding gap for I's bootstrap draw file.

All three phase fits retain heavy residual tails. Numerical agreement is not
model calibration, biological validation, paper-wide cohort approval or production
verification. XML annotation checks establish label content and marker counts;
they do not establish painted visibility or exact marker placement. The unselected
canvas-overdraw correction and remaining renderer checks are separate work.
