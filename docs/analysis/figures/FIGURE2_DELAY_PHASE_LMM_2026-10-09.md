# Delay G: observed medians with separately fitted contrasts

The author requested another version with a log-response LMM, log-baseline
adjustment and fish random effects, retaining observed medians and showing
fitted contrasts separately. This candidate preserves the frozen legacy metric,
bout-only windows, cohort and missingness. It compares the global spline against
a new phase-aware model; it does not freeze a panel, choose a paper model or
change the shared full assembly. Historical artifacts are preserved.

## Exactly what is shown

Top: blue control, magenta Delay. For each fish/trial, mean of finite valid
adjacent bout response frames [0,9) divided by equivalent baseline mean [-15,0).
Take the equal-fish condition median at each trial. Unsmoothed lines; reference
1. Bands are the already authenticated pointwise 95% whole-fish bootstrap
percentile CIs: 5,000 resamples, seed 10, within condition, same selected fish
trajectory across trials, preserving NaNs. There are 28 control and 29 Delay
cohort fish, with variable trial contributions and 4,811 eligible rows.

Bottom: gray dashed is the previous global spline, purple is the new phase-aware
LMM. Both show the common-baseline adjusted contrast:

`[control minus Delay log response at trial] - [average control minus Delay in Pre5–14]`.

Positive is greater response suppression in Delay relative to control and Pre.
These are fixed-effect population contrasts, not individual-fish predictions
and not fitted condition medians of the displayed ratios. Fish random effects
account for dependence during fitting; they are zero for population prediction.
The common log-baseline value is the model-input mean; its additive term cancels
from the condition contrast. Bands are pointwise 95% Gaussian/Wald CIs from
the fixed-effect covariance, including uncertainty in the Pre reference.
They do not use the 5,000 descriptive bootstrap resamples. Neither band is a
simultaneous onset band. Phase curves are segmented to avoid drawing a false
continuity across phase boundaries. No new D/M/R or trial-star strip is drawn;
the full trial tests are saved with their declared BH90 correction.

## Defined model comparison

Both models fit natural log of positive arithmetic bout-window response means,
with natural log of the baseline mean as covariate, no log offset, control
reference, fish random intercept plus global scaled-trial random slope. ML
estimation; primary L-BFGS optimizer; no automatic random-intercept fallback.

Global fixed model:
`ln(response) ~ ln(baseline) + condition * cubic_spline(global_trial, df=5)`.
It has 13 fixed coefficients and actual interior knots at trials 35 and 65.

Phase-aware fixed model:
`ln(response) ~ ln(baseline) + condition * (phase + pre_linear_trial + training_spline4 + test_spline4)`.
It has 25 fixed coefficients. Pre5–14 is linear, training15–64 has its own
cubic df4 spline, and test65–94 another cubic df4 spline. Each phase has its
own intercept; continuity is not imposed. Each spline uses a schedule-defined
normalized trial coordinate and one interior knot at its phase midpoint.
Inactive phase basis values are zero. This avoids a training spline basis
directly imposing curvature in Pre, although shared baseline/covariance
parameters still use the complete data. Exact Patsy formulas and basis values
are saved in the candidate specification and phase-features.csv.

The boundaries and basis complexity were set before these candidate fits,
motivated by known experimental phases, not selected to minimize p-values.
This is nevertheless same-data exploratory model comparison after seeing
the old results. No claim of independent confirmation is made.

## Results and limitations

| Quantity | Global spline | Phase-aware candidate |
| --- | --- | --- |
| Fixed coefficients | 13 | 25 |
| AIC (lower preferred under its assumptions) | -6095.30 | -6099.46 |
| BIC (lower preferred under its assumptions) | -5985.17 | -5911.58 |
| Residual excess kurtosis (normal reference 0) | 7.423 | 7.363 |
| Median genuine adjacent-trial residual lag1 | .136 | .125 |
| Significant Pre contrasts, BH90 | 10 | 0 |
| Significant trial contrasts, BH90 | 5–71 | 19–70 |
| Mean training change, log-response units | .07788 | .07876 |
| Mean test change, log-response units | .02873 | .03042 |

AIC weakly favors phase-aware; BIC favors global because the new model is more
complex. Neither criterion establishes calibration, biological correctness,
or an onset model. The residual shape barely improves; heavy tails, unequal
spread and missingness selection remain unresolved. AIC/BIC here use the
software's observation-count convention for repeated data; treat them as
descriptive comparison, not a definitive fish-level validation criterion.

Both primary models, alternate Powell phase fit and random-intercept-only
phase sensitivity pass convergence, full-rank, positive covariance and the
corrected structured-MLE Hessian checks. Boundary warnings remain recorded.
Powell agrees on trials19–70; removing the random slope extends this to19–74.
All 57 phase leave-one-fish-out refits converge without detected singularity;
their maximum change-contrast shift is .01777 log-response units. These
influence refits use the lighter convergence/rank/random-covariance gate,
not full curvature reviews for every omitted-fish fit.

Trial19 is not an inferred learning onset: it is the first rejection in this
particular fit and family. Onset needs an approved meaningful effect,
simultaneous uncertainty, persistence and calibration; none is computed here.
Not forcing training curvature into Pre improves interpretability, but does
not ensure absence of a genuine Pre trend or make model uncertainty negligible.

## Was historical LogMedian better?

It may be a better summary for the question "what is typical vigor while
bouting?" Within-window medians resist rare extreme values, whereas arithmetic
means retain the influence of strong bursts. Whether those bursts are noise
or meaningful behavior is a scientific question. Logging an arithmetic mean
afterward does not make that mean as resistant as a within-window median.

Historical LogMedian logs the processed positive vigor values before computing
response and baseline window medians and subtracting them. The current log
ratio computes arithmetic response/baseline means first, then logs the ratio.
They summarize different aspects of the frame distribution. A window with
values [1,1,10] has mean4 and median1; the distinction is not merely an axis
rescaling. Even medians after log versus logs after median can differ under
even-sample interpolation conventions.

My recommendation is to retain the current means for this controlled model
comparison, and evaluate a fresh within-window LogMedian outcome as a bounded
sensitivity on the same cohort, windows, bout masks and preprocessing. Do not
bundle the change with the historical rolling-median/downsampling steps or
choose it because it yields attractive stars. For a typical-intensity claim,
I lean toward the within-window median; for an average-intensity claim with
meaningful bursts, the arithmetic mean is more directly aligned.

Do not copy the old inference literally: the historical LogMedian preparation
also applies log(x+1) to already log-transformed window summaries. A clean
median-based ANCOVA would model the response window's median log vigor with
the baseline window's median log vigor covariate, without a second log, subject
to its own eligibility and residual checks. This turn does not implement or
select that new outcome; it implements the requested arithmetic-mean LMM
comparison and explains the alternative.

Sources: [NIST on mean/median location](https://www.itl.nist.gov/div898/handbook/eda/section3/eda351.htm),
[statsmodels mixed-effects documentation](https://www.statsmodels.org/stable/mixed_linear.html).
Historical source evidence is in the [bout-only comparison](FIGURE2_DELAY_BOUT_ONLY_COMPARISON_2026-10-09.md).

## Artifacts and validation

Bundle:
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/20261009T144413666038Z-delay-phase-lmm/`.
Contains PNG/SVG/PDF, model input, schedule-defined basis, exact formulas,
coefficients/covariances, contrast matrices/tables, diagnostics/residuals,
optimizer/random-structure sensitivities, 57 influence refits and hashed scripts.
Two meaningful phase tests pass: phase basis isolation and recovery of a pure
training shift without an artificial Pre/Test effect. The three descriptive
resampling/multiplicity tests also pass. The global model's contrast curve
reproduces the prior saved contrasts. SVG parsing, export hashes, plotted
source equality and visual legibility were checked. This is a scientific
exploratory render and does not invoke the panel-freeze validator.
