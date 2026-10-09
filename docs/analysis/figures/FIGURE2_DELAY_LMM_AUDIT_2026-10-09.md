# Confirmation of every Delay LMM and log-display meaning

Audit requested by the author on 2026-10-09 after reviewing the dense statistical
annotations. Scope is the 57-fish bout-only Delay review, not every repository
model, Trace panels, learning classification or full figure assembly.

## Audit result

All 15 saved models were independently refitted from the authenticated saved
model input, using statsmodels directly with their recorded formulas, ML
estimation (`reml=False`), optimizer and random-effects structure. All fixed
coefficients reproduce within 2.3e-16. All fits converge, have full fixed-design
rank and positive random/fixed covariance eigenvalues. No fallback was used.
The D, joint interaction, M, R and trial statistics plus Holm/BH corrections
were independently reconstructed and reproduce the saved tables.

| Model family | Count | Fixed effects | Random effects | Annotations |
| --- | --- | --- | --- | --- |
| Main block | 1 | ln(response) ~ ln(baseline) + condition * block | fish intercept + scaled-trial slope | joint test; 8 D interaction terms, Holm8 |
| Main longitudinal | 1 | ln(response) ~ ln(baseline) + condition * cubic spline(trial), 5 basis columns | fish intercept + scaled-trial slope | 90 baseline-referenced trial contrasts, BH90 |
| Local block | 9 | ln(response) ~ ln(baseline) + condition * centered trial | fish intercept | M at block center, BH9; R slope, raw and separate BH9 |
| Optimizer sensitivity | 2 | same as respective main model | same | Powell versus L-BFGS |
| Random-intercept sensitivity | 2 | same as respective main model | fish intercept | assess random-slope dependence |

All models use the frozen legacy metric, bout-only arithmetic window means,
positive responses/baselines, baseline [-15,0), response [0,9), trials 5–94.
No additive log offset is used. Control is the condition reference and Pre-train
is the block reference. The input has 4,811 observations, 57 fish overall;
Train5 and Test2 local fits have 56 contributing fish because one control fish
has no eligible ratio in that block. No missing outcomes are changed to zero.
The observed curves give each fish one ratio per trial; the likelihood uses all
eligible repeated observations, with dependence modeled at fish level, and is
not literally an equal-weight fit of the plotted condition medians.

## Diagnostic bug found and fixed

The old helper called `model.hessian(result.params)`. The result array packs
direct covariance entries, but the default model decodes array inputs as
square-root covariance parameters. This evaluated curvature away from the MLE.
The helper now passes `result.params_object`, as statsmodels itself does when
computing fitted covariance. This fixes the diagnostic, not the fit or p-value
calculation. The audit refitted all 15 models and evaluated both forms.

All corrected Hessians are negative definite: largest eigenvalues range from
about -118.26 to -6.42. The main block value is -57.96 and main spline -57.06.
The previous curvature numbers must not be used as valid MLE checks. The
17 learning-onset tests pass, including a regression assertion comparing the
recorded Hessian with the structured MLE Hessian and rejecting the array form.
Historical bundles remain intact, with this correction recorded separately.

## Why the stars look implausibly dense

The independent trial-design calculation reproduces all 67 stars, trials 5–71.
The actual spline interior knots are trials **35 and 65**, with endpoints 5/94.
There is no knot or separate phase at the training start, trial 15. A single
global cubic spline carries a shared shape through pre-training and training.
Each black star tests:

`(control - Delay adjusted log response at t) - average(control - Delay in Pre5–14)`.

It is a two-sided test, not a direct condition difference, not a test of the
median ratio, and not the original legacy per-trial test. The ten Pre contrasts
average zero by construction, but each can differ from that average. All ten
Pre trials reject here: early Pre estimates are negative and later ones
positive. This is mathematically consistent with a fitted trend; it cannot be
read as acquisition before training. Strong cancellation near the Pre average
also shrinks both the contrast and its SE. Dense stars are not 67 independent
discoveries and cannot localize onset.

Powell reproduces the same 5–71 range. Removing the random slope changes it to
5–75 (71 trials), so the endpoint is not insensitive to random-effects choice.
This is an observed sensitivity, not evidence that either structure is the
correct final model. Previous 57 leave-one-fish-out refits have a maximum trial
contrast shift of .01772 log-response units and mean training contrast range
.06573–.08215. Converged refits alone do not establish influence adequacy.

## Adequacy remains unresolved

Numerical reproduction is not scientific validation. Main longitudinal
residual excess kurtosis is 7.423 (normal reference 0); local block values
range 3.328–12.483. Several fits emit the recorded boundary warning. Covariance
eigenvalues pass the explicit singularity rule, but warnings cannot be erased
by the status `ok`. Residual SD is .126 control versus .119 Delay in the main
model; this comparison alone does not establish homoscedasticity. Correctly
paired adjacent-trial residual lag1 is .136 overall median, and does not
validate the full serial covariance. The older residual helper could pair
trials across gaps; the audit uses only genuinely adjacent scheduled trials.

Coverage/missingness selection, variable numbers of bout samples per window,
serial covariance and heavy tails remain material. The 5,000 descriptive fish
bootstraps do not refit the LMM and do not calibrate these model Wald p-values.
The review omitted the Plans' categorical-trial sensitivity, simultaneous onset
bands and onset calibration; no onset/extinction claim was completed.

Do not accept the current black-star strip as final trial inference. The next
bounded comparison should retain the data/metric and compare a phase-aware
trajectory with a categorical-trial LMM sensitivity, examining fit, residuals
and fish-level uncertainty rather than selecting the smallest p-values. Keep
block-level evidence separate. A fish-level robustness result describes a
different estimand and cannot validate every local/trial annotation.

## What the new log line means

For fish i at trial t, let R be the arithmetic mean of eligible bout frames in
the response window and B the corresponding baseline mean. The ratio version
plots `median_fish(R/B)`. The log version plots `median_fish(ln(R/B))`.

Example: response mean .12 and baseline mean .15 give ratio .8 and ln(.8)=-.223.
This is the same fish/window intensity comparison on another display scale.
Across fish, take the median of those log values and bootstrap that statistic.
It is not a fitted curve; the log and ratio displays use identical LMM tests.
With an even number of fish, arithmetic averaging of the middle two values
means `median(ln(ratio))` need not be exactly `ln(median(ratio))`.

The historical LogMedian pipeline instead logged positive vigor upstream and
used `median_window(ln(vigor_response)) - median_window(ln(vigor_baseline))`,
after its own rolling median/downsampling steps, then summarized across fish.
It changes the within-window statistic as well as the display scale. For a
simple unsmoothed example, response frames [1,1,10] versus baseline [1,1,1]
give a mean ratio of 4 and log mean ratio ln(4)=1.386. The window median of logs
is 0 in both windows, so their difference is 0. Logging after taking a mean
does not undo the influence a large frame already had on that mean.

## Saved evidence

Audit bundle:
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/20261009T143706055966Z-delay-lmm-audit/`.
It includes all 15 corrected diagnostics, independently recalculated test
tables/matrices, optimizer/random-structure trial sensitivities, spline knots,
exact audit/helper scripts, authenticated source identity and output hashes.
The prior plot/statistical artifacts are preserved; no panel was frozen.
