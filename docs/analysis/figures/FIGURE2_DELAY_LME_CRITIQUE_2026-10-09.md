# Delay G: critique of the Plans, legacy LME and new review

The author authorized this exploratory run on2026-10-09, fixed the legacy
metric, and accepted5,000 whole-fish resamples/seed10 for the displayed median
CI. The bootstrap is descriptive uncertainty, not5,000 refits of the LME.
Neither a preliminary plot nor numerical convergence closes scientific gates.

## What the Plans improve

`Plans/02_ANALYSIS_AND_STATISTICS.md` correctly starts from a biological
estimand, identifies fish as independent units, retains condition in population
comparisons, and requires a simpler fish-level robustness analysis and failure
handling. `Plans/03_LEARNING_ONSET_IMPLEMENTATION.md` correctly distinguishes
block evidence, trial evidence, persistent onset and extinction. A first
isolated significant trial cannot establish onset; disappearing significance
cannot establish extinction.

Their unresolved weakness is that many consequential choices remain open:
outcome scale, baseline adjustment, covariance, day/rig structure, coverage,
contrast family, spline complexity, minimum meaningful effect and persistence.
The implemented defaults (including late Test2/3, offset1e-6, spline5 and K3)
are engineering defaults, not automatically accepted biological definitions.
Random intercepts/slopes do not guarantee correct residual correlation or
homoscedasticity. Bootstrap calibration and model adequacy require evidence at
the actual fish counts. Separate correction families are defensible for an
exploratory review, but do not collectively control one paper-wide error rate.
This run therefore implements a bounded G review, not every onset/learner
deliverable in the Plans.

## What is wrong with literal legacy reuse

February14 and March24 `5_NormalizedVigorPlotting_LogMedian.py` snapshots are
identical and preserve the original D/M/R/star hierarchy. However:

- Gold D uses unadjusted individual interaction coefficients; that is not a
  single joint condition-by-block test.
- At one trial, one observation per fish cannot separately identify a fish
  random intercept and observation noise.90 separately fitted LMEs are not an
  adequate longitudinal model.
- `log(response+1)` is sensitive to the units of small activity values: adding1
  to rad/ms versus deg/ms is not the same scientifically meaningful transform.
  The new review uses log(response+1e-6) with all current responses positive.
- The plot shows median response/baseline ratios, while the ANCOVA tests
  log response adjusted for baseline. Those are different estimands; their
  uncertainty and tests must not be conflated.
- M and R ask about a local mean and local slope. They do not prove acquisition
  onset. Raw R should remain visibly exploratory; its FDR result can differ.
- Skipping failed fits, selecting coefficient terms implicitly and borrowing
  stars from an image hides analysis failures and contrast identity.

## Defined review method

29 Delay +28 control fish; CS trials5–94; legacy metric; baseline[-15,0),
response[0,9). Sample means over valid adjacent frames include stationary
frames. Current eligibility retains finite positive baselines, nonnegative
finite responses and at least one valid sample per window; no stricter
coverage rule was silently added. Upstream signal/alignment and coverage
approval remain open.

The block model and spline5 longitudinal model use log response, log baseline
covariate, condition-by-time terms and fish random intercepts/slopes. Nine
local block LMEs use random intercepts and centered within-block trial. D uses
Holm over8 nonreference block interactions. M and R use separate BH9 families;
raw R is displayed separately. Black stars use BH90 on two-sided longitudinal
contrasts: control-minus-Delay at each trial minus its average in Pre5–14.
They describe differential change, not the legacy raw condition contrast.
Trials5–14 remain in the declared90-trial family. A future paper choice to
test only post-pre trials would be a new, recorded family, not a silent edit.

The curves remain unsmoothed. Each whole-fish bootstrap draw samples28 control
and29 Delay fish with replacement separately. A duplicate fish brings its
entire90-trial trajectory twice, including NaNs; all trials use the same fish
draw. Quantiles of5,000 condition medians produce the pointwise95% CI. Seed10
fixes the draw sequence. Missingness is preserved, not imputed. Preserving it
does not eliminate bias from informative missingness.

## Results and limitations

| Requested lane | Fresh legacy-metric result |
| --- | --- |
| Joint condition × block test | Wald χ²8=19.1485, p=.014085 |
| D individual interactions | None survives Holm8 (Train4 raw p=.014732, adjusted=.117857) |
| M local block means | Train4: estimate−.11649 log response, BH p=.006939 → M** |
| R local slopes | Train3: estimate−.01386 log response/trial, raw p=.030317 → R*; BH p=.272856 |
| R FDR | No block below .05 |
| Black trial stars | No differential-change trial survives BH90 |

All main/local fits and prespecified optimizer/intercept sensitivities passed
convergence, finite positive fixed covariance and Hessian curvature checks.
All57 leave-one-fish-out block/longitudinal fits passed their numerical fit
checks. Powell main fixed coefficients differ by at most about.00017; changing
the random structure affects nuisance coefficients and some contrasts, so
random-effects structure still needs scientific review. Passing influence
refit convergence does not by itself establish influence robustness.

Residual diagnostics are a material limitation: longitudinal residual
skewness−1.97, excess kurtosis20.85, with strong negative tails and changing
spread. Median within-fish lag1 correlation is.093, but that single summary
does not validate the full serial covariance. Gaussian Wald p-values and
their annotations are provisional. The preview explicitly flags this; it
must not become the final paper inference by approving its artwork.

A prespecified simple fish-level all-training-versus-pre log-ratio contrast
gives Delay-minus-control suppression=.13696, fish-bootstrap95% CI
.08480–.19074. It supports overall training suppression descriptively while
the more localized ANCOVA tests ask different questions. This is same-data
robustness, not independent confirmation. Neither result localizes onset or
extinction. No simultaneous onset band was computed in this scoped review.

Saved bootstrap endpoint change from the first2,500 draws to all5,000 reached
.02350 ratio units at the most variable endpoint. Discrete medians at28/29
fish can move between order statistics; the requested5,000-draw result is
preserved without claiming perfect Monte Carlo convergence.

Recommended next statistical work is a bounded outcome/covariance adequacy
review with fish-level robust inference, explicit coverage/day sensitivity
and calibration. Keep the current figure as an exploratory preview and keep
all nonsignificant or failed results visible in the tables.

## Artifacts and verification

Analysis bundle:
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/20261009T072921130316Z-delay-lme/`.
Corrected display:
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/20261009T073349568633Z-delay-display/`.
These preserve model input, fish ratios,5,000 draw indices/median trajectories,
full coefficient/covariance/test tables, residuals, sensitivities and57
leave-one-fish-out results. Source dependencies verified and plotted ratios
reproduced the existing G data. Three meaningful tests verify whole-fish
resampling, missingness, seed/order invariance and retention of failed tests
in multiplicity families. SVG/PNG/PDF share one plotted-data source.
Old sources and versions, H/I, Figure1 and the shared assembly are preserved.
