# Current G versus independent exploratory results — 2026-10-15

This is a read-only comparison of the selected 2026-10-09 G freeze with the
independent `20261014-analysis-02` result selected by its state pointer.
`current-G-comparison-2026-10-15.json` binds the numerical sources, hashes and
derived comparisons. No figure, frozen source or analysis result was edited.

## What is reproduced

The same 57 fish, 5,130 scheduled trials, 4,811 eligible outcomes, raw positive
bout-frame LogMedian values and windows are used. Missing bouts are not zero.
The current G descriptive trajectory takes medians across fish, with pointwise
percentile intervals from 5,000 whole-fish resamples, seed 10. The independent
review uses mean descriptive trajectories for its direct-difference target;
it does not reproduce the displayed across-fish median as its model estimand.

The new adjusted-response candidate uses the same trial-model family as
current G: response LogMedian with baseline LogMedian as a covariate, separate
Pre linear and Train/Test df4 cubic spline bases, phase-specific intercepts,
condition interactions and fish random intercept/slope. Population contrasts
are control-minus-Delay change from average Pre5–14. The fitted trial
contrasts differ by at most 1.5123e-8 log units; SEs by at most 5.5495e-8.
The alternative direct-difference candidate removes the baseline covariate
and instead models response-minus-baseline LogMedian.

Numerical tables in the relocated `20261009T145647405899Z-delay-logmedian`
directory were used only after matching their SHA-256 hashes to the selected
freeze inventory. Six matching sources include the phase contrast, coefficient,
covariance and contrast-matrix files, D tests and fish-level outcomes. A
different model-input file in that directory did not match the selected hash
and was not treated as the authoritative frozen model-input artifact.

## The full current G statistical analysis

Current G has distinct test families, with conditional Gaussian mixed models:

- D: eight baseline-adjusted condition-by-block interactions relative to Pre,
  Holm corrected. Training blocks 1–5 and Test1 pass; Test2/3 do not.
- M: nine local condition differences, BH corrected. Train2–5 and Test1 pass.
- R: nine local slope differences, both raw and BH rows. Train1 and Test1 pass.
- Trial change: BH across 90 trial contrasts from the phase-aware model.
  Trials 19–71 pass. This is the selected phase-aware result, not the older
  global-spline sensitivity with pre-training significance.
- A saved joint block-interaction Wald test has p=1.2945e-22; it is an omnibus
  test under that fitted model, not an onset or effect-size test.

Test3 local M/R failed as singular and is explicitly unavailable. These are
different hypotheses: M is a level difference, R a slope difference, D a change
relative to Pre. Their rows are not independent replications of one claim.
The M/R annotation details were inspected from the saved selected sidecar;
the exact original M/R CSV was not found among the accessible relocated tables.

The independent workflow did not refit and calibrate these identical D/M/R
families. It compares planned phase averages and trial changes with one joint
bootstrap-t family per target (90 trial contrasts plus three phase contrasts).
The new 999 whole-fish model refits differ from the old 5,000 descriptive median
resamples. Resample count alone does not explain wider intervals: estimands,
model refitting and simultaneous maximum-statistic uncertainty differ.

## Matched phase periods

The periods below are late training 55–64, early testing 65–74 and late testing
85–94. Positive contrasts indicate conditioned-direction suppression.

| Method | Late training | Early testing | Late testing |
|---|---|---|---|
| Current G phase-model average, derived now | 0.0947 | 0.0651 | 0.0262 |
| New adjusted model, joint interval | 0.0947 [0.0117, 0.1778] | 0.0651 [−0.0270, 0.1573] | 0.0262 [−0.0847, 0.1370] |
| New direct-difference model, joint interval | 0.1029 [0.0386, 0.1673] | 0.0738 [0.0063, 0.1413] | 0.0144 [−0.0586, 0.0874] |
| Unsmoothed fish-change comparator, joint interval | 0.0978 [−0.0477, 0.2434] | 0.0849 [0.0015, 0.1684] | 0.0162 [−0.0740, 0.1064] |

For context, applying ordinary 1.96-SE pointwise uncertainty to these averages
using the frozen phase covariance gives [0.0579, 0.1315], [0.0243, 0.1060] and
[−0.0229, 0.0753]. These are derived in this comparison, not originally
displayed phase-summary intervals and not Holm-adjusted D tests. They show why
identical point estimates can support different uncertainty statements.

Current D coefficients use Delay-minus-control, the opposite sign convention:
Train5 −0.09115 (Holm p=7.61e-6), Test1 −0.06173 (p=0.01153), Test3 −0.01892
(p=0.88897). Their categorical mean function differs from the phase spline;
these are directionally consistent with the phase-model averages above.

The unsmoothed comparator anchors each fish to its available Pre mean and
averages available fish changes; it has a different estimand. Its late-training
joint interval includes zero. A simpler pointwise fish-phase sensitivity has
a positive late-training interval, so joint versus pointwise scope matters
there too. Recording-date resampling retains positive late-training intervals
for both LMM targets; early testing remains baseline-treatment dependent.
Random-intercept, US-overlap and leave-one-fish-out fits are saved sensitivities,
not separate author-approved primary analyses.

## Acquisition, cohorts and validation

Current G has no acquisition estimator or onset claim. Trial19 is its first
BH-significant trial, not an estimate of biological acquisition. BH FDR control
does not produce a simultaneous confidence band or an onset interval; stars
against zero also do not test a prespecified meaningful magnitude. The new
zero-threshold three-trial rule has a nominal first supported run at trial22
in both smooth candidates, but calibration is not approved. At the unapproved
10% threshold the point crossings are 48/50 with no finite latest bound.
Neither the old trial19 star nor the new nominal zero-threshold result
contradicts that unresolved meaningful-threshold timing.

Historical exclusion projections increase late-training LMM estimates from
approximately 0.103/0.095 to 0.124/0.120 for the combined recipe (24 control,
27 Delay). A separate technical protocol-flag sensitivity removes only Delay
`20221122_03`, giving 0.096/0.083 (28/28). Both keep positive nominal
late-training LMM intervals; neither localizes 10% timing. Behavioral filtering
changes the population and cannot validate the primary cohort. Missing outcomes
do not by themselves require whole-fish exclusion. A malformed raw protocol
timestamp is genuine technical evidence warranting review, not proof the
whole recording must be discarded.

Expanded simulations fail full-curve coverage in all ten abrupt and ten
transient examples for both smooth targets. Because current G shares this
mean family, the concern also applies to its trial-level interpretation.
The simulations do not directly estimate the error rates of its full BH/Holm
D/M/R families and do not establish that every current p-value is false.
Heavy residual tails and behavior-dependent availability remain concerns.
The unsmoothed comparison's 92–98% scenario coverage is promising but includes
wide/unbounded onset intervals and cannot validate the adjusted LMM.
Individual candidates assess heterogeneity rather than repeat the population
test; their conditional timing uncertainty remains unvalidated.

## Assessment

No numerical extraction or selected trial-model contrast error was found.
Current G is a reproducible exploratory display whose own record already
acknowledges heavy tails, unavailable Test3 local fits and no onset claim.
It is not a completed validated confirmatory analysis. The new evidence
strengthens the reason to qualify the trial stars and early-testing inference;
it does not erase the observed trajectories or reverse the late-training
direction. Precise onset, model-independent late-training significance,
whole-cohort movement effects and extinction are not established by current G.
