# Learning-Onset Analysis Completion

**Status:** Condition-aware block and longitudinal routes, robustness,
diagnostics, and three-panel figure implemented; Gate O/S approval,
paper-scale calibration, and paper execution open.

**Input dependency:** A reviewed, immutable cohort and authenticated
`cohort-trial-outcomes` artifact from the
[cohort plan](./01_COHORT_IMPLEMENTATION.md). This plan does not decide fish
membership. Its trial-level eligibility may change model contributions
without changing the cohort.

The [analysis and statistics plan](./02_ANALYSIS_AND_STATISTICS.md)
owns the primary estimand, outcome family, condition contrast, model choice,
uncertainty, diagnostics, and validation decision. Approved choices are
recorded in [DECISIONS.md](./DECISIONS.md). The
[LME pipeline reference](../docs/analysis/LME_PIPELINE_AND_PARAMETERS.md)
documents current formulas, public parameters, artifacts, and failure gates.

## Questions and implemented route

The analysis asks whether test and control trajectories change differently,
by which prespecified block a difference is supported, and whether an exact
trial onset can be localized. Fish are the population units. An isolated
per-trial p-value cannot establish onset.

The implemented route produces fish-weighted descriptive trajectories, a
condition-by-block mixed model with planned contrasts, one condition-aware
longitudinal spline model with a categorical-trial sensitivity, fish-bootstrap
simultaneous trial bands, a persistent onset rule, fish-level
permutation/bootstrap robustness, leave-one-fish-out and residual diagnostics,
authenticated tables, and a three-component figure. It uses one selected
metric and a cohort-authenticated, reason-coded model-input eligibility table.
The older condition-pooled exploratory LME/permutation/bootstrap routes are
not substitutes for this analysis.

The proposed onset rule finds the earliest scheduled trial whose lower
simultaneous confidence bound for test-versus-control change from pre-training
exceeds `delta_min` for `K` consecutive eligible trials. `delta_min`, `K`, the
contrast direction, and the trial family require approval before a paper fit.
No qualifying run is reported as **onset not localized**; bootstrap samples
without onset remain in the uncertainty summary. A first supported block is
reported separately because it answers a coarser question.

## Remaining work

1. Freeze the primary metric and outcome, test/control conditions, baseline
   and response windows, pre-training reference, blocks, contrast direction,
   `delta_min`, `K`, multiplicity family, spline complexity, random-effects
   fallback, bootstrap size/seed, and publication-stopping diagnostics.
   Choose an outcome-appropriate model family; do not automatically apply a
   Gaussian LME to movement probability or bout counts.
2. Calibrate false-onset rate, interval coverage, and onset recovery at the
   expected paper fish/trial counts. Include null, pre-existing difference,
   abrupt, gradual, transient, missingness, unequal coverage, influential
   fish, and singular-fit scenarios. Verify effect sign, fish weighting,
   factor references, multiplicity, and failure handling.
3. Run the approved configuration on the frozen paper cohort. Review the
   block and longitudinal fits, simultaneous band, fish-level robustness,
   categorical sensitivity, coverage, convergence, covariance, Hessian where
   available, residuals, and leave-one-fish-out influence.
4. Approve panel data and the three figure components: fish/condition trial
   trajectories, adjusted trial contrasts with simultaneous uncertainty and
   qualified onset marker, and planned block contrasts. Every visible value
   must trace to saved cohort, configuration, model, and source identities.
   Failed diagnostics suppress inferential annotations.

## Exit gate

The paper result tests an approved condition-versus-control change estimand,
uses fish as the independent units, reports block and trial evidence as
distinct claims, and either localizes onset with a prespecified meaningful
effect, persistence, simultaneous uncertainty, and passing diagnostics or
states that onset was not localized. The fish-level robustness result agrees
or its disagreement is explained. The figure regenerates from authenticated
panel data. The earlier [onset design](./Archive/LEARNING_ONSET_LME.md) and
[combined implementation plan](./Archive/REMAINING_COHORT_AND_LEARNING_ONSET_IMPLEMENTATION.md)
remain historical records.
