# Learning-Onset Inference for Trial Trajectories

> **Archived incomplete on 2026-09-23 as a superseded design proposal.** The
> condition-aware analysis is implemented but its scientific choices and
> paper-scale validation remain open. The active delivery plan is
> [learning-onset completion](../03_LEARNING_ONSET_IMPLEMENTATION.md);
> current behavior is documented in the
> [LME pipeline reference](../../docs/analysis/LME_PIPELINE_AND_PARAMETERS.md).
> Gate O/S decisions belong to the
> [analysis and statistics plan](../02_ANALYSIS_AND_STATISTICS.md).

**Status:** Design proposed; scientific choices and paper-scale execution remain open  
**Scope:** Condition-aware population learning across trials, block-level evidence,
trial-level localization, onset uncertainty, and the corresponding pooled figure  
**Depends on:** approved outcome, technical assessment, frozen cohort, shared
movement detector, and paper-scale coverage review

Detailed implementation order, artifact contracts, and acceptance gates are in
[Learning-Onset Analysis Completion](../03_LEARNING_ONSET_IMPLEMENTATION.md).

## Objective

Estimate when the conditioned/test population first shows evidence of learning
relative to control while respecting fish as the independent biological unit.
The main figure has trial number on the x-axis and conditioned-response strength
on the y-axis, with raw fish-level evidence, model-estimated condition
trajectories, uncertainty, and a prespecified learning-onset result.

The analysis must distinguish three questions:

1. Is there evidence that the test and control trajectories differ over time?
2. By which prespecified phase or block is learning established?
3. At which individual trial can the difference first be localized, with what
   uncertainty?

The third question is not answered by taking the first isolated significant
test from many independent per-trial models.

## Current implementation status

| Capability | Status | Location or limitation |
| --- | --- | --- |
| Legacy pooled trial trajectory | Implemented, archived | `Archive/historical-scripts/5_NormalizedVigorPlotting.py::run_trial_by_trial` |
| Legacy global condition-by-block LME | Implemented, archived | Baseline-adjusted log response; fish grouping |
| Legacy block-local condition-by-trial LME | Implemented, archived | Separate model per block |
| Legacy per-trial condition LME + BH-FDR | Implemented, archived | Separate model per trial; not a valid onset estimator |
| Artifact-producing legacy extraction | Implemented, archived | `Archive/package/src/classical_conditioning/analysis/legacy_statistics.py` |
| Authenticated candidate model input | Implemented | Carries fish, condition, trial, block, response, and baseline |
| Active candidate LME | Implemented only as exploratory plumbing | Omits condition and condition-by-time terms |
| Condition-aware longitudinal model | Implemented, not paper-run | Spline condition-by-trial model; scientific configuration remains unfrozen |
| Categorical-trial sensitivity | Implemented, not paper-run | Optional condition-by-categorical-trial mixed model; paper-scale feasibility remains untested |
| Planned block and trial contrasts | Implemented, not paper-run | Named block and trial contrasts relative to pre-training |
| Simultaneous trialwise uncertainty | Implemented, not paper-run | Within-condition fish bootstrap maximum-deviation band |
| Defined learning-onset rule and interval | Implemented, not paper-run | Configurable `delta_min`, persistence, and bootstrap onset distribution |
| Strict diagnostic/publication gates | Substantially implemented | Rank, convergence, covariance singularity, Hessian where available, simultaneous-band, robustness, and leave-one-fish-out gates; optimizer/random-structure sensitivity tables and residual diagnostic figure are implemented, while paper-scale review remains |
| Fish-weighted descriptive ratio figures | Implemented | Frozen-cohort selected-block, trial-number, and event-aligned response/baseline figures; no inferential annotations |
| Learning-trajectory publication panel | Implemented, not paper-run | Three panels use authenticated model input, simultaneous trial contrasts, and block contrasts |
| Coefficient, coverage, residual, and panel-data artifacts | Implemented, not paper-run | Authenticated tables are published with the analysis |

The descriptive builders and learning-onset route now consume
`cohort-trial-outcomes`. A reviewed paper cohort and approved O/S configuration
are still required before the resulting inference is paper-authoritative.

The two-stage discarding command does not calculate LME trial eligibility.
This analysis keeps its explicit cohort-authenticated eligibility table;
ineligible rows and fish with no model-eligible trials remain in sample flow.
The exploratory behavioral screen never changes the primary model cohort.

## Scientific definitions to freeze before fitting

### Population contrast

- Reference condition: control.
- Test condition: delay for the delay experiment, or the named conditioned
  condition for another experiment.
- Biological unit: fish.
- Repeated unit: trial within fish.
- Primary contrast: test-versus-control difference in change from the
  prespecified pre-training reference, not merely a static condition
  difference.

### Outcome and direction

Select one primary conditioned-response outcome before examining inferential
results. For a continuous activity/vigor outcome, the recommended model scale
is:

```text
y = log(response + c)
baseline = log(pre-CS baseline + c)
```

Model response with baseline as a covariate. Define the reported contrast so
that a positive value always means a stronger conditioned response in the test
condition. For motor suppression this can be reported as control minus test on
the adjusted log-response scale.

The plot may show a directly interpretable fish-trial score such as
`log(baseline + c) - log(response + c)`, but inferential annotations must be
identified as model-adjusted contrasts if the fitted response is not exactly
the plotted score.

Do not automatically reuse a Gaussian LME for every outcome:

- continuous positive activity/intensity: log-linear LME if diagnostics pass;
- movement occurrence/probability: binomial GLMM or fish-clustered GEE;
- bout counts/rates: Poisson or negative-binomial model with observation-time
  exposure;
- conditional intensity: analyze only where defined and report the associated
  movement-occurrence result.

### Meaning of "learning established"

Freeze a smallest biologically relevant condition contrast, `delta_min`, and a
persistence length, `K`, before fitting. Recommended initial definition:

> Learning is established at the earliest scheduled trial `t` for which the
> lower simultaneous 95% confidence bound for the test-versus-control change
> from pre-training exceeds `delta_min` for `K = 3` consecutive eligible
> trials.

For a motor-suppression outcome, the trial contrast is:

```text
D(t) = [control - test at trial t]
       - [control - test during pre-training]
```

If no trial satisfies the rule, report that onset was not localized; do not
substitute the trial with the smallest p-value. Report a fish-bootstrap
confidence interval or distribution for the onset trial. Also report the first
block satisfying its prespecified block-level contrast, because it will often
be more stable than an exact-trial claim.

`delta_min = 0` is a statistical-difference definition. A nonzero value is
preferable when a biologically meaningful minimum effect can be justified.

## Recommended model hierarchy

### Model 1 — primary block/phase LME

Use a compact, prespecified model for the primary population claim:

```text
log_response ~ log_baseline + condition * C(block_or_phase)
               + approved_day_or_rig_terms
groups = fish_id
random effects = 1 + scaled_trial
```

- Use ML for nested fixed-effect model comparisons and REML for final
  coefficient estimation if appropriate.
- Trial must be centered/scaled before a random slope is fitted.
- Test condition-related terms jointly; do not treat one interaction
  coefficient as the global interaction test.
- Prespecify a simpler random-intercept-only fallback. Use it only under a
  written singularity rule, not because it produces a preferred p-value.
- Do not use a random slope for condition: condition is normally constant
  within fish.
- Do not default to a random slope for baseline without a biological and
  numerical justification.

Planned contrasts should include:

1. pre-training test versus control balance/change reference;
2. test-versus-control change in each named training/test block;
3. one primary late-training or test contrast;
4. multiplicity correction across the limited planned block family.

### Model 2 — trial-resolution longitudinal LME

Use one longitudinal model, not one independently fitted LME per trial. Two
candidate implementations should be compared on blinded/synthetic criteria:

```text
# Smooth trajectory, preferred when the curve is gradual
log_response ~ log_baseline + condition * spline(scaled_trial, df=4..6)
               + approved_day_or_rig_terms
groups = fish_id
random effects = 1 + scaled_trial
```

```text
# Trial-categorical sensitivity, if sample size and rank support it
log_response ~ log_baseline + condition * C(trial_number)
               + approved_day_or_rig_terms
groups = fish_id
random effects = 1 + scaled_trial
```

The spline degrees of freedom or categorical trial range must be frozen before
examining the condition contrasts. The categorical implementation is useful
for showing that smoothing did not create the onset, but it may be too
parameter-heavy to serve as the primary model.

Derive adjusted test-versus-control contrasts at every scheduled trial from
the single fitted model. Use a simultaneous confidence band or a fish-cluster
max-T/bootstrap procedure over the complete trial family. BH-FDR contrasts may
be retained as an explicitly exploratory comparison to the legacy method.

The archived per-trial LME should not be revived as the primary route. Within
one trial there is ordinarily one row per fish, so its fish random intercept is
weakly or non-identifiable; fit failures were silently skipped, and the legacy
coefficient lookup did not construct a safe named contrast when more than two
conditions were present.

### Model 3 — fish-level robustness analysis

Each fish contributes one prespecified change score, for example late test
minus pre-training on the signed CR scale. Compare change scores between test
and control with a condition-label permutation test and fish bootstrap
interval. This is the minimum robustness analysis and must preserve condition.

### Optional onset-focused sensitivity

A segmented or change-point mixed model may estimate a population onset
parameter directly. Because breakpoint inference is non-regular and sensitive
to curve shape, use it only as a bounded sensitivity analysis unless simulation
shows reliable recovery at the actual fish and trial counts.

## Figure specification

The main panel should contain:

- one observation per fish per trial before population aggregation;
- faint fish trajectories or a companion fish-level panel;
- equal-fish-weight condition estimates;
- 95% fish-aware/model-based intervals;
- block boundaries and phase labels;
- the prespecified primary block contrast;
- the onset estimate and interval, if localized;
- an optional compact trial-contrast strip based on simultaneous intervals.

Do not use ordinary row-level seaborn bootstrap intervals. Do not cover the
panel with raw p-value stars. A separate contrast panel showing effect size and
confidence bounds is preferred.

## Diagnostics and publication-stopping rules

Every fitted model must publish:

- observations, fish, fish per condition, and trials per fish/block;
- fixed-effect design rank and reference categories;
- optimizer, convergence state, gradient/Hessian information when available;
- random-effect covariance, boundary and singularity diagnostics;
- residual-versus-fit and distribution diagnostics for continuous outcomes;
- influence and leave-one-fish-out results;
- optimizer and allowed random-structure sensitivity;
- missingness/coverage by condition and trial;
- all planned contrasts, including non-significant results.

A failed, rank-deficient, non-converged, disallowed singular, or materially
influence-dependent fit cannot generate a final onset annotation.

## Required artifacts

1. Cohort-authenticated fish-trial model input.
2. Frozen model/contrast configuration with factor references.
3. Global-model coefficient and joint-test table.
4. Block planned-contrast table.
5. Trial contrast table with simultaneous intervals.
6. Onset summary containing `delta_min`, `K`, estimate, interval, and failure
   reason when not localized.
7. Fish-level robustness results.
8. Diagnostic and influence tables/figures.
9. Figure panel-data artifact and publication figure provenance.

## Validation tests

- Recover a known condition-by-time effect in synthetic clustered data.
- Recover no onset under a null trajectory at the declared false-positive rate.
- Distinguish a static pre-existing condition difference from learning-related
  change.
- Detect delayed, gradual, transient, and non-persistent effects.
- Verify reference categories, sign direction, row-order invariance, and equal
  fish weighting.
- Verify that duplicating trials within one fish does not behave like adding
  fish.
- Verify multiplicity and simultaneous-band construction.
- Verify that singular/non-converged fits suppress onset and final annotations.
- Compare block, spline, categorical-trial, and fish-summary recovery at the
  expected paper-scale fish counts before selecting the final route.

## Implementation sequence

1. Freeze outcome, direction, pre-training reference, block family,
   `delta_min`, and `K`.
2. Build the cohort-authenticated fish-trial table and coverage report.
3. Implement Model 1 and the fish-level robustness analysis.
4. Implement both bounded Model 2 candidates and run synthetic recovery tests.
5. Select the trial representation by recovery, diagnostics, and
   interpretability—not smallest p-value.
6. Implement simultaneous trial contrasts and onset uncertainty.
7. Run paper-scale diagnostics and influence analysis.
8. Generate the panel-data artifact and figure only from an accepted fit.

## Exit gate

The result answers a named condition-versus-control change estimand; fish are
the independent population units; the global, block, and trial-localized
claims form one prespecified hierarchy; onset has a frozen persistence and
effect-size definition plus uncertainty; diagnostics pass; and a condition-aware
fish-level analysis agrees or its disagreement is explained.
