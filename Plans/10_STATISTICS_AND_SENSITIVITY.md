# Step 10 — Population Statistics, Diagnostics, and Sensitivity Analyses

**Status:** In progress (workshop recorded; Gate S open)  
**Done:** Workshop; model-input; LME; fish-permutation; fish-bootstrap.  
**Open:** Gate S freeze; planned contrasts.  
**Change class:** Behavior-preserving legacy branch plus candidate scientific correction
**Depends on:** Step 09 and scientific gate S  
**Unlocks:** Confirmatory results and final paper figures

## Objective

Regenerate legacy statistics, define and run one prespecified corrected
inferential strategy, expose all model diagnostics, and quantify robustness to
scientifically plausible choices.

Before freezing Gate S, complete Work Package **10.0** (methodology workshop):
criticize legacy and default LME routes and consider drastically different
inferential families. Working record:
[STATISTICS_METHODOLOGY_WORKSHOP.md](./STATISTICS_METHODOLOGY_WORKSHOP.md).

## Scientific gate S

Freeze before confirmatory execution:

- primary scientific estimand;
- primary cohort;
- primary outcome;
- time/phase representation;
- model family and link;
- fixed effects;
- random-effects structure;
- reference categories;
- planned contrasts;
- multiple-comparison families;
- bootstrap unit, iterations, seed, and interval;
- convergence and singularity criteria;
- missing-data assumptions;
- sensitivity analyses;
- failure handling.

Gate S also names the validation mode and the exact observations that can
support the primary confirmatory result.

## Confirmatory dataset rule

If anticipatory group/phase effects contributed to metric selection, the
primary confirmatory p-value or interval must use:

- the untouched permanent confirmation set; or
- outer-test predictions/effects from the frozen nested fish-level
  cross-fitting procedure.

A model refitted to the complete cohort after effect-informed metric selection
may provide a descriptive full-data estimate, but it is not labeled as the
independent primary confirmation. If metric selection was entirely
outcome-blind, that fact and its evidence must be recorded before a full-cohort
confirmatory model is permitted.

## Work packages

### 10.0 Post-refactor statistics-methodology workshop

Do not freeze Gate S while only “implement the LME from DECISIONS.md” is on the
table. Parallel to the learner workshop (Step 11.0), revisit inference from
first principles:

- define the primary estimand (population conditioning effect, trajectory
  shape, bout hazard, predictive suppression, etc.);
- criticize the legacy Mann-Whitney / per-trial LME / ratio / weak-bootstrap
  stack as a scientific design, not only as buggy code;
- criticize the default fish-random-intercept LME plan, including when another
  family is more honest for probability, counts, curves, or causal contrasts;
- compare drastically different families (fish-level permutation, hierarchical
  Bayes, GEE/marginal GLM, functional/GAM curves, bout point-process, HMM
  occupancy, design-based / randomization inference, multivariate multi-metric,
  pure predictive evaluation);
- choose one engineering default for shared six-metric plumbing and at least
  one alternative family strong enough to falsify a fragile LME story;
- record rejection reasons and kill criteria before inspecting confirmatory
  fits on paper-scope data.

This workshop may keep, demote, or replace mixed-effects as the primary route.
Fixture fish are for pipeline tests only.

### 10.1 Reproduce legacy statistics

Extract current:

- block and phase nonparametric tests;
- trial-by-trial tests;
- correction procedures;
- mixed-effects formulas;
- optimizers and jitter;
- bootstrap behavior.

Save warnings and convergence behavior. Legacy equivalence does not convert a
fragile model into an approved model.

### 10.2 Define model-input artifacts

Every model receives a frozen table containing only explicit columns and rows.
Save:

```text
model_input_id
source outcome artifact
cohort hash
outcome definition
included rows/fish/trials
transforms
reference categories
```

Model fitting must be independently reproducible from this artifact.

**Local fixture progress:** `candidate-model-input-v1` publishes the shared
authenticated table used by both `candidate-mixed-effects-v1` and
`candidate-fish-permutation-v1` (`candidate-model-input` CLI). Cohort hash and
Gate S reference categories remain open.

### 10.3 Implement primary longitudinal model

The exact model is a scientific decision. Candidate structures may cover:

- continuous positive/skewed total activity;
- binary or binomial movement probability;
- proportion/occupancy outcomes;
- conditional positive intensity;
- repeated phases/blocks/trials within fish.

Use one primary model rather than unrelated per-trial models as the main
evidence, unless the scientific specification explicitly justifies otherwise.

### 10.4 Implement fish-aware uncertainty

Options:

- fish-level bootstrap;
- hierarchical bootstrap resampling fish first and trials second;
- model-based intervals with a separate fish-level robustness analysis.

Frames are never independent bootstrap units for population claims.

**Local fixture progress:** `candidate-fish-bootstrap-v1` implements fish-unit
percentile CIs on the shared early→late estimand (`candidate-fish-bootstrap`
CLI). Hierarchical bootstrap remains open.

### 10.5 Implement planned contrasts

Record:

- contrast name;
- estimand;
- coefficient combination;
- comparison family;
- raw and adjusted result;
- direction expected, if prespecified.

Use joint tests for multi-parameter interactions where appropriate.

### 10.6 Enforce diagnostics

Check and save:

- optimizer success;
- convergence flag;
- gradient or available convergence evidence;
- boundary variance;
- singular covariance;
- Hessian status;
- residual/outlier diagnostics appropriate to the model;
- observations and fish;
- model rank;
- warnings.

A diagnostic failure returns a failed result requiring disposition. It does not
return a successful table with missing coefficients or a string in place of a
model.

### 10.7 Run sensitivity analyses

Minimum set:

- primary versus legacy cohort;
- alternative technical-coverage threshold;
- total activity versus movement probability versus conditional intensity;
- selected metric versus legacy point-15 benchmark;
- selected metric versus the manuscript-described segment-speed sum;
- reasonable window variants;
- alternative baseline treatment;
- fish-level versus hierarchical bootstrap;
- robust/nonparametric fish-level summary;
- model random-effects alternatives justified in advance;
- relevant smoothing/threshold variants retained from metric validation;
- influence of rig/day or identified batch structure.

Treat these as robustness evidence, not an unrestricted search for
significance.

### 10.8 Build statistical result schema

```text
result_id
analysis_recipe
scientific_status
model_id
model_input_id
cohort_id
outcome_id
metric_id
formula
grouping_variable
reference_categories
random_effects
contrast_id
estimate
standard_error
confidence_interval
p_value
adjusted_p_value
correction_family
fish_count
observation_count
converged
diagnostic_status
```

## Required tests

- Fixed synthetic coefficient direction
- Correct group/reference coding
- Correct number of fish and observations
- Deterministic bootstrap with fixed seed
- Equal fish weighting
- Joint interaction test behavior
- Multiplicity-family membership
- Diagnostic failure propagation
- Input row order invariance
- Legacy regression values in pinned environment

## Deliverables

- Frozen model-input artifacts
- Legacy statistical equivalence report
- Primary model specification
- Model result and diagnostic artifacts
- Planned contrast table
- Bootstrap outputs
- Sensitivity-analysis matrix
- Effect-size and manuscript-impact summary

## Exit gate

The primary result can be independently regenerated from a versioned
trial-outcome artifact; diagnostics pass or have an approved failure
disposition; validation-partition rules are satisfied; and all planned
contrasts and sensitivities are reported.

## Failure conditions

Do not declare confirmatory results if:

- the primary model was selected after inspecting alternatives;
- convergence/singularity is ignored;
- fish identity is absent from repeated observations;
- bootstrap units are frames or unqualified rows;
- multiplicity families are undefined;
- model inputs cannot be frozen and counted.
- the primary result reuses effect-informed metric-selection observations
  without a valid nested/cross-fitted analysis.
