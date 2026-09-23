# Remaining Cohort and Learning-Onset Work

> **Archived incomplete on 2026-09-23 after separation of concerns.** Current
> cohort work is in [Paper Cohort Completion](../01_COHORT_IMPLEMENTATION.md);
> current model and onset work is in
> [Learning-Onset Analysis Completion](../03_LEARNING_ONSET_IMPLEMENTATION.md).
> The sections below preserve the former combined plan and are not a live
> status authority.

**Purpose:** Produce a defensible learning curve, determine whether and when
the test condition differs from control, and ensure the same fish are used
throughout the population analysis.

This is the active implementation plan for cohort selection and learning-onset
inference. The retired [learning-onset design](./LEARNING_ONSET_LME.md)
and [cohort boundary plan](./SINGLE_COHORT_AND_EXCLUSION.md) remain
historical evidence. Current fitted behavior and public parameters are in the
[LME pipeline reference](../../docs/analysis/LME_PIPELINE_AND_PARAMETERS.md).
Gate O/S scientific choices are owned by the
[analysis and statistics plan](../02_ANALYSIS_AND_STATISTICS.md) and
recorded when approved in [DECISIONS.md](../DECISIONS.md).

## Implementation status

Implemented in code: the single cohort-applied trial table and sample flow;
trial eligibility; the block interaction model and planned contrasts; the
spline and categorical-trial longitudinal models; fish-bootstrap simultaneous trial bands; the
persistent onset rule; fish-level permutation/bootstrap robustness; leave-one-
fish-out diagnostics; rank-deficiency gating; coefficient, adjusted-trajectory,
coverage, residual, and figure panel-data tables; authenticated result tables;
and the three-component final figure.

Still required before a paper result exists: review and approve the completed
exclusion inventory, approve and freeze the paper cohort, choose the primary outcome and meaningful
effect threshold, install and run the project statistical environment,
calibrate the method on synthetic datasets, inspect diagnostics, and run the
approved configuration on the paper data. Older learner and exploratory
population consumers also still need migration to the cohort boundary.

## What the completed analysis will answer

The completed work will answer three related questions:

1. **Does learning occur?**  
   Does the response in the test condition change across training differently
   from the control condition?

2. **By which block is learning established?**  
   In which prespecified group of trials is there stable evidence that the
   test condition differs from control relative to pre-training?

3. **At which trial can learning first be localized?**  
   What is the earliest trial at which the test-versus-control difference is
   large enough, statistically supported, and persistent across consecutive
   trials?

The analysis will not define onset as the first isolated trial with `p < .05`.

## Analyses that will be delivered

### Analysis 1 — descriptive learning trajectory

**Question:** What do the fish and condition trajectories look like across
trials?

For every fish and CS trial:

```text
activity ratio = response activity / pre-CS baseline activity
```

Lower ratios mean stronger motor suppression. The data will first remain at
fish-by-trial level. Condition summaries will then give each fish equal weight.

**Outputs:**

- fish-by-trial table;
- number of contributing fish per condition and trial;
- median and IQR by condition and trial;
- explicit missingness and eligibility reasons.

**Status:** Implemented. Durable fish and condition panel data, coverage, and
trial eligibility are published with the learning-onset analysis.

### Analysis 2 — primary block-level mixed model

**Question:** Is there a test-versus-control learning effect, and by which
prespecified block is it established?

Recommended model:

```text
log_response ~ log_baseline + condition * block
groups = fish
random effects = fish intercept + scaled trial slope
```

The model will test:

- the overall condition-by-block interaction;
- test versus control within each planned block;
- change in the test-versus-control contrast relative to pre-training;
- the primary late-training/test contrast.

The block contrasts will use a prespecified multiplicity correction. The first
supported block will be reported as the block by which learning is established.

**Outputs:**

- overall interaction test;
- adjusted effect and confidence interval for each block;
- corrected p-values for the limited planned block family;
- first supported block, or “not established by a block”;
- model diagnostics and fish counts.

### Analysis 3 — trial-resolution longitudinal mixed model

**Question:** At which trial does the test condition begin to differ from
control relative to pre-training?

One longitudinal model will be fitted across all trials. The primary candidate
is:

```text
log_response ~ log_baseline + condition * spline(trial)
groups = fish
random effects = fish intercept + scaled trial slope
```

A categorical-trial model may be used as a sensitivity analysis if the sample
size supports it. The analysis will not fit an unrelated LME at every trial.

The fitted model will generate a test-versus-control learning contrast at each
scheduled trial. Confidence intervals will be simultaneous across the complete
trial family, so scanning many trials does not inflate the evidence.

**Outputs:**

- adjusted control and test trajectories;
- test-versus-control change at every trial;
- pointwise and simultaneous confidence intervals;
- contributing fish counts at every trial;
- trial-level model diagnostics.

### Analysis 4 — learning-onset rule

**Question:** What is the earliest persistent trial-level difference?

Before fitting, define:

- `delta_min`: the smallest meaningful test-versus-control learning effect;
- `K`: the required number of consecutive supported trials, initially proposed
  as three.

Recommended rule:

> Learning onset is the earliest trial for which the lower simultaneous 95%
> confidence bound exceeds `delta_min` for `K` consecutive eligible trials.

Fish-level bootstrap resampling will show how stable the onset is. Some
bootstrap samples may have no localized onset; those results must remain in the
uncertainty summary.

**Outputs:**

- onset trial, when localized;
- persistence run used to establish onset;
- bootstrap distribution and interval;
- probability that onset is not localized;
- explicit failure reason when no onset can be claimed.

### Analysis 5 — simple fish-level robustness check

Each fish will contribute one prespecified change score, such as late test
minus pre-training on the signed CR scale. The change scores will be compared
between test and control with a condition-label permutation test and a fish
bootstrap interval.

This checks whether the main conclusion is visible without relying on the full
longitudinal model. It will not be used to estimate an exact onset trial.

## Figures that will be delivered

### Figure 1 — learning trajectory across trials

**Axes:**

- x: CS trial number;
- y: response activity divided by pre-CS baseline activity.

**Visible elements:**

- faint trajectory for every fish;
- condition median at every trial;
- fish IQR as a descriptive ribbon;
- contributing fish count;
- block and phase boundaries;
- statement that lower ratios mean stronger suppression.

This is the intuitive data figure. Its IQR is descriptive and will not be
presented as the uncertainty of the condition difference.

### Figure 2 — adjusted test-versus-control effect across trials

**Axes:**

- x: CS trial number;
- y: adjusted test-versus-control learning contrast, oriented so positive means
  stronger learning in the test condition.

**Visible elements:**

- estimated trial contrast;
- simultaneous 95% confidence band;
- zero-effect reference;
- `delta_min` reference;
- onset marker and interval, only if the onset rule and diagnostics pass.

This is the figure that supports the claim about the trial at which learning is
established.

### Figure 3 — planned block contrasts

A compact point-and-interval or forest plot will show the adjusted
test-versus-control change in each prespecified block. It will identify the
first supported block and the primary late-test contrast.

### Supporting descriptive figures

The following already implemented figures remain useful but do not establish
onset:

- selected Late Pre-train, Early Test, and Late Test ratios;
- event-aligned response/baseline trajectories around CS onset.
- pooled configured-catch scaled-total-activity profiles; and
- declared ten-trial-block scaled-total-activity profiles.

The catch/block substrate and its post-classification orchestration are governed
by the
[integrated single-metric cohort/CR-profile plan](../07_INTEGRATED_ANALYSIS_AND_CR_PROFILES.md).
These descriptive figures use the frozen cohort but do not define it.

## Fish exclusion and cohort work

This is a separate housekeeping requirement that protects all analyses above.

### Required outcome

One reviewed cohort determines fish membership once. All population analyses
then read the same cohort-applied trial table. A model or figure must not create
its own fish list.

Primary inclusion uses a prespecified technical policy independent of response
strength, movement during a response window, learner status, or imaging
availability. Processing failures and trial/outcome ineligibility remain
separate from fish exclusion. The [selection inventory](../EXCLUSION_AND_SELECTION_INVENTORY.csv)
records active bypasses and legacy rules; the technical command behavior is in
the [discarding assessment guide](../../docs/analysis/DISCARDING_ASSESSMENT.md).

The paper inventory and draft manifest must retain every expected recording,
including missing, incomplete, failed, technically invalid, and pending cases.
The reviewed manifest records reasons, reviewer, timestamp, policy identity,
source QC identity, and condition counts. A changed decision receives a new
cohort ID rather than editing a frozen manifest. Named sensitivity populations
must be explicit and may include stricter technical/coverage, legacy
behavioral, and imaging-valid groups; none replaces the behavior-primary
cohort. The exploratory legacy screen and merged learner-input prerequisite
never set `primary_included`.

The population boundary authenticates the cohort manifest and each source
trial-outcome artifact, rejects unknown/duplicate/missing or
condition-mismatched fish and stale hashes, applies membership once, and
publishes sample flow. Every population consumer must use that artifact and
the matching cohort and assessment hashes. Trial eligibility is reason-coded
per outcome without changing fish membership; fish with no eligible rows stay
visible in flow reports. Counts must reconcile from inventory through models
and figures. No supported downstream command may read active legacy discard
lists, accept an independent population fish list, or discard fish inside
plotting. These invariants need row-order, mismatch, response-perturbation,
hash-invalidation, and sample-flow tests before the exit gate passes.

### Steps

1. List every current place where recordings, fish, trials, or rows are
   discarded.
2. Classify each rule as technical fish exclusion, processing failure,
   trial-level validity, analysis eligibility, or display-only selection.
3. Run `assess-discarding`: technical readiness first, then the exploratory
   source-linked preprocessing screen and merged learner-input check.
4. Decide the primary technical inclusion policy without looking at learning
   strength.
5. Freeze the reviewed paper cohort from technical columns only.
6. Build one authenticated `cohort-trial-outcomes` table.
7. Store excluded, failed, missing, and included fish in a sample-flow table.
8. Store outcome-specific missingness in an eligibility table instead of
   changing cohort membership.
9. Make the LME, permutation/bootstrap, learner analysis, and population
   figures consume the cohort table and matching selection-assessment hash,
   with no private downstream discard filters.

### Outputs

```text
cohort-trial-outcomes.parquet
cohort-sample-flow.parquet
analysis-eligibility.parquet
```

Every final result will report:

- frozen cohort identity and hash;
- total included fish by condition;
- contributing fish by trial or block;
- ineligible observations and reasons.

## Implementation order

### Step 1 — make the cohort input authoritative

- complete the discard/filter inventory;
- run the two-stage technical and exploratory assessment bundle; keep LME
  trial eligibility in the cohort-authenticated inference route;
- freeze the paper cohort;
- build `cohort-trial-outcomes` and sample flow;
- stop population commands from accepting arbitrary fish lists.

**Done when:** every downstream analysis receives the same fish population and
selection-assessment hash, no downstream consumer re-filters privately, and all
counts reconcile.

### Step 2 — freeze the statistical choices

Decide and record:

- primary metric and outcome;
- control and test conditions;
- baseline and response windows;
- pre-training reference;
- blocks/phases;
- effect direction;
- `delta_min` and `K`;
- block multiplicity method;
- spline complexity;
- random-effects fallback;
- bootstrap size and seed;
- diagnostic failure rules.

**Done when:** one configuration exists before the paper models are fitted.

### Step 3 — validate the block analysis for paper use

- run the cohort-authenticated model input on the reviewed paper population;
- confirm the condition-by-block model and planned contrasts match approved
  Gate O/S settings;
- review the fish-level robustness result and diagnostic artifacts.

**Done when:** the analysis can say whether learning occurred and by which
block, or clearly report that it was not established.

### Step 4 — validate the trial and onset analysis for paper use

- fit the implemented condition-aware longitudinal route on the reviewed
  cohort and approved configuration;
- calibrate simultaneous trial contrasts, the persistent onset rule, and fish
  bootstrap uncertainty at the paper sample size;
- review the categorical-trial sensitivity model's numerical feasibility.

**Done when:** the analysis can report an onset trial with uncertainty or state
that onset was not localized.

### Step 5 — approve final panel data and figures

- verify that saved panel data contain every plotted fish value, summary,
  contrast, interval, and annotation;
- review the three implemented figure components against the approved analysis;
- verify cohort, configuration, model, and source hashes in provenance;
- confirm diagnostic failure suppresses inferential annotations.

**Done when:** every visible element can be reproduced from saved panel data.

## Required validation

Deterministic tests now cover null/pre-existing differences, gradual and
transient persistence behavior, sign direction, row-order invariance,
experiment-scoped fish identity, equal-fish weighting, and Holm adjustment. A
clustered synthetic spline-recovery test is included when the optional mixed-
model dependencies are installed. Repeated simulation for false-onset rate,
coverage, missingness, influential-fish, and singular-fit calibration at the
paper sample size remains required before the paper run.

Before applying the method to the paper data, test synthetic datasets with:

- no learning;
- a pre-existing condition difference but no learning;
- abrupt persistent learning;
- gradual learning;
- transient non-persistent differences;
- missing trials and unequal coverage;
- one influential fish;
- a singular random-slope structure.

The tests must show acceptable false-onset behavior, interval coverage, onset
recovery, and correct failure handling at the expected fish and trial counts.

## What is deliberately not part of this plan

- inventing a new learner classifier;
- imaging analyses;
- new behavioral metrics;
- a separate LME at every trial;
- event-aligned functional inference;
- unrestricted model searching;
- redesigning the entire repository.

## Completion criterion

The work is complete when the same frozen fish cohort feeds all analyses, the
condition-by-block model answers whether learning occurred, the longitudinal
model estimates test-versus-control effects across trials, the persistent rule
either localizes onset with uncertainty or explicitly fails to do so, and the
three figure components are reproducible from saved panel data.
