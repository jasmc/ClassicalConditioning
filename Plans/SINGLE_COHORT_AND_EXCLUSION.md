# Single Cohort and Fish-Exclusion Boundary

**Status:** Core cohort boundary and selection inventory implemented; paper policy/cohort and full consumer migration remain  
**Scope:** Inventory every fish-level exclusion, decide inclusion once, freeze
one reviewed cohort manifest, and make all population analyses consume it  
**Depends on:** complete paper recording inventory and approved technical QC
policy

Detailed implementation order, artifact contracts, and acceptance gates are in
[Remaining Cohort and Learning-Onset Implementation](./REMAINING_COHORT_AND_LEARNING_ONSET_IMPLEMENTATION.md).

## Objective

Stop different stages from silently analyzing different sets of fish. Preserve
all discoverable recordings through per-recording processing and QC, make one
reviewed fish-level inclusion decision, and apply that decision exactly once at
the boundary between per-recording outcomes and population analysis.

This plan distinguishes:

- **fish exclusion:** membership in a named population;
- **processing failure:** a recording that could not produce required data;
- **trial/row validity:** whether a particular outcome is observed;
- **analysis-specific eligibility:** whether an included fish can contribute to
  one estimand, without redefining the global cohort;
- **display selection:** choosing example fish, never changing inference.

This is also the sole active plan for collapsing every legacy discard/filter
decision into one supported, non-destructive analysis step. No separate
discard plan is required.

## Current implementation findings

### Legacy pipeline

| Stage | Current behavior |
| --- | --- |
| Stage 1 preprocessing/QC | Writes `Fish to discard.txt` and `Discarded_fish_IDs.txt`; may move raw or processed files |
| Stage 3 grouping | Loads the discard list for included/discarded heatmap grids but does not clearly enforce it for condition-level pooled data |
| Stage 4 scaled vigor | `APPLY_FISH_DISCARD = True` by default and filters locally |
| Stage 5 normalized vigor/LME | `APPLY_FISH_DISCARD = False` by default; also applies an independent minimum-trial fish filter |
| Learner variants | Apply different feature-completeness rules; only the WIP variant has explicit discard-aware controls |

The stage-5 minimum-trial implementation retains a fish when any one block
passes the threshold, then keeps its sparse rows from other blocks. This is an
OR-logic bug, not an approved cohort rule.

### Active refactored pipeline

Implemented:

- immutable reviewed cohort schema;
- validated uniqueness and inclusion rules;
- canonical cohort hash;
- `freeze-cohort` and `apply-cohort` commands;
- strict many-to-one application that fails on unknown fish;
- fixture-scale cohort plumbing.

Still missing:

- the main `run-pipeline` selection does not require a cohort ID;
- the candidate runner and metric comparison accept recording lists directly;
- the new `cohort-trial-outcomes` builder is implemented, but has not been run
  on a reviewed paper cohort;
- the new learning-onset model, robustness checks, and response-ratio figures
  authenticate the cohort hash; older metric-comparison, learner, and
  exploratory population consumers still need migration;
- the learning-onset and response-ratio figure routes use the authoritative
  cohort population artifact; older exploratory consumers still need migration;
- the learning-onset route now publishes unified trial eligibility; older
  outcome and learner routes still use local completeness filters;
- paper-scale technical QC policy and reviewed cohort remain unfrozen.

Interim figure status:

- the selected-block, trial-number, and event-aligned ratio commands require a
  frozen cohort ID and now derive population membership from
  `cohort-trial-outcomes`;
- selected-block finite-ratio coverage is represented as analysis eligibility
  in the aggregation frame, not written back as cohort membership; the
  learning-onset route separately publishes durable trial eligibility;

## Authoritative boundary

Use the following pipeline structure:

```text
raw inventory
  -> per-recording intake/preprocessing/metrics/movement/trial outcomes
  -> per-recording QC and processing-status table
  -> UNIFIED SELECTION ASSESSMENT
  -> reviewed immutable cohort manifest
  -> APPLY COHORT ONCE
  -> cohort trial-outcome artifact
  -> population summaries, statistics, learners, and figures
```

Do not discard fish during raw discovery, preprocessing, metric calculation,
movement detection, or per-recording trial-outcome generation. Those stages
must retain evidence needed to decide technical validity and account for
failures.

The one application point should create a versioned artifact such as:

```text
cohort-trial-outcomes
```

It concatenates authenticated per-recording trial outcomes, joins the frozen
cohort manifest once, retains the chosen population, and records the cohort ID,
logical hash, source outcome hashes, included/excluded fish counts, and sample
flow. Every population consumer reads this artifact rather than independently
filtering per-recording files or fish lists.

## Unified selection-assessment step

The supported refactor calculates every fish-, block-, trial-, model-, and
classifier-eligibility decision in one named step, provisionally
`selection-assessment-v1`. It replaces the legacy moments at which fish
or rows disappeared:

1. immediately after preprocessing in Stage 1;
2. when Stages 3 and 4 re-read the Stage-1 discard list;
3. while preparing normalized-vigor/LME data in Stage 5 and, in the refactor,
   the LME used to establish the learning trial; and
4. while preparing features in the learner-classification scripts.

The step runs after authenticated per-recording outcomes/profiles and the
processing-status inventory exist, but before the reviewed cohort is frozen or
any population model/classifier is fitted. One step may calculate several
dispositions; it must not collapse their different scientific meanings into a
single Boolean.

### Inputs and invalidation

Required inputs are the complete recording inventory, processing/QC status,
authenticated per-recording trial outcomes and temporal/bout evidence, resolved
experiment trial/catch/block definitions, the selected metric/outcome contract,
the learning-onset eligibility configuration, and the learner feature and
validation recipe.

Changing a policy, metric, detector, outcome window, LME eligibility setting,
or learner feature recipe creates a new assessment execution and hash.

### Assessment bundle

Publish one immutable bundle:

```text
fish-selection-assessment.parquet
block-selection-assessment.parquet
trial-analysis-eligibility.parquet
selection-sample-flow.parquet
selection-assessment-summary.json
```

The fish table contains separate fields, not one overloaded exclusion flag:

```text
technical_valid
technical_reason_codes
primary_candidate_included
legacy_stage1_pass
legacy_stage1_reason_codes
legacy_stage5_any_block_pass
legacy_stage5_all_required_blocks_pass
learning_onset_eligible_trial_count
learner_classification_eligible
learner_ineligible_reason_codes
```

The reviewed `primary_included` value is frozen only from the approved
technical policy. Every other field is sensitivity or downstream eligibility
evidence and cannot overwrite it.

### Legacy Stage-1 behavior screen

Calculate the complete `legacy-discard-interim-v1` status in this unified step:

- last-US timing/viability and a 0-5 s bout response;
- a 0-5 s bout for every configured Train US trial;
- the same rule for Re-Train trials when present;
- at least three bout-bearing trials in both baseline and assay-specific CR
  windows for each of Late Pre-train 10-14, Early Test 65-69, and Late Test
  90-94; and
- all required checks and blocks pass.

Store every component result and reason. This is a behavior-dependent
legacy-equivalent sensitivity status, never primary technical inclusion.

### Legacy Stage-5 and learning-onset filtering

Reproduce and expose, without silently adopting:

- rows dropped for missing normalized vigor, trial number, block, baseline, or
  response;
- counts of finite rows per fish and block;
- the historical minimum of six trials per fish/block; and
- the historical bug that retained a fish when any one block passed.

Publish both exact `legacy_stage5_any_block_pass` and corrected
`legacy_stage5_all_required_blocks_pass` statuses. Neither defines the primary
cohort. The supported learning-onset analysis consumes its resolved row-level
eligibility from the same bundle: window coverage, finite response and
baseline, positive ratio baseline, and block assignment. Fish with no eligible
model rows remain visible in cohort and sample-flow reporting.

### Legacy learner filtering

Calculate classifier eligibility before classification rather than dropping
fish inside classifier code. For every versioned learner recipe record:

- missing normalized-vigor, trial, ten-trial-block, five-trial-block,
  baseline, or response inputs;
- required epoch totals (the currently configured legacy recipes use at least
  six trials in each selected epoch);
- at least three trials in every required five-trial block;
- acquisition coverage for Early/Late Pre-train versus Late Train/Early Test;
- extinction/recovery coverage for Late Train/Early Test versus Test 5/Late
  Test;
- the intersection across all features selected by that recipe; and
- variant-specific transform requirements, including strictly positive finite
  normalized vigor where a log transform requires it.

Before classifier execution the bundle records eligibility and reasons. After
execution, ineligible fish become `Unclassified`; they never disappear.
Controls retain the Reference role even when classifier diagnostics are
calculated for them.

### Single-boundary requirements

- Never move, rename, or delete raw or processed files.
- Never write or consume active `Fish to discard.txt` or
  `Discarded_fish_IDs.txt` files.
- LME and learner code consume assessment eligibility and may not implement
  private filters.
- Stage 3/4 equivalents may display assessment status but may not re-apply a
  discard list or create a different population.
- Plotting code may filter only explicit display examples, never inference.
- A rule change invalidates the assessment bundle and dependent artifacts, not
  the source data.

## Cohort policy

### Primary behavior cohort

Primary inclusion should depend only on prespecified technical criteria that
cannot be caused by learning, for example:

- source triplet exists and identity is resolvable;
- protocol and condition assignment are valid;
- acquisition/frame integrity passes the approved rule;
- tracking/tail data meet a prespecified recording-level technical threshold;
- required stimulus timing is available;
- the recording successfully produces the canonical trial-outcome artifact.

Do not require response-window movement, a calculable conditional-vigor value,
or a minimum apparent CR. Learning can cause immobility, so those criteria can
condition inclusion on the outcome.

### Named sensitivity populations

Freeze additional columns/population identities in the same manifest rather
than creating new ad hoc discard files:

- strict technical-validity population;
- prespecified minimum-coverage population;
- behavioral-engagement sensitivity population;
- legacy-equivalent population;
- imaging-valid/multimodal population, which must never replace the primary
  behavior cohort.

The legacy-equivalent behavioral population is specified as
`legacy-discard-interim-v1` in the
[integrated cohort/CR-profile plan](./Analysis/4_INTEGRATED_SINGLE_METRIC_COHORT_AND_CR_PROFILES.md).
It records final-US viability, post-US bouts, and the Late Pre-train, Early
Test, and Late Test baseline/CR bout criteria with all-component AND logic.
Because those checks depend on behavior, the unified assessment computes them
only as sensitivity statuses. The primary-cohort review ignores them; after
learner status exists, cross-tabulate their effect by condition and learner
stratum. Never translate failure into `primary_included = false`.

### Trial and outcome eligibility

The unified assessment calculates row-level validity fields before model or
classifier execution and carries them through cohort application. If a model
requires finite baseline/response values or minimum window coverage, generate an
analysis-eligibility table containing:

```text
selection_assessment_id
selection_assessment_hash
analysis_id
experiment_id
recording_id
fish_id
trial_id
metric_id
outcome_id
eligible
ineligible_reason
```

The cohort ID/hash are added by validated join after cohort freeze; they are
not prerequisites for calculating eligibility across the complete inventory.

This table may determine which observations contribute to one model, but it
must not rewrite `primary_included`. Fish with no eligible observations remain
visible in the sample-flow and missingness reports.

Learner-classification eligibility is likewise a downstream status, not a new
global fish discard.

## Work packages

### 1. Complete the exclusion inventory

The machine-readable inventory is maintained in
[EXCLUSION_AND_SELECTION_INVENTORY.csv](./EXCLUSION_AND_SELECTION_INVENTORY.csv).
It distinguishes implemented validity/eligibility handling from active
population consumers that still bypass the cohort boundary.

Create a machine-readable register of every current selection operation with:

```text
module/function
stage
unit: recording | fish | trial | frame | row
criterion
default state
pre-outcome or post-outcome
current effect
replacement disposition
```

The first inventory must cover:

- recording discovery and condition filters;
- intake failures and `continue_on_error` behavior;
- legacy Stage 1 discard criteria and file movement;
- Stage 3/4/5 discard flags;
- stage-5 minimum-trial logic;
- candidate frame/detector validity;
- trial-outcome coverage and missingness;
- candidate model-input filtering;
- metric-comparison complete-trial filtering;
- all learner feature-completeness rules;
- imaging availability and cross-modal joins;
- example-fish/display selectors.

Classify each operation as retained technical QC, cohort decision, row-level
validity, analysis eligibility, display-only selection, or behavior to remove.

### 2. Freeze the policy before reviewing outcome effects

Record the primary and sensitivity criteria, thresholds, missingness handling,
and reviewer process. Generate condition-blinded QC views where practical.
Outcome-dependent behavioral engagement cannot determine the primary cohort.

### 3. Build the unified selection assessment

Calculate all technical, legacy-equivalent, learning-onset, and learner
eligibility statuses for the complete inventory. Reconcile counts and freeze
the assessment hash before cohort review, model fitting, or classification.

### 4. Generate a complete draft manifest

Start from the frozen paper inventory, not from successfully processed fish
alone. Include every expected recording and explicit dispositions for missing,
failed, technically invalid, included, and pending records. Extend the current
manifest or pair it with a coverage table so failed recordings cannot disappear
before cohort review.

### 5. Review and freeze the cohort

Use the implemented `cohort-manifest-v1` validation and hashing. Never edit a
frozen manifest in place; issue a new cohort ID for any change. Preserve a
review copy, reasons, reviewer, timestamp, policy ID, source QC identity, and
condition counts.

### 6. Build the single cohort-applied outcome artifact — implemented

The supported builder now:

1. authenticates the cohort manifest;
2. authenticates every requested per-recording trial-outcome artifact;
3. reconciles the manifest against authenticated trial-artifact availability;
4. applies the selected population exactly once by validated join;
5. writes the cohort-filtered long table and sample-flow artifacts;
6. records cohort ID/hash in its summary and completion marker.

Unknown fish, duplicate fish, condition disagreement, absent expected fish,
and stale source hashes must fail the build.

Paper-scale reconciliation against the external experiment inventory and batch
processing report remains part of the cohort-freeze step because those inputs
are not present in this repository checkout.

### 7. Route every population consumer through that artifact

Change the supported interfaces so the following require `cohort_id` and read
the cohort-applied outcome artifact:

- metric and outcome cohort summaries;
- learning-onset and other mixed-effects models;
- fish permutation and bootstrap;
- learner analysis;
- population and learner-stratified figures;
- result and figure registries.

Artifacts must record cohort ID, logical hash, included fish count by
condition, contributing fish count by outcome, and eligibility-reason counts.

### 8. Remove downstream cohort redefinition

- Remove active `APPLY_FISH_DISCARD`-style switches.
- Do not read `Discarded_fish_IDs.txt` from supported downstream modules.
- Replace local minimum-trial fish filters with explicit coverage/eligibility
  tables governed by the frozen policy.
- Never move or delete raw data as part of exclusion.
- Keep archived legacy scripts unchanged for reproduction; wrap legacy
  comparisons with an explicit legacy cohort manifest rather than editing
  their scientific behavior.
- Prevent plotting code from filtering fish except display-only selectors that
  cannot alter inference.

### 9. Reconcile counts and produce sample flow

For every population result, publish:

```text
inventory fish
intake complete / failed
technical valid / invalid
primary included / excluded
outcome eligible / ineligible
model contributing fish by condition
```

Counts must reconcile exactly. Differences between outcomes are permitted only
when accompanied by named eligibility reasons; they must not appear as an
unannounced cohort change.

## Required tests

- Manifest application is invariant to row order.
- Unknown, duplicate, missing, or condition-mismatched fish fail loudly.
- A fish excluded by the manifest appears in no population artifact.
- Every included fish remains in the cohort flow even when an outcome is
  missing.
- Primary inclusion is unchanged when response-window behavior is altered in a
  synthetic test.
- Outcome eligibility can change without changing cohort membership.
- Every population artifact carries the same cohort ID and logical hash.
- Changing a cohort hash invalidates all cohort-dependent cached artifacts.
- Population counts reconcile from inventory through figures.
- No supported downstream module reads legacy discard text files or exposes an
  independent fish-discard flag.
- Legacy-equivalent and sensitivity cohorts never overwrite the primary cohort.
- `legacy-discard-interim-v1` records component reasons, enforces all required
  blocks, and performs no file moves or deletes.
- Learner status and legacy-discard status cannot alter the primary cohort hash.
- Stage-1, Stage-5/LME, and learner eligibility are produced by one assessment
  execution; supported downstream modules contain no duplicate filtering.
- Exact Stage-5 any-block behavior and corrected all-required-block behavior
  are both visible and never confused.
- Learner feature eligibility is calculated before fitting and an ineligible
  fish becomes Unclassified rather than disappearing.
- Imaging availability cannot alter behavior-primary inclusion.

## Implementation sequence

1. Build and review the complete exclusion inventory.
2. Implement and run the unified selection-assessment step.
3. Decide the primary technical policy and named sensitivity populations with
   paper-scale counts available.
4. Generate, review, and freeze the paper cohort manifest.
5. Build `cohort-trial-outcomes` and its sample-flow artifact.
6. Make model input and the learning-onset analysis consume the unified
   assessment eligibility first.
7. Migrate metric comparison, permutation/bootstrap, figures, and learner
   analysis.
   The post-classification catch/block and complete single-metric run follow
   the integrated cohort/CR-profile plan.
8. Add guards that reject population runs without a frozen cohort identity and
   matching selection-assessment hash.
9. Remove active downstream fish-discard switches and document archived legacy
   differences.
10. Regenerate population artifacts and reconcile all sample sizes.

## Exit gate

Every expected recording has an explicit disposition; one reviewed immutable
manifest defines each named population; the primary manifest is independent of
the hypothesized behavioral response; cohort membership is applied exactly once
to create the population outcome artifact; every downstream result authenticates
the same cohort and selection-assessment hashes; all legacy Stage-1, Stage-5,
LME, and learner filters are evaluated once in the unified assessment;
outcome-specific missingness is reported as eligibility rather than hidden fish
exclusion; and all sample counts reconcile through the final figures.
