# Integrated Single-Metric Cohort Analysis and CR-Profile Figures

**Status:** Shared label-independent profile substrate implemented; automatic
paper run waits for Gate L and a frozen learner-representation manifest
**Depends on:** approved metric/outcome contract, frozen primary technical
cohort, and [learner classification](./2_LEARNER_CLASSIFICATION.md)
**Delivers:** one authenticated post-classification analysis run plus focused
rerender commands for population and learner-stratified outputs

## Purpose

Run the complete single-metric population and learner-stratified analysis from
one immutable cohort and one frozen learner representation. The routine run
must produce the ratio, catch, block, learning-onset, diagnostic, and
learner-stratified outputs without allowing any consumer to construct its own
fish population.

This plan integrates rather than supersedes:

- [Single Cohort and Fish-Exclusion Boundary](../SINGLE_COHORT_AND_EXCLUSION.md);
- [Learner Classification and Stratified Analysis](./2_LEARNER_CLASSIFICATION.md);
- [Learner-Stratified Outputs and Validation](./3_LEARNER_STRATIFIED_OUTPUTS.md);
- [Learning-Onset Inference](../LEARNING_ONSET_LME.md); and
- [Figures and Reproducible Reporting](./FIGURES_AND_REPRODUCIBLE_REPORTING.md).

## Scientific boundary and execution order

The classifier cannot precede every cohort decision: it needs stable technical
membership and stable outcome artifacts as inputs. Conversely, the complete
learner-stratified figure run cannot precede the classifier. Use this order:

```text
approved preprocessing + one selected metric
  -> unified technical/legacy/LME/learner eligibility assessment
  -> label-independent primary technical cohort
  -> authenticated cohort trial outcomes and temporal profiles
  -> frozen learner representation/classification manifest
  -> integrated population + learner-stratified analysis run
  -> registered figures and release evidence
```

The primary population LME remains label-independent even when it is executed
by the final post-classification command. Learner labels annotate the frozen
cohort; they never add or remove primary-cohort fish.

## Cohort and legacy-discard policy

### Primary cohort

The primary cohort uses only prespecified technical criteria that cannot be
caused by learning. Missing or immobile behavior is represented by outcome or
classification eligibility, not primary fish exclusion.

All technical inclusion, legacy sensitivity, learning-onset eligibility, and
learner-feature eligibility are calculated together by the unified assessment
defined in the single-cohort plan. LME and classifier modules consume those
results; they do not discard again.

### `legacy-discard-interim-v1`

Preserve the historical behavior-dependent discard procedure as a named,
non-destructive sensitivity policy. It is not technical QC, is not
paper-approved, and must not define the primary cohort or the classifier's
reference population.

The policy records each component separately:

1. the final scheduled US trial has a recorded US end at least 0.4 s after
   onset and at least one detected bout in 0-5 s;
2. every configured Train US trial has at least one detected bout in 0-5 s;
3. Re-Train US trials, when present, satisfy the same bout rule;
4. each of Late Pre-train (CS 10-14), Early Test (CS 65-69), and Late Test
   (CS 90-94) has at least three distinct trials with a detected bout in the
   baseline window `[-15, 0]` s;
5. each of those three blocks also has at least three distinct trials with a
   detected bout in the assay-specific CR window; and
6. every required component and every selected block must pass (AND logic).

The unified evaluator reads authenticated artifacts, writes component reasons
and a policy hash, and never moves, renames, or deletes source or processed
files. It computes this status before cohort review, but the review may use only
technical columns for primary membership. After learner labels exist, report
the legacy status by condition and learner stratum. It is a sensitivity
population and sample-flow cross-tab, not a replacement cohort.

## Experiment-owned catch and block definitions

For `allDelay`, `all3sTrace`, and `all10sTrace`, CS trials
`25, 39, 53, 59, 65` are catch trials. Trial 65 is both the first Early Test
trial and a catch trial. Catch membership and declared ten-trial blocks come
from `ExperimentSpec`; commands and plotting modules must not maintain private
trial lists.

## CR-profile estimands

Catch and block temporal profiles use `Scaled total activity` and the existing
temporal-profile coverage field. Values below the configured coverage threshold
are missing, never zero-filled.

Aggregation order is fixed:

```text
trial + fish + time bin
  -> trials pooled within fish and profile group
  -> one fish value per group and time bin
  -> condition/learner-stratum median and fish IQR
```

Every fish has equal cohort weight. The catch figure pools all five configured
catch trials within fish. The block figure uses every declared CS ten-trial
block. Panel data report contributing trials and fish for every time bin.

Same-data learner/non-learner profiles are descriptive classifier
characterization. Learner-group inference requires the held-out, cross-fitted,
or independent validation mode frozen at Gate L.

## Final integrated command contract

After a learner manifest is frozen, `run-pipeline` receives:

```text
cohort_id
selection_assessment_id / selection_assessment_hash
cohort_metric
learner_representation_id / classifier_execution_id
validation_mode
run_figures = true
```

The command authenticates one selection-assessment hash, one cohort hash, one
selected metric, one learner manifest hash, and all source outcome/profile
hashes. It then produces:

1. primary and named-sensitivity sample flows, including the legacy-discard
   cross-tab;
2. cohort-authenticated single-metric comparison/summary;
3. the complete label-independent learning-onset analysis and diagnostics;
4. selected-block, trial-ratio, and event-aligned-ratio figures;
5. pooled catch and declared ten-trial block profiles;
6. learner-representation sample flow and learner-stratified versions of the
   applicable temporal and trajectory outputs; and
7. figure/result provenance proving common cohort, metric, classifier, and
   validation identities.

Dedicated commands remain available for focused rerenders. They must enforce
the same authentication and may not accept arbitrary fish lists.

## Implementation status

Implemented now:

- experiment-owned catch flags for CS 25, 39, 53, 59, and 65 in all three
  migrated experiments;
- experiment-owned declared CS block resolution;
- coverage-masked, trials-within-fish then equal-fish aggregation for scaled
  temporal profiles;
- focused cohort catch-profile and block-profile figure commands; and
- tests for catch membership, Early Test catch 65, block resolution,
  equal-fish aggregation, and coverage masking.

Still required before the learner workstream can execute:

- the unified `selection-assessment-v1` implementation covering legacy
  Stage 1, Stage-5/LME, and learner-feature eligibility without file moves or
  downstream duplicate filtering;

Gated on the learner workstream:

- the canonical learner/continuous-representation manifest and validated join;
- stratum-aware rendering and validation-mode-specific inference;
- the learner-stratum impact report for the already-computed
  `legacy-discard-interim-v1` status;
- `run-pipeline` orchestration of the entire cohort/learner figure family; and
- paper-scale execution and figure registration.

## Required tests before completion

- All migrated experiments expose exactly catches 25, 39, 53, 59, and 65.
- The legacy sensitivity evaluator enforces every component and every block,
  records component reasons, and performs no file moves.
- Altering response behavior cannot alter primary technical membership.
- Learner labels cannot alter the primary cohort hash.
- LME and learner modules reproduce the eligibility rows from the unified
  assessment and contain no private discard implementation.
- Catch/block aggregation is trial-within-fish then equal-fish, with explicit
  coverage and counts.
- Every table, model, and figure in one integrated run carries identical cohort,
  selection-assessment, metric, learner-manifest, validation-mode, and source
  hashes as applicable.
- Diagnostic failure suppresses inferential annotations without suppressing
  descriptive sample-flow and profile outputs.

## Exit gate

One routine post-classification command regenerates all registered population
and learner-stratified outputs from a single frozen technical cohort and a
single selected metric. Primary membership is independent of learning
expression; behavior-dependent legacy screening is visible only as a named
sensitivity analysis; all catches include trial 65; and every output is
traceable to common authenticated identities.
