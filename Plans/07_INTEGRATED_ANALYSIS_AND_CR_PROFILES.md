# Integrated Analysis and CR Profiles

**Status:** Routine corrected run now schedules available population figures;
learner/paper panels remain blocked on Gate L, approved signed data, and
independent timing validation
**Depends on:** approved metric/outcome contract, frozen primary technical
cohort, and [learner classification](./05_LEARNER_ANALYSIS.md)
**Delivers:** one authenticated post-classification analysis run plus focused
rerender commands for population and learner-stratified outputs

## Purpose

Run the complete single-metric population and learner-stratified analysis from
one immutable cohort and one frozen learner representation. The routine run
must produce the ratio, catch, block, learning-onset, diagnostic, and
learner-stratified outputs without allowing any consumer to construct its own
fish population.

This plan integrates rather than supersedes:

- [Paper Cohort Completion](./01_COHORT_IMPLEMENTATION.md);
- [Learning-Onset Analysis Completion](./03_LEARNING_ONSET_IMPLEMENTATION.md);
- [Learner Representation and Analysis](./05_LEARNER_ANALYSIS.md);
- [Learner Outputs and Validation](./06_LEARNER_OUTPUTS_AND_VALIDATION.md);
- [Learning-Onset Analysis Reference](../docs/analysis/LME_PIPELINE_AND_PARAMETERS.md); and
- [Figures and Reproducible Reporting](./08_FIGURES_AND_REPRODUCIBLE_REPORTING.md).

## Scientific boundary and execution order

The classifier cannot precede every cohort decision: it needs stable technical
membership and stable outcome artifacts as inputs. Conversely, the complete
learner-stratified figure run cannot precede the classifier. Use this order:

```text
approved preprocessing + one selected metric
  -> technical assessment, then exploratory legacy-rule screening
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

The two-stage command records technical evidence and a source-linked
exploratory legacy-rule projection. Learning-onset trial eligibility remains
in its cohort-authenticated inference route; learner model-derived feature
availability remains for later learner analysis.

### Exploratory legacy-rule projection

Preserve the historical behavior-dependent discard procedure as a named,
non-destructive sensitivity policy. It is not technical QC, is not
paper-approved, and must not define the primary cohort or the classifier's
reference population.

The policy records each component separately:

1. the final observed US trial has a recorded US end at least 0.4 s after
   onset and at least one detected bout in 0-5 s;
2. every observed Train US trial has at least one detected bout in 0-5 s;
3. Re-Train US trials, when present, satisfy the same bout rule;
4. each of Late Pre-train (CS 10-14), Early Test (CS 65-69), and Late Test
   (CS 90-94) has at least three distinct trials with a detected bout in the
   baseline window `[-15, 0]` s;
5. each of those three blocks also has at least three distinct trials with a
   detected bout in the assay-specific CR window; and
6. every required component and every selected block must pass (AND logic).

The evaluator reads authenticated artifacts, writes component reasons and an
assessment hash, and never moves source or processed files. It also records
one merged learner-input check; it does not run a classifier. Only technical
columns may inform primary review. After learner labels exist, report the
exploratory status by condition and learner stratum.

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

After a learner manifest is frozen, `run-pipeline` receives the relevant
scientific identities. It has no figure or intake enable switch:

```text
cohort_id
selection_assessment_id / selection_assessment_hash
metric
learner_representation_id / classifier_execution_id
validation_mode
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
  equal-fish aggregation, and coverage masking;
- unconditional raw inventory and verified intake with a durable operational
  ready/incomplete/failed ledger and explicit retry;
- stable-path corrected candidate analysis and bounded routine scheduling of
  detector, per-fish, comparison, and five cohort figure families when their
  authenticated inputs are available;
- a proposed A–G/A–I/A–G/A–G paper panel registry whose blocked reasons are
  included in each run summary. Paper panels are not silently filled by
  descriptive figure families.

Still required before the learner workstream can execute:

- approval of the numerical technical policy, reviewed primary cohort, and
  validation of the two-stage audit on paper-scale inputs;

Gated on the learner workstream:

- the canonical learner/continuous-representation manifest and validated join;
- stratum-aware rendering and validation-mode-specific inference;
- the learner-stratum impact report for the exploratory legacy-rule status;
- learner-figure rendering from a frozen representation and signed independent
  timing panel data; and
- paper-scale execution and figure registration.

## Required tests before completion

- All migrated experiments expose exactly catches 25, 39, 53, 59, and 65.
- The legacy sensitivity evaluator enforces every component and every block,
  records component reasons, and performs no file moves.
- Altering response behavior cannot alter primary technical membership.
- Learner labels cannot alter the primary cohort hash.
- LME trial eligibility and learner model-derived feature failures remain
  explicit without changing technical cohort membership.
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
