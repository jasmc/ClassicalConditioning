# Paper Cohort Completion

**Status:** Cohort infrastructure and two-stage assessment implemented on
fixtures; numerical paper policy, reviewed cohort, and consumer migration open.

**Scope:** Decide fish membership once from label-independent technical
evidence, freeze it, and make every population consumer use the same cohort.
Learning-onset model choices and validation are owned by
[Learning-Onset Implementation](./03_LEARNING_ONSET_IMPLEMENTATION.md).

## Boundary and evidence

The [selection inventory](./EXCLUSION_AND_SELECTION_INVENTORY.csv) records
active, legacy, and deferred selection operations. The implemented
[`assess-discarding` guide](../docs/analysis/04_DISCARDING_AND_SELECTION.md) specifies
the technical evidence and exploratory legacy-rule projection. The latter,
including its merged learner-input prerequisite, cannot set primary membership.

Primary inclusion must use a prespecified technical policy independent of
response strength, response-window movement, learner status, and imaging
availability. Processing failure, fish exclusion, trial/outcome eligibility,
and display selection are distinct. LME trial eligibility remains in the
cohort-authenticated inference route; it cannot change the frozen fish set.

The paper inventory and draft manifest must account for every expected
recording, including missing, incomplete, ambiguous, failed, technically
invalid, and pending cases. The reviewed manifest records inclusion reasons,
reviewer, timestamp, policy identity, source QC identity, and condition
counts. A change to a frozen decision requires a new cohort ID. Named
sensitivity populations may be stricter technical/coverage, behavioral, or
imaging-valid groups; none replaces the behavior-primary cohort.

## Remaining work

1. Reconcile the selection inventory with all current population consumers,
   including manual recording selectors, older metric comparison and
   exploratory inference, learner analysis, and plotting.
2. Validate the technical and exploratory assessment on the complete paper
   inventory. Reconcile source, processing, rule, and fish counts, including
   recordings that failed before trial outcomes were produced.
3. Approve the numerical technical policy before reviewing outcome effects.
   Record primary and sensitivity criteria, thresholds, missingness handling,
   and reviewer process. Use condition-blinded QC where practical.
4. Generate the complete draft manifest, review it, and freeze the paper
   cohort. Never edit a frozen manifest in place.
5. Build the authenticated `cohort-trial-outcomes` table and sample-flow
   artifacts for that cohort. Migrate all supported population consumers to
   the same cohort-applied table and matching cohort/assessment hashes.
6. Remove downstream fish-list, discard-file, and local minimum-trial
   population switches. Retain outcome-specific, reason-coded row eligibility
   without silently changing fish membership.
7. Reconcile inventory, intake, technical review, inclusion, outcome
   eligibility, and contributing-fish counts through models and figures.

The implemented cohort boundary authenticates the manifest and each source
trial-outcome artifact, applies membership once by validated join, and
publishes sample flow. A paper run and full consumer migration remain open.

## Required evidence and exit gate

- Unknown, duplicate, missing, condition-mismatched fish and stale artifact
  hashes fail rather than disappearing.
- Manifest application is row-order invariant. Changing response behavior or
  learner labels cannot change the primary cohort hash.
- An included fish with no eligible outcome rows remains visible in sample
  flow; every ineligible row has a reason.
- Cohort hash changes invalidate dependent results. Every supported
  population table, model, and figure carries the matching cohort identity.
- No supported downstream command reads active legacy discard lists or
  constructs a private inference population.
- Counts reconcile from complete inventory through final figures, with named
  eligibility reasons for any outcome-specific difference.

The work is complete when Gate C0 is approved, a reviewed immutable C1 cohort
is frozen, all population consumers authenticate that cohort, and the sample
flow reconciles. Earlier detailed cohort plans remain in Git history; this
plan owns the current requirements.
