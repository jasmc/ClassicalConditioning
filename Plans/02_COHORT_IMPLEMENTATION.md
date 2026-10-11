# Fish Exclusion, Eligibility and Paper Cohort



**Scope:** Decide fish membership once from label-independent technical
evidence, freeze it, and make every population consumer use the same cohort.
Learning-onset model choices and validation are owned by
[Learning-Onset Implementation](04_LEARNING_ONSET_IMPLEMENTATION.md).

## Boundary and evidence

The [selection inventory](../docs/analysis/audits/EXCLUSION_AND_SELECTION_INVENTORY.csv) records
active, legacy, and deferred selection operations. The implemented
[`assess-discarding` guide](../docs/analysis/reference/04-discarding-and-selection.md) specifies
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

## Current automation and historical plan

Automatic assessment already exists: the corrected `run-pipeline` route invokes `assess-discarding` after per-fish processing and before exploratory comparison. A focused assessment command is also available. These technical/exploratory tables do not remove raw/processed files, publish an approved cohort or silently authorize paper exclusions. Frozen-cohort application is a separate downstream operation. Whether a job is currently running requires its run ledger; code presence alone does not establish an active job.

This work was already planned on 2026-09-02 in Git revision `83cbeee`, `Plans/steps/08_FULL_REPROCESSING_QC_AND_COHORT.md`. That version separated local fixture plumbing from paper C0/C1 review. Its decision register deferred the historical per-block OR/AND completeness issue pending real cohort counts. The present Plan 02 is the successor, before Analysis and Statistics, not a new competing plan. See the [current/historical route audit and decision proposal](../docs/analysis/audits/fish-selection-policy-review-2026-10-11.md).

## Author-requested consolidation before later review

On 2026-10-11 the author requested combining **all legacy discarding steps**, with review later. The [combined review package](../docs/analysis/audits/fish-selection-policy-review-2026-10-11.md) now covers preprocessing, propagation through grouping/scaled plots, normalized-vigor row/completeness and paired-test filters, all four learner variants and display selectors. Preserve each original rule, order, default, threshold/configuration, AND/OR logic and bypass; do not silently intersect all rules into a new primary population. Later review must approve the disposition and numerical policy before activation. Current automatic assessment omits some of these later families; documentation consolidation does not change that execution.

## Policy review to finalize before paper inference

Retain the already decided boundary: one label-independent technical primary cohort. Finalize the unresolved numerical criteria and approved implementation through this sequence:

1. Inventory every exclusion/eligibility consumer and its current default, population effect, rationale, threshold and provenance. Include post-preprocessing checks, legacy discard-list propagation, normalized-vigor completeness, model trial eligibility, learner feature failures and manual example selectors.
2. Create a rule-by-rule review table assigning each check to technical fish membership, processing disposition, outcome-specific trial/block eligibility, learner eligibility, named sensitivity analysis or display-only selection. US-response/CR-bout strength and classifier labels cannot determine primary membership under the existing decision.
3. On the complete authenticated inventory, summarize technical evidence and compare candidate tracking/timing/coverage thresholds and historical completeness rules by experiment/condition. Report candidate fish/trial/block counts and uncertainty without using favorable condition effects to choose thresholds. Evaluate whole-session versus required-window coverage and consequences of partial loss; do not turn an unprocessed/unrequested case into a technical exclusion automatically.
4. Agree the actual technical thresholds, handling of partial recordings, reevaluation of processing failures, missing-data statuses and review process. For each analysis define its required windows/blocks and minimum usable information separately. Test historical OR versus all-required-block logic on real counts; approve the intended estimand and row eligibility instead of silently globally removing fish.
5. Record author-approved C0 policy, parameters, scope and evidence in the scientific decision register and a versioned policy definition. Draft defaults (`min_matched_frames=1`, `min_valid_frame_fraction=0.0`) are plumbing defaults, not approved paper thresholds. This plan and the draft review do not freeze them.
6. Review the per-fish dispositions and freeze C1 with the explicit cohort command. All expected recordings retain a status/reason; confirmed exclusions remain in sample flow. Publish numerical manifests on the external artifact root. Changing an accepted policy or cohort creates a new identity and invalidates dependent results.
7. Migrate supported population consumers to that cohort once. Apply analysis-specific eligibility as auditable reason-coded masks, including contributors with no usable outcome; forbid plotting/statistics scripts from constructing private fish lists. Retain original historical routes and frozen outputs for reproduction.

**Policy exit:** The approved rule table distinguishes membership from row/model/display eligibility, every numerical threshold has author-reviewed evidence, one C1 manifest is authoritative, and all downstream counts reconcile. Until that evidence is available, the numerical policy remains open; no automatic exclusion switch is newly enabled by this document.

## Remaining implementation work

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
