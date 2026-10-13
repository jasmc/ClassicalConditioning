# Fish Discarding, Eligibility and Paper Cohort



**Scope:** Decide fish membership once from label-independent technical
evidence, freeze it, and make every population consumer use the same cohort.
Learning-onset model choices and validation are owned by
[Learning-Onset Implementation](03_FIGURE_2_LEARNING_ONSET.md).

## Boundary and evidence

The [selection inventory](../records/scientific-evidence/audits/EXCLUSION_AND_SELECTION_INVENTORY.csv) records
active, legacy, and deferred selection operations. The implemented
[`assess-discarding` guide](../docs/paper/methods/recording-quality-and-cohorts.md) specifies
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

This work was already planned on 2026-09-02 in Git revision `83cbeee`, `Plans/steps/08_FULL_REPROCESSING_QC_AND_COHORT.md`. That version separated local fixture plumbing from paper C0/C1 review. Its decision register deferred the historical per-block OR/AND completeness issue pending real cohort counts. The present Plan 02 is the successor, before Analysis and Statistics, not a new competing plan. See the [current/historical route audit and decision proposal](../records/scientific-evidence/audits/fish-selection-policy-review-2026-10-11.md).

## Author-requested consolidation before later review

On 2026-10-11 the author requested combining **all legacy discarding steps**, with review later. The [combined review package](../records/scientific-evidence/audits/fish-selection-policy-review-2026-10-11.md) now covers preprocessing, propagation through grouping/scaled plots, normalized-vigor row/completeness and paired-test filters, all four learner variants and display selectors. Preserve each original rule, order, default, threshold/configuration, AND/OR logic and bypass; do not silently intersect all rules into a new primary population. Later review must approve the disposition and numerical policy before activation. Current automatic assessment omits some of these later families; documentation consolidation does not change that execution.

## Detailed consolidation and later-review workflow

This sequence implements the author's request to collect all legacy discarding steps first and review the solution later. The documentation/source inventory is available; executing the evaluation, approving rules and enabling the final policy are future work. Progress belongs only in the implementation index. Plan 02 remains before Plan 06.

### 2.1 Establish the source and recording inventories

- Authenticate all nine legacy scripts: preprocessing/discarding, example plotting, grouping, scaled vigor, normalized vigor, and nominal/new/improved/WIP learners. Include their helper/configuration dependencies and earlier revisions where a rule changed.
- For every selection-related operation, record its source revision/hash, function/call site, execution stage, affected unit, inputs and output consumers. Distinguish actual instantiated settings from unused dataclass defaults or comments.
- Record the complete expected paper recording denominator supplied by Plan 01, including incomplete, ambiguous, failed, unprocessed and unavailable recordings. Identify which saved inputs can authenticate each legacy recipe; absent inputs remain explicit.
- Extend the existing [source index](../records/scientific-evidence/audits/legacy-selection-source-index-2026-10-11.json) and [selection inventory](../records/scientific-evidence/audits/EXCLUSION_AND_SELECTION_INVENTORY.csv), rather than creating another independent fish list.

**Deliverable:** One source-linked rule catalogue and expected-recording inventory. Source-line candidates are reviewed into actual operations; comments and repeated calls are not counted as independent rules.

### 2.2 Collect post-preprocessing screening

Document and later reproduce these as distinct original rules:

1. Readability and presence of per-fish inputs; raw/processing failures and ambiguous identity.
2. Last observed US viability: observed event/end timing and the original bout-bearing window.
3. Train US bout response in the original selected trials/windows.
4. Re-Train US response, including the original absent-Re-Train bypass.
5. Baseline bout-bearing trial completeness in each original required block.
6. CR-window bout-bearing trial completeness, including experiment-specific windows and empty-selected-CS guards.
7. Resulting text lists, reasons and original file-move behavior as historical provenance only.

Capture exact boundaries, counts, event interpretation, AND/OR logic, enabled/disabled flags and fail/unevaluable behavior. `RUN_DISCARD=False` in the current legacy source is recorded accurately. Supported evaluation must never reproduce destructive file movement. Preserve valid immobility separately from tracking failure and undefined bout-only vigor.

**Deliverable:** Individual rule definitions and source-specific preprocessing-screen recipes, with no numerical policy approval implied.

### 2.3 Trace propagation into grouping and scaled-vigor consumers

- Follow `Discarded_fish_IDs.txt` and reason files through grouping, per-fish grids, pooled data and scaled-vigor processing/plots.
- Record that grouping can split heatmaps without equivalently filtering pooled rows; record scaled-vigor list application and its defaults.
- Trace every reused list to the rules that produced it. Do not count reapplication of a previously excluded fish as additional independent attrition.
- Include manual condition/fish/trial/example selectors and metadata restrictions. Distinguish processing scope, illustration selection and actual model population effects.

**Deliverable:** Producer-to-consumer graph showing where nominally related plots/tables use different populations.

### 2.4 Collect normalized-vigor plotting and statistical filters

Keep the later families separate from post-preprocessing checks:

1. Optional discard-list reuse (`APPLY_FISH_DISCARD=False` by default here).
2. Age, setup/rig, condition, alignment, block and trial-window selection.
3. Missing/nonfinite normalized vigor, baseline/response and trial/block fields removed by model preparation.
4. Minimum-trial fish selection in `prepare_main_df`: the executable **ANY-block** rule, its configured minimum and the remaining sparse rows retained after it passes.
5. Paired-test complete-block/pair subsets, missing-block skip behavior and analysis-specific contributing counts.
6. Other plotting/model finite-row, available-pair and coverage subsets discovered in the source scan.

The ANY-block rule and a possible all-required-block alternative are different recipes. Preserve the inherited behavior before comparing alternatives; an all-required-block fix is not silently approved by this plan. A paired contrast can require complete pairs without removing that fish from unrelated analyses.

**Deliverable:** Explicit fish-, trial-, block- and contrast-level eligibility definitions, with the original order and defaults preserved.

### 2.5 Collect all four learner variants

- Record each variant's input fields, finite/positive transformations, selected acquisition/recovery features, epoch trial counts and constituent-block completeness checks.
- Keep the actual configured thresholds, selected features and variant-specific differences; a merged pre-fit prerequisite is not equivalent to running four complete classifiers.
- Include later feature/model failures, intersections of fish with required features and minimum-fish model-feasibility gates. Identify whether a failure removes rows, marks a fish ineligible or stops a model.
- Preserve controls as references and eligible/insufficient/uncertain statuses separately. Labels are outcomes of a method, not primary inclusion criteria.
- Coordinate learner-specific decisions with Plan 04. Assessment and final learner-method validation remain distinct; do not fit new classifiers merely to populate the exclusion catalogue.

**Deliverable:** Four source-specific eligibility profiles plus a comparison of shared versus variant-specific requirements and unevaluable cases.

### 2.6 Define one evaluation contract, without activating a primary policy

Specify a future audit-only evaluator that reads authenticated inputs and publishes one combined evidence bundle:

- Stable experiment/condition/fish identity, recording inventory status and source/configuration/recipe hashes.
- Per-operation unit (recording, fish, frame, trial, block, pair, feature or model), observed evidence, exact parameter values, rule status and reason.
- `pass`, `fail`, `not_applicable` and `not_evaluable` states with explicit semantics. Missing input is not a pass; unavailable processing is not automatically a biological exclusion.
- Dependencies and applicability: evaluate independent checks even when an earlier rule fails, and mark genuinely dependent checks unevaluable when their required inputs are absent.
- Original stage order and cumulative attrition for each original executable recipe, plus independent rule effects and overlap. Comparisons across incompatible metrics/windows/versions remain labelled as different recipes.
- Fish/trial/block/model counts by experiment/condition and complete expected denominator; distinguish exclusions, pending processing and analysis-specific noncontribution.

Reuse the existing technical/exploratory assessment where it matches the intended recipe. Add the missing normalized-vigor and later learner families to the common evidence contract when implemented; do not force all rules into one global Boolean intersection. Rules with no authenticated reproduction inputs remain documented and unevaluated.

**Deliverable:** An auditable evaluation specification and, when the evaluation work is authorized and inputs are ready, a combined external rule matrix, flow/overlap tables and provenance bundle. No current source data, cohort or frozen figure is changed.

### 2.7 Prepare the later author-review packet

The packet must show:

- Every inherited rule and default, grouped by original stage, with source evidence and actual population/row effects.
- Which rules are already evaluated by `assess-discarding`, which are absent from it, and which later consumers currently bypass the shared cohort.
- Counts for original recipes and explicitly labelled alternatives, including ANY-block versus required-block completeness where evaluable.
- Proposed disposition of each rule: technical primary membership, processing disposition, named sensitivity, outcome-specific eligibility, learner eligibility, display selection or redundant propagation.
- Unresolved thresholds, partial-recording treatment, minimum usable information, per-contrast block requirements and unevaluable cases. Do not select settings using favorable condition effects.
- A draft coherent pipeline and versioned policy definition, identifying accepted boundaries separately from proposed numerical choices.

**Review checkpoint:** Present the combined packet to the author and record decisions later. Gathering legacy steps is currently authorized; applying them all to the paper population is not. The label-independent technical primary boundary is already decided, but numerical C0 criteria and C1 membership remain open. No draft cutoff or disabled switch becomes approved automatically.

### 2.8 Implement and freeze only after the policy review

1. Record approved C0 criteria, parameters, exceptions, scope, rationale and evidence in the scientific decision register and versioned policy configuration. The assessment defaults of one matched frame and zero minimum valid-frame fraction are not approved paper criteria.
2. Validate the approved evaluator against the expected paper inventory, preserving failed/pending cases and distinguishing technical inclusion from behavior-dependent sensitivity rules.
3. Generate and review one draft per-fish cohort manifest, then freeze C1 with the explicit cohort command. Save numerical payloads externally and retain policy/provenance records in the repository.
4. Build authenticated cohort trial outcomes and sample flow. Migrate supported population models/figures to this cohort once; replace private downstream fish lists with explicit cohort-authenticated eligibility artifacts.
5. Keep outcome/model/learner/display eligibility reason-coded without rewriting primary membership. A fish with no eligible rows still appears in sample flow.
6. Publish primary and separately identified approved sensitivity results. Later changes require a new policy/cohort identity and invalidate dependent results; historical methods and accepted figure freezes remain unchanged.

**Deliverable:** One approved policy, one reviewed immutable paper cohort, supported consumer migration and complete inventory-to-figure count reconciliation. This phase cannot be declared complete from the existence of assessment code or fixture tests.

### 2.9 Required implementation checks

- Source/configuration hashes, original default switches and original AND/OR/bypass semantics are reproducible.
- Repeated discard-list propagation does not create double-counted exclusions; separate original recipes retain their own denominators.
- Unknown/duplicate/mismatched identities and stale inputs fail explicitly; unavailable inputs remain unevaluable.
- Assessments are deterministic and auditable; no evaluator moves/deletes scientific source files.
- Changing response strength or learner labels cannot change primary technical membership.
- Fish membership is applied once, joins are validated and row/model eligibility cannot silently alter the primary cohort.
- Counts reconcile at recording, fish, trial, block, contrast, feature/model and displayed-contributor levels.
- Author approval is required before draft numerical criteria drive a new paper cohort. No scientific method or threshold is frozen by editing this plan.

The existing cohort boundary and assessment infrastructure supply part of this workflow. The complete legacy evaluation, later policy review and paper-scale consumer migration remain future work.

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
