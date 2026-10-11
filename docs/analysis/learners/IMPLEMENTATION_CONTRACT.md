# Learner implementation contract

Technical interface, provenance and compatibility requirements for [Plan 05](../../../Plans/05_LEARNER_ANALYSIS_AND_VALIDATION.md). Method choice, execution phases and acceptance gates are owned by that plan. Classification-specific requirements apply only when Gate L approves categories; otherwise use the approved score or trajectory with the same identity, cohort and validation contracts.

## Existing variants

Track separately:

```text
legacy-nominal
legacy-new
legacy-improved
legacy-wip
```

Do not rename one `canonical` merely because it is the easiest to execute.

## Scientific gate L

Approve the learner representation and all applicable settings below.
Classifier-specific settings apply only if categorical classification is selected:

- classifier implementation and version;
- activity metric and detector feeding the classifier;
- response transformation;
- feature definitions and directions;
- minimum trial/coverage rules;
- reference/control distribution;
- threshold calibration;
- point versus conservative rule;
- primary learner label;
- random seed and resampling;
- validation mode;
- confirmatory versus descriptive use.

## Implementation requirements

### 1. Wrap existing classifiers

Each legacy implementation receives:

- explicit input artifact;
- explicit configuration;
- isolated output path;
- no reliance on mutable module globals at the interface;
- versioned output manifest;
- preserved current results under the pinned environment.

Use adapters first; do not rewrite algorithms before characterization.

### 2. Define classifier protocol

```python
class LearnerClassifier(Protocol):
    classifier_id: str
    implementation_version: str

    def classify(
        self,
        trial_outcomes: pd.DataFrame,
        config: LearnerClassifierConfig,
    ) -> pd.DataFrame:
        ...
```

### 3. Build classification manifest

Minimum fields:

```text
experiment_id
condition_id
fish_id
cohort_role
classification_eligible
ineligible_reason
learner_status
analysis_stratum
classifier_id
classifier_version
classifier_execution_id
classifier_config_hash
input_metric_id
input_metric_version
movement_detector_id
feature values and uncertainty
distance/statistic and threshold
votes/rules
cohort_hash
cohort_policy_id
cohort_policy_hash
selection_assessment_hash
input_trial_outcomes_hash
classification_trial_set_id
evaluation_trial_set_id
code commit
validation_mode
generated_at
```

Preserve classifier evidence; do not export only a Boolean label.

### 4. Evaluate classifier candidates

On identical frozen inputs, compare:

- control false-positive rate;
- label stability under fish/trial resampling;
- feature uncertainty;
- sensitivity to transforms and thresholds;
- minimum-trial behavior;
- missingness;
- agreement/disagreement between variants;
- out-of-sample performance;
- dependence on legacy versus selected activity metric.

Include alternatives emerging from Phase 1 of the governing learner plan rather than limiting the
comparison to the four inherited classifiers.

### 5. Prevent circular analysis

Label every use as one of:

```text
descriptive_classifier_characterization
held_out_trials
held_out_phase
cross_fitted
independent_experiment
```

If labels and compared behavior come from the same trials/windows, plots are
descriptive and inferential p-values do not independently demonstrate learning.

### 6. Revisit classification after metric selection

Legacy distal-vigor labels remain historical outputs. If a new classifier uses
whole-tail activity, it receives a new classifier recipe and requires new
validation.

Do not assess a new activity metric primarily by how well it matches labels
created from the old metric.

### 7. Join labels safely

Join on the canonical fish key and condition/experiment metadata required by
the schema. Enforce many-to-one validation and stop on unmatched or duplicate
manifest rows.

The join annotates the frozen cohort and must not redefine it. Preserve every
primary-cohort fish as reference, learner, non-learner, or unclassified. A
behavior-dependent legacy policy may be joined later as sensitivity metadata,
but it cannot add/remove primary fish or define the classifier's reference
population.

The exploratory assessment publishes one merged learner-input prerequisite,
not four classifications. Classifier-specific model-derived feature failures
are recorded during later learner analysis. Ineligible primary-cohort fish
remain explicit as `Unclassified` rather than silently disappearing.

### 8. Build stratified outcomes

Reuse the outcome/profile functions governed by Plans 03 and 08. Do not duplicate:

- time binning;
- block mapping;
- catch-trial selection;
- movement probability;
- conditional intensity;
- bootstrap.

Always retain reference, learner, non-learner, and unclassified strata with
sample-flow counts.

Catch membership and ten-trial blocks resolve from `ExperimentSpec`. For the
three migrated experiments, the catch set is CS 25, 39, 53, 59, and 65; trial
65 is the first Early Test trial. Reuse the cohort catch/block aggregation and
coverage masking implemented for the integrated profile figures so labeled
and unlabeled panels cannot drift scientifically.

## Required tests

- Unique manifest key
- Stable restoration of source fish IDs
- Same fish ID remains distinct across experiments/conditions as required
- Controls retain reference role
- Ineligible fish remain explicit
- Many-to-one join validation
- Fixed-seed classifier reproducibility
- Legacy label regression
- No classification-data leakage in held-out/cross-fit modes
- Fish-first profile aggregation
- Primary cohort hash is unchanged by labels or classifier eligibility
- Catch/block selectors are identical in population and stratified outputs
- Classification and evaluation trial sets are disjoint in held-out mode
- Behavior-dependent legacy sensitivity status never becomes primary inclusion
- The merged pre-fit learner-input prerequisite matches the authenticated
  two-stage assessment; method-specific feature eligibility is recorded later
- Classifier execution contains no independent fish/row discard path

## Deliverables

- Legacy classifier adapters and equivalence reports
- Classifier comparison report
- Approved classifier decision record, if selected
- Post-refactor learner-methodology workshop record, including alternatives
  considered and power/simulation findings
- Classification manifest
- Unmatched and sample-flow reports
- Labeled trial/temporal derived artifacts
- Validation-mode-specific learner results

## Traceability and claim requirements

Every learner label is traceable to an input metric, cohort, feature recipe,
classifier, threshold, code, and validation mode. Any confirmatory
learner-stratified claim uses genuinely held-out, cross-fitted, or independent
evidence. The decision to classify at all is justified against continuous and
longitudinal alternatives after the non-learner refactor is complete.

## Relationship to the population analysis

Population analysis and learner analysis have separate scientific roles. The
population result should not be made conditional on a favorable learner split,
but the paper is not complete until the learner workstream has an approved
representation, validation mode, and set of outputs. A failed categorical
classifier is therefore a result to explain and replace with an approved
continuous or model-based representation, not a reason to omit individual
learning from the paper.

The complete rendering run may occur after the learner manifest is frozen, but
the primary population estimand, cohort, and LME remain label-independent. The
post-classification command is orchestration timing, not permission to choose
population membership or statistical settings from the observed learner split.

## Output contract

The active implementation must also preserve the detailed execution contract
from the archived learner-stratified plan:

- label CS-aligned, US-aligned, per-trial, and panel-ready derived tables from
  one canonical manifest rather than recomputing labels in plotting scripts;
- report reference, conditioned learner, conditioned non-learner, conditioned
  unclassified, and unmatched fish explicitly;
- analyze movement occurrence, fraction moving, bout count, conditional vigor,
  and any approved integrated movement outcome rather than relying only on
  vigor conditional on movement;
- support individual-fish, selected-trial, catch-trial, grouped-trial,
  five-trial-block, phase, and trial-by-trial views;
- aggregate samples to trials and trials within fish before aggregating fish;
- record contributing-fish coverage for every temporal bin;
- use fish-level uncertainty, or hierarchical fish/trial resampling when the
  estimand requires it;
- produce classification/sample-flow tables, individual-fish atlases,
  temporal profiles, trajectories, and validation-mode-specific results;
- separate panel-data generation from rendering and make the learner rule,
  cohort, classifier/score recipe, alignment, windows, and validation mode
  available to every figure sidecar; and
- complete at least one held-out, cross-fitted, or independent validation
  before a confirmatory learner claim.

These requirements supplement the interface and traceability requirements above. The full schemas, output inventory, module
layout, configuration, validation checklist, phases, pilot, and definition of
done are maintained in
[outputs and validation specification](OUTPUTS_AND_VALIDATION.md).
