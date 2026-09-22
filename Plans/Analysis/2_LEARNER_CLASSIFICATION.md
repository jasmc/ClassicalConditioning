# Learner Classification and Stratified Analysis

**Status:** Active — required learner-analysis workstream for the paper
**Change class:** Legacy reproduction plus separately validated scientific analysis
**Depends on:** Frozen preprocessing inputs, a unified selection-assessment
bundle, an approved C1/Gate O outcome contract, the population-analysis design,
and scientific Gate L
**Unlocks:** Learner-focused paper claims, figures, and stratified outputs

The post-classification single-metric run is specified in
[the integrated cohort/CR-profile plan](./4_INTEGRATED_SINGLE_METRIC_COHORT_AND_CR_PROFILES.md).
It does not change this plan's prerequisite: classification consumes a frozen,
label-independent primary technical cohort and authenticated outcome artifacts.

This file restores the complete operational content of archived Plan 11 as an
active plan. The historical wording remains preserved in
[the archived Plan 11](../Archive/11_LEARNER_CLASSIFICATION.md). The complete
detailed execution design is active in
[Learner-Stratified Outputs and Validation](./3_LEARNER_STRATIFIED_OUTPUTS.md),
with its original source retained in the archive.

Learner analysis is required for the paper. A hard learner/non-learner label is
not assumed in advance: continuous, longitudinal, probabilistic, and categorical
representations must be compared. If categorical classification is rejected,
the accepted continuous or model-based learner representation still has to
produce the learner-focused paper analysis.

## Objective

Preserve all current classifier variants as explicit legacy methods, select or
develop one canonical representation only after corrected outcomes are
available, and prevent circular learner-stratified inference. Before selecting
any classifier, conduct an end-to-end scientific brainstorming and design
review after the relevant preprocessing, cohort, and outcome contracts are
frozen.

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

Approve:

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

## Work packages

### 11.0 Post-refactor learner-methodology workshop

Do not begin canonical classifier selection while preprocessing, outcomes,
selection assessment, cohorts, or primary inference are still changing. Once the non-learner refactor
is fully implemented and its artifacts are frozen, revisit the learner question
from first principles:

- define whether the target is a discrete class, continuous learning score,
  latent trajectory, responder probability, or another estimand;
- separate biological learning, behavioral expression, measurement quality,
  and classifier eligibility;
- reconsider whether classification is necessary for the scientific question;
- compare supervised, unsupervised, semi-supervised, longitudinal, and
  model-based alternatives;
- explore PCA/component analyses as descriptive structure diagnostics, without
  assuming that principal components define learners;
- use power analysis or simulation to assess identifiable effect sizes,
  class-separation limits, validation sample requirements, and uncertainty;
- examine whether trial trajectories, change points, mixture models,
  hierarchical models, or continuous scores are more informative than a
  binary label;
- define independent, held-out, or cross-fitted validation before inspecting
  candidate performance.

Record all considered approaches, assumptions, rejection reasons, and decision
criteria. This workshop may conclude that no canonical learner classifier
should be used. PCA and power analysis are candidate tools for investigation,
not preselected methods or evidence of validity.

Explicitly include a critique of the original single-fish learner
classification approach itself (not only alternative methods): whether
classifying individual fish as discrete "learners" versus "non-learners" is
scientifically well-founded given the underlying continuous, noisy signal;
what assumptions that framing bakes in; and whether a population-level or
continuous-score framing better matches the data-generating process.

This work package is not required to choose the activity metric, but it is
required for the paper's learner workstream. Begin the final method comparison
only after its input metric, cohort, and outcomes are stable. See
[the decision register](../DECISIONS.md) (Gate L).

### 11.1 Wrap existing classifiers

Each legacy implementation receives:

- explicit input artifact;
- explicit configuration;
- isolated output path;
- no reliance on mutable module globals at the interface;
- versioned output manifest;
- preserved current results under the pinned environment.

Use adapters first; do not rewrite algorithms before characterization.

### 11.2 Define classifier protocol

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

### 11.3 Build classification manifest

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

### 11.4 Evaluate classifier candidates

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

Include alternatives emerging from Work Package 11.0 rather than limiting the
comparison to the four inherited classifiers.

### 11.5 Prevent circular analysis

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

### 11.6 Revisit classification after metric selection

Legacy distal-vigor labels remain historical outputs. If a new classifier uses
whole-tail activity, it receives a new classifier recipe and requires new
validation.

Do not assess a new activity metric primarily by how well it matches labels
created from the old metric.

### 11.7 Join labels safely

Join on the canonical fish key and condition/experiment metadata required by
the schema. Enforce many-to-one validation and stop on unmatched or duplicate
manifest rows.

The join annotates the frozen cohort and must not redefine it. Preserve every
primary-cohort fish as reference, learner, non-learner, or unclassified. A
behavior-dependent legacy policy may be joined later as sensitivity metadata,
but it cannot add/remove primary fish or define the classifier's reference
population.

Classifier code must not apply its own trial-count or complete-case filter.
It consumes the learner-eligibility rows produced by the unified selection
assessment. Eligible fish are classified; ineligible primary-cohort fish are
emitted as `Unclassified` with the assessment reason codes.

### 11.8 Build stratified outcomes

Reuse Step 09 outcome functions. Do not duplicate:

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
- Learner eligibility exactly matches the frozen unified assessment
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

## Exit gate

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

## Detailed output contract retained from the source plan

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

These requirements supplement, rather than replace, the work packages, tests,
deliverables, and exit gate above. The full schemas, output inventory, module
layout, configuration, validation checklist, phases, pilot, and definition of
done are maintained in
[the detailed companion](./3_LEARNER_STRATIFIED_OUTPUTS.md).
