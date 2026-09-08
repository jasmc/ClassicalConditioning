# Step 11 — Versioned Learner Classification and Stratified Analysis

**Status:** Not started  
**Change class:** Legacy reproduction plus separately validated scientific analysis  
**Depends on:** Fully implemented non-learner refactor, Steps 09-10, and scientific gate L
**Unlocks:** Learner-stratified paper outputs, if approved

## Objective

Preserve all current classifier variants as explicit legacy methods, select or
develop one canonical classifier only after corrected outcomes are available,
and prevent circular learner-stratified inference. Before selecting any
classifier, conduct an end-to-end scientific brainstorming and design review
after the rest of the refactor is fully implemented.

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
cohorts, or primary inference are still changing. Once the non-learner refactor
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

This work package is **not required** for the candidate activity-metric
comparison (Step 07/09/10) and is explicitly deferred until after that
comparison is complete. See [Plans/DECISIONS.md](./DECISIONS.md) (Gate L).

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

## Optionality

The primary group-level paper analysis must not be blocked by learner
classification unless a manuscript claim explicitly depends on learner
stratification.
