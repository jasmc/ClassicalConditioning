# Analysis and Statistics Design

**Status:** Overall paper strategy and approval open; a condition-aware
learning-onset route and older exploratory inference scaffolds are implemented
**Scope:** Population effects, longitudinal analysis, uncertainty, sensitivity,
individual heterogeneity, and the required learner-analysis workstream
**Depends on for final execution:** approved cohort, outcome definitions, and
shared bout-segmentation configuration
**Does not require for discussion:** final paper-scale artifacts

## Purpose

Design one coherent analysis strategy for the behavioral paper. Start with the
scientific questions and experimental structure, then choose the smallest set
of models that answer them honestly. Improve the existing LME where it is a
reasonable model and compare it against a simpler fish-level analysis.

Learner heterogeneity is a required paper workstream. It must produce a
scientifically interpretable account of individual learning and the paper's
learner-related outputs. It must also test whether the final representation
should be a continuous score, responder probability, latent trajectory,
binary/multiclass label, or a combination. Requiring learner analysis does not
justify choosing a hard threshold before that comparison. The operational
requirements live in
[Learner Representation and Analysis](./05_LEARNER_ANALYSIS.md).

The implemented exploratory model-input, LME, permutation, and bootstrap code
is described in [the current LME reference](../docs/analysis/07_LEARNING_ONSET_LME.md).
The newer condition-aware implementation and its remaining paper gates are in
[Learning-Onset Analysis Completion](./03_LEARNING_ONSET_IMPLEMENTATION.md).
Earlier statistics and learner plans remain in Git history. Their learner
requirements are in the active learner plan.

## Questions to resolve before choosing a model

1. What is the main biological claim: a conditioned-versus-control difference
   in change, a condition-by-time trajectory, a response during a named CS
   window, or something else?
2. What is the independent experimental unit? For population claims it is the
   fish, not frames or trial rows.
3. Which outcome most directly represents the claim: total activity, movement
   probability, conditional movement intensity, or bout initiation/rate?
4. What time scale matters: prespecified phase/block contrast, trial trajectory,
   or the full event-aligned curve?
5. If the paper claims extinction, what evidence shows that an established
   conditioned response subsequently declines or is absent? Prespecify the
   comparison and uncertainty; loss of statistical significance alone is not
   evidence of extinction.
6. Is individual heterogeneity adequately described by continuous fish effects,
   or is there evidence for distinct learner classes?
7. Which observations are exploratory, and which—if any—remain independent for
   confirmation?

## Non-negotiable analysis rules

- Bout segmentation is shared across all metrics.
- Fish are the population-level independent units; repeated trials and frames
  remain nested within fish.
- Condition identity must be present in every population comparison. An
  overall early-to-late change pooled across control and conditioned fish does
  not test conditioning.
- The effect direction must be named once and used consistently in artifacts,
  models, figures, and prose.
- Different outcome types do not automatically receive the same Gaussian LME.
- Failed, non-converged, rank-deficient, or disallowed singular fits cannot
  produce publication-ready coefficients or figures.
- Exploratory alternatives are compared using prespecified criteria, not by
  choosing whichever gives the smallest p-value.
- A learner label derived from an outcome cannot independently confirm a
  difference in that same outcome.
- Learner analysis is required for the paper, but a categorical learner label
  is adopted only if the scientific review and validation support it.

## Current implementation findings to resolve before paper approval

- The older exploratory LME route does not implement the required condition
  contrast or its interaction with learning time. The newer learning-onset
  route implements condition-aware block and longitudinal models, but its
  scientific configuration and paper-scale validation remain open; see
  [learning-onset implementation](./03_LEARNING_ONSET_IMPLEMENTATION.md).
- The older exploratory permutation and bootstrap routes pool conditions and
  test overall change. The learning-onset route has condition-aware fish-level
  robustness checks; their paper-scale calibration and interpretation remain
  open.
- Bout-rate handling does not yet use a matched baseline bout-rate estimand.
- Suppression-effect sign documentation is inconsistent.
- Cohort identity and the final scientific choices for factor references,
  planned contrasts, multiplicity, and fit-failure rules are not frozen for
  paper use, even where engineering implementations exist.
- Existing deterministic and synthetic tests do not replace calibration at the
  paper sample size or a reviewed paper run.

## Discussion and exploration programme

### 2.1 Define claims, estimands, and outcome hierarchy

Write one primary claim in plain language and translate it into one estimand.
Name one primary outcome; secondary outcomes should answer distinct biological
questions rather than repeat the same claim. Record:

- conditioned and comparison populations;
- baseline and response windows;
- early/late or continuous-time definition;
- direction and scale of the effect;
- aggregation order from frames to trials to fish;
- missingness and minimum-coverage rules; and
- whether the result is exploratory or confirmatory.

### 2.2 Explore the data without silently selecting the final model

Use paper-scale descriptive summaries to understand distributions, zeros,
coverage, fish-level trajectories, condition balance, day/rig structure, and
outliers. Keep a decision log of what was inspected and which later choices
could have been influenced. Exploration may motivate candidates; it does not
turn the same observations into untouched confirmation data.

### 2.3 Improve and challenge the LME

If the primary outcome is continuous and an LME is plausible, evaluate a small
prespecified family around a scientifically meaningful formula such as a
condition-by-time or condition-by-phase interaction, with fish as the grouping
unit. Decide rather than assume:

- raw, transformed, or model-family scale;
- baseline as subtraction, covariate, or repeated period;
- categorical phase/block versus continuous trial trajectory;
- random intercept alone versus a justified fish-level slope;
- whether day/rig effects are estimable;
- reference categories and planned contrasts; and
- covariance, residual, influence, and optimizer diagnostics.

Reject or demote the LME if the outcome distribution is incompatible, the
design cannot support the random-effects structure, fits are persistently
singular, or conclusions depend on a small number of fish.

### 2.4 Compare one strong simpler alternative

Implement a condition-aware fish-level contrast as the minimum robustness
analysis. Each fish contributes one prespecified effect; compare those effects
between conditioned and control groups using a design-appropriate permutation
or robust interval. This sacrifices trajectory detail but makes the biological
unit and effective sample size explicit.

Consider GEE/cluster-robust GLM when a population-average probability or rate
is primary. Consider a hierarchical Bayesian model only if it solves a concrete
problem that the LME/GEE cannot and priors plus diagnostics can be reviewed.
GAM/functional curves, point-process models, and HMMs remain exploratory tools
unless the scientific question specifically requires them.

### 2.5 Match models to outcome families

- Continuous positive activity/intensity: assess log-normal, Gamma, or another
  justified positive model; a Gaussian LME requires acceptable residuals.
- Movement probability/occupancy: use binomial information with fish clustering
  rather than treating proportions as unconstrained Gaussian values without
  justification.
- Bout counts/rates: use the observation time as exposure and consider Poisson
  or negative-binomial behavior; do not pair response bout rate with baseline
  total activity.
- Event-aligned curves: use them descriptively unless a prespecified functional
  inferential method and multiplicity rule are justified.

### 2.6 Analyze individual heterogeneity and complete the learner workstream

First show fish-level effects and trajectories continuously. Estimate how much
fish differ, how uncertain each effect is, and whether apparent subgroups are
stable under trial/fish resampling. Ask whether a threshold adds scientific
meaning beyond a continuous score.

The paper requires learner-focused analysis. Compare continuous scores,
random-slope distributions, change points, mixture models, latent trajectories,
responder probabilities, and categorical classes where scientifically
plausible. This comparison decides how the required learner result is
expressed; it does not decide whether individual learning is examined at all.

If categorical classification proceeds, freeze its inputs, threshold,
eligibility, uncertainty, and validation mode. Same-data learner/non-learner
comparisons are descriptive. Confirmatory claims require held-out, cross-fitted,
or independent evidence. If categorical classification is rejected, the paper
must still report the approved continuous or model-based learner analysis and
the evidence for not imposing classes. Full requirements are in
[the learner plan](./05_LEARNER_ANALYSIS.md).

### 2.7 Define uncertainty, multiplicity, and sensitivity

Specify one primary uncertainty method and a bounded sensitivity set. Relevant
checks include:

- fish bootstrap or model-based intervals with fish-level robustness;
- cohort and coverage rules;
- outcome and baseline definitions;
- selected metric versus legacy benchmark;
- smoothing/threshold settings for the shared detector;
- random-effects alternatives justified before fitting;
- day/rig influence; and
- multiplicity across the limited family of primary/secondary contrasts.

Do not run an unrestricted grid and report only favorable variants.

### 2.8 Freeze diagnostics and failure behavior

For each adopted model, define required convergence evidence, singularity and
boundary rules, model-rank checks, residual or calibration diagnostics,
influence checks, minimum fish/observation counts, and optimizer sensitivity.
A failed fit produces a failed result artifact and no final figure annotation.

### 2.9 Build reproducible analysis artifacts and tests

Version the final model input, formula, factor references, cohort/outcome IDs,
contrasts, uncertainty configuration, diagnostics, and results. Add synthetic
recovery tests for the intended condition effect, null behavior, factor coding,
row-order invariance, fish weighting, multiplicity, and failure propagation.

## Decision sequence

1. Discuss and record the primary claim, estimand, outcome hierarchy, and the
   required learner estimand.
2. Review paper-scale descriptive distributions and sample structure.
3. Prespecify the improved LME or replacement model plus one strong alternative.
4. Freeze cohort, model input, contrasts, diagnostics, uncertainty, and
   validation status.
5. Implement only the approved primary route and bounded sensitivity analysis.
6. Complete the learner workstream using the approved continuous, categorical,
   or combined representation and its specified validation mode.

## Deliverables

- analysis discussion and decision log;
- primary claim, estimand, outcome hierarchy, and effect direction;
- improved LME specification or documented reason for replacement;
- one condition-aware fish-level robustness analysis;
- model diagnostics and publication-stopping criteria;
- planned contrasts and multiplicity family;
- bounded sensitivity plan;
- completed learner-methodology decision and required learner outputs, with a
  justified continuous, categorical, or combined representation;
- versioned model-input, result, diagnostic, and test artifacts; and
- methods-ready interpretation including limitations.

## Exit gate

The primary population result can be regenerated from an approved cohort and
outcome artifact; it directly tests the conditioned-versus-control estimand;
diagnostics pass; fish-level uncertainty and the bounded robustness analysis
agree or their disagreement is explained; all model choices are traceable; and
any learner-stratified claim is clearly labeled descriptive or uses independent
validation. The learner-plan exit gate must also pass before the paper's learner
claims or learner figure are designated final.
