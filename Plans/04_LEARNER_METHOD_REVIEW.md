# Learner Method Review

**Status:** Scientific discussion for Gate L; no learner method or paper label is approved here

**Scope:** Existing recordings and experiment conditions only; no additional experiments are possible

**Relationship:** Supplements [the active learner plan](./05_LEARNER_ANALYSIS.md), [the analysis design](./02_ANALYSIS_AND_STATISTICS.md), and [the decision register](./DECISIONS.md). It does not replace their cohort, outcome, validation, or approval requirements.

## The question to answer

“Which fish learned?” combines several different questions. A defensible analysis should distinguish:

1. **Population conditioning:** Did conditioned fish change differently from controls across the scheduled experience?
2. **Individual expression:** How large and reliable was the conditioned-direction behavioral change in each fish?
3. **Prediction:** Does early individual behavior predict later behavior that was not used to define it?
4. **Discrete classes:** Does the distribution support distinct biological learner types rather than continuous heterogeneity?

The first question belongs to the label-independent population analysis. The learner workstream addresses the other three. A fish's observed change also contains measurement error, spontaneous activity changes, habituation, fatigue, and other influences. Each fish experienced only its assigned condition, so its unconditioned counterfactual trajectory is unobserved. No classifier can turn these data into a directly observed biological ground truth for that fish. An individual score should therefore be described as evidence of *conditioned behavioral expression*, with uncertainty, rather than proof of an underlying binary ability to learn. Related work on individual response heterogeneity explains why observed within-subject changes need not equal true individual effects [1].

## Constraints from this repository

- Gate L requires a learner-focused paper result but leaves the continuous, longitudinal, probabilistic, and categorical representation open. A hard label is optional [active plan](./05_LEARNER_ANALYSIS.md).
- The primary fish cohort must be fixed using label-independent technical criteria. The current two-stage assessment records a merged pre-fit learner-input prerequisite; method-specific feature eligibility and reasons remain for the learner analysis. Neither may redefine the primary cohort [decision register](./DECISIONS.md).
- The chosen activity metric, detector, outcome transformation, windows, and population inference are not yet paper-frozen. This review cannot select a final method by comparing results on two local fixture fish.
- Existing outcomes must keep immobility distinct from missing tracking. A conditional-on-movement vigor score alone can miss strong suppression [analysis findings](../docs/analysis/audits/ANALYSIS_FINDINGS.md).
- Acquisition and later recovery must be reported separately. All four legacy scripts require acquisition-direction suppression *and* later recovery-direction change for their final learner label; that conjunction defines a particular trajectory phenotype, not learning itself [legacy behavior matrix](../docs/analysis/legacy/LEARNER_VARIANT_BEHAVIOR_MATRIX.md).

## Methods that do not require an LME learner classifier

| Method | Quantity estimated | Key assumptions and failure modes | Existing-data validation | When it may be preferable to an LME |
| --- | --- | --- | --- | --- |
| Prespecified fish-level acquisition contrast | One continuous early-to-late change per fish, oriented toward conditioned suppression; compare its distribution with controls | Throws away detailed trajectory shape; baseline drift and unequal coverage can bias a simple difference; an individual change is not an individual causal effect | Trial and block deletion, control comparison within experiment, null/gradual/abrupt simulations, prediction of later untouched trials | Transparent primary description when a prespecified phase contrast matches the biological claim and model assumptions are doubtful |
| Control-reference rank or calibrated score | How extreme a fish's directional contrast is relative to eligible controls | Controls must be comparable on experiment, day, outcome definition, coverage, and other determinants; small control samples give coarse tails; a tail rank is not `P(fish learned)` | Calibrate reference using controls excluded from score construction where feasible; leave-one-control-out diagnostics; report reference counts and uncertainty; test held-out prediction | Useful if a limited operational “unusual relative to controls” flag is needed and a reference distribution is credible |
| Individual time-series or change-point analysis | Whether a within-fish trajectory changes, and an onset region if an abrupt persistent change is plausible | Scheduled phases are ordered and cannot be freely permuted; gradual learning can look like no change point; noise and autocorrelation can create false shifts | Compare with no-change and gradual-change simulations; inspect control fish; check onset stability when trials or blocks are omitted | Useful for an onset question when individual records have enough repeated observations |
| State-space model | A latent trial-by-trial response trajectory and uncertainty in its onset, strength, and persistence | State dynamics and observation model are assumptions; latent “learning” may absorb fatigue or performance changes; a binary-correct-response model from another assay cannot be transplanted directly to these activity data | Posterior/predictive checks or equivalent likelihood diagnostics, simulation recovery, held-out trial prediction, comparison with a simpler score | Useful if time-varying individual trajectories materially improve description or prediction [2] |
| Functional trajectory analysis, including FPCA | Continuous modes of variation in fish trajectories | Components depend on scaling, smoothing, missingness, and alignment; a component or cluster of scores is not a validated learner class | Resampling stability, reconstruction/prediction error, sensitivity to missingness and time-grid choices | Useful to describe trajectory shapes without imposing one parametric curve [3] |
| Mixture or trajectory clustering | Candidate groups with different fitted score distributions or trajectories | Skewness, outliers, error structure, and rigid within-class assumptions can create spurious classes [4] | Compare with flexible one-population models; simulate the null; assess class size, resampling stability, and held-out prediction | Only if the scientific question truly concerns classes and the separation survives these checks |

Mechanistic associative-learning models could also be considered if the CS–US schedule and observation model are specified well enough to separate associative strength from motor expression. They should compete against simple non-associative time or fatigue models. A fitted parameter called “learning rate” is not automatically biological evidence of learning. Supervised classification has no independent fish-level training labels here: training on a legacy label would reproduce its definition rather than validate it.

### A practical continuous starting point

Start with a prespecified acquisition contrast for each fish on the selected, zero-inclusive primary outcome. Define the baseline and acquisition trial sets, direction, aggregation, minimum coverage, and any smallest meaningful change before inspecting the learner distribution. Keep fish-level estimates, contributing-trial counts, and missingness reasons. Summarize the full distribution by condition and experiment. This is a **candidate representation**, not an approved estimator or a finalized threshold.

Quantifying uncertainty requires more work than resampling trial rows independently. Trials are ordered and their behavior may be serially dependent. If each phase contains few usable trials, even a moving-block or within-phase bootstrap can be unstable or erase the trend of interest. Compare analytic, time-series, and resampling approaches on simulations matching the actual trial count, serial dependence, missingness, and plausible gradual or abrupt effects. Fish are the independent units for population inference; within-fish resampling addresses a different source of uncertainty.

### What a control-tail threshold would mean

A directional rank such as `(1 + number of calibration controls at least as extreme) / (1 + number of calibration controls)` describes extremeness under the calibrated control reference. A conformal-style false-positive guarantee needs exchangeability of the evaluated null fish and calibration controls, a score fixed without using calibration outcomes, and independent units at the fish level. Those conditions may fail when experiments, days, coverage, or technical quality differ. Under the displayed rank convention, the smallest attainable value is `1/(n_controls + 1)`; a strict `p < 0.05` rule cannot even trigger with 19 or fewer calibration controls. Reusing the same controls to choose the score, fit a reference distribution, and claim calibrated error would need additional accounting. Any threshold must be evaluated out of sample and its uncertainty reported. Conformal methods provide useful tools for false-positive control under their assumptions, but do not establish a biological learner probability [5].

### Language for any categorical result

If Gate L selects categories, separate insufficient data from uncertain evidence. `Unclassified` should mean the frozen eligibility assessment found inadequate usable information. Among eligible fish, a positive label can mean a prespecified criterion for conditioned-direction expression was met; a remainder label should not be called proven “non-learning.” “No meaningful conditioned response detected” is a narrower behavioral statement. Claiming evidence that a meaningful effect is absent requires a prespecified equivalence margin and enough precision to exclude such an effect. Any resulting label must retain the underlying continuous score, uncertainty, rule, reference identity, and validation mode.

## How to compare candidates without choosing the most favorable result

After the metric, cohort, and outcome contracts are frozen, use a bounded comparison of a simple continuous contrast, a control-calibrated directional score if controls suffice, and one longitudinal method only if the scientific question requires trial dynamics. Use the same authenticated trial outcomes and frozen selection assessment for each route. Compare interpretability, eligible sample size, null calibration, sensitivity to windows and transformations, missingness behavior, simulation performance, and prediction of prespecified later observations. If a complex method adds no stable information beyond the simple contrast, prefer the simpler result. Include legacy labels in an agreement and sensitivity report, not as training truth or a selection target.

The four inherited variants are historical comparators. They differ in response transformation, middle epoch, threshold, direction rule, and reference calculation [legacy behavior matrix](../docs/analysis/legacy/LEARNER_VARIANT_BEHAVIOR_MATRIX.md). Their final labels all mix acquisition with recovery. Reproducing them is useful for explaining differences from the earlier analysis, but none is made canonical by this document.

## Validation and interpretation using existing trials

Same-trial plots by a label derived from those trials describe what the rule selected. A learner-versus-remainder test on those same outcomes is circular as evidence that the selected fish learned. A prespecified held-out set can instead ask whether a score based on earlier trials predicts later conditioned behavior. The later trials must not influence score design, threshold selection, reference fitting, eligibility decisions that depend on their behavioral values, or labeling of the fish. A phase holdout may change what the score measures; for example, an acquisition-only rule cannot be assumed equivalent to a legacy acquisition-plus-recovery rule. Report this as **predictive validation of later expression**, not direct validation of a true learner class. If no sufficiently independent trial set remains, keep the individual analysis descriptive.

The final Gate L review should document whether any candidate has adequate individual precision and control calibration, whether apparent classes remain stable against a continuous alternative, and whether the labels would add interpretable information to the paper. A continuous result still fulfills the required learner workstream when categorical claims cannot be defended.

## References

1. [Response heterogeneity: Challenges for personalised medicine and big data approaches in psychiatry and chronic pain](https://pmc.ncbi.nlm.nih.gov/articles/PMC5820606/).
2. [Smith et al., Dynamic Analysis of Learning in Behavioral Experiments](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729979/). The published model analyzes binary task responses; adaptation to these motor outcomes would require a new observation model.
3. [Longitudinal functional principal component analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC3131008/).
4. [Extracting Spurious Latent Classes in Growth Mixture Modeling With Nonnormal Errors](https://pmc.ncbi.nlm.nih.gov/articles/PMC5965610/).
5. [Conformal Prediction Sets with Limited False Positives](https://proceedings.mlr.press/v162/fisch22a.html). Its guarantees apply under the method's stated calibration assumptions; this review does not assert they hold for these fish.
