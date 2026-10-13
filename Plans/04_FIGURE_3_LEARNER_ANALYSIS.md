# Figure 3 — Learner Analysis and Validation

**Scope:** Individual conditioned behavioral expression, Figure 3, and learner-dependent Figure 4/CR-profile outputs using existing recordings. No additional experiments are available. A categorical learner/non-learner label is optional; an approved continuous or longitudinal result still completes the learner workstream.

**Depends on:** Frozen corrected preprocessing and selected metric/detector; authenticated technical/exploratory assessment; label-independent primary cohort (C1); approved outcomes (O) and the necessary population-analysis contract (S). Final method selection waits for stable inputs. Gate L approves the learner representation and validation design.

**Unlocks:** Reviewed learner-focused claims and Figure 3, plus learner-aware outputs in [Plan 05](05_FIGURE_4_CR_PROFILES.md), [figure reporting](07_FIGURE_BUILD_TOOLS.md) and [the Figures 1–4 release](08_FIRST_RELEASE_FIGURES_1_TO_4.md). Population membership and inference remain independent of learner results.

## Phase 1: Method review and Gate L decision

1. Hold the end-to-end methodology workshop after the non-learner prerequisites are stable. Distinguish population conditioning, individual expression, held-out prediction and evidence for discrete classes. Critique whether the original binary learner framing is scientifically justified.
2. Compare a prespecified continuous acquisition contrast, a control-calibrated score when eligible controls suffice, and longitudinal/state-space/change-point approaches when trial dynamics warrant them. Consider component/PCA diagnostics and mixture/clustering alternatives without treating components or clusters as validated learner classes.
3. Separate acquisition from recovery, immobility from missing tracking, and biological interpretation from classifier eligibility. Preserve the legacy nominal/new/improved/WIP variants as historical comparators, not ground truth or a canonical method by default.
4. Design simulations/power checks, individual uncertainty, control calibration and validation splits before comparing candidate results. Compare null performance, gradual/abrupt effects, serial dependence, missingness, window/metric sensitivity, resampling stability and later held-out prediction on identical authenticated inputs.
5. Record alternatives, assumptions, rejection reasons, feasible precision and the rationale for imposing or declining categories. Approve Gate L in the [decision register](../docs/paper/decisions.md): estimand, representation, implementation/version, metric/detector, transformation, features/direction, coverage, reference/calibration, any meaningful-effect margin or threshold/rule, seed/resampling, validation mode and permitted claims.

**Exit:** One justified representation and prospective validation design are approved. Rejecting categories does not remove the required individual-learning analysis. Independent new-cohort validation is unavailable; insufficient existing-data independence limits claims to descriptive results.

Scientific comparison and literature: [method-review reference](../docs/paper/methods/individual-learning.md).

## Phase 2: Implementation and scientific validation

1. Wrap and characterize the four legacy classifiers with explicit immutable inputs/configuration, isolated external destinations and versioned manifests; preserve algorithm identity and authenticated regression results before adaptation.
2. Implement the approved representation and evidence-rich fish-level manifest. Preserve experiment/condition/fish identity, cohort/policy and assessment hashes, metric/detector/method versions, source/code identity, features or scores/trajectories, uncertainty, eligibility reasons, any rules/thresholds and disjoint construction/evaluation trial sets.
3. Annotate derived CS/US time-series, trial summaries and panel data through validated many-to-one joins. Stop on duplicate keys; report unmatched fish. Keep controls as references and insufficient-information fish explicit. Model-specific feature failures never remove fish from the frozen primary cohort or create an independent discard path.
4. Reuse approved outcome functions and Plan 05 aggregation/selectors for movement occurrence, fraction moving, bouts, conditional vigor and approved zero-inclusive activity. Resolve catches and blocks from experiment configuration; aggregate samples → trials → within fish → across fish, with contributing counts and fish-level or justified hierarchical uncertainty.
5. Validate the chosen method with the prespecified existing-trial/phase holdout or cross-fitting design. Evaluation outcomes must not inform score design, thresholds, reference fitting, behavioral eligibility or corresponding labels. Same-data strata describe rule-selected behavior; their differences are not independent confirmation of learning.
6. Run manifest/identity, cohort-hash, eligibility, join-cardinality, fixed-seed, legacy-regression, selector-consistency, coverage, fish-first aggregation and leakage checks. Revisit and revalidate the recipe if upstream metric, outcomes or preprocessing change.

**Exit:** Auditable derived artifacts and validation results satisfy Gate L; the primary cohort and original datasets remain unchanged. Inadequate precision/calibration or validation is reported as a limitation, never concealed by favorable learner splits.

Interfaces and tests: [implementation contract](../records/scientific-evidence/designs/individual-learning-implementation.md). Schemas and QC: [outputs and validation specification](../records/scientific-evidence/designs/individual-learning-outputs-and-validation.md).

## Phase 3: Figure 3 and downstream outputs

1. Produce representation/eligibility and sample-flow tables, feature/score distributions, individual examples/atlas, trial trajectories and validated learner-focused results. For approved categories, retain underlying continuous evidence and reference/unclassified counts; otherwise use the approved scores or trajectories.
2. Supply individual-fish, selected single-trial, catch, grouped/five-trial-block, phase and trial-by-trial views with movement occurrence alongside conditional vigor. Reuse configured pooled catches and ten-trial selectors for the integrated run; do not silently substitute one block definition for another.
3. Keep panel-data generation separate from rendering. Every sidecar/caption identifies cohort/input/method hashes, eligibility, alignment, windows, event identity, contributing fish/trials, uncertainty and validation mode. Adapt downstream Figure 4 grouping to the approved representation rather than forcing categories.
4. Integrate with Plans 05 and 07 and review Figure 3 plus applicable downstream claims for the first Figures 1–4 release. Save one consolidated external review containing requested versions/provenance; numerical payloads and media remain external. Panel/figure freezing uses the explicit freeze gate and its scientific, structural and visual checks.

**Exit:** Reviewed panels and downstream results express the approved representation faithfully, show coverage and uncertainty, and distinguish descriptive characterization from held-out prediction or confirmatory evidence.

## Completion criteria

- [ ] Prerequisite inputs, cohort and outcome/inference contracts are authenticated and stable.
- [ ] Method workshop, comparisons, simulation/power evidence and Gate L decision are recorded.
- [ ] One approved representation is implemented with a versioned evidence-rich manifest.
- [ ] Unique keys, eligibility, cohort preservation, joins, coverage and shared selectors pass verification.
- [ ] Non-circular existing-data validation supports every inferential claim; unsupported claims remain descriptive.
- [ ] Figure 3 and learner-dependent Figure 4/profile outputs preserve approved semantics and provenance.
- [ ] Original scientific data and historical classifier results remain intact; permanent outputs are external.

This is the sole active learner plan, including the former method-review and output/validation workstreams. Detailed requirements remain in [learner documentation](../docs/paper/methods/README.md). Track progress only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md).

## Preserved detailed requirements

The [implementation design](../records/scientific-evidence/designs/individual-learning-implementation.md) and [outputs/validation design](../records/scientific-evidence/designs/individual-learning-outputs-and-validation.md) preserve unique conditional schemas, joins, output families and validation requirements. Use them with the approved representation; categorical examples are conditional on Gate L. Current scientific principles live in [shared methods](../docs/paper/methods/individual-learning.md), with [interface guidance](../docs/running-analysis/individual-learning-interfaces.md).
