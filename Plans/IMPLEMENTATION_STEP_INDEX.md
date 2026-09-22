# Implementation Status Index

This is the only live plan-status board. Archived plans retain detailed
historical acceptance criteria; current descriptive behavior lives under
`docs/analysis/`.

## Status legend

| Status | Meaning |
| --- | --- |
| Not started | No implementation work accepted |
| Discussion/design | Scientific or interface decisions precede implementation |
| In progress | Work is active but its exit gate has not passed |
| Fixture-only | Software works on bounded fixtures but is not paper-authoritative |
| Deferred | Preserved work outside the current critical path |
| Complete | Scoped exit gate passed and evidence is recorded |
| Archived — incomplete | Historical plan retired explicitly before its exit gate passed |
| Superseded | Replaced by a named active plan or documentation file |

## Active work

| Workstream | Status | Main open requirement | Authority |
| --- | --- | --- | --- |
| Governance and paper baseline | Open release prerequisite | Paper-scope baseline manifest, configuration freeze, and output fingerprints | [Governance](./GOVERNANCE.md), archived Plan 00 |
| Preprocessing and metric decision | Fixture-only / scientific decision open | Gate P and Gate T1: preprocessing, smoothing, shared detector parameters, selected metric, and validation status | [Metrics and bouts](../docs/analysis/METRICS_AND_BOUTS.md), [Decisions](./DECISIONS.md) |
| Paper cohort and unified selection | Fixture-only / infrastructure implemented | Implement one selection-assessment step covering technical, legacy Stage-1, Stage-5/LME, and learner eligibility; then approve Gate C0 and freeze the reviewed C1 cohort | [Cohort/onset implementation](./REMAINING_COHORT_AND_LEARNING_ONSET_IMPLEMENTATION.md), [single-cohort plan](./SINGLE_COHORT_AND_EXCLUSION.md) |
| Outcomes and population inference | Implemented, not paper-run | Freeze Gate O/S, run paper cohort, review recovery/calibration and diagnostics; block, longitudinal, simultaneous contrast, onset, robustness, and figure routes now exist | [Cohort/onset implementation](./REMAINING_COHORT_AND_LEARNING_ONSET_IMPLEMENTATION.md), [learning-onset plan](./LEARNING_ONSET_LME.md) |
| Learner analysis | Active required workstream | Gate L estimand/method, classifier or continuous representation, manifest, non-circular validation, and learner outputs | [Learner classification](./Analysis/2_LEARNER_CLASSIFICATION.md) |
| Integrated cohort/CR profiles | In progress; post-classification orchestration gated | Shared catch/block substrate is implemented; learner manifest, sensitivity evaluator, common-hash `run-pipeline` route, and paper run remain | [Integrated cohort/CR profiles](./Analysis/4_INTEGRATED_SINGLE_METRIC_COHORT_AND_CR_PROFILES.md) |
| Figures and interfaces | In progress | Registry, notebooks, paper panels, publication dimensions/theme, and visual regression | [Figures and reproducible reporting](./Analysis/FIGURES_AND_REPRODUCIBLE_REPORTING.md) |
| Release and legacy retirement | Not started | Immutable paper release and verified reproduction before legacy retirement | [Releases](./RELEASES_AND_LEGACY_RETIREMENT.md) |

## Deferred work

| Workstream | Status | Activation condition |
| --- | --- | --- |
| Semantic provenance and legacy conversion | Deferred | Stable schema plus concrete cross-environment, multi-version, or release need |
| Behavior/imaging integration | Deferred | Canonical behavior identities and recipes stable; imaging work explicitly authorized |
| Tail mechanistic analyses | Deferred | Core metric, shared detector, outcome, and cohort contracts approved |

## Historical implementation record

| Former step | Status | Recorded result or remaining gap |
| --- | --- | --- |
| 00 Governance and baseline | Archived — incomplete | No full paper-scope baseline/configuration/output-fingerprint evidence found |
| 00A Single-fish local intake | Complete | Pilot intake; 14 tests; unchanged source hashes |
| 00B Codebase characterization | Complete | Legacy routes characterized; LogMedian pilot authenticated |
| 01 Package and environment | Archived — incomplete | Foundation implemented; final pinned numerical baseline not found |
| 02A Artifact integrity and publication | Complete for implemented routes | SHA-256 source/output lineage, lossless intake verification, transactional publication |
| 03 Domain and configuration | Complete for `allDelay` fixtures | Resolved config, FishKey, trial map, stage hashes, CLI |
| 04 Ingestion and raw validation | Complete for local fixtures | Readers, raw validation, Gate T0, two tracking audits |
| 05 Legacy preprocessing equivalence | Complete for local fixtures | `legacy-paper-v1`; historical pickle timebase classified |
| 06 Tail representation and candidates | Archived — incomplete | Three-metric fixture chain retained; Gate P and known-answer coverage remain open |
| 07 Metric validation and selection | Archived — incomplete | Shared bout segmentation decided/implemented; Gate T1 remains open |
| 08 Full reprocessing, QC, and cohort | Archived — incomplete | Batch/cohort plumbing exists; paper cohort remains unfrozen |
| 09 Outcomes and temporal profiles | Archived — incomplete | Two-fixture corrected outcomes exist; Gate O remains open |
| 10A Exploratory inference scaffolding | Complete as engineering scaffold | Authenticated input, draft LME, fish permutation/bootstrap; no paper approval |
| 11 Historical learner plan | Superseded, content restored active | All operational requirements now live in the active learner plan |
| 12 Historical figure plan | Superseded, content merged active | CLI/notebooks/figure system plus paper registry now share one active plan |

## Scientific gates

| Gate | Required decision | Blocks |
| --- | --- | --- |
| G0 | Paper experiments, claims, figures, owners, and reference run | Paper-authoritative work |
| T0 | Raw tracking semantics and available coordinates | Tail representation; resolved for current fixtures |
| P | Frame loss, synchronization, interpolation, filtering, time, and gaps | Corrected paper preprocessing |
| T1 | Activity metric, shared detector input, smoothing, thresholds, units, and confirmation design | Corrected paper metric |
| C0 | Technical inclusion, engagement, missingness, primary/sensitivity populations | Cohort construction |
| C1 | Reviewed cohort instance and immutable hash | Cohort-dependent outcomes and inference |
| O | Outcomes, windows, scaling, binning, direction, and aggregation | Canonical outcome build |
| S | Estimand, model, random effects, contrasts, uncertainty, multiplicity, diagnostics, validation mode | Confirmatory population inference |
| L | Learner estimand, representation, method, features, power, threshold if used, and validation | Learner claims and Figure 3 |
| F | Figure semantics, dimensions, labels, sample sizes, and formats | Final figures |

Gate answers live in [DECISIONS.md](./DECISIONS.md). They are revisited only
when new evidence changes a decision.

## Critical path

```text
paper baseline
  -> approve preprocessing + shared bout detector + metric
  -> freeze paper cohort
  -> freeze outcomes and population inference
  -> complete learner-method decision and validation
  -> build registered paper figures
  -> freeze and verify paper release
```

The learner-method review can begin conceptually alongside population analysis,
but final classifier/score inputs and confirmatory validation depend on stable
preprocessing, cohort, and outcomes.

## Release milestones

- **R1 legacy-equivalent:** explicit configuration, pinned environment,
  canonical mirrors, and known-issue record.
- **R2 candidate-development:** candidate metrics, shared segmentation,
  validation/comparison outputs, and recorded metric decision.
- **R3 corrected paper candidate:** approved preprocessing, cohort, outcomes,
  population and learner results, diagnostics, and regenerated figures.
- **R4 paper release:** immutable bundle tied to manuscript, code, environment,
  source inventory, cohort, results, figures, and hashes.

## Updating status

For each accepted change, record the affected workstream, behavior-preserving or
scientific-correction class, versions and upstream identities, validation
evidence, and any invalidated downstream artifacts. Do not mark unrelated work
complete and do not push unless explicitly requested.
