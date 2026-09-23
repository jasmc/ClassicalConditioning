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
| Paper cohort and two-stage assessment | Fixture-only / infrastructure implemented | Validate technical and exploratory legacy-rule assessment on paper inputs; approve Gate C0 and freeze the reviewed C1 cohort; migrate population consumers | [Current assessment behavior](../docs/analysis/DISCARDING_ASSESSMENT.md), [cohort completion](./01_COHORT_IMPLEMENTATION.md) |
| Learning onset and population inference | Implemented, not paper-run | Freeze Gate O/S, calibrate and run approved models on the frozen paper cohort, review diagnostics and figures | [Learning-onset completion](./03_LEARNING_ONSET_IMPLEMENTATION.md), [analysis and statistics design](./02_ANALYSIS_AND_STATISTICS.md) |
| Learner analysis | Design open; no canonical package route | Gate L estimand/method, continuous or categorical representation, manifest, non-circular validation, and learner outputs | [Learner analysis](./05_LEARNER_ANALYSIS.md) |
| Integrated cohort/CR profiles | In progress; post-classification orchestration gated | Shared catch/block substrate is implemented; learner manifest, sensitivity evaluator, common-hash `run-pipeline` route, and paper run remain | [Integrated cohort/CR profiles](./07_INTEGRATED_ANALYSIS_AND_CR_PROFILES.md) |
| Figures and interfaces | In progress; Figure 1 C/D trace command implemented | Registry, approved paper panels, publication dimensions/theme, and visual regression | [Figures and reproducible reporting](./08_FIGURES_AND_REPRODUCIBLE_REPORTING.md) |
| Tail mechanistic analyses | Planned after Figure 4 | Complete and review Figure 4 pooled-learner CR analysis, then implement exploratory tail analyses with its approved identities and timing conventions | [Tail mechanistic analyses](./09_TAIL_MECHANISTIC_ANALYSES.md) |
| Final analysis release | Not started | Freeze one verified analysis record after code, scientific decisions, paper results, and figures are ready | [Final analysis release](./10_FINAL_ANALYSIS_RELEASE.md) |

## Deferred work

| Workstream | Status | Activation condition |
| --- | --- | --- |
| Behavior/imaging integration | Deferred | Canonical behavior identities and recipes stable; imaging work explicitly authorized |

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
| 05 Legacy preprocessing equivalence | Complete for local fixtures | `legacy-paper`; historical pickle timebase classified |
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
  -> complete Figure 4 pooled-learner CR analysis
  -> carry out tail mechanistic analyses
  -> freeze and verify the one final analysis release
```

The learner-method review can begin conceptually alongside population analysis,
but final classifier/score inputs and confirmatory validation depend on stable
preprocessing, cohort, and outcomes.

## Final analysis release

Make one release after the critical path is complete. Freeze the code,
configuration, data and cohort identities, analysis results, diagnostics,
figures, and manuscript links described in the
[final analysis release plan](./10_FINAL_ANALYSIS_RELEASE.md). Candidate and
legacy runs are evidence used during analysis, not earlier releases.

## Updating status

For each accepted change, record the affected workstream, behavior-preserving or
scientific-correction class, versions and upstream identities, validation
evidence, and any invalidated downstream artifacts. Do not mark unrelated work
complete and do not push unless explicitly requested.
