# Implementation status index

The only live workstream-status board. A scoped panel freeze is not whole-figure or paper-release approval.

| Number | Workstream | Status |
|---|---|---|
| 01 | [Paper baseline and input validation](01_PAPER_BASELINE_AND_INPUT_VALIDATION.md) | Not started |
| 02 | [Paper Cohort Completion](02_COHORT_IMPLEMENTATION.md) | In progress |
| 03 | [Analysis and Statistics Design](03_ANALYSIS_AND_STATISTICS.md) | Design / validation open |
| 04 | [Learning-Onset Analysis Completion](04_LEARNING_ONSET_IMPLEMENTATION.md) | Design / validation open |
| 05 | [Learner Method Review](05_LEARNER_METHOD_REVIEW.md) | Design / validation open |
| 06 | [Learner Representation and Analysis](06_LEARNER_ANALYSIS.md) | Design / validation open |
| 07 | [Learner Outputs and Validation](07_LEARNER_OUTPUTS_AND_VALIDATION.md) | Design / validation open |
| 08 | [Integrated Analysis and CR Profiles](08_INTEGRATED_ANALYSIS_AND_CR_PROFILES.md) | In progress |
| 09 | [Figures and Reproducible Reporting](09_FIGURES_AND_REPRODUCIBLE_REPORTING.md) | In progress |
| 10 | [First release: Figures 1–4](10_FIRST_RELEASE_FIGURES_1_TO_4.md) | Not started |
| 11 | [Supplementary Figures and Data](11_SUPPLEMENTARY_FIGURES.md) | Design / validation open |
| 12 | [Tail Mechanistic Analyses](12_TAIL_MECHANISTIC_ANALYSES.md) | Design / validation open |
| 13 | [Behavior and Imaging Integration Plan](13_BEHAVIOR_IMAGING_INTEGRATION.md) | Design / validation open |
| 14 | [Repository cleanup and deferred external archive](14_REPOSITORY_CLEANUP_AND_EXTERNAL_ARCHIVE.md) | Complete; external payloads verified and pruned |

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

Gate answers live in [DECISIONS.md](../docs/analysis/decisions/DECISIONS.md). They are revisited only
when new evidence changes a decision.


## Dependencies and first release

Baseline/input validation → reviewed cohort → outcomes and justified inference → validated learner representation → Figure 4 analysis → reviewed Figures 1–4 → first release. Figure infrastructure and Figure 1/2 review can progress alongside upstream work; paper claims wait for their applicable scientific evidence.

Supplementary composition, tail mechanistic analyses and imaging do not block the first release. Cleanup is independent of analysis order. Phase A retains pending payloads; Phase B requires verified JOAQUIM transfer.

## Updating status

Record only scoped accepted changes and their evidence. Keep completed descriptions in documentation, decisions in the decision register, and per-file transfer state in the archive inventory. Do not mark unrelated work complete.
