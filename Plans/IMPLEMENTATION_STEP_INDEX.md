# Implementation status index

The only live workstream-status board. A scoped panel freeze is not whole-figure or paper-release approval.

| Number | Workstream | Status |
|---|---|---|
| 01 | [Paper input validation and reproducibility](01_PAPER_INPUT_VALIDATION_AND_REPRODUCIBILITY.md) | Infrastructure implemented; paper-scale validation open |
| 02 | [Fish discarding, eligibility and paper cohort](02_DISCARDING_AND_COHORT_IMPLEMENTATION.md) | G historical-rule projections and separate technical exclusion sensitivity run; primary 57 fish preserved; paper-wide evaluation and C0/C1 approval open |
| 03 | [Figure 2 — learning onset](03_FIGURE_2_LEARNING_ONSET.md) | Independent G exploratory workflow completed and reviewed; baseline/threshold selection and timing calibration open; current freeze preserved |
| 04 | [Figure 3 — learner analysis and validation](04_FIGURE_3_LEARNER_ANALYSIS.md) | Descriptive Delay draft and full atlas retained; later Trace drafts remain separate. [Corrected BLUP comparison](../docs/running-analysis/figure3-blup-comparison.md) completed for all 57 Delay/control fish: 999 calibration refits and four independent 499-dataset scenarios, all successful; matched pointwise coverage ≈95%, null family-bound rejection ≈5%. Full per-fish/legacy comparison published on JOAQUIM. No method selected; Gate L, meaningful margin/eligibility, full-assay validation and claims remain open. |
| 05 | [Figure 4 — CR profiles](05_FIGURE_4_CR_PROFILES.md) | In progress |
| 06 | [Analysis and statistics](06_ANALYSIS_AND_STATISTICS.md) | G expanded shape/availability simulations and unsmoothed comparison completed; smooth curve coverage fails for abrupt/transient truth; scientific approval open ([review](../records/scientific-evidence/analyses/figure-02-g-independent/implementation-and-review-2026-10-15.md)) |
| 07 | [Reliable figure builds and panel replacement](07_FIGURE_BUILD_TOOLS.md) | Generic assembler safeguards implemented ([audit](../records/scientific-evidence/audits/figure-build-integrity-2026-10-12.json)); current-selection routing, paper reproduction, final-scale QC and review/state integration open. [Nature target](../docs/paper/figures/shared/nature-style.md): profile/policy checks and single-snapshot shared plotting/axis/stimulus support implemented ([evidence](../records/figure-reviews/plotting-theme-integration-20261016.md)); [N01–N28 and F01–F22](07_FIGURE_BUILD_TOOLS.md#5-nature-main-figure-presentation-profile) track remaining renderer/QC/export/freeze integration; scoped Figure 1/2 measured styling/frames published through guarded state; scientific/role, accessibility and production coverage remain partial. [2026-10-17 handoff H01–H10](07_FIGURE_BUILD_TOOLS.md#6-nature-completion-handoff-2026-10-17) consolidates remaining work and the latest box-only/no-further-resize instruction; learner counts belong to Figure 3 only |
| 08 | [First release — Figures 1–4](08_FIRST_RELEASE_FIGURES_1_TO_4.md) | Not started; R01–R10 acceptance gates and required manifest defined; scientific and production prerequisites remain open |
| 09 | [Supplementary figures](09_SUPPLEMENTARY_FIGURES.md) | Design / validation open |
| 10 | [Tail mechanisms](backlog/10_TAIL_MECHANISTIC_ANALYSES.md) | Design / validation open |
| 11 | [Behavior/imaging integration](backlog/11_BEHAVIOR_IMAGING_INTEGRATION.md) | Design / validation open |
| 12 | [Study/discussion artifact storage](12_STUDY_ARTIFACT_STORAGE.md) | Implemented: 18 study sources externalized with exact recovery; four readers updated, 57-entry library published. Existing HTML paths/bytes and notes retained; protected historical originals remain exact compatibility records. |
| 13 | [Script command organization](13_SCRIPT_COMMAND_ORGANIZATION.md) | Implemented needed audit repairs: scoped command tree, shared helpers/guards, 29 completed-driver retirements plus one helper facade, verified intake quarantine and drive navigation. Scientific authority unchanged; later guarded figure reconciliation complete; stale test/import expectations repaired; CI removed by author direction on 25 October. |

Renumbered consecutively on 2026-10-12. The map is recorded in [plan renumbering](../records/repository-archive/plan-renumbering-20261012/record.json). Old identifiers in the historical tables below refer to their original numbering.

## Completed and retired work packages

| Identifier | Record | Completion |
|---|---|---|
| 14 | [Repository cleanup and deferred external archive](../records/repository-archive/plan-policy-review-20261011/completed-repository-cleanup-work-package.md) | Complete; external payloads, historical scripts, configurations and [documentation organization](../records/repository-archive/documentation-cleanup-20261011/README.md) verified; [configuration record](../records/repository-archive/configuration-cleanup-20261011/README.md) |

Completed maintenance on 2026-10-17: [organization reconciliation](../records/repository-history/2026-10-17-organization-reconciliation/README.md) implements the authority/preservation fixes, navigation, documentation ownership, script classification and routine local checks. All scripts remain retained; any archive/delete dispositions remain separate author choices. This maintenance does not close scientific gates.

Maintenance follow-up on 2026-10-23: [script lifecycle and CI](../records/repository-history/2026-10-23-script-lifecycle/README.md) resolves the 14 pending classifications and five historical-candidate recommendations, catalogues the guide repair and initially implemented organization CI. The 25 October author direction removes CI; local checks remain supported. The later authorized script reorganization completed the proposed archive-helper disposition with verified external recovery.

Maintenance on 2026-10-23: [repository and SSD review organization](../records/repository-history/2026-10-23-repository-output-review/README.md) inventories project/output ownership, repairs maintained navigation and publishes a reading library for Figure 2 G/H/I and Figures 3/4. Scientific selections and gates are unchanged.

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

Gate answers live in [DECISIONS.md](../docs/paper/decisions.md). They are revisited only
when new evidence changes a decision.


Maintenance on 2026-10-25: [CI removal and regression follow-up](../records/repository-history/2026-10-25-ci-removal/README.md) removes the hosted workflow, disables repository Actions and fixes obsolete legend test expectations. All 47 repository tests and 83 targeted figure tests pass. [Read-only reconciliation](../records/figure-reviews/authority-reconciliation-20261025/diagnosis.json) identifies later Figure 1 H and Figure 2 C/F/I scientific drafts behind the four hash mismatches; local state/caption reconciliation was initially open and is completed in the follow-up below. Follow [the completion follow-up](07_FIGURE_BUILD_TOOLS.md#completion-follow-up-25-october-2026) before further styling or release claims.

## Dependencies and first release

Baseline/input validation → reviewed cohort → outcomes and justified inference → validated learner representation → Figure 4 analysis → reviewed Figures 1–4 → first release. Figure infrastructure and Figure 1/2 review can progress alongside upstream work; paper claims wait for their applicable scientific evidence.

Supplementary composition, tail mechanistic analyses and imaging do not block the first release. Cleanup is independent of analysis order. Phase A retains pending payloads; Phase B requires verified JOAQUIM transfer.

## Updating status

Record only scoped accepted changes and their evidence. Keep completed descriptions in documentation, decisions in the decision register, and per-file transfer state in the archive inventory. Do not mark unrelated work complete.

Completion follow-up on 2026-10-25: [guarded saved-draft reconciliation](../records/figure-reviews/authority-reconciliation-20261025/README.md) resolves all four Figure 1/2 artifact conflicts and refreshes the ordered collection. The [BLUP comparison](../docs/paper/figures/figure-03/blup-comparison.md) now contains completed numerical calibration/validation and an SSD-owned decision study. Accepted freezes, Figure 3 selection, primary cohort and Figure 4 alternatives remain unchanged.
