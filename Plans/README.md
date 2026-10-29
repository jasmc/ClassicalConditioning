# ClassicalConditioning implementation plans

This folder contains future work, this navigation page and the [implementation index](IMPLEMENTATION_STEP_INDEX.md). Numbers order work packages; explicit dependencies determine execution.

| Plan | Responsibility |
|---|---|
| [00 Remaining work and execution order](00_REMAINING_WORK_AND_EXECUTION_ORDER.md) | Cross-plan priorities, dependencies and acceptance map; detailed work remains in its owning plan |
| [01 Paper input validation and reproducibility](01_PAPER_INPUT_VALIDATION_AND_REPRODUCIBILITY.md) | Authenticate paper inputs, processing and reproducibility evidence |
| [02 Fish discarding, eligibility and paper cohort](02_DISCARDING_AND_COHORT_IMPLEMENTATION.md) | Combine legacy rules for later review and establish one approved cohort |
| [03 Figure 2 — learning onset](03_FIGURE_2_LEARNING_ONSET.md) | Implement and validate Figure 2 acquisition/onset analysis |
| [3.1 Figure 2 — alternative analyses](03.1_FIGURE_2_ALTERNATIVE_ANALYSES.md) | Population model comparison, common comparison contract and shared execution checklist |
| [04 Figure 3 — learner analysis and validation](04_FIGURE_3_LEARNER_ANALYSIS.md) | Select and validate individual-learning representation and Figure 3 |
| [4.1 Figure 3 — alternative analyses](04.1_FIGURE_3_ALTERNATIVE_ANALYSES.md) | Corrected BLUP, direct-contrast and individual-trajectory comparison |
| [05 Figure 4 — CR profiles](05_FIGURE_4_CR_PROFILES.md) | Build Figure 4 conditioned-response profiles and the consistent integrated analysis run |
| [5.1 Figure 4 — response models and pooling](05.1_FIGURE_4_BOUT_AVAILABILITY_AND_POOLING.md) | Individual/population LME and GAMM comparisons, bout availability, and CR shape, magnitude and peak timing |
| [06 Analysis and statistics](06_ANALYSIS_AND_STATISTICS.md) | Approve outcomes, models, uncertainty and inferential claims |
| [07 Reliable figure builds and panel replacement](07_FIGURE_BUILD_TOOLS.md) | Current selections, reproduction, final-scale QC and reviews; [Nature completion handoff](07_FIGURE_BUILD_TOOLS.md#6-nature-completion-handoff-2026-10-17) bundles remaining implementation and latest author constraints |
| [08 First release — Figures 1–4](08_FIRST_RELEASE_FIGURES_1_TO_4.md) | Scientific/presentation acceptance gates and immutable release manifest for the first four figures |
| [09 Supplementary figures](09_SUPPLEMENTARY_FIGURES.md) | Supporting controls, sensitivity analyses and supplementary panels |
| [10 Tail mechanisms](backlog/10_TAIL_MECHANISTIC_ANALYSES.md) | Later tail-mechanism analyses |
| [11 Behavior/imaging integration](backlog/11_BEHAVIOR_IMAGING_INTEGRATION.md) | Optional imaging integration |
| [12 Study reader verification](12_STUDY_ARTIFACT_STORAGE.md) | Remaining browser/notebook/responsive/print QA; ongoing source and note safeguards live in the study policy |

The first release covers main Figures 1–4 and necessary scientific evidence. Supplementary composition, tail mechanisms and imaging are separate. Required learner validation and main-claim controls remain mandatory.

Decimal subplans sit under their parent workstreams: 3.1, 4.1 and 5.1 use filenames beginning 03.1_, 04.1_ and 05.1_. The parent plan retains scientific ownership; explicit dependencies determine execution.

## Completed plan accounts

- [Folder-audit follow-up, Plan 15](../records/repository-history/2026-10-28-folder-audit-implementation/completed-plan-15.md): full completed plan; [implementation and preservation](../records/repository-history/2026-10-28-folder-audit-implementation/README.md).

- [Script command organization, Plan 13](../records/repository-history/2026-10-24-scripts-audit/completed-script-command-organization-plan.md): exact completed account; later maintenance is recorded in the [folder-audit implementation](../records/repository-history/2026-10-28-folder-audit-implementation/README.md).

## Authorities and lifecycle

- [Governance](../docs/project-maintenance/governance.md) records durable rules.
- [Decisions](../docs/paper/decisions.md) records approved scientific choices.
- [Paper review agenda](../docs/paper/review.md) and [panel concerns](../docs/paper/figures/README.md) keep criticism separate from approval.
- [Freeze index](../records/frozen-analyses/selections.md) identifies exact scoped freezes.
- [Archive instructions](../records/repository-archive/README.md) describe pending external payloads.

Put remaining work in its owning numbered plan; update progress only in the implementation index. Move implemented descriptions to documentation. Historical versions remain in Git. No plan approves a scientific change by itself.

Plans 01–13 were renumbered consecutively on 2026-10-12. The author-requested
00 roadmap was added on 2026-10-25 without renumbering those work packages.
Numbers describe the narrative/work-package order; explicit dependencies determine
execution. Statistics is listed after Figure 4, but approved inferential design
is still required before final figure analyses. Former learner subplans were
consolidated; completed repository cleanup is retained in the historical
implementation index.

## Placement and lifecycle

Add a numbered plan only for remaining implementation work; do not reuse a
retired number. Keep goals, dependencies, tasks and acceptance in their owning
plan, with workstream progress only in the implementation index. When work is
complete, retain its evidence in records and its operating rules in docs,
updating active references before retiring the plan. Split mixed completed and
future material without losing unresolved tasks. Dated historical statements
do not serve as current status or scientific approval.

See the [folder policy](../docs/project-maintenance/structure.md) for shared preservation and ownership rules.

## Retired-plan navigation exceptions

`13_SCRIPT_COMMAND_ORGANIZATION.md` and
`15_REPOSITORY_FOLDER_AUDIT_FOLLOW_UP.md` are compact compatibility pointers for
links in preserved audit records. Their full bodies live only in repository
history. They are neither active plans nor progress boards; retain these useful
reading routes while the immutable audit links depend on them.
