# ClassicalConditioning implementation plans

This folder contains future work, this navigation page and the [implementation index](IMPLEMENTATION_STEP_INDEX.md). Numbers order work packages; explicit dependencies determine execution.

| Plan | Responsibility |
|---|---|
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

The first release covers main Figures 1–4 and necessary scientific evidence. Supplementary composition, tail mechanisms and imaging are separate. Required learner validation and main-claim controls remain mandatory.

Decimal subplans sit under their parent workstreams: 3.1, 4.1 and 5.1 use filenames beginning 03.1_, 04.1_ and 05.1_. The parent plan retains scientific ownership; explicit dependencies determine execution.

## Authorities and lifecycle

- [Governance](../docs/project-maintenance/governance.md) records durable rules.
- [Decisions](../docs/paper/decisions.md) records approved scientific choices.
- [Paper review agenda](../docs/paper/review.md) and [panel concerns](../docs/paper/figures/README.md) keep criticism separate from approval.
- [Freeze index](../records/frozen-analyses/selections.md) identifies exact scoped freezes.
- [Archive instructions](../records/repository-archive/README.md) describe pending external payloads.

Put remaining work in its owning numbered plan; update progress only in the implementation index. Move implemented descriptions to documentation. Historical versions remain in Git. No plan approves a scientific change by itself.

Plans were renumbered consecutively on 2026-10-12. Numbers describe the narrative/work-package order; explicit dependencies determine execution. Statistics is listed after Figure 4, but approved inferential design is still required before final figure analyses. Former learner subplans were consolidated; completed repository cleanup is retained in the historical implementation index.
