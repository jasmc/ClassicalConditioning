# Remaining work and execution order

Author-requested coordination plan, 25 October 2026. The scope is completion
of the current Figure 2 correction, research-review delivery and the first
Figures 1–4 release. Detailed tasks and acceptance stay in the owning plans
below; progress stays only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md).
This roadmap does not approve a scientific choice, select a candidate or request
a new freeze, reflow, deletion or hosted CI.

## Priorities and ownership

| Order | Remaining work | Owning plan | Dependency and completion evidence |
|---|---|---|---|
| 1 | Verify and publish the prepared Figure 2 canvas-overdraw correction | [07: immediate verification follow-up](07_FIGURE_BUILD_TOOLS.md#immediate-verification-and-delivery-follow-up) | Authenticate the recorded candidate and current inputs; collect fresh exact browser frame/visual evidence; resolve affected coverage; publish through supported scoped guards and refresh the collection last. Preserve geometry and scientific data. |
| 2 | Close source/runtime provenance gaps and validate the paper input denominator | [01: provenance follow-up](01_PAPER_INPUT_VALIDATION_AND_REPRODUCIBILITY.md#saved-analysis-provenance-follow-up) | Recover exact historical sources/runtime identities where possible; distinguish an independently reproduced candidate from original-run evidence. Reconcile every expected recording and timing/processing case. Missing evidence remains explicit. |
| 3 | Approve the primary technical cohort and defensible outcomes/inference | [02](02_DISCARDING_AND_COHORT_IMPLEMENTATION.md), [06: statistical follow-up](06_ANALYSIS_AND_STATISTICS.md#saved-ghi-statistics-scientific-follow-up), then [03](03_FIGURE_2_LEARNING_ONSET.md) | Establish label-independent C0/C1 membership, outcome/model/claim choices and applicable calibration, dependence, missingness and influence evidence. Run the approved analysis on authenticated inputs. Arithmetic agreement does not close these decisions. |
| 4 | Select and validate one Figure 3 individual-learning representation | [04](04_FIGURE_3_LEARNER_ANALYSIS.md) | Review the completed BLUP comparison and unchanged legacy comparators. Resolve Gate L, information/uncertainty requirements and permitted claims; complete applicable assay validation. Keep acquisition, recovery and any extinction claim distinct; categories remain a scientific choice. |
| 5 | Finish Figure 4 using the approved representation and common cohort | [05](05_FIGURE_4_CR_PROFILES.md) | Use approved inputs/outcomes and the Figure 3 representation; complete within- and across-experiment comparisons with valid controls, joins, eligibility and non-circular interpretation. Adapt outputs if categories are declined. |
| 6 | Finish figure production and review-artifact verification | [07 H02–H10](07_FIGURE_BUILD_TOOLS.md#6-nature-completion-handoff-2026-10-17), [12: reader verification](12_STUDY_ARTIFACT_STORAGE.md#reader-verification-follow-up) | Complete scientific/element mappings, applicable style/accessibility checks, actual vector/PDF font/resource inspection, standalone/collection equivalence, responsive and print review. Exercise the newly delivered studies' notebook controls without touching personal notes. |
| 7 | Assemble and approve the first release | [08 R01–R10](08_FIRST_RELEASE_FIGURES_1_TO_4.md#release-preparation-and-acceptance-gates) | Reconcile all applicable scientific and production evidence, exact selections, captions, source tables and reproduction identities. Prepare the immutable release manifest; publish an authorized release/tag only after its gates pass. |

## Work that can proceed together

Start the exact-browser check and provenance recovery independently. If browser
access or storage is unavailable, preserve the prepared candidate and record
the missing evidence; continue source/documentation work. Do not substitute old
browser measurements or regenerate artwork to make a check pass.

Paper input validation precedes final cohort-dependent inference. Scientific
review can use the existing studies while inputs are being authenticated, but
final model/learner validation must use the approved input/cohort identities.
Figure 4 depends on that representation and cohort. Role registration, source
tables, accessibility assessment and review-reader QA can progress alongside
scientific work; certify final delivery only against the eventual selected bytes.

Further composition/resizing remains outside the current visual scope. Use the
latest box-only/no-further-resize instruction and Plan 07's scoped procedure.
Do not implicitly add Figures 3/4 to the selected main-paper collection.

## Evidence to reuse

- [Canvas correction record](../records/figure-reviews/figure-02/canvas-overdraw-20261025.json):
  the correction is prepared and native-reviewed; fresh browser verification
  and guarded selection remain necessary.
- [G/H/I saved-statistics audit](../records/scientific-evidence/analyses/figure-02-ghi-statistical-audit.json):
  numerical reconciliation and exact recovery aliases already exist. Its open
  source/runtime and scientific limitations are the follow-up scope.
- [Completed BLUP comparison](../docs/running-analysis/figure3-blup-comparison.md):
  reuse the Delay/control calibration and stress evidence; it does not select
  a learner method or establish full-assay validation.
- [Current Figure 2 methods](../docs/paper/figures/figure-02/current-methods.md)
  and [current figures](../docs/paper/figures/README.md): resolve scientific and
  presentation authority before resuming. Historical Methods remain preserved.

Repository/script organization, needed storage dispositions, current-versus-
historical documentation reconciliation and CI removal have implementation
evidence in the index. Do not reopen them as general cleanup or recreate CI.
Repeat checks or analyses only when changed inputs or a concrete unresolved
issue require it.

## Boundaries and completion

Required validation and source evidence for a main-paper claim remain necessary
even if no supplementary figure is published. Supplementary composition
([09](09_SUPPLEMENTARY_FIGURES.md)), tail mechanisms
([10](backlog/10_TAIL_MECHANISTIC_ANALYSES.md)) and imaging
([11](backlog/11_BEHAVIOR_IMAGING_INTEGRATION.md)) remain separate unless the
release's approved claims explicitly depend on them.

The work is complete when the applicable owning-plan acceptance criteria and
Plan 08 release gates have evidence and the remaining scientific decisions have
explicit scoped dispositions. An unavailable historical original may be recorded
as unrecoverable after documented searches; any replacement analysis needs its
own provenance and scientific acceptance. It must not be relabelled as the old run.
No missing evidence, unsupported claim or unreviewed candidate becomes approved
through this plan. Keep full study/discussion Markdown and HTML on JOAQUIM;
retain plans, durable rules and compact evidence links in the repository.
