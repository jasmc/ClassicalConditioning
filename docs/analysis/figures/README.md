# Numbered figure references

**Current authority update, 2026-10-10:** [Scoped freeze index](freezes/README.md) identifies Figure 1 E/F–H V12 and Figure 2 D/E B/G. Older formulas, proposed registry status and examples below describe historical/proposed routes and must not override those selections. Exact frozen bytes are preserved; current paper assemblies and remaining scientific gates still require review.

For every panel or whole-figure freeze, use the
[figure element identities, shared style and freeze contract](FIGURE_ELEMENT_SPECIFICATION.md)
and [versioned element/style configuration](../../../configs/paper-figures/figure-elements.json).
The [repository instructions](../../../AGENTS.md) require applying the defaults
to a reviewable candidate and resolving only outstanding scoped exceptions.
The specification inventories the older assemblies and newer panel revisions;
it does not retroactively restyle historical artifacts. New freezes use the
[freeze command](../../../scripts/freeze_figure.py); its checks run only at
freeze time. The [SciFigEditor source review](SCIFIGEDITOR_REVIEW_2026-10-09.md)
records related semantic editing ideas.

| Number | Document | Purpose |
| --- | --- | --- |
| 01 | [Paper panel provenance](01_PAPER_PANEL_PROVENANCE.md) | Main and supplementary panels: inputs, calculation, renderer, and status. |
| 02 | [Paper figure specification](02_PAPER_FIGURE_SPECIFICATION.md) | Intended layout, shared visual rules, freeze conditions, and command. |
| 03 | [Figure pipelines](03_FIGURE_PIPELINES.md) | Legacy and corrected figure families and how they differ. |
| 04 | [Review variants](04_REVIEW_VARIANTS.md) | Version-by-version comparison of generated review alternatives. |
| 05 | [Draft comparison](05_DRAFT_COMPARISON.md) | Draft image versus current code and proposed panels. |
| 06 | [Figure 4 learner profiles](06_FIGURE4_LEARNER_PROFILES.md) | Classifier manifest, run order, signed signal, and provenance. |
| 07 | [Figure 4 handover](../../maintenance/transfers/FIGURE4_TRANSFER.md) | Current checkout, transfer requirements, run commands, and remaining paper gates. |

Dated Figure 4 execution and readiness records:

- [Delay cohort readiness, 2026-09-24](DELAY_FIGURE4_READINESS_2026-09-24.md) — authenticated cohort and provisional classifier comparisons.
- [Fixed 3sTrace exploratory run](TRACE3_EXPLORATORY_RUN_2026-09-24.md) — partial-assay execution, corrected response window, and provisional learner identities.

Use the current handover for transfer instructions; the old working-tree patch is retained only as historical evidence. The approved full three-assay Figure 4 still depends on reviewed cohorts and Gate L.

For the intended Figure 1 lettering with protocol B, see the [C/D trace version guide](FIGURE1_CD_TRACE_VARIANTS.md) before choosing the paired tail-angle and vigor presentation.
The [pre-CS baseline review](BASELINE_WINDOW_REVIEW.md) inventories which analyses currently use [−15, 0) s, [−20, 0) s, or an earlier-than−15 s reference.

The [machine panel registry](../../../configs/paper-figures/behavior-paper.json) records proposed panel IDs and blocked reasons. The [review variant registry](../../../configs/paper-figures/review-variants.json) records generated alternatives.
