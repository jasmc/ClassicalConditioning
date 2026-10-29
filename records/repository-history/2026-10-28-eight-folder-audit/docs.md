# Independent audit: docs

Audit date: 28 October 2026. Baseline: `702193b9591b62bf12820bc2b01ca5e8928f9528`.
Focused review of ownership/navigation, maintenance rules, relevant operating
guides and compatibility exceptions; not a manuscript-by-manuscript review.

## Ownership

Paper guidance owns purposes, approved definitions, explicit choices and scoped
state. Running-analysis owns operational interfaces. Project-maintenance owns
repository/storage procedures. Exclude completed audit narratives, future-work
progress and full study manuscripts except preserved compatibility originals.

## Findings and dispositions

- **D1 — Correct an obsolete launcher and default.** [Learning models](../../../docs/running-analysis/learning-models.md)
  line 168 recommends absent `run-allDelay-full-windows.ps1` and describes the
  default as `tail_length_weighted_angular_l1`. The supported dispatcher is
  `scripts/workflows/windows/delay.ps1`; its `pipeline-analysis` stage passes
  arguments to `stages/delay_pipeline_analysis.ps1`, whose `MetricId` default is
  `legacy_distal_angular_speed` (line 24). Replace the paragraph with the exact
  dispatcher/stage route and current default. Preserve historical analysis IDs
  and describe metric changes as separate sensitivity work, not an approved
  replacement method. PowerShell execution remains untested in this audit.
- **D2 — Qualify migration wording.** [Structure guidance](../../../docs/project-maintenance/structure.md)
  says local narratives remain pending Plan 12, whereas Plan 12 and the source
  registry record implemented source migration. Revise that paragraph to say
  ordinary study sources resolve externally, protected originals remain local,
  and remaining reader QA is tracked in Plan 12. The source registry currently
  contains 20 entries; avoid turning that count into another maintained status
  claim. Consumers are assistant routing and study-source readers.
- **D3 — Retain compatibility pages.** `docs/current-figures.md`,
  `docs/panel-review.md` and `docs/maintenance/archive/ARTIFACT_INDEX.md` are
  short navigation aliases; protected specification/review originals under
  `docs/analysis/figures/` are registered exceptions. Their apparently old
  hierarchy is intentional. No blanket deletion, merge or renaming is proposed.

## Proposed instruction wording — not active

Owner: `docs/project-maintenance/structure.md`; `docs/README.md` routes to it.

> Classify documentation by purpose: maintained definitions and explicit decisions
> belong in paper guidance, executable usage in running-analysis, and repository
> procedures in project-maintenance. Keep future work in Plans and completed
> evidence in records; full study manuscripts follow the external study policy.
> Link to the canonical owner instead of repeating its rules or current status.
> When updating a command example, verify its current path, dispatch stage and
> defaults against source. Preserve registered historical/byte-sensitive pages.

## Verification and limits

Existing local link/structure checks pass; inline-code command names escaped
those checks, explaining D1. Compare edited examples statically with the actual
stage and run existing configuration/command tests. No document was externalized
or rewritten. Approved definitions, every legend and the scientific correctness
of historical discussions were outside this review.
