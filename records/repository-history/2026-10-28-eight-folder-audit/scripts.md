# Independent audit: scripts

Audit date: 28 October 2026. Baseline: `702193b9591b62bf12820bc2b01ca5e8928f9528`.
Focused static review of command placement, several later trace workflows,
imports, temporary roots and publication behavior. No command producing science,
artwork or figure state was executed for classification.

## Ownership

Keep supported entry points, scoped workflows, required compatibility facades
and explicit reproduction/audit tools. Reusable implementation belongs in src;
completed recipes need disposition and exact recovery, not automatic replay.

## Findings and dispositions

- **S1 — Reconcile later command growth with the declared layout.** There are
  67 top-level commands, including the seven standing facades; all 67 have
  catalogue links. [The guide](../../../scripts/README.md) and Plan 13 describe
  the earlier scoped organization but do not explain the later 60 root entries.
  The static scan found 105 script-to-script import occurrences, including bare
  sibling imports. This is placement/dependency debt, not proof that the tools
  are unused. Prioritize the eight explicitly mapped candidates in the
  reconciliation, extracting helpers before changing paths.
- **S2 — Remove temporary-host assumptions from supported publication routes.**
  `finish_figure2_trace10.py:14`, `finish_trace10_example.py:11`,
  `finish_trace10_learner_selection.py:20` and `publish_figure2_trace10.py:12`
  embed an 18 October Windows visualization directory. The last command also
  mutates another driver's `FINAL` and `TEMP` globals before reuse. Introduce
  explicit candidate/evidence paths and parameterized helpers in the supported
  successor routes; preserve old inputs/evidence and historical originals.
  Verify flag/receipt contracts without executing publication during refactoring.
  The selected 10sTrace-example state binds `finish_trace10_example.py` by its
  exact current source hash. That candidate therefore requires original-source
  recovery before a move; ordinary import repair alone is insufficient.
- **S3 — Preserve science when extracting helpers.**
  `process_trace10_additions.protocol` is consumed by source-alias tests;
  `finalize_trace3_additions.check_run` and
  `resume_trace10_additions.dispositions` implement tested completion/recovery
  contracts. Move these reusable contracts to an owning package module before
  retiring/moving their drivers. Keep protocol-specific timing and stage rules
  distinct. Source relocation alone does not authorize replaying intake or
  updating selected state.

## Proposed instruction wording — not active

Owner: `docs/project-maintenance/script-lifecycle.md`; scripts README stays the
command catalogue and links to that owner.

> Place new commands under their maintained operation/figure/panel scope. Keep
> root facades only for explicit compatibility contracts. Document the command's
> lifecycle role, inputs, outputs and modifying/publication modes in its catalogue
> entry or operating guide. Reusable helpers belong in src; do not reuse a driver
> by overriding its globals. Accept explicit candidate/evidence locations rather
> than a personal temporary workspace. Trace callers, source bindings and recovery
> before a move or retirement; preserve scientific definitions and guards.

## Verification and limits

The external evidence includes import edges and exact temporary-path lines.
Static imports omit dynamic/resource/subprocess relationships; the candidate
consumer inventory adds textual references but is not proof of full closure.
Targeted state inspection performed 54 hash checks across 39 directly bound JSON metadata files;
four records exceeded the bounded content-review limit and remain uninspected
internally. No artwork/numerical payload was read. The separate full external
checker owns byte/inventory validation, not this content-review sample.
Later moves require provenance-closure, compatibility/import and focused
behavior tests. The other root commands were inventoried, not individually
cleared for relocation or retirement; no command is approved for deletion here.
