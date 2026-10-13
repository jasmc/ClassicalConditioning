---
name: paper-figures
description: Show, create, edit, compare, verify or freeze paper figures and panels in the ClassicalConditioning repository, using scoped selections, editable SVGs and guarded publication. Use for figure legends and linked-panel consistency; exclude unrelated analysis and repository cleanup.
---

# Paper figures

Use this skill only in the ClassicalConditioning repository. Resolve all paths
below from its root; do not depend on a machine-specific checkout path. Read
[repository protections](../../../AGENTS.md) and the authoritative
[iteration workflow](../../../docs/paper/figures/shared/iteration-workflow.md)
before figure work. Maintain procedures there, rather than copying them here.

## Resolve the requested state

Read [Current figures](../../../docs/paper/figures/README.md), the requested
figure's selection records, legend and scoped `current-state.json` when present.
Check storage and authenticate bound sources before loading or regenerating.
Resolve relocated artifacts through archive indexes; do not choose by age.

- **Show:** follow the recorded `default_display` and open/show its authenticated
  SVG itself. The author selected v10 as the sole Figure 1 working version on
  2026-10-13; default display is the active draft. Show frozen selections only
  when explicitly requested. Identify unfrozen scopes and preserve scientific freezes.
- **Create/edit/compare:** capture starting hashes before changing code or
  inputs. Implement repeatable edits in supported code. Preserve addressable
  SVG groups/text/paths, source mappings and authentic embedded raster data.
  Maintain one recorded working SVG per frozen/draft state and one
  self-contained HTML review with requested versions, legends and provenance.
  Use temporary/in-memory previews and remove task-created render files after
  verification. An XML editor does not imply graphical drag-and-drop support.
- **Verify:** read the selected semantic registry, distinguish structural IDs
  from scientific roles, check linked-panel relationships and measure the
  exact saved SVG at final assembly scale. Perform real renderer/visual review;
  record unresolved measurements and partial coverage honestly. Create a
  companion legend for each new panel/figure; edit existing legends only when
  definitions or necessary explanations change. Keep essential units, axes,
  ticks and scientific marks visible; move explanatory clutter to the legend.

## Use supported tools within their implemented scope

Run commands from the repository root with `.venv/bin/python`. Read the
workflow's concurrency/publication section for destinations and command options.

- `scripts/figure_working_copy.py verify` authenticates Figure 1 working state.
  Use `begin --ticket <temporary-ticket.json>` before Figure 1 edits.
  `prepare --output-dir <existing-external-directory> --revision <revision>`
  bootstraps/reuses authenticated copies only when needed. Never refresh over
  unreviewed manual edits.
- `scripts/build_figure_legend_review.py --baseline <ticket.json> --output
  <new-external-review.html>` publishes the Figure 1 draft/review with guarded
  hashes, measurements and atomic state update, preserving Figure 2.
- For other figures, use their supported builder and
  [assembly safeguards](../../../docs/paper/figures/shared/assembly-preflight.md).
  The two tools above currently target Figure 1; do not direct another figure
  through them or invent unsupported flags. If needed, implement equivalent
  scoped state/locking support before publishing that figure.

## Explicit freeze only

Read the workflow's freeze section and the
[element specification](../../../docs/analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md).
Use [figure-elements.json](../../../configs/paper-figures/figure-elements.json)
for role identifiers, numerical defaults and plot-family rules. Prepare a
separate styled candidate with exact source/data mappings, final-scale evidence,
protection and approved scoped exceptions. The gate validates; it does not
classify elements, apply styles or perform scientific/visual review.

Run `.venv/bin/python scripts/freeze_figure.py --candidate <candidate.json>
--check-only`; resolve every issue, then run with `--output <new-freeze.json>`.
Complete independent candidate work before asking about unresolved scientific
choices or exceptions. Record exception ID, role, property/value, reason, scope
and approval evidence. Prior scoped approval persists; appearance is not approval.
After success update scoped selection/navigation and presentation state, then
inventory earlier versions for author disposition. Preserve all pending versions.

For approved transfer/pruning, follow the
[archive procedure](../../../records/repository-archive/README.md) and use
`scripts/repository_archive.py`; never treat this skill as cleanup authorization.
