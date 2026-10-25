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

For creation, style edits, verification or journal-rule work, read the maintained
[Nature target](../../../docs/paper/figures/shared/nature-style.md). It separates
journal requirements from exact author/project choices, including axis geometry.
Use its profile lookup and explicit shared-theme adapter for new paper candidates;
use the workflow’s scoped migration route for an explicit artwork update.
Remaining production/scientific checks stay open. Existing selected
artwork is not automatically compliant; showing it does not request migration.
Pre-Nature styling is retired for new paper work. Unprofiled theme calls and
historical style contracts remain only for compatibility/reproduction; do not
reuse their numerical typography as an alternative target. The
[retirement record](../../../records/repository-archive/2026-10-17-retired-styling/README.md)
indexes archived guidance and retained dependencies.
Preserve historical contracts and use the existing scoped registry/publication
pipeline; presentation checks do not execute an unrequested freeze.

For routine styling, query only the needed profile section and named registry
roles/groups. Load one profile snapshot per build and pass the same theme to
axes/stimulus helpers inside `matplotlib.rc_context()`. Consult plans/history
when a rule is unresolved, scientific scope
changes or supported checks expose a gap. Reuse the implemented helper and
source-bound evidence rather than copying per-panel values or adding another tool.

## Resolve the requested state

Plural main-paper requests (for example, “show me the figures of the paper”)
use the ordered collection: run `scripts/main_paper_figures.py build` with
`.venv/bin/python`, then `display-target`, and open/inspect its authenticated
HTML in the built-in browser. This refreshes from selected scoped figure states
and requires current passing layout evidence. A request for a particular figure
or panel still shows only that scope and its legend. See the workflow's
**Main-paper collection** section; collection inclusion does not create a freeze.

Read [Current figures](../../../docs/paper/figures/README.md), the requested
figure's selection records, legend and scoped `current-state.json` when present.
Check storage and authenticate bound sources before loading or regenerating.
Resolve relocated artifacts through archive indexes; do not choose by age.

- **Show:** open only the requested figure/panel’s selected SVG and associated
  legend in minimal self-contained HTML in the built-in browser. No page header,
  navigation, notes, provenance display, other figures/panels, version switches
  or comparisons unless explicitly requested. Keep provenance nonvisual. Follow scoped
  state (`default_display` and `default_display_format`); Use the single current pair resolved from state.
  Keep the matching editable SVG as the figure artifact; open it when requested.
  Identify unfrozen scopes. Show frozen selections/comparisons only on request.
  A panel request needs a view of that panel alone, preserving authenticated
  source coordinates/resources; scrolling to a panel in a full figure is not
  isolation. Do not create extra standalone working SVGs for panel previews.
- **Create/edit/compare:** capture starting hashes before changing code or
  inputs. Implement repeatable edits in supported code. Preserve addressable
  SVG groups/text/paths, source mappings and authentic embedded raster data.
  Maintain one active working SVG and its matching self-contained HTML review
  with requested versions, legends and provenance. Preserve historical records
  according to author dispositions; do not create competing working versions.
  Use temporary/in-memory previews and remove task-created render files after
  verification. An XML editor does not imply graphical drag-and-drop support.
- **Style edits:** resolve named roles and existing consistency groups before
  editing. Apply local requests locally; preserve and recheck recorded group
  requirements. Propagate a shared change only across the author-named scope,
  using supported code, registry updates and final-scale evidence. Semantic
  lookup alone does not automatically restyle other panels.
- **Ensure consistency:** implement the named scoped relationship/property in
  supported draft code, and update the existing draft registry and version-bound
  evidence as described in the workflow’s consistency-request procedure. A
  request does not by itself revise shared freeze rules or scientific definitions.
  Register related-panel groups by type: layout (equal measured gaps, shared
  axis layout), scientific (authenticated data/cohort/control/experiment links)
  and presentation style (named role groups, including cross-figure axis titles).
  Check each declared relationship at final scale; containment alone does not
  verify spacing. Preserve distinctions between visual grouping and shared data.
- **Verify:** read the selected semantic registry, distinguish structural IDs
  from scientific roles, check linked-panel relationships and measure the
  exact saved SVG at final assembly scale. Perform real renderer/visual review;
  record unresolved measurements and partial coverage honestly. Create a
  companion legend for each new panel/figure, with an explicit source path/hash
  and panel association in state. A figure legend’s panel-specific paragraph may
  serve as the panel legend; do not duplicate files unnecessarily. Show the
  associated legend with the SVG by default. Edit existing legends only when
  definitions or necessary explanations change. Keep essential units, axes,
  ticks and scientific marks visible; move explanatory clutter to the legend.

## Use supported tools within their implemented scope

Run commands from the repository root with `.venv/bin/python`. Read the
workflow's concurrency/publication section for destinations and command options.

- `scripts/figure_working_copy.py display-target` resolves the authenticated
  current Figure 1 HTML, SVG and legend paths from state. Open that exact HTML;
  do not copy a historical review to a generic `figure1.html` or reuse an old
  browser URL without checking its backing file and hash. Inspect the actual
  page before reporting a successful open.
- `scripts/figure_working_copy.py verify` authenticates Figure 1 working state.
  Use `begin --ticket <temporary-ticket.json>` before Figure 1 edits.
  `prepare --output-dir <existing-external-directory> --revision <revision>`
  bootstraps/reuses authenticated copies only when needed. Never refresh over
  unreviewed manual edits.
- `scripts/paper/shared/build_legend_review.py --baseline <ticket.json> --output
  <new-external-review.html>` publishes the Figure 1 draft/review with guarded
  hashes, measurements and atomic state update, preserving Figure 2. Author direction
  on 2026-10-13: update the same active HTML/SVG pair with `--update-current`;
  do not create a new working version for each edit. Preserve approved freeze candidates.
- For other figures, use their supported builder and
  [assembly safeguards](../../../docs/paper/figures/shared/assembly-preflight.md).
  The two tools above currently target Figure 1; do not direct another figure
  through them or invent unsupported flags. If needed, implement equivalent
  scoped state/locking support before publishing that figure.

## Explicit freeze only

Read the workflow's freeze section and the
[element specification](../../../docs/paper/figures/shared/figure-element-specification.md).
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

## Invisible layout frames

Future paper/supplementary/analysis figure edits follow the workflow's
**Invisible layout frames** section: shared outside blank strip, explicitly
owned content, final-scale browser evidence and scoped freeze checks. The
shared inset is author-selected and active in the layout policy. Do not retrofit
current drafts without a scoped edit request; future evidence must bind the active policy.
