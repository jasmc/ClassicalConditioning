---
name: paper-figures
description: Show, create, edit, compare, verify or freeze paper figures and panels in the ClassicalConditioning repository, using scoped selections, editable SVGs and guarded publication. Use for figure legends and linked-panel consistency; exclude unrelated analysis and repository cleanup.
---

# Paper figures

Resolve paths from the ClassicalConditioning repository root. Apply
[AGENTS.md](../../../AGENTS.md); the [iteration workflow](../../../docs/paper/figures/shared/iteration-workflow.md#task-routes)
owns procedures. Read its common authority rules and the sections for the actual
task, rather than loading the entire workflow and all linked references.

## Resolve scope and state

Start with [Current figures](../../../docs/paper/figures/README.md), then the
requested scope's selection records, `current-state.json`, associated legend
and bound evidence. Authenticate consumed sources using supported resolution;
check external storage first. A newer filename, cache or open tab is not state.
Scientific selection and presentation selection remain separate. Identify
unfrozen scope; a panel freeze never approves the whole assembly.

Choose the relevant workflow route:

| Request | Read in the workflow | Action |
|---|---|---|
| Show one figure/panel | Display and working formats | Open the exact state-selected minimal SVG-and-legend HTML; inspect the rendered scope. |
| Show the main-paper figures | Main-paper collection | Run `.venv/bin/python scripts/main_paper_figures.py build`, then `display-target`; inspect the authenticated collection. |
| Compare retained versions | Ordered version browsing; maintained review library | Use the supported ordered browser; preserve each version's legend and source bindings. Browsing does not change selection. |
| Create/edit a draft or legend | Operational commands and concurrency; Verification before accepting a candidate; Linked panels and legends | Capture pre-edit hashes and use the scope's supported guarded builder. |
| Verify selected artwork | Verification before accepting a candidate; relevant scoped mappings/evidence | Authenticate inputs, inspect the exact rendering and report partial coverage. Verification does not select or freeze science. |
| Style or consistency work | Semantic registry and linked-panel checks; applicable geometry/frame sections | Resolve named roles and recorded relationships; update scoped registry/evidence and measure actual final-scale output. |
| Freeze | What happens when the author freezes; element specification | Prepare a separate candidate and complete the explicit gate. |

For new presentation/style work or verification against journal rules, also read
[Nature guidance](../../../docs/paper/figures/shared/nature-style.md). Query the
needed `nature-main` profile roles once per build and reuse the shared theme.
Historical defaults remain reproduction-only. Do not restyle just to display.
For future layout, read **Invisible layout frames** and the active policy;
existing drafts need a scoped edit request before retrofit. Read plans/history
only for unresolved rules, changed scientific scope or gaps exposed by checks.

## Task-specific safeguards

- **Display:** show only the requested SVG and matching scoped legend by default;
  follow state `default_display` and `default_display_format`.
  other scopes, visible notes/provenance, comparisons, versions and frozen views
  require an explicit request. Keep provenance nonvisual. Panel isolation must
  preserve authenticated coordinates/resources; scrolling a full figure is not
  isolation. Keep the editable SVG available without creating a competing
  standalone working SVG for a panel preview. Verify the actual saved page before
  reporting a successful open; report unavailable browser capabilities honestly.
- **Edit:** preserve manual/uncommitted work, authentic raster data, addressable
  SVG groups/text/paths and scientific mappings. Repeatable edits belong in
  supported code. Maintain the scope's one active SVG/HTML pair; preserve
  historical versions according to recorded dispositions. Use temporary previews
  and remove task-created render files after verification.
- **Consistency:** a local request changes its named scope/property. Preserve
  recorded group requirements; propagate a shared change only through the named
  scope. Record layout, scientific and presentation relationships distinctly.
  IDs, similar styling and frame containment do not prove data identity, spacing
  or full consistency. Preserve unresolved coverage and scientific definitions.
- **Legend:** each new scope needs a matching source-bound legend; a figure's
  panel paragraph plus necessary shared definitions can serve without duplication.
  Update existing legends only where definitions/explanations change. Keep units,
  essential ticks and scientific marks in the artwork.
- **Publication:** snapshot state, working artifacts and bound inputs before
  edits, then recheck under locks. Publish verified artifacts/evidence before the
  atomic state pointer. A conflict or missing storage stops publication.

Run supported commands from the root with `.venv/bin/python`; consult the
[tool catalogue](../../../scripts/README.md) for implemented scope/options.
`figure_working_copy.py` and `paper/shared/build_legend_review.py` target Figure 1;
use its `begin` ticket and `--update-current` route. Do not send other figures
through those tools or invent flags. Use their scoped builders and
[assembly safeguards](../../../docs/paper/figures/shared/assembly-preflight.md),
implementing equivalent guards if necessary before publication.

## Explicit freeze only

Use the [element specification](../../../docs/paper/figures/shared/figure-element-specification.md)
for identities/mapping requirements and the resolved Nature profile for current
presentation values; do not reuse historical compatibility typography. Complete
independent candidate work before asking about unresolved scientific choices or
exceptions. Prior scoped approvals persist; appearance is not approval.

Run `.venv/bin/python scripts/freeze_figure.py --candidate <candidate.json>
--check-only`, resolve every issue, then publish with `--output <new-freeze.json>`.
The gate validates prepared mappings/evidence; it does not classify elements,
style artwork, perform visual review or choose scientific methods. Update
accepted scoped selection/navigation and presentation state after success;
inventory previous versions/dependencies for author disposition. Preserve pending
versions. The [archive procedure](../../../records/repository-archive/README.md)
governs separately authorized transfer/pruning; this skill authorizes no cleanup.
