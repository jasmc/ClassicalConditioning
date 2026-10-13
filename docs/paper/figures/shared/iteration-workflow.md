# Figure iteration, semantic mapping and chat handoff

Standing author directions: 2026-10-09 through 2026-10-12. The repository
[AGENTS.md](../../../../AGENTS.md) contains the concise rules; this document
contains procedures. Neither creates a scientific selection or an implicit freeze.

The repository [paper-figures skill](../../../../.agents/skills/paper-figures/SKILL.md)
routes figure requests to this workflow and the supported tools. It can be
selected automatically or invoked explicitly as `$paper-figures`. Root
instructions require reading it for figure work even if discovery is unavailable.
Keep the workflow here as the single procedural authority; the skill is an
entrypoint, not another scientific selection or style specification.

## Two separate current states

Scientific authority remains the scoped selections linked by
[Current figures](../README.md). Panel selections supersede older
whole-assembly sources only within their own scope.

[Figure 1 current state](../figure-01/current-state.json) is the sole presentation
pointer. It separates:

- `scientific_selection`: paths/hashes of the accepted scoped freeze records;
  a summary, not another source of scientific authority.
- `frozen_display`: reusable editable SVG assembled from those frozen panels,
  with authenticated source mappings. No whole-figure freeze is inferred.
- `draft`: the active presentation candidate, its editable SVG, consolidated
  HTML review and hashes. Showing the frozen display never resets this draft.
- Verification, open questions and reproduction information for resuming work.

The older verification/text-relocation/style-audit files remain version-bound
review evidence. Their paths do not override current state. If another figure
needs ongoing work, give it one similarly scoped state record; do not duplicate
scientific definitions or create several competing “latest” manifests.

## Display and working formats

Author selection on 2026-10-13: v10 is the sole Figure 1 working SVG.
“Show Figure 1” follows `default_display` in state, now `draft`. The author
authorized removal of the other standalone working SVGs; historical HTML
reviews and scientific freezes remain preserved. Requesting frozen selections
explicitly does not change the working version or imply a whole-figure freeze.

Open/show the SVG itself using the app's file panel. SVG contains editable XML
text, vector groups, paths and uniquely namespaced resource IDs. It can also be
opened in a graphical SVG editor, but opening it in a source editor does not
provide drag handles. A raster preview/HTML link alone does not fulfill the
editable display request. Do not claim a queued app open succeeded visibly.

Frozen originals remain immutable. The working SVG is a derived copy. Manual
edits are detected by its recorded hash: preserve them, reconcile with the
supported generator and review them before publishing a new candidate. Do not
silently regenerate over an edited copy. Authentic raster data inside a vector
SVG, such as heatmap cells, stays intact.

Use one self-contained HTML for author review, with requested versions, draft
legends and provenance embedded. The author's standalone SVG preference permits
one recorded frozen working copy and one recorded active draft copy; avoid
redundant media formats. Display previews are temporary/in-memory and removed
after verification. Permanent media belong on an explicit external destination.

## Operational commands and concurrency

The supported [working-copy tool](../../../../scripts/figure_working_copy.py)
creates vector-native frozen/draft copies from the authenticated existing review,
checks selected scientific records, imports available frozen registries and
measures the resulting SVGs. Bootstrap, with an existing external directory:

```sh
.venv/bin/python scripts/figure_working_copy.py prepare --output-dir '/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews' --revision 2026-10-12-v2
.venv/bin/python scripts/figure_working_copy.py verify
```

`prepare` reuses existing verified copies; `--refresh` creates new versioned copies while preserving older ones. It refuses changed/missing bound
artifacts instead of discarding edits. `verify` is read-only. Bootstrap does not
recompute science, style a frozen source or create a freeze.

Before editing, capture a starting ticket:

```sh
.venv/bin/python scripts/figure_working_copy.py begin --ticket /tmp/figure-01-edit-start.json
```

The [legend/layout generator](../../../../scripts/build_figure_legend_review.py)
now checks state/working-copy/source hashes, makes a new external HTML plus
editable draft SVG, measures the final SVG and updates state under an exclusive
lock. It also updates the version-bound text mapping within that transaction.

```sh
.venv/bin/python scripts/build_figure_legend_review.py --baseline /tmp/figure-01-edit-start.json --output '/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/<new-review-name>.html'
```

Capture starting hashes before editing code or candidate inputs, not only when
running a generator. At publication recheck them as well as the generator's
captured inputs. The generator requires the pre-edit ticket after state bootstrap and rechecks
it during publication. Code/source changes during rendering are also checked.
Guards cannot recover an earlier baseline an editor never recorded. A conflict stops
publication. Preserve both changes and reconcile explicitly. Do not remove a
state lock merely because it exists; verify that its owning process is gone.

Save new artifacts exclusively; verify their bytes; publish updated evidence;
replace the state JSON atomically last. Missing storage stops publication.
Shared HTML reviews retain unchanged other-figure assemblies. Do not regenerate
other figures just to update a Figure 1 display.

## Verification before accepting a candidate

Measure from the final nested SVG, not the styling recipe. The working-copy
measurement report records effective font dimensions/weights, affine coordinate
maps, stroke width ranges (or straight-line normal widths), tick lengths and
measured tick-to-label anchor distances. Anchor distance is not glyph-edge pad.
Explicit renderer/visual evidence is still needed for padding, clipping,
shared-axis intent, missing pixels and colourbar interpretation.

[Initial effective-presentation audit](../../../../records/figure-reviews/figure-01/effective-presentation-audit-2026-10-12-v2.json)
binds its measurements to the two SVG hashes. Later candidates have their own
version-bound report linked from state. Any discrepancy is unresolved evidence,
not a passed style claim. Full source hashes, tick labels, data geometry,
resources and ancestor transforms must be accounted for. Intentional display
transforms need named mapping/review evidence; an unchanged path alone cannot
prove its coordinate meaning is unchanged.

Check relevant tests, then inspect the exact saved rendering at final assembly
scale. Record only reviews actually performed. An SVG editing handle is not a
scientific classification; legacy elements without authenticated mappings stay
unresolved for any new freeze.

## What happens when the author freezes

1. Resolve the exact selected candidate, panel scope, scientific definitions and
   inputs. Preserve previous frozen bytes/hashes. A scientific change requires a
   selected scientific source, not a presentation exception.
2. Build/verify the semantic element registry from named renderer inputs and
   existing authenticated mappings. Assign unique IDs and record SVG/Matplotlib
   types, scientific role, quantity/units, data/event source, normalization,
   coordinate system, tick values and composite/mappable relationships. Record
   explicit/high classification evidence and a protection class. Never infer a
   role from colour, position or artist order without renderer/source evidence.
3. Style a separate candidate using common defaults at the intended final width.
   Preserve scientific inputs/results. Record each approved exception's ID,
   element role, property/value, reason, scope and actual approval evidence.
   The freeze request authorizes freezing; ask only about unresolved decisions
   after independent candidate work is complete.
4. Perform scientific-mapping, structure and visual reviews. Check effective
   opacity, stacking, shared axes, clipping, colourbar meaning and raster pixels.
   Record real evidence for renderer properties the SVG cannot prove.
5. Run the explicit [freeze gate](../../../../scripts/freeze_figure.py) in
   `--check-only` mode. Resolve every issue before running with `--output` to
   publish a new manifest. The gate checks ID coverage, compatible roles/types,
   mapped scientific geometry/text/resources/ancestor transforms, style values,
   final-scale fonts/strokes/tick lengths, visibility and hashes. Unsupported
   transforms/CSS or missing evidence fail; noninteractive cases cannot invent
   approval. Ordinary rendering never invokes this gate.
6. The gate rehashes inputs and exclusively writes the new manifest with the
   specification version/hash, registry/resolved styles, scoped exceptions,
   scale, source/data/export hashes, checks and authorization. It records
   already-created exports; it neither styles them nor guesses their meanings.
7. Update scoped current-selection/navigation records and current display state
   for the accepted scope. A panel freeze does not approve the full assembly.
   Then inventory earlier versions and dependencies, with keep/archive/delete
   recommendations. The author must review and choose which previous versions
   to retain, archive or delete. Keep all pending originals until those choices
   are recorded; freezing alone is not disposal approval. Execute only approved
   dispositions, verify external copies before pruning, preserve frozen bytes
   and shared active reviews, and update indexes/navigation. Report completed
   and pending dispositions. Check-only runs or rule edits do not trigger cleanup.


The [element/style specification](../../../analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md) and
[versioned role contract](../../../../configs/paper-figures/figure-elements.json)
remain the detailed freeze authority. The gate validates the registry prepared
by the agent/renderer; it does not itself establish scientific truth or conduct
visual review. The [archive procedure](../../../../records/repository-archive/README.md)
governs verified transfer and pruning.

## Current Figure 1 handoff

A–D, E and F–H V12 have scoped freezes; no whole-figure freeze is recorded.
The legacy-style/shorter-heatmap candidate remains the active unfrozen draft.
Figure 2 is preserved. The current D/E draft uses the same D measured frame table, named semantic
artists, matching axis rectangles and the D block labels (trial 9: Pre-Train).
E is recomputed from D's stored vigor with the inherited E bout mask; it no
longer claims to contain the exact frozen V12 rows. The author resolved the arrow relationship on 2026-10-13: the five arrows
are now in F, marking the Delay trials shown in D/E. G has no draft arrows.
See state for exact copies/hashes and evidence, and the editable
[Figure 1 legend](../figure-01/legend.md),
[editorial notes](review.md) and
[style audit](../../../../records/figure-reviews/panel-style-consistency-audit.json) for context.

A new chat needs only the active figure/panels, this workflow/state link and any
new author comments or storage limitation; read the saved records before work.

## Semantic registry and linked-panel checks

### Where consistency is established

- **Scientific definitions:** scoped accepted selections linked by
  [Current figures](../README.md), with shared definitions under
  [paper methods](../../methods/README.md). Each semantic registry maps stable
  element IDs to scientific roles, inputs, units, transforms and relationships.
  Matching appearance does not establish matching science.
- **Shared presentation contract:**
  [figure-elements.json](../../../../configs/paper-figures/figure-elements.json)
  is authoritative for reusable role IDs, numerical style defaults and
  plot-family rules; the
  [element specification](../../../analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md)
  explains them. These apply to separate future freeze candidates on explicit
  freeze requests, with author-approved exceptions limited to recorded scopes.
- **Rendering foundation:**
  [theme.py](../../../../src/classical_conditioning/figures/theme.py) supplies
  reusable fonts, dimensions, condition colors and axis helpers. Its use alone
  does not certify compliance: full automatic application of the role/style
  contract is not implemented. Draft edits follow selected author directions.
- **Verification:** compare linked-panel scientific identities and data-axis
  rectangles, then measure fonts/strokes/ticks through all transforms at final
  assembly scale and inspect the actual saved rendering. Store version-bound
  evidence linked from scoped state. The explicit freeze gate validates full
  visible-element mappings, protected science, styles, evidence and hashes
  against the versioned contract before a new freeze can be published.

Current coverage is partial: Figure 1 state records explicit D/E relationships
and selected F annotations; legacy A–D lack a complete new-freeze registry.
Historical style audits do not certify current artifacts. Consult the exact
state and evidence before claiming any panel or whole-figure consistency.

### Linked panels and legends

The registry supplies scientific meaning; the SVG structural inventory supplies
editable addresses. Neither substitutes for the other. AI reads the selected
source evidence, assigns roles and relationships, then the renderer applies
presentation to those explicit artists. Inspect actual exports and validate
identity/layout relationships before publishing a draft. The explicit freeze
gate additionally requires full visible-element coverage and genuine review
and protection evidence; a draft registry is not an accepted freeze.

Related panels must declare shared fish/cohort, trials, frame/time hashes,
coordinate origin and event definitions. Raw and scaled displays declare their
common numeric input, eligibility, bins, baseline and normalization. Alignment
checks compare data-axis rectangles and row labels/centres, not allocated outer
viewports; colourbars are outside the matched body. Use CS-relative time where
applicable and verify label/tick placement visually at final assembly scale.

Create a legend for each new panel/figure. Preserve it during presentation-only
edits; update just the affected definitions when inputs/meaning change. Keep
units and essential scientific marks in the figure, moving method explanations
and unnecessary identifiers into the legend. Embed the legend in the single
consolidated review.
