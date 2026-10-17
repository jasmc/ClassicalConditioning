# Figure iteration, semantic mapping and chat handoff

Standing author directions: 2026-10-09 through 2026-10-13. The repository
[AGENTS.md](../../../../AGENTS.md) contains the concise rules; this document
contains procedures. Neither creates a scientific selection or an implicit freeze.

The repository [paper-figures skill](../../../../.agents/skills/paper-figures/SKILL.md)
routes figure requests to this workflow and the supported tools. It can be
selected automatically or invoked explicitly as `$paper-figures`. Root
instructions require reading it for figure work even if discovery is unavailable.
Keep the workflow here as the single procedural authority; the skill is an
entrypoint, not another scientific selection or style specification.

## Nature target and migration status

Author direction, 2026-10-16: use Nature's figure guidance for the next
paper-figure presentation revision, including the complete element registry,
scientific identities, typography, layout, captions and delivery checks.
The [maintained Nature target](nature-style.md) owns the selected rules and
exact project choices. The [Nature implementation checklist](../../../../Plans/07_FIGURE_BUILD_TOOLS.md#5-nature-main-figure-presentation-profile)
owns remaining code/artwork work, known fixes and sequence;
the [source comparison](../../../../records/figure-reviews/nature-guidance-20261016.md)
links the official guidance. Track implementation in the existing plan index.

The `nature-main` profile resolver, shared-theme adapter and scoped Figure 1/2
migration tooling are implemented; final migration publication and broader
renderer/production integration remain pending.
Use the coding entry point in the maintained guidance for exact values and input
hashes. Profile lookup does not publish state or execute the freeze gate.
Author direction, 2026-10-17: pre-Nature styling is retired for new paper work.
Its numerical instructions have been removed from current procedures and
[archived with their original bytes](../../../../records/repository-archive/2026-10-17-retired-styling/README.md).
Unprofiled renderers and historical contracts remain compatibility/reproduction
dependencies, not alternative paper styling targets. Resolve new typography and
stroke values through the Nature profile. Preserve compatible
author rules, including invisible frames, absolute axis spacing, right-aligned
keys and typed panel/scientific relationships. Current saved drafts remain
authoritative until replacements pass scoped guarded publication. This direction
does not select new science, freeze a figure or authorize historical cleanup.

## Two separate current states

Scientific authority remains the scoped selections linked by
[Current figures](../README.md). Panel selections supersede older
whole-assembly sources only within their own scope.

Each scoped current state is its figure's presentation authority, including
[Figure 1](../figure-01/current-state.json) and
[Figure 2](../figure-02/current-state.json). Figure 1's state separates:

- `scientific_selection`: paths/hashes of the accepted scoped freeze records;
  a summary, not another source of scientific authority.
- `frozen_display`: provenance/evidence for the assembly of frozen panels. Its
  derived working SVG may be absent under an author-approved single-copy policy.
  No whole-figure freeze is inferred.
- `draft`: the active presentation candidate, its editable SVG, consolidated
  HTML review and hashes. Showing the frozen display never resets this draft.
- Verification, open questions and reproduction information for resuming work.

The older verification/text-relocation/style-audit files remain version-bound
review evidence. Their paths do not override current state. If another figure
needs ongoing work, give it one similarly scoped state record; do not duplicate
scientific definitions or create several competing “latest” manifests.

## Display and working formats

Author direction on 2026-10-13: maintain one active Figure 1 working SVG and its matching review, resolved from current state. The initial v10 selection has been superseded by the author-requested edits. Update the same active pair with `build_figure_legend_review.py --update-current`; do not accumulate working versions. The approved v15 freeze candidate remains preserved separately.
“Show Figure 1” follows `default_display` in state, now `draft`, and opens its
authenticated HTML review (`default_display_format: html-svg-and-legend`). The author
authorized removal of the other standalone working SVGs; historical HTML
reviews and scientific freezes remain preserved. Requesting frozen selections
explicitly does not change the working version or imply a whole-figure freeze.

Opening a specific figure or panel displays only its selected working SVG and
associated legend, in minimal self-contained HTML in the built-in browser. The
legend is the sole default addition to the SVG. Do not add page headings,
navigation, review notes, provenance displays, other figures/panels, historical
versions or comparison controls unless the author explicitly requests them.
Keep source/provenance metadata nonvisual. An expanded review/comparison does
not replace the default minimal view or select a different working version.

Resolve Figure 1 with `.venv/bin/python scripts/figure_working_copy.py
display-target`; scoped state is authoritative for other figures. Check the
HTML, working SVG and associated legend path/hashes first. Open the exact
recorded HTML from its external destination. Do not copy an older combined
review to a generic localhost `figure1.html` or trust a previously open URL.
If an existing server route is used, verify its backing file/hash against state;
otherwise open a route serving the recorded file directly. Inspect rendered
content for the requested scope before reporting that it is open. Reuse a
matching recorded minimal page, or wrap the authenticated SVG using
`svg_editing_page` in the supported review builder; do not redraw science.
It embeds native SVG so vector element addresses remain available for browser
annotation. Annotation guides code edits; it does not promise graphical drag
handles. For panel-only requests, extract the authenticated panel scope with
its required resource definitions and coordinate transforms and include only
its legend content (plus necessary shared definitions). Preserve other panels
and the full working artifact; do not just scroll/crop a full-figure page or
create a competing standalone working SVG. Record the panel view’s source hash,
scope and bounds when one is produced.

A link alone does not fulfill an open/show request. If localhost serving is
needed, serve the external page without copying media or serving the repository.
Do not claim queued opens succeeded visibly or annotation was tested without
evidence. If annotation is unavailable, report the limitation and accept focused
comments. Preserve historical reviews separately and open them only on request.

The SVG remains the editable/publication artifact, with vector groups, text,
paths, source mappings and authentic embedded rasters. Open/show the SVG itself
on an explicit SVG request. A source editor supports XML edits; it does not
provide graphical drag handles. Do not claim a queued app open succeeded visibly.

Frozen originals remain immutable. The working SVG is a derived copy. Manual
edits are detected by its recorded hash: preserve them, reconcile with the
supported generator and review them before publishing a new candidate. Do not
silently regenerate over an edited copy. Authentic raster data inside a vector
SVG, such as heatmap cells, stays intact.

Use one active minimal HTML view and its matching working SVG. Include only
the scoped associated legend; retain provenance as nonvisual metadata. Produce
expanded review/comparison HTML only when explicitly requested. Figure 1’s
single-author-selected-draft policy permits only the active working SVG; frozen
scientific originals and historical reviews remain preserved. Future edits
publish a new matching pair and make it current under the guard; prior-version
dispositions follow author authorization. Avoid redundant media formats. Display previews are temporary/in-memory and removed
after verification. Permanent media belong on an explicit external destination.

## Operational commands and concurrency

Author direction on 2026-10-15 permits an internal Mac source cache of all
panels, including unfrozen artwork, for offline layout/aesthetic work. See
[Offline figure artwork](offline-artwork.md) for the local folder, exact scope,
copy evidence and draft-publication procedure. It is a source mirror; it does
not replace either figure's active presentation pointer or mirror raw analysis
dependencies.

The supported [working-copy tool](../../../../scripts/figure_working_copy.py)
creates vector-native frozen/draft copies from the authenticated existing review,
checks selected scientific records, imports available frozen registries and
measures the resulting SVGs. Bootstrap, with an existing external directory:

```sh
.venv/bin/python scripts/figure_working_copy.py prepare --output-dir '/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews' --revision 2026-10-12-v2
.venv/bin/python scripts/figure_working_copy.py verify
```

`prepare` reuses existing verified copies. Under Figure 1’s single-draft policy
it returns the selected copy and rejects `--refresh`; it must not recreate a
frozen working copy. Other bootstrap states may use `--refresh` to create
versioned copies while preserving older ones. It refuses changed/missing bound
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
Expanded shared reviews retain unchanged other-figure assemblies; minimal
working views omit them without modifying their records or sources. Do not regenerate
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

A–D, E and F–H have scoped freezes; the current F–H selection is the author-frozen 2026-10-13 historical-fish 1 s direct log-frame median managua_r version (Trace 20230307_12). V12 remains preserved as superseded F–H evidence; no whole-figure freeze is recorded.
The legacy-style/shorter-heatmap candidate remains the active unfrozen draft.
Figure 2 is preserved. The current D/E working pair uses the same D measured frame table, named semantic
artists, matching axis rectangles and the D block labels (trial 9: Pre-Train).
E’s accepted 2026-10-13 scoped freeze uses D’s stored vigor with the inherited bout mask: unbinned raw baseline and scaling, no black-trace log, 0–5 display, no baseline reference, exact current F bands/H color legend, and two continuous CS boundaries. Older E/V12 freezes remain preserved. D and whole-figure draft approval remain separate. The author resolved the arrow relationship on 2026-10-13: the five arrows
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

### When the author asks to “ensure consistency”

Treat a request such as “ensure consistency between D and E” as authorization
to implement and verify the named consistency in the active draft, and to record
it for future edits. Resolve the intended property from the request, annotations
and existing directions: scientific identity/processing, axes and alignment,
or presentation such as fonts, strokes, colors and labels. Do not assume two
panels must share all scientific definitions merely because they should match
visually. If the intended reference or scientific choice remains ambiguous,
show the relevant versions/options for the author to choose while completing
independent checks.

Extend the existing scoped draft semantic registry and state/evidence rather
than creating a competing “consistency registry”. Give each request a stable ID
in the scoped state’s `consistency_requests` list. Record the author’s direction,
panel/element IDs or scientific roles, property/relationship, reference and
expected value or tolerance, affected input/artifact hashes, implementation,
verification-evidence references and status (verified, partial or unresolved).
Map scientific relationships in the semantic registry; bind presentation
requirements and measured results in the version-bound presentation evidence.
Link full records from state when they are stored separately. A request is a
requirement, not proof that the figure already satisfies it.

Apply repeatable changes in supported renderer/assembly code. Recheck these
recorded requirements after later edits to the affected elements. Measure the
saved SVG at final assembly scale and inspect the matching rendered review;
record only checks actually performed. Preserve partial coverage and unresolved
items rather than declaring whole-figure consistency. Capture and recheck
starting hashes under the publication lock; publish registry/evidence and the
matching HTML/SVG before updating the atomic state pointer.

Do not modify an immutable accepted registry or frozen source. Changes to data,
masks, timing, units or normalization are scientific draft changes and must be
recorded explicitly. A scoped consistency request does not automatically modify
`figure-elements.json`, shared defaults, or other figures. Update the shared
role/style contract only when the author establishes a reusable rule with clear
scope; preserve versioning and approved exceptions. Normal draft verification
does not invoke the freeze gate.

### Style edits and propagation

Semantic lookup identifies scientific roles and named element groups; it is not
an automatic style synchronization engine. Full propagation of the reusable
role/style contract across all renderers is not implemented. Existing draft
registry coverage is partial, so report unmapped targets rather than inferring
roles from appearance.

Before a style edit, find the relevant named roles and recorded
`consistency_requests`. A local instruction changes only its stated scope. Keep
and recheck existing group requirements for the affected property. A shared
instruction (for example “use 9 pt axis labels throughout Figure 1”) authorizes
updating that property's named group in supported code and draft evidence;
use the consistency-request procedure above to record and verify it. Updating
registry text or search results alone does not change SVG styles. If a local
instruction conflicts with an established shared requirement, make the conflict
visible and resolve the intended scope with the author. Never silently change
unrelated panels, scientific definitions or accepted freezes.

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

Create an associated legend for each new panel/figure. Record its editable
source path/hash and figure/panel scope in state. Figure-level panel paragraphs
may supply panel legends without duplicate files; include needed shared
definitions. Preserve legends during presentation-only edits; update just the
affected definitions when inputs/meaning change. Keep
units and essential scientific marks in the figure, moving method explanations
and unnecessary identifiers into the legend. Show the scoped legend with its SVG in the minimal HTML. Full
legends, notes and comparisons beyond that scope require an explicit request.

## Invisible layout frames (author direction 2026-10-15)

This ordinary-draft layout requirement applies to future main, supplementary
and analysis figures. It does not change scientific definitions or historical
freezes. The author selected **1.0 mm** (tighter) on 2026-10-15 after the
calibration review. The active numerical authority and author evidence are in
`configs/paper-figures/layout-frames.json`; the former 1.5 mm recommendation
was not selected. New freezes using the current style contract require this
active policy and fresh frame evidence. Earlier calibration reports retain
their original pre-activation policy hashes as historical review evidence.

A frame is nonvisual allocation geometry: no fill, stroke, hit target or export
overlay. The figure and edge-panel allocations share the outside blank strip;
there is no second added figure margin. Outer allocations reach the canvas
boundary. Adjacent allocations may touch, leaving two insets between their
contents. Positive extra gaps require named grouping reasons and measured widths.
Standalone figures/panel exports have one outside strip. Associated HTML legends
are outside the artwork and are not included in its physical bounds.

Include panel letters, axis labels/ticks, titles, annotations, strokes, markers
and images in explicit owning panel groups. Shared colourbars and annotations
have their own figure-level `shared` allocation. Identify background surfaces by
explicit IDs; never discard white marks by appearance. Align actual data-axis
rectangles as well as named content edges. Natural aspect ratios and internal
whitespace are allowed: do not stretch science, crop marks, change scientific
coordinates or reduce established font sizes just to fill the available space.

### Interfaces and supported routes

`figures/layout_frames.py` supplies validation, inset geometry, metadata,
measurements and evidence authentication. SVG metadata `id="layout-frames"`
contains `scope` (`figure` or `panel`), `final_size_mm`, `inset_mm`, `allocations`
with `id`, `kind`, `box_mm` and `group_ids`, explicit `background_ids`, optional
`grouping_gaps`, named `alignments` and `data_axes` element mappings. Coordinates
are physical millimetres. Approved scientific clipping requires explicit clip
IDs and source evidence in `scientific_clips`; other clipping is removed for a
second render so viewport clipping cannot conceal margin intrusion. Unmapped,
unsupported and unmeasured content remains unresolved.

- Generic assembler: opt in through layout `layout_frames`, supplying
  `final_width_mm`, `inset_mm` and optional relationships. Panel boxes retain
  their explicit placement; content boxes are inset automatically. Labels must
  be placed explicitly within those bounds. Visible layout borders, crop repairs
  and aspect-distorting fits are rejected. `--panel` preserves the panel's final
  assembly scale and changes evidence scope to `panel`.
- Shared Matplotlib exporter: pass `layout_frames=<spec>` to `export_matplotlib_figure`.
  The figure size must already match the declared physical size. Tight cropping
  is disabled only during that export; the global theme and historical exports
  remain unchanged. A publication SVG is required for exact-artifact evidence.
  Allocate artists explicitly; axes are not automatically interpreted as panels.
- Figure 1 review builder: `--layout-frames <spec.json>` binds explicit metadata
  without rearranging the draft. Existing publication snapshots/locks apply.
  Effective-presentation reports include frame status. This does not infer
  ownership from anonymous structural handles or certify a legacy layout.

Existing direct `savefig`/SVG-writing scripts, including specialised Figure 2–4
and legacy analysis routes, are not automatically frame-aware. The route inventory
in `records/figure-reviews/layout-frame-implementation.json` lists those entrypoints.
Future work must use the supported adapters or report incomplete coverage; a
missing frame mapping/evidence cannot pass a new freeze. No current draft was
converted as part of introducing these tools.

### Browser evidence and publication

Run `.venv/bin/python scripts/check_layout_frames.py <exact.svg> --review
<new-external.html>` to prepare a separate comparison/check page. Open it in the
built-in browser and use **Measure saved SVGs in this browser**. Save its JSON
output, then run the checker again with `--browser-report <report.json>
--evidence <new-external.json>`. The checker is read-only unless an explicit
review/evidence destination is supplied. It never writes presentation state.
The review page is a verification aid, not the default minimal figure display.

The collector waits for fonts and image decoding, measures the actual SVG in a
browser and rasterises each owned allocation on transparent backgrounds at
600 dpi, including painted strokes, text, markers and raster content. A second
render removes unapproved clipping and nested viewport cropping. Expanded raster
bounds prevent the root canvas from concealing overflow. Unsupported effects,
missing ownership, decode failures and measurement limits fail unresolved.
The physical measurement tolerance is 0.1 mm, not permission to use a narrower
nominal margin. Named data-axis alignments are measured through SVG transforms.
Visual inspection at final scale is still required; numerical containment alone
does not prove aesthetic quality or legibility.

Evidence binds SVG bytes, metadata, final size/scope, policy and Python/browser
collector hashes. Changed inputs require fresh evidence. Provide the resulting
JSON as hash-bound `verification.layout_frames: {path, sha256}` in a new freeze
candidate. The gate re-evaluates measurements and rejects stale evidence,
calibration-only outputs, wrong panel scope and mismatched final width. A panel
freeze does not certify its surrounding figure. No legacy freeze is rewritten.
Publish future draft artifacts and evidence under the existing guards before
updating the atomic scoped presentation pointer.

Calibration uses `scripts/calibrate_layout_frames.py --output <new-external.html>`
with 1.0, 1.5 and 2.0 mm presets at 89 and 183 mm widths. It includes authenticated
selected excerpts as reference, with no compliance claim. It never changes current
SVG/HTML pairs or presentation pointers. Store permanent reviews on JOAQUIM;
if storage or sources are unavailable, record the limitation and continue
independent code work. Choose the inset only after reviewing the concrete page.

## Main-paper collection

Author selection on 2026-10-15: Figures 1 and 2, including current unfinished
panels, form the main-paper presentation collection. Inclusion is not a new
scientific freeze. Particular figure/panel requests keep the existing scoped
minimal views; only plural paper-figure requests use the collection.

Use `.venv/bin/python scripts/main_paper_figures.py build` to refresh from the
ordered selections in `docs/paper/figures/main-paper-current-state.json`, then
`display-target` to authenticate the exact recorded HTML. Open it in the built-in
browser and inspect the rendered figures and legends before reporting success.
A changed figure requires fresh current frame evidence before refresh; do not
serve an old collection as current. Add subsequent main figures only on explicit
selection. Frozen-only or comparison requests remain separately scoped.

The collection is self-contained native SVG, Figure 1/legend then Figure 2/legend,
at each selected SVG's native publication width, with responsive uniform screen
fitting and 9 pt serif legends. The initial 183-mm selection was superseded for
the current Nature revision by the author's 2026-10-17 compact 136-mm preference
where feasible. Never enlarge an intermediate-width SVG to fill 183 mm.
It has no visible review UI. SVG IDs and references are namespaced and each
artwork uses a declarative shadow root. Embedded font faces have figure-specific
names and are registered at document level because Chromium does not register
font faces declared only inside a shadow root. Provenance remains nonvisual.
Printing preserves artwork dimensions; long legends can continue on later pages.
No journal-template compliance is claimed.

Permanent HTML is `/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/main-paper-figures.html`.
The builder snapshots selected states/artifacts, scientific-selection records,
legends, frame evidence, policy and implementation; rechecks under scoped locks;
and updates the collection pointer last with rollback on failure. It refuses
unrecorded output collisions, manual hash changes, missing storage and stale
frame evidence. An unchanged authenticated build is reused. It never edits
individual figure state or recomputes science.

### Current assemblies: explicitly authorized layout migration

The same author request explicitly authorizes applying the active 1.0 mm policy
to both current assemblies. `scripts/layout_main_paper_figures.py` provides
`initialize`, `arrange`, `refine` and `publish`. Capture the source ticket before
implementation/edits, binding both scoped states, selected SVG/HTML/legend files,
scientific selections and the layout policy. Supply it to `initialize --ticket
<ticket.json> --directory <existing-external-directory>`. The tool adds explicit
panel/shared ownership; it never classifies scientific roles from appearance.

For browser geometry bootstrap, serve only the external reviews directory on
localhost port 8766 and use `scripts/measure_main_paper_layout.cjs` with its
measurement input, bounds output and installed Chromium-family executable.
The browser helpers require Playwright available through Node's module lookup.
Run `arrange --directory <directory>`, then
`scripts/check_main_paper_layout.cjs <directory> <browser-executable>
<temporary-screenshot-directory>` to collect exact browser reports. `refine`
can use those measured painted/unclipped bounds to repack, followed by another
`arrange` and fresh browser collection. The generic `check_layout_frames.py`
remains the assessment authority. Inspect exact saved renderings at final scale.

The migration retains 183 mm width and translates explicitly owned content;
it does not resize plots or their typography. Rows retain their inherited
vertical relationships. Allocations touch, with a 1 mm required inset and a
0.05 mm rendering reserve per content edge; intrinsic whitespace is allowed.
Source data-axis clipPaths remain explicitly bound by ID, source SVG and
clip-definition hash. Incidental nested viewport cropping is removed, and the
legacy Figure 2 top-row canvas background is painted before subsequent rows.
Named checks cover Figure 1 D/E row geometry and F/G/H heatmaps, and Figure 2
A/B/C heatmaps and D/E plot-body geometry. Other existing scientific/presentation
coverage remains partial; this is not whole-figure scientific certification.

Only after passing assessments and visual review, run `publish --directory
<directory> --visual-review-note <actual-review-description>`. It authenticates
the pre-edit source ticket, preserves the previous selected state/HTML/SVG under
the explicit external task directory, checks every drawable's attributes/text,
publishes fresh registry/presentation/frame evidence and updates the same active
SVG/HTML pairs under their scoped state guards. Existing immutable scientific
selections and freezes remain unchanged. Old measurement reports remain
historical evidence; collector fixes require fresh reports for current artifacts.

Collection rendering QA is repeatable with
`scripts/verify_main_paper_collection.cjs <repository-root> <browser-executable>
<temporary-output-directory>` against that external-only local server. It checks
publication width, figure/legend placement, responsive overflow and page errors,
and writes temporary screenshots, standalone comparison renders, a browser JSON
report and a print PDF for page-by-page inspection. Keep the final hash-bound
verification record under `records/figure-reviews/`; remove temporary media after
review. A refreshed collection clears the previous verification reference.

### Typed panel relationships and related-panel spacing

Author correction, 2026-10-15: containment and matching axis sizes alone do not
establish a correct layout. Register relationships with explicit types:

- **Layout:** Figure 1 F/G/H share the displayed-trial Y coordinate and require
  equal gaps between heatmap data-axis rectangles. Figure 2 A/B/C require equal
  gaps between their two-heatmap data bodies; preserve equal spacing within each
  pair. Figure 2 D/E/F and G/H/I require equal outer panel-content gaps while
  retaining their different plot structures and natural sizes.
- **Scientific:** Figure 1 D/E/F share the example fish/trial data relationship:
  D/E common stored inputs and measured times, E's documented raw-frame
  normalization, and the exact selected F rows used in E's color bands. This
  relationship does not imply that D/E/F belong to one layout row or share a
  normalization. Retain authenticated frame/trial/source mappings.
- **Presentation style:** Register axis-title text artists separately from tick,
  condition and panel titles. Resolve the Nature profile's `axis.label` role
  for new axis-title edits, including colorbar quantity labels and Figure 2 H;
  query `--role axis.label` for the current value. Existing state-bound
  pre-Nature measurements remain historical evidence. Wording, units and scientific scales
  are unchanged. This does not certify all remaining typography or science.

Keep these in each current draft's existing semantic registry and
`consistency_requests`, with source identities, exact artifact hashes and fresh
measurements. Never infer shared controls, experiments, estimators or outcomes
from a layout group. Figure 2 H retains its different ratio outcome; C/F/I are
placeholders. Any future scientific group needs actual source mappings, not an
appearance-based identity. A freeze must resolve required element and scientific
relationship coverage within its requested scope.

SVG frame metadata `spacing_groups` declares stable group IDs, ordered target
IDs, `target: data_axes|content`, `axis: x|y` and `minimum_gap_mm`. The checker
compares consecutive edge-to-edge gaps at final scale within the active 0.1 mm
measurement tolerance, checking painted and unclipped content separately.
Composite data bodies use explicit `data_axes.element_ids` lists; the browser
measures their union. Equal allocation widths or equal panel-letter positions
are not substitutes for the declared spacing target.

The earlier `correct_panel_relationships.py` correction recipe is retained for
reproduction of its recorded pre-Nature candidates. For new selected Figure 1/2
presentation work, use the Nature migration route below with the same typed
relationships and exact browser measurements. Do not run the historical recipe
to restore retired typography. Recollect evidence after affected edits and
refresh the collection only after guarded publication.

### Physical axis geometry and related example labels

Author direction, 2026-10-15: axis typography and spacing must agree in absolute
publication units across horizontal/vertical axes, panels and figures. Use the
resolved Nature roles `axis.tick`, `axis.tick_label` and `axis.label` for
tick strokes, typography and physical inputs; do not use the unoverlaid
historical contract as the presentation target.
Compensate for every assembly transform; equal source attributes are not proof.
Measure 3 pt from the outward tick endpoint to the nearest tick-label geometric
glyph edge, and 3 pt from the outer tick-label envelope to the nearest axis-title
edge. These edge gaps are distinct from renderer anchor/baseline padding.
Keep each tick and label's scientific coordinate fixed along the axis. Shared
titles use the envelope of their explicitly registered labels. Retain intentional
tickless categorical/statistical axes and register the inapplicable tick-gap check.

Figure 1 D/E example block/trial labels use their profile-bound annotation roles,
with matching declared fonts/weights and a separate column outside the Y title.
Do not let row descriptions
occupy the title-to-tick-label strip. Reallocate/translate whole panels and canvas
space as needed, preserving data shapes, scientific clipping and related-panel
alignments/spacing. Record requirements in the existing draft's
`presentation_styles.axis_geometry`, example-label group and consistency request,
with exact saved-SVG measurements and explicit partial scientific coverage.
`style_axis_geometry.cjs` stages registered leaf changes and `--verify` measures
the saved candidate without editing it; `publish_axis_geometry.py` protects all
other scientific payloads and uses the guarded state publisher after passing
physical geometry, invisible-frame and visual checks. Refresh the collection
after publication. Recheck these groups after subsequent affected edits.

Author additions, 2026-10-15: every plot color/stimulus key has the canonical
`legend.key` role and a specific `plot.legend.condition-key` or
`plot.legend.stimulus-key` tag in the existing draft registry and SVG. Align
the whole key's right edge to its explicitly named data-axis/reference right
edge, within 0.01 mm at publication size; resolve typography through `legend.key`
in the Nature profile.
Direct schematic labels next to a stimulus are annotations, not color keys.
Keep sample/cohort counts in the caption rather than in key text or headings.
Use the displayed condition names `3sTrace` and `10sTrace` everywhere.

Distinguish `heading.condition` (canonical `title.condition`) from experiment
scope: center each D/E condition heading over its own data rectangle, with a
common baseline within the panel. The Delay experiment contains both Control
and Delay conditions; a mixed-condition trajectory must not carry an ambiguous
top "Delay" heading. Its key identifies the conditions; its caption identifies
the experiment and cohort counts. Keep E's own shared-scale Y spine and ticks;
its numerical labels and quantity title may remain shared with D, as heatmap
panels retain local ticks while sharing trial labels. Register the restored
artists and their authenticated common-scale mapping explicitly.

The supported annotation stage is `refine_paper_annotations.py`, followed by
`align_paper_annotations.cjs`; its `--verify` mode measures the saved candidate.
This stage may reposition the categorical statistical rows and the complete G
trajectory plot to keep full-size labels separate, preserving data paths,
values, text and markers. It does not recompute hypotheses or statistics.


### Nature presentation migration of selected main drafts

For an explicit migration request, `scripts/nature_main_figures.py` stages the
selected Figure 1/2 SVGs and uses the existing physical-axis, annotation and frame
checks, then `publish_state` for scoped guarded publication. It does not execute
historical plotting code or select cached panels. `prepare --directory <existing
external directory>` captures the states, sources, legends, registries, selections,
profile inputs and implementation hashes before artwork edits. A changed manual
source fails publication; implementation revisions are recorded explicitly.
Never reuse a source ticket to authorize changes in scientific inputs.

The supported sequence is: prepare; `style_axis_geometry.cjs`; then
`align_paper_annotations.cjs`. The scoped `refine_nature_layout.cjs` is a one-time
composition refinement of those candidates: Figure 1 fish/condition headings are
separated, and Figure 2 G is enlarged uniformly. Reapply the axis/annotation
helpers after refinement. `--center-labels` on the axis helper centers actual
glyphs and, for profile-bound inputs, title groups on their registered axis
extent. It does not change tick values or data coordinates.

Generate check pages using `publish_axis_geometry.py pages`. Collect exact
browser bounds with `check_main_paper_layout.cjs`; use `nature_main_figures.py
repack --exact --directory ...` to pack those measurements. A geometric packing
bootstrap (`measure_nature_packing.cjs`) is explicitly nonpublication evidence
and can overestimate unpainted content; it must never replace the exact collector.
The 2026-10-17 author-selected compact route uses `compact --directory ...`
once, preserving the preceding candidates and their presentation maps. It
uniformly reduces Figure 2 A/B/C and the empty I placeholder, compensates text,
and retains G's larger statistical inset. Rerun axis/annotation/stroke helpers
and exact bounds, then `repack --exact --width-mm 136 --directory ...`.
After the initial compact exact render, `pair-spacing --directory ...` transfers
1 mm into each Figure 2 Control/condition pair so the unchanged 6-pt endpoint
labels remain separate. It records the complete moved axis and heading groups;
its exact-derived packing estimate is provisional, never publication evidence.
Recenter shared X titles and collect fresh exact bounds after this step.
Changing the root width also changes its viewBox width at the same physical
unit scale; source protection rejects a global shrink of data/font units.
Repacking prepares both candidates before writing either and fails at the Nature
height limit. The per-owner resizes are uniform; all selected raster bytes,
data paths, timing, values and scientific texts remain protected.

`check_nature_presentation.cjs` applies profile-resolved physical stroke roles
and checks every mapped visible text and stroke; `--verify` is read-only. Named
renderer roles identify spines, traces, median/IQR, stimulus and boundary marks;
unclassified strokes retain their inherited hierarchy within the journal range
and remain explicitly unclassified scientifically. Physical stroke compensation
uses non-scaling strokes to avoid directional distortion from historical nested
axis transforms. Tick geometry remains with the existing axis checker. This is
not complete scientific registration or a production PDF-font/accessibility gate.

After the last change, regenerate pages and collect fresh exact painted/unclipped
frame reports, axis `--verify`, annotation `--verify` and Nature `--verify` reports.
Inspect both exact candidates at their native publication size and the dense G inset. Only then run
`nature_main_figures.py publish --directory ... --visual-review-note <actual
review>`. It authenticates all inputs, checks the source/candidate scientific
payload, saves prior artwork/captions/state and verified evidence, then updates
the same selected SVG/HTML pairs through scoped locks and atomic pointers.
Concise captions remain below 300 whitespace-delimited words; detailed previous
wording is retained in [figure-specific Methods](figure-specific-methods.md).
Refresh the authenticated collection last and inspect browser and print output.

The HTML renderer uses each SVG’s native millimetre dimensions for its inner
sheet inside an outer responsive transform. This uniformly scales text, markers and non-scaling SVG strokes on
narrow screens; printing restores the native publication scale. No scientific
freeze, missing-panel approval or full Nature/submission-readiness follows.

Dense G annotations use explicitly registered phase-header artists rather than
the empty historical X-axis container. Redundant D/M/R prefixes may be omitted
from significance glyphs only through recorded source/display aliases: every
star and its unchanged labeled hypothesis row are preserved and guarded. Shared
tick path definitions delegate width to their registered instances using
`stroke-width:inherit`; a definition’s specified width must not silently override
the measured instance width. Ordinary and non-scaling stroke measurement have
separate tested paths. Pure measurement/CLI imports do not load unrelated
scientific parquet inputs.

The 2026-10-16/17 migration was interrupted by disconnected storage before
publication; [resume evidence](../../../../records/figure-reviews/nature-main-migration-20261017.md)
records the exact status. Read selected state before treating any candidate as
current.
