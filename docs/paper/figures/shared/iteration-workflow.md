# Figure guide

This is the single maintained guide for figure display, saved-input previews,
scientific element mappings, Nature presentation, layout, publication and freeze.
[AGENTS.md](../../../../AGENTS.md) owns essential repository protections; the
[paper-figures skill](../../../../.agents/skills/paper-figures/SKILL.md) routes requests here.
The [unified contract](../../../../configs/paper-figures/figure-elements.json) supplies
machine-readable roles, current style values, rule origins and frame policy.
Scoped scientific selections and presentation states remain separate authorities.
Read only the task route and relevant sections; neither this guide nor contract
resolution selects science, edits artwork or creates a freeze.

For new heatmaps, apply the standing [robust baseline normalization decision](../../decisions.md#2026-10-31--robust-baseline-normalization-for-all-heatmaps)
and its recorded aggregation order. This scientific method change does not
rewrite selected/frozen artwork or turn a display request into recomputation.

## Task routes

For selected artwork, read **Two separate current states** and resolve only the
requested scope's state, scientific selections, legend and consumed bindings.
For a disposable preview from explicit saved inputs, start with **Lightweight work**.
Choose the smallest route that covers the request; expand it when the change
actually affects another relationship. The checks below preserve source
identity and publication safeguards without turning each edit into a freeze.

| Request | Inputs and sections | Required checks and outcome |
|---|---|---|
| Display a figure/panel | Selected SVG/legend; **Display and working formats** | Authenticate saved bytes and inspect the exact rendered scope. No restyling or state selection. Plural main-paper display uses **Main-paper collection** and its guarded build. |
| Compare retained versions | Requested bundles; **Ordered version browsing** and the [review library](../review-library.md) | Preserve each legend/source binding; browsing does not change selection. |
| Disposable preview | Explicit saved inputs; **Lightweight work** | Verify hashes, units/interpretation and provisional status; retain the complete recovery bundle. No state bootstrap, full registry, journal audit or global navigation refresh. |
| Legend-only edit | Selected artwork, candidate legend and starting ticket; **Operational commands and concurrency**, **Linked panels and legends** | Check affected definitions and rendered legend; preserve SVG bytes. A scientific interpretation change uses the scientific route. |
| Presentation edit | Selected bundle, starting ticket and declared candidate changes; **Operational commands and concurrency**, **Verification before accepting a candidate** | Check the change boundary, scientific payload and affected final-scale properties; inspect the saved candidate. SVG publication requires fresh bound frame evidence; legend-only publication does not. |
| Consistency or layout | Edit inputs plus named roles/relationships; **Semantic registry and linked-panel checks**, relevant frame/axis sections | Update scoped registry/evidence, measure affected relationships and report partial coverage. Whole-layout work also uses **Composition review**. |
| Scientific work | Claim/estimand, selected definitions and affected input chain; [governance](../../../project-maintenance/governance.md), [decisions](../../decisions.md) and [methods](../../methods/README.md) | Resolve relevant open decisions before execution; use scientific validation and reconcile methods/legend. Styling permission does not select a method. |
| Explicit freeze | Exact approved candidate/bundle; **What happens when the author freezes** and its specification | Complete mappings, genuine reviews and check-only gate, then guarded publication of the complete required panel/assembly/letter set. Ordinary edits/exports never invoke this route. |

Use headings for bounded reads. Read **Requirements, choices and precedence** for new styling or requested
journal verification, and the frame sections for layout work. Source authentication,
manual-edit protection and applicable scientific/render checks remain required.
An unavailable source or publication conflict stops dependent publication; continue
independent code/doc work and report the concrete limitation. Ask only about
unresolved scientific choices, conflicting author directions or proposed changes
to standing design preferences. Existing scoped approvals remain in force.
Plans own future work; scoped state and exact bindings own current presentation.

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

Follow the scoped state's `default_display` and `default_display_format`.
Figure 1's author-selected single-draft policy maintains one active SVG and
matching review; update that pair through guarded presentation publication.
Preserved freeze candidates, historical HTML and scientific freezes remain
separate. An explicit frozen view never resets the draft or implies a whole-figure
freeze. Exact versions and approved dispositions resolve through state and
[Figure 1 guidance](../figure-01/README.md), not a procedural handoff summary.

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

### Ordered version browsing

Author direction, 2026-10-27: the SSD review library presents Figures 1–4 in
numerical order, with one browsing page per figure. Panel letters and version
labels are ordered naturally within their recorded families. Buttons display one
saved version at a time with its associated legend/context and original sources.
Keep joint work intact; isolate panels only through explicit source-bound
ownership mappings. The supported review-library builder/checker owns discovery,
coverage and guarded publication. Include embedded HTML/archive versions and
record aliases, supporting illustrations, historical gaps and unresolved items.
This expanded browser is requested discovery/comparison; it does not replace
minimal scoped display, select a version, restyle artwork or create a freeze.

## Lightweight work

Choose the smallest applicable route. Display authenticates the requested saved
view; it does not restyle, rebuild or certify it against a newer contract.
`figure_working_copy.py display-target --state <state.json>` supports nested
working states and flat exploratory states. `verify` retains comprehensive
verification. Inspect the actual saved browser rendering before claiming an open.

Disposable exploration requires explicit saved inputs, units/interpretation,
a matching legend and provisional status. Use the shared `write_preview` helper
in `classical_conditioning.figures.preview`, or the assembler's
`--preview --output <fresh-task-directory> --legend <legend.md>` route.
Preview output stays outside the repository and cannot overwrite an iteration.
It requires no new paper state, publication ticket, complete role registry,
journal-compliance audit, analysis rerun or global navigation update. The shared
theme remains the styling entry point when plotting new numerical artwork.

Figure 2G supports `render --preview --inputs-dir <saved-ticket-directory>
--output-dir <fresh-task-directory>` without paper-state checks or numerical
refitting. The Figure 3 atlas supports `render --preview --inputs-dir
<saved-atlas-directory> --fish <ID> --output-dir <fresh-task-directory>`; repeat
`--fish` to render selected cards only. Saved input hashes are still checked.

The helper stores native SVG, minimal self-contained HTML, input/output hashes,
renderer identity and source snapshots. Its compact registry sidecar is required
for scientific editing/recovery, but visual display remains self-contained.
`read_registry` accepts both inline legacy and compact sidecar representations.
Historical reproduction remains embedded and unchanged. Do not retrofit selected
artwork. Do not subset fonts needed for future text edits.

When continuing an iteration, carry forward its accepted edits and all referenced
resources, including a registry sidecar referenced by the SVG metadata. Verify
the resulting bundle can resolve those resources from its own directory; copying
the SVG alone is insufficient. Reuse the current review tab where practical and
authenticate the served bytes after replacement, accounting for browser caching.
On requested retention or freeze, preserve the final bundle and required recovery
inputs on JOAQUIM. Keep recovery/intermediate inputs identified as dependencies,
separate from the author-selected versions in navigation.

Retain only versions requested for saving, comparison or promotion:
`figure_working_copy.py retain-preview --manifest <preview.json>
--output-dir <existing-SSD-version-directory> --scope <scope>`.
Retention preserves the bundle without selecting or freezing it. Finish retention
by registering all chosen bundles in one browser update:
`review_library.py build --changed <bundle-directory> --scope figure-02`.
Repeat `--changed` and `--scope` for multiple bundles/figures. This incremental
route preserves existing inventory and records partial coverage. Changed sources
require exact authenticated historical recovery; absent that recovery, reconcile
and use an explicit full build; a missing full publication never triggers an implicit drive-wide scan.
Display and scratch previews do not refresh navigation. Keep task previews while
needed; do not delete user material as housekeeping.

For managed presentation publication, follow **Operational commands and concurrency**.
Retention is distinct from selecting a managed draft or completing a formal freeze.

## Operational commands and concurrency

The author requested consolidation of the former Mac source cache into the SSD
on 2026-10-28. See [Artwork recovery guidance](offline-artwork.md) for its retained
versions and exact recovery mapping. Dated source snapshots do not replace either
figure's active presentation pointer or mirror raw analysis dependencies. A future
offline copy must bind its source/state hashes; publishing a derivative still
requires the scoped guards.

For ordinary edits, use selected artwork and the presentation-only branch of
[the review builder](../../../../scripts/paper/shared/build_legend_review.py).
Maintain the scope's one active SVG/HTML pair rather than competing current files.
Capture a ticket before changing code or candidate inputs:

```sh
.venv/bin/python scripts/figure_working_copy.py begin --state <state.json> --ticket <fresh-ticket.json>
.venv/bin/python scripts/paper/shared/build_legend_review.py --state <state.json> --baseline <fresh-ticket.json> --candidate-legend <candidate-legend.md> --update-current
```

For an SVG edit, supply `--candidate-svg <candidate.svg>`,
`--presentation-changes <changes.json>`, `--candidate-layout-evidence <evidence.json>`
and `--visual-review-note <actual-review>`; `--candidate-legend` is optional then.
The change list names stable `element_id` values and permitted `attributes` or
`text: true`. The generic route permits typography and text positioning, including
transforms on text-only groups. It rejects coordinate edits to data graphics and
their ancestors. Use the existing scoped mapping-aware assembly/layout tools for
those edits; preserve values, axis mappings, clips and associated scientific marks.
Changing scientific units, definitions or interpretation requires the scientific
route even when expressed as text. Data paths/resources cannot be edited through
the generic route. It preserves previous selected bytes, binds measurements to
the SVG's native physical width and publishes state last without reconstruction.

### Bootstrap and historical reconstruction

Only a missing working state calls for the supported working-copy bootstrap,
using an existing external destination:

```sh
.venv/bin/python scripts/figure_working_copy.py prepare --output-dir '<existing-external-figure-01-assembly-current-directory>' --revision <revision>
.venv/bin/python scripts/figure_working_copy.py verify
```

`prepare` reuses verified copies and refuses changed/missing inputs. Under Figure
1's single-draft policy it rejects `--refresh`; other supported bootstrap states
may preserve versioned copies. It does not restyle or recompute science.
Working-copy `verify` is read-only; similarly named commands elsewhere may
recompute statistics, so consult their documented behavior before execution.

The review builder without candidate SVG/legend arguments enters **Figure 1
numerical reconstruction**, requiring `--output`. That branch rejects other scoped
states before importing numerical dependencies. It is a reproduction/scientific
operation, not the command for an ordinary display, legend or presentation edit.
Keep historical reconstruction commands and dependencies available separately.

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

Choose tests and measurements for the affected behavior and relationships, then
inspect the exact saved rendering at its native physical assembly size. A local
label edit does not require redesigning other panels or completing a freeze
registry. Fresh frame evidence remains required for managed SVG publication.
Measure current candidates at their declared native width, cross-checking frame
dimensions; the low-level historical measurement API retains its explicit/default
width for reproduction only. Record only reviews actually performed. An SVG
editing handle is not a scientific classification; legacy elements without
authenticated mappings stay unresolved for any new freeze.

## Semantic registry and linked-panel checks

### Scientific element identities

Map stable element IDs to quantities, units, source/data hashes, transformations,
timing, eligibility, normalization, coordinate systems and relationships from
authenticated inputs. Structural XML IDs, colors and appearance do not establish
scientific identity. Keep original source-panel identifiers when display order changes.
`trace.scaled_vigor_series` supports framewise vigor with explicit baseline/scaling
context; `heatmap.vigor` supports direct SVG rectangles or explicitly mapped cell
groups. These representations do not change scientific definitions.
Historical [1.1.1 specification](../../../analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md)
and frozen contract copies remain immutable reproduction dependencies.

### Where consistency is established

- **Scientific definitions:** scoped accepted selections linked by
  [Current figures](../README.md), with shared definitions under
  [paper methods](../../methods/README.md). Each semantic registry maps stable
  element IDs to scientific roles, inputs, units, transforms and relationships.
  Matching appearance does not establish matching science.
- **Shared presentation contract:**
  [figure-elements.json](../../../../configs/paper-figures/figure-elements.json)
  owns reusable role IDs, compatible representations and historical defaults.
  New presentation values come from the resolved [current presentation values](#coding-entry-point-and-exact-values),
  preserving scientific definitions rather than inheriting historical estimator
  choices from plot-family fields. The
  [scientific element identities](#scientific-element-identities)
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

Read coverage from the exact scoped state and version-bound evidence. Legacy
or partial mappings remain unresolved for a new freeze; historical style audits
do not certify a later artifact or whole-figure consistency.

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

When a selected scientific draft changes, reconcile its current methods reference,
caption and discovery descriptions with that exact scoped state. Preserve dated
and hash-bound Methods as historical evidence; label their scope and link to the
current reference rather than rewriting their original definitions. Current
Figure 2 uses [its working-draft methods](../figure-02/current-methods.md).

For a statistical audit, bind actual saved inputs, formulas, coefficients,
covariance, diagnostics, correction families and rendered annotation sources.
Current command defaults do not establish historical run settings. The
[G/H/I audit route](../../../../scripts/paper/figure_02/panels_g_h_i/audit_saved_statistics.py)
reconciles saved arithmetic and selected SVG labels without refitting or changing
science. Keep detailed Markdown/HTML discussion on JOAQUIM and compact evidence
links locally. Missing source/runtime bindings and unverified renderer coverage
remain explicit; successful arithmetic does not approve a model or freeze it.

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

## Nature target and migration status

New presentation work uses `nature-main` from the unified contract through the
shared theme. Existing artwork remains selected until scoped guarded publication.
Read actual coverage from state-bound evidence; implementation progress belongs
in the existing [plan index](../../../../Plans/IMPLEMENTATION_STEP_INDEX.md).
Historical compatibility styles are reproduction dependencies only.

## Requirements, choices and precedence

Bind each rule to its origin: journal requirement/recommendation, standing
author rule, project default or scoped exception. The
[Nature specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
cover visible axes/ticks, units in parentheses, editable type, 5–7 pt ordinary
text and 8 pt bold upright lowercase panel letters. The
[production guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
sets 89/183 mm widths and a 170 mm artwork-height limit.
Nature's [final-submission page](https://www.nature.com/nature/for-authors/final-submission)
also allows 120–136 mm widths where necessary and asks authors to identify
essential panel alignments; its final stroke range is 0.25–1 pt.
Choose the smallest appropriate width for content and legibility; 183 mm is
an available double-column size, not a requirement for every figure. Never
spread uniformly reduced panels apart just to fill that width. On 2026-10-17
the author selected compact 136-mm compositions where feasible for the current
Figure 1/2 revision. Record each native width in its SVG, frames and state;
retain physical text/tick/stroke sizes when changing the canvas. Intermediate
widths remain an explicit composition choice, not a claim of editorial approval.

Specific editorial instructions override generic web guidance for that
submission. Keep main/Extended Data and review/production requirements distinct.
Source conflicts are recorded in the source review. No promise of automatic
compliance follows from choosing a font or citing this page.

For substantial design work, use the workflow's
[composition review](iteration-workflow.md#composition-review) to assess purpose,
panel proportions, reading order and final-size readability together. Journal
ranges constrain the design; they do not choose the best composition. A local
edit checks affected properties and relationships. Preserve author-selected
spacing/frames; present a concrete alternative for choice if a preference causes
a demonstrated problem. Do not add numerical aesthetic thresholds that the
profile or an approved scoped relationship does not establish.

## Coding entry point and exact values

The [unified contract](../../../../configs/paper-figures/figure-elements.json)
owns exact resolved styles, semantic roles, rule origins and embedded frame policy.
The `nature-main` profile name is a supported view of that single contract; it
does not overlay another active configuration. Use this small lookup rather than re-reading plans,
copying numbers into a renderer or introducing another theme/exporter:

```bash
.venv/bin/python -m classical_conditioning.figures.presentation_profiles
.venv/bin/python -m classical_conditioning.figures.presentation_profiles --role axis.tick
```

Other sections are `typography`, `colors`, `roles`, `rules` and `theme`.
Use `--section styles` for all resolved style roles, or `--role <canonical-id>`
for one existing element role and its applicable axis-geometry rules.
Output includes the contract input hash and explicitly requires render evidence.
Historical evidence retains its original configuration and implementation hashes;
it is not rewritten to certify the unified contract. Saved display remains
authenticated by scoped state. New publication requires fresh relevant evidence
bound to the unified contract, even when its numerical policy values are unchanged.
For new or author-requested presentation candidates, opt in through the existing
shared API; legacy calls keep their original defaults:

```python
import matplotlib as mpl

from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme, style_axes

profile = load_presentation_profile("nature-main")
with mpl.rc_context():
    theme = apply_theme(profile=profile)
    # Construct, style and export the figure inside this context.
    # style_axes(ax, theme=theme, xlabel="Time (s)", ylabel="Activity")
    # Non-theme artists use identified role values from profile.contract["styles"].
```

Load once per build and reuse the same snapshot for role values and evidence.
`get_theme(profile=profile)` also accepts that resolved object without re-reading
files; `apply_theme(profile="nature-main")` is the shorter option when a build
needs only the theme. Passing both `theme` and `profile` is rejected. No global
profile cache can hide an edited config; a later build resolves fresh bytes.

The selected Arial face must be available; a paper-wide alternative requires
an explicit recorded profile change, not machine-dependent fallback. SVG/PDF
embedding and glyph checks remain separate. Theme adaptation handles shared
Matplotlib parameters; it does not restyle every existing artist or convert a
historical freeze contract into a newly approved candidate. Bind profile inputs
and code hashes to the existing evidence before guarded publication.
The resolved view retains historical plot-family fields for compatibility;
these do not select current estimators, cohorts or scientific axes. Resolve
those through scoped selections, including the ratio/log-difference distinction.

The contract separates journal ranges from our exact role sizes, stroke hierarchy,
black text, tick geometry, caption-review styling and author-selected spacing.
The cited Nature guidance does not prescribe universal tick direction/count,
glyph gaps, marker diameter or right-key alignment. The active frame inset is
read from the contract’s `layout_frames.policy` section, not copied into another competing policy.
For routine edits, use the recorded version and query the relevant section.
Recheck official guidance when the target journal/stage changes or before actual
submission; a local role change does not require repeating the full research.

## Axes, ticks and labels

Apply these rules to ordinary X/Y axes and the applicable components of
colorbars, shared axes, secondary/twin axes and inset axes. Schematic coordinates
are identified separately. Register affected components and visibility decisions
in the scoped draft evidence; complete visible-element coverage belongs to the
explicit freeze route. Report the actual coverage of a requested journal review.

- **Meaning:** record quantity, units or dimensionless status, coordinate
  transform/scale, limits, inversion, category order and source values. Preserve
  numerical definitions: log differences and ratios are distinct. Styling does
  not authorize starting every axis at zero, taking another logarithm, changing
  a reference from one to zero, or changing limits to match another panel.
- **Sides:** ordinary plots use left/bottom spines at the data-rectangle edge;
  top/right are off unless a registered family needs them, such as boxed
  heatmaps. Default spine outward offset is zero. Keep local spines/ticks when
  numerical labels or quantity titles are shared, including Figure 2 E.
- **Major/minor ticks:** retain authenticated tick values and categories.
  Minor ticks are off by default; logarithmic/other exceptions specify values,
  lengths and widths explicitly. Do not generate dense minor ticks merely by
  switching a renderer style. Endpoint ticks and marks must remain unclipped.
- **Positions and alignment:** X tick labels are horizontal and centered on
  their ticks; left Y labels have aligned right edges and centers aligned to
  their ticks. Other sides/rotations need an explicit role-specific rule.
  Center axis titles on the relevant data-axis extent; quantity titles on Y
  are vertical. Condition headings center on their own data rectangles and
  share a baseline within their named group. Multiline titles need line-spacing
  and visual-center checks, not only matching text anchors.
- **Gaps:** measure along the appropriate outward axis normal: painted tick
  end to nearest label glyph edge, then the outer tick-label envelope to the
  nearest axis-title glyph edge. Include minus signs, exponents, scientific
  offset text and multiline labels in the envelope. Allocate exponent/offset
  text explicitly so that it neither collides with a title nor detaches from
  its axis. A Matplotlib `pad=3` or `labelpad=3` is only an input; it does not
  establish the required physical gap.
- **Number/text formatting:** use sentence case without a terminal full stop,
  quantities with units in parentheses, SI number–unit spacing in numeric
  expressions, a readable minus sign, and consistent precision for a quantity.
  Do not add unsupported digits or invent a unit for a dimensionless measure.
  Preserve exact categorical names `3sTrace`/`10sTrace`; durations remain
  `3 s`/`10 s`. Define phase abbreviations and unusual units in the caption.
- **Sharing:** name members, property shared, reference axis/label and reason
  for each suppressed component. Distinguish sharing coordinates, labels,
  normalization and cohorts. Equal colors/ranges do not prove equivalent
  estimators. Colorbars carry quantity, units, ticks, limits, normalization,
  missing/out-of-range meaning and explicit mappable ownership.
- **Evidence:** measure the exact saved SVG through all transforms, with
  unclipped painted bounds and resolved fonts at publication scale. Use the
  applicable bound measurement tolerance; do not treat tolerance as a new
  target gap. Recheck after embedding/export. A nominal style dictionary,
  screenshot or partial axis inventory cannot establish full compliance.

These retained defaults must be reconciled with each existing scientific plot
family; no source selection, tick value or scientific exception is inferred.

`style_axes` styles ordinary axes; pass `show_yticklabels=False` while retaining
`show_yticks=True` to keep local ticks/spines with shared numerical labels.
The equivalent X flags are independent too. Quantity-title sharing remains an
explicit group/caller decision. Existing profile-styled axes receive the declared
font, colors, spine widths and tick caps as well as sizes. The stimulus-window
helper uses the CS/US role widths and layer order without changing event timing.

## Strokes, colors, fonts and physical sizes

Check effective final stroke widths, including marker edges, colorbar outlines,
arrow stems, schematics and statistical lines through nested transforms. Resolve
line caps/joins, physical dash arrays and offsets, marker/arrowhead sizes and
error-bar cap dimensions by role; the profile names measurements still needed
in the pilot. Fill-only bands/cells need no artificial border. An alpha of 0.2
is opacity, not a 0.2-pt stroke. Keep meaningful translucent layers distinct and
inspect their composite against the actual background.

Use the bound CS/US, condition and colormap tokens; a palette example is not an
instruction to recolor selected scientific data. Inspect color-vision confusion,
contrast, print/grayscale legibility where relevant, zero/missing/saturated colors
and uncertainty-band visibility. Distinguish recommendations from hard numerical
checks: there is no universal Nature data-line contrast ratio, marker diameter,
opacity or dash spacing. Recheck the same semantic role across panels/figures.

Resolve normal/bold faces, Symbol/Greek and mathematical glyphs, minus signs,
superscripts/subscripts and multiline leading explicitly. Text stays editable;
font outlines or visually plausible fallback are insufficient. Check actual PDF
font types and resource embedding, not only `pdf.fonttype=42`/`svg.fonttype=none`.
Rasterized axis labels cannot pass an editable-text requirement by nominal size.

Artwork dimensions include all painted letters, ticks, keys, colorbars and
strokes; captions stay outside. Preserve natural image/data ratios and compensate
fonts/strokes after uniform resizing. Use alphabetical, logically related panels
and efficient whitespace without abandoning registered physical frames.

## Related content and delivery

Keep typed layout, presentation-style and scientific relationships in the
existing registry. Retain right-aligned `legend.key` groups with represented
series and a named right edge. Keep experiment headings distinct from condition
headings; move cohort counts and explanatory clutter to captions. Preserve
related-panel alignments and data-body spacing, natural aspect ratios and all
authentic raster content. Frames alone do not certify good use of whitespace.

Inspect color accessibility, label collisions and scientific data visibility.
Avoid decorative grids, hatching and shadows; meaningful guides/dashes remain
identified. Define every error bar/band, sample unit, relevant n, comparison and
statistical symbol using source evidence. Detailed methods remain recoverable
when captions are shortened. Never invent missing results to fill a panel.

Keep editable native SVG and production vector output with embedded resources,
fonts and RGB content. Inspect PDF font types/editability separately from browser
appearance; do not outline or flatten text. Native photographic resolution,
scale bars, image integrity and source-data checks apply by asset type, not by
blindly upsampling discrete heatmaps. Record any asset/source conflict.

The [formatting guide](https://www.nature.com/nature/for-authors/formatting-guide)
distinguishes final captions from artwork and Extended Data from Supplementary
Information. Use separate delivery rules for those scopes. Include concise
production instructions for essential named alignments in the delivery evidence,
without adding visible review notes to the SVG/legend collection.

## Conditional content and policy coverage

The maintained source review covers the figure guide's specifications,
production, integrity and Extended Data sections plus Nature's formatting and
final-submission guidance. The following rules apply only when that content is
introduced; they are not excuses to apply a main-artwork profile to other media:

- Sequences use the guide's monospaced one-letter format; chemical structures
  require its dedicated structure guide/template, not guessed plot styling.
- Gels/blots require original uncropped evidence; document composites/splices,
  mark boundaries and intentional duplication, and apply adjustments equally
  to controls. Preserve authenticated sources for all other image types too.
- Extended Data has a separate delivery target: 180 × 170 mm, one figure per
  page, JPEG/TIFF/EPS, RGB and at most 10 MB. Its raster-resolution advice differs
  from main photographs. Keep editable masters; flatten only the requested
  delivery derivative. An exceptionally long legend can occupy a second page.
- Extended Data tables need their own formatting: 7-pt type, restrained rules,
  legible footnotes and single-page composition. They are not ordinary plot axes.

The [Extended Data guide](https://research-figure-guide.nature.com/figures/extended-data-formatting-guidelines/)
and the dated source review retain conflicts and exact stage-specific details.
No complete ED exporter/profile is implemented by the main-theme adapter.

The [figure integrity page](https://research-figure-guide.nature.com/figures/image-integrity/)
has broad wording against generative imagery; the current
[AI policy](https://www.nature.com/nature-portfolio/editorial-policies/ai)
uses a risk/verification framework. Record this evolving policy distinction.
Coding assistance must preserve inspectable source, reproducibility, human
scientific decisions and required disclosure. It is not authority to invent
scientific imagery, alter authenticated raster content or supply missing results.

## Invisible layout frames (author direction 2026-10-15)

This ordinary-draft layout requirement applies to future main, supplementary
and analysis figures. It does not change scientific definitions or historical
freezes. The author selected **1.0 mm** (tighter) on 2026-10-15 after the
calibration review. The active numerical authority and author evidence are in
the unified contract’s `layout_frames.policy` section; the former 1.5 mm recommendation
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

### Design frames and letters first

Author direction, 2026-10-30: when designing a new figure or a new figure
layout, start at the intended final physical size. Allocate the figure, panel
and shared-element frames first, using placeholders for panel contents. Place
the panel letters in their owning frames before adding the actual panel artwork.
Use consistent letter placement relative to the corresponding frame edges,
and align baselines across panels in the same row. Follow the selected inset,
outside-margin and inter-panel rules above. Placeholders and any visible frame
outlines are layout aids only; remove them from the finished artwork, whose
frames remain invisible. A display-only request does not create a new layout.

Then insert authenticated panel artwork into its allocated frame, preserving
scientific coordinates, aspect ratios, typography and ownership. Design shared
elements such as colour bars with explicit allocations and mappings; use one
bar for multiple panels only when their quantities, normalization, palette,
limits and interpretation match. Revise the scaffold if the actual content
needs more space rather than stretching or cropping scientific artwork.

Review the composition as a whole at the selected Nature physical dimensions.
Use named alignment targets for the requested relationships: row content tops
and bottoms, common left/right content edges, comparable letter-to-content
offsets, and the true data-area widths/heights. Heterogeneous panels can have
different internal whitespace; equal outer boxes do not establish these content
relationships. Rebalance schematic sizes, card widths, allocations and shared
elements together when needed, rather than moving letters alone. With an
explicit request to widen a plot, adjust its display-coordinate mapping and
associated axes, ticks, guides, clipping and colour bands consistently; preserve
the underlying values/estimand and physical text/stroke sizes. Uniformly enlarge
schematic artwork when its natural proportions need to be retained.

For an existing figure, retain its selected artwork and use its recorded frames
as the starting scaffold when a scoped layout edit or freeze is requested; do
not rebuild or retrofit it merely to satisfy the order of operations. Before
freezing a whole figure, measure letter anchors relative to the final frame
edges and row baselines, inspect rendered glyph bounds and clipping, and check
the actual data axes, content and shared elements against their allocations.
Record any intended exception and its approval; a passing containment check
alone does not establish letter alignment. Bind the measurements and visual
review to the exact candidate SVG. If a required relationship cannot be
measured or justified, leave it unresolved rather than asserting a pass.

Use rendered painted/glyph bounds for claims about visible content alignment.
A nested SVG viewport or `getBoundingClientRect()` of a container can include
blank space and is not itself proof of a visible top/bottom edge. For plot-size
claims, measure the actual data-axis rectangles or endpoints, excluding Y titles,
tick labels and trial descriptions. Explicit layout guides may support frame
measurement only when their agreement with the actual plotted axes is verified;
measuring an intended guide alone does not verify the plot. Recheck affected
relationships after resizing and record what was actually measured, its final
physical units, scope and tolerance. A general frame-containment pass does not
certify every editorial alignment.

Include panel letters, axis labels/ticks, titles, annotations, strokes, markers
and images in explicit owning panel groups. Shared colourbars and annotations
have their own figure-level `shared` allocation. Identify background surfaces by
explicit IDs; never discard white marks by appearance. Align actual data-axis
rectangles as well as named content edges. Natural aspect ratios and internal
whitespace are allowed: do not stretch science, crop marks, change scientific
coordinates or reduce established font sizes just to fill the available space.

### Composition review

For a new assembly or substantial layout revision, establish the figure's
scientific purpose and reading order from its owning guidance before arranging
panels. Keep the intended panel set unless a scope change is requested. At the
selected final physical size, use the frame-first scaffold above, then review:

- Panel proportions and density against their scientific purpose; allocate more
  room to dense evidence while preserving natural image/schematic proportions.
- Actual content/data-axis edges, letter offsets, shared elements and intentional
  grouping whitespace. Equal frames alone do not establish visual balance.
- Readability of the smallest ticks, labels and statistical marks; the hierarchy
  should guide the reader without obscuring data, uncertainty or event timing.
- Color accessibility, meaningful missing/zero/saturated values, and the legend's
  explanation of sample units, uncertainty and statistical comparisons.

Inspect the whole saved composition and its densest areas at native publication
size; responsive browser enlargement is not evidence of print legibility.
After revision, recheck affected relationships and the resulting composition.
For a small local edit, inspect its context and dependent relationships instead
of repeating this full review. Keep journal requirements, recommendations and
author choices distinct using the maintained [presentation guidance](#requirements-choices-and-precedence).
If a standing preference creates a demonstrated design problem, show the concrete
tradeoff and alternative for the author's choice; do not silently change defaults.

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
in `records/figure-reviews/shared/layout-frame-implementation.json` lists those entrypoints.
Future work must use the supported adapters or report incomplete coverage; a
missing frame mapping/evidence cannot pass a new freeze. No current draft was
converted as part of introducing these tools.

### Browser evidence and publication

Run `.venv/bin/python scripts/paper/shared/verify_layout_frames.py <exact.svg> --review
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

Calibration uses `scripts/paper/shared/calibrate_layout_frames.py --output <new-external.html>`
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

Resolve the permanent HTML from the collection's `html.path` through
`display-target`; do not reconstruct it from an older directory convention.
For storage/portability verification, authenticate the state-selected saved
wrapper with `display-target`; do not rebuild SSD artwork to make a check pass.
The final recovered wrapper is bound by its completion receipt and recovery
evidence in the collection state.

The builder snapshots selected states/artifacts, scientific-selection records,
legends, frame evidence, policy and implementation; rechecks under scoped locks;
and updates the collection pointer last with rollback on failure. It refuses
unrecorded output collisions, manual hash changes, missing storage and stale
frame evidence. An unchanged authenticated build is reused. It never edits
individual figure state or recomputes science.

### Current assemblies: explicitly authorized layout migration

This is a historical recipe, not a standing instruction to migrate current work.
The exact pre-consolidation guide is preserved through the
[consolidation receipt](../../../../records/figure-reviews/shared/figure-guide-consolidation-20261031.json).
Use the [tool catalogue](../../../../scripts/README.md) for supported command usage;
resolve selected state and capture pre-edit tickets before any scoped migration.

### Typed panel relationships and related-panel spacing

Record layout, scientific and presentation-style relationships separately in the
scoped registry and `consistency_requests`. Named scientific relationships require
source/data mappings; a layout group does not imply shared cohorts or estimators.
Figure-specific relationships live with [Figure 1 assembly](../figure-01/assembly.md)
and [Figure 2 assembly](../figure-02/assembly.md). Register spacing targets as data
axes or painted content, measure actual gaps at final size and recheck after edits.

### Physical axis geometry and related example labels

Use profile roles and rendered glyph bounds for absolute physical gaps, preserving
tick coordinates. Register intentional tickless axes and inapplicable gap checks.
Keys carry canonical `legend.key` and specific condition/stimulus-key tags; align
their right edge to a named reference within 0.01 mm. Direct schematic labels are
annotations. Keep cohort counts in captions and preserve `3sTrace`/`10sTrace` names.
Figure-specific example-label/heading relationships belong in the assembly guides.

### Nature presentation migration of selected main drafts

Use supported staging, exact browser measurement and guarded publication tools
from the [catalogue](../../../../scripts/README.md). Historical compact/repacking
sequences and one-time corrections remain in the archived pre-consolidation guide;
do not run them merely to resume a selected draft. Verify tools are read-only by
default; `--apply` and `--center-labels` are explicit edits and conflict with `--verify`.
Existing raster data, scientific text, statistics, source mappings and coordinate
meaning remain protected; no migration or production-compliance claim is inferred.

## What happens when the author freezes

### Freeze readiness and completion

Once an explicit freeze is requested, inspect the scoped registry, source
bindings and validator compatibility early, alongside the layout work. A valid
scientific registry for a panel or an older style profile does not establish
whole-assembly coverage under the current Nature profile. Classify failures by
cause: incomplete source-grounded mappings/reviews, actual artwork defects,
validator limitations, or author decisions. Complete mappings and reviews from
the authenticated renderer/data evidence where available. Diagnose and repair
applicable tooling incompatibilities with focused validation; do not change
scientific definitions, weaken the gate, invent evidence or approve exceptions
merely to obtain a pass. Recheck after relevant fixes rather than repeatedly
reporting the same known failure count as the outcome of a freeze request.

The final freeze request binds the exact author-approved assembly and legend.
Preserve those bytes before further work. Revisions that change visible artwork
need a new review; changes confined to mappings, evidence or supported validation
must still bind the same artifact hashes. Author approval or a preserved
presentation checkpoint is not a completed formal freeze. If completion remains
blocked by unavailable evidence or unresolved validation, report the concrete
remaining work and the actual saved/selected status without claiming a freeze.
Record any authorized presentation selection separately; it does not discharge
the outstanding freeze request.

### Candidate verification and publication

Author direction, 2026-10-30: a request to freeze a whole figure assembly also
requests a freeze of each constituent panel as it appears in that assembly,
including the panel letters and their positions. This applies to the pending
Figure 1 whole-assembly request. Treat the assembly and its panels as one
version-bound freeze set:

- Bind each panel's content, scientific definitions/inputs, scoped legend,
  final physical size, frame allocation, placement and ancestor transforms to
  the exact approved assembly hash. Preserve original source-panel identifiers
  when displayed lettering differs. Freeze each panel's as-assembled version;
  an older panel freeze alone does not certify later content or presentation.
- Bind every panel letter's identity, text/style, owning panel, final canvas
  position and offsets from its frame and named plot/schematic reference edges.
  Record anchors and rendered glyph bounds distinctly in physical units. These
  positions are frozen properties, not merely a passing alignment observation.
- Prepare and validate the panel candidates and the whole-assembly candidate
  through the explicit gate. Use source-bound panel isolation at the assembly's
  final scale, with its required resources and coordinate mappings; do not
  redraw science or reflow a standalone panel during extraction. Keep shared
  colour bars/keys under their assembly ownership and link them from the panels
  they explain. Reference joint scientific records and dependencies rather
  than duplicating them as purported independent scientific analyses.
- Preserve prior freezes and publish all validated panel/assembly manifests
  before committing the complete freeze-set selection under the publication
  guards. Record panel-manifest references in the assembly record and each
  panel's parent-assembly identity. Report a whole-figure freeze complete only
  when both the assembly and every required panel/letter binding have passed.
  A failed panel check leaves the complete set pending; partial success must
  be identified as partial. Later changes to content, size, placement or letter
  geometry require an explicitly revised version and its affected checks.

1. Resolve the exact selected candidate, panel scope, scientific definitions and
   inputs. Preserve previous frozen bytes/hashes. A scientific change requires a
   selected scientific source, not a presentation exception.
2. Build/verify the semantic element registry from named renderer inputs and
   existing authenticated mappings. Assign unique IDs and record SVG/Matplotlib
   types, scientific role, quantity/units, data/event source, normalization,
   coordinate system, tick values and composite/mappable relationships. Record
   explicit/high classification evidence and a protection class. Never infer a
   role from colour, position or artist order without renderer/source evidence.
3. Verify the candidate against the resolved current Nature profile at the
   intended final width; historical compatibility typography is not the target.
   If style changes are needed, prepare a separate revised candidate for review
   and preserve the previously approved artwork. Preserve scientific inputs/results.
   Record each approved exception's ID,
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


The [scientific mapping requirements](#scientific-element-identities) and
[versioned role contract](../../../../configs/paper-figures/figure-elements.json)
remain the detailed freeze authority. The gate validates the registry prepared
by the agent/renderer; it does not itself establish scientific truth or conduct
visual review. The [archive procedure](../../../../records/repository-archive/README.md)
governs verified transfer and pruning.

## Current Figure 1 handoff

Use [Figure 1 guidance](../figure-01/README.md), its scoped state and legend.
Current versions, approvals and unresolved coverage do not live in this guide.
