# Reliable Figure Builds and Panel Replacement

## Purpose

Make the existing tools usable across Figures 1–4: resolve the correct selected panel, reproduce or replace it without changing unrelated panels or scientific definitions, and publish an auditable review through the existing workflow.

AGENTS.md defines review, storage and freeze rules. This plan covers their implementation across supported paper builds. Scientific choices belong to Plans 01–06; release packaging belongs to Plan 08. Unavailable or unapproved evidence must be reported explicitly.

## Implementation to reuse

The assembler now provides no-write source/geometry preflight, external-artifact resolution, original/candidate provenance, reproduction commands, baseline comparisons and locked transactional publication. It checks panel payloads, clipping and shared assembly context; protects bound inputs and accepted/managed artifacts; and rolls back failed publication. These are existing building blocks, not tasks to implement again.

Use the [preflight and replacement guide](../docs/paper/figures/shared/assembly-preflight.md). The [implementation audit](../records/scientific-evidence/audits/figure-build-integrity-2026-10-12.json) records 424 passing tests and 70 unchanged frozen records for that implementation. Those checks used synthetic assembly inputs; they do not establish complete paper routing, scientific identity or final visual quality for Figures 1–4.

Reuse [paper-panel orchestration](../src/classical_conditioning/figures/paper_panels.py), [selected-assembly wiring](../scripts/update_latest_figure_assemblies.py), semantic/theme validators and the existing guarded review/current-state workflow. Do not add a second selection registry, review generator or state writer.

## Remaining work

### 1. Connect current selections to supported builds

Connect registry entries, scoped selection records, presentation state and assembly configurations to exact source hashes and rendering routes. The generic assembler authenticates explicitly supplied inputs; it does not discover the scientifically current selection.

A newer scoped panel selection takes precedence over an older whole-figure assembly. Keep frozen selection, active draft and release readiness distinct. A review precursor must never silently substitute for a selected frozen panel.

**Acceptance:** Each supported route resolves the recorded current source or reports a clear missing, stale, blocked or unavailable result. Source selection is never based on filename age.

### 2. Complete available paper reproduction routes

Extend existing commands for approved, available panels, including Figure 3 and current-source wiring for the other figures. Connect the assembler preflight and provenance to these routes. Document exact inputs, settings, renderer and reproduction command; identify unresolved scientific dependencies rather than inventing defaults.

Exercise selected-panel preview, candidate replacement and reproduction on authenticated paper inputs when available. Check untouched panel identities and protected scientific payloads as well as intended geometry. The generic synthetic tests already cover publication failures and replacement safeguards; add route-specific checks where they establish something new.

**Acceptance:** Each available release panel/figure has a supported reproducible command and meaningful preflight/run record. Failures preserve prior accepted work; no build reports a partial result as complete. Missing scientific approvals remain owned by their scientific plans.

### 3. Connect final assembly checks

Apply existing semantic mappings, source identities, dimensions, effective fonts/strokes, clipping, legends and protected-payload checks at final assembly scale. Generic geometry and XML integrity checks are implemented; they do not verify scientific roles, visual appearance or effective styling across renderers.

Use real renderer/visual evidence and approved references where available. Report incomplete semantic coverage and unresolved measurements explicitly. Keep scientific corrections distinguishable from presentation edits.

**Acceptance:** Supported paper builds detect wrong/stale sources and unintended assembly changes, with evidence for relevant final-scale measurements and visual checks. The explicit freeze gate remains a separate author-requested operation.

### 4. Connect builds to the existing review workflow

Pass verified build artifacts and their provenance into the existing consolidated external HTML review and guarded working-copy workflow. Preserve requested versions, editable SVGs, legends, settings and reproduction information. Use the existing figure-state publication lock and snapshot checks when updating a current pointer; assembler-side publication alone does not publish figure state.

**Acceptance:** A reader can locate the exact reviewed artifact and reproduce or replace its panels. Saved artifacts/evidence are verified before atomic state publication; concurrent work is preserved. Missing external evidence remains explicit. Permanent media/data stay external.

### 5. Nature main-figure presentation profile

Author direction, 2026-10-16: Nature is the target for the next paper-figure
presentation revision. Define the complete rule set and record everything to
implement/fix before returning to the Figure 2 A/B/C layout decision. This
section is the implementation checklist, not a claim that existing figures
already follow it. The [source comparison](../records/figure-reviews/nature-guidance-20261016.md)
records official links, source conflicts and observed gaps. Scope covers main
Figures 1–4, captions, linked methods/source data and later Extended Data.
The target is Nature itself; another journal needs its own checked profile.

#### Rule ownership and implementation shape

The [maintained Nature target](../docs/paper/figures/shared/nature-style.md)
owns requirements, usage and detailed axis rules; the versioned machine profile
owns exact overrides and rule origins. This plan owns implementation tasks and
known fixes; the dated source review owns research/audit evidence. Do not copy
role values into plans, AGENTS.md or the skill.

The versioned `nature-main` profile and read-only resolution through the existing
contract/shared theme are implemented. Complete remaining validator and scoped
builder integration without claiming automatic full-artwork enforcement.
Keep one semantic registry and one guarded publication workflow. Bind per-rule
origins, source URLs/date, profile hash and submission stage into evidence.
Retain byte-sensitive specifications and accepted contract copies; profile
resolution must preserve legacy reproduction and scoped freeze compatibility.
The adapter is explicit; legacy defaults and selected artwork have not changed.

#### Complete implementation checklist

Each ID below must have an implementation/verification disposition. These are
future tasks, not another progress board or semantic registry. Progress remains
in the implementation index; exact results belong in version-bound evidence.

| ID | Area | Required implementation and acceptance evidence |
|---|---|---|
| N01 | Authority and baseline | Snapshot selected states, SVG/HTML pairs, legends, existing registries, profile/policy/code and scoped science selections before edits. Authenticate current source bytes under the publication lock; preserve manual/concurrent work. |
| N02 | Complete element coverage | Inventory every visible text, path, line, marker, patch, image, key, colorbar, guide, statistical glyph and schematic. Assign stable unique IDs, canonical roles, owner/allocation and source aliases. Group IDs and XML positions alone are insufficient. Hidden/unused definitions have explicit visibility; do not invent roles from color. |
| N03 | Scientific identities | Map experiment, condition, matched-control cohort, fish, trial/frame/block and eligibility to authenticated sources. Record estimators, units, event definitions and transformations. Layout/style sharing cannot imply common cohorts or equal scientific meaning. |
| N04 | One profile implementation | Add explicit journal/stage/profile resolution to existing theme/export/validation routes. Bind source/version/hash and per-rule origin: journal requirement, author rule, project default or scoped exception. No competing export/state/selection pipeline. |
| N05 | Main dimensions | Check artwork size independently of captions, margins and HTML browser zoom. Support full/single-column targets and the height cap; include all letters, marks, strokes and colorbars in actual bounds. Full page depth is not an artwork allowance. |
| N06 | Physical typography | Measure all fonts through nested SVG transforms. Use resolved profile role values; no ordinary text below 5 or above 7 pt. Inventory statistical phase headers, stars, fish IDs, counts and rasterized labels too. Do not treat a hidden text node as visible compliance or substitute anchor spacing for glyph bounds. |
| N07 | Font fidelity | Verify actual resolved font, glyph coverage, embedding rights and editable PDF/SVG output. Test Unicode minus, degrees, Greek, superscripts/subscripts and mathematical symbols; resolve the guide's Symbol-font instruction explicitly rather than allowing incidental fallback. Sequence/chemical notation rules are applicable only if such content is introduced. No outlined lettering, Type 3 text or whole-figure rasterization. |
| N08 | Panel letters and references | Use lowercase displayed letters at the target size; preserve immutable scientific/source IDs and uppercase historical aliases. Update caption/manuscript/review references and the displayed-to-source mapping atomically; avoid global string replacement. |
| N09 | Axis semantics and visibility | Implement the maintained target’s detailed axis rules, including secondary/twin/inset and colorbar axes, minor/major ticks, spine sides/offsets and scientific multiplier/offset text. Label numerical axes with quantities and units, retaining bounds, tick values/order, categorical meanings, scale type and coordinate origin. Use sentence case, no terminal full stop, SI number–unit spacing and consistent numerical precision; preserve categorical name exceptions. Each shared/suppressed label needs explicit membership/reason. Keep local plot spines/ticks, including E; schematic/annotation coordinates are identified separately. |
| N10 | Axis geometry | Same absolute tick length, width, color, pattern and cap for X/Y; measure both directions. Check label centers against tick positions, tick-end-to-label-edge gaps, title-to-label-envelope gaps and multiline/rotated titles. Tickless categorical exceptions remain explicit, not guessed. Test rendered glyph gaps separately from renderer pad/labelpad values, both signs/unequal-width labels and multiplier text. |
| N11 | Full stroke hierarchy | Check every stroke, including paths reached through SVG use/defs, marker edges, guides, error bars and schematics. Enforce 0.25–1 pt at final scale; compensate uniform resizing and audit directional width under transforms. Distinguish fill-only areas from strokes. |
| N12 | Markers, patterns and opacity | Choose physical marker/arrow/cap sizes per role in a rendered pilot, then reuse them. Preserve fish/summary distinctions, clipping, dash lengths, effective opacity and layer order. Avoid decorative hatching, background grids and drop shadows; scientific line patterns remain explicitly meaningful. Statistical comparison lines remain straight without end caps; do not conflate IQR stems/caps with significance lines. |
| N13 | Frames and grouping | Preserve shared outside blank strip and normal two-inset content gaps. Measure related data rectangles and named edges, not equal allocation widths. Larger gaps need an explicit grouping reason and dimension. Frame containment is necessary but not proof of good whitespace or equal spacing. |
| N14 | Layout and aspect ratios | Recompose logically linked panels at publication size. Preserve natural plot/image ratios, data coordinates and important annotations; no independent X/Y squeeze or clipped marks to force a fit. Uniform size changes need font/stroke compensation. Record any new desired data-axis aspect as a deliberate presentation decision. |
| N15 | Keys and headings | Tag every color/stimulus key as legend.key plus its subtype, identify members/represented series, and align the whole key to a named right edge. Black key text plus colored glyphs; no redundant counts. Center condition headings over their own data rectangles with common baselines. An experiment heading cannot masquerade as a single-condition heading. |
| N16 | Color accessibility | Review color-vision deficiencies, text/background contrast and grayscale discrimination where useful. Target text contrast above 4.5:1; do not invent one numerical contrast test for all data marks. Keep meanings independent of color alone. Black text is the default; add graphical keys where colored headings currently carry identity. Record palette/series mappings and actual resolved colors; changes to frozen raster colors require a separately specified candidate. |
| N17 | Heatmaps and colorbars | Record cell coordinates, matrix shape/binning, interpolation, missing values, zero, limits, normalization, units and mappable links. Colorbars get their own allocation and role members. Shared keys must not imply estimator equivalence; preserve phase/trial mappings and authenticated raster bytes. |
| N18 | Source images and schematics | Authenticate original assets, retain separate editable annotations/scale bars where applicable, document crops/composites/reuse and distinguish drawings from data. Use scale bars rather than magnification factors; retain explanatory schematics, remove only superfluous decoration. Check duplication, permissions and image integrity. No generative image synthesis/inpainting or artificial resolution claims. Record the 300/450-dpi source conflict: target native 450 dpi for production photographs, with 300-dpi assets flagged for source/editor resolution, never artificial upsampling. Do not apply this rule to discrete heatmap matrices. |
| N19 | Statistical reporting | Link every statistical line/star/row to its exact contrast/result, test/tail, multiplicity family, sample unit and applicable source. Define uncertainty type, level, aggregation and resampling unit. Distinguish repeated trials from independent fish; exact n may vary by block. Missing outputs stay unavailable, never fabricated to complete a caption. |
| N20 | Concise captions and Methods | Draft short figure title, panel descriptions, key definitions, exact relevant n, central tendency/uncertainty and essential statistics. Move detailed processing into shared Methods with a traceable old-to-new paragraph map. Keep the complete prior legend; do not delete scientific definitions to reach a word count. Count by a documented method and cross-check against the reporting summary. |
| N21 | Source-data package | Map panel/series/contrast IDs to source numerical tables and reproducible code/dependencies. Include exclusions, missingness and changing denominators. Reuse analysis evidence owned by Plans 01–06; availability/code/reporting statements must match actual accessible artifacts. No new scientific approval follows from formatting. |
| N22 | Review versus submission | Keep scoped SVG+legend HTML and ordered main collection. Do not apply the artwork font cap to the off-artwork review caption. Export separate editable final artwork and manuscript captions; use an editor-selected submission format rather than treating HTML as a submission file. |
| N23 | Export preflight | Verify native SVG and vector PDF appearance, text/font editability, resource embedding, color space, dimensions, file-size budget and no linked resources. Avoid lossless-looking previews that actually flatten text/data. Check numerical/raster hashes and source-to-export lineage, not screenshots alone. Include concise production instructions for essential registered alignments. Support recorded 120–136 mm compositions when needed rather than falsely rejecting all intermediate widths. |
| N24 | Extended Data and other outputs | Define a separate Nature Extended Data profile/delivery route, preserve editable masters, and apply the dedicated dimensions/encoding/size requirements and naming. Resolve the typically ten-item ED budget with the manuscript; distinguish ED figures, supplementary data/tables/video and internal analysis atlases. Do not silently label extra submission figures as Supplementary Information. No result may be moved there solely to pass the main-figure height check without an explicit composition decision. |
| N25 | Browser and print verification | Measure exact saved SVGs with painted/unclipped overflow checks and active hashes. Inspect at final size for legibility, collisions, dashes, colors and clipping. Compare collection embeddings with standalone selected SVGs, test narrow screens, inspect every print page, and disclose nonzero rasterization differences rather than claiming pixel identity. |
| N26 | Failure and publication tests | Exercise stale states/sources/profile hashes, changed manual SVGs, unavailable storage, incomplete mappings, unembedded fonts, limits, alias errors, invalid sharing and publication rollback. Evidence/artifacts publish before scoped state; refresh the collection last. Preserve originals and approved candidates. |
| N27 | Durable guidance | Keep AGENTS.md/skill as short routes to maintained guidance; the workflow owns procedure and the plan owns future work. Reconcile permitted consumers with profile resolution; remove unconditional legacy defaults from Nature routes. Preserve byte-sensitive specifications and frozen contract copies. Add superseding guidance/compatible versioned profiles rather than editing accepted history. |
| N28 | Freeze boundary | Separate formatting pass, semantic/scientific completeness and release readiness. A partial draft may be reviewed with explicit gaps; it must not be called fully Nature compliant. An explicit freeze still requires complete scoped registration and the existing freeze gate. No cleanup/disposal or new scientific freeze is authorized here. |

#### Known fixes and decisions to carry into the first audit

The following are grounded in repository contracts, selected state, prior
version-bound evidence or author notes. The selected artwork hashes and canvas
dimensions were rechecked on 2026-10-16; no fresh full visual/element audit is
claimed here.

| ID | Observed issue / decision | Required resolution |
|---|---|---|
| F01 | Main heights about 215.46 / 203.42 mm | Recompose both figures; do not merely reduce the complete image and leave unreadable fonts. |
| F02 | 8-pt ordinary text and uppercase/10-pt letter defaults | Migrate all affected roles, aliases and references together; retain previous scoped approvals as history. |
| F03 | Summary/median-IQR defaults are 1.2 pt | Lower the Nature-profile defaults to the target hierarchy and inspect actual paths, not just config values. |
| F04 | Legacy tiny G statistical headers/glyphs | Audit every one, allocate categorical row/column space and preserve hypothesis mappings. Earlier work enlarged row labels but did not certify all annotation typography. |
| F05 | `paired_block_ratio` defaults still put conditioned fish first and omit other local spines | Reconcile with author-selected control-first layouts and local E ticks. Introduce an accurately named log-difference plot family for current D/E; preserve historical ratio records/aliases and y=1 references where scientifically appropriate. Never turn ratio data into log differences through styling. |
| F06 | Colored condition headings | Replace their color-only identity with black headings plus explicit condition keys while keeping centered column alignment. |
| F07 | Figure 1 E has excess space before its colorbar | Reallocate D/E data regions and shared colorbar column together. Keep E and the F–H color scales right-aligned without separating D/E excessively or stretching scientific geometry. |
| F08 | F–H equal body gaps can still be excessively large | Recheck whole-row whitespace, outside frames and named alignments together. Register which gaps are normal and which are grouping gaps; do not equate equal spacing with optimal spacing. |
| F09 | Figure 2 A/B/C bodies differ in aspect from Figure 1 F–H | Compare (i) side-by-side pairs with shorter bodies and (ii) matched control above paired within each experiment. Measure trial/bin legibility, overall figure height, column alignment and control identity. Stacking remains an option, not an approved switch or a scientific pipeline change. |
| F10 | A/B retain different estimators under a common displayed color range | Audit the sharing contract; use separate, correctly labelled scales or explicitly justified shared visual-key semantics pending scientific harmonisation. Equal limits alone are not proof of interchangeable measurements. The scientific change belongs to its owning plan. |
| F11 | H is a ratio while D/E/G are log differences | Preserve independent scientific axes/references/units and explain the distinction. Matching typography does not authorize shared limits or outcomes. |
| F12 | Current legends roughly 774 / 794 words | Draft concise captions plus linked Methods; keep exact sample sizes, missingness, normalization and inference qualifications recoverable. |
| F13 | Axis inventory/legacy XML IDs cover only part of each figure; structure preflight reports fig-1E/F/G/H registry/state disagreements | Extend the existing registry to all elements and relationships. Trace disagreements through exact scoped selections, source IDs and displayed aliases; do not infer scientific equivalence from a successful layout pass. Resolve ownership/scientific evidence gaps before a complete compliance/freeze claim. |
| F14 | Control-first panel/source aliases and selected Delay example links | Preserve current displayed F=Control, G=Delay, H=3sTrace; D/E examples and arrows/bands link to the selected Delay source in G. Do not reintroduce old display-letter assumptions. |
| F15 | C/F/I are Pending; B/H and whole figures have unfinished scopes | Retain honest placeholders in drafts. Final submission readiness needs those results completed or an explicitly approved figure composition; do not manufacture data or silently drop panels. |
| F16 | Many renderers/measurement helpers hard-code prior typography | Parameterize all consumers and ensure a future edit cannot silently reintroduce old sizes, key placement or wording. Verify actual selected output after route changes. |
| F17 | Late-stage label scaling/packing changes | Recheck label centers against tick values, multiline titles, rotated Y labels, trial-description columns, right-aligned keys and statistical-row separation after every reflow. |
| F18 | Nature web pages differ on some dimensions/formats | Keep source/stage precedence and the conservative working targets documented; follow specific editorial instructions if received. Do not silently merge incompatible main/ED requirements. |
| F19 | The target table was stored in a plan; AGENTS.md/skill lacked direct Nature routing | Corrected during the 2026-10-16 review: maintained rules now live under shared paper guidance; plan owns implementation. Keep entrypoints concise and distinguish selected policy from executable enforcement. |
| F20 | Renderer padding can be mistaken for geometric spacing | `theme.py` passes 3-pt pad/labelpad; this does not prove 3-pt glyph-edge gaps. The supported SVG correction/measurement route exists, but generators/exporters need profile-bound final checks too. Preserve that route rather than implementing another fixer. |
| F21 | Historical element specification is byte-sensitive; current structure preflight reports preserved-hash mismatches | Preserve it and frozen contract copies. Resolve existing discrepancies through provenance review; do not rewrite expected hashes or accepted history to silence checks. Future profile resolution must support prior exact reproductions while applying new targets to new candidates, without implicit freeze execution. |
| F22 | Axis edge cases and authority wording were underspecified | Added minor ticks, sides/offsets, twin/inset/secondary axes, numeric formatting and multiplier ownership. Removed AGENTS.md's stale hard-coded v10 and workflow wording that implied only Figure 1 has presentation state; scoped state remains authoritative. |

#### Implementation locations and sequence

1. **Baseline and audit (N01–N03, F01–F22).** Resolve all current selections;
   classify each finding as confirmed, measured, source-derived or unverified.
   Evidence rows carry element/relationship ID, source hash, expected value,
   actual final measurement, unit/tolerance, reproduction and disposition.
   If JOAQUIM is absent, continue code/docs and authenticated offline design;
   stale local snapshots are not current authority. Record unavailable checks.
2. **Rules and consumers (N04–N12, N27).** Use the implemented profile resolver
   and explicit `theme.py` adapter. Complete `export.py`, `figure_workspace.py`,
   scoped renderer integration and compatible freeze validation, with profile
   hashes and honest coverage. Preserve immutable/byte-bound contract copies and
   explicit legacy reproduction; do not replace them in place to change defaults.
   Parameterize `style_axis_geometry.cjs`, `align_paper_annotations.cjs`, their
   Python preparation/publication helpers and older assembly entry points.
   Resolve conflicts in the specification/workflow/skill at the same time.
   Keep draft validation separate from explicit freeze execution.
3. **Representative pilot, then all roles (N06–N19).** Render one dense example
   of traces, boxed heatmaps, paired block summaries, trajectories with bands,
   statistical annotations and schematics. Choose/record missing marker/dash
   dimensions and test actual font/glyph embedding. Use the validated roles
   across Figures 1/2 and future Figures 3/4 without revisiting scientific gates.
4. **Layout and captions (N13–N20).** Reflow Figure 1 first, then compare Figure
   2 heatmap arrangements within the complete figure height budget. Review
   caption/Methods pairs and panel aliases together. A needed result split,
   changed estimator, palette/normalization or exception needs a concrete scoped
   decision; ordinary reversible presentation choices can proceed.
5. **Delivery and verification (N21–N26).** Implement source-data/export checks,
   main and ED packaging, full renderer/print QA and failure tests. Extend
   `layout_frames.py`/its browser collector and the existing collection checker
   only where needed. Bind profile, policy and implementation hashes to the
   existing scoped registry and evidence, publish under guards and refresh the
   collection last. Single-figure requests retain scoped display behavior.

#### Completion gate

Every checklist ID has an implemented check or an explicitly scoped human
review with evidence; unknown properties do not pass. All visible roles have
the required identities and final-scale measurements. Every author relationship
is rechecked on the new bytes. Captions, Methods and numerical/source mappings
agree; unresolved scientific work is neither approved nor hidden by a style
pass. A draft can be presented with gaps, but a submission/Nature-ready claim
requires all applicable checks, including completed selected scientific content.
The current saved drafts remain authoritative until reviewed replacements are
published through the existing guarded workflow. This planning step does not
publish figures, rewrite their captions, freeze science or remove history.

## Scope and completion

This plan does not choose cohorts, learner methods, examples, trial matching, pooling, centring or presentation exceptions. It connects approved choices without recomputing or refreezing them. Historical frozen snapshots remain immutable; candidates remain separate.

Additional notebooks, inspection APIs and a general command framework are optional, not first-release prerequisites. Background ideas remain in the [preserved tooling design](../records/repository-history/tooling-designs/figure-build-tools.md).

Complete this plan when the four integration requirements and applicable Nature-profile checks above are demonstrated for supported Figures 1–4 release builds, with unavailable scopes and their owning dependencies explicitly recorded. Generic assembler implementation alone is not completion. Track progress only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md).
