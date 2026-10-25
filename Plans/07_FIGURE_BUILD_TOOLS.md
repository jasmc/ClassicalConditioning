# Reliable Figure Builds and Panel Replacement

## Purpose

Make the existing tools usable across Figures 1–4: resolve the correct selected panel, reproduce or replace it without changing unrelated panels or scientific definitions, and publish an auditable review through the existing workflow.

AGENTS.md defines review, storage and freeze rules. This plan covers their implementation across supported paper builds. Scientific choices belong to Plans 01–06; release packaging belongs to Plan 08. Unavailable or unapproved evidence must be reported explicitly.

For the current Nature work, start with the
[2026-10-17 completion handoff](#6-nature-completion-handoff-2026-10-17): latest
author constraints, authenticated baseline, H01–H10 work packages and acceptance.
The N/F tables below retain the detailed requirement/finding identifiers.

## Implementation to reuse

The assembler now provides no-write source/geometry preflight, external-artifact resolution, original/candidate provenance, reproduction commands, baseline comparisons and locked transactional publication. It checks panel payloads, clipping and shared assembly context; protects bound inputs and accepted/managed artifacts; and rolls back failed publication. These are existing building blocks, not tasks to implement again.

Use the [preflight and replacement guide](../docs/paper/figures/shared/assembly-preflight.md). The [implementation audit](../records/scientific-evidence/audits/figure-build-integrity-2026-10-12.json) records 424 passing tests and 70 unchanged frozen records for that implementation. Those checks used synthetic assembly inputs; they do not establish complete paper routing, scientific identity or final visual quality for Figures 1–4.

Reuse [paper-panel orchestration](../src/classical_conditioning/figures/paper_panels.py), [authenticated current collection](../scripts/paper/shared/collection.py), semantic/theme validators and the existing guarded review/current-state workflow. Do not add a second selection registry, review generator or state writer.

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
The adapter is explicit; legacy defaults remain available for exact historical
reproduction. Selected Figure 1/2 presentation drafts have since migrated through
the guarded route. Resolve their current states and read the
[2026-10-17 handoff](#6-nature-completion-handoff-2026-10-17) before resuming;
neither this dated plan nor a browser tab selects an artifact.

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

The following retain the original 2026-10-16 audit findings and stable finding
IDs. They are not a list of defects all still present in current artwork. Later
measured changes and outstanding work are distinguished in the handoff below
and in state-bound evidence. Do not rerun the original migration or apply an old
finding without checking the current selected bytes and latest author scope.

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

### 6. Nature completion handoff, 2026-10-17

This is the execution handoff for the remaining N01–N28 and F01–F22 work, merged
here rather than creating a competing numbered plan. Plan 07 owns tooling,
semantic/presentation checks and supported figure delivery. [Plan 08](08_FIRST_RELEASE_FIGURES_1_TO_4.md)
owns acceptance of the Figures 1–4 release. Plans 01–06 own scientific evidence
and unresolved scientific decisions. Update live progress only in the
[implementation index](IMPLEMENTATION_STEP_INDEX.md); record measurements under
`records/` or in authenticated external evidence, not in another status registry.

#### Latest author scope: preserve before improving

- Keep the Nature typography, axes, ticks, physical spacing, strokes, keys,
  naming, semantic mappings and invisible-frame rules already applied.
- The final clarification was **“do nothing but marking with the black box”**,
  followed by the instruction to retain the other Nature changes. The black
  136 × 170 mm rectangle is an HTML review guide. It must not replace invisible
  allocation metadata, modify the artwork, become a scientific element or enter
  final submission artwork. Its review wrapper may reserve caption clearance
  without changing native SVG dimensions or transforms.
- No further shrinking, stretching, automatic repacking, canvas change, stacking
  or rollback to the old eLife artwork follows from this plan or from displaying
  the guide. The author previously allowed the eLife outer dimensions, then
  restricted the immediate edit to the box. Do not infer an instruction to
  restore old sizes or undo Nature styling. Prepare a concrete proposal only
  when a later scoped layout request calls for it.
- **Learner counts belong to Figure 3 only**, under [Plan 04](04_FIGURE_3_LEARNER_ANALYSIS.md).
  Do not add learner counts, learner percentages or learner classifications to
  Figures 1/2, their captions, registries or Nature checks. Fish/cohort counts
  and block-specific denominators remain required where applicable; they are
  distinct from learner counts. This handoff does not choose a learner rule.
- Do not recompute cohorts, timing, binning, normalization, significance,
  estimators or missing results as presentation work. Preserve frozen originals,
  manual edits, selected dependencies and unrelated Figure 3/4 work. No freeze,
  release, archive, deletion or historical cleanup is authorized by this plan.

#### Start here: authority and measured baseline

Read `AGENTS.md`, the repository paper-figures skill, the authoritative
[iteration workflow](../docs/paper/figures/shared/iteration-workflow.md) and the
[maintained Nature target](../docs/paper/figures/shared/nature-style.md). Resolve
`docs/paper/figures/figure-01/current-state.json`, Figure 2's corresponding state
and `main-paper-current-state.json`; authenticate their SVG, caption, registry,
scientific-selection and evidence references. Check storage before reading media.
Use the offline artwork procedure only within its documented snapshot limits.

The 2026-10-17 read-only audit found the following; these are dated observations,
not replacement selection pointers or blanket compliance claims:

| Checked area | Observation and reuse boundary |
|---|---|
| Selected artwork | Figure 1: approximately 136 × 163.70 mm; Figure 2: 136 × 156.47 mm. Both SVG hashes matched their scoped states. Do not apply the original 215.46/203.42-mm findings as current dimensions. |
| Invisible frames | Both exact painted/unclipped checks passed against the active policy hash. Panel and shared-colorbar ownership and the selected inset remain present. |
| Relationships | Figure 1 F/G/H layout and D/E/displayed-G scientific links remain registered. Figure 2 A/B/C, D/E/F and G/H/I layout groups remain registered; their declared spacing checks passed. |
| Larger gaps | F1 F/G/H data-body gaps ≈11.15 mm; F2 A/B/C pair-body gaps ≈11.13 mm; F2 D/E/F whole-content gaps ≈7.15 mm; F2 G/H/I whole-content gaps ≈3.89 mm. The target types differ. Equal measured gaps do not mean every adjacent content gap is the normal two-inset gap, or that composition is finally approved. |
| Styling and axes | State-bound reports cover measured typography, strokes, registered axes, key/heading alignment and dense G annotations. The axis inventory reports no unresolved ownership; this does not establish complete scientific identity for every mark. |
| Incomplete roles | The audited presentation reports contain 55 Figure 1 and 372 Figure 2 stroke instances with generic `unclassified.presentation.stroke` tags. Audit by element/instance and source before assigning scientific roles; these counts are not missing-data or learner counts. |
| Captions | Concise current captions and preserved detailed definitions exist. Authenticate current caption/Methods references; do not replace them with the original long legends or shorten them again without a reason. |
| Delivery evidence | Browser/responsive and two-page print review exist. Embedding screenshots have nonzero pixel differences; the DOM comparison includes hidden zero-box nodes and needs correction before numerical equivalence can be claimed. Actual production-PDF font/editability validation remains open. |
| Scientific scope | Figure 2 C/F/I remain Pending. B/H retain distinct unfinished estimators; H ratios are not the D/E/G log-difference outcome. No whole-figure freeze or submission readiness follows. |

The external review area is
`/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/nature-main-20261016/`.
`size-guide.html` and its evidence are an explicitly annotated review;
`main-paper-figures.html` is the collection destination resolved through state.
Neither a filename nor an older `candidate.html` is authority. The chronological
[migration record](../records/figure-reviews/nature-main-migration-20261017.md)
contains interruption-era wording; reconcile its latest disposition against
current state before following any old resume instruction. Do not run `prepare`,
`compact` or `pair-spacing` over the existing directory as a generic resume step.

<a id="immediate-completion-order-25-october-2026"></a>

#### Completion follow-up, 25 October 2026

The [reconciliation record](../records/figure-reviews/authority-reconciliation-20261025/README.md)
closes the four current Figure 1/2 SVG/HTML binding conflicts. The exact PC
Figure 2 state, matching captions, current semantic mappings and saved frame
measurements were recovered/authenticated. Figure 1 H now binds its recorded
20230419_10 unfrozen example; Figure 2 binds the previously published fixed10sTrace
C/F/I and earlier expanded B/E/H draft. Caption/HTML/SVG agreement passes;
SVG/HTML artwork bytes and accepted freezes remain unchanged. The ordered
collection was refreshed last through the supported guard. External structure,
current-artifact and archive authentication passes without exceptions beyond
the already recorded exact author-approved archive absences.

Policy/helper/collector source hashes differ only by Git LF/CRLF transport.
The supported adapter proves those byte variants, preserves original reports
and measurement arrays, and revalidates the exact saved Edge evidence. It does
not claim a new browser run. Temporary native rendering and embedded-font/image/
reference checks supplement the saved evidence; production-ready claims remain
out of scope of that transport check.

The D/E clipping investigation now identifies opaque full-canvas backgrounds
in the overlapping E/F viewports as a concrete cause. The
[supported repair](../docs/running-analysis/figure2-canvas-repair.md) prepares
an unselected candidate changing only those two fills; all other SVG bytes
remain identical. Native before/after rendering restores the hidden LT points,
labels and statistical marks, with no changed pixels outside the target canvas
regions. Ten focused safety/publication tests pass. The exact candidate,
matching HTML, original snapshots and evidence live on JOAQUIM.

Fresh exact browser frame measurements are still required before selecting that
candidate. The old Edge report is not evidence for the changed SVG. Scientific/
role registration, accessibility, collection equivalence and actual production-
PDF checks under H02–H10 also remain open. No further automatic repacking or
artwork resizing follows. Keep the author’s box-only constraint.
Any changed artwork needs fresh exact-render measurements; transport equivalence
cannot authenticate different SVG bytes or modified source logic.

The [corrected BLUP comparison](../docs/running-analysis/figure3-blup-comparison.md)
now completes its Delay/control calibration and independent stress runs. It
supplies decision evidence, not Gate L or a new Figure 3 selection. Plans 01–06
still own cohort, scientific method/claim decisions and applicable full-assay
validation; Plan 08 consumes those approvals for release. CI is removed and
repository Actions disabled; all relevant checks are local.

#### Work packages and acceptance evidence

Execute H01 first. H02–H06 can progress without changing artwork geometry.
H07 composition implementation waits for a later scoped request. H08–H10 depend
on authenticated inputs and the applicable earlier checks. IDs below bundle the
existing N/F requirements; they do not introduce new semantic roles or approvals.

**H01 — Reconcile authority, coverage and reproducibility (N01, N04, N27, N28).**
Capture starting hashes for the exact scoped inputs and only the implementation
files being changed. Distinguish a rule being recorded, implemented, measured,
human-reviewed, scientifically accepted and release-ready. Reconcile stale
“migration pending” guidance/resume notes and any historical baseline assertions
without rewriting dated evidence. Audit Figure 2's legacy
`assembly_scale.A_native_scale = 1` against the source-relative allocation map;
do not use that field as proof of current native scale. Check generator/evidence
hashes, source/display aliases and caption provenance together. Produce one dated
coverage report mapping every applicable N/F ID to existing evidence, remaining
work or a justified not-applicable disposition. Missing storage or changed manual
inputs stop publication; independent code/doc work can continue.

**H02 — Complete element and scientific registration (N02, N03, N09, N17–N19,
N21; F10, F11, F13, F14).** Extend each existing draft registry, not a new index.
For every visible element and referenced instance, record stable draft identity,
canonical role/subtype, renderer/source alias, visibility, owner/allocation,
style group and source/result mapping where scientific. Include glyphs inside
text spans, markers and marker edges, bands, clipping boundaries, statistical
rows/brackets/stars, arrows, stimuli, scale bars, colorbars, raster matrices and
schematics. Distinguish resource definitions from rendered instances. Preserve
source IDs; add aliases or deterministic draft IDs without modifying frozen bytes.

Map experiment, condition and matched control separately. Bind cohort/FishKey,
trial/frame/block, exclusions, estimator, aggregation, baseline, event windows,
normalization, uncertainty and statistical contrast/correction family to exact
sources. Represent layout, style, shared-coordinate and scientific relationships
as different types. Record which elements share which axis or mappable, and why
labels are suppressed. A color, structural ID or successful bounds check is not
scientific evidence. No invented mappings or approval may close an unknown.
Acceptance requires an inventory-to-render reconciliation and zero unexplained
visible instances for any scope claimed complete; unknown scopes stay explicit.

**H03 — Finish typography and axis edge cases (N06–N10, N15; F02, F04–F06,
F17, F20, F22).** Reuse the resolved profile and existing axis/annotation helpers.
Keep exact values in configuration, not per-panel patches. Verify X/Y ticks,
colorbars, shared/secondary/inset axes and schematic-coordinate exceptions at
native size through all transforms. Check tick values and side/visibility,
length/normal stroke/cap, label centering, nearest painted-glyph gaps, title
centering, multiline/rotated text and offset/exponent envelopes. Include nested
spans and inherited/reference styles; a parent text style is insufficient.

Keep E's local vertical spine at its data edge while sharing only the explicitly
named labels. Check D/E example block/trial labels at identical physical sizes;
the same applies across figures for each shared role. Preserve `3sTrace` and
`10sTrace`, lowercase displayed panel aliases, right-aligned graphical keys and
black key text. Center condition headings on their own columns and check common
baselines. Keep Figure 2 G's misleading experiment/condition heading absent and
its cohort counts in the caption. Resolve actual normal/bold/Greek/math fonts,
glyph coverage and line spacing. Evidence must name each measured role and exact
SVG; author-selected geometry is not a universal Nature-prescribed number.

**H04 — Complete the graphical-role pilot (N11, N12, N17, N19; F03, F04).**
Classify remaining generic strokes from renderer/source context. Audit effective
directional widths, referenced tick paths, marker outlines, schematic arrows,
comparison lines and colorbar outlines. Define missing physical marker diameter,
arrowhead, error-cap and dash-length/offset choices by role using a representative
rendered pilot; record their origin as project/author choices rather than inventing
Nature mandates. Check line caps/joins, opacity multiplication, uncertainty-band
visibility, fill-only regions and layer order. Scientific dashed guides remain
meaningful; decorative patterns/grids/shadows require review, not indiscriminate
deletion. Preserve the statistical contrast and star count when relocating or
restyling marks. Reuse the ordinary/non-scaling-stroke calibration and test use/defs
inheritance, horizontal/vertical marks and nested anisotropic transforms. Acceptance
is role-specific measured and visual evidence, not only a permitted-width range.

**H05 — Accessibility and source-image integrity (N07, N16–N18, N23).**
Inspect normal vision, relevant color-vision simulations and grayscale where useful;
check text contrast and condition/phase/uncertainty discrimination without color
alone. Inventory actual palette/opacity/missing/zero/saturation meanings, including
CS/US, Control/condition traces and heatmap keys. Preserve authenticated raster
bytes; an accessibility-driven recoloring needs a separate traceable candidate,
not silent editing of accepted cells. Confirm native image resolution by asset
type, editable scale bars/annotations, crop/composite/reuse provenance and relevant
permissions. Distinguish drawings from measured images. Do not upsample to claim
new detail, apply photographic DPI rules to discrete matrices, or synthesize data.
Resolve the guide's source/stage conflicts and any actual glyph fallback explicitly.

**H06 — Close caption, Methods and source-table lineage (N08, N19–N22;
F10–F12, F14, F15).** Reconcile panel aliases, counts/sample units, variable
denominators, omitted data, keys, units, central tendency/uncertainty and every
statistical symbol with the selected sources. Preserve A/B estimator differences
and H ratio semantics; styling must not harmonize them. Verify concise captions
against their retained detailed definitions and manuscript references, including
an explicit old-to-new definition map where still missing. Resolve shared Methods
into the canonical scientific-methods home; avoid divergent duplicate definitions.
Map exported numerical tables to panel/series/contrast IDs, code, configuration,
cohort/exclusion evidence and reproduction commands. Missing outputs or controls
remain explicit and owned by Plans 01–06. Keep learner counts out of Figures 1/2.

**H07 — Composition proposal only when requested (N05, N13, N14, N25;
F01, F07–F09, F17).** Keep invisible figure/panel/shared allocations, the shared
outside strip and normal two-inset content spacing. Measure content envelopes
and data rectangles separately; document each larger grouping gap with actual
width, members, reason and review disposition. Reassess D/E/colorbar proximity,
F/G/H shared-axis layout and each Figure 2 row together. A named gap or equal
spacing is not aesthetic approval. If later requested, compare side-by-side and
stacked Control/condition arrangements for A/B/C against Figure 1 F/G/H ratios,
legibility and control identity. Stacking and changed scientific aspect targets
remain unselected. Use suitable journal dimensions, not a compulsory 183-mm canvas
or automatic 136-mm shrink. Respect the latest no-resize/box-only instruction;
this work package currently authorizes planning and measurement, not reflow.

**H08 — Native export and collection equivalence (N05, N07, N18, N22–N25).**
Use the existing exporter/collection renderer. Validate final editable vector
artwork and manuscript captions separately from review HTML. Inspect the actual
PDF's font types/embedding/subsets, editable glyphs and vectors, RGB resources,
physical dimensions, native raster resources and linked-resource absence. A browser
print preview or font configuration flag does not certify production export.
Keep main, Extended Data and other delivery profiles separate; implement only
the output scope needed for the release, with conditional content marked N/A.

Extend `verify_main_paper_collection.cjs` rather than adding a competing checker.
Compare standalone and embedded saved SVGs using corresponding visible geometry,
computed styles/fonts, raster payloads and screenshots. Exclude hidden zero-box
nodes/resource-only definitions from geometric comparison; explain subpixel
capture/antialiasing differences and bind meaningful tolerances instead of
claiming pixel identity. Check all figures/legends in order, self-containment,
unique IDs/internal references, stylesheet/font isolation, native-size print,
uniform narrow-screen fitting including non-scaling strokes, clipping and every
printed page. Ensure the black guide is absent from submission/normal collection
artwork. Bind final collection verification to the exact HTML hash and publish
its evidence before any verification pointer. Reuse unchanged verified output.

**H09 — Efficient durable integration and failure handling (N01, N04, N21,
N23, N25–N28; F13, F16, F19–F22).** Use one resolved profile per build and pass
the same theme through plotting helpers inside scoped Matplotlib rc contexts.
Extend `presentation_profiles.py`, `theme.py`, `export.py`, `paper_panels.py`,
`figure_workspace.py` and the existing figure-specific adapters only where the
coverage report identifies a gap. Route ordinary edits by named role/group, not
by repeatedly reading every plan, running every renderer or copying constants.
Keep a small supported build/check/publication entrypoint and explicit legacy
reproduction; consolidate disposable migration steps into reproducible supported
adapters where warranted. Migrate Figures 3/4 only within their requested scope.

Cache/reuse expensive renderer evidence only when exact SVG, fonts/resources,
profile/policy, collector/helper code, dimensions and scientific-selection inputs
are unchanged. A changed owner/shared style invalidates affected relationships;
unchanged unrelated figures need no rerender. Keep AGENTS.md and the skill short;
maintained guidance owns rules, the workflow owns procedures, state owns selection
and this plan owns future work. Resolve stale historical-default wording without
changing byte-bound historical contracts. Check-only and display remain read-only.

Add meaningful route checks for stale/manual inputs, changed profiles/fonts,
unsupported widths/effects, absent storage, missing role/source mappings, duplicate
IDs, false axis sharing, unembedded resources, source/display alias mistakes and
state/output write failures. Test multi-figure publication interruptions: recovery
must identify exactly which scope was published, preserve prior state/artifacts
and never expose a mixed collection as verified. Reuse existing rollback tests
where they already establish the property. Preserve all preexisting work.

**H10 — Reviewed delivery, optional freeze and release handoff (N20–N28).**
Record actual visual review at native size, measurements, protection checks and
open scientific coverage on exact saved bytes. Publish artifacts/evidence before
atomic scoped state under existing locks; refresh the ordered collection only
after its selected scopes pass. Particular figure/panel requests show only their
own selected SVG and legend; plural paper requests use the authenticated collection.
Do not add Figures 3/4 to the selected collection implicitly. A later explicit
freeze requires complete scoped element/scientific registration and the existing
freeze gate, never a generic-stroke or nominal-style substitute. Hand the verified
artifact/evidence references to Plan 08 without silently closing scientific gates.

#### Minimum evidence row and completion contract

For each applicable check, record: N/F ID; figure/panel/element or typed
relationship ID; canonical role; source/selection and artifact hashes; rule origin
and profile/policy version/hash; expected and actual values with physical units;
tolerance and measurement method; tool/font/resource hashes; exact reproduction
command; result (`pass`, `fail`, `unresolved` or justified `not applicable`);
reviewer/review evidence; and owning scientific dependency where relevant.
Separate scientific meaning, presentation conformance and storage/publication
status. Human review is valid evidence only for the explicitly reviewed scope.

Complete the handoff when H01–H10 and all applicable N/F requirements have a
traceable disposition, the selected output is reproducible, role/scientific
coverage is complete for any scope claimed complete, and Plan 08 has the required
delivery references. Do not claim final Nature or release readiness while
applicable production/scientific checks remain unresolved. The planning request
does not execute these packages, resize artwork, alter selections or freeze data.

## Scope and completion

This plan does not choose cohorts, learner methods, examples, trial matching, pooling, centring or presentation exceptions. It connects approved choices without recomputing or refreezing them. Historical frozen snapshots remain immutable; candidates remain separate.

Additional notebooks, inspection APIs and a general command framework are optional, not first-release prerequisites. Background ideas remain in the [preserved tooling design](../records/repository-history/tooling-designs/figure-build-tools.md).

Complete this plan when the four integration requirements and applicable Nature-profile checks above are demonstrated for supported Figures 1–4 release builds, with unavailable scopes and their owning dependencies explicitly recorded. Generic assembler implementation alone is not completion. Track progress only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md).
