# Nature presentation migration: interrupted before publication

Status on 2026-10-17: **storage reconnected; compact revision and final publication pending**. JOAQUIM was authenticated after reconnection on 2026-10-16, then disconnected again. No selected Figure 1/2 state or collection pointer was published by this migration. Do not show the previous collection as the updated result.

## Preserved work

Staging destination (reconnected): `/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/nature-main-20261016/`. It holds starting-ticket hashes, exact source SVG copies/registries, presentation maps, candidate SVGs, concise captions, browser reports and a candidate HTML. Preserve all of these; do not initialize over them or substitute the older offline panel cache.

The candidates apply embedded Arial text, lowercase panel aliases, profile-based typography/strokes, uniform allocation resizing, physical axis geometry, right-aligned keys, separated heatmap headings, a larger Figure 2 G inset, explicit phase headings and redundant statistical-prefix aliases. The latter retain every significance star and hypothesis row. Full old caption definitions are preserved in `docs/paper/figures/shared/figure-specific-methods.md`.

## Verification completed and limits

- 51 targeted Python tests pass locally, including source/raster/data protection, aspect-ratio rejection, statistical alias integrity, physical stroke measurement, tick-template inheritance, current/source panel aliases, stale inputs and publication rollback.
- The browser stroke calibration passed on the mounted drive: physical 1-pt horizontal/vertical strokes through anisotropic transforms, ordinary strokes, dashed strokes and inherited tick-template widths. Evidence is staged as `stroke-calibration.json`.
- Exact saved-SVG painted/unclipped frame checks and visual review passed for earlier candidate revisions at 183 mm. Later dense-annotation refinements require their exact final reports to be authenticated again after reconnecting; do not infer passing final evidence from the earlier revisions.
- Arial glyph coverage was checked for all Figure 1/2 text; no missing characters were found. Actual production-PDF font/export and full accessibility checks remain open.
- Combined final collection screen/standalone comparison, responsive and print checks remain pending.
- Nature profile selection and these tests do not establish complete scientific registration, full Nature compliance, submission readiness or a whole-figure freeze. Composition whitespace and the unapproved Figure 2 stacking option remain F07–F09. Figure 2 B/H remain distinct unfinished estimators; C/F/I stay Pending.

## Resume

1. Confirm JOAQUIM is mounted; inspect the saved staging ticket and exact candidate/report hashes. Keep source, selection, registry and manual-edit guards active. Implementation changes are recorded separately; do not relax science/source locks.
2. Recheck the final candidate SVGs with the existing axis/annotation/Nature verification helpers and exact layout collector if reports are missing/stale. Inspect the dense G phase/row labels and H key at native size; the publisher rejects their collisions.
3. Run the scoped Nature publisher with an actual visual-review note only after all required checks pass. It preserves prior source/caption/state evidence and updates the same working SVG/HTML pair through guarded publication.
4. Build and authenticate the ordered collection using `scripts/main_paper_figures.py build` and `display-target`. Run `verify_main_paper_collection.cjs`, compare standalone/embedded renderings, inspect print pages, then open the exact current HTML in the built-in browser. Do not publish a freeze.

## Selected authority at interruption

- Figure 1: state SHA-256 `7954b59179d576bd7f56803504e09474c60073d63822c1cedf318b7bb55de50d`; selected SVG `/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-01-v16-editing.svg`, SHA-256 `e0f2db1f613c1cd149db5697f152054d570ea6dfa33d4efb10a27cc4648727bf`.
- Figure 2: state SHA-256 `7a40978f11e3e869e46f15fa556526f0b225775f7ff0cffb7c886228e5e27d3e`; selected SVG `/Volumes/JOAQUIM/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/reviews/figure2-current-minimal.svg`, SHA-256 `7decdf68b414f4a34cc6f01a345ec5a1d7381768b86787cdf6aeca73379903d3`.

## Implementation hashes

- `scripts/nature_main_figures.py`: `22a8e09f15630fae3bc41c601371589dedbd45b1e5e8e98608079e2db7d8c56e`.
- `scripts/style_axis_geometry.cjs`: `0336c6be16446dbfa83dcda3f3c565cc650f4268ce8a51d5b4ed8bb9c6d59620`.
- `scripts/align_paper_annotations.cjs`: `860dd114ca7ebf30cf459d484ec51cb0b890aa32f86c6349f51f69ff6a607173`.
- `scripts/check_nature_presentation.cjs`: `5c05826402f0d0f978e68457610b643270552e768572509531ac9b3f38132735`.
- `scripts/main_paper_figures.py`: `8582329211ccae862fee7922071ac30f6ee34d72a35f10414ed13395c8ef012b`.
- `scripts/register_figure_axes.py`: `9ac47dc0d709dfaa92effe837eca2365262be5c2992bdb9a198d68d1286d857a`.
- `src/classical_conditioning/figure_workspace.py`: `f3a23ca320b116517d7b50a66a390d3debd7c0d5f9f176b8c4036687e5048d4c`.
- `src/classical_conditioning/figures/layout_frames_browser.js`: `ad2d9fb1ba817f6de6002162cac0c2b927962182f95cdee3f39f9d4b761ecbb9`.
- `tests/test_nature_main_figures.py`: `e95617a529c5db001449347bf774290237607c7410b4ed0b8b40bef4d9be2e9a`.
- `tests/check_non_scaling_strokes.cjs`: `fc0ed4b667ea10f5af289f26bbcb40a37384c4a97cbd22ecf0de4c6c986c9f7e`.

## Author spacing correction, 2026-10-17

The author rejected the excessive separation in the 183-mm candidate and selected
“Compact 136 mm where feasible”. Official Nature final-submission guidance was
rechecked: 89/183 mm standard, 120–136 mm conditional; the production guide asks
for the smallest appropriate size and minimal whitespace. Native-width rendering
and compact composition are being verified. Do not publish the earlier spread-out
candidate. Preserve physical role sizes, scientific aspect ratios and source locks.

## Guarded publication and subsequent author steering

The compact 136-mm drafts were published successfully before the author's next
message. Figure 1: 136 × 163.69779259402097 mm; Figure 2: 136 ×
156.46844886241252 mm. Both exact painted/unclipped frame reports, physical-axis
checks, annotation checks and Nature style checks passed. Source paths, rasters,
data values and native aspect ratios remained protected; 54 targeted tests passed.
The collection was rebuilt with HTML SHA-256
`3c1de5fc9157a335406ab060e7c96c43e5f0587fc5018614126e07af1e767010`.
Browser/responsive checks and two-page print rendering passed visual review.
Standalone/embedded computed styles match; screenshot pixel differences reflect
subpixel capture origins, and the geometry comparison still needs hidden zero-box
nodes excluded before claiming a complete numerical comparison. No whole-figure
scientific freeze or submission readiness was established.

The author then requested a black-stroked box marking the desired Nature area,
no shrinking, and permission to retain the original eLife canvas width/height.
Clarification is pending on restoring original panel sizes versus enlarging only
the canvas. Do not treat the compact composition as the final approved layout.
Create an explicitly annotated review guide, preserving originals and current
state; do not confuse its visible black guide with the invisible-frame policy.
