# Nature paper-figure target

Selected by the author on 2026-10-16 for the next paper-figure revision.
This is the maintained presentation target, not a passing compliance report.
Read-only profile resolution and an explicit shared-theme adapter are implemented.
A scoped Figure 1/2 migration route now measures typography, axes, strokes, frames
and related-panel gaps through the existing guards; see the workflow. Remaining
consumers, complete scientific identities, accessibility and production checks
remain pending.
Showing selected artwork does not trigger migration. The
[iteration workflow](iteration-workflow.md) owns procedures and publication;
[Plan 07](../../../../Plans/07_FIGURE_BUILD_TOOLS.md#5-nature-main-figure-presentation-profile)
owns remaining implementation; the
[source review](../../../../records/figure-reviews/nature-guidance-20261016.md)
owns dated source comparisons and audit findings.

Author direction, 2026-10-17: pre-Nature styling is retired as a target for new
paper work. Existing unprofiled renderers and immutable contracts remain
compatibility/reproduction dependencies, not another supported paper style.
The [retirement record](../../../../records/repository-archive/2026-10-17-retired-styling/README.md)
indexes verified external snapshots of the superseded guidance. Use this target
and its resolved role values for new work; change a paper-wide styling choice
through an explicit recorded profile revision.

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

## Coding entry point and exact values

The [versioned profile](../../../../configs/paper-figures/nature-main.json)
owns exact overrides and per-rule origins. It resolves the existing semantic
contract and active layout policy; it does not copy their role registry or
scientific identities. Use this small lookup rather than re-reading plans,
copying numbers into a renderer or introducing another theme/exporter:

```bash
.venv/bin/python -m classical_conditioning.figures.presentation_profiles
.venv/bin/python -m classical_conditioning.figures.presentation_profiles --role axis.tick
```

Other sections are `typography`, `colors`, `roles`, `rules` and `theme`.
Use `--section styles` for all resolved style roles, or `--role <canonical-id>`
for one existing element role and its applicable axis-geometry rules.
Output includes all three input hashes and explicitly requires render evidence.
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

The profile separates journal ranges from our exact role sizes, stroke hierarchy,
black text, tick geometry, caption-review styling and author-selected spacing.
The cited Nature guidance does not prescribe universal tick direction/count,
glyph gaps, marker diameter or right-key alignment. The active frame inset is
read from `layout-frames.json`, not copied into another competing policy.
For routine edits, use the recorded version and query the relevant section.
Recheck official guidance when the target journal/stage changes or before actual
submission; a local role change does not require repeating the full research.

## Axes, ticks and labels

Apply these rules to ordinary X/Y axes and the applicable components of
colorbars, shared axes, secondary/twin axes and inset axes. Schematic coordinates
are identified separately. Register every component and visibility decision.

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

## Authority and implementation boundary

Keep existing semantic role identifiers and state/publication tools. The
machine profile supplies versioned numerical overrides and policy preflight in
that pipeline; full rendered-artifact validation remains to integrate, and it does not create another registry or exporter. Bind profile/rule
origin and implementation hashes into version-bound evidence. Preserve
byte-sensitive historical specifications and frozen contracts; update permitted
consumers to resolve the new profile rather than rewriting accepted history.
Presentation validation of requested edits is distinct from an explicit freeze.

Selected drafts change only after the scoped migration passes verification and
guarded publication. Read their state-bound profile/evidence for the exact coverage.
No whole-figure scientific freeze or submission-readiness claim follows from
this guidance, collection inclusion or a check-only run.
