# Figure element identities, shared style and freeze contract

Version **1.0.0**, author instruction recorded **2026-10-09**. The
[machine-readable specification](../../../configs/paper-figures/figure-elements.json)
is authoritative for role identifiers and numerical style defaults. This file
explains their meaning and records the inspected Figure 1-2 sources. The root
[repository instructions](../../../AGENTS.md) apply this contract whenever the
author requests a **panel or whole-figure freeze**.

These defaults were selected for future review/freeze candidates. This change
adds documentation, configuration and standing instructions; it does not load
the JSON into renderers, add a CLI gate, restyle existing figures, change their
scientific calculations, or replace historical freezes. Future freeze work must
apply and verify the contract explicitly. A configuration reference alone is
not proof of compliance.

## Existing foundations and current source inventory

The existing [theme](../../../src/classical_conditioning/figures/theme.py)
provides DejaVu Sans, physical dimensions, CS/US colors, ticks, spines and export
defaults. Its `FigureTheme`, `apply_theme`, `style_axes`, `condition_color` and
`add_stimulus_window` are reusable foundations. The existing
[exporter](../../../src/classical_conditioning/figures/export.py) provides
`FigureProvenance.artist_mappings`, `assign_axes_semantic_ids`, a sidecar
`artist_registry`, semantic SVG validation and transactional publication.
Extend these facilities when integrating this specification; do not introduce
a competing export format. Current renderers do not all use these facilities.

Related records are the [paper specification](02_PAPER_FIGURE_SPECIFICATION.md),
[figure reporting plan](../../../Plans/08_FIGURES_AND_REPRODUCIBLE_REPORTING.md),
[panel comments](../../../Plans/PANEL_REVIEW_COMMENTS.md) and
[legacy style configuration](../../../legacy/helpers/plotting_style.py).

The following inventory is a **2026-10-09 inspection snapshot**, not a new
selection or scientific approval. Resolve the current scoped selection again
at each freeze, rather than choosing the newest filename or copying an older
assembly's appearance. Other review variants can coexist without superseding
the registered primary selection.

| Figure/panels and inspected revision | Source identity and observed content |
| --- | --- |
| Fig1 assembly, last written 2026-10-06 | [Layout](../../../configs/paper-figures/figure1-assembly.json); `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/figure1-preview.svg` and `.png`. Assembly retains older F-H sources and G fish `20230307_12`. |
| Fig1 A-C | Frozen `frozen/2026-10-05-ABCD/Fig1_PanelA_frozen.svg`, `Fig1_PanelB_frozen.svg`, `Fig1_PanelC_frozen.svg` beneath the Fig1 storage root. A is authored apparatus/illumination artwork; B/C are direct SVG schematics, not Matplotlib artists. CS bars/symbols, violet US symbols, condition keys, timing, phase cards, legends and labels recur. |
| Fig1 D | Same frozen folder, `Fig1_PanelD_frozen.svg`; [freeze record](../../../configs/paper-figures/figure1-freeze.json) binds its hash. Five measured tail-angle traces, CS boundaries, recorded US where present, phase text, axes/ticks and labels. Renderer evidence: [trace builder](../../../scripts/build_figure1_trace_panels.py). |
| Fig1 E, assembly v4 | `traces/Fig1_PanelE_RawVigor_HeatmapOverlay_allTrials_v4.svg` beneath the Fig1 storage root. Raw vigor plus exact stored 0.5-s heatmap-bin rectangles on a secondary axis; orange zero reference, event guides and clipping note. Evidence: [overlay builder](../../../scripts/build_figure1_vigor_overlay.py). This is not authenticated by the newer F-H selection. |
| Fig1 F-H, inspected legacy layout | [Scoped current-analysis record](../../../configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json) retains the C/D and direct-mean C sources in the [legacy layout review](../../../reviews/fgh_legacy_layout_20261009/README.md). Its `render_layout.py` produces `FGH_C_BoutSamples_legacy_layout.svg`, `FGH_D_BoutSamples_legacy_layout.svg`, `FGH_C_DirectBins_legacy_layout.svg` and the earlier `FGH_Version4_DirectBinMedians_legacy_layout.svg`. Primary registered variant is `C_BoutSamples`; G is `20230310_08`. Continuous trial axes, boxed heatmaps, one shared colorbar, phase labels left of F, and five example arrowheads at G trials 9/17/63/66/93. |
| Fig1 F-H Version 4, selection updated during this inspection | The scoped record now binds Version 4 to [baseline-20 data manifest](../../../reviews/fgh_version4_baseline20_20261009/data_manifest.json) and `FGH_Version4_DirectBinMedians_legacy_layout.svg` in that folder. This separate scientific revision uses [-20,0) bin-median references, includes valid non-bout contributions as negative infinity, and leaves trials undefined when the reference is not finite. Negative-infinite display values with finite references use the palette's negative endpoint; NaNs remain missing. The registered primary C-bout variant is unchanged. This additional source/manifest inspection does not replace the earlier visual inspection or constitute new approval. |
| Fig2 assembly, last written 2026-10-08 | [Layout](../../../configs/paper-figures/figure2-assembly.json); `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/figure2-main.svg` and `.png`. A/B heatmaps, older D/E block plots, G/H trajectories; C/F/I are placeholders for unavailable authenticated 10sTrace panels. Placeholder text/borders are presentation elements, not scientific results. |
| Fig2 A/B, population heatmaps | Sources under `figure2-assembly/sources/20261008T200908002798Z/`: `Fig2_PanelA_allDelay_signed-pre15.svg`, `Fig2_PanelB_all3sTrace_signed-pre15.svg`. Equal-fish pooled signed log-vigor bins, condition headings, CS guides, phase separators, axes and colorbars. Evidence: [population review builder](../../../scripts/populate_figure2_available.py). |
| Fig2 D/E, newer author-frozen style | `figure2-assembly/row2-block-ratio-review/style-freeze.json` and `analysis-selection.json` select `style-baseline-freeze/20261009T110531835455Z/Fig2_PanelD_reference-style.figure.json` and `Fig2_PanelE_reference-style.figure.json`. Adjacent SVG/PNGs show conditioned fish left, control right, paired trajectories, connected medians/fish IQR, PT/ET/LT, shared left axis, straight statistical lines and opaque black ratio-one references. These newer appearances are not in the inspected full assembly. |
| Fig2 G/H, trajectories and uncertainty reviews | The assembly uses earlier descriptive trajectories; separate row3 reviews exist. Inventory condition curves, bands, ratio-one reference, phase boundaries, global-trial ticks and legends. Determine each band's actual estimator/method from its selected sidecar; the separate [Delay review](FIGURE2_DELAY_LME_CRITIQUE_2026-10-09.md) must not be conflated with the assembly. |

All abbreviated `figure2-assembly/...` paths in this inventory are beneath
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/`. Exact source/export
hashes are in the existing selection records and sidecars; a new freeze must
verify the selected actual bytes, not reuse this inventory as a manifest.

### Observed differences and semantic coverage

- Typography: the theme uses 8-pt labels/titles and 7-pt ticks/legends. Frozen
  D/E source plots use headings 25 pt, y labels 23 pt, categorical x labels
  22 pt and y ticks 19 pt. Newer F-H sources use 8-pt axis/colorbar labels,
  7-pt ticks/fish IDs, 10-pt condition headings and 14-pt panel letters.
  These are source sizes; assembly scaling changes their final sizes.
- CS guides: newer F-H uses `#0d8136`, 2.4 pt, alpha 0.8, both boundaries
  solid. The Fig2 heatmap builder uses `#168241`, 0.65 pt. Fig1 B/C schematic
  builders use CS `#0d7f3c` and US `#78358c`; the shared theme uses CS
  `#0d8136` and US `#702e78`. Scientific events need independent identities
  despite similar colors.
- US guides: newer F-H uses violet `#964bad`, 0.6 pt, dotted, only over
  training rows at the selected assay time. A protocol guide is not evidence
  of a measured US; authenticate its event source before assigning the role.
- Spines/ticks: trace plots use left/bottom spines; newer F-H uses four
  0.6-pt gray spines, 2.5-pt outward ticks and -20/0/20-s labels. D/E uses
  one 1.4-pt left spine, no right-subpanel spines, and categorical labels with
  no x tick marks. The proposed shared system retains these plot-family
  structures while standardizing typography and thickness.
- Transparency/stacking: D/E has fish line RGBA alpha 0.37, point face alpha
  0.30, edge alpha 0.40, and black median/IQR alpha 0.72. Its ratio-one
  reference is black, 0.85 pt, alpha 1, zorder 0. Fig1 E rectangles have
  translucent orange fills and opaque borders; they sit behind raw traces.
- Scientific scales: F-H C/D sample and direct-mean C displays use +/-1;
  direct-bin-median Version 4 uses +/-0.25. Fig2 A/B pooled signed values
  use +/-0.25. Sharing `managua_r` does not make these measures equivalent.
  Missing heatmap values are black in these reviews; theme defaults instead
  use gray. Neither missingness nor a black cell implies zero.
- IDs: inspected F-H C and Version 4 SVGs have 17 explicitly named scientific
  groups each, including `CS_G_0`, `frozen_samples_G`, phase labels and example
  arrows, but no `axes__`, `axis__`, `tick__` or `spine__` registry IDs. The
  selected D/E E SVG names its two `baseline-one` references, but its sidecar
  has no `artist_registry`. The shared exporter currently identifies ticks by
  index and does not capture a complete resolved style for every artist.
- SVG composition already prefixes source IDs to prevent collisions and
  normalizes font families. Prefixing a generic `line2d_7` does not identify
  its scientific role. Preserve source-to-assembled ID mappings at freeze.

## Scientific role catalog

Role identifiers below are exact keys in the machine specification. Classes
are common implementations; a handmade schematic records its real `SVG.*`
type rather than pretending to be a Matplotlib artist. Not every panel uses
every role.

| Role identifiers | Matplotlib/SVG representation and scientific metadata |
| --- | --- |
| `stimulus.cs.onset`, `stimulus.cs.offset` | `Line2D` / SVG line/path; event identity, timing source, alignment and extent. |
| `stimulus.us.actual`, `stimulus.us.expected` | Recorded versus protocol-expected US; retain event source and trial scope. |
| `stimulus.window`, `stimulus.symbol` | Rectangle/polygon or schematic bar/circle; CS/US interval, pulse duration, short/long US or CS-only catch. |
| `baseline.window` | Span/rectangle; baseline bounds and alignment from selected analysis. |
| `reference.time.zero`, `reference.signal.zero`, `reference.response_ratio.equal_baseline` | `Line2D`; axis quantity and reference definition. Ratio equality is y=1; time zero is commonly x=0. |
| `trace.tail_angle`, `trace.raw_vigor`, `trace.heatmap_bin_overlay` | Measured angle/activity lines or exact saved-bin patches; metric, units, transformations, fish/trial and source fields. |
| `trajectory.fish`, `observation.fish` | `Line2D` / `PathCollection`; biological pairing, condition and trial/block. |
| `summary.curve`, `summary.median` | Summary line/markers; estimator, aggregation order and population. |
| `uncertainty.iqr`, `uncertainty.confidence_interval` | Stems/caps or bands; IQR is Q25-Q75, CI carries level/method/resampling unit and pointwise/simultaneous status. |
| `heatmap.vigor`, `heatmap.coverage`, `colorbar.scale` | `PatchCollection`, `QuadMesh`, `AxesImage` and colorbar axes; measure, normalization, limits, missingness and mappable links. |
| `phase.separator.heatmap`, `phase.boundary`, `phase.label` | Lines/text; named session transitions and exact trial boundary. |
| `annotation.example_trial` | Polygon/arrow/annotation; selected fish/trial, selection source and provisional/approved status. |
| `axes.container`, `axis.component` | Axes, x/y axis or SVG group; data/colorbar context, quantity, units and transform. |
| `axis.tick`, `axis.tick_label`, `axis.label`, `axis.spine` | Tick line/text, axis title, spine; dimension, side, major/minor kind, value/category and visibility. |
| `title.panel`, `title.condition`, `label.panel_letter`, `label.fish_id` | Text roles independent of scientific data; condition headings retain condition identity. |
| `legend.key` | Legend and member glyphs; links to the represented roles/series. |
| `annotation.statistical_comparison`, `annotation.statistical_label` | Lines/`ConnectionPatch` and text; exact contrast, saved result, test/correction and scientific status. |
| `annotation.note`, `schematic.preparation` | Notes and authored apparatus/session geometry, including provisional status and clipping. |

### Element records and existing API compatibility

Use `{figure}__{panel}__{subpanel}__{role-slug}[__{semantic-instance}]` for new
instance IDs. Include condition/fish/trial or tick dimension/side/kind/value
where needed. For example:

```text
fig1__g__trace__stimulus-cs-onset
fig2__e__trace__reference-response-ratio-one
fig1__g__trace__axis-tick-x-bottom-major-zero
```

The JSON contains complete example records. Required record fields are
`element_id`, `scientific_role`, `artist_type`, `figure_id`, `panel_id`,
`subpanel_id`, `scientific_context`, `coordinate_system`, `geometry`,
`style_role`, `resolved_style` and `required_in_svg`. Structural elements use
explicit not-applicable scientific fields; do not invent a metric or fish.

Keep historical gids as aliases and preserve the existing sidecar metadata.
When integrating, add these fields to existing `artist_registry` entries.
Convert the contract's boolean `required_in_svg` to the current exporter's
`"true"`/`"false"` strings at that boundary until its API is migrated. Shared
structural entries need actual `artist_type` values too; registration cannot
be limited to artists already carrying a gid. ID assignment must be idempotent
and must not silently replace a caller's scientific identity.

Give each composite summary a `composite_id`; its median line/marker and IQR
stems/caps have separate instance IDs, constituent roles and `composite_part`
values. Colorbar components similarly link to their `mappable_ids`. Register
legend glyphs and labels under the legend group and underlying scientific role.
Hidden artists retain metadata with visibility and export requirement false;
rendered scientific artists require SVG presence.

Coordinate metadata must distinguish data, axes/figure fractions, blended
transforms and SVG user space. An `axvline` usually combines data x and axes
fraction y. Schematic positions must not be treated as measured seconds unless
the schematic explicitly defines that mapping. One artist can represent both
CS onset and time zero using additional `scientific_roles`; apply the specific
stimulus style and do not draw duplicate coincident lines.

Record style values explicitly: font family/style/weight/size, line width and
pattern, color, marker geometry where applicable, visibility, clipping,
padding, zorder and effective opacity. Inspect both artist alpha and embedded
RGBA alpha; Matplotlib can override rather than multiply those values. Record
the actual exported fill/stroke opacity. Null artist alpha does not itself
mean transparency.

## Uniform style and scientific boundaries

All typography uses DejaVu Sans, normal style/weight unless specified below.
Sizes are **effective final assembled sizes**, in typographic points.

| Style | Shared default |
| --- | --- |
| Labels/titles | 8 pt; label padding 3 pt, title padding 4 pt |
| Tick/legend text | 7 pt; legends without frames |
| Panel letters | 10 pt, bold |
| Spines/ticks | 0.5 pt; outward ticks length 2 pt, padding 3 pt |
| CS/US guides | 0.7 pt, alpha 1; CS `#0d8136`, US `#702e78` |
| Guide patterns | Solid CS onset/actual US; dashed CS offset; dotted expected US |
| Zero/one reference | Black, 0.6 pt, alpha 1, zorder 0, behind data |
| Measured traces | Black, 0.6 pt, alpha 1 |
| Fish trajectories/points | Condition colors; trajectories 0.6 pt; alpha 0.3 |
| Summary curves | 1.2 pt; black median/IQR summaries alpha 0.72 |
| Windows/uncertainty bands | Alpha 0.18/0.2; no visible fill edges |
| Heatmaps/colorbars | `managua_r`, black missing values, alpha 1; colorbar outline 0.5 pt |
| Heatmap phase separators | White, 0.8 pt, alpha 1 |
| Statistical comparison lines/text | 0.7 pt / 7 pt; straight lines without end caps |

Condition colors come from `ConditionSpec.color_rgb_255` through
`condition_color`, not a separately invented palette. Existing manually chosen
condition colors are inspection evidence; carry them forward only with scoped
approval if they differ. Shapes and marker sizes not fixed by these defaults
must be recorded in resolved styles and made legible in the candidate.

Use a provisional final double-column width of **183 mm**. Record the actual
source-to-final transforms. For uniform scaling by k, source fonts/strokes need
to be target/k to achieve the target sizes. SVG viewBox, nested transforms,
unit conversion and any non-scaling strokes must be included. Resolve
nonuniform scaling before freeze; a correct source fontsize is insufficient.
Inspect individual panel and whole-figure views for clipping and legibility.

Structural plot-family rules are declared in JSON: continuous traces/trial
ratios have left/bottom spines; heatmaps retain four boxed spines with repeated
labels suppressed on shared axes; paired block plots have a shared left axis
on the conditioned subpanel and no spines/y labels on the control subpanel.
Block labels remain visible without x tick marks. Data limits, tick locations,
phase boundaries and PT/ET/LT block definitions remain scientific inputs.

Do not standardize scientific transformations through style:

- C/D scaled vigor uses its recorded quantile formula and +/-1 limits.
- Signed-log vigor/direct-bin-median displays use their own recorded
  transformation and +/-0.25 limits; record natural-log reference units.
  The newer Version 4 baseline-20 revision retains negative-infinite non-bout
  values as lower-endpoint display values when its baseline reference is
  finite; keep this scientific rule distinct from NaN missingness. An
  undefined reference remains undefined, not an alternative zero reference.
- Pooled heatmaps record equal-fish aggregation separately from single-fish
  values. Coverage has its own numerator/denominator and scale.
- Angle traces explicitly record degrees or radians; raw vigor records rad/ms
  and the actual metric. Ratios are dimensionless even if an older label says AU.
- Missing values remain distinct from zero. Shared colorbars require identical
  measure, units, normalization and limits.
- Event timing, baseline bounds, summary estimators, CI methods, cohorts and
  statistical decisions are scientific definitions, never style exceptions.

## Freeze workflow and scoped exceptions

For **every panel or whole-figure freeze**, the agent must identify the selected
sources, assign/verify element records, apply the shared styles to a separate
candidate, and inspect it at its intended final scale. Verify source/data
hashes and retain historical frozen artifacts. Complete independent review
work before asking about a decision that remains unresolved.

Ask only about an unresolved exception: a previously selected appearance
conflicting with defaults, stimulus visibility, ambiguous event identity, or a
necessary departure to avoid illegible/clipped content. Explicit prior author
approval remains valid within scope and is not re-asked. If the user has
explicitly replaced that earlier choice, follow the newer instruction. A
renderer value or inspected screenshot alone is not approval.

Each approved exception records `exception_id`, `scientific_role`, `property`,
`value`, `reason`, `scope` and `approval_evidence`. Scope identifies figure,
panel/subpanel/element and applicable revision or future use. Cite the decision
or author instruction and date; resolve values into each affected element's
style. Do not propagate a Fig2 D/E exception to unrelated panels. Scientific
ambiguities require a scientific decision record, not a presentation waiver.
In particular, distinguish D/E's already approved black, alpha-1 reference
from the proposed thinner default; preserve any explicitly approved scoped
thickness until the author supersedes it.

A freeze manifest follows the JSON `freeze_record_contract`: selected IDs and
selection record; specification ID/version/path/SHA-256; final width and
source-to-final transforms; element registry/resolved styles; approved
exceptions; source/data paths and SHA-256; final export paths and SHA-256;
verification evidence; and freeze authorization. Hash actual final bytes.
Store final SVG hashes in the separate freeze manifest to avoid circular
embedded-sidecar hashes. Preserve the current exporter's integrity convention.

The author's request to freeze supplies authorization; no redundant permission
question is required after exceptions are resolved. If a required decision is
pending, do not finalize the freeze. Noninteractive command-line workflows
must report the unresolved exceptions with failure status, rather than guessing
approval. These are standing requirements for future integration/workflows,
not a claim that the current CLI already enforces them.

### Acceptance checklist

- Inventory the exact source revision and all meaningful visible roles,
  including schematic, secondary-axis, colorbar and statistical components.
- Match inventory roles to JSON keys; require unique IDs and complete
  scientific mappings, tick values and composite membership.
- Check effective final typography, widths, opacity, stacking, shared axes,
  missingness, clipping and colorbar meaning against the defaults/exceptions.
- Validate SVG presence of required semantic IDs and its sidecar mappings;
  preserve source-to-assembly identity links.
- Record only author-approved scoped exceptions; keep scientific definitions
  and scientific approval status separate from presentation compliance.
- Verify unchanged data/source hashes for presentation-only work, validate
  JSON and documentation links, and preserve existing freezes.
