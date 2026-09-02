# Scientific Figure Pipeline Plan

> **Implementation authority:** Active sequencing and shared architecture are
> defined by `MASTER_ANALYSIS_MIGRATION_PLAN.md` and
> `IMPLEMENTATION_STEP_INDEX.md`.

## Purpose

Build a reproducible and maintainable figure-production system for the larval zebrafish classical-conditioning paper.

The system should:

- Generate scientifically correct figures from explicit data artifacts.
- Apply one visual language across every panel and major figure.
- Reuse a panel in standalone, manuscript, supplementary, and presentation layouts.
- Assemble related analytical plots without losing axis alignment.
- Combine plots, protocol schematics, and images when necessary.
- Allow later interactive layout adjustment without changing scientific content.
- Preserve traceability from a submitted panel back to its data, analysis settings, code, and included fish.
- Export publication-ready SVG and PDF, plus raster formats when required.

This plan complements `../docs/analysis/ANALYSIS_ISSUES.md`. Figure refactoring must not hide or freeze known analytical problems. Scientifically relevant preprocessing and statistical issues should be resolved before final manuscript figures are approved.

## Guiding principles

### 1. Scientific correctness precedes visual refinement

A visually polished panel is not publication-ready unless its:

- Input dataset is identified.
- Fish cohort is explicit.
- Analysis parameters are recorded.
- Units and axis definitions are correct.
- Statistical annotations are generated from saved results.
- Caption can be derived from the actual analysis.

### 2. Every edit type has one authoritative home

| Edit type | Authoritative location |
| --- | --- |
| Raw-data processing | Analysis code |
| Cohort selection | Versioned cohort manifest |
| Statistical analysis | Analysis code and saved result tables |
| Panel-ready aggregation | Data-preparation functions |
| Data visual encoding | Panel rendering functions |
| Fonts, colors, strokes, and dimensions | Shared figure theme |
| Plot-only multi-panel layout | Matplotlib figure specification |
| Mixed-media layout and persistent callouts | Inkscape template or equivalent assembly layer |
| Figure caption and manuscript reference | Paper repository |
| Final submission conversion | Export pipeline |

An edit should not be made manually in the final SVG if it belongs in analysis code, the panel renderer, theme, or layout specification.

### 3. Generated figures are products of versioned sources

The primary sources are:

- Analysis code
- Panel-ready data or immutable references to it
- Statistical result tables
- Figure specifications
- Theme tokens
- Cohort manifest
- Inkscape templates and human-authored annotation layers

SVG, PDF, PNG, and TIFF are generated outputs. A final hand-refined SVG may be archived separately but must never silently replace the reproducible source.

### 4. Use vector graphics by default

- Use SVG as the editable intermediate format.
- Use PDF as the primary manuscript/submission format when accepted.
- Use PNG or TIFF only for previews or journal raster requirements.
- Keep SVG text as text during iteration.
- Convert text to paths only if a journal explicitly requires it and preserve an editable master.

## Scope

### Included

- Existing Matplotlib and Seaborn data figures
- Example-fish traces and heatmaps
- Scaled-vigor panels
- Normalized-vigor panels
- Statistical summary panels
- Protocol and experimental schematics
- Future learner-classification figures after the analysis is validated
- Major and supplementary figure assembly
- Figure-level labels, legends, colorbars, annotations, and exports
- Interactive layout adjustment
- Visual and scientific quality control

### Not included initially

- A custom browser-based drag-and-drop editor
- A general-purpose SVG rewriting engine
- Automatic AI approval of scientific correctness
- Refactoring all figure scripts simultaneously
- Confirmatory learner-classification figures before one canonical analysis is established

## Target architecture

```text
Raw and processed data
          |
          v
Validated analysis and cohort manifest
          |
          v
Panel data-preparation functions
          |
          +--------------------------+
          |                          |
          v                          v
Panel-ready data tables       Statistical result tables
          |                          |
          +-------------+------------+
                        |
                        v
Reusable panel renderers
draw_*(ax, panel_data, style)
                        |
          +-------------+-------------+
          |                           |
          v                           v
Standalone panels          Matplotlib compound figures
          |                           |
          +-------------+-------------+
                        |
                        v
SVG analytical panels or compound figures
                        |
          +-------------+-------------+
          |                           |
          v                           v
Plot-only final figures     Mixed-media Inkscape templates
          |                           |
          +-------------+-------------+
                        |
                        v
Major figure SVG/PDF + raster derivatives
                        |
                        v
Deterministic, accessibility, visual, and scientific QC
```

## Repository responsibilities

### Analysis repository

Location:

```text
C:\Users\Public\More projects\ClassicalConditioning
```

This repository should own:

- Panel data preparation
- Statistical outputs
- Fish/cohort manifests
- Shared plot theme
- Data-driven panel rendering
- Plot-only compound figures
- Standalone panel exports
- Figure provenance records

Suggested structure:

```text
ClassicalConditioning/
|-- figures/
|   |-- __init__.py
|   |-- theme.py
|   |-- specs.py
|   |-- registry.py
|   |-- compose.py
|   |-- export.py
|   |-- provenance.py
|   |-- qc.py
|   |-- panels/
|   |   |-- protocol.py
|   |   |-- example_trace.py
|   |   |-- heatmap.py
|   |   |-- scaled_vigor.py
|   |   `-- normalized_vigor.py
|   `-- compound/
|       |-- delay_summary.py
|       |-- trace_summary.py
|       `-- comparison_summary.py
|-- figure_specs/
|   |-- figure_1.py
|   |-- figure_2.py
|   `-- supplementary_figure_1.py
|-- build/
|   |-- panel_data/
|   |-- statistics/
|   |-- panels/
|   |-- compound/
|   |-- provenance/
|   `-- qc/
|-- plotting_style.py
`-- figure_saving.py
```

### Paper repository

Location:

```text
C:\Users\Public\More projects\Paper\Learning paper
```

This repository should own:

- Major-figure layout involving heterogeneous media
- Experimental schematics and illustrations
- Persistent callouts and cross-panel annotations
- Figure labels when applied at the manuscript assembly level
- Captions and manuscript references
- Submission-ready exports

Suggested structure:

```text
Learning paper/
`-- figures/
    |-- templates/
    |   |-- Figure_1.template.svg
    |   `-- Figure_2.template.svg
    |-- schematics/
    |-- generated-panels/
    |-- masters/
    |   |-- Figure_1.generated.svg
    |   `-- Figure_2.generated.svg
    |-- final/
    |   |-- Figure_1.pdf
    |   `-- Figure_2.pdf
    `-- qc/
```

Generated outputs may be excluded from Git if large, but final archival figures and their provenance manifests should be retained in a defined release or manuscript snapshot.

## Workstreams

## Workstream A: Resolve scientific foundations

### Objective

Ensure that figure infrastructure is built on scientifically valid and explicitly defined outcomes.

### Tasks

1. Review and prioritize all critical items in `../docs/analysis/ANALYSIS_ISSUES.md`.
2. Define the canonical mathematical specification for:
   - Tail-movement vigor
   - Spatial and temporal filtering
   - Bout detection
   - Trial alignment
   - Baseline windows
   - Conditioned-response windows
   - Scaled vigor
   - Normalized vigor
   - Missing-data treatment
3. Resolve fish-exclusion behavior across every pipeline stage.
4. Create a fish-level cohort manifest containing:
   - Fish ID
   - Experimental condition
   - Source files
   - Technical validity
   - Behavioral inclusion status
   - Exclusion reason
   - Analysis cohort name
5. Add synthetic or small reference tests for foundational calculations.
6. Reprocess raw data after the canonical definitions are implemented.
7. Produce canonical panel-ready tables and statistical result tables.

### Deliverables

- Validated analysis specification
- Tested preprocessing functions
- Cohort manifest
- Corrected processed datasets
- Saved statistical results
- Dataset and result provenance metadata

### Exit criteria

- Code and manuscript describe the same calculations.
- Expected fish counts are asserted and reproduced.
- Foundational metric tests pass.
- No final figure relies on undocumented manual data selection.

## Workstream B: Establish the figure design system

### Objective

Create one source of truth for all cross-panel visual properties.

### Tasks

1. Extend the existing `PlotStyleConfig` or introduce an immutable `FigureTheme`.
2. Define physical publication dimensions:
   - Single-column width
   - Double-column width
   - Maximum figure height
   - Standard panel gaps
   - Outer margins
3. Define typography:
   - Font family
   - Axis-label size
   - Tick-label size
   - Legend size
   - Panel-label size and weight
   - Mathematical text policy
4. Define line and marker roles:
   - Axis stroke
   - Data stroke
   - Confidence-band edge
   - Statistical bracket
   - Baseline/reference line
   - Marker sizes
5. Define semantic colors:
   - Every experimental condition
   - CS and US
   - Baseline/reference
   - Excluded/missing data
   - Statistical annotations
6. Define standard axis behavior:
   - Spine visibility
   - Tick direction
   - Tick length
   - Label padding
   - Grid policy
7. Define panel-label placement relative to panel bounds.
8. Define export settings for SVG, PDF, PNG, and TIFF.
9. Expose the theme as Python and optionally serialize it to JSON for paper-repository tooling.

### Deliverables

- `figures/theme.py`
- Theme documentation and a visual style specimen
- Named semantic palette
- Standard size conversion helpers
- Shared axis, legend, and panel-label helpers

### Exit criteria

- A single theme change updates all migrated panels.
- Condition colors are identical across scripts.
- Test exports are legible at final physical size.
- No migrated renderer defines its own font family or arbitrary palette.

## Workstream C: Separate analysis, rendering, and saving

### Objective

Turn monolithic plotting routines into reusable panel components.

### Panel interface

Each migrated panel should use:

```python
def prepare_<panel_name>_data(
    source_data,
    analysis_config,
) -> PanelData:
    ...
```

```python
def draw_<panel_name>(
    ax,
    panel_data,
    style,
) -> PanelArtists:
    ...
```

The drawing function must:

- Accept an existing `Axes` or `SubplotSpec`.
- Avoid loading files.
- Avoid analytical aggregation not represented in `PanelData`.
- Avoid creating the major figure.
- Avoid saving or closing figures.
- Avoid deciding its panel letter.
- Return legend handles, labels, and optional colorbar mappables.

### Migration order

Migrate one representative panel from each plot family:

1. Example trace
2. Heatmap
3. Time-resolved condition line plot
4. Normalized-vigor block summary
5. Statistical boxplot or point summary
6. Protocol schematic

After the pattern is validated, migrate remaining panels incrementally.

### Backward-compatible wrappers

Maintain standalone generation:

```python
def save_<panel_name>_standalone(panel_data, output_path):
    fig, ax = ...
    artists = draw_<panel_name>(ax, panel_data, style)
    ...
    save_publication_figure(fig, output_path)
```

### Deliverables

- Panel-ready data classes or typed mappings
- Reusable renderers
- Standalone wrappers
- Reference exports for comparison

### Exit criteria

- The same renderer can be used standalone and in a major figure.
- Rendering does not change cohort membership or statistics.
- Old and refactored figures are compared and intentional differences documented.
- No data-driven annotation is manually recreated in the assembly layer.

## Workstream D: Build major-figure composition

### Objective

Assemble consistent major figures while preserving analytical relationships.

### Plot-only figures

Use Matplotlib `subplot_mosaic` or nested `GridSpec` when panels require:

- Shared axes
- Identical plot dimensions
- Shared limits
- Shared legends
- Shared colorbars
- Direct visual comparison

Define typed figure specifications:

```python
FIGURE_2 = MajorFigureSpec(
    figure_id="figure_2",
    width_mm=180,
    height_mm=125,
    mosaic=[
        ["protocol", "protocol", "trace"],
        ["heatmap", "summary", "summary"],
    ],
    width_ratios=[1.0, 1.0, 1.2],
    height_ratios=[0.9, 1.0],
    panels={...},
)
```

The composer should control:

- Canvas size
- Panel positions
- Row and column ratios
- Gaps and margins
- Shared legends and colorbars
- Panel labels
- Figure-level annotations

### Mixed-media figures

Use an Inkscape master template when combining:

- Protocol or apparatus schematics
- Data plots
- Microscopy or raster images
- Anatomical illustrations
- Persistent arrows and cross-panel callouts

Start with direct Inkscape templates. Evaluate `figurefirst` only through a limited proof of concept because of its maintenance and compatibility risk.

Use `svgutils` only for:

- Simple fixed-grid SVG assembly
- Legacy standalone SVGs
- Panels that do not require mathematically aligned axes

### Deliverables

- Figure composer
- Typed figure specifications
- One plot-only major figure
- One mixed-media template
- Shared panel-label and legend utilities

### Exit criteria

- Major figures regenerate without manually moving analytical panels.
- Shared axes and legends are correct.
- Panel labels are controlled only by the composer or paper template.
- Updating one panel does not require rebuilding its layout manually.

## Workstream E: Add interactive adjustment

### Objective

Allow rapid visual layout refinement while retaining reproducibility.

### Stage 1: Interactive Matplotlib preview

Use Jupyter or VS Code with:

```python
%matplotlib widget
```

Support inspection, zooming, and axis-limit selection.

### Stage 2: Configuration-driven widgets

Add `ipywidgets` controls for presentation-only properties:

- Figure width and height
- Row and column ratios
- Horizontal and vertical gaps
- Margins
- Axis limits
- Tick positions
- Legend location
- Panel-label offsets
- Title visibility

Accepted values must be copied back into the versioned figure specification.

### Stage 3: Inkscape adjustment

Use Inkscape for mixed-media layout and persistent annotations:

- One layer or group per panel
- Separate layers for labels, annotations, and legends
- Stable IDs such as `panel_A`, `panel_B`
- Relative links during local iteration when reliable
- Embedded final master for portability

### Guardrails

Interactive tools may change:

- Panel placement
- Dimensions
- Spacing
- Label placement
- Legend placement
- Presentation-only axis visibility

They may not change:

- Data points
- Cohort membership
- Statistical results
- Error bars
- Significance values
- Analysis windows
- Condition mappings

### Deliverables

- Interactive preview notebook
- Persisted layout specifications
- Inkscape template conventions
- Manual-edit policy

### Exit criteria

- Layout changes are reproducible after restarting the environment.
- No accepted change exists only in transient widget state.
- Manual final edits are clearly separated from generated artwork.

## Workstream F: Build a safe SVG layer

### Objective

Use SVG structure for validation and constrained refinement without making arbitrary XML rewriting the primary styling mechanism.

### Tasks

1. Preserve text as text:

```python
plt.rcParams["svg.fonttype"] = "none"
```

2. Assign semantic `gid` values to important artists where practical:

```python
line.set_gid("condition-delay")
xlabel.set_gid("axis-label-x")
```

3. Add an SVG audit tool that reports:
   - Document dimensions and `viewBox`
   - Fonts in use
   - Font sizes
   - Colors and opacities
   - Raster image references
   - External links
   - Missing panel IDs
   - Invalid XML
4. Permit constrained normalization of:
   - Known font roles
   - Known panel-label roles
   - Document metadata
   - Explicit semantic classes or IDs
5. Do not blindly replace all fills, strokes, paths, or text.
6. Record manual SVG changes separately and port recurring changes back to code or templates.

### Semantic SVG export contract

Publication SVGs must be machine-interpretable without relying only on
Matplotlib's generated element order.

#### Stable semantic IDs

Assign unique `gid` values before saving:

```text
figure__figure-2
panel__A
axes__A__main
axis__A__x
axis-title__A__x
tick__A__x__000
tick-label__A__x__000
spine__A__left
series__A__condition-delay__median
interval__A__condition-delay__ci95
stimulus__A__cs
legend__A__main
annotation__A__planned-contrast-1
```

At minimum, every panel, axes, axis title, tick label, spine, legend, data
series, uncertainty component, stimulus mark, and statistical annotation must
either have an explicit semantic ID or appear in the semantic sidecar as an
intentionally unclassified element.

#### Embedded reproduction metadata

Add an inert JSON payload inside the SVG `<metadata>` element. It contains:

```json
{
  "figure_id": "Figure_2",
  "figure_spec_version": "1.0",
  "analysis_recipe": "corrected-paper-v2",
  "source_commit": "commit hash",
  "source_file": "src/classical_conditioning/figures/figure_2.py",
  "source_symbol": "build_figure_2",
  "source_hash": "sha256",
  "reproduction_snippet": "build_figure('Figure_2', recipe='corrected-paper-v2', mode='publication')",
  "input_artifacts": [],
  "cohort_hash": "sha256",
  "artist_registry": {}
}
```

The snippet is a concise exact invocation, not a copy of the complete source
file and not executable JavaScript. Raw observations are never embedded in SVG
metadata.

#### Sidecar remains authoritative

Write `<figure-name>.figure.json` beside the SVG. It contains the complete
provenance, panel specifications, data-field mappings, and artist registry.
The SVG embeds a compact subset plus the sidecar hash. This protects software
interpretation from renderer-specific XML changes and allows PDF/PNG exports to
share the same manifest.

#### Structural validation

Validate:

- unique IDs;
- every declared artist ID exists in the SVG;
- every important SVG group maps to one semantic component;
- axes, tick, data-series, legend, and annotation relationships;
- no external scripts, fonts, images, or CDN resources;
- valid XML and internal references;
- embedded metadata and sidecar hashes agree;
- a no-op parse/write cycle preserves the rendered figure.

### Deliverables

- SVG auditor
- Optional constrained normalizer
- Semantic-ID conventions
- Embedded reproduction metadata schema
- Per-figure semantic artist registry and sidecar JSON
- SVG manual-edit policy

### Exit criteria

- The audit can identify style drift without modifying scientific marks.
- Normalization only targets explicit roles.
- Generated SVG remains editable and standards-compliant.
- Software can map panels, axes, ticks, series, intervals, stimuli, legends,
  and annotations to their generating specification and code invocation.

## Workstream G: Export and provenance

### Objective

Make every final panel traceable and generate all required formats consistently.

### Export policy

For every approved major figure, generate:

- SVG: editable vector master
- PDF: manuscript/submission vector output
- PNG: preview at 300 or 600 DPI
- TIFF: only if required by the target journal

Do not rely on `bbox_inches="tight"` without checking final physical dimensions; it can change canvas size. Prefer explicit margins for final manuscript figures.

### Provenance manifest

Save a JSON record beside every figure:

```json
{
  "figure_id": "Figure_2",
  "generated_at": "ISO-8601 timestamp",
  "git_commit": "commit hash",
  "experiment": "allDelay",
  "cohort_manifest": "path and hash",
  "input_artifacts": [],
  "analysis_config": {},
  "figure_spec": {},
  "theme_version": "version or hash",
  "software_versions": {},
  "panels": {
    "A": {
      "renderer": "draw_protocol_panel",
      "data_artifact": "..."
    }
  }
}
```

The full manifest also stores `source_file`, `source_symbol`, `source_hash`,
`reproduction_snippet`, `artist_registry`, and the embedded SVG metadata hash.

### Tasks

1. Extend `figure_saving.py` with multi-format publication export.
2. Add physical-dimension validation.
3. Record Git commit and package versions.
4. Record input artifact paths and hashes.
5. Record the cohort-manifest hash.
6. Record resolved analysis and figure settings.
7. Prevent silent overwrite unless explicitly requested.

### Deliverables

- Multi-format export utility
- Per-figure provenance JSON
- Reproducible output naming convention
- Build summary

### Exit criteria

- Every manuscript panel can be traced to code, data, cohort, and settings.
- Re-running an unchanged figure produces equivalent output and provenance.
- Final dimensions match the target journal specification.

## Workstream H: Figure quality control

### Objective

Catch structural, perceptual, accessibility, and scientific errors before submission.

### Level 1: Deterministic checks

Verify:

- Expected panel count and IDs
- Final width and height
- Minimum font size
- Allowed font families
- Valid SVG XML
- No clipped labels or axes
- No text outside the canvas
- No unexpected external file references
- No unexpected rasterization
- Approved semantic colors
- Expected sample-size labels
- Required units on axes
- Required output formats

### Level 2: Accessibility and perceptual checks

Render:

- At final physical size
- At 100% view
- At 300 and 600 DPI
- In grayscale
- Under deuteranopia simulation
- Under protanopia simulation
- Under tritanopia simulation

Review:

- Text legibility
- Condition distinguishability
- Whitespace balance
- Visual hierarchy
- Legend ambiguity
- Overlap and clipping
- Confidence-band visibility
- Heatmap color-scale interpretation

### Level 3: Scientific review

A human reviewer must confirm:

- Correct data artifact
- Correct fish cohort and sample size
- Correct condition labels and colors
- Correct axes and units
- Correct baseline and response windows
- Correct statistics and correction method
- Correct uncertainty representation
- Correct caption
- No graphical change that alters interpretation

### Optional AI-assisted critique

A vision model may assist with:

- Panel-label alignment
- Whitespace
- Overlap
- Font legibility
- Apparent color-accessibility problems
- Visual hierarchy

It must not approve scientific correctness or automatically modify data marks. All proposed changes require review and should be expressed as specific, auditable layout or style changes.

### Deliverables

- QC script
- QC contact sheet
- Figure review checklist
- Signed-off review record for final manuscript figures

### Exit criteria

- Deterministic checks pass.
- Accessibility variants remain interpretable.
- A human scientific reviewer signs off each major figure.
- The caption and provenance manifest agree.

## Workstream I: Build automation

### Objective

Regenerate figures through one explicit command rather than a sequence of manual script edits.

### Tasks

1. Introduce a small command-line interface or Python module entry point:

```powershell
python -m figures.build --figure Figure_2
```

2. Support:
   - Build one panel
   - Build one major figure
   - Build all manuscript figures
   - Run QC
   - Export all formats
3. Use explicit configuration files or typed specifications rather than editing `RUN_*` constants.
4. Fail clearly on:
   - Missing input artifacts
   - Cohort-count mismatch
   - Model-result mismatch
   - Missing fonts
   - Invalid dimensions
   - Failed QC
5. Consider a Makefile or `justfile` only after the underlying Python commands are stable.

### Deliverables

- Figure build command
- Reproducible build log
- Failure-safe validation
- Optional paper-level build integration

### Exit criteria

- A documented command regenerates an entire major figure.
- Missing or inconsistent inputs cause a visible failure.
- The build does not depend on filesystem “first” or “latest” file selection.

## Pilot implementation

Do not begin by migrating every figure. Use one scientifically important figure containing representative panel types.

### Proposed pilot

A conditioning summary figure with:

- Protocol schematic
- Example fish trace
- Population scaled-vigor heatmap
- Normalized-vigor block summary
- Statistical comparison

This tests:

- Simple and compound panels
- Shared colors and typography
- Matplotlib composition
- Mixed-media assembly
- Legends and colorbars
- Persistent annotations
- Provenance
- Accessibility

### Pilot stages

1. Freeze the current figure and its source artifacts.
2. Confirm the underlying analysis is valid.
3. Create panel-ready data tables.
4. Migrate each panel to `prepare_*` and `draw_*`.
5. Build a Matplotlib-only draft.
6. Export panel and compound SVGs.
7. Build an Inkscape template for the heterogeneous final layout.
8. Add provenance and QC.
9. Compare with the original figure.
10. Review with the manuscript author.

### Pilot success criteria

- All panels regenerate from explicit inputs.
- One style source controls the complete figure.
- Panels can also be exported standalone.
- Layout can be changed without changing panel code.
- Updating one panel does not destroy annotations.
- Figure passes deterministic and accessibility QC.
- A reviewer can identify the source data and cohort for every panel.

## Prioritized implementation sequence

### Priority 0: Scientific blockers

- Resolve vigor, bout detection, scaling, missingness, and cohort inconsistencies.
- Do not approve final manuscript figures before these are addressed.

### Priority 1: Foundations

- Shared immutable theme
- Physical dimensions
- Semantic colors
- Multi-format export
- Provenance format

### Priority 2: Reusable panels

- Extract representative renderers
- Preserve standalone wrappers
- Remove hard-coded panel letters
- Remove data loading from drawing functions

### Priority 3: Major-figure composition

- Typed specifications
- `subplot_mosaic` and nested `GridSpec`
- Shared legends and colorbars
- Panel-label placement

### Priority 4: Mixed-media and interactivity

- Inkscape template conventions
- Interactive configuration notebook
- Limited `figurefirst` proof of concept

### Priority 5: QC and automation

- SVG audit
- Accessibility renderings
- Human review checklist
- One-command builds

### Priority 6: Scale to all figures

- Migrate remaining main figures
- Migrate supplementary figures
- Integrate caption and manuscript references
- Archive a reproducible submission snapshot

## Decisions to make before implementation

1. Which journal or target page dimensions should define figure widths?
2. Which major figure should be the pilot?
3. Which figures are plot-only versus mixed media?
4. Which condition color palette is final?
5. Which font is acceptable to the target journal and available on all build systems?
6. Should final figure SVG/PDF files be committed, released separately, or both?
7. Which analytical issues must be corrected before panel migration begins?
8. Who provides scientific sign-off for each final figure?

These choices should be recorded in the figure specification or project documentation rather than remaining implicit.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Visual refactor changes scientific content | Separate preparation from drawing; compare saved panel data and statistics |
| Manual SVG edits are lost after regeneration | Store persistent changes in code or template layers |
| `figurefirst` is incompatible with the environment | Test only on the pilot; retain Matplotlib/Inkscape fallback |
| SVG normalizer changes semantic colors | Restrict edits to explicit semantic IDs/classes |
| Generated SVG diffs are noisy | Prioritize code, specs, tokens, and provenance in review |
| Fonts differ across systems | Validate installed fonts and embed or package according to journal policy |
| Linked Inkscape panels break | Use relative links during iteration and embed archival masters |
| Confidence intervals treat trials as independent | Use fish-level or hierarchical resampling |
| Excluded fish re-enter downstream figures | Assert cohort membership at panel-data generation |
| Old and new outputs coexist ambiguously | Use run IDs and immutable output directories |
| Final-size text becomes unreadable | Include physical-size and raster-preview QC |
| AI visual review introduces unsafe edits | Restrict AI to suggestions; require human review |

## Definition of done for a manuscript figure

A figure is ready for manuscript submission only when:

- [ ] Its analysis has passed the relevant scientific validation.
- [ ] Every panel has an explicit input artifact.
- [ ] The fish cohort and sample size are recorded.
- [ ] Statistical annotations come from saved results.
- [ ] Shared visual tokens are applied.
- [ ] Panel dimensions match the target journal.
- [ ] Panel labels are assigned in one composition layer.
- [ ] Legends and colorbars are unambiguous.
- [ ] SVG and PDF exports remain vector where expected.
- [ ] Required raster outputs have the correct DPI.
- [ ] Deterministic QC passes.
- [ ] Grayscale and color-vision-deficiency previews remain interpretable.
- [ ] No labels or marks are clipped.
- [ ] The caption agrees with the actual panel data and analysis.
- [ ] A provenance manifest is saved beside the figure.
- [ ] The SVG contains compact inert reproduction metadata.
- [ ] Important artists have stable semantic IDs and a validated registry.
- [ ] A human scientific reviewer has approved the figure.
- [ ] Any manual refinement is archived separately and documented.

## Immediate next steps

1. Select one pilot major figure.
2. Resolve the pilot’s analysis blockers from `../docs/analysis/ANALYSIS_ISSUES.md`.
3. Confirm the target journal dimensions and font requirements.
4. Implement the immutable figure theme.
5. Extend `figure_saving.py` with publication export and provenance.
6. Extract the pilot’s first `prepare_*` and `draw_*` panel pair.
7. Compose the plot-only draft with Matplotlib.
8. Evaluate whether the final pilot requires an Inkscape template.
9. Add deterministic and accessibility QC.
10. Review the pilot before migrating the remaining manuscript figures.
