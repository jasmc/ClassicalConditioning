# Scientific Figure Pipeline Plan

## Documentation baseline

This plan is tailored to:

- `bf46bf7b02baaf6d8138d9772f255881f86c3c78`
- Source feature commits `505e29a` and `88fe4fc`
- Cherry-picked equivalents `4d6939f` and `f372101`
- Working-tree LogMedian compatibility adaptations dated 2026-08-28

Recommendations from the earlier `b0dfcb3` plan are retained only where they
remain applicable to the LogMedian tree.

## Goal

Produce consistent, editable, reproducible scientific-paper figures while
keeping these concerns separate:

```text
scientific data and statistics
        |
panel-ready tables
        |
reusable panel renderers
        |
major-figure layout
        |
vector export and final annotation
        |
scientific and accessibility review
```

## Current baseline

Useful existing foundations:

- `plotting_style.py`: centralized Matplotlib/Seaborn defaults
- `figure_saving.py`: filename-safe export
- `heatmap_utils.py`: shared heatmap preparation and phase axes
- `pipeline_utils.py`: shared heatmap and trial-metric helpers
- `analysis_utils.add_component()`: positioned figure text
- LogMedian steps 3-6: a consistent zero-centered analytical family
- Learner-stratified pipeline: SVG/PNG export, fish-first summaries, panel data,
  coverage tables, and provenance

Current limitations:

- Plot functions often load, transform, draw, label, and save in one function.
- Panel letters and layout are embedded in scripts.
- Figure dimensions and save settings vary.
- Scientific and display transformations are not always named separately.
- Major figures are not assembled from reusable panel functions.
- Figure artifacts do not consistently identify input data and cohort.

## Design principles

1. A figure never chooses its scientific cohort.
2. A renderer does not load files or run statistics.
3. Analytical values and display-only normalization use different names.
4. Condition colors have one semantic definition.
5. Panel labels are assigned by the major-figure composer.
6. SVG and PDF are canonical figure outputs.
7. Manual SVG edits do not alter data marks.
8. Every paper figure has a provenance record.

## Target interfaces

### Panel data preparation

```python
def prepare_temporal_profile_panel(
    temporal_data: pd.DataFrame,
    result_table: pd.DataFrame,
    config: TemporalProfileConfig,
) -> TemporalProfilePanelData:
    ...
```

### Panel rendering

```python
def draw_temporal_profile_panel(
    ax: matplotlib.axes.Axes,
    panel_data: TemporalProfilePanelData,
    style: PanelStyle,
) -> PanelArtists:
    ...
```

Renderer rules:

- Accept an existing `Axes` or `SubplotSpec`.
- Do not read files.
- Do not decide fish inclusion.
- Do not recalculate inferential statistics.
- Do not create the full major figure.
- Do not save or close the figure.
- Return legend/colorbar handles.

### Major figure specification

```python
@dataclass(frozen=True)
class MajorFigureSpec:
    figure_id: str
    width_mm: float
    height_mm: float
    mosaic: tuple[tuple[str, ...], ...]
    panels: Mapping[str, PanelSpec]
```

Do not create `Figure1`, `Figure2`, and `Figure3` subclasses.

## Shared design system

Extend `plotting_style.py` with immutable publication tokens:

- Single- and double-column width
- Maximum figure height
- Font family
- Axis, tick, legend, and panel-label sizes
- Axis, data, reference, and annotation strokes
- Marker sizes
- Panel gaps and outer margins
- Condition colors
- CS and US colors
- Missing/unclassified colors
- Export settings

Keep condition color separate from learner status. For learner-stratified
figures, facet by stratum or use line style/saturation while retaining the
condition's color identity.

## Composition strategy

### Matplotlib composition

Use `subplot_mosaic` or nested `GridSpec` when panels need:

- Shared axes
- Identical dimensions
- Shared limits
- Shared legends
- Shared colorbars
- Direct temporal comparison

This should be the default for:

- Control versus conditioned time courses
- Block and phase summaries
- Catch-trial grids
- Learner-stratified temporal profiles
- Heatmap groups

### Inkscape composition

Use an Inkscape template when combining:

- Protocol schematics
- Apparatus drawings
- Microscopy or raster images
- Data plots
- Persistent arrows and callouts

Keep:

```text
Figure_2.template.svg
Figure_2.generated.svg
Figure_2.final.svg
```

Recurring edits belong in code or the template, not only in `final.svg`.

### SVG assembly libraries

Do not make a dormant dependency central to the pipeline.

- Evaluate `figurefirst` on one complex figure only.
- Use `svgutils` for simple fixed SVG grids or legacy panels.
- Keep plot-only analytical layouts in Matplotlib.

## Interactive adjustment

### Immediate

Use VS Code/Jupyter with an interactive Matplotlib backend to inspect:

- Limits
- Spacing
- Legends
- Labels

### Reproducible controls

Add `ipywidgets` or a small configuration notebook for:

- Width and height
- Row/column ratios
- Margins and gaps
- Axis limits and ticks
- Legend position
- Panel-label offsets

Accepted values must be copied into the versioned figure specification.

### Final vector refinement

Use Inkscape for:

- Fine annotation placement
- Scale bars
- Schematic callouts
- Mixed-media alignment

Do not manually alter:

- Data paths
- Error bands
- Statistical markers
- Tick values
- Color scales

## Export policy

For each approved major figure:

| Format | Purpose |
| --- | --- |
| SVG | Editable vector source |
| PDF | Submission and archive |
| PNG | Preview at 300-600 DPI |
| TIFF | Only when required by the journal |

Keep:

```python
plt.rcParams["svg.fonttype"] = "none"
```

Check fixed physical dimensions after export. `bbox_inches="tight"` can change
the canvas and should not be used blindly for journal-sized figures.

## Figure provenance

Save `Figure_X.provenance.json` containing:

- Documentation/code baseline
- Figure ID
- Panel IDs
- Input artifact paths and hashes
- Experiment
- Alignment
- Cohort/manifest hash
- Analysis configuration
- Statistical result references
- Figure specification
- Theme version
- Package versions
- Random seeds
- Export dimensions

## Quality control

### Deterministic

- Expected axes/panel count
- Physical dimensions
- Required labels and units
- Allowed fonts and minimum size
- Correct condition colors
- Correct legend entries
- Expected colorbar range
- No clipped artists
- Valid SVG
- No unexpected rasterization

### Accessibility

Render:

- Final physical size
- Grayscale
- Deuteranopia simulation
- Protanopia simulation
- Tritanopia simulation

### Scientific review

A human confirms:

- Input dataset
- Fish cohort and `n`
- Alignment
- Trial/block definitions
- Baseline and CR windows
- Statistical annotation
- Caption
- No manual alteration of data marks

AI-assisted visual critique may suggest layout and accessibility changes but
must not approve scientific correctness.

## Migration order

### Phase 1: freeze one pilot figure

Select a figure containing:

- Protocol schematic
- Example fish
- LogMedian heatmap
- Trial/block summary

Record its current source artifacts and appearance.

### Phase 2: theme

- Add physical dimensions.
- Centralize semantic colors and strokes.
- Create a visual style specimen.

### Phase 3: reusable panels

Extract, in order:

1. Example trace
2. Heatmap
3. Temporal line profile
4. Block summary
5. Statistical summary

Retain standalone wrappers.

### Phase 4: major composition

- Build plot-only figure with `subplot_mosaic`.
- Add shared legends and colorbars.
- Add panel labels at composition level.

### Phase 5: mixed media

- Build one Inkscape template.
- Test linked panels locally.
- Embed panels for archival output.

### Phase 6: provenance and QC

- Add figure manifest.
- Add structural tests.
- Add accessibility contact sheet.
- Add human sign-off checklist.

### Phase 7: scale

Migrate remaining main and supplementary figures only after the pilot works.

## Definition of done

- [ ] Scientific analysis for every panel is approved.
- [ ] Panel data is an explicit artifact.
- [ ] Cohort and sample size are recorded.
- [ ] Renderer is reusable.
- [ ] Shared theme is applied.
- [ ] Layout is versioned.
- [ ] SVG and PDF are vector.
- [ ] Required raster export has correct DPI.
- [ ] Deterministic checks pass.
- [ ] Accessibility variants are interpretable.
- [ ] Caption matches analysis.
- [ ] Provenance is saved.
- [ ] Human scientific review is complete.

