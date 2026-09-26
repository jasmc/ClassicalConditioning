# Figure 1 vector assembly preview

This is a **layout preview**, not an approved manuscript figure or a change to
the scientific panel registry. The working map has A preparation, B
condition-timing inset, C session sequence, D/E matched tail-angle/vigor
traces, and F/G/H Delay/3sTrace/control single-fish heatmaps.
The panel map can be changed in the JSON without changing the composer.

## Rebuild

From the repository root, the default output goes directly to the SSD:

```powershell
& '.venv-trace\Scripts\python.exe' scripts/build_figure1_panel_a.py
& '.venv-trace\Scripts\python.exe' scripts/build_figure1_panel_b.py --variant v4
& '.venv-trace\Scripts\python.exe' scripts/build_figure1_panel_c.py --variant v2
& '.venv-trace\Scripts\python.exe' scripts/build_figure1_trace_panels.py
& '.venv-trace\Scripts\python.exe' scripts/build_figure1_legacy_vigor_heatmaps.py
& '.venv-trace\Scripts\python.exe' scripts/assemble_svg_figure.py configs/paper-figures/figure1-assembly.json --export png
```

Use `--watch` to rebuild when a source SVG or the layout JSON changes. Use
`--strict` only after every slot has a selected source; it rejects placeholders.
For one-panel review, add `--panel A` (or another panel ID); this writes
`panels/Fig1_PanelA.svg` and the requested PNG/PDF under the same SSD figure
folder. Other figure manifests can set their own `figure_id` for the same
naming convention.
For example, refine the A scheme at `schemes/Fig1_PanelA_Setup_DejaVuSans_v1.svg`, run
`--panel A --export png`, inspect `panels/Fig1_PanelA.png`, then run the full command
above. The layout can also carry text repairs, crop, source choice, or position
changes for one panel. B can be removed from a later layout without touching
C's scheme. `--watch --panel A --export png` refreshes its preview as its
source SVG changes.
The same script can assemble other figures from another JSON with `canvas` and
`panels`. Each panel has an `id`, `box` `[x,y,width,height]`, and optional
`source` path relative to the layout's `storage_root`. Sources may be SVG, PNG, or JPEG; SVG
remains vector, while a generated raster plot is embedded at its original
resolution. `view_box` crops a compound SVG without editing its source.
Changing a panel's `source` or replacing that file changes
only that panel's content on the next build. The output SVG embeds its sources
as vector elements and prefixes their IDs and CSS classes. The `.svg.json`
sidecar records input hashes and missing slots. The repository holds only the
layout recipe and composer; scheme assets, fonts, assembled figures, and
individual panel previews live under
`J:\ClassicalConditioning Outputs\ORGER-JOAQUIM\outputs\figure1-assembly\`.

The five supplied J: drive SVGs are retained byte-for-byte in the SSD `schemes/`
folder. Names ending `_source.svg` are archival artwork; versioned SVGs are
editable proposals. The current layout selects A's font-normalized SVG,
B exploratory v4, and C exploratory v2. Earlier B and C proposals remain in
`schemes/`, with their earlier combined review at `figure1-exploratory-BC.svg`.
The supplied files are named:

| Original | SSD scheme name |
| --- | --- |
| Asset 6 | `Fig1_PanelA_Setup_source.svg` |
| Asset 10 | `Fig1_PanelB_ConditionTiming_source.svg` |
| Asset 7 | `Fig1_PanelsB-C_ProtocolAlternative_source.svg` |
| Asset 11 | `Fig1_PanelsB-C_SessionTimeline_source.svg` |
| Asset 5 | `Fig1_PanelsB-C_SessionTimelineAlternative_source.svg` |

For an exploratory replacement without changing the selected panel in the
layout, use `--replace-panel-source B=schemes/Fig1_PanelB_ConditionTiming_exploratory.svg`
with `--output` set to a separate SSD SVG path. This works with `--panel B` for
a standalone review and without it for the whole figure. The override and
its source hash are recorded in the output's `.svg.json` sidecar.
The override drops crop and font-repair instructions tied to the selected
source, so a clean replacement C SVG is not cropped using the old compound
Illustrator file's coordinates.

## Typography

The repository's scientific plot theme uses bundled **DejaVu Sans**. The
selected A, B, and C source SVGs now declare DejaVu Sans themselves, as do
the F/G/H vector heatmaps. A's outlined "Basal illumination" glyphs were
replaced with live DejaVu Sans text; its original vector artwork is retained
in the untouched source asset. B/C exploratory versions use live text and
consistent regular/bold roles. New figure layouts can set
`font_family`; the default is DejaVu Sans. The regular and bold font files and
their license are stored in the SSD `fonts/` folder. For Inkscape export,
`font_directory` supplies a local Fontconfig file so PNG/PDF use those same
glyphs rather than substituting another system font.

## Artwork changes in this preview

| Change | Reason |
| --- | --- |
| A retains Asset 6's apparatus; its embedded letter and outlined basal-illumination text are replaced by live DejaVu Sans. | Aligns typography and panel lettering with the scientific plots. |
| B v4 gives Control three separate example-trial lanes. Each has one US dot, a question-mark time tick, and a full-size CS bar matching the paired rows. | Makes the one-US-per-trial relationship explicit while showing that Control onset varies. |
| C v2 omits the final viability-check card and legend item; its four phase cards expand to use the width. | Focuses C on the session phases while retaining the protocol counts and short/long US distinction. |
| F/G/H are individual fish heatmaps from the legacy distal angular speed metric; all use the same signed bout-log recipe, 0.5 s bins, DejaVu Sans, `managua_r`, and fixed −0.25…+0.25 colour limits. | Makes the three examples directly comparable and keeps all heatmap cells as SVG rectangles. |
| D/E show the same authenticated Delay fish (`20221115_07`) and global CS trials 9, 17, 63, 66, and 93. D is pre-CS-median-centred tail angle in degrees; E is unscaled frame-level legacy vigor in rad/ms. | Keeps the paired traces measured, matched by frame/trial, and independently replaceable as SVG panels. |
| The selected A SVG converts Illustrator `rgba()` to RGB plus opacity. | Prevents Inkscape from rendering the translucent apparatus circle as black. |

The active C protocol retains the 500 ms final event in its data specification;
it is deliberately absent from the visible v2 panel. Historical C v1 remains
available. All F/G/H fish IDs and source hashes are recorded in the SVG JSON
sidecars next to their panel data.

## Remaining slots

The current D/E traces and F/G/H heatmaps are review examples rather than a
final choice of manuscript fish or activity metric. Each source can be changed
in the layout manifest and rebuilt without regenerating unrelated panels.
