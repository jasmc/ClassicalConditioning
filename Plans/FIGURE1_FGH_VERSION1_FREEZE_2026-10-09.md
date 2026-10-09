# Figure 1 F/G/H: selected Version 1 freeze

The user approved freezing Version 1 on 8 October 2026 and requested layout improvements using the frozen pooled-fish heatmap and the uploaded legacy SVG. The selection was recorded on 9 October 2026.

## Frozen processing and values

- F: Delay, fish 20221115_07; G: 3 s Trace, fish 20230307_12; H: Control, fish 20221115_09.
- Global CS trials 5-94; display interval [-20,20) s relative to CS onset.
- Calculate framewise angular speed before shared bout detection. Calculate the bout summary after detection.
- Use the median of eligible natural-log vigor observations for each detected bout within the displayed trial window, then repeat that median on its eligible samples.
- Use eligible samples within [-15,0) s for each trial's P10, P50 and P90. There is no half-second binning anywhere in this selected version. Longer eligible bout durations therefore contribute more baseline samples.
- Apply C = clip((x - P50) / ((P90 - P10) / 2), -1, +1). Retain managua_r, limits -1 and +1, and its original zero colour.
- Render each eligible sample interval without bridging missing data. Consecutive equal-valued samples are losslessly encoded as runs, with no averaging.
- H trial 16 retains an undefined scale because its baseline quantile range is zero. Black areas represent unavailable display values, not zero scaled vigor.

The numeric freeze is `reviews/fgh_v1_layout_freeze_20261009/frozen-version1/freeze.json`. It records SHA256 hashes for the copied run tables, approved original figures, selected processing code and baseline statistics, and binds the original full sample tables by path and hash. The source comparison manifest is retained for provenance; Version 2 is not selected.

## Revised layout

Each panel now has a continuous tall global-trial axis, four thin spines, outward ticks, time ticks at -20/0/20 s, white phase separators after trials 14 and 64, compact Pre/Train/Test labels, condition-coloured titles and a slim right colourbar. Green CS guides and the existing training US guide are retained. The label is "Scaled log vigor (C)" and its limits remain +/-1; the pooled figure's signed-vigor units and limits were not transferred.

The legacy reference contributes tall proportions and phase labels. The frozen pooled reference contributes the boxed axes, ticks, continuous trial axis and white separators. Reference copies and hashes are retained under `reviews/fgh_v1_layout_freeze_20261009/references`.

## Verification and files

The revised SVGs were checked against all 21,853 frozen sample runs: F 9,474; G 9,072; H 3,307. Sample values and intervals match exactly. Exported rectangle positions, widths, heights and palette fills were checked numerically. Label bounds were checked and PNG/PDF renderings were inspected.

- Outputs: `reviews/fgh_v1_layout_freeze_20261009/polished-layout`, with separate SVG/PNG/PDF panels and a three-page PDF.
- Validation: `polished-layout/layout_validation.json` within that review folder.
- Registration: `configs/paper-figures/figure1-fgh-version1-freeze-20261009.json` and the scoped F/G/H field in `figure1-freeze.json`.
- Earlier scientific assessment: `Plans/FGH_ANALYSIS_STEP_ASSESSMENT_2026-10-08.md`.

This freeze selects the requested processing and numeric display; it does not resolve the scientific limitations recorded in the earlier assessment. The existing A-D freeze remains intact. Panel E and full-figure assembly are outside this change.

## Current layout: shared colourbar

At the user's follow-up request, the current F/G/H presentation is a single aligned row with one shared managua_r colourbar at the far right. Pre/Train/Test labels appear to the right of H, and the "Global CS trial" axis title has been removed; trial-number ticks remain. The recommended colourbar name is "Vigor relative to baseline": zero is the trial's baseline median, and positive/negative values indicate higher/lower log vigor on the retained C scale. Values are still clipped at +/-1.

Current SVG/PNG/PDF outputs and validation are in `reviews/fgh_v1_layout_freeze_20261009/shared-colourbar`. The scoped configuration's `current_layout` field points to this revision. All frozen numeric data and the earlier layout exports are preserved.

## Selected fish replacement

The user subsequently selected `20230310_08` for the 3 s Trace example (G), replacing `20230307_12`. The Version 1 recipe remains fixed. G was rebuilt from its own hash-verified camera, protocol, corrected angles and angular-validity coverage. F and H's displayed sample-run tables were copied byte-for-byte from their freeze.

The revised selection and output row are in `reviews/fgh_v1_trace20230310_08_20261009`, with a new freeze manifest preserving the previous freeze as provenance. The scoped Figure 1 configuration now points to this selection and the shared-colourbar layout. The previous fish and its exports remain available in the earlier review folder.

## Additional comparison rows

The user requested a D-scaled bout-median sample row and a Version 2 row using direct eligible log vigor averaged into 0.5 s bins, with baseline quantiles taken from finite baseline-bin scalars. These additional reviews are in `reviews/fgh_D_samples_and_direct_bins_20261009`. D uses `clip((x-P50)/max(P50-P10,P90-P50),-1,+1)` on unbinned bout-median samples. The Version 2 displayed row retains C from the original comparison, pending any user preference for D. Both use the current fish and shared-colourbar layout. They do not supersede the Version 1 C freeze.

## HTML summary

`reviews/fgh_latest_summary_20261009/index.html` collects all three latest rows with 20230310_08 as G: selected Version 1 C, Version 1 D, and Version 2 C. It embeds the figures and PDF downloads and includes bullet-point processing steps, the baseline-voting comparison, scale formulas and source verification. SVG links point to the workspace exports. Browser checks confirmed three loaded figures, three PDF downloads, no page errors and no horizontal overflow on desktop or mobile. Source export hashes were checked when building the summary.

## Subsequent processing correction

The user clarified complete detected bouts beyond the plotting crop, timepoint-based baseline statistics with repeated full-bout median values, Version 2 without median substitution in its displayed means, and baseline-median zero. The corrected rows are now in `reviews/fgh_full_bouts_baseline_samples_20261009`; the HTML points to them. Previous freezes and outputs above are historical versions. `Plans/FGH_REQUESTED_METHOD_CORRECTION_2026-10-09.md` records the decisions, exact formulas, verification, and the distinction between Version 2's baseline reference median and its displayed baseline-bin median.

The user then approved a Version 2 exception: its baseline statistics must come from finite unscaled baseline-bin means so its displayed baseline median is zero. Its current row is in `reviews/fgh_direct_bins_own_reference_20261009`. The HTML combines that row with the unchanged complete-bout C/D sample versions. The earlier predominantly blue row is historical and no longer the current Version 2 display.
