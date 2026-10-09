# Corrected complete-bout analysis

The user confirmed: use complete detected bouts beyond the plotting crop; baseline statistics use every eligible baseline timepoint carrying its corresponding bout median; retain Version 2 without bout-median substitution in its displayed means. The user also clarified that the baseline median should map to zero.

## Current pipeline

- Fish: F Delay 20221115_07; G 3 s Trace 20230310_08; H Control 20221115_09.
- Reconstruct raw angular speed and run the unchanged shared detector with additional context around each trial. Check source hashes and alignment. Every selected complete bout ends safely within its loaded frame segment; at least 3.3 s of context margin was available. Eligibility and direct log values match the prior data exactly.
- Calculate each selected bout's median natural-log vigor over all its eligible support, including samples outside [-20,+20) s. Repeat the median on the bout's eligible samples. Complete support is saved for independent readback.
- For every trial, calculate P10, m=P50 and P90 from all eligible [-15,0) s timepoints carrying their complete-bout median values. The same reference statistics apply to all three versions. Neither bouts nor bins get a single vote. Quantiles use NumPy linear interpolation.
- C sample display: `clip((bout_median_log-m)/((P90-P10)/2),-1,+1)`, with no binning.
- D sample display: `clip((bout_median_log-m)/max(m-P10,P90-m),-1,+1)`, with no binning.
- Version 2 C display: each half-second cell averages direct eligible framewise log vigor, without bout-median substitution; subtract the shared m, divide by `(P90-P10)/2`, then clip. The baseline reference is not calculated from its bin means.
- Display trials 5-94 in [-20,+20) s, using the current shared managua_r colourbar and layout. Missing or undefined values remain black. H trial 16 retains a zero quantile range and an undefined display.

## Meaning of zero and scaling terminology

Both C and D map the baseline reference median exactly to zero. They are median-centred percentile scales, rather than ordinary affine min-max scaling. Mapping P10 to -1 and P90 to +1 with ordinary min-max centres their midpoint and cancels prior median subtraction; it would not generally preserve median zero.

The C/D sample baseline timepoint medians are zero to a maximum absolute residual of about 1.2e-15. Version 2's direct-log cell means use that same bout-median reference, so their displayed baseline-cell median need not be zero. Its measured ranges are Delay [-1,-0.303], 3 s Trace [-1,-0.645] and Control [-1,-0.511]. This reflects different display and reference populations, not an arithmetic failure.

Complete-bout medians can use post-CS values when a baseline bout crosses CS onset. Such occurrences total 38 for Delay, 66 for 3 s Trace and 13 for Control. This is a consequence of the confirmed full-bout rule.

## Artifacts and verification

The three current rows are `FGH_C_BoutSamples_shared_colourbar`, `FGH_D_BoutSamples_shared_colourbar` and `FGH_C_DirectBins_shared_colourbar`, each in SVG/PNG/PDF. `baseline_statistics.csv` records the common 270 trial references. Complete-bout support, summaries, cropped frame tables, display runs and bins are saved per fish. `data_manifest.json`, each variant's validation and `numeric_verification.json` record checks and hashes. Independent readback recomputed full-bout medians, all baseline quantiles, sample transforms and direct-bin means/scales; SVG geometry and colour fills and PDF renderings were checked.

These outputs implement the user's corrections. Earlier freezes and figures remain preserved as history. The HTML summary in `reviews/fgh_latest_summary_20261009/index.html` now presents these corrected rows.
