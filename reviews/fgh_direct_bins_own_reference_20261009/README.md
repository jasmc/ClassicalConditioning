# Version 2: direct bins with their own baseline reference

The user approved: "Use baseline-bin means; guarantee displayed median zero (recommended)" after the predominantly blue display exposed a mismatch between direct-bin means and a bout-median reference.

- Fish remain Delay 20221115_07, 3 s Trace 20230310_08 and Control 20221115_09.
- Each cell is the mean of direct eligible framewise natural-log vigor inside its 0.5 s interval. There is no bout-median replacement in these values. The 90 x 80 raw cell means and eligible frame counts are unchanged.
- For each trial, collect the finite unscaled baseline-cell means in [-15,0) s (bin indices 10-39). Each finite bin contributes one vote. Calculate P10, m=P50 and P90 with NumPy linear quantiles.
- Display `clip((bin_mean-m)/((P90-P10)/2),-1,+1)`, retaining the shared managua_r palette and layout.
- The displayed baseline median is zero in all 269 defined fish/trials, with maximum absolute numerical residual 2.8e-15. Control trial 16 retains a zero quantile range and remains undefined. Zero denotes the median, rather than every baseline cell or the baseline mean.
- The C/D full-bout sample versions are unchanged and continue to use eligible unbinned timepoints carrying complete-bout medians for their own baseline statistics.

The previous blue direct-bin row remains in `reviews/fgh_full_bouts_baseline_samples_20261009` as the earlier reference-mismatch result. The current HTML summary points to this fixed direct-bin row. `data_manifest.json`, `numeric_verification.json`, and `C_DirectBins_validation.json` record inputs, approval, formulas, unchanged cell values, baseline checks and export hashes.
