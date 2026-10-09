# Two additional F/G/H review versions

F: Delay 20221115_07. G: 3 s Trace 20230310_08. H: Control 20221115_09. Both new rows retain the shared managua_r colourbar at +/-1, the latest axes and phase labels, and the "Vigor relative to baseline" label.

## D with bout medians, no binning

The eligible samples and bout-median log vigor are taken from the selected Version 1 frame tables. Baseline observations are eligible displayed samples within [-15,0) s, with no half-second aggregation. Let m = P50 and z = sample's bout-median log vigor minus m. The displayed value is:

`D = clip(z / max(m-P10, P90-m), -1, +1)`

This is one common denominator for both signs, rather than a different scale for positive and negative values. Longer eligible bout durations still contribute more baseline samples. Lossless runs encode contiguous samples with identical displayed values, without bridging gaps.

## Version 2: direct log vigor, 0.5 s bins

Use eligible framewise natural-log vigor directly, without replacing it by bout medians. Each half-second cell is the mean of its finite eligible log-vigor samples. Cells have edges from -20 to +20 s; trials 5-94 produce 90 x 80 cells per fish. Empty cells remain missing.

Calculate P10, m=P50 and P90 from the finite baseline-bin scalars within [-15,0) s (bin indices 10-39). Each finite bin contributes once, regardless of its eligible frame count. The displayed Version 2 uses C, consistent with the original C-mode comparison:

`C = clip((bin_mean-m) / ((P90-P10)/2), -1, +1)`

The user was offered a choice of C or D for Version 2; in the absence of a reply, C was retained from the original comparison. D-scaled bin values are also recorded in the numeric table for reproducibility, but this review row displays C.

## Provenance and validation

Both variants use the existing hash-bound frame tables, so raw-vigor reconstruction, bout detection, eligibility and CS alignment remain identical. Quantiles use NumPy's linear method. A zero/undefined denominator produces missing values, without borrowing another trial's scale. Black denotes a missing displayed value, not zero.

`data_manifest.json` records source hashes, formulas, support counts and numeric-table hashes. `baseline_statistics.csv` records each trial's quantiles, denominators and finite baseline counts. Individual tables preserve sample intervals for D and half-second means, frame counts and C/D values for Version 2. The exported SVG rectangles are checked against table values, time bounds, trial positions and palette fills; PDFs are rendered for inspection.

These are additional review variants. The selected Version 1 C freeze and its registration remain unchanged.
