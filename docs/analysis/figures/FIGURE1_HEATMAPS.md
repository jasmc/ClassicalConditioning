# Figure 1 single-fish heatmaps: signal and colour scale

The current review assembly uses three individual fish: F Delay
`20221115_07`, G 3 s Trace `20230307_12`, and H unpaired Control
`20221115_09`. Each heatmap has one row per CS trial, grouped as Pre-Train
(global CS trials 5–14), Train (15–64), and Test (65–94). The four priming
presentations are not displayed. Columns span −20 to +20 s relative to CS
onset in 0.5 s bins. Green guides mark CS onset and offset; purple Train guides
mark the verified paired US in F and G. H has no fixed CS-relative US guide.

The raw per-frame candidate is `legacy_distal_angular_speed_rad_per_ms`:
absolute wrapped frame-to-frame change in the sum of local tail angles,
divided by the measured frame interval. It is a historical benchmark; opposing
segment changes can cancel. This is also the *unscaled raw* vigor in E.

For each fish and CS trial, the heatmap recipe uses only valid, positive
activity frames belonging to detected moving bouts. It takes the natural log
of the raw metric and subtracts the median log value in that trial's
pre-CS window [−20, 0) s. It then gives each moving-bout frame its bout's
median centred log value. A heatmap cell averages those frame-assigned values
in a 0.5 s bin. Missing moving-bout or baseline data remain NaN and display
black. This is implemented in
`src/classical_conditioning/analysis/temporal_profiles.py:_signed_bout_log_vigor`
and `src/classical_conditioning/figures/signed_bout_heatmap.py:calculate_fish_heatmaps`.

All three panels use the same `managua_r` palette and fixed display limits
−0.25 to +0.25. Values beyond those limits saturate at the endpoint colours;
the stored panel data are not clipped. Zero denotes the trial's pre-CS median
log vigor, not zero movement. The colour scale is not per-fish min–max or
0–1 normalization. A log difference of +0.25 corresponds to about 1.28 times
the baseline raw metric, and −0.25 to about 0.78 times, before the bout/bin
aggregation. The three fish are therefore displayed on a comparable scale.

The narrow historical display range saturates many valid cells: 45.8% in
Delay F, 40.7% in 3 s Trace G, and 7.0% in Control H. Nonmissing cells number
3,654/7,200, 6,011/7,200, and 3,470/7,200 respectively, as measured from
the saved panel-data Parquet files. Black cells are missing signal, not zero.
This saturation should be reviewed before choosing a final manuscript colour
range; the stored values have not been clipped.

The editable SVGs, per-bin Parquet files, and SHA-256 sidecars are written
directly to `J:\ClassicalConditioning Outputs\ORGER-JOAQUIM\outputs\figure1-assembly\heatmaps\`
by `scripts/build_figure1_legacy_vigor_heatmaps.py`. This figure is a review
example; these fish and the legacy metric have not been frozen as the final
manuscript selection.
