# Rebuilt F, G and H: exact corrected Panel E processing

Only panels F, G and H were created in this rebuild. No full Figure 1 assembly was generated and no other panel or frozen artifact was edited.

Outputs: `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/fgh-rebuild-panel-e-contract-20261007/`. Builder: `rebuild_fgh_from_frames.py` in this folder. Each panel has PNG, PDF, SVG, a data Parquet and a processing sidecar. The build manifest records verified inputs, trial baselines, frame counts, checks and output hashes.

## Exact sequence

1. Verify saved v5 input hashes, source manifest, camera hash, corrected-angle completion marker, protocol hash and saved heatmap hash before consuming numerical data. Reconstruct from source angles; do not reuse stored signed bins as the calculation input. The stored bins are loaded afterwards as a comparison.
2. Estimate presumed acquisition cadence from the camera's stable reference. Acquired time is `reference arrival epoch + (FrameID − reference FrameID) × inferred interval`. Camera arrival intervals are not instantaneous vigor denominators. Estimated rates: F 702.571512 FPS, G 702.550276 FPS, H 702.570288 FPS, approximately 1.42334 ms per frame. Measured Cycle onset is subtracted to obtain relative seconds. This is an inferred clock, not hardware exposure timing.
3. Sum the 16 acquired local tail angles in radians to obtain cumulative distal bend B. Invalid position/timestamp rows become NaN. Compute `v = abs(atan2(sin(B[t]−B[t−1]), cos(B[t]−B[t−1]))) / delta_time_ms`. Require valid neighboring rows, consecutive FrameIDs, and positive delta time no greater than 10 ms. Invalid derivatives remain NaN. This preserves the selected `legacy_distal_angular_speed` semantics; no other metric is substituted.
4. Detect bouts with 2 s contextual margins beyond the plotted window. Smooth raw vigor with a centered 7-frame full-support median. Compute centered 21-frame maximum minus centered 401-frame minimum. Require finite envelope and angular coverage ≥0.8. Threshold at 4 deg/ms = 0.06981317 rad/ms; merge gaps shorter than 14.2857 ms without crossing invalid support; reject durations below 57.1429 ms and bouts with raw peak below 1 deg/ms = 0.01745329 rad/ms. These settings preserve the v5 recipe. The earlier arrival-clock audit used a different inferred sample count for its minimum window; do not confuse it with these 401-frame windows.
5. Crop each trial to [−20,20) s. Define ONE eligible mask: detector-valid AND moving AND bout_id>0 AND finite raw vigor AND raw vigor>0. Raw and every transformed signal use precisely this same support. Excluded frames stay NaN. No all-frame raw/conditional transformed mismatch is introduced.
6. Calculate the trial-specific reference `b_j = median(ln(v_i))` using only eligible frames in that trial's [−15,0) s window. Then `z_i = ln(v_i) − b_j`. It is subtraction, with no fish-level reference, percentile-range division, min–max scaling or stored-value clipping. The reference weights eligible frames, so longer bouts have greater influence; it is not a median of equally weighted baseline bouts. No baseline means the centred trial is entirely NaN.
7. For each detected bout, take the median z over its eligible frames within the displayed trial window. Repeat that one number only on the bout's eligible frames. A bout crossing a bin boundary keeps the same median across its contributing frames. A bout crossing the trial window is summarized over its visible eligible portion. Baseline selection may include part of a bout, but the bout summary uses its visible eligible portion across the whole trial window.
8. Assign a frame to `k = floor((relative_seconds + 20)/0.5)`. Bin k is [−20+0.5k, −19.5+0.5k). Average only finite repeated bout medians in the bin. If two bouts contribute, weight their medians by the number of their contributing frames. Divide by eligible finite count, not total frames or nominal 0.5 s duration. No contribution means NaN. This is conditional movement amplitude, not zero-inclusive total activity or bout frequency. Coverage counts/fractions are exported separately.
9. Render with managua_r and a shared [−0.25,+0.25] colour range. Stored values outside this range remain unchanged; endpoint colours saturate. Black means no eligible contribution. Each half-second cell is a bin summary, not evidence that movement occupied the entire cell. Trial rows are chronological: 5–14 Pre-Train, 15–64 Train, 65–94 Test. Green guides mark CS onset/offset (0/10 s); purple training guide marks paired US at 9 s in F and 13 s in G. H has no paired-US guide.

## Checks and findings

All 21,600 independently reconstructed bins reproduce v5 with identical NaNs. Maximum error is below 5.3×10⁻¹⁵. For F's trials 9,17,63,66,93, all FrameIDs, eligible masks, raw values, centred frame logs and repeated bout medians match the corrected Panel E frame artifact; their 400 bins match as well.

The independently rebuilt numeric values are therefore unchanged from v5. This establishes consistency with the documented E contract. It does not establish that the complete upstream preprocessing is scientifically approved: spatial filtering, 700 FPS resampling and other legacy differences remain under the separate preprocessing audit. None were silently introduced in this figure-only rebuild.

The key mathematical distinction is **mean of repeated bout medians versus mean of individual centred logs**. For F trial 93, interval [−19.5,−19.0) s:

| Bout | Eligible frames in bin | Centred bout median |
|---|---:|---:|
| 8425 | 255 | +0.289767 |
| 8426 | 41 | −0.138417 |

The heatmap value is `(255×0.289767 + 41×−0.138417)/296 = +0.230458`. The mean individual centred log on those SAME 296 frames is **−0.083087**. The positive heatmap cell follows the corrected E bout-median contract; it is not a colour inversion. Using framewise logs would produce a different signal and requires an explicit change in scientific definition.

Across finite bins, opposite signs occur in 2,350 F bins, 2,144 G bins and 1,967 H bins when those two summaries are compared. This is why treating a framewise centred-log trace as the direct input to these heatmaps gives a misleading comparison.

Trial baseline median subtraction makes the median eligible FRAME log in the baseline zero. It does not force the mean baseline bin, the mean of repeated bout medians, or every baseline cell to zero. Bright baseline cells therefore do not by themselves establish failed centring.

Empty bins: F 754/7,200, G 698/7,200, H 3,272/7,200. H's black coverage must not be interpreted as negative or zero signed vigor. The independent frame-to-bin verification image is supporting evidence for F; it is not a redesigned Panel E.

No candidate is frozen. The prior response's substantive failures were going beyond F–H and providing a stored-bin redraw instead of an independent reconstruction. This rebuild addresses those failures without claiming an arithmetic defect that the measured checks do not show.
