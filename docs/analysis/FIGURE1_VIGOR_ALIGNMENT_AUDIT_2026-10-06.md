# Figure 1E/F alignment audit, 6 October 2026

**Subsequent finding:** these tests establish consistency within the arrival-clock artifacts, not correctness of that clock as exposure timing. The user identified legacy buffered-arrival correction. [The subsequent timing repair](LEGACY_ACQUISITION_TIMING_REPAIR_2026-10-06.md) shows that acquisition-time reconstruction removes the broad detector invalidity in the displayed windows. Earlier conclusions about there being no upstream processing defect must be read with that correction.

The current legacy-metric assembly has no demonstrated frame, CS-origin, or half-second bin alignment defect. The dominant source of absent orange bands is detector validity loss around invalid derivatives, followed by movement masking. Bout medians and the independent display axes explain additional differences in apparent amplitude and support. No production signal or panel was changed.

## Reproduction and evidence

Run `.venv-trace/Scripts/python.exe scripts/audit_figure1_vigor_alignment.py` from the repository. Outputs are written directly to `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/audit-vigor-alignment-20261006`: `audit.json`, `joined_frames.parquet`, `bins.csv`, `example_frames.csv`, `diagnostic.svg`, and `diagnostic.png`. The diagnostic places raw vigor, detector inputs, detector support, and exact stored bins in separate aligned rows.

Both E v4 and F v2 sidecars were read before their inputs. All sidecar hashes for raw frames, events, panel data, SVGs, corrected metric/preprocessed inputs, stimulus events and source manifest matched. The movement completion marker and its summary hash matched; the summary identifies the same corrected-v1 metric hash used by E/F. Historical F: paths in the detector summary refer to the same hashed content now accessed on J:; there is no evidence of a different source version.

For trials 9, 17, 63, 66 and 93, all 140,515 frames inside [-20,20) were joined by FrameID to the metric and movement sources. Saved raw vigor equals the source legacy metric exactly, including NaNs. Source metric and movement FrameID/AbsoluteTime arrays match exactly. Every saved CS-relative time equals `(AbsoluteTime - measured Cycle.Beg)/1000`, with maximum error **0 ms**. All **400 bins** reproduce exactly with matching NaN masks. Bin index is `floor((seconds + 20)/0.5)`, using left-inclusive, right-exclusive edges `-20 + k*0.5`. Frame timestamps are nonuniform; median frame intervals are approximately 1.419 ms. FrameStep gaps are zero in these windows, but long measured intervals exist. Bins are frame-weighted, not duration-weighted.

The trace preparation can include a frame at exactly +20 s, whereas the heatmap excludes +20 s. The audit explicitly excludes that endpoint; this documented endpoint difference cannot explain interior shifted/absent bands.

## Measured missing-band example

Trial **93**, measured CS onset **1668533635624 ms**, bin **49**, **[4.5,5.0) s**:

- 353 raw frames; maximum vigor **4.1241254788 rad/ms** at FrameID **8619063**, AbsoluteTime **1668533640413 ms** (4.789 s), measured derivative interval **0.1998 ms**.
- 0 detector-valid, 0 moving, 0 eligible frames. Stored and reconstructed bin are both **NaN**, not zero.
- One invalid derivative in this bin: FrameID **8619056**, time **4.788 s**, interval **12.9589 ms**, raw metric NaN. Angular coverage is 1.0, and FrameStep is 1. Corrected preprocessing rejects derivative intervals exceeding **10 ms**.
- The 7-frame full-occupancy median propagates that NaN to 7 smoothed frames. The centered 403-frame full-occupancy rolling minimum then makes the envelope missing across the entire selected bin. All 353 envelopes are NaN; no coverage failure occurs.

Across all five windows, only **232/140,515** derivatives are invalid, but **94,774/140,515 (67.45%)** envelopes are nonfinite. This is a substantial amplification of sparse invalid derivatives by the configured full-occupancy windows. Detector-valid fractions are **30.7%, 30.8%, 37.2%, 33.2%, 30.9%**; eligible fractions are **8.7%, 9.0%, 12.4%, 13.7%, 16.5%**. Missing bins are **50, 51, 43, 38, 25** out of 80 respectively. Raw peaks remain visible where the detector cannot evaluate the signal.

## Detector and aggregation interpretation

The saved masks were independently reproduced from metric values with 2 s context on both sides of each trial window; all valid/moving flags match. Detector configuration: legacy distal angular speed, 7-frame median (~10 ms), centered max 21 frames (~28.6 ms) minus centered min 403 frames (~571.4 ms), threshold **4 deg/ms = 0.06981317 rad/ms**, amplitude gate **1 deg/ms = 0.01745329 rad/ms**, minimum duration **57.1429 ms**, maximum interbout gap **14.2857 ms**, minimum angular coverage **0.8**. These are appropriate to the documented legacy route and correctly converted to radians. The windows resolve to sample counts using the recording median interval; they are not exact timestamp-duration rolling windows. They are centered, so support can extend before/after instantaneous raw peaks. Bout duration and gap gates use measured DeltaTimeMs.

Only valid positive moving frames with positive bout IDs contribute. Baseline is the median natural log of those frames in **[-15,0)**. Each bout's median centered log value, computed on its eligible support within the trial window, replaces its eligible instantaneous values. Bins average those repeated medians, weighted by eligible frame count. Consequently, an isolated large peak need not create a large positive band; an eligible bout below baseline can produce a negative band despite positive raw vigor. A near-zero band can be visually tiny. No missingness-to-zero conversion was found.

The current E overlay uses independent axes: raw vigor **0–0.6 rad/ms**, signed log vigor **-1.25 to +0.75**. Orange zero lies 62.5% up the plotting area, while black zero lies at its bottom. Vertical correspondence therefore has no amplitude meaning. Raw values above 0.6 are clipped visually; F's ±0.25 colour limits saturate colours without altering stored values.

## Scope and conclusion

The missing-band example is explained by an explicit upstream validity/smoothing rule, not an x-axis offset. This behavior reproduces the implementation, whose full-occupancy rule is documented as preserving the historical dropna behavior. It is not enough evidence to label the implementation defective or silently relax masking. Its large coverage loss merits scientific review before using the detector for inference. The provenance explicitly says thresholds are historical, not validated against video labels, and the detector is not paper-approved. Blinded/video validation and sensitivity analysis remain needed to judge whether these rules identify real bouts adequately.

E remains under review. A–D frozen assets, freeze/assembly configs, earlier variants, and current assembly were preserved. No alternative metric was substituted. The audit diagnostic is evidence, not a replacement panel or visual redesign.

## User correction: raw and scaled must share bout-frame support

The user clarified that both raw and scaled vigor in the comparison must use the same frames, selected through bout detection. The existing all-frame raw versus masked/binned overlay does not satisfy that comparison requirement even though its underlying timing and arithmetic reproduce correctly.

`scripts/build_figure1_same_frame_vigor_review.py` creates a new review in `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/audit-vigor-same-frames-20261006/`. A single `valid & moving & bout_id>0 & finite(Vigor) & Vigor>0` mask is applied to untransformed raw vigor, framewise centred log vigor, and the existing signed bout-median signal. Excluded frames are NaN in all three. Assertions verify identical finite-frame support and unchanged raw amplitudes. The plot uses separate aligned columns for raw, scaled, and exact stored bins, with no y clipping. The light scaled trace preserves framewise changes; the dark trace shows the bout median actually used by the heatmap.

Shared supported frame counts for trials 9/17/63/66/93 are 2445/2525/3472/3852/4628. Averaging the supported bout medians reproduces all 400 F bins, including NaNs, within 3.2e-15 floating-point error. Existing panels and source processing remain preserved. This repairs the comparison's support mismatch; it does not change the detector's documented loss of coverage.

## Repeat with explicit NaNs from corrected angles

At the user's request, `scripts/repeat_figure1_nan_mask_audit.py` repeats corrected angles → raw legacy vigor → detector → signed bins. It sets invalid angle-position/timestamp rows to NaN, invalid raw derivatives to NaN, detector-invalid vigor to NaN, and uses a shared eligible-bout mask for compared raw/scaled signals. Output is `audit-vigor-explicit-nans-repeat-20261006/explicit_nan_frames.parquet`, plus `repeat.json` and a repeated same-frame comparison generated from that new data. Authenticated source files are preserved.

There are zero invalid angle-position rows in the contextual windows inspected. This audit therefore does **not** establish missing angles as the cause. Valid angles on both sides of an excessive measured interval can still yield an invalid derivative: the current derivative rule rejects intervals >10 ms. The independently calculated raw vigor matches the stored candidate metric to 1e-12 (excluding only the first context row, whose predecessor is outside the read). In [-20,20), trials 9/17/63/66/93 have 48/48/43/46/47 invalid raw measurements. Recalculated valid/moving flags agree exactly with stored flags. Recalculated 400 signed bins agree exactly, including missingness. No NaNs were interpolated or converted to zero. Invalid positions and invalid derivatives have distinct validity criteria; rejecting a derivative does not itself make its endpoint angle a missing position measurement.
