# Legacy acquisition timing: audit and repairs

## The original code already models buffered arrivals

The user pointed to `legacy/scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py`, calling `legacy/helpers/analysis_utils.py::framerate_and_reference_frame`. The older detailed implementation in `legacy/modules/my_functions.py` explains that timestamps record when the computer catches frames, estimates a stable cadence, models accumulated buffer lag, and flags capacity-related frame-loss evidence. Therefore interpreting every camera-log interval as a true acquisition interval omitted an existing legacy assumption. This supersedes the earlier audit's conclusion that reproducing the arrival-clock detector was sufficient to explain the signal-processing correctness.

The intended sequence is: identify stable reference → estimate capture cadence over a long span → check exported-ID loss and buffer-capacity evidence → infer frame acquisition times → resample to the target 700 FPS grid → assign protocol events using that reconstructed clock → calculate vigor and bouts. A delayed arrival followed by rapid catch-up must not be used as the instantaneous denominator for acquisition-based vigor under this model.

Before edits, the exact legacy helper returned 702.5715108257 FPS, reference FrameID 430474, no loss evidence on the full camera log. With the reader's historical 13,999-row startup discard it returned 702.5715226006 FPS, reference FrameID 443457, no loss evidence. The algorithm's broad estimate was already sound for this recording; its edge handling and downstream clocks were not.

## Verified defects repaired

- Stable-window indexing subtracted the first FrameID to infer row positions, which fails when IDs are missing. The shared `src/classical_conditioning/preprocessing/acquisition_timing.py` uses actual row positions and estimates elapsed duration per actual FrameID span.
- Missing FrameIDs were not checked directly. They now contribute an explicit missing-ID count. The historical buffer-size divisor is treated as a capacity threshold, not an individual lost-frame count. Loss evidence prevents silently generating a constant-cadence time model.
- Failure to find stable anchors left zero sentinels and could silently produce empty slices/NaN rate. It now raises a descriptive error; duplicate/reversed IDs and nonmonotonic/nonfinite times are rejected.
- The helper accepted a figure filename but ignored it. Its timing diagnostic is restored.
- `interpolate_data` mapped frame positions to the expected grid but interpolated the logged arrival clocks. It now reconstructs elapsed/absolute clocks at the expected rate before stimulus assignment, while preserving the correct `expected/predicted` mapping of source positions.
- The preprocessing script then rebuilt a **700 FPS resampled** timeline at the **estimated source rate**, after protocol assignment. That mismatched rate could distort elapsed duration by about 43.5 seconds over this recording. That second clock overwrite was removed.
- `stim_in_data` truncated reconstructed fractional milliseconds to integers. The reconstructed time remains float64. A single Cycle/Reinforcer row also collapsed to a 1D array and was silently skipped; list-based selection preserves two dimensions.

The detailed older `legacy/modules/my_functions.py` remains an archival reference. The function invoked by the user's current legacy script delegates to the new shared estimator. Historical input files, corrected-v1 artifacts and frozen panels are preserved.

## Real-fish repeat

`scripts/audit_figure1_legacy_cadence.py` verifies the camera/source hashes, estimates cadence, reconstructs acquisition times, and independently recalculates raw legacy distal angular speed, the configured detector, signed bout medians, and 0.5 s bins for the five Figure 1 trials. It uses original acquired frames at the estimated cadence, without 700 FPS resampling or adding spatial/temporal angle filters; this isolates the clock change from other legacy preprocessing differences. Consequently it is a versioned timing review, not an assertion of bit-for-bit equivalence to the entire historical pipeline.

Results are in `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/audit-legacy-cadence-20261006/`: frames, bins, `audit.json`, and `presumed_cadence_review.svg/.png`.

Repaired estimate: **702.5715121468 FPS**, interval **1.4233426530 ms**, reference **430473**, zero missing exported IDs, maximum post-reference delay **28.1438 ms**, no buffer-capacity exceedance. In the selected windows there are no invalid derivatives under the presumed acquisition clock. Detector coverage becomes 100% (movement eligibility remains a separate mask).

| Trial | Arrival-clock detector valid | Presumed-clock detector valid | Missing bins, arrival → presumed |
| --- | ---: | ---: | ---: |
| 9 | 30.7% | 100% | 50 → 20 |
| 17 | 30.8% | 100% | 51 → 11 |
| 63 | 37.2% | 100% | 43 → 14 |
| 66 | 33.2% | 100% | 38 → 4 |
| 93 | 30.9% | 100% | 25 → 0 |

Pooled displayed raw maxima change substantially (trial 93: 4.1241 → 1.1915 rad/ms). This is recalculation with the reconstructed acquisition denominator, not y clipping or graphical scaling. Framewise bout raw/scaled supports are asserted identical; reconstructed bins match independent NaN-ignoring grouped means within 1e-12. Baseline remains [-15,0). New columns 2 and 3 show the same bout signal on matched axes, before and after bin averaging.

The model presumes constant exposure cadence and that FrameIDs represent the capture sequence; it does not claim recovered hardware exposure timestamps. The epoch is anchored to a stable reference's logged AbsoluteTime and retains unknown reference latency and millisecond precision. Existing corrected-v1 data specifically use the arrival clock; this audit provides separate reconstructed-clock evidence rather than silently rewriting those versioned sources.

## Validation

Five acquisition-timing regression tests pass: delayed arrivals with catch-up, one missing ID with nontrivial indexes, missing stable anchors/duplicate IDs, buffer evidence distinct from missing IDs, and stimulus labelling on a correctly reconstructed expected-rate clock. Ten existing corrected-preprocessing tests and fourteen temporal-profile tests pass. Frozen A–D and earlier output variants were retained.
