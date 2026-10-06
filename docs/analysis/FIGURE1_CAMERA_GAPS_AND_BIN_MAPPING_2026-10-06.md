# Figure 1 camera intervals and column 2 → 3 audit

**Subsequent finding:** the user identified the legacy frame-rate/buffer analysis and presumed acquisition-time reconstruction. The causal interpretation below is superseded by [the legacy timing audit and repairs](LEGACY_ACQUISITION_TIMING_REPAIR_2026-10-06.md). The measured raw timestamp and bin-contribution evidence remains valid.

## Long intervals originate in the raw camera log

`scripts/audit_figure1_camera_gaps.py` verifies complete SHA-256 hashes of the original camera and tracking TXT logs and their intake Parquets. Raw excerpts are saved in `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly/audit-camera-gaps-and-bin-mapping-20261006/`. Camera excerpt FrameID, ElapsedTime and AbsoluteTime match the intake values. The corrected frames preserve camera elapsed time, and all 232 >10 ms intervals in the five selected trial windows agree exactly with camera-log intervals.

Across the full recording, 8,344,615 camera rows have **zero missing FrameIDs**. There are **13,936 intervals >10 ms**, all between consecutive FrameIDs. Their median is 13.5054 ms, maximum 95.2944 ms. Median time between long-interval events is 0.844115 s; 10th–90th percentiles are 0.823813–0.865984 s. This recurring pattern is not missing angle rows introduced by preprocessing.

Every long interval (with ten following frames available) is followed within those ten frames by an interval below 0.5 ms; the immediately following interval has median 0.2354 ms. For the audited trial-93 example:

| FrameID | Interval since previous row (ms) |
| --- | ---: |
| 8619055 | 1.1337 |
| 8619056 | 12.9589 |
| 8619057 | 0.2172 |
| 8619058 | 0.1793 |
| 8619059 | 0.1752 |
| 8619060 | 0.1690 |
| 8619061 | 0.2645 |
| 8619062 | 0.2549 |
| 8619063 | 0.1998 |

The original tracking log has finite angles and coordinates for these consecutive rows. The >10 ms event is a timestamp spacing event, not demonstrated loss of angle measurements. AbsoluteTime also shows a 13 ms step followed by multiple rows within the same millisecond. The raw ElapsedTime pattern therefore is not introduced by integer AbsoluteTime rounding.

The recurring delay followed by rapid catch-up is consistent with acquisition software receiving/processing buffered frames or delayed timestamping. This is an inference, not an established hardware diagnosis. The logs do not specify whether ElapsedTime timestamps exposure, driver delivery, or downstream processing. The repository contains no acquisition workflow establishing those semantics. We must inspect that workflow or hardware timestamps before deciding whether the intervals reflect real capture pauses or host timing jitter. Consecutive exported FrameIDs alone do not prove uninterrupted physical exposures.

This distinction also matters to the raw vigor peaks: derivatives divide by these measured intervals, including submillisecond intervals after a delay. If those are software-arrival intervals rather than exposure intervals, both the long-gap rejection and the raw peak amplitudes may be scientifically inappropriate. No substitute clock or threshold was applied during this audit.

## Why the prior columns 2 and 3 appear different

Column 2 previously superimposed two different signals: light individual framewise centred logs, and dark bout medians. Column 3 averages **only the dark bout medians**, repeated on eligible bout frames. It does not average the light trace directly. Column 2's y range was dominated by the low individual logs; column 3 used a much narrower scale. That display obscured the actual correspondence.

Concrete bin: trial 93, **[-15.5,-15.0) s**, 62 contributing frames from two bouts:

| Bout ID | Contributing frames | Bout median |
| --- | ---: | ---: |
| 9910 | 44 | +0.1521219025 |
| 9911 | 18 | +0.2125895818 |

The heatmap bin is `(44*0.1521219025 + 18*0.2125895818)/62 = +0.1696770352`. The individual centred logs in those same frames average **-0.2576260882**. Medians and means need not have the same sign when individual values include strongly negative tails. Each bout median is calculated from its eligible support across the trial window, not recomputed separately within each bin.

The bar fills the full half-second **as a bin summary** even though only short pieces of bout support contributed. It does not assert continuous movement during the whole half-second. NaNs and unsupported frames contribute neither values nor denominator. Empty bins remain NaN. There is no measured temporal shift.

`direct_bout_to_bin_review.svg` removes the extra framewise trace from column 2 and uses identical signed y limits for columns 2 and 3. Raw and scaled bout signals retain identical eligible-frame masks. Its full 400-bin reconstruction check passes within 3.2e-15. `column2_to_column3.svg` and `bin_contributions.csv` show the numerical example. Existing panels and alternatives remain preserved.
