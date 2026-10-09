# Two single-fish C-mode review versions

Requested 8 October 2026. F Delay 20221115_07; G 3 s Trace 20230307_12; H Control 20221115_09. Trials 5-94, displayed [-20,20) s, baseline [-15,0) s. Only new review outputs are created.

## Terminology and shared upstream steps

**Framewise raw vigor** is absolute wrapped distal bend angular change divided by the inferred acquisition interval, in rad/ms. It is calculated before detection, on valid adjacent frames. The bout detector uses a centred median-smoothed version of that signal, its rolling max-minus-min envelope, and duration/gap/peak rules. **Bout vigor** or **bout summary** is calculated after detection from values belonging to the detected bout. The same word vigor had been used for these different stages; neither the bout median nor the final C value is needed to detect the bout.

All camera/protocol/angle/coverage sources and detector settings match the preceding fresh review. Eligible means detector-valid, moving, positive bout ID, finite positive raw angular speed. Excluded observations stay NaN. Natural log is used in both variants. There are no new tracking filters, thresholds or interpolated observations.

## Version 1: bout medians, no binning anywhere

1. Select the visible trial observations.
2. Take the natural log of eligible raw frame vigor.
3. Calculate each detected bout's median log vigor from its eligible observations inside the visible [-20,20) crop.
4. Repeat that median only on the original eligible samples belonging to the bout.
5. Calculate P10, P50 and P90 directly from the repeated values at eligible baseline samples in [-15,0), using NumPy linear quantiles. Every baseline sample gets one vote; longer eligible bout portions therefore contribute more votes. No half-second calibration bins are constructed.
6. Apply `C = clip((sample_value - P50) / ((P90-P10)/2), -1, +1)` to the repeated sample values.
7. Display those values on their sample intervals. A sample interval is centred on its timestamp and has width equal to acquisition cadence, clipped at the visible edges. Identical adjacent values are encoded as one vector run, without averaging; gaps in eligibility or FrameID remain gaps. Run widths retain actual sample support duration. This is a lossless representation of the repeated sample signal, not a 0.5 s display.

The visible-crop bout median is deliberately retained for this requested comparison. The preceding critique's crop-dependence and cross-stimulus mixing concerns still apply; this review does not approve that choice.

## Version 2: direct log vigor, 0.5 s bins, no bout medians

1. Select the same visible observations and use exactly the same eligibility mask.
2. Take natural log of eligible raw frame vigor.
3. For each half-open 0.5 s interval, average the finite log values of samples inside that interval. There is no per-bout median replacement. Every trial has 80 scalars; empty intervals remain NaN.
4. Calculate P10, P50 and P90 from the finite 30 possible baseline-bin scalars in [-15,0), with one vote per finite bin.
5. Apply `C = clip((bin_value - P50) / ((P90-P10)/2), -1, +1)` to the bin scalars.
6. Display one explicit 0.5 s rectangle per trial/bin, using 81 time edges.

The within-bin scalar is **mean of logs**, not log of mean raw vigor. It still describes intensity conditional on detected eligible movement rather than all observed activity.

## Display and comparison interpretation

Both variants use `managua_r`, symmetric C limits [-1,+1], zero at palette midpoint #582948, and the same panel geometry and CS/US guides. A nonpositive or missing baseline scale makes the corresponding trial undefined/NaN. Empty support and undefined scale display as black. No scale is borrowed and no missing observation is zero-filled.

The user explicitly selected sample-based baseline statistics for Version 1. Thus this comparison changes three coupled choices: bout summarization, display resolution, and the baseline voting unit. It does not isolate the bout-median operation alone. The separate `C_trial_baseline_statistics.csv` records each trial's reference and denominator for both versions.

## Files and validation

- `C_BoutSamples_FGH.png` and `C_DirectBins_FGH.png`: the two full F/G/H review galleries.
- Six panel PNG/SVG/PDF files: each version and fish separately.
- `Panel*_sample_data.parquet`: per-observation time, FrameID, eligibility, detected bout ID, eligible raw/log vigor, repeated bout-median log and sample C values. The raw column here is masked to eligibility; raw framewise speed was calculated before detection.
- `Panel*_display_sample_runs.csv`: exact vector run intervals, original frame bounds, sample counts and C values for Version 1.
- `Panel*_direct_halfsecond_bins.csv` and Parquet: 80 explicit bin scalars and C values per trial for Version 2.
- `manifest.json`: source/code hashes, complete definitions, undefined trials and exported geometry checks.

The direct scalar bins are independently checked against the saved scientific audit's directly aggregated frame logs. For sample display, every finite sample is accounted for by a constant-valued contiguous run and the SVG checks each run's fill, position and width against its original sample intervals. The binned SVG checks require exactly 7,200 cells per panel and matching scalar fills. Baseline medians are checked after centring/scaling in each version's own voting unit.
