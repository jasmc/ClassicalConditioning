# Handoff: bout-only log vigor scaffold

**Reference:** Figure 2 D/E version B, frozen 2026-10-09: [review](../reviews/figure2_B_freeze_20261009/comparison.html), [D freeze](../reviews/figure2_B_freeze_20261009/D.freeze.json), [E freeze](../reviews/figure2_B_freeze_20261009/E.freeze.json). This selection covers D/E only.

## Raw vigor → B

- Authenticate corrected metric, movement-state and protocol files against source manifests. Join frames one-to-one by FrameID/measured AbsoluteTime; verify identity and chronology. Read needed windows without copying recordings.
- Raw metric: `legacy_distal_angular_speed_rad_per_ms`, absolute wrapped frame-to-frame change in the sum of local tail angles divided by measured frame interval (rad/ms).
- Mask invalid/discontinuous and **no-bout frames to NaN before every calculation**. For logs retain finite, strictly positive bout values. Never zero-fill, interpolate missing periods or add pseudocounts.
- Align to CS onset. Baseline **[-15,0) s**; response **[0,9) s Delay**, **[0,13) s 3 s Trace**, stopping before expected US. Endpoints are left-inclusive/right-exclusive.
- Log eligible frames first. **B_trial = median(ln(response vigor)) − median(ln(baseline vigor))**. Empty windows yield NaN and exclusion. Save sample counts/exclusion reasons. No historical rolling smoothing/downsampling is restored.
- Median eligible trial B per fish/block: PT10–14, ET65–69, LT90–94; minimum eligible trials D=3/E=1. Plot equal-fish median/IQR. Zero means unchanged; `100*(exp(B)-1)` expresses proportional change.
- B measures typical **bout-frame intensity**, not movement frequency or total activity. Longer bouts contribute more frames; individual bouts are not equally weighted. No-bout exclusions can be informative.

```python
v = np.where(valid & adjacent & moving & np.isfinite(raw_vigor), raw_vigor, np.nan)
L = np.full(v.shape, np.nan)
positive = np.isfinite(v) & (v > 0)
L[positive] = np.log(v[positive])
t = (absolute_time_ms - cs_onset_ms) / 1000
base = L[(t >= -15) & (t < 0)]
response = L[(t >= 0) & (t < response_end_s)]  # 9 or 13
base, response = base[np.isfinite(base)], response[np.isfinite(response)]
B_trial = np.median(response) - np.median(base) if len(base) and len(response) else np.nan
```

## Adapt the scaffold, preserving selected definitions

| Panel | Required adaptation |
| --- | --- |
| Fig1 E | Preserve selected fish/trials. Raw trace: native rad/ms with NaN gaps. New B-consistent log trace: `L(t)-baseline_median`. Any heatmap overlay must use the exact saved bins. |
| Fig1 F/G/H | New B candidate: each trial/time bin = `median(L_bin)-baseline_median`; save bin width, endpoints and eligible counts. Empty bins stay NaN. |
| Fig2 A/B/C | First build fish-level trial/time cells; explicitly choose equal-fish population median or mean and save contributing-fish counts. |
| Fig2 D/E/F | Reuse exact frozen D/E exports/data/stats. F needs authenticated 10 s Trace inputs and protocol-specific windows/cohort. |
| Fig2 G/H/I | Compute B for every required trial, then equal-fish summaries/uncertainty. Refit any model for this outcome; old arithmetic-mean ratio/LMM fits cannot be relabeled. |

**Coverage:** D/E caches contain only 15 selected block trials. Full heatmaps/trajectories require additional authenticated frame windows; arithmetic means cannot recover log medians. C/F/I remain unavailable without authenticated 10 s Trace inputs.

**Existing Fig1 F/G/H:** selected `C_BoutSamples` is quantile-scaled, not B. Version 4 uses 0.25-s log-bin means and a [-20,0) mean-bin reference. Preserve both; make B a separate candidate. Resolve current scoped E/F/G/H records before the older assembly. Current examples: F=20221115_07, G=20230310_08, H=20221115_09.

## Statistics, validation and export

- Fish are statistical units; pair eligible fish across blocks. Frozen D/E use exact two-sided sign tests for paired changes/zero, reporting and excluding zero ties. Independent comparisons use Brunner–Munzel t tests (rank/probability effects; not pure median differences), requiring ≥10 fish/group in this recipe.
- One **Holm36** family: 12 zero-reference +12 paired-change +6 between-group block +6 between-group change tests. Define an appropriate family before analysing other panels. Pointwise bootstrap intervals are separate from corrected p-values; IQR is not CI. Descriptive example panels need no invented stars.
- Review fish independence/day/tank design. Frozen tests do not adjust clustering; outcome/test selection remains exploratory.
- Save source hashes/settings, auditable window/bin summaries, counts/exclusions, fish outcomes, summaries/tests/effects and SVG/PDF/PNG/exporter sidecars. Verify no-bout outliers cannot change results, empty windows stay NaN, unit conversion cancels in `L-b`, and joins do not duplicate fish/frames.
- Register semantic elements/mappings, keep zero behind data and show missing cells explicitly. Style a separate candidate at final size, inspect it, run `freeze_figure.py --check-only`, then `--output` after passing.
- D/E are **54.9 × 53.2 mm** at provisional 183-mm assembly width. Enlarge the older assembly's shorter row-2 boxes; squeezing exports changes effective fonts/strokes. Whole-figure freeze requires its own assembly checks.

Entry points: [mask](../src/classical_conditioning/analysis/bout_vigor.py), [trial calculation](../scripts/review_figure2_block_log_median.py), [tests](../src/classical_conditioning/analysis/bout_block_statistics.py), [renderer](../scripts/prepare_figure2_B_freeze.py), [freeze gate](../scripts/freeze_figure.py). Frozen bundle includes small code/specification snapshots and source manifests.
