# Delay G: bout-only correction and historical log audit

Follow-up: [all-model confirmation and corrected Hessian audit](FIGURE2_DELAY_LMM_AUDIT_2026-10-09.md).
The prior saved curvature values used the wrong parameter representation;
all 15 fits and test tables reproduce, and the corrected curvature checks pass.

## Author instruction and corrected outcome

On 2026-10-09 the author specified: only frames with bouts count; nonbout
periods are ignored as NaN vigor. The previous all-valid-frame G review was
incorrect for this requested outcome. Preserve it for provenance, but supersede
it for the selected Delay review. The author also authorized separate log and
unlogged display candidates. These are exploratory renders, not panel freezes.

Keep the frozen metric `legacy_distal_angular_speed`, rad/ms. Use the saved
conditional-intensity fields: arithmetic means over finite, detector-valid,
adjacent frames with `moving=True` (positive bout ID). Baseline [-15,0), response
[0,9), CS trials 5â€“94. Each fish ratio is response mean / baseline mean. A
no-bout window stays NaN, never zero; a ratio requires both positive window
means. This measures vigor conditional on bouting, not movement probability or
total activity. Do not silently change cohort, normalization windows or metric.

There are 57 cohort fish (28 control, 29 Delay), 4,811 usable fish-trial ratios
out of 5,130 scheduled; 319 are missing. Trial coverage varies: control 21â€“28,
Delay 26â€“29. Retaining missingness does not establish missing-at-random: changes
in which fish bout can affect the conditional-intensity comparison.

## Did the historical plots use logs?

Yes, in the LogMedian branch. A statement that logs were used only for the
model is correct for the earlier mean-ratio branch, but incomplete for the
February/March code. The exact Git snapshots below were extracted and hashed.

| Historical branch | Plot outcome | Reference | Model preparation |
| --- | --- | --- | --- |
| Feb 6 standard, commit `e4fe3f48d66916174f7fcaa4a5a18be5d49431f3` | response arithmetic mean / baseline arithmetic mean of raw bout vigor | 1 | `log(response+1)`, baseline `log(baseline+1)` |
| Feb 14 LogMedian, commit `2f63ef48361d6e80d1b8e1393894d7e5a3dedf55` | response median of already natural-log vigor minus baseline median of already natural-log vigor | 0 | Step 5 applies `log(x+1)` again to the window summaries |
| Mar 24 LogMedian, commit `bf46bf7b02baaf6d8138d9772f255881f86c3c78` | same as Feb 14; both grouping and plotting scripts are byte-identical | 0 | same second transform |

`3_FishGrouping_LogMedian.py` masks nonbouts to NaN, uses a right-aligned
10-frame rolling median and every-10-frame downsampling, then explicitly calls
`np.log` on positive vigor at line 293. The final save masks nonbouts again.
`5_NormalizedVigorPlotting_LogMedian.py` computes response-minus-baseline window
medians of those log values. Thus it transforms plotted data upstream, despite
the retained `Vigor (deg/ms)` column name. That old label is misleading.

The additional model `log(x+1)` at lines 494â€“495 is a second transform when fed
the LogMedian grouping output. Values at or below -1 make this invalid or
nonfinite. This warrants an input audit; it does not prove which historical
figures or fitted results actually used that upstream file.

The uploaded appearance around 1 is consistent with the raw-ratio branch, but
an image cannot authenticate its generating source. Nor is a difference of
medians of log vigor automatically exactly the log ratio of raw medians under
all interpolation conventions.

Snapshots and source-manifest with SHA-256 are under the analysis bundle's
`historical-comparison/`. No historical artifact was overwritten or restyled.

## Two current display candidates

1. **Ratio:** condition median of individual bout-only arithmetic-mean ratios.
2. **Log ratio:** take the natural log of each individual ratio, then calculate
   the condition median and bootstrap. Reference zero; negative means lower
   response bout vigor. This keeps the same window means, unlike historical
   LogMedian. The two versions use exactly the same fitted tests and marks.

Blue = control, magenta = Delay. Lines are unsmoothed trial medians. Bands are
pointwise 95% percentile CIs for those medians: 5,000 resamples, seed 10, drawing
whole fish trajectories within condition and preserving NaNs. In plain terms,
each resample draws fish with replacement; a selected fish brings every trial
and missing trial with it. Repeat 5,000 times to estimate median uncertainty.
Seed 10 fixes the random sequence. An IQR describes fish spread; a CI describes
uncertainty about a summary; bootstrap is the method producing this CI.

Prefer the ratio version for interpretation (0.9 means 10% lower response bout
intensity relative to baseline). Keep the log version as a sensitivity/display
alternative. Neither CI overlap nor log display substitutes for a model test.

## Corrected fitted results and criticism

The outcome change required fresh fits; all previous all-frame p values are
superseded for this bout-only review. Model: natural log positive bout response,
natural log positive bout baseline covariate, no offset. Random fish intercept
and trial slope for block/spline models; random intercept for local block fits.

| Annotation | Bout-only result |
| --- | --- |
| Joint block interaction | Wald chi-square(8)=100.8922, p=2.8051e-18 |
| D, Holm8 | Train1â€“5 and Test1 |
| M, BH9 | Train2, Train4, Train5, Test1 |
| R raw | Train1 and Test3 |
| R, BH9 | Train1 only |
| Black stars, BH90 | trials 5â€“71 (67 trials) |

D compares individual interactions with Pre5â€“14. M is a centered local block
condition difference; R is a local slope difference. Trial stars test the
adjusted control-minus-Delay log-response contrast at each trial minus its
average in Pre5â€“14. These tests do not directly test the plotted median ratios.
Pre stars mean departures from an average Pre contrast; they are not learning
before training. Each black star marks p_FDR<.05; its size does not encode p.

All main/local, alternate optimizer and random-intercept sensitivity fits pass
numerical gates; all 57 leave-one-fish-out refits pass. Longitudinal residual
skewness .291, excess kurtosis 7.423 and median fish lag1 .136. Tails remain
heavy. Convergence and successful refits do not validate Gaussian Wald tests
or demonstrate effect stability under every sensitivity. Same-data fish-level
training suppression difference .07835, bootstrap CI [.02964,.12808], is a
different estimand. Maximum bootstrap endpoint change from 2,500 to 5,000 draws
is .02760 ratio units. These results remain exploratory.

The Plans' fish-level independence, contrast identity and onset/extinction
distinctions are appropriate. Covariance, coverage, day/rig sensitivity and
calibration remain unresolved. Legacy separate-per-trial random-intercept fits
have one observation per fish and cannot separately identify that variance.
Separate correction families are not one global paper-wide correction.

**Fitting recommendation:** keep observed medians in the main descriptive
panel. Use log-response mixed effects with log-baseline adjustment for inference.
Show fitted condition contrasts separately. Compare the current global spline
with a phase-aware piecewise/spline model having explicit pre/training/test
boundaries; the global spline can carry training curvature into Pre. The 67
stars, including Pre, make this sensitivity particularly necessary before
accepting trial inference. An arbitrary polynomial or fitted exponential is
not supported as the primary fit without a biological shape hypothesis. Do
not infer onset from first significance or extinction from loss of significance.

## Artifacts

Fresh analysis:
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/20261009T141045627289Z-delay-boutonly-lme/`.
It preserves exclusions, model input, test tables, fits, residuals, influence
refits, bootstrap draws, historical snapshots and hashes. Figure1, H/I and the
shared full assembly are unchanged. Panel I remains unavailable/inconclusive.
