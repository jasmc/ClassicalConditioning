# Delay matched summary comparison and storage recovery

The author requested historical LogMedian, a matched mean/median comparison,
and all versions in one file. The standing metric decision remains
`legacy_distal_angular_speed`; all new analyses ignore non-bout frames.
These are exploratory outcome/model alternatives, not new panel freezes.

## Storage and reproducibility

J was inaccessible. The author explicitly authorized F if inputs match.
All **402** recorded source-data hashes in the local pre15 cohort sidecar
match the original F artifacts, including frames, movement state, outcomes,
events and cohort records. The technical cohort hash remains
`9ef4b9297c939e0d34a99a4d606d0f9c8c2a42a0a7b9aee20d947823ef66f2d5`.
No raw/frame files were copied. J exports were preserved, and the new F
previews are labeled regenerated rather than byte-identical recovered exports.

The inspected, rebuildable uv package cache was removed, freeing an observed
1,148,518,400 bytes on C. Installed environments, scientific data, Git,
historical exports and the active bundled runtime were preserved. Details:
`reviews/disk_cleanup_delay_20261009.json`.

F root:
`F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/`.
`f-recovery-current.json` identifies the completed regenerated versions.

## Exact definitions

Use valid, adjacent (`FrameStep == 1`) moving/bout frames, with finite legacy
angular speed. Verify `moving == (bout_id > 0)`. Baseline [-15,0), response
[0,9), CS trials 5–94; no-bout windows stay missing. The 57 fish (28 control,
29 Delay) have 5,130 scheduled rows and 4,811 matched eligible rows.

| Version | Per-fish outcome | Population display |
| --- | --- | --- |
| Arithmetic means | mean(response)/mean(baseline) | median across fish; reference 1 |
| Log mean ratio | ln(mean response/mean baseline) | median across fish; reference 0 |
| Literal medians | median(response)/median(baseline) | median across fish; reference 1; descriptive only |
| Historical LogMedian summary | median(ln positive response frames) minus median(ln positive baseline frames) | median across fish; reference 0; fresh LMMs |

The first two and literal medians retain finite zero vigor. LogMedian excludes
nonpositive values before logging: 18,610 baseline and 10,859 response frames.
Therefore historical LogMedian differs from literal medians both because of
this positivity filter and because even-count median interpolation does not
commute with logging. The largest individual literal-median ratio versus
exp(LogMedian difference) discrepancy is 0.17238. The matched-input plot
states the common initial bout mask and the difference in zero treatment.

The new historical-summary version adds no historical rolling median or
downsampling. It logs once and does not repeat the legacy second log(x+1).
It reproduces the summary on current corrected inputs, not the entire old
February/March pipeline. Historical references remain in the earlier audit.

Every descriptive bootstrap uses **5,000 whole-fish draws within condition,
seed 10**, carrying all scheduled trials and missingness together. The mean,
literal-median and exponentiated LogMedian comparisons verify identical draw
indices. Bands are pointwise percentile 95% CIs. The separate IQR page shows
the middle 50% of observed fish with the same mean-ratio median lines.

## Model results and criticism

All 15 regenerated mean main/local/sensitivity fits reproduce the earlier
results and pass numerical gates. All four mean global/phase comparison fits
pass. The 14 LogMedian attempted fits comprise five block/global/phase/
sensitivity fits and nine local fits: **13 pass; Test3-local fails** with
random-intercept variance about 1.35e-12. Its M/R tests remain NaN and display
`n/a`. Both BH9 families retain nine scheduled tests, treating unavailable
tests as p=1 during adjustment, rather than shrinking the family. No fallback
or changed singularity threshold was used.

| Outcome / trial model | D Holm8 | M BH9 | R raw / BH9 | Black-star trials |
| --- | ---: | ---: | ---: | --- |
| Means / global | 6 | 4 | 2 / 1 | 5–71, including all Pre |
| Means / phase | 6 | 4 | 2 / 1 | 19–70 |
| LogMedian / phase | 6 | 5 | 2 / 2 | 19–71 |

D compares each non-Pre block interaction with Pre. M is the local condition
difference at centered trial; R is the local condition slope difference.
The trial tests compare fitted control-minus-Delay differences with their
average Pre contrast and use BH90. They do not directly test displayed medians.
All 57-fish omission checks in both phase analyses pass, as do the 57 paired
global/block omission checks for means. Numerical success does not establish
Gaussian Wald calibration or learning onset.

LogMedian does **not** improve tails here: phase residual excess kurtosis
8.027, compared with 7.363 for means. Its Test3 local fit also fails. A median
is defensible for a typical-intensity claim, but the present results do not
justify declaring historical LogMedian statistically better. For average
intensity, retain means; strong bursts belong to that estimand. Literal
window medians avoid the frame-level positivity filtering, but this literal
median version currently has descriptive curves only, not its own fitted
inference. Do not transfer either outcome's stars onto the literal-median plot.

The Plans' fish-unit, condition-contrast, diagnostics and simultaneous-band/
persistent-onset requirements remain appropriate. Heavy tails, residual
covariance, day/rig sensitivity and informative missing bout windows remain
unresolved. The phase-aware model reduces global-curve borrowing across phase
boundaries, but no first-star onset or loss-of-significance extinction claim
is justified. The outcome measures conditional bout intensity, not movement
probability or total activity.

## Single-file deliverable

`all-versions-comparison/Delay_all_versions_comparison.pdf` contains 11 pages:
definitions, band/test explanations, matched summaries, CI/IQR, mean ratio,
log mean ratio, mean observed/fitted contrasts, mean phase marks, historical
LogMedian marks, mean/LogMedian fitted contrasts, and model criticism.
Its text was extracted and all 11 rendered pages visually inspected. Labels,
clipping, stars, unavailable tests and captions were reviewed. The adjacent
`pdf-manifest.json` records its SHA-256 and source images. Old all-frame panels
are superseded and explicitly discussed rather than used as current candidates.

## Mean of logged vigor check — 2026-10-09

The author requested a fifth descriptive outcome: mean(ln positive response
bout frames) minus mean(ln positive baseline bout frames). Population lines
remain medians across fish. Fresh authenticated frame extraction reproduces
all previous window LogMedian values and positive-frame counts; both outcomes
have the same 4,811 usable rows among 5,130 scheduled rows, for the same 57 fish.
No bands, statistical marks, LMM fits, smoothing or pseudocounts are added.
Five window-definition tests pass, including mean-log versus log-mean,
nonpositive/empty handling and unit-conversion cancellation.

The three-panel preview shows LogMedian, mean-log and their overlay, all on
shared y limits. Blue is control; magenta is Delay. Solid overlay lines are
mean-log; dashed lines are LogMedian. The broad observed pattern is similar,
with trial-specific differences. No scientific preference is selected from
this exploratory visual comparison.

Outputs are under
`F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/20261009T174114195765Z-delay-meanlog-check/`.
The PNG/SVG, condition medians, fish-trial window summaries and authenticated
source/settings/export records preserve this candidate separately.


## Selected G freeze — 2026-10-09

The author selected native Historical LogMedian with its own D/M/R and
phase-aware LMM statistics. Scoped current selection:
`configs/paper-figures/figure2-G-logmedian-freeze-20261009.json`.
The immutable freeze is `F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/frozen/20261009-G-historical-logmedian/Fig2_G_Historical_LogMedian.freeze.json`.
It passed the version1.1.1 figure gate with no exceptions. Final standalone
size is183×98mm; the whole assembly is older and was not changed.

Blue is control; magenta is Delay. Per-fish values are median(ln positive
response bout frames) minus median(ln positive baseline bout frames), then
median across fish. Windows are[-15,0) and[0,9), CS5–94. Non-bout and empty
windows remain missing. Bands are pointwise95% CIs from5,000 whole-fish
resamples within condition, seed10. Data and fitted results were not changed.
D Holm8:6; M BH9:5; R raw/BH9:2/2; phase trial BH90 stars19–71.
Test3 local M/R remains unavailable and is shown n/a; heavy tails and
onset-inference limitations remain. The primary plot is observed medians,
with model tests separately aligned above it, not a fitted curve.

All alternatives, including mean(log frames), are documented in the single
self-contained `reviews/figure2_delay_all_versions_20261009.html`. This
supersedes the old comparison PDF and loose previews mentioned above.
Its embedded verified ZIP contains deduplicated alternative data, model
outputs, code and vector/unique raster sources; selected scientific payload
resolves to immutable canonical freeze data. Redundant PNG/PDF derivatives
and PDF QA renders were discarded after consolidation; their hashes and
disposition remain in the manifest. Historical audit snapshots retain their
original paths, now resolvable through this manifest.

Cleanup removed353 stale files
(34,286,026bytes).
`reviews/delay_versions_cleanup_20261009.json` records every original hash,
archive mapping, discarded derivative and path. Original Digested Data,
other panel reviews and all historical freezes were preserved; J untouched.
Four exporter tests and five window-definition tests pass.
