# Figure 2 A/B numerical review, 19 October 2026

Author request: investigate why B is generally closer to baseline than A;
identify scientific or analytical causes without assuming an error.

## Scope and evidence

The reviewed working SVG is SHA256
`3daba1407e6a70b3942af77acceae3d4bfd1f686e3cb5bca3d0a7ac63bbcb4ed`.
A uses the selected immutable 57-fish freeze; B uses the selected 179-fish
draft, including 90 controls and 89 fixed-3sTrace fish. The four timing-flagged
fish remain excluded from B/E/H. Figure state, artwork, statistics and scientific
definitions were not changed by this review.

Reproduction: `scripts/audit_figure2_heatmap_ab.py`, then
`scripts/audit_figure2_heatmap_ab_followup.py`. These are exclusive-output review
scripts, not figure builders. Preserve the existing evidence directory before
any further run. Numerical evidence is on JOAQUIM at
`J:/ClassicalConditioning Outputs/current/reviews/figure-02-3strace/matched-delay-statistics/ab-analysis-review/`:
`audit.json`, `followup.json`, `pooled-summary.csv`,
`individual-bin-summary.csv`, `fish-window-summary.csv`,
`subgroup-pooled-summary.csv`, `phase-summary.csv` and `raster-follow-up.json`.
JSON evidence binds source/generator hashes and the selected SVG.

## Findings

1. **More fish reduce fluctuations in pooled cells.** Median contributors per
   cell are A Control 21 / Delay 25 and B Control 80 / Trace 79. Finite-cell
   fractions are 75.9% / 81.9% versus 89.3% / 88.8%. Control baseline-cell SD
   falls from 0.04240 in A to 0.02336 in B. Using 1,000 fixed-size B subsets of
   28 controls without replacement raises median absolute baseline colour from
   0.01626 in full B to a median 0.02652, close to A's 0.02796. Thus sample size
   accounts for much of the background difference. Subset quantiles are
   descriptive resampling diagnostics, not confidence intervals or tests.

2. **The mean CS shift is also smaller and more heterogeneous in this Trace
   cohort.** All values below are averages of the selected population cells,
   in natural-log vigor difference units. Phases use acquisition Pre5–14,
   Train15–64 and Test65–94. The common CS window [0,9) precedes both US times.

   | Conditioned population | A Delay | B 3sTrace |
   |---|---:|---:|
   | Mean [0,9), all 90 trials | −0.05250 | −0.01117 |
   | Mean [0,9), Pre | −0.00116 | +0.02668 |
   | Mean [0,9), Train | −0.07625 | −0.02573 |
   | Mean [0,9), Test | −0.03001 | +0.00050 |
   | Training own post-US 3 s window | +0.08929 | +0.21250 |

   Post-US windows are [9,12) for Delay and [13,16) for Trace; training includes
   the accepted catch trials. Among per-fish [0,9) means over all 90 trials,
   28/29 Delay fish and 55/89 Trace fish are negative. These signs are descriptive
   and are not learner classifications. The individual-bin median absolute
   deviations are much more similar: 0.10166 Delay versus 0.09302 Trace, so the
   paler pooled B does not mean every individual fish remains near baseline.
   The original B subgroup has pooled [0,9) mean +0.00012; additions have
   −0.02052. Expansion did not simply dilute a stronger negative original mean.

3. **Median centering does not force an arithmetic population baseline mean
   to zero.** Every defined fish/trial baseline-bin median is zero to numerical
   precision. Nevertheless the pooled baseline mean is about −0.022 for A
   versus −0.007 for B. Median/mean differences, skew and availability therefore
   contribute to the background offset. This follows the accepted estimator;
   it is not a hidden second normalization or zero imputation.

## Checks and limits

- Frozen A bin-table hashes match the selected freeze. Each included B cached
  bin-table hash matches its calculation record. Independent regrouping
  reproduces both selected pooled tables exactly. Baseline subtraction errors
  and duplicate fish/trial/bin counts are zero. Undefined baselines propagate
  missing values rather than zeros (166/5130 A fish/trials, 150/16110 B).
- The binning/calculation implementation differs from A's freeze only in
  canonical source-identity resolution. Detector configuration and helper
  implementations, including cadence estimation, are identical to A's saved
  source. Recorded cadence ranges overlap. Included heatmap records have no
  exported-FrameID or buffer-capacity loss flags.
- Fresh authenticated recorded-event reads verify all 89 conditioned B fish
  against the fixed-13 s specification, including per-trial consistency and
  expected paired/catch structure. Increasing-trace timing did not explain B.
- Every cell-centre colour in all four 90×40 maps exactly matches its selected
  numerical matrix under the same `managua_r` ±0.25 scale. B's PNG is byte-pixel
  equal to the current rendering recipe. A's initial full-pixel comparison
  differed only along historical nearest-neighbour raster boundaries; the
  follow-up confirms all 3600 cell-centre colours per A map match exactly.
- The heatmap measures vigor conditional on eligible detected-bout frames.
  Near-zero colour does not establish unchanged bout frequency, total movement,
  or lack of learning. These diagnostics are not new inferential tests.
- A combines 2022/2024 recordings and B uses 2023 recordings. Protocol and
  recording cohort/year are confounded, so this review does not attribute a
  biological cause to Delay versus Trace or establish that learning is weaker.
- Cached original-frame summaries were authenticated and independently
  repooled; large raw tracking/frame files were not regenerated. Baseline
  estimator and missingness sensitivity beyond the reported summaries remain
  possible further analyses, rather than changes to the accepted definition.

Conclusion: the observed contrast has a supported combination of averaging,
cohort response heterogeneity and median-centering effects. No analysis or
colour-mapping defect explaining the contrast was found within this scope.
