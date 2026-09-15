# Legacy Learner-Variant Behavior Matrix

## Status and scope

This document characterizes the four preexisting learner scripts as executable
legacy behavior. It does not select a canonical classifier. Their outputs are
not interchangeable, and none should be treated as the corrected analysis.

## Shared structure

All four scripts:

- consume stage-5 per-fish/per-trial normalized-vigor artifacts;
- split configured 10-trial blocks into ordered 5-trial sub-blocks;
- build acquisition and extinction/recovery epoch-change features;
- require at least three trials in every included 5-trial block and six total
  trials in each active epoch;
- retain only fish with complete data for every selected feature;
- require at least ten complete fish before classification;
- use control-anchored mixed models when at least three controls are available,
  with fallback to unanchored models;
- extract fish-specific epoch random slopes and associated uncertainty;
- use control covariance information to identify multivariate departures;
- produce classifications plus optional diagnostics and per-fish figures.

## Behavior matrix

| Dimension | `6_LearnersQuantification.py` | `6_LearnersQuantification_new.py` | `6_LearnersQuantification_improved.py` | `6_LearnersQuantification_WIP.py` |
| --- | --- | --- | --- | --- |
| Input discovery | Newest matching gzip pickle in pooled-data root | CSV preferred, then newest pickle across root and legacy subfolders | Same broad CSV/pickle search | Broad search plus explicit-path, NaN-filter, and selected-fish controls |
| Modeled response | `log(Mean CR + 1)` adjusted for `log(baseline + 1)` | Same | `log(Mean CR)` adjusted for `log(baseline)` | `log(Normalized vigor)` by default; baseline is not a covariate |
| Acquisition late epoch | Late Train + Early Test | Train 6-9 + Late Train + Early Test | Train 6-9 + Late Train + Early Test | Train 6-9 + Late Train + Early Test |
| Extinction early epoch | Late Train + Early Test | Train 6-9 + Late Train + Early Test | Train 6-9 + Late Train + Early Test | Train 6-9 + Late Train + Early Test |
| Multivariate gate | Ledoit-Wolf Mahalanobis distance | Ledoit-Wolf Mahalanobis distance | One-sided whitened joint statistic; Mahalanobis is diagnostic | Same one-sided joint statistic |
| Threshold | Bootstrap 75th percentile, 1,000 resamples | Bootstrap 75th percentile, 50 resamples; optional theoretical chi-square | Empirical 90th control percentile (`alpha=0.10`) | Empirical 95th control percentile (`alpha=0.05`) |
| Direction rule | Point and conservative CI-shifted votes | Point and Gaussian directional-probability votes | All features must point in the expected direction | All features must point in the expected direction |
| Final decision | Distance outlier plus two votes | Distance outlier plus two votes; probability threshold 0.60 | Joint statistic above control threshold plus unanimous direction | Same, with stricter control threshold |
| Control reference | Full control sample | Full control sample | Leave-one-out mean for each control fish | Leave-one-out mean for each control fish |
| Main CSV default | Disabled | Enabled, canonical filename | Enabled, canonical filename | Enabled with `_wip` suffix |

## Feature semantics

Acquisition is coded as an expected negative change in movement response.
Extinction/recovery is coded as an expected positive change. The original
script uses only Late Train and Early Test for the shared late-training epoch;
the other three use six 5-trial blocks spanning Train 6 through Early Test.
Because coverage is required in every included block, this changes both the
feature estimate and the eligible cohort.

## Uncertainty and covariance

The mixed-model routes combine fixed epoch-effect uncertainty with conditional
random-slope variance. Non-finite or non-positive total standard errors are
floored to `0.01`. Mahalanobis routes use Ledoit-Wolf shrinkage, pseudoinverse
fallback, and per-fish standardized-Euclidean fallback. The improved and WIP
routes instead use per-fish directional z-scores, a shrinkage-whitened
one-sided quadratic statistic, and finite-sample empirical p-values.

## Validation and failure behavior

The original and new scripts include normality, feature-correlation, PCA, and
learning-versus-performance diagnostics. The improved and WIP routes calibrate
their joint threshold directly from controls and use leave-one-out control
references to reduce self-influence. Model failures commonly trigger fallback
models rather than hard failure; robust readers also try alternate pickle
decoders. These success-shaped fallbacks are legacy behavior to preserve and
report, not a design for the corrected classifier.

## Consequence

The scripts disagree on source-file precedence, epoch definitions, response
mathematics, thresholds, uncertainty voting, false-positive target, and output
defaults. A single label such as `learner` therefore does not identify a stable
scientific estimand. Learner classification remains deferred until corrected
outcomes and cohorts are frozen.
