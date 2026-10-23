# Delay response-profile model draft

The 2026-10-22 exploratory pilot implements part of
[Plan 5.1](../../Plans/05.1_FIGURE_4_BOUT_AVAILABILITY_AND_POOLING.md).
Its [scoped draft state](../paper/figures/figure-04/delay-model-draft-state.json)
binds the editable SVG, minimal HTML review, legend, analysis, semantic registry
and rendered layout evidence. It selects no scientific method or learner rule.

The external directory is
`J:\ClassicalConditioning Outputs\current\reviews\figure4-delay-model-draft-20261022`.
JOAQUIM must be connected. The [starting ticket](../../records/figure-reviews/figure-04/delay-model-draft-start-20261022.json)
captures previous authority and the authenticated Figure 3 exploratory atlas.
Original versioned corrected-v1 metric, corrected-v2 movement and corrected-v3
temporal files are authenticated against that atlas and temporal completion
metadata. The production source resolver is unchanged.

## Data and model

All 57 source fish remain in the inventory: 28 controls and 29 Delay fish.
Saved legacy-WIP flags define 16 learner-flagged and 10 nonlearner-flagged Delay
fish; three unclassified Delay fish are outside the modeled groups. Controls
are unsplit. These retrospective labels use overlapping acquisition/recovery
data and are not validated learner classes.

The outcome is signed conditional bout log vigor, with the existing estimator,
0.5-s bins, a trial-specific [−20,0) s baseline and coverage ≥0.9. Catch trials
25, 39, 53, 59 and 65 are summarized by median within fish. No-bout intensity
remains unavailable; technically inadequate coverage remains unavailable.
Observed moving fraction is plotted separately with equal fish weight.
Numerical contributor counts are saved per fish and time bin.

The joint Gaussian ML model uses group-specific cubic B-splines (six degrees
of freedom) over −5 to 12 s. It fits 1,756 observed bins from 54 fish. A fish
intercept/time-slope covariance passed the original fit but only 354/399
whole-fish bootstrap refits. The entire bootstrap was restarted for a fish
intercept model; all 399 refits passed. No draws from different structures are
mixed. Both structures' diagnostics are retained. This stability rule supports
an exploratory display, not scientific selection of the residual covariance.

Panels show group curves, direct group contrasts, fitted contrast maxima,
observed movement, two illustrative conditional fish fits, and maxima timing.
Population curves have pointwise bootstrap ranges. Contrasts have nominal
joint bands across three contrasts and 18 pre-US bin centers. Extrema use
the 0.25–8.75 s grid and descriptive bootstrap ranges, not calibrated peak-time
confidence intervals. Conditional fish fits share a population shape with
fish intercepts; these are not separate per-fish spline fits.

Control-minus-learner-flagged suppression has a fitted maximum at 4.25 s, but
its bootstrap timing range extends from 3.75 to 8.75 s, with about 22% of
maxima at a search boundary. The control-minus-nonlearner-flagged maximum is
at the final bin, with about 52% boundary maxima. Precise biological peak
timing is therefore unsupported. No p-values or independent learning claims
are supplied. Informative bout availability and temporal residual dependence
remain limitations; a penalized GAMM comparison has not been run.

## Build and review

The [builder](../../scripts/build_figure4_delay_model_draft.py) has sequential
`extract`, `model`, `render` and `publish` stages, with source/output guards.
The committed ticket describes this specific run; it is not a generic permission
to overwrite a selected draft. Existing cache files authenticate before reuse.
The [browser checker](../../scripts/check_figure4_delay_model_draft.cjs) accepts
the external draft directory and a temporary PNG destination. It measures the
saved SVG through the shared layout collector at 600 dpi. Temporary inspection
images are removed after review.

The draft uses the resolved `nature-main` profile on a native 183 × 135 mm
canvas with shared 1-mm invisible frame insets. Exact saved-artifact layout
evidence passed, and the SVG/legend were visually reviewed in Edge and the
built-in browser. This does not establish full production compliance or freeze
the figure. Four [known-answer tests](../../tests/test_figure4_profile_models.py)
cover contrast signs, whole-fish resampling, missing intensity and conditional
Gaussian predictions. Future model comparisons and validation remain owned by
Plan 5.1 and the implementation index.
