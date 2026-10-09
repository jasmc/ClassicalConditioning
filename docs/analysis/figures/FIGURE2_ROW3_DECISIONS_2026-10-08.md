# Figure 2 G–I: metric decision and future LME annotations

The author explicitly reaffirmed **freeze legacy metric** on 2026-10-08.
For this row, the frozen metric is **tail bend angular speed**,
`legacy_distal_angular_speed`, frame column
`legacy_distal_angular_speed_rad_per_ms`, units rad/ms. This reaffirms
`docs/analysis/METRIC_SELECTION_2026-10-06.md`; it does not restore historical
interpolated timing or legacy preprocessing. The pre-CS baseline remains the
previously frozen **[-15,0) s**. Cohort, coverage, response-window, upstream
signal/alignment and statistical approval remain separate and pending.

The author's “LLM stats test” was clarified by the author to mean **LME/LMM
linear mixed-effects modeling**. Preserve the intention to add saved,
model-derived statistical annotations to G/H later, using the uploaded draft
as a layout reference. Do not interpret the prior no-marks descriptive
candidate as a decision to abandon statistical annotations. No existing stars
or letters in the upload are accepted test results.

## Historical code recovered

The historical source at both `2f63ef48361d6e80d1b8e1393894d7e5a3dedf55`
(2026-02-14) and `bf46bf7b02baaf6d8138d9772f255881f86c3c78`
(2026-03-24) is `5_NormalizedVigorPlotting_LogMedian.py`. It is byte-identical
between these two snapshots. It contains `run_trial_by_trial` and the draft's
annotation hierarchy. Exact snapshot exports and hashes are stored in the
versioned SSD row-three decision bundle under
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/`.
The currently retained historical route is
`legacy/scripts/5_NormalizedVigorPlotting.py`; it is not assumed byte-identical
to the February/March file or the exact generating code of the uploaded image.

| Draft lane | Historical calculation |
| --- | --- |
| Gold D | Individual condition × block interaction coefficient p-values < .05 from the global model; not a single joint interaction test, and not multiplicity-adjusted in that lane |
| Gray M | Local block mean difference, BH/FDR-adjusted across blocks |
| Red R (raw) / R (FDR) | Local block condition × centered-trial slope difference, raw / BH-adjusted across blocks |
| Black stars | Separate per-trial condition tests, BH/FDR-adjusted across trials |

These models use log(response + 1), with log(baseline + 1) as a covariate and
fish grouping. They do not directly test the plotted median ratio. The
historical per-trial mixed models normally have one observation per fish and
are problematic for estimating fish random effects. Failed fits are skipped.
This is provenance of the old design, not acceptance of its inference method.

The current longitudinal replacement is
`src/classical_conditioning/analysis/inference/learning_onset.py`, with its
parameter reference in `docs/analysis/07_LEARNING_ONSET_LME.md`. The exploratory
annotation renderer `scripts/render_figure2_legacy_stats_lme_review.py`
already combines descriptive ratios, inference lanes and model contrast bands,
but currently supports a saved **angular-L1 Delay** model. That model failed
influence diagnostics and did not localize simultaneous onset. It cannot be
reused as inference for the frozen legacy metric, or transferred to Trace.
Future annotation requires metric/cohort/window-matched saved fits, named
contrasts, multiplicity rules and accepted diagnostics. No model was rerun
or scientific annotation added in this decision step.

## Band recommendation — not yet approved

Bootstrap is a resampling method; a confidence interval (CI) is an uncertainty
interval. The saved bootstrap version already is a **95% bootstrap CI of the
condition median** (100 draws, seed10); “bootstrap” and “CI” are not competing
plot options. Its per-trial bands are pointwise, not simultaneous. IQR uses the
25th/75th percentiles of fish values, representing fish spread, not uncertainty
in the median.

Given the author's intended population comparison with later LME annotations,
recommend median with a 95% fish-bootstrap CI for the main G/H curves, and keep
median/IQR plus individual-fish coverage in the supporting review. Proposed
implementation: resample whole fish trajectories within each condition, retain
missingness, 5,000 draws, seed10, and check interval stability. Existing
100-draw bands remain preserved historical/display alternatives. A bootstrap
CI of a median is not the model CI or a test of the between-condition learning
contrast; annotation must come from the accepted model's saved results, not
band overlap. This recommendation changes no band-selection status.

Panel I remains a placeholder and 10sTrace remains **inconclusive** until
authenticated cohort and processed inputs are found. Main assembly and shared
manifest are unchanged by this row-specific decision.

## Author instruction and completed review — 2026-10-09

The author reaffirmed **always use the legacy metric**, accepted **5,000
whole-fish resamples within condition, seed10, preserving missingness**, and
authorized a new Delay G LME preview with D/M/R/trial-star lanes. These are now
implemented; the earlier "band recommendation not approved" section records
the previous state. Use the frozen legacy metric for subsequent main-paper
row work. Do not silently switch metrics; older alternatives remain historical
sensitivity artifacts.

The analysis script is `scripts/render_figure2_delay_legacy_metric_lme.py`.
Its metric is fixed to `legacy_distal_angular_speed`. The display script is
`scripts/render_figure2_delay_review_display.py`. Fresh fits use the active
condition-aware LME code on the57-fish saved cohort. Trial contrasts come from
one spline LME and compare differential change against Pre5–14; this is an
explicit improvement over the separate-per-trial legacy LME. D is Holm-adjusted;
M and R use separate BH families; raw R is also visible as requested. No stars
are generated for failed fits or nonsignificant adjusted tests.

The saved analysis directory is
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/20261009T072921130316Z-delay-lme/`.
The corrected display directory is
`J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/20261009T073349568633Z-delay-display/`.
All numerical fit gates passed, but residuals are strongly heavy-tailed; the
annotation preview is **exploratory and not approved paper inference**. See
`FIGURE2_DELAY_LME_CRITIQUE_2026-10-09.md` for model/Plans criticism and exact
lane results. No onset/extinction claim, 10sTrace source, shared assembly or
shared-manifest change is made by this row-specific work.
