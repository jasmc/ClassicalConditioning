# Independent Figure 2 G analysis

This exploratory route implements the author-requested 2026-10-14 review
separately from the selected G freeze and Figure 2 display. It never edits
selection/state pointers, frozen artifacts, raw files or cohort membership.
Scientific decisions remain open; this is a way to inspect candidate outcomes.

## Run and outputs

Run from the repository root with `.venv/bin/python`:

```sh
.venv/bin/python scripts/paper/figure_02/panel_g/run_independent_review.py \
  --output-dir '/Volumes/JOAQUIM/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-G-independent-review/NEW-ANALYSIS-ID' \
  --stage all --bootstrap 999 --workers 3
```

The destination must be new and external. Each stage writes a new immutable
directory under `stages/`; `analysis-state.json` selects the latest complete
directory. A run lock serializes writers. Source/output hashes and the artifact
inventory are verified before the state pointer is published atomically.
Completion metadata is written before the manifest. Failed candidates remain
inspectable without replacing preceding results. Changed or added artifacts
stop reuse. Changing numerical settings requires a new run destination.

Stages are `prepare`, `legacy`, `population`, `stabilize`, `sensitivities`,
`individual`, `timing-review`, `hyperparameters`, `cohorts`, `calibrate`,
`robust-calibrate`, `validation-review` and `report`; `verify` checks the selected
stage and original G. `validation-review` regenerates calibration summaries
from saved replicates without refitting.
Bootstrap and robustness fits retry L-BFGS with Powell using the same model;
attempts and failures remain explicit. Failed full models withhold inference
and skip stabilization without inventing missing bootstrap files.

`--prepare-from /absolute/path/to/20261014-analysis-01` imports authenticated
numerical input/QC, verifies all 401 bound sources and the exact frozen G
differences again. It preserves the parent reconstruction's own source lineage.

`--calibration-repetitions` and `--calibration-bootstrap` control simulation
work. Defaults are ten repeats per scenario and 99 resamples for the LMM study;
this remains feasibility/shape/availability evidence. The separate unsmoothed
comparison uses `--robust-calibration-repetitions` (default 100 independent
experiments per scenario) and 499 resamples, with binomial uncertainty on
coverage/false-supported-run rates. It cannot validate another estimator.
`--cohort-bootstrap` defaults to 199 refits per sensitivity cohort/target.

Permanent numerical tables, diagnostics, code snapshots, SVG, associated legend
and self-contained `review.html` reside in the external run directory. Run
configuration, input lineage and protected-original hashes accompany them.
The review is an independent scientific draft, not the current Figure 2 view.
`--stage report --rebuild-review` publishes a new report stage and retains the
preceding HTML/SVG/legend. Manual edits stop it. Resolve the latest HTML through
`analysis-state.json`; stale browser tabs do not select a current analysis.

## Historical discarding sensitivities

The author authorized February/March discarding experiments on 2026-10-14.
`legacy` reads Git revisions `2f63ef4` and `bf46bf7`, including preprocessing,
LogMedian plotting, experiment/general configuration and annotation helpers.
It saves source snapshots/hashes, per-fish rules, bout counts, overlap,
cumulative attrition and named membership tables before estimating effects.

Recipes include no additional discarding; the historical ANY-block six-trial
filter; baseline ≥3 or CR ≥3 bout trials in each chosen five-trial QC block;
their intersection; all Train US responses; combined preprocessing projection
with a last-US proxy; and an all-required-blocks six-trial alternative.
Effects/timing are compared across every recipe, without selecting by significance.
Behavior-based selection changes the population and can favor an outcome.

These are projections onto authenticated corrected detector/event data.
Original segmented historical Bout/700 FPS pickle data are unavailable here.
The historical annotated last-US end is explicitly unevaluable; raw US duration
is a named proxy in one alternative recipe. No projected cohort is labelled an
exact old cohort. Historical switches are recorded as disabled; sensitivity
activation is separately authorized. The selected G and primary cohort do not change.

## Candidate population analyses

Authenticated frame/movement/protocol inputs reconstruct the positive
bout-window LogMedian values. The rebuilt eligible differences must match the
exact selected G numerical hash/values before inference. Raw inventory and
reason-coded availability account for all observed recordings and scheduled
fish/trial outcomes. Technical failure, no bout and nonpositive vigor remain
distinct. The main candidates retain all selected G fish; named sensitivity
recipes separately report their removals and contributors.

Two targets are compared without selecting a primary one:

- Direct response-minus-baseline LogMedian.
- Response LogMedian with baseline as a covariate.

Both use the selected phase-aware basis family and fish random intercept/slope.
An accelerated Gaussian ML backend uses group sufficient statistics and
Woodbury identities for exactly this model. Fixed-effect uncertainty uses the
full observed-information Schur complement, including covariance nuisance
parameters. Known-answer tests compare likelihood, contrasts and standard
errors with statsmodels, including missing scheduled trials. On the complete
G input, maximum contrast discrepancies were below 2e-7 log units and relative
SE discrepancies below 0.004%; the comparison is saved in
`records/scientific-evidence/analyses/figure-02-g-independent/fast-lmm-equivalence-2026-10-14.json`.
The reference backend remains available as `fit(..., backend="statsmodels")`.
This numerical equivalence does not validate the scientific model assumptions.
Cached whole-fish resamples use integer multiplicities of the original fish's
sufficient statistics, equivalent to repeated copies with distinct cluster IDs;
known-answer tests check this against explicit duplicated-fish fitting.
The curve is control-minus-Delay change relative to average Pre trials 5–14.
Planned phase averages use late training 55–64, early testing 65–74 and late
testing 85–94. Plotted descriptive means match the direct-difference target;
they are not the frozen median-across-fish curve.

Whole-fish bootstrap refits within condition preserve each sampled trajectory's
missingness and scheduled trial numbers. A bootstrap-t maximum over 90 trial
contrasts and three phase contrasts supplies one simultaneous family per
candidate target. At least 95% of requested refits must pass numerical checks;
otherwise inferential bands/timing are withheld. This numerical gate does not
approve Gaussian adequacy or independence assumptions.

Residuals, true-adjacent serial correlations, leave-one-fish-out changes,
random-intercept sensitivity, simpler fish-level phase effects and actual-US
window-overlap sensitivity are saved. The simpler robustness intervals are
pointwise sensitivity evidence, not an additional joint claim family.
Additional sensitivities resample recording dates as whole clusters and fit
categorical trial effects without smoothing. The unapproved 95% detector-valid-frame
stress criterion is distinct from both requiring movement and independently
assessing coordinate tracking quality; it retains no G outcomes and is not applied.
Detector validity includes rolling-envelope/derivative/tail-coverage eligibility.
Descriptive movement
occurrence/time-moving views include valid no-bout trials. A schedule/event audit
separates declared prior pairing counts from actual nearest-US timing evidence.
Prototype fields named `baseline_technical_valid_frames` and
`response_technical_valid_frames` count detector-valid adjacent finite vigor
frames. Their names must not be interpreted as independent fish-quality diagnoses.

Acquisition candidates use three consecutive scheduled training trials.
Threshold fractions 0%, 5%, 10% and 20% are an explicitly unapproved exploration
grid, transformed by `-log(1-fraction)`. Model crossing, uncertainty bounds and
first supported run are separate. A missing upper bound does not establish
absence of learning. Acquisition runs cannot cross into testing.

## Individual candidates and missing observations

Training-only hierarchical spline and random-walk state-space covariance models
estimate individual conditioned-direction expression. Fixed-effect uncertainty
is integrated at estimated covariance parameters; covariance-parameter
uncertainty remains unaccounted for. Their simultaneous empirical-Bayes credible
envelopes are conditional model outputs, not calibrated acquisition intervals.

Full scheduled-grid crossing bounds keep unavailable trials as possible onset
locations. An additional field identifies the first supported run actually
observed and whether a point-estimated run contains model predictions. Never
compress time to the next three available observations.

The prototype information rule is three pre-training and ten training outcomes;
it is not an approved paper eligibility policy. It does not alter membership.
Leave-one-control-out validation refits references and covariance parameters
without that control's training observations; its retained summaries describe
the same scheduled-grid latent timing as full fits and simulations, while
observed supported runs stay separate. Validation trajectories are saved and
the same three-Pre/ten-training information rule applies in all three routes.
Testing is excluded from model/hyperparameter fitting.
The held-out forecast assesses training-end persistence into available trials
65–69, acknowledging the reinforcement transition; it is not extinction inference.

`hyperparameters` perturbs variances by factors 0.5 and 2 and AR1 by ±0.1,
recording expression/timing sensitivity. It does not integrate covariance
uncertainty or approve individual intervals. The unsmoothed comparator anchors
each fish by its own available Pre mean and uses means of contributing fish
changes. Its 90 trial and three fish-level phase summaries share a bootstrap-t
family. This is a separate estimand from the baseline-adjusted LMM.

## Interpretation and remaining gates

The primary baseline treatment, meaningful threshold, technical cohort policy,
minimum local coverage, recording/day independence and final model choice remain
author decisions. The prototype local coverage floor is six fish per condition,
not an approved numerical policy. Outcome availability may be informative:
conditional vigor is undefined for no-bout windows, rather than zero activity.

Simulation studies and technical sensitivities are engineering/review evidence.
They do not approve population or individual timing uncertainty. Fit/optimizer
success does not replace calibration, missingness and fit-failure review. The
[detailed 15 October evidence](../../records/scientific-evidence/analyses/figure-02-g-independent/implementation-and-review-2026-10-15.md)
owns the completed-run counts, coverage results, exclusions and interpretations.

## Technical sensitivity command

```sh
.venv/bin/python scripts/paper/figure_02/panel_g/run_technical_sensitivity.py \
  --parent /absolute/selected/stage \
  --output-dir /absolute/new/external/destination
```

This source-only sensitivity records its protocol rule before fitting,
authenticates acquisition/preprocessing/event sources and leaves the primary
cohort unchanged. Saved raw-source and supplemental evidence belong with the
selected report; this command does not reconstruct intended timestamps.

The owning workstreams remain Plans 02 (cohort), 06 (estimand/model), 03
(population timing), 04 (individual expression) and 07 (figure reporting).
An explicit later freeze request is required to replace an accepted selection.
