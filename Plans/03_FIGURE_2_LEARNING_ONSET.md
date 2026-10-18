# Figure 2 — Learning-Onset Analysis



**Input dependency:** A reviewed, immutable cohort and authenticated
`cohort-trial-outcomes` artifact from the
[cohort plan](02_DISCARDING_AND_COHORT_IMPLEMENTATION.md). This plan does not decide fish
membership. Its trial-level eligibility may change model contributions
without changing the cohort.

The [analysis and statistics plan](06_ANALYSIS_AND_STATISTICS.md)
owns the primary estimand, outcome family, condition contrast, model choice,
uncertainty, diagnostics, and validation decision. Approved choices are
recorded in [DECISIONS.md](../docs/paper/decisions.md). The
[LME pipeline reference](../docs/running-analysis/learning-models.md)
documents current formulas, public parameters, artifacts, and failure gates.

## Questions and implemented route

The analysis asks whether test and control trajectories change differently,
by which prespecified block a difference is supported, and whether an exact
trial onset can be localized. Fish are the population units. An isolated
per-trial p-value cannot establish onset.

The paper's possible extinction claim is a separate question: after a
conditioned response is established, does it decline or become small during
later unreinforced trials relative to an approved reference? The current route
localizes acquisition onset, not extinction. A later nonsignificant contrast
does not by itself establish loss of the response.

The implemented route produces fish-weighted descriptive trajectories, a
condition-by-block mixed model with planned contrasts, one condition-aware
longitudinal spline model with a categorical-trial sensitivity, fish-bootstrap
simultaneous trial bands, a persistent onset rule, fish-level
permutation/bootstrap robustness, leave-one-fish-out and residual diagnostics,
authenticated tables, and a three-component figure. It uses one selected
metric and a cohort-authenticated, reason-coded model-input eligibility table.
The older condition-pooled exploratory LME/permutation/bootstrap routes are
not substitutes for this analysis.

The proposed onset rule finds the earliest scheduled trial whose lower
simultaneous confidence bound for test-versus-control change from pre-training
exceeds `delta_min` for `K` consecutive eligible trials. `delta_min`, `K`, the
contrast direction, and the trial family require approval before a paper fit.
No qualifying run is reported as **onset not localized**; bootstrap samples
without onset remain in the uncertainty summary. A first supported block is
reported separately because it answers a coarser question.

## Remaining work

1. Freeze the primary metric and outcome, test/control conditions, baseline
   and response windows, pre-training reference, blocks, contrast direction,
   `delta_min`, `K`, multiplicity family, spline complexity, random-effects
   fallback, bootstrap size/seed, and publication-stopping diagnostics.
   Choose an outcome-appropriate model family; do not automatically apply a
   Gaussian LME to movement probability or bout counts.
   If claiming extinction, also freeze its starting population/period,
   direction, smallest meaningful remaining effect or decline, persistence
   rule, uncertainty and multiplicity family, and how fish without supported
   acquisition are handled.
2. Calibrate false-onset rate, interval coverage, and onset recovery at the
   expected paper fish/trial counts. Include null, pre-existing difference,
   abrupt, gradual, transient, missingness, unequal coverage, influential
   fish, and singular-fit scenarios. Verify effect sign, fish weighting,
   factor references, multiplicity, and failure handling.
   Separately calibrate false extinction claims and recovery of gradual,
   abrupt, transient, and absent declines using the actual test-trial layout.
3. Run the approved configuration on the frozen paper cohort. Review the
   block and longitudinal fits, simultaneous band, fish-level robustness,
   categorical sensitivity, coverage, convergence, covariance, Hessian where
   available, residuals, and leave-one-fish-out influence.
4. Approve panel data and the three figure components: fish/condition trial
   trajectories, adjusted trial contrasts with simultaneous uncertainty and
   qualified onset marker, and planned block contrasts. Every visible value
   must trace to saved cohort, configuration, model, and source identities.
   Failed diagnostics suppress inferential annotations.
5. If the approved claim requires extinction timing, implement and validate a
   separate late-test decline or loss analysis with fish-level uncertainty.
   Report an unlocalized or unsupported extinction result explicitly; do not
   infer it from onset outputs or a disappearing p-value.

## Exit gate

The paper result tests an approved condition-versus-control change estimand,
uses fish as the independent units, reports block and trial evidence as
distinct claims, and either localizes onset with a prespecified meaningful
effect, persistence, simultaneous uncertainty, and passing diagnostics or
states that onset was not localized. The fish-level robustness result agrees
or its disagreement is explained. The figure regenerates from authenticated
panel data. Any extinction claim has its own prespecified definition,
validated uncertainty, and diagnostics, or is labeled unsupported. Earlier
onset designs remain in Git history; this plan owns the current exit gate.

## GAMM exploration — 2026-10-18

**Purpose:** test whether a hierarchical generalized additive mixed model
(GAMM) improves population inference for Delay versus Control, particularly
around rapid acquisition and the Training–Test transition. This is a bounded
exploration under Plan 06, not approval to replace current G. Preserve existing
analyses, artwork and selection/freeze records; save future runs separately.

### What the other-chat proposal adds—and what needs correcting

A GAMM estimates nonlinear trajectories with penalized smooths and fish-level
variation. It could supply global, block and trial contrasts from one fit.
However, current G and the independent revision already use nonlinear spline
mixed models. The proposed comparison concerns smoothing, phase structure and
dependence, rather than a straight line versus a curve. A Gaussian GAMM is a
valid candidate; “generalized” does not require changing the response family.
No separate population GAMM comparison has yet been established by the reviewed
implementation evidence.

The proposal's Delay-minus-Control **level difference** also differs from G's
Control-minus-Delay **change relative to Pre**. A pre-existing group difference
can produce separation without differential learning. Keep these questions
explicit, including the sign. Per-trial pre-CS baseline adjustment and the
Pre-training reference are two different operations.

### Practical sequence

1. **Audit inputs and the boundary.** Reuse authenticated outcomes and report
   fish/trial availability by condition and phase. Check actual response windows,
   US timing, intertrial gaps, session boundaries and context changes. Measuring
   before the first omitted US motivates continuity into the first Test trial,
   conditional on unchanged cues/context; it does not require equal observations
   or smooth derivatives across that boundary.
2. **Specify the targets before comparing fits.** Retain change from Pre as the
   learning-related contrast; show level separation separately. Compare direct
   response-minus-baseline outcomes with baseline-adjusted response, using the
   same inputs where possible. Explain their different interpretations: baseline
   adjustment may remove behavior affected by training. Define the global null
   explicitly—no level difference, or no differential change. Retain planned
   late-Train, early-Test and late-Test averages; do not select favorable windows.
3. **Fit a small candidate set.** Use the existing phase-aware spline LMM as
   reference; compare a penalized phase-aware GAMM and a version linking Train
   into Test with continuity but allowing subsequent change. Keep Pre treatment
   explicit. Retain categorical-block LMM and the existing unsmoothed fish-level
   comparator. Bound basis complexity and random-effects alternatives in advance.
   Assess residual tails, fit stability and within-fish dependence; any correlation
   model must respect scheduled gaps rather than treating adjacent available
   observations as consecutive trials. Do not rank models by significance alone.
4. **Make inference comparable.** Use the same contrast direction, reference,
   evaluation grid and fish weighting. Specify multiplicity across global, block
   and trial claims: one model does not make them one hypothesis. Evaluate
   whole-fish refit bootstrap uncertainty, including smoothing/covariance
   estimation and failed refits. Distinguish pointwise intervals from simultaneous
   bands. For timing, specify the meaningful threshold, scheduled-trial persistence
   and handling of missing trials and resamples without onset before calculation.
5. **Challenge the method before recommending it.** Simulate null, pre-existing
   offsets, gradual/abrupt acquisition, plateaus and transient effects at the
   observed fish counts and trial schedule. Include autocorrelation and plausible
   behavior-related availability. Compare curve and phase-average coverage,
   false detections, timing recovery, unlocalized onsets and fit failures. Use a
   small pilot to reject poor candidates, then enough independent simulations to
   report Monte Carlo uncertainty. Set acceptance criteria before inspecting
   comparative results. AIC or successful numerical reproduction is insufficient.

### Trade-offs to review

| Candidate | Advantage | Main criticism / caveat |
|---|---|---|
| Existing phase spline LMM | Reproducible benchmark; phase changes allowed | Fixed flexibility can miss abrupt/transient shapes; separate phases permit jumps |
| Penalized GAMM | Estimates smoothness; can represent rapid change and plateaus | Can oversmooth transitions; smoothing and dependence complicate uncertainty |
| Continuity-linked Train/Test GAMM | Uses the pre-omission argument | Wrong boundary assumptions can spread effects across phases; continuity does not justify one rigid smooth |
| Categorical-block LMM | Fewer assumptions about the within-block curve | Coarse timing; block averaging can conceal brief effects |
| Unsmoothed fish comparator | Makes fish contributions explicit; avoids smooth-curve bias | Wider uncertainty; incomplete trials can change contributing fish |

### Non-negotiable caveats and stopping rules

- Incomplete trials do not automatically require discarding fish. No-bout
  windows have undefined conditional intensity, not zero intensity. Mixed models
  accept unequal trial counts but do not solve informative availability. Keep
  cohort/exclusion decisions in Plan 02; any restriction is a labeled sensitivity.
- Existing smooth-model calibration showed severe full-curve undercoverage for
  abrupt/transient scenarios. A GAMM must address that failure empirically;
  flexibility alone is no validation. See the
  [implementation review](../records/scientific-evidence/analyses/figure-02-g-independent/implementation-and-review-2026-10-15.md)
  and [comparison with current G](../records/scientific-evidence/analyses/figure-02-g-independent/current-G-comparison-2026-10-15.md).
- Supported trial intervals are not onset confidence intervals. Detectable
  divergence is not biological acquisition; nonsignificance is not equivalence
  or extinction. If timing remains unreliable, restrict conclusions to supported
  phase averages and explicitly report timing as unlocalized.
- Population curves do not identify individual learners. The summary's proposed
  classifier features are discussion, not an authenticated description of legacy
  algorithms. Individual classification remains in Plan 04.

**Deliverable:** one concise comparison report with input/configuration IDs,
eligible counts, estimates and intervals, fit/bootstrap failures, calibration
results and disagreements between methods. Recommend retaining, replacing or
demoting each candidate, with reasons. Record unresolved choices for review;
any new primary analysis or figure requires a subsequent explicit decision.
