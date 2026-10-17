# Figure 3 discussion and author directions — 16 October 2026

This is a dated summary of this chat, including the author's corrections,
implemented exploratory outputs and unresolved work. It is not a scientific
approval, a freeze, or a replacement for scoped presentation state, Plan 04,
the decision register or the implementation index.

## 1. What the author asked to accomplish

Review the existing Figure 3 learner-analysis plan and statistical plan,
criticize the proposed analysis, inspect the inherited code and outputs, and
build actual exploratory material and a main-paper draft. The starting documents
were Plans/04_FIGURE_3_LEARNER_ANALYSIS.md and
Plans/06_ANALYSIS_AND_STATISTICS.md.

The scientific aim is to find an appropriate, accurate and defensible way to
quantify individual conditioned expression and, if justified, identify learners
and nonlearners. The author also asked which approach finds more learners while
avoiding excessive loss of fish. Those are comparison criteria to report;
maximizing learner counts or retention alone does not establish accuracy.

The author subsequently requested a statistically corrected BLUP-based version,
inspired by the legacy code, and a comparison of all the candidate versions
before choosing the final method. This request is still open: the corrected
BLUP comparison has been started but has not been completed or published.

## 2. Standing author choices from this chat

### Figure 3 versus Figure 4

Figure 3 develops and illustrates the individual-learning analysis and its
underlying evidence. Aggregate behavioral comparisons of Control versus
Learner, Control versus Nonlearner, and Learner versus Nonlearner, both within
each experiment and across experiments, belong in **Figure 4**. The author
explicitly corrected their placement during the discussion. Controls can appear
in Figure 3 as references/calibration evidence without moving the Figure 4
behavioral comparison there.

The main Figure 3 must contain **one selected analysis path**. It must not be a
catalogue of competing methods. The subsequent request to compare all versions
is for a separate exploratory/author review used to select that path. It does
not reverse the single-path requirement for the final main figure.

### Metric, trial outcome and scales

Use only the legacy tail-bend angular-speed metric,
`legacy_distal_angular_speed`. The author requested deletion of Figure 3 work
concerning the alternative metric; the recorded disposition of the angular-L1
draft and sidecars remains in the repository archive records. This is not
authorization to delete other historical work or the legacy classifier variants.

For the main individual trial plots use the **Figure 2 G outcome**:

`median ln(positive CR-bout vigor) − median ln(positive baseline-bout vigor)`.

For the available Delay protocol, the baseline window is [−15,0) seconds and
the CR window is [0,9) seconds relative to CS onset. The author explicitly
confirmed this outcome after the distinction from the heatmap estimator was
explained. Do not apply another logarithm or silently substitute a log ratio
of arithmetic bout means. Exact legacy classifier inputs remain separately
available for audit and comparison.

Heatmaps use native `managua_r` with fixed display limits **−0.25 to +0.25**,
undefined intensity black, and unclipped numerical values. The matching
heatmap estimator is a direct 1-second median of eligible positive bout-frame
log vigor, minus each trial's median finite baseline-bin medians. This is
related to, but not identical to, the G trial estimator. The selected condition
colors are Control `#00aeef` and Delay `#ec008c`.

Conditional no-bout vigor is undefined, not an invented zero. Movement/fraction
moving must remain visible separately. Detector-eligible time is not an
independent raw-tracking completeness measurement. Missingness, immobility,
technical cohort exclusions and method-specific eligibility are distinct.

### Individual examples and supplementary data

Show a few deliberately selected, illustrative fish in the main paper, with
heatmaps and trial observations that make the score or classification
understandable. The author explicitly allowed selecting clear examples, provided
**all fish are shown in supplementary material**. Selection is explanatory,
not representative sampling or independent validation. Include uncertain,
discordant and unclassified cases in the complete exploratory material.

Retain original scheduled trial numbering, phase boundaries, configured catches
and actual US timing. Use common scales when comparing main examples and retain
all observations rather than clipping unfavorable points. The legacy fish-grid
idea motivated the complete individual-fish atlas.

## 3. What the legacy review established

The four original variants are nominal, new, improved and WIP. They differ in
response transformations, selected epochs, completeness requirements, control
references, uncertainty rules and thresholds. They are historical comparators,
not ground-truth learner labels or successively validated upgrades.

All four use mixed-model fish-specific epoch effects/BLUPs and require both
acquisition-direction suppression and later recovery-direction change. They
therefore select an acquisition-plus-recovery phenotype. The nominal middle
window is trials 60–69; the other versions use trials 40–69. Both windows include
early Test observations. Late Test 85–94 also enters the recovery feature.
Their Test plots cannot be presented as held-out validation.

The legacy models do not all consume the G outcome: nominal/new use a
log(response mean + 1) model with log(baseline mean + 1) adjustment; improved
uses log response/baseline means without the added one; WIP uses log normalized
vigor. These are transformations of the same legacy metric, not interchangeable
estimands. Unit-sensitive transformations and historical preprocessing changes
prevent treating historical and current counts as identical results.

Fresh application of the unchanged algorithms to the current bout-only pilot
produced:

| Original rule | Delay flagged / eligible | Delay unclassified / all 29 | Control flagged / eligible | Control unclassified / all 28 |
|---|---:|---:|---:|---:|
| Nominal | 6/27 | 2/29 | 1/23 | 5/28 |
| New | 8/26 | 3/29 | 3/23 | 5/28 |
| Improved | 9/26 | 3/29 | 3/23 | 5/28 |
| WIP | 16/26 | 3/29 | 2/23 | 5/28 |

WIP finds the most learners among these legacy recipes, but that does not prove
it is the most scientifically accurate. Same-data control flags are not an
independently established false-positive rate. Model warnings and numerical
fallbacks remain part of the legacy record rather than being silently repaired
inside a supposedly unchanged historical method.

The original heatmap-grid code did not establish a complete, identity-verified
atlas or equivalent numerical/model inclusion. Old plotting normalizations,
per-fish color scales, discarded-fish lists and renumbered trials were reviewed
as traps to avoid. A cumulative BLUP feature line is not a fitted trial-by-trial
learning trajectory.

## 4. Exploratory outputs that were completed

A complete **57-fish Delay/control atlas** was built: 29 Delay and 28 control
fish, with 5,130 scheduled CS trial rows (trials 5–94). It contains all four
legacy decisions, eligibility reasons, continuous features, original classifier
inputs, G trial plots, heatmaps and movement/availability information. It has
filters and preserves all fish, including unclassified entries.

The initial atlas illustrations were 20240618_08 (all four rules flag),
20240612_08 (none flag), 20240702_01 (disagreement), and 20221123_02
(near the WIP threshold). Their selection did not validate the labels.
Historical 59-fish and 53-fish 3sTrace legacy-metric drafts were also identified
and preserved; they do not supply the current pilot or establish current state.

Its authority is docs/paper/figures/figure-03/exploratory-state.json. Its
external root is:

`/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure3-learner-exploration-g-style-20261015`.

Only Delay/control processed inputs were authenticated for this pilot. Trace
inputs remain a scope gap; this must not be described as a Trace exclusion or
an all-experiment result.

JOAQUIM was temporarily disconnected during main-figure preparation. The author
said they would reconnect it, and it subsequently became available. Work used
the recorded numerical sources after reconnection, rather than substituting a
stale browser copy.

## 5. Main-paper drafts and the author's correction

The first main-paper draft mixed a new descriptive G acquisition contrast with
a comparison of the four legacy rules and three rule-based examples. The author
rejected this: **the figure must contain only the best analysis path**.
The mixed-method draft was preserved, not treated as the selected final result.

A second draft removed the legacy comparison, WIP labels and rule-disagreement
panel. Its current composition is:

| Panel | Current descriptive draft |
|---|---|
| a | G trial outcome and acquisition-score definition |
| b | One acquisition score per fish, all 57 fish retained |
| c | Larger contrast: 20240618_08 |
| d | Intermediate contrast: 20240618_02 |
| e | Near-zero contrast: 20240612_08 |

The candidate acquisition score is:

`median(G outcome, Pre 5–14) − median(G outcome, late Train 40–64)`.

Positive means greater response suppression. It separates acquisition from
recovery and excludes Test observations from this calculation. Prior inspection
of Test prevents retrospectively calling it a prospectively untouched holdout.
The two epoch medians are marked on the individual plots. The three main trial
axes share −0.6 to +0.6; heatmaps retain ±0.25.

All 57 fish have mathematically estimable contrasts, but defined-trial coverage
ranges from 4–10 in Pre and 5–25 in late Train. This is not proof of adequate
precision or an approved classification-eligibility rule. No new binary learner
labels or validated individual intervals were introduced.

The current main-paper presentation authority is
**docs/paper/figures/figure-03/current-state.json**. At this checkpoint it selects:

`/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure3-main-paper-single-path-20261016/review.html`.

The displayed draft was opened at http://127.0.0.1:8776/review.html, but the state
and its bound file hashes, not a localhost tab, determine the selected version.
The native editable SVG is 183 × 158 mm. Twelve targeted tests and exact-render
layout checks passed for that draft. It remains unfrozen. The exploratory atlas,
previous draft, other figures and main-paper collection were preserved.

## 6. BLUP discussion and the correction to the recommendation

The author asked whether BLUPs had been considered. The response clarified that
BLUPs were present in the legacy analysis but absent from the current descriptive
main draft. The assistant acknowledged that it had simplified too early:
**no completed comparison establishes that the median contrast is better than a
properly implemented BLUP-based approach**. Failure of particular earlier
individual uncertainty procedures does not rule out BLUP models generally.

The author then requested a version using BLUPs as in the legacy code, but
statistically corrected, to compare the versions before choosing the best one.
This is authorization to build and compare candidates, not approval of a
particular corrected model, threshold or claim of statistical correctness.

Inspection of the WIP implementation identified a two-stage control-anchor
procedure: the second fit includes random epoch slopes but omits a fixed epoch
effect, then adds the estimated control anchor back. Its reported uncertainty
combines anchor and conditional random-effect terms as independent quantities;
this is not a full joint prediction-error calculation. Other legacy variants
and fallback paths must retain their original identities in the comparison.

The corrected candidate being implemented uses:

- G trial outcomes on the selected legacy metric;
- jointly fitted condition-by-epoch fixed means;
- correlated fish-specific random epoch means, equivalent to a random intercept
  and epoch slope;
- fish acquisition estimated as the **fixed plus random** contrast, not the
  random deviation alone;
- a separately reported excess contrast relative to the control fixed change,
  with shared-reference covariance accounted for;
- separate control/Delay residual variances and a within-epoch AR1 option using
  original scheduled gaps;
- refitted parametric prediction-error bootstrap uncertainty, with independent
  simulation checks planned before inferential display or classification;
- acquisition and recovery reported separately, rather than requiring recovery
  to qualify as an acquisition response.

A BLUP is not a probability that a fish learned. Even a positive control-relative
change is not a directly observed individual causal effect. Model assumptions,
serial dependence, coverage/missingness, uncertainty calibration, sensitivity,
retention and control comparability remain part of the assessment.

## 7. BLUP implementation checkpoint — not a completed comparison

A supported statistical prototype was added at
src/classical_conditioning/analysis/review/blup_comparison.py, with independent
dense-matrix tests in tests/test_blup_comparison.py. A first REML fit was exercised
on the 57-fish G acquisition data and a five-refit smoke run completed. Eight targeted tests passed: five new
dense-matrix/BLUP tests and three existing acquisition-contrast tests. These are
implementation checks, not scientific validation or a finished result.

The external work directory is:

`/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure3-blup-comparison-20261016`.

It contains a starting ticket binding source identities and protected states.
No corrected BLUP comparison HTML, completed simulation report, new categorical
manifest or selected BLUP Figure 3 has been published at this checkpoint. The
main-paper state still selects the descriptive single-path draft above.

The next scientific work is to complete and validate the BLUP candidate,
compare it against the unpooled mean and median contrasts and the unchanged
legacy recipes, and make the differences inspectable in a separate review.
Compare like estimands where possible and explicitly identify differences in
outcome, windows and recovery requirements. Do not choose a winner solely from
learner counts or claim that retaining sparse fish guarantees accurate estimates.
Only after that assessment should one analysis path replace the main draft.

## 8. Decisions still open and protections retained

The final learner estimand and representation, adequate precision/coverage,
control reference, meaningful-effect margin or threshold if used, uncertainty
calibration and permitted claims remain unresolved under Plan 04 / Gate L.
The author's interest in learner/nonlearner examples does not by itself approve
a binary classification rule. Continuous results remain a candidate, not a
rejection of the author's requested BLUP comparison.

All new paper candidates follow the selected Nature presentation guidance and
shared profile, native editable SVG/associated legend workflow, active 1 mm
invisible frame policy and guarded publication. Those presentation checks do
not approve science or freeze a figure. Preserve manual/uncommitted changes,
source data, original legacy results, earlier drafts and unrelated figures.
Permanent numerical/media/review artifacts remain on JOAQUIM; local code,
tests, guidance and this dated discussion record remain in the repository.

The latest author-supplied AGENTS.md instructions replace earlier AGENTS.md
instructions. The current selection is resolved from scoped state rather than
a hard-coded version name. No scientific freeze, historical cleanup or deletion
of preserved versions was requested in this discussion.
