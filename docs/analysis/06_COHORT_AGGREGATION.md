# Cohort aggregation and response/baseline figures

This document distinguishes the active candidate cohort summaries from the
archived legacy pooling route. It records how the new response/baseline figures
aggregate observations and why their ratio displays do not themselves carry
mixed-model p-values.

The candidate route remains exploratory. A paper cohort, primary outcome, and
confirmatory statistical configuration have not yet been frozen.

## Current standardized-difference cohort comparison

The existing `figure-metric-comparison` figure compares all three candidate
metrics using the same outcome definition. For each recording, metric, and
alignment, it:

1. takes the arithmetic mean over time within each trial's baseline and
   response windows;
2. retains only trials with values in both windows and takes their arithmetic
   mean within the recording; and
3. takes an equal-recording arithmetic mean within each condition or the
   pooled cohort summary.

Its recording-level displayed value is not a response/baseline ratio:

```text
(mean response - mean baseline) / SD(trial baseline means)
```

The plot contains one dot per recording and a translucent condition mean for
each candidate metric. The title calls this fish-equal because the ordinary
use case is one recording per fish; technically the implemented weighting unit
is the recording. The route rejects duplicate fish only in the legacy cohort
builder, not in this candidate plotting stage, so a future paper route must
enforce the frozen fish cohort before interpreting it as fish-equal.

## How legacy pooling differs

Legacy scaled-vigor pooling concatenates all included recordings within a
condition before aggregating. At each exact time, it takes a median over all
non-missing pooled samples; within a displayed time bin it takes the arithmetic
mean of those exact-time medians. Thus it does not assign an equal final weight
to each fish or recording. A fish with more valid samples or trials can have
more influence.

The legacy normalized-vigor route first calculates, for each fish and trial:

```text
mean(response window) / mean(baseline window)
```

It then makes block/trial summaries and legacy statistical outputs. The
historical normalised-vigor blocks use five CS trials, despite the experiment
configuration also having ten-trial block names.

| Question | Active standardized comparison | Legacy scaled-vigor pooling | Legacy normalized vigor |
| --- | --- | --- | --- |
| Initial unit | recording/trial profile | all pooled samples in a condition | fish/trial ratio |
| Within-window reduction | arithmetic mean | exact-time median | arithmetic mean |
| Cohort reduction | equal-recording arithmetic mean | mean of exact-time medians | varies by block/trial route |
| Displayed effect | standardized response minus baseline | scaled-vigor time course | response / baseline |
| Ratio displayed? | No | No | Yes |

These are different estimands. A figure must name its aggregation and must not
be described as a reproduction of another route merely because their axes look
similar.

## New descriptive ratio figures

The new figures deliberately use a third, explicitly fish-weighted
aggregation. They require an authenticated frozen `cohort_id` and derive
population membership from `cohort-trial-outcomes`, where the primary cohort
has already been applied. They write the cohort hash into figure provenance,
and manual recording lists are deliberately not accepted. The event-aligned
figure uses the recording identities from that table to load authenticated
temporal profiles; it does not create another fish-selection rule.

Only `total-activity` and `conditional-intensity` are currently accepted. The
other candidate outcomes lack a matched baseline quantity in the trial-output
contract, so rendering a response/baseline ratio for them would invent an
estimand.

### Selected blocks: `figure-cohort-selected-block-ratio`

The x-axis contains three prespecified five-CS-trial blocks:

| Display label | CS trial numbers |
| --- | --- |
| Final Pre-train | 10–14 |
| Early Test | 65–69 |
| Late Test | 90–94 |

For every eligible trial, the figure calculates `response / pre-CS baseline`.
It then takes the **median across that fish's trials in the selected block**.
A fish needs at least three finite trial ratios in a block. The visible dot is
that fish-level median. Within each condition, the line is the median across
fish and the error bar spans the fish interquartile range (Q25–Q75).

This three-ratio requirement is analysis eligibility, not a new fish discard.
The aggregation frame retains an eligibility row and reason for every cohort
fish/block; only eligible rows enter the visible block summary. Legend sample
sizes expose changes in contributing fish across blocks. The learning-onset
route now publishes a durable trial eligibility table. The selected-block
plot's separate three-finite-ratio display rule remains visible in its panel
data and is not treated as fish exclusion.

This makes fish, rather than samples or trials, the population weighting unit.
The median/IQR choice is descriptive and robust to ratios inflated by a small
but positive baseline. It is not a confidence interval and is not an LME
effect estimate.

```powershell
classical-conditioning figure-cohort-selected-block-ratio `
  --project-dir "<SAVE-DIR>" --analysis-id <ANALYSIS-ID> `
  --cohort-id <FROZEN-COHORT-ID> `
  --metric tail_length_weighted_angular_l1 --outcome total-activity
```

### Trial-number trajectory: `figure-cohort-trial-ratio`

This is the descriptive plot that directly matches the requested axes: CS
trial number on x and `response / pre-CS baseline` on y. Each fish contributes
at most one fish-level value per trial before the condition median and fish IQR
are calculated. Faint traces show individual fish; legend sample sizes report
whether the contributing fish count changes over trials.

```powershell
classical-conditioning figure-cohort-trial-ratio `
  --project-dir "<SAVE-DIR>" --analysis-id <ANALYSIS-ID> `
  --cohort-id <FROZEN-COHORT-ID> `
  --metric tail_length_weighted_angular_l1 --outcome total-activity
```

This remains descriptive. The condition-aware longitudinal model, simultaneous
intervals, onset rule, diagnostic gates, and panel-data artifact are implemented
in the separate learning-onset route. They still require scientific approval
and a paper-cohort run before the publication panel is accepted.

### Event-aligned trajectory: `figure-cohort-event-aligned-ratio`

For each CS-aligned trial, the figure divides every time-bin value by that
trial's arithmetic mean in the pre-CS baseline window (−15 to 0 s). It then
takes the median across trials at every time bin within each fish, followed by
the median and Q25–Q75 across fish in each condition. Faint traces are
fish-level trajectories; the solid line and ribbon are the cohort median and
fish IQR. The vertical line is CS onset and the horizontal line at one is the
trial-normalised baseline.

```powershell
classical-conditioning figure-cohort-event-aligned-ratio `
  --project-dir "<SAVE-DIR>" --analysis-id <ANALYSIS-ID> `
  --cohort-id <FROZEN-COHORT-ID> `
  --metric tail_length_weighted_angular_l1 --outcome total-activity
```

This is not legacy exact-time pooling: ratios are first normalised within a
trial, then reduced within fish, then across fish. It prevents an individual
fish gaining extra cohort weight merely by contributing more trials or frames.
The aggregation frame counts contributing trials for every fish/time point and
the legend shows the range of contributing fish counts. Missing or zero
baselines can still make the contributing set vary with time; this must be
reviewed as outcome eligibility, especially for conditional intensity, which
is undefined when no movement bout is present.

For all three ratio figures, a smaller activity ratio means stronger motor
suppression. The ratio is therefore not automatically a positively directed
"CR strength" score. Any inferential contrast must state and enforce its sign
convention explicitly.

## LME annotations: legacy record and active choice

The legacy normalized-vigor route fit three families of fish-grouped LMEs on
`log(response + 1)` with `log(baseline + 1)` as a covariate:

1. **One global condition × ten-trial-block model** over all retained trials.
2. **One condition × centred-trial model inside each ten-trial block**, whose
   condition main effect and condition-by-trial slope were FDR-corrected across
   blocks.
3. **One condition model per trial**, whose condition p-values were
   Benjamini–Hochberg FDR-corrected across trials.

It used fish as the grouping variable and a baseline random-effect formula in
the frozen legacy reproduction. These are faithfully documented/reproducible
legacy outputs, not approved learning-onset inference. In particular, the
separate per-trial models cannot establish the onset by choosing the first
isolated significant p-value; the legacy cohort and exclusions were also
stage-specific. At a single trial there is normally only one observation per
fish, so a fish random intercept is weakly or non-identifiable there; failed
fits were also caught and skipped, which makes missing results difficult to
distinguish from genuine absence of evidence. With more than two conditions,
the historical code also selected the first matching coefficient rather than
constructing a named test-versus-control contrast.

The active [learning-onset plan](../../Plans/03_LEARNING_ONSET_IMPLEMENTATION.md) replaces
that hierarchy with:

1. a prespecified **condition × block/phase LME** for the population claim;
2. a **single condition-aware longitudinal LME** (spline or prespecified
   categorical-trial sensitivity) for trial-resolution contrasts; and
3. a fish-level condition-aware robustness analysis.

The implemented route requires a frozen outcome/direction, condition reference, contrast
family, effect threshold, persistence rule, diagnostics, and simultaneous
trial uncertainty before an onset result can be annotated. The current
older candidate mixed-effects scaffold omits condition and condition-by-time
terms, so it is not valid input for either new figure's condition comparison.
The separate learning-onset route does include those terms but is not yet
approved for paper inference.

Accordingly, the initial ratio figures intentionally show **no p-value stars,
no legacy per-trial p-values, and no onset claim**. Their visible fish-level
summaries and provenance provide the descriptive visual basis for future
accepted Model 1/Model 2 artifacts. Once those models satisfy the plan's
diagnostic gates, their effect sizes and simultaneous intervals—not raw ratio
p-values—can be added as separate contrast panels or clear model-derived
annotations.

## Learner stratification

Learner/non-learner panels are intentionally not part of these figures. The
repository contains multiple incompatible historical classifier variants. A
label may be used only after the learner-method review selects and freezes a
method; statistics on the same trials used to create that label are descriptive
unless they are held out or cross-fitted. See the active learner-analysis plan
for the full validation requirements.
