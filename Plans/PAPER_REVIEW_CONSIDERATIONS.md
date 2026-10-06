# Paper review considerations

Use this agenda when reviewing the path from raw recordings to Figures 1–4.
It raises questions that affect more than one panel; it does not approve a
scientific choice or create another status board. Record resolved choices in
[DECISIONS.md](./DECISIONS.md), implementation in the relevant numbered plan,
and progress only in the [implementation index](./IMPLEMENTATION_STEP_INDEX.md).
Review the proposed panel layout in the
[paper figure specification](../docs/analysis/figures/02_PAPER_FIGURE_SPECIFICATION.md).

## Before Figures 1 and 2: source data and population

1. **Do recorded event times match the declared protocol?** The
   [fixed-trace raw-protocol audit](../configs/trace-preflight.json) supports
   a 13 s 3sTrace paired-US latency, and the active
   configuration now declares 13 s. Verify paired-training `Reinforcer` events
   for every authenticated Figure 4 fish before drawing expected-US guides.
   Keep actual US timing distinct from the conditioned-response window, and
   rebuild any artifacts made under the former 9 s configuration. See
   [analysis and statistics](./02_ANALYSIS_AND_STATISTICS.md) and
   [figure reporting](./08_FIGURES_AND_REPRODUCIBLE_REPORTING.md).
2. **What makes a recording usable, and what makes a fish a cohort member?**
   Show source-linked technical failures, exploratory behavior-dependent checks,
   fish identities, duplicate or repeated recordings, and counts at each
   transition. Freeze the label-independent primary inclusion policy before
   treating pooled results as paper evidence. The candidate metric comparison
   currently weights recordings equally; ordinary one-recording-per-fish use
   does not establish a fish-weighted paper result. See
   [cohort implementation](./01_COHORT_IMPLEMENTATION.md) and the
   [aggregation audit](../docs/analysis/06_COHORT_AGGREGATION.md).
3. **Which vigor measure and shared bout detector support the paper?** Compare
   all three candidates with the same measured-time, validity, smoothing, and
   bout rules. State the selection criteria and review evidence before choosing
   one metric; then carry its identity through all main and supplementary
   figures. See [the T1 decision](./DECISIONS.md) and
   [supplementary figures](./10_SUPPLEMENTARY_FIGURES.md).

## Figure 1 and Figure 2: display and inference

4. **Can a reader trace a single-fish panel into the pooled view?** Record the
   chosen fish and trials. Confirm Figure 1 and Figure 2 heatmaps use the same
   signed bout-log-vigor bin definition and display limits, with one fish value
   per pooled bin. Name the response/baseline ratio as a different outcome;
   distinguish the routine all-valid-frame heatmap if shown as a diagnostic.
   Show valid-bin fractions and contributing-fish counts alongside pooled
   heatmaps. See [figure reporting](./08_FIGURES_AND_REPRODUCIBLE_REPORTING.md)
   and [supplementary figures](./10_SUPPLEMENTARY_FIGURES.md).
5. **What does an empty time bin mean?** Keep no-bout and invalid measurements
   distinct from zero movement. Pair conditional vigor with movement
   probability or another occurrence measure, and expose coverage before
   comparing groups or times. See
   [supplementary figures](./10_SUPPLEMENTARY_FIGURES.md).
6. **What exactly supports learning and extinction claims?** Freeze the
   fish-level block and trial estimands, contrasts, uncertainty, multiplicity,
   model diagnostics, and eligibility before placing inferential marks on
   Figure 2. Acquisition onset and extinction need separate definitions and
   validation; a later nonsignificant contrast alone does not locate
   extinction. See [analysis and statistics](./02_ANALYSIS_AND_STATISTICS.md)
   and [learning-onset completion](./03_LEARNING_ONSET_IMPLEMENTATION.md).

## Figure 3 and Figure 4: individual learning and response profiles

7. **Which learner representation survives validation?** Review continuous
   fish effects and heterogeneity before deciding whether categories add
   defensible meaning. State the evaluation split for any inferential learner
   claim. If the approved result is continuous or model-based, adapt Figure 4's
   current categorical renderer and grouping to match it. Same-data strata
   remain descriptive. See [method review](./04_LEARNER_METHOD_REVIEW.md),
   [learner analysis](./05_LEARNER_ANALYSIS.md), and
   [figure reporting](./08_FIGURES_AND_REPRODUCIBLE_REPORTING.md).
8. **Which trials can support independent response timing?** Identify each
   fish's label-building and evaluation trials, the exact pooled catch set,
   verified expected-US time, and a rule for fish without a measurable
   response. Keep post-US paired-training activity out of anticipatory timing
   claims. See [supplementary figures](./10_SUPPLEMENTARY_FIGURES.md).

## Figure handoff and final release

9. **What evidence travels with every main and supplementary panel?** Check
   metric and cohort identities, source and recipe hashes, fish/trial counts,
   missingness, calculation, inference status, diagnostics, and visual review.
   Assign final supplementary numbers only after content is approved. The
   release should freeze one coherent record of the selected panels and their
   inputs. See [figure reporting](./08_FIGURES_AND_REPRODUCIBLE_REPORTING.md),
   [supplementary figures](./10_SUPPLEMENTARY_FIGURES.md), and
   [final release](./12_FINAL_ANALYSIS_RELEASE.md).
