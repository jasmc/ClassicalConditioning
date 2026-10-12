# Paper review considerations

Use this agenda when reviewing the path from raw recordings to Figures 1–4.
It raises questions that affect more than one panel; it does not approve a
scientific choice or create another status board. Record resolved choices in
[DECISIONS.md](scientific-decisions.md), implementation in the relevant numbered plan,
and progress only in the [implementation index](../Plans/IMPLEMENTATION_STEP_INDEX.md).
Use [Current figures](current-figures.md) to distinguish accepted frozen panels from the proposed remainder.

Collect concrete panel-specific criticisms and follow-up questions in
[panel review comments](panel-review.md). Its reusable entry format
applies to every main and supplementary panel; Figure 2 D/E currently include
the pairing, clipping, interpretation, contrast-coverage and multiplicity
review. It is an agenda rather than another implementation-status board.

## Before Figures 1 and 2: source data and population

1. **Do recorded event times match the declared protocol?** The
   [fixed-trace raw-protocol audit](analysis/audits/trace-preflight.json) supports
   a 13 s 3sTrace paired-US latency, and the active
   configuration now declares 13 s. Verify paired-training `Reinforcer` events
   for every authenticated Figure 4 fish before drawing expected-US guides.
   Keep actual US timing distinct from the conditioned-response window, and
   rebuild any artifacts made under the former 9 s configuration. See
   [analysis and statistics](../Plans/06_ANALYSIS_AND_STATISTICS.md) and
   [figure reporting](../Plans/07_FIGURE_BUILD_TOOLS.md).
2. **What makes a recording usable, and what makes a fish a cohort member?**
   Show source-linked technical failures, exploratory behavior-dependent checks,
   fish identities, duplicate or repeated recordings, and counts at each
   transition. Freeze the label-independent primary inclusion policy before
   treating pooled results as paper evidence. The candidate metric comparison
   currently weights recordings equally; ordinary one-recording-per-fish use
   does not establish a fish-weighted paper result. See
   [cohort implementation](../Plans/02_DISCARDING_AND_COHORT_IMPLEMENTATION.md) and the
   [aggregation audit](analysis/reference/06-cohort-aggregation.md).
3. **Which vigor measure and shared bout detector support the paper?** Compare
   all three candidates with the same measured-time, validity, smoothing, and
   bout rules. The author already selected tail bend angular speed; preserve that choice and review remaining preprocessing/detector validation. Carry its identity through all main and supplementary
   figures. See [the T1 decision](scientific-decisions.md) and
   [supplementary figures](../Plans/09_SUPPLEMENTARY_FIGURES.md).

## Figure 1 and Figure 2: display and inference

4. **Can a reader trace a single-fish panel into the pooled view?** Record the
   chosen fish and trials. Preserve the frozen Figure 1 V12 definition. Figure 2 population
   pooling is not selected by that single-fish freeze; review its proposed
   adaptation and one-value-per-fish weighting explicitly. Treat legacy
   response/baseline ratios as a different historical outcome;
   distinguish the routine all-valid-frame heatmap if shown as a diagnostic.
   Show valid-bin fractions and contributing-fish counts alongside pooled
   heatmaps. See [figure reporting](../Plans/07_FIGURE_BUILD_TOOLS.md)
   and [supplementary figures](../Plans/09_SUPPLEMENTARY_FIGURES.md).
5. **What does an empty time bin mean?** Keep no-bout and invalid measurements
   distinct from zero movement. Pair conditional vigor with movement
   probability or another occurrence measure, and expose coverage before
   comparing groups or times. See
   [supplementary figures](../Plans/09_SUPPLEMENTARY_FIGURES.md).
6. **What exactly supports learning and extinction claims?** Preserve the exact frozen D/E B and G estimands, tests/models,
   uncertainty, multiplicity and recorded limitations. Other Figure 2 panels
   require their own accepted definitions and evidence before inference. Acquisition onset and extinction need separate definitions and
   validation; a later nonsignificant contrast alone does not locate
   extinction. See [analysis and statistics](../Plans/06_ANALYSIS_AND_STATISTICS.md)
   and [learning-onset completion](../Plans/03_FIGURE_2_LEARNING_ONSET.md).

## Figure 3 and Figure 4: individual learning and response profiles

7. **Which learner representation survives validation?** Review continuous
   fish effects and heterogeneity before deciding whether categories add
   defensible meaning. State the evaluation split for any inferential learner
   claim. If the approved result is continuous or model-based, adapt Figure 4's
   current categorical renderer and grouping to match it. Same-data strata
   remain descriptive. See [the unified learner plan](../Plans/04_FIGURE_3_LEARNER_ANALYSIS.md),
   [method review](analysis/learners/METHOD_REVIEW.md), and
   [figure reporting](../Plans/07_FIGURE_BUILD_TOOLS.md).
8. **Which trials can support independent response timing?** Identify each
   fish's label-building and evaluation trials, the exact pooled catch set,
   verified expected-US time, and a rule for fish without a measurable
   response. Keep post-US paired-training activity out of anticipatory timing
   claims. See [supplementary figures](../Plans/09_SUPPLEMENTARY_FIGURES.md).

## Figure handoff and final release

9. **What evidence travels with every main and supplementary panel?** Check
   metric and cohort identities, source and recipe hashes, fish/trial counts,
   missingness, calculation, inference status, diagnostics, and visual review.
   Assign final supplementary numbers only after content is approved. The
   release should freeze one coherent record of the selected panels and their
   inputs. See [figure reporting](../Plans/07_FIGURE_BUILD_TOOLS.md),
   [supplementary figures](../Plans/09_SUPPLEMENTARY_FIGURES.md), and
   [final release](../Plans/08_FIRST_RELEASE_FIGURES_1_TO_4.md).


## Current-selection and release check

Review each Figure 1–4 panel using the individual entries in panel review comments. The selected metric and bout-only policy are approved; D/E B and G selections and Figure 1 E/V12 supersede earlier unselected discussions within their scopes. Figure 2 population pooling choices remain open. Release only main Figures 1–4 with necessary evidence; do not require supplementary composition or follow-on tail/imaging work.
