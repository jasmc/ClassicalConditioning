# Exploratory Legacy-Rule Discarding

**Status:** Source-linked pre-classification screening implemented; paper approval is not implied.

`assess-discarding` runs this second stage immediately after technical
assessment. Every named check has a source comment in the implementation and
source metadata in the rule table. The default combined exploratory status
applies all checks. `--disable-check RULE_ID` can be repeated to inspect the
effect of omitting a check; each run replaces the stable derived assessment
bundle and records its new input hash and step-by-step fish counts. This combined population is a new rule projection on
the selected refactored metric, not the population produced by any one legacy
script.

| Rule ID | Legacy operation and executable behavior |
| --- | --- |
| `readable_fish` | [Preprocessing discard loop](../Archive/historical-scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py): unreadable per-fish data were discarded. `RUN_DISCARD` was false by default; the assessment still evaluates it. |
| `last_us` | Same file, `check_viability`: the last **observed** US event needs an end at least 0.4 s after onset and a bout in the inclusive 0–5 s window. |
| `train_us` | Same file, `check_train`: observed Train US trials need a bout in the inclusive 0–5 s window. |
| `retrain_us` | Same file, `check_retrain`: apply that rule only when Re-Train is declared; ordinary late US events are not automatically Re-Train. |
| `baseline_bouts` | Same file, `check_baseline`: Late Pre-train 10–14, Early Test 65–69, and Late Test 90–94 each need three trials with a bout in the inclusive −15–0 s window. |
| `cr_bouts` | Same file, `check_cr`: the same blocks each need three bout-bearing trials in the experiment's inclusive CR window (0–9, 0–13, or 0–20 s). The executable empty-selected-CS bypass is reported. Trial 65 counts in Early Test even though it is a catch trial. |
| `discard_propagation` | [Grouping](../Archive/historical-scripts/3_FishGrouping.py) used the discard list for heatmap grids but not pooled rows; [scaled vigor](../Archive/historical-scripts/4_ScaledVigorPlotting.py) applied it by default; [normalized vigor](../Archive/historical-scripts/5_NormalizedVigorPlotting.py) did not by default. These reuses are recorded, not counted as fresh independent checks. |
| `learner_inputs` | One merged pre-fit prerequisite from [original](../Archive/historical-scripts/6_LearnersQuantification.py), [new](../Archive/historical-scripts/6_LearnersQuantification_new.py), [improved](../Archive/historical-scripts/6_LearnersQuantification_improved.py), and [WIP](../Archive/historical-scripts/6_LearnersQuantification_WIP.py): finite positive baseline/response/ratio; at least six trials in each selected epoch and three in every required five-trial block. This does not run four classifiers or decide learner labels. |

The legacy normalized-vigor missing-window and minimum-trial/block filter
family is deliberately **not** part of this script. Model-derived learner
feature failures remain for the later learner analysis. An unevaluable rule is
never silently treated as a pass. Neither this exploratory result nor a
disabled check can alter the reviewed primary technical cohort. The command
never writes or reads active legacy discard lists and never moves files.

Outputs include technical and exploratory fish tables, rule-level statuses,
detail evidence, cumulative flow, source identity, and the assessment hash.
