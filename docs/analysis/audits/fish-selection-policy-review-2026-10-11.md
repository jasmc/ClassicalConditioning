# Fish exclusion and eligibility: combined legacy review

**Review pending.** Author instruction, 2026-10-11: combine all legacy discarding steps; review them later. This document collects their behavior and ownership in one package. It does not approve numerical thresholds, enable exclusion switches, recompute a cohort or change accepted panel freezes. [Plan 02](../../../Plans/02_DISCARDING_AND_COHORT_IMPLEMENTATION.md) owns the review before [Analysis and Statistics](../../../Plans/06_ANALYSIS_AND_STATISTICS.md).

## What currently runs automatically

The supported corrected pipeline invokes [`assess-discarding`](../../../src/classical_conditioning/analysis/discarding.py) from [`run_pipeline`](../../../src/classical_conditioning/pipeline.py) after per-fish processing and before exploratory metric comparison. Tests assert that ordering. Its technical evidence and exploratory legacy-rule projection are automatic **within an invoked run**, not an always-running service. This code audit does not claim a job is currently running.

Assessment does not freeze paper membership or move/delete source data. Its default policy is draft; `min_matched_frames=1` and `min_valid_frame_fraction=0.0` are implementation defaults, not author-approved paper thresholds. Processing scope and authenticated output readiness influence `technical_ready`; an unrequested, unavailable or failed processing case is not automatically a biologically excluded fish.

Reviewed membership is separately frozen by `freeze-cohort` and applied by the cohort table. Trial/model eligibility is subsequently reason-coded without changing that cohort. Some exploratory/manual routes still bypass that shared population and require migration; this review does not assert every consumer is already compliant.

## Historical plan recovered

Git revision `83cbeee` (2026-09-02) contains `Plans/steps/08_FULL_REPROCESSING_QC_AND_COHORT.md`: full expected inventory, corrected reprocessing/QC, reviewed primary/sensitivity cohorts, explicit C0 policy and C1 cohort freeze. Its local two-fish fixture is explicitly not the paper cohort. `Plans/DECISIONS.md` at that revision deferred per-block completeness, including historical OR versus required-block logic, until real cohort counts could be reviewed. Plan 02 is the current successor.

## Combined legacy stages

The combined package preserves where a rule acts and its original recipe. Gathering rules together does not mean AND-ing every check into one new primary exclusion switch. List propagation is not a new independent check, model sample-size gates are not fish quality rules, and example selection is not population eligibility.

| Stage/source | Behavior to preserve for review | Executable default or scope | Proposed review ownership; not approval |
|---|---|---|---|
| Preprocessing `run_discard` | Failed/unreadable per-fish inputs; viability, Train/Re-Train, baseline and CR checks below; legacy logs/file movement | `RUN_DISCARD=False`; runs only if explicitly enabled | Processing disposition versus technical/behavioral screening must be distinguished |
| Last observed US viability | Last observed US end at least 0.4 s after onset and bout in inclusive 0–5 s window | Inside enabled preprocessing discard route | Physiological/behavioral screen; cannot silently become primary membership |
| Train US response | Bout in each selected observed Train US trial within inclusive 0–5 s | Same route; preserve exact block matching and event availability | Behavioral/sensitivity review |
| Re-Train US response | Same check when the source route has a Re-Train block; absence has legacy bypass behavior | Same route; report absent-block versus unevaluable cases | Behavioral/sensitivity review |
| Baseline bout completeness | At least 3 bout-bearing trials in each selected five-trial block, baseline inclusive −15–0 s | Source block set/windows preserved, including Late Pre-train, Early Test and Late Test | Eligibility/sensitivity; no automatic primary exclusion from activity alone |
| CR bout completeness | At least 3 bout-bearing trials per required block in experiment-specific inclusive CR interval | Preserve selected-CS/empty-selection guards, trial mapping and original thresholds | Outcome-dependent eligibility/sensitivity |
| Grouping (`3_FishGrouping`) | Reads `Discarded_fish_IDs.txt`; separates included/discarded heatmap grids, but pooled rows are not equivalently filtered | `APPLY_FISH_DISCARD=True` when list exists | Propagation/display difference, not another biological criterion |
| Scaled vigor (`4_ScaledVigorPlotting`) | Reads/reuses the same list for plotted/processed tables | `APPLY_FISH_DISCARD=True` when list exists | Propagation; review population consistency |
| Normalized vigor (`5_NormalizedVigorPlotting`) | Optional reuse of list at aggregation/model/plot entry points | `APPLY_FISH_DISCARD=False` | Propagation is disabled by default, other row/statistical filters still exist |
| Normalized pooled metadata scope | Optional age and rig/setup-color restrictions; condition, alignment, selected blocks/windows | Age/setup defaults are `all`; selected plot/experiment routes define remaining scope | Experiment scope/display/analysis selection; cannot be mistaken for generic QC |
| Normalized finite-row preparation | `prepare_main_df` drops missing normalized vigor, trial/block, baseline and response fields | Applied when that model preparation route runs | Reason-coded row eligibility |
| Normalized ANY-block fish rule | `prepare_main_df` retains fish meeting `minimum_trials_per_fish_per_block` in **any** block, then retains their remaining rows | Executable ANY-block behavior, not an all-block requirement | Preserve for comparison; final analysis-specific completeness decision remains open |
| Normalized paired block test | Wilcoxon branch builds a subset with all requested blocks; skips when block coverage is missing | Conditional statistical branch | Paired-contrast eligibility, not whole-cohort exclusion |
| Normalized plotting/statistics finite subsets | Nonfinite outcome removal, available pairs, requested blocks and coverage-dependent plotting/test subsets | Per consumer; retain original selected windows and denominators | Row/block/display/model eligibility and contribution counts |
| Learner variants: input preparation | Missing fields, normalized outcomes/baseline/response validity and selected features | Separate nominal/new/improved/WIP recipes; configuration overrides may differ from dataclass defaults | Pre-fit learner eligibility, not primary membership |
| Learner epoch/block completeness | Required feature epochs and per-five-trial-block counts; active recipes include 6 trials per epoch and 3 per constituent block | Preserve actual instantiated configuration and acquisition/recovery feature selections for each variant | Learner-specific eligibility; do not silently unify variants |
| Learner feature completeness | Intersection of fish with required acquisition/recovery/model features; finite transforms and model-derived failures | Later in feature/model calculation, outside the current pre-fit assessment | Learner eligibility with explicit unavailable/uncertain status |
| Learner model sample-size gate | Minimum fish with all features can abort a model fit | Variant/configuration dependent | Model feasibility, not exclusion of individual fish from paper cohort |
| Example-fish plotting (`2_ExampleFishPlotting`) | Selected fish, conditions, trials and display windows | Manual illustration scope; trace all source selectors in the index | Display selection only |

**Source evidence:** [selection source index](legacy-selection-source-index-2026-10-11.json) covers all nine current legacy scripts with SHA-256 and selection-related source-line candidates, including the four learner variants. Candidate lines include comments and are not counts of independent rules. The [selection inventory](EXCLUSION_AND_SELECTION_INVENTORY.csv) covers current/refactored consumers as well. Historical executable semantics remain in their original files; this documentation does not modify them.

## Relationship to the existing assessment

The current `assess-discarding` projection already includes readability, last/Train/Re-Train US, baseline/CR bout checks, propagation metadata and one merged pre-fit learner-input prerequisite. It deliberately omits normalized-vigor row/minimum-trial/paired-block families and later model-derived learner failures. Those omitted families are now part of this **combined review package**, not newly activated code. Its combined exploratory status is not the exact cohort of any single inherited script.

Keep each legacy variant, order, AND/OR logic, bypass, disabled flag and dependency visible. Evaluate all rules on a common authenticated source when a later review run is authorized; output individual rule statuses, cumulative effects, overlap, unevaluable reasons and counts at fish/trial/block/model levels. Do not count reapplication of the same discard list as independent attrition.

## Boundary already decided; numerical solution still open

The current [scientific decisions](../../scientific-decisions.md) require a label-independent technical primary cohort. Response-window movement, CR strength, learner labels and imaging availability cannot define it. Bout-only vigor can be undefined for a no-bout window; valid immobility is not proof of tracking failure or justification to remove a fish. Outcome-specific missingness/eligibility remains explicit.

Before freezing the final C0/C1 solution, review which legacy checks remain technical membership criteria, which become sensitivity/eligibility/display rules, their exact values and window/block semantics, treatment of partial/failed cases and actual paper-scale count consequences. The historical rules are collected now; author approval and the population-changing run occur later. No legacy file-move behavior is carried into the supported design.
