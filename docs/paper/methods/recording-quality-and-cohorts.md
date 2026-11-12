# Observation-only fish assessment

**Scope:** Technical reference for supported commands and implementation/comparison routes. Some recipes below are historical or exploratory. Accepted paper methods are defined by [Current figures](../figures/README.md) and the exact frozen selections; this manual does not supersede them.

The implemented `assess-discarding` command publishes a technical evidence
audit followed by source-linked exploratory legacy-rule screening. Neither
stage excludes fish, freezes a cohort or constitutes paper approval. All new
membership decisions remain `pending_review`; an approved technical-policy
file also cannot activate exclusions through this command. The unfinished policy,
paper-scale validation, and cohort review are tracked in the
[active cohort implementation plan](../../../Plans/02_DISCARDING_AND_COHORT_IMPLEMENTATION.md).

## Technical stage

The command starts from the complete raw inventory and retains incomplete,
ambiguous, requested-but-undiscovered, and failed recordings. Its technical
table records identity/condition, tracking-header schema, processing status,
authenticated artifact lineage, matched and valid frame counts, and protocol
timing evidence.

The optional technical policy JSON can set `approval_status`,
`min_matched_frames`, and `min_valid_frame_fraction`. The default is a draft
evidence audit. An approved policy also requires `approved_by` and
`approved_at`; the command records its evidence without assigning a primary
candidate disposition. Only approved label-independent technical evidence can inform the
reviewed primary manifest. CR strength, US bouts, and learner features cannot.

| Policy field | Default | Accepted value |
| --- | --- | --- |
| `approval_status` | `draft` | `draft` or `approved`. |
| `min_matched_frames` | `1` | Positive integer. |
| `min_valid_frame_fraction` | `0.0` | Number from 0 to 1. |
| `approved_by` | absent | Reviewer identity required when status is `approved`. |
| `approved_at` | absent | Review date/time required when status is `approved`. |

The policy parser is [`load_technical_policy`](../../../src/classical_conditioning/analysis/discarding.py). The routine JSON also selects `assessment_metric` and optional `disabled_discard_checks`; see [03 parameters](../../running-analysis/parameters.md#1-routine-run-inputs).

The assessment bundle uses stable derived paths under
`Processed data/Discarding/<analysis_id>/` and authenticates its inputs with an
assessment hash. Reruns replace the compatibility assessment files atomically,
while observation evidence is retained in immutable snapshots. Raw and
processed inputs are never moved or deleted, and frozen reviewed cohort
manifests remain immutable. The focused command is
`classical-conditioning assess-discarding --raw-dir ... --project-dir ...
--analysis-id ... --experiment ... --metric ...`.

## Exploratory stage

`assess-discarding` runs this second stage immediately after technical
assessment. Every named check has a source comment in the implementation and
source metadata in the rule table. The default combined exploratory status
applies all checks. `--disable-check RULE_ID` can be repeated to inspect the
effect of omitting a check; each run replaces the stable derived assessment
bundle and records its new input hash and step-by-step fish counts. This
combined population is a new rule projection on the selected refactored metric,
not the population produced by any one legacy script.

| Rule ID | Legacy operation and executable behavior |
| --- | --- |
| `readable_fish` | [Preprocessing discard loop](../../../legacy/scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py): unreadable per-fish data were discarded. `RUN_DISCARD` was false by default; the assessment still evaluates it. |
| `last_us` | Same file, `check_viability`: the last **observed** US event needs an end at least 0.4 s after onset and a bout in the inclusive 0–5 s window. |
| `train_us` | Same file, `check_train`: observed Train US trials need a bout in the inclusive 0–5 s window. |
| `retrain_us` | Same file, `check_retrain`: apply that rule only when Re-Train is declared; ordinary late US events are not automatically Re-Train. |
| `baseline_bouts` | Same file, `check_baseline`: Late Pre-train 10–14, Early Test 65–69, and Late Test 90–94 each need three trials with a bout in the inclusive −15–0 s window. |
| `cr_bouts` | Same file, `check_cr`: the same blocks each need three bout-bearing trials in the experiment's inclusive CR window (0–9, 0–13, or 0–20 s). The executable empty-selected-CS bypass is reported. Trial 65 counts in Early Test even though it is a catch trial. |
| `discard_propagation` | [Grouping](../../../legacy/scripts/3_FishGrouping.py) used the discard list for heatmap grids but not pooled rows; [scaled vigor](../../../legacy/scripts/4_ScaledVigorPlotting.py) applied it by default; [normalized vigor](../../../legacy/scripts/5_NormalizedVigorPlotting.py) did not by default. These reuses are recorded, not counted as fresh independent checks. |
| `learner_inputs` | One merged pre-fit prerequisite from [original](../../../legacy/scripts/6_LearnersQuantification.py), [new](../../../legacy/scripts/6_LearnersQuantification_new.py), [improved](../../../legacy/scripts/6_LearnersQuantification_improved.py), and [WIP](../../../legacy/scripts/6_LearnersQuantification_WIP.py): finite positive baseline/response/ratio; at least six trials in each selected epoch and three in every required five-trial block. This does not run four classifiers or decide learner labels. |

The compatibility tables above retain their original rule family. Additional
observation tables cover normalized-vigor missing fields, the six-trial
ANY-block rule, selected paired-block completeness and each learner variant's
instantiated input/epoch/block requirements. These are labeled projections on
corrected outcomes. Original pooled-table metadata and window NaN fractions
are not invented when unavailable. Model-derived learner feature failures remain
unevaluable until the learner actually runs. An unevaluable rule is
never silently treated as a pass. Neither this exploratory result nor a
disabled check can alter the reviewed primary technical cohort. The command
never writes or reads active legacy discard lists and never moves files.

Outputs include technical and exploratory fish tables, rule-level statuses,
detail evidence, cumulative flow, source identity, and the assessment hash.

## Reading and retaining the evidence

`assessment-summary.json` contains an `observation` reference with the immutable
summary path, SHA-256 and evidence identity. `observation-current.json` is only a
navigation pointer. Retained evidence lives under
`Processed data/Discarding/<analysis_id>/evidence/<evidence_id>/`, with matching
Parquet and CSV versions of:

- `inventory`: all discovered/requested/expected cases, processing reasons and pending decisions.
- `historical-checks` and `historical-details`: evaluated results, enabled flags and independent reasons.
- `later-checks`: normalized-vigor and variant-specific learner projections.
- `window-evidence`: saved outcome values and available sample/coverage/bout counts.

Independent checks continue after a failure. Disabling a historical check changes
the hypothetical compatibility flow, but its evaluated evidence remains recorded.
No table here is an active discard list or a reviewed cohort.

Use repeated `--expected-recording-id ID` arguments to retain expected recordings
that are absent from discovery without adding them to the processing selection.
The command's `--observation-only` flag documents the always-enabled mode; there
is no exclusion activation option.

For exact statistical preparation evidence from a saved legacy table, additionally
pass `--legacy-pooled-input PATH --legacy-required-block BLOCK` (repeat the block
argument). The CSV/Parquet must already have the original condition/metadata scope
and block remapping; retain its normalized-vigor, baseline, response and other
original columns. The observer records source bytes, required blocks, LME row
and ANY-block preparation, broad `dropna()`/block medians and the complete-block
Wilcoxon branch. It neither fits tests nor feeds those projected rows to an
analysis. Pickles and automatic newest-file discovery are not accepted.

## Existing analyses and contribution records

Population metric summaries, model-input/LME/resampling consumers, cohort sample
flow and eligibility, learning-onset analysis and Figure 4 sample flow now attach
observation references to their existing summaries. Sidecars retain actual
contributions or the already-calculated eligibility/diagnostic tables. Included
fish with no usable observations stay visible in sample flow or contribution
counts. Legacy learner workers additionally observe preparation/coverage/feature
calls and record model failures without rerunning those operations.

Observers run on copies or already-calculated results and cannot replace analysis
inputs. Failures are recorded as observation failures. A failed pipeline assessment
does not block population comparisons or change their recording list. Existing
analytical masks, scientific definitions and reviewed cohorts retain their
recorded behavior. No paper-scale assessment, policy review or new cohort freeze
is implied by these engineering changes.
