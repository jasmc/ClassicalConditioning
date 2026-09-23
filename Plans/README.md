# ClassicalConditioning Plans

## Purpose

This folder contains future work and the small set of governance files needed
to manage it. Descriptions of current architecture, behavior, audits, and legacy
routes live under [`docs/analysis/`](../docs/analysis/README.md).

All processing and artifacts remain local. No plan authorizes a scientific
correction by itself: decisions are recorded, implemented under stable recipe
names with authenticated inputs and settings, and validated before they become
paper-authoritative.

## Start here

- [Analysis governance](./GOVERNANCE.md) — durable engineering and scientific invariants
- [Decision register](./DECISIONS.md) — approved decisions and open gates
- [Implementation status index](./IMPLEMENTATION_STEP_INDEX.md) — the only live status board
- [Analysis and statistics](./Analysis/1_ANALYSIS_AND_STATISTICS.md) — population inference and model design
- [Remaining cohort and learning-onset implementation](./REMAINING_COHORT_AND_LEARNING_ONSET_IMPLEMENTATION.md) — ordered implementation work, artifacts, tests, and exit gates
- [Single cohort and exclusion boundary](./SINGLE_COHORT_AND_EXCLUSION.md) — technical cohort review, exploratory sensitivity, and downstream eligibility boundaries
- [Exclusion and selection inventory](./EXCLUSION_AND_SELECTION_INVENTORY.csv) — machine-readable register of every active, legacy, and deferred selection rule
- [Learner classification and stratified analysis](./Analysis/2_LEARNER_CLASSIFICATION.md) — required paper workstream
- [Integrated single-metric cohort and CR profiles](./Analysis/4_INTEGRATED_SINGLE_METRIC_COHORT_AND_CR_PROFILES.md) — post-classification cohort/learner orchestration and catch/block figures
- [Figures and reproducible reporting](./Analysis/FIGURES_AND_REPRODUCIBLE_REPORTING.md) — CLI, notebooks, panel data, and paper figures
- [Releases and legacy retirement](./RELEASES_AND_LEGACY_RETIREMENT.md)

## Active plans

### Analysis

| Plan | Role |
| --- | --- |
| [Analysis and statistics](./Analysis/1_ANALYSIS_AND_STATISTICS.md) | Claims, estimands, outcomes, LME alternatives, diagnostics, uncertainty, and population robustness |
| [Remaining cohort and learning-onset implementation](./REMAINING_COHORT_AND_LEARNING_ONSET_IMPLEMENTATION.md) | Critical-path delivery plan connecting the cohort boundary to onset inference and the final figure |
| [Learner classification and stratified analysis](./Analysis/2_LEARNER_CLASSIFICATION.md) | Required learner-method review, manifest, validation, outputs, and non-circular inference |
| [Non-LME learner methods review](./Analysis/LEARNER_CLASSIFICATION_NON_LME_REVIEW.md) | Active Gate L discussion of continuous scores, trajectories, probabilistic methods, and validation; no method approved |
| [Learner-stratified outputs and validation](./Analysis/3_LEARNER_STRATIFIED_OUTPUTS.md) | Detailed schemas, joins, temporal analyses, modules, output inventory, phases, and validation checklist |
| [Integrated single-metric cohort and CR profiles](./Analysis/4_INTEGRATED_SINGLE_METRIC_COHORT_AND_CR_PROFILES.md) | Shared catch/block estimands, legacy-discard sensitivity boundary, and final post-classification orchestration |
| [Figures and reproducible reporting](./Analysis/FIGURES_AND_REPRODUCIBLE_REPORTING.md) | Stage interfaces, notebooks, panel data, figure registry, paper automation, and QC |

Learner analysis is required for the paper. Whether its final representation is
continuous, probabilistic, categorical, or combined remains a scientific
decision, not an omission option.

### Release

- [Reproduction releases, documentation, and legacy retirement](./RELEASES_AND_LEGACY_RETIREMENT.md)

## Deferred plans

Deferred means preserved future work that does not block the present behavioral
paper path.

- [Schema, semantic provenance, and legacy conversion](./Deferred/SCHEMA_SEMANTIC_PROVENANCE_AND_LEGACY_CONVERSION.md)
- [Behavior and imaging integration](./Deferred/BEHAVIOR_IMAGING_INTEGRATION.md)
- [Tail mechanistic analyses](./Deferred/TAIL_MECHANISTIC_ANALYSES.md)

## Documentation and evidence

- [Analysis documentation index](../docs/analysis/README.md)
- [Current pipeline guide](../docs/analysis/CURRENT_PIPELINE_GUIDE.md)
- [Architecture](../docs/analysis/ARCHITECTURE.md)
- [Activity metrics, smoothing, and shared bouts](../docs/analysis/METRICS_AND_BOUTS.md)
- [Analysis findings](../docs/analysis/audits/ANALYSIS_FINDINGS.md)
- [Issue register](../docs/analysis/audits/ISSUE_REGISTER.md)

Documentation explains what exists or what was observed. It does not compete
with this folder as an implementation queue.

## Archive

[`Archive/`](./Archive/README.md) contains three clearly distinguished classes:

1. completed scoped plans with recorded evidence;
2. plans archived incomplete by explicit decision; and
3. superseded source plans retained for history.

Archive does not automatically mean complete. Historical numbered filenames
remain unchanged because their identifiers are part of the project record.
The active `Analysis/` plans use numeric prefixes to make their reading order
explicit; the cross-cutting active plans at the `Plans/` root use descriptive
names.

## Lifecycle

- Add future implementation to the narrowest applicable active plan.
- Record cross-cutting scientific decisions in `DECISIONS.md`.
- Update status only in `IMPLEMENTATION_STEP_INDEX.md`.
- Move stable descriptions of implemented behavior to `docs/analysis/`.
- Archive a plan only after completion, explicit retirement, or replacement by
  a named active document.
- When archiving, record which of those three reasons applies and update links
  in the same change.
- Keep user-facing commands and formats synchronized with the repository README.

## Working rule

Later stages may be prototyped on synthetic or local fixtures, but they do not
become paper-authoritative while their upstream scientific contracts, cohort,
validation mode, or release evidence remain open.
