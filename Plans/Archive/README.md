# Plan Archive

This folder preserves completed scoped plans, explicitly retired incomplete
plans, superseded source plans, and historical notes. Archive status does not
by itself mean that an exit gate passed; the sections below state which kind of
archive each file represents.

## Completed steps

| Step | Completed | Scope | Evidence |
| --- | --- | --- | --- |
| [00A — First local single-fish intake](./00A_SINGLE_FISH_LOCAL_INTAKE.md) | 2026-08-29 | Pilot intake | `fe02193`, `6e3c4dd`; 14 tests; unchanged source hashes |
| [00B — Preexisting codebase characterization](./00B_PREEXISTING_CODEBASE_CHARACTERIZATION.md) | 2026-08-30 | Characterization | Legacy routes characterized; LogMedian pilot authenticated |
| [03 — Domain and configuration](./03_DOMAIN_AND_CONFIGURATION.md) | 2026-08-31 | `allDelay` local | `resolve-config`, FishKey, trial map, stage hashes |
| [04 — Ingestion and raw validation](./04_INGESTION_AND_RAW_VALIDATION.md) | 2026-08-31 | Local fixtures | Readers, validate-raw, Gate T0, tracking audits (2 fish) |
| [05 — Legacy preprocessing equivalence](./05_LEGACY_PREPROCESSING_EQUIVALENCE.md) | 2026-08-31 | Local fixtures | `legacy-paper-v1`; pickle reciprocal timebase classified |
| [02A — Artifact integrity and publication](./02A_ARTIFACT_INTEGRITY_AND_PUBLICATION.md) | 2026-09-15 | Implemented package routes | SHA-256 source/output lineage, lossless intake verification, and transactional publication |
| [10A — Exploratory inference scaffolding](./10A_EXPLORATORY_INFERENCE_SCAFFOLDING.md) | 2026-09-15 | Engineering scaffolding | Authenticated model input, exploratory LME, fish permutation, and fish bootstrap; no paper inference approval |

## User-directed incomplete archives

These files are archived for plan-pack cleanup, not because their exit gates
passed. Their uncompleted requirements remain visible in the files and the
implementation index.

| Step | Archived | Remaining status |
| --- | --- | --- |
| [00 — Governance and baseline](./00_GOVERNANCE_AND_BASELINE.md) | 2026-09-15 | No paper-scope baseline manifest, configuration freeze, or output fingerprint evidence found |
| [01 — Package and environment](./01_PACKAGE_AND_ENVIRONMENT.md) | 2026-09-15 | Foundation implemented; final Step 00-scoped pinned numerical reference not found |
| [06 — Tail representation and candidates](./06_TAIL_REPRESENTATION_AND_CANDIDATES.md) | 2026-09-15 | Candidate implementation retained; Gate P and remaining known-answer checks did not pass |
| [07 — Metric validation and selection](./07_METRIC_VALIDATION_AND_SELECTION.md) | 2026-09-15 | Shared bout segmentation implemented; Gate T1 metric/smoothing/threshold/partition selection remains open |
| [08 — Full reprocessing, QC, and cohort](./08_FULL_REPROCESSING_QC_AND_COHORT.md) | 2026-09-15 | Fixture batch/cohort plumbing retained; paper-scale QC and C1 cohort hash remain open |
| [09 — Outcomes and temporal profiles](./09_OUTCOMES_AND_TEMPORAL_PROFILES.md) | 2026-09-15 | Two-fixture corrected outcomes retained; C1/Gate O paper outcome contract remains open |
| [Single cohort and exclusion boundary](./SINGLE_COHORT_AND_EXCLUSION.md) | 2026-09-23 | Consolidated into the active cohort plan; paper policy, C1 cohort, and consumer migration remain open |
| [Technical assessment](./TECHNICAL_ASSESSMENT.md) | 2026-09-23 | Implemented command behavior moved to the discarding assessment guide; numerical paper policy and reviewed cohort remain open |
| [Learning-onset inference design](./LEARNING_ONSET_LME.md) | 2026-09-23 | Superseded by the active onset plan and current LME reference; Gate O/S choices and paper-scale validation remain open |
| [Combined cohort and learning-onset work](./REMAINING_COHORT_AND_LEARNING_ONSET_IMPLEMENTATION.md) | 2026-09-23 | Split into separate active cohort and onset plans; neither exit gate has passed |
| [Schema, semantic provenance, and legacy conversion](./SCHEMA_SEMANTIC_PROVENANCE_AND_LEGACY_CONVERSION.md) | 2026-09-23 | Optional engineering design retired unstarted; no concrete need for semantic hashing or a resolver in the single-release analysis |

## Archived source plans

| Plan | Why archived |
| --- | --- |
| [Codebase migration plan](./CODEBASE_MIGRATION_PLAN.md) | Durable rules consolidated in active governance; dual-lane script path retired |
| [Master analysis migration plan](./MASTER_ANALYSIS_MIGRATION_PLAN.md) | Durable invariants consolidated in `../GOVERNANCE.md`; numbered roadmap no longer represents the active plan structure |
| [Repository migration map](./REPOSITORY_MIGRATION_MAP.md) | Repository rules consolidated in governance; current architecture/behavior moved to documentation |
| [Scientific figure pipeline plan](./SCIENTIFIC_FIGURE_PIPELINE_PLAN.md) | Operational content merged into `../08_FIGURES_AND_REPRODUCIBLE_REPORTING.md`; interactive HTML frozen |
| [Paper figure automation plan](./PAPER_FIGURE_AUTOMATION_PLAN.md) | Registry, paper-panel scope, blockers, build sequence, and QC merged into the active figure plan |
| [Step 11 learner classification](./11_LEARNER_CLASSIFICATION.md) | Exact historical source retained; all work packages, tests, deliverables, and exit criteria restored in `../05_LEARNER_ANALYSIS.md` |
| [Learner-stratified vigor plan](./LEARNER_STRATIFIED_VIGOR_ANALYSIS_PLAN.md) | Exact historical source retained; its detailed design is restored in `../06_LEARNER_OUTPUTS_AND_VALIDATION.md` |
| [Tail dynamics and vigor plan](./TAIL_DYNAMICS_VIGOR_ANALYSIS_PLAN.md) | Implemented behavior documented in `../../docs/analysis/METRICS_AND_BOUTS.md`; exploratory mechanisms moved to `../09_TAIL_MECHANISTIC_ANALYSES.md` |
| [Former staged release roadmap](./RELEASES_AND_LEGACY_RETIREMENT.md) | R1–R4 staging superseded by the one [final analysis release](../10_FINAL_ANALYSIS_RELEASE.md); no staged release gate was completed |

## Archived notes

| Note | Why archived |
| --- | --- |
| [Pickle timebase handoff](./HANDOFF_2026-08-31_PICKLE_TIMEBASE.md) | Decision recorded; active handoff moved to two-fish pipeline notes |
| [Two-fish candidate handoff](./HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md) | Local-fixture evidence retained; active Gate O/S requirements live in the analysis/statistics plan |
| [Historical LogMedian pipeline](./HISTORICAL_LOGMEDIAN_PIPELINE.md) | Historical route evidence; covered by Archive Step 00B; not on critical path |
| [Retired five-metric pilot evidence](./RETIRED_FIVE_METRIC_PILOT_EVIDENCE.md) | Local detector/smoothing evidence retained after the active set was reduced to three metrics |
| [Statistics methodology workshop](./STATISTICS_METHODOLOGY_WORKSHOP.md) | Completed critique retained; active discussion and decisions moved to unified Step 10 |
| [Analysis readiness discussion](./ANALYSIS_READINESS_DISCUSSION.md) | Blockers and decision agenda merged into the active analysis/statistics plan and status index |

When a step completes:

1. move its file from `Plans/<STEPFILE>.md` to `Plans/Archive/<STEPFILE>.md`;
2. add completion date, implementation commit IDs, validation evidence, and
   final artifact/version IDs;
3. update this index, `../README.md`, and
   `../IMPLEMENTATION_STEP_INDEX.md`;
4. move stable descriptions of implemented behavior to `docs/analysis/`.

Archived plans are historical evidence. Later corrections create new plan
steps or amendments rather than rewriting completion history.

**Do not archive yet:** the numbered active plans in `../`,
deferred plans still preserving future work, `../GOVERNANCE.md`,
`../DECISIONS.md`, or `../IMPLEMENTATION_STEP_INDEX.md`.
