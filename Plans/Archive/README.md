# Completed Plan Archive

This folder contains detailed implementation plans whose exit gates have
passed for the scoped work that was authorized (pilot / local fixtures /
`allDelay` where noted). Archived files live flat in this folder (no nested
`steps/` or `Notes/` directories).

## Completed steps

| Step | Completed | Scope | Evidence |
| --- | --- | --- | --- |
| [00A — First local single-fish intake](./00A_SINGLE_FISH_LOCAL_INTAKE.md) | 2026-08-29 | Pilot intake | `fe02193`, `6e3c4dd`; 14 tests; unchanged source hashes |
| [00B — Preexisting codebase characterization](./00B_PREEXISTING_CODEBASE_CHARACTERIZATION.md) | 2026-08-30 | Characterization | Legacy routes characterized; LogMedian pilot authenticated |
| [03 — Domain and configuration](./03_DOMAIN_AND_CONFIGURATION.md) | 2026-08-31 | `allDelay` local | `resolve-config`, FishKey, trial map, stage hashes |
| [04 — Ingestion and raw validation](./04_INGESTION_AND_RAW_VALIDATION.md) | 2026-08-31 | Local fixtures | Readers, validate-raw, Gate T0, tracking audits (2 fish) |
| [05 — Legacy preprocessing equivalence](./05_LEGACY_PREPROCESSING_EQUIVALENCE.md) | 2026-08-31 | Local fixtures | `legacy-paper-v1`; pickle reciprocal timebase classified |

## Archived source plans

| Plan | Why archived |
| --- | --- |
| [Codebase migration plan](./CODEBASE_MIGRATION_PLAN.md) | Absorbed into MASTER §3.3 and REPOSITORY_MIGRATION_MAP; dual-lane script path retired |
| [Scientific figure pipeline plan](./SCIENTIFIC_FIGURE_PIPELINE_PLAN.md) | Operational home moved to Step 12; interactive HTML frozen (Gate F) |

## Archived notes

| Note | Why archived |
| --- | --- |
| [Pickle timebase handoff](./HANDOFF_2026-08-31_PICKLE_TIMEBASE.md) | Decision recorded; active handoff moved to two-fish pipeline notes |
| [Historical LogMedian pipeline](./HISTORICAL_LOGMEDIAN_PIPELINE.md) | Historical route evidence; covered by Archive Step 00B; not on critical path |

When a step completes:

1. move its file from `Plans/<STEPFILE>.md` to `Plans/Archive/<STEPFILE>.md`;
2. add completion date, implementation commit IDs, validation evidence, and
   final artifact/version IDs;
3. update this index, `../README.md`, and
   `../IMPLEMENTATION_STEP_INDEX.md`;
4. keep the master plan active and current.

Archived plans are historical evidence. Later corrections create new plan
steps or amendments rather than rewriting completion history.

**Do not archive yet:** Steps 01–02 and 06–13 (still open), active domain
annexes (`TAIL_DYNAMICS_…`, learner plan, deferred imaging),
`DECISIONS.md`, `IMPLEMENTATION_STEP_INDEX.md`, or the master plan.
