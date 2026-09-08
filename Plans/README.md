# ClassicalConditioning Plan Pack

## Purpose

This folder contains the implementation plan for migrating the current analysis
into a tested, versioned, reproducible Python package while:

1. reproducing the current numerical results before changing scientific
   behavior;
2. replacing pickle as the canonical storage format;
3. supporting independently runnable analysis stages;
4. comparing versioned tail-activity metrics without overwriting historical
   vigor;
5. producing an auditable corrected paper-analysis release;
6. later adding optional imaging without changing canonical behavior outputs.

All data processing and artifact storage are local. The implementation must not
use Databricks, upload artifacts to Databricks, or require Databricks services.
LLM calls may assist with code and planning, but bulk raw/processed data is not
included in those calls.

No plan in this folder authorizes a scientific correction by itself. Scientific
changes require the decision and validation gates defined in the master plan.

## Start here

- **Active handoff:** [HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md](./HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md)
- [Consolidated decisions](./DECISIONS.md) — living decision authority
- [Implementation step index](./IMPLEMENTATION_STEP_INDEX.md) — **only status board**
- [Statistics methodology workshop (Step 10.0)](./STATISTICS_METHODOLOGY_WORKSHOP.md)
- [Master analysis migration plan](./MASTER_ANALYSIS_MIGRATION_PLAN.md)
- [Analysis architecture and alternative routes](./ANALYSIS_ARCHITECTURE.md)
- [Repository migration map](./REPOSITORY_MIGRATION_MAP.md)
- [Analysis issue traceability](./ANALYSIS_ISSUES_TRACEABILITY.md)
- [Tail dynamics and vigor analysis plan](./TAIL_DYNAMICS_VIGOR_ANALYSIS_PLAN.md)
- [Deferred behavior and imaging integration plan](./BEHAVIOR_IMAGING_INTEGRATION_PLAN.md)

## Document roles

| Document | Role |
| --- | --- |
| This README | Hub and lifecycle only |
| `DECISIONS.md` | Decisions and remaining gates |
| `IMPLEMENTATION_STEP_INDEX.md` | Living Done/Open status |
| `MASTER_ANALYSIS_MIGRATION_PLAN.md` | Engineering constitution |
| `REPOSITORY_MIGRATION_MAP.md` | File moves and package layout |
| `ANALYSIS_ARCHITECTURE.md` | Diagrams and route matrix |
| `ANALYSIS_ISSUES_TRACEABILITY.md` | Issue ↔ gate register |
| Domain annexes | Tail, learner, imaging |
| `00_…`–`13_…` step files | Implementation work packages (flat in this folder) |
| `Archive/` | Completed steps and archived source plans |

## Related audits and domain plans

- [Analysis file index](../docs/analysis/ANALYSIS_FILES_INDEX.md)
- [Analysis issues](../docs/analysis/ANALYSIS_ISSUES.md)
- [Inherited imaging pipeline critique](../docs/analysis/IMAGING_PIPELINE_CRITIQUE.md)
- [Learner-stratified vigor plan](./LEARNER_STRATIFIED_VIGOR_ANALYSIS_PLAN.md)
- [Archived scientific figure pipeline](./Archive/SCIENTIFIC_FIGURE_PIPELINE_PLAN.md) — operational home is Step 12
- [Archived codebase migration plan](./Archive/CODEBASE_MIGRATION_PLAN.md) — absorbed into MASTER §3.3 and REPOSITORY_MAP
- [Repository overview](../README.md)

If two implementation documents conflict, use the master plan's decision
hierarchy and record the resolution in `DECISIONS.md` before implementation.

## Step files

Step status (Done/Open, dependencies, exit evidence) lives only in
[IMPLEMENTATION_STEP_INDEX.md](./IMPLEMENTATION_STEP_INDEX.md). Active step
plans live directly under `Plans/`; completed scoped steps live under
`Plans/Archive/`.

| Step | Document |
| --- | --- |
| 00 | [Governance and baseline](./00_GOVERNANCE_AND_BASELINE.md) |
| 00A | [First local single-fish slice](./Archive/00A_SINGLE_FISH_LOCAL_INTAKE.md) — complete |
| 00B | [Preexisting codebase characterization](./Archive/00B_PREEXISTING_CODEBASE_CHARACTERIZATION.md) — complete |
| 01 | [Package and environment](./01_PACKAGE_AND_ENVIRONMENT.md) |
| 02 | [Artifacts, schemas, and provenance](./02_ARTIFACTS_SCHEMAS_AND_PROVENANCE.md) |
| 03 | [Domain and configuration](./Archive/03_DOMAIN_AND_CONFIGURATION.md) — complete (`allDelay`) |
| 04 | [Ingestion and raw validation](./Archive/04_INGESTION_AND_RAW_VALIDATION.md) — complete (local fixtures) |
| 05 | [Legacy preprocessing equivalence](./Archive/05_LEGACY_PREPROCESSING_EQUIVALENCE.md) — complete (local fixtures) |
| 06 | [Tail representation and candidates](./06_TAIL_REPRESENTATION_AND_CANDIDATES.md) |
| 07 | [Metric validation and selection](./07_METRIC_VALIDATION_AND_SELECTION.md) |
| 08 | [Full reprocessing, QC, and cohort](./08_FULL_REPROCESSING_QC_AND_COHORT.md) |
| 09 | [Outcomes and temporal profiles](./09_OUTCOMES_AND_TEMPORAL_PROFILES.md) |
| 10 | [Statistics and sensitivity](./10_STATISTICS_AND_SENSITIVITY.md) |
| 11 | [Learner classification](./11_LEARNER_CLASSIFICATION.md) |
| 12 | [Figures, CLI, and notebooks](./12_FIGURES_CLI_AND_NOTEBOOKS.md) |
| 13 | [Releases and legacy retirement](./13_RELEASES_AND_LEGACY_RETIREMENT.md) |

## Deferred parallel integration track

The optional two-photon imaging route is intentionally outside the numbered
behavior migration sequence. It may begin only after its prerequisites in
[the behavior and imaging integration plan](./BEHAVIOR_IMAGING_INTEGRATION_PLAN.md)
are met. It does not block behavior-only steps or releases, and it must not
create a second behavior preprocessing route.

## Working rule

Complete a step only when its exit gate passes. A later step may be prototyped
using synthetic inputs, but it must not produce paper-authoritative results
from an upstream artifact that has not passed its gate.

## Plan lifecycle

- Active step plans live directly under `Plans/` (for example `06_….md`).
- After a step passes its exit gate, move its file to `Plans/Archive/`
  (`git mv Plans/<STEPFILE>.md Plans/Archive/<STEPFILE>.md`).
- Update this README and the implementation index in the same documentation
  commit so links remain accurate.
- Keep the master plan current; do not archive it while the migration is active.
- Update the repository README whenever user-facing commands, formats, or
  active analysis versions change.
- Do not duplicate Done/Open status here; update the step index only.
- Do not recreate `Plans/steps/` or `Plans/Notes/` folders.

See [the archive index](./Archive/README.md) for completed steps and archived
source plans.

## Historical analysis notes

- [Archived handwritten LogMedian pipeline interpretation](./Archive/HISTORICAL_LOGMEDIAN_PIPELINE.md)
- [Archived pickle timebase handoff](./Archive/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md)
