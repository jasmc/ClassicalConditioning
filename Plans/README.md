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

- **Active handoff:** [Notes/HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md](./Notes/HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md)
- [Consolidated decisions](./DECISIONS.md)
- [Statistics methodology workshop (Step 10.0)](./Notes/STATISTICS_METHODOLOGY_WORKSHOP.md)
- [Master analysis migration plan](./MASTER_ANALYSIS_MIGRATION_PLAN.md)
- [Analysis architecture and alternative routes](./ANALYSIS_ARCHITECTURE.md)
- [Repository migration map](./REPOSITORY_MIGRATION_MAP.md)
- [Implementation step index](./IMPLEMENTATION_STEP_INDEX.md)
- [Analysis issue traceability](./ANALYSIS_ISSUES_TRACEABILITY.md)
- [Tail dynamics and vigor analysis plan](./TAIL_DYNAMICS_VIGOR_ANALYSIS_PLAN.md)
- [Deferred behavior and imaging integration plan](./BEHAVIOR_IMAGING_INTEGRATION_PLAN.md)

## Source plans and audits

The master plan integrates, but does not replace, these repository documents.
The analysis-issues audit is treated as a risk and evidence register: its
findings must be verified and resolved, but its proposed remedies do not become
scientific decisions automatically.

- [Current codebase migration plan](./CODEBASE_MIGRATION_PLAN.md)
- [Analysis file index](../docs/analysis/ANALYSIS_FILES_INDEX.md)
- [Analysis issues](../docs/analysis/ANALYSIS_ISSUES.md)
- [Inherited imaging pipeline critique](../docs/analysis/IMAGING_PIPELINE_CRITIQUE.md)
- [Learner-stratified vigor plan](./LEARNER_STRATIFIED_VIGOR_ANALYSIS_PLAN.md)
- [Scientific figure pipeline plan](./SCIENTIFIC_FIGURE_PIPELINE_PLAN.md)
- [Repository overview](../README.md)

If two implementation documents conflict, use the master plan's decision
hierarchy and record the resolution before implementation.

## Step files

| Step | Document | Primary result |
| --- | --- | --- |
| 00 | [Governance and baseline](./steps/00_GOVERNANCE_AND_BASELINE.md) | Frozen current pipeline and paper scope |
| 00A | [First local single-fish slice](./Archive/steps/00A_SINGLE_FISH_LOCAL_INTAKE.md) — complete | Lossless local ingestion and acquisition integrity report |
| 00B | [Preexisting codebase characterization](./Archive/steps/00B_PREEXISTING_CODEBASE_CHARACTERIZATION.md) — complete | Executable behavior contracts before replacement |
| 01 | [Package and environment](./steps/01_PACKAGE_AND_ENVIRONMENT.md) | Installable, testable Python foundation |
| 02 | [Artifacts, schemas, and provenance](./steps/02_ARTIFACTS_SCHEMAS_AND_PROVENANCE.md) | Lossless Parquet/JSON artifact contract |
| 03 | [Domain and configuration](./Archive/steps/03_DOMAIN_AND_CONFIGURATION.md) — complete (`allDelay`) | Immutable, stage-scoped resolved configuration |
| 04 | [Ingestion and raw validation](./Archive/steps/04_INGESTION_AND_RAW_VALIDATION.md) — complete (local fixtures) | Explicit, validated raw scientific tables |
| 05 | [Legacy preprocessing equivalence](./Archive/steps/05_LEGACY_PREPROCESSING_EQUIVALENCE.md) — complete (local fixtures) | Exact reproduction of current point-15 processing |
| 06 | [Tail representation and candidates](./steps/06_TAIL_REPRESENTATION_AND_CANDIDATES.md) | Versioned shared tail representation and candidate metrics |
| 07 | [Metric validation and selection](./steps/07_METRIC_VALIDATION_AND_SELECTION.md) | Frozen metric recipe selected without test-set leakage |
| 08 | [Full reprocessing, QC, and cohort](./steps/08_FULL_REPROCESSING_QC_AND_COHORT.md) | Corrected full dataset and frozen cohort |
| 09 | [Outcomes and temporal profiles](./steps/09_OUTCOMES_AND_TEMPORAL_PROFILES.md) | Fish-aware trial and time-resolved outcomes |
| 10 | [Statistics and sensitivity](./steps/10_STATISTICS_AND_SENSITIVITY.md) | Diagnosed primary inference and robustness results |
| 11 | [Learner classification](./steps/11_LEARNER_CLASSIFICATION.md) | Versioned, non-circular learner analysis |
| 12 | [Figures, CLI, and notebooks](./steps/12_FIGURES_CLI_AND_NOTEBOOKS.md) | Reusable figures and consistent execution interfaces |
| 13 | [Releases and legacy retirement](./steps/13_RELEASES_AND_LEGACY_RETIREMENT.md) | Immutable legacy and corrected paper releases |

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

- `Plans/steps/` contains work that is not yet implemented.
- After a step passes its exit gate, move its detailed plan to
  `Plans/Archive/steps/`.
- Update this README and the implementation index in the same documentation
  commit so links and status remain accurate.
- Keep the master plan current; do not archive it while the migration is active.
- Update the repository README whenever user-facing commands, formats, or
  active analysis versions change.

See [the archive index](./Archive/README.md) for completed steps.

## Historical analysis notes

- [Handwritten LogMedian pipeline interpretation](./Notes/HISTORICAL_LOGMEDIAN_PIPELINE.md)
- [Archived pickle timebase handoff](./Archive/Notes/HANDOFF_2026-08-31_PICKLE_TIMEBASE.md)
