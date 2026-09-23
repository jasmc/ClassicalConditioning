# Analysis Documentation

This folder documents the current analysis behavior and proposed architecture, records
audits, and preserves legacy references. Future work belongs under `Plans/`;
status belongs only in `Plans/IMPLEMENTATION_STEP_INDEX.md`.

## Start by task

| If you need to… | Read this |
| --- | --- |
| Install and run the candidate route for the first time | [Root README](../../README.md), then [candidate workflow](./USER_WORKFLOW.md) |
| Find an output, review hashes/markers, or understand reuse | [Output and provenance](./OUTPUT_AND_PROVENANCE.md) |
| Recover from a failure without breaking lineage | [Troubleshooting](./TROUBLESHOOTING.md) |
| Understand candidate metrics or shared bouts | [Activity metrics, smoothing, and shared bouts](./METRICS_AND_BOUTS.md) |
| Render or interpret a candidate figure | [Figure guide](./FIGURE_GUIDE.md) |
| Translate project terminology | [Glossary](./GLOSSARY.md) |
| Review source ownership and call order | [Current pipeline guide](./CURRENT_PIPELINE_GUIDE.md) |
| Review cohort aggregation or response/baseline figures | [Cohort aggregation and figures](./COHORT_AGGREGATION_AND_FIGURES.md) |
| Review learning-onset analysis | [LME pipeline and parameter reference](./LME_PIPELINE_AND_PARAMETERS.md) |
| Review technical and exploratory selection evidence | [Discarding assessment](./DISCARDING_ASSESSMENT.md) |

## Architecture and implementation reference

- [Proposed multimodal architecture](./ARCHITECTURE.md)
- [Current pipeline guide](./CURRENT_PIPELINE_GUIDE.md)

## Figures

- [Legacy and refactored figure pipelines](./figures/FIGURE_PIPELINES.md)
- [Paper draft-to-current figure comparison](./figures/PAPER_DRAFT_COMPARISON.md)

## Audits and issue tracking

- [Analysis findings](./audits/ANALYSIS_FINDINGS.md)
- [Issue register](./audits/ISSUE_REGISTER.md)
- [Inherited imaging critique](./audits/IMAGING_PIPELINE_CRITIQUE.md)

Audits describe evidence and risk. They do not, by themselves, authorize or
schedule implementation.

## Legacy reference

- [Analysis files and historical execution order](./legacy/ANALYSIS_FILES_INDEX.md)
- [Preexisting codebase behavior map](./legacy/CODEBASE_BEHAVIOR_MAP.md)
- [Legacy versus corrected workflow](./legacy/LEGACY_VS_CORRECTED_WORKFLOW.md)
- [Learner-variant behavior matrix](./legacy/LEARNER_VARIANT_BEHAVIOR_MATRIX.md)

## Historical records

- [Implementation status snapshot, 2026-09-01](./archive/CURRENT_IMPLEMENTATION_STATUS_2026-09-01.md)
- [Former combined architecture/status map](./archive/FINAL_ARCHITECTURE_STATUS_MAP.md)
- [README reorganization record](./archive/README_REORGANIZATION.md)
- [Historical pipeline and heatmap audit](./archive/PIPELINE_AND_HEATMAP_AUDIT.md)

These records are retained for provenance but are not updated and are not
current status authority. Repository maintenance guidance lives in the
[maintenance guide](../maintenance/REPOSITORY_GUIDE.md).
