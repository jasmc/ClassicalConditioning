# Numbered analysis documentation

Read the guides in order for the active corrected behavior route. The numbers describe the analysis flow; the paper figure and historical folders have their own numbered indexes. The [root README](../../README.md) is the short setup and run entry point. Code and authenticated output manifests remain the source of exact run values.

| Step | Guide | Fast answer |
| --- | --- | --- |
| 01 | [Run workflow](01_USER_WORKFLOW.md) | How to configure, run, resume, and inspect a run. |
| 02 | [Current pipeline](02_CURRENT_PIPELINE.md) | Stage order, source ownership, and trust boundaries. |
| 03 | [All analysis parameters](03_ANALYSIS_PARAMETER_INDEX.md) | Run fields, fixed stage settings, experiment values, and the full LME reference. |
| 04 | [Discarding and selection](04_DISCARDING_AND_SELECTION.md) | Technical assessment, exploratory rules, and reviewed cohort boundary. |
| 05 | [Metrics and bouts](05_METRICS_AND_BOUTS.md) | Candidate signals, shared movement detector, and interpretation. |
| 06 | [Cohort aggregation](06_COHORT_AGGREGATION.md) | Fish weighting, ratios, and cohort figure data. |
| 07 | [Learning-onset LME](07_LEARNING_ONSET_LME.md) | Every public LME parameter, estimand, fit, diagnostic, and output. |
| 08 | [Candidate figures](08_CANDIDATE_FIGURES.md) | Descriptive figure families and render commands. |
| 09 | [Output and provenance](09_OUTPUT_AND_PROVENANCE.md) | Where artifacts live and how they are authenticated. |
| 10 | [Troubleshooting](10_TROUBLESHOOTING.md) | Recovery from known run failures. |
| 11 | [Glossary](11_GLOSSARY.md) | Project terms. |
| 12 | [Proposed multimodal architecture](12_PROPOSED_MULTIMODAL_ARCHITECTURE.md) | Future architecture, distinct from implemented behavior. |

## Paper figures

Start with [01 panel-by-panel provenance](figures/01_PAPER_PANEL_PROVENANCE.md): it covers every planned main panel (Figure 1A–H through Figure 4C) and supporting analysis grouped by parent figure, including the source, calculation, renderer, and current gap. Then use the [02 paper specification](figures/02_PAPER_FIGURE_SPECIFICATION.md) for the layout proposal and the [figure folder index](figures/README.md) for review variants and Figure 4 detail.

## Evidence and history

- [Audits](audits/README.md) record findings and open issues; they are evidence, not current run instructions.
- [Legacy references](legacy/README.md) describe inherited implementations and comparisons.
- Dated documentation versions remain in Git history; use the current guides and implementation index for status.
- [Repository maintenance](../maintenance/01_REPOSITORY_GUIDE.md) covers document ownership.

Current implementation status and unfinished scientific gates live in [Plans/IMPLEMENTATION_STEP_INDEX.md](../../Plans/IMPLEMENTATION_STEP_INDEX.md). No candidate output becomes a paper result merely because it renders.
