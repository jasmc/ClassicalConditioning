# README review and reorganization plan

This document independently reviews the root [README](../../README.md) as it
exists on 2026-09-18. It is a documentation plan, not a replacement for the
README. No README content has been removed or moved as part of this change.

## What is working well

- It identifies the supported candidate-only route and clearly distinguishes
  archived historical code.
- It explains provenance, output safety, raw-data immutability, and the
  difference between generated tables, QC evidence, metadata, and figures.
- It gives unusually useful scientific context for metrics, bout detection,
  coverage, and figure interpretation.
- It contains concrete install, configuration, and command examples.

## Critique

1. **It serves too many readers in one linear document.** A new user needs an
   install-and-run path, whereas an analyst reviewing the validity of a bout
   outcome needs the scientific-method material. They must currently scroll
   through each other’s material.
2. **The operational quick start is buried under detail.** The first runnable
   command appears early, but selecting raw data, copying the example config,
   locating outputs, and reviewing success are spread across several sections.
3. **The document repeats its conceptual boundaries.** Candidate status,
   metric-versus-detector separation, and provenance are each explained well,
   but related explanations occur in multiple distant locations. Cross-links
   could let one canonical explanation serve several sections.
4. **The command reference is useful but flat.** It lacks an explicit mapping
   from a routine workflow to its diagnostic/recovery command and expected
   artifact. Readers must infer the dependency order from surrounding text.
5. **Figures combine production instructions and interpretation.** The exact
   invocation examples belong close to command usage; detailed interpretation,
   colour scaling, and limitations form a reference guide that merits its own
   page.
6. **Scientific status is too late for a fast reader.** The candidate route’s
   exploratory status should appear in the opening workflow summary as well as
   in the dedicated status section.
7. **Maintenance guidance is mixed with user-facing analysis guidance.** The
   repository-cleanup section is valuable but should not compete with the
   pipeline documentation for attention.
8. **Some support documents are described only by title.** A reader cannot
   quickly tell which document answers reproducibility, input schema,
   troubleshooting, or scientific-decision questions.

## Implementation status

The following work is complete:

- The README now has a top-level table of contents, a new-reader path, and a
  detailed current save-directory tree.
- `USER_WORKFLOW.md`, `OUTPUT_AND_PROVENANCE.md`, `TROUBLESHOOTING.md`,
  `FIGURE_GUIDE.md`, `GLOSSARY.md`, and `docs/maintenance/REPOSITORY_GUIDE.md`
  now exist and are linked from both the root README and the task-oriented
  analysis index.
- The root README retains its original detailed material during this migration.
  This deliberate temporary duplication means no scientific or operational
  content was lost while the new guides were introduced.

The metrics/bouts, figures, and repository-maintenance sections have now been
consolidated into their specialist guides, with concise README summaries and
links. Remaining editorial consolidation is limited to the configuration and
command reference: preserve their first-run details in the README while moving
only repetition to the workflow guide, one topic at a time with a
commit-level content inventory and redirect link.

## No-content-loss reorganization plan

The target is a short root README that is an accurate landing page, with
specialist material moved verbatim first and then edited only for clarity.
Every move must leave a redirecting link in the original location and be
checked against the content inventory below.

| Phase | Destination | Content to move or consolidate | Completion evidence |
| --- | --- | --- | --- |
| 1 | Root `README.md` | Keep project purpose, scientific-status warning, install, 5-minute candidate run, output-tree summary, command index, and links. | **In progress:** navigation is complete; detailed duplicated material awaits reviewed consolidation. |
| 2 | `docs/analysis/USER_WORKFLOW.md` | Expand “Run from a config file,” raw-file rules, pipeline stages, failure semantics, resuming, and output review. | **Implemented:** includes an end-to-end workflow, review order, and safe resume guidance. |
| 3 | `docs/analysis/METRICS_AND_BOUTS.md` | Consolidate the active metric definitions, shared detector explanation, mathematical conventions, and scientific limitations. | **Existing canonical guide:** retain and cross-link; expand only after a scientific decision changes. |
| 4 | `docs/analysis/FIGURE_GUIDE.md` | Move all figure commands, axes, masked-cell meaning, scaling, colour bars, and cohort-figure interpretation. | **Implemented:** contains output modes, command, interpretation, and colour/coverage rules. |
| 5 | `docs/analysis/OUTPUT_AND_PROVENANCE.md` | Expand the save-directory tree, marker/hash semantics, overwrite behaviour, and artifact troubleshooting. | **Implemented:** directory contract, lineage/reuse rules, and output-review checklist. |
| 6 | `docs/maintenance/REPOSITORY_GUIDE.md` | Move repository-folder cleanup guidance and historical-archive policy. | **Implemented:** active/archive boundary and raw-data/maintenance rules. |
| 7 | `docs/analysis/README.md` | Turn the existing analysis index into a task-oriented documentation map. | **Implemented:** a task-to-guide table now leads the index. |

## Documentation gaps and placeholders to add

The first-pass guides below now exist. Their remaining work is bounded expansion
only where a claim can be verified from code or an explicit scientific decision;
the documentation must not invent unverified behaviour.

| Placeholder | Required explanation before it is considered complete |
| --- | --- |
| `docs/analysis/TROUBLESHOOTING.md` | **Implemented baseline.** Add verified error-message-to-recovery examples when stable command output is available. |
| `docs/analysis/OUTPUT_AND_PROVENANCE.md` | **Implemented baseline.** Add a generated exact filename-pattern appendix from recipe constants; do not hand-maintain guessed stems. |
| `docs/analysis/USER_WORKFLOW.md` | **Implemented.** Add screenshots only if a stable representative dataset can be published. |
| `docs/analysis/FIGURE_GUIDE.md` | **Implemented.** Add figure-specific provenance-sidecar field examples after they are verified against an exported figure. |
| `docs/maintenance/REPOSITORY_GUIDE.md` | **Implemented.** Update when project data-storage policy changes. |
| `docs/analysis/GLOSSARY.md` | **Implemented.** Keep it synchronized with new recipe-facing terminology. |

## Guardrails for the reorganization

1. Move text with history-preserving commits; do not delete a source paragraph
   until its destination and inbound README link exist.
2. Preserve every command example and state whether it is routine, diagnostic,
   benchmark-only, or historical.
3. Do not turn scientific caveats into implementation claims. Missing evidence
   remains a named limitation or placeholder until it is supplied.
4. Update the root TOC and the analysis documentation index after every move.
5. Verify Markdown links and examples after each phase; documentation is part
   of the reproducibility interface.
