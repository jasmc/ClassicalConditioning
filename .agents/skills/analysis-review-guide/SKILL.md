---
name: analysis-review-guide
description: Create or revise a requested substantial ClassicalConditioning HTML study guide or saved-analysis comparison artifact with editable sources.
---

# Analysis Review Guide

Short interpretations and simple figure comparisons stay in chat unless a study
artifact is requested. Use the project figure workflow for exploratory previews.

Create a guide the user can understand and question without another AI explanation.
It explains saved evidence and its limits; it does not select methods, approve
cohorts or freeze figures. Resolve supporting paths below from this skill folder;
resolve project sources through that project's policy and recorded authority.

## Evidence and scientific meaning

Inspect the relevant conversation, selections, code, run settings, saved results
and figures. Use recorded authority, not filenames or browser tabs. Inspect only
sources needed for the requested questions; use targeted reads and summaries
rather than loading whole inventories, large tables or generated HTML into model
context. Build tables, links and repeated case views programmatically.

Present a concise outline and material gaps before building; ask only about
consequential missing scope/preferences and continue independent work. No extra
approval checkpoint is implied. Keep author instructions, recommendations,
corrections and explicit decisions distinct.

- Separate discussed, implemented, tested, run, validated, published and selected
  statuses. Code presence does not prove execution; publication does not prove
  scientific validation. Distinguish scientific and presentation selections.
- Define the scientific question, estimand, metric/outcome, units of replication,
  transformations, timing, aggregation, sign and missing/zero meanings before
  judging a method. Bind counts to actual denominators and eligibility policies.
- Trace inputs → filtering → estimation → uncertainty/reference → decision rule
  → saved results. Distinguish primary exclusions, undefined outcomes,
  model/feature eligibility, unclassified cases and criterion failures. Do not
  add overlapping exclusions as if disjoint. Exhaustive requests need complete
  case accounting, including explicit unavailable evidence.
- Distinguish original historical runs, later reapplications and proxies. When
  inputs, windows, transformations or estimands differ, the comparison does not
  isolate algorithmic improvement. More positives, narrower bands or optimizer
  success do not establish scientific accuracy. Same-data agreement is not
  independent validation; formulas and tests do not validate a model's claims.
- Make incomplete work visible. Do not invent results, redraw saved science or
  rerun models to fill a guide. Separately authorized analyses retain separate
  execution/status evidence. Synthetic teaching examples must be clearly labelled
  and isolated from real results, files and decisions.

Read [guide-design.md](references/guide-design.md) for content planning, statistical
comparisons and decision worksheets. It supplies evidence fields and scientific
distinctions to use when relevant, not a mandatory chapter template.

## Ownership and publication

Resolve canonical study sources through the [project storage policy](../../../docs/project-maintenance/review-artifacts.md#current-source-resolution) before choosing paths. Keep the
canonical editable source (normally Markdown, or an existing structured source)
with its HTML in the designated artifact store. Plans, durable instructions,
approved definitions, supported code/tests and authority records keep their
project homes. Use the existing JOAQUIM topic workspace.
Do not import another project's mount, counts or panel structure.

Use one canonical topic workspace and stable guide/section IDs. Preserve manual
edits, distinct historical discussions and saved notes. Read
[artifact-lifecycle.md](references/artifact-lifecycle.md) when setting up,
reorganizing, moving or revising a decision workspace; it owns discovery,
revision identity, exact recovery and note-safe route retirement.
An external guide may retain source snapshots without replacing their authority.
Carry explicit author choices into the existing scoped decision workflow with
evidence; notes, checkboxes and provisional preferences never publish approval.

## Reading experience

Prefer a self-contained offline HTML unless another format is requested. Use a
relative-asset bundle when required by policy/preference and state that dependency. Embed authentic
saved artwork with matching legends and source identities; use the project's
figure workflow for isolation/scaling. Core interpretation should not require
repository access. Keep provenance outside the main reading narrative.

Use plain-language explanations with progressive equations, assumptions, settings
and code references. Explain what each plot establishes and leaves unresolved;
keep important limits beside results. Choose interactions
that aid this reader: method/case selection, worked arithmetic or a clearly
synthetic demonstration. Avoid a large gallery when a selector serves comparison.

For substantial study guides, provide a suggested reading path, chapter
navigation/search, expandable technical/historical detail, responsive tables and
print styling. Include section-specific understanding/questions/provisional
choices, preserving existing notebook behavior.
Read [html-behavior.md](references/html-behavior.md) when building these controls
or changing notes. Its optional [review-notebook.js](assets/review-notebook.js)
provides the established notebook/search/import behavior, not a scientific
renderer or mandatory chapter layout. Reuse it or equivalent existing code.

## Verify and deliver

Authenticate consumed sources and final HTML using the project's existing
manifest and guards; do not introduce competing selection state. Optional
standalone check, using this skill's absolute path:

```sh
python3 "<skill-root>/scripts/check_guide.py" /absolute/path/guide.html \
  --manifest /absolute/path/guide-provenance.json
```

Use `--offline` only for the embedded single-file contract, not a legitimate
relative-asset bundle. The design reference describes the optional manifest;
the checker verifies static integrity/hashes, not science or browser behavior.

Open the exact saved artifact with permitted browser tools. Inspect meaningful
reading views, authentic figure/legend pairing, links/downloads, responsive
layout and print output as applicable. Exercise changed controls and preserve
notes. New or changed notebook behavior needs the fresh-context persistence,
export/import, invalid-input and storage-failure checks in `html-behavior.md`;
never test against the user's actual notebook. A content-only revision need not
repeat unaffected notebook-code tests, but must preserve IDs/semantics and check
any affected note compatibility. Record only verification actually performed;
report unavailable capabilities rather than bypassing restrictions.

Deliver the exact file, reading entry point and substantive gaps. Keep a compact
topic-to-section/evidence coverage map for requests spanning many iterations.
Creating a guide does not finish an analysis, select a winner, freeze artwork
or independently verify every scientific claim.
