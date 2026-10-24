# Script command organization

Author request, 24 October 2026: audit every file in `scripts/`, assess stale
or redundant code, and propose coherent naming by figure/panel scope.

The [audit record](../records/repository-history/2026-10-24-scripts-audit/audit.json)
identifies the inspected Git revision, all-file inventory and report hashes.
The full per-file recommendations, source/consumer evidence and proposed paths
belong on JOAQUIM, alongside their Markdown/HTML reading report. Publication
status and any temporary review location are explicit in that record.
This plan proposes changes; it does not execute script moves or retire recipes.

## Findings and intended outcome

There are 111 tracked files: 93 Python, 11 CommonJS, five PowerShell, one browser
JavaScript and the README, totaling 18,952 lines. Forty-seven ignored bytecode
files are generated caches, not additional supported tools.

The recommendations account for every tracked file:

| Proposed treatment | Files |
|---|---:|
| Retain maintained command/verification capability | 32 |
| Extract shared implementation into package modules | 15 |
| Extract live helpers before retiring the dated recipe | 14 |
| Retain explicit historical reproduction/audit capability | 18 |
| Consolidate platform launchers into protocol stages | 5 |
| Completed-recipe archival candidates, subject to verified disposition | 26 |
| Retain compact navigation | 1 |

These are roles, not proof of runtime usage or authorization to delete files.
A catalogue mention or historical evidence reference does not prove current
operation; absence of an importer does not prove that a CLI is unnecessary.
Keep scientific definitions and selected/frozen evidence separate from code
organization. Exact frozen snapshots are never updated to a refactored version.

## Priority 0 — clarify and repair operational boundaries

- Repair the producer/consumer path mismatch between
  `plot_train_test_transition_metrics.py` and `plot_metric_response_magnitude.py`.
  The former sets repository-local `outputs/` and creates media there; the latter
  reads through `external_output`. Use one explicit authenticated external
  input/output contract, preserve previous results, and reject unavailable storage.
- Split modifying and verifying interfaces. `check_nature_presentation.cjs`
  rewrites candidate SVGs by default; `align_paper_annotations.cjs` also edits by
  default, and `style_axis_geometry.cjs --center-labels` changes artwork.
  Reserve verify/check routes for validation and evidence collection, make
  mutation explicit, and retain guarded publication outside those operations.
- Correct misleading entry-point documentation without changing scientific
  definitions. `freeze_figure1_panels.py` verifies an existing A–D freeze;
  `prepare_figure2_B_freeze.py` prepares outcome version B for D/E, not panel B;
  `build_figure2_heatmap_row.py` has an F/G/H docstring but handles the Figure 2
  six-heatmap composition with A-only freeze scope.
- Stop describing `update_latest_figure_assemblies.py` as the generic current
  viewer: its Figure 2 path can resolve current state, but its Figure 1 assembly
  uses older E/V12 selection logic and its default output is dated 10 October.
  Keep imported helpers until extraction; current display follows scoped state
  and the supported main-paper collection.

Acceptance: synthetic tests cover external output agreement and missing-storage
failures; modifying operations cannot be invoked accidentally as read-only
checks; active navigation names the actual scope. No scientific model reruns.

## Priority 1 — extract libraries and break circular imports

Use existing `figure_workspace`, `figure_freeze`, `external_artifacts`, `theme`
and `layout_frames` implementation. Add focused modules where necessary;
do not create another state authority, freeze workflow or plotting theme.

Three direct/lazy import cycles need removal:

1. `build_figure_legend_review` ↔ `publish_figure1_fgh_one_second`.
2. `layout_main_paper_figures` ↔ `correct_panel_relationships`.
3. `review_figure2_block_log_median` ↔ `review_figure2_bout_only`.

The legend-review module has 954 lines and 21 distinct direct Python importers.
Extract generic HTML/font/scoped-SVG helpers first. Move pure Figure 1 layout
helpers out of `scripts/figure1_layout_spacing.py`. Extract `SOURCES` and
`bundle_markdown` from the completed G supplement driver before retiring it:
the supported G guide builder imports them today. Replace mutable `OUT` globals
and cross-script callbacks with explicit parameters.

Share browser DOM/CTM/address/font-readiness utilities and launcher options.
Keep geometric bootstrap, painted/unclipped layout evidence, scientific-role
verification and mutation as distinct contracts. Preserve guide storage keys,
source identities and downloaded payloads when sharing notebook/download helpers.

The conservative AST comparison found no large identical function-body groups;
sharing is architectural, not a basis to combine different scientific analyses.
The D/E log-median script exactly matches a frozen snapshot but has live
importers. Preserve its frozen original and each distinct outcome/formula/window.

Acceptance: package modules import without script execution or mounted SSD;
cycles disappear; synthetic regression tests preserve exact calculation inputs,
source bindings, scoped selection and publication guards. No broad golden-byte
rewrite of historical scientific records.

## Priority 2 — migrate thin entry points to coherent scoped names

Proposed structure; create folders only for real retained operations:

```text
scripts/
  README.md
  maintenance/
  reviews/
  workflows/
    windows/
    macos/
    trace/
    metrics/
  paper/
    shared/
    figure-01/
      panel-a/
      panel-b/
      panel-c/
      panel-e/
      panels-d-e/
      panels-f-g-h/
    figure-02/
      assembly/
      panel-a/
      panel-b/
      panel-g/
      panels-d-e/
      panels-d-g/
      panels-e-h/
    figure-03/
      supplement/
```

Within a scope use actions such as `build.py`, `prepare_freeze.py`,
`publish.py`, `verify_*.py`, `audit_*.py`, `render_*.py` and
`build_study_guide.py`. Use `run_` for computations/orchestration. Keep method and
historical version names where they distinguish scientific reproduction.
Shared importable implementation belongs under `src/classical_conditioning/`,
using Python module names with underscores; commands can mirror the existing
figure/panel directory slugs. A single monolithic script per figure is unnecessary.

Record source-panel identity and display-panel identity separately. Figure 1
F/G/H changed display order; an old C/D script can represent today's D/E trace
roles. Do not mechanically rename panel letters from filenames. Figure 4 has
package/CLI analysis and rendering already; create no empty wrapper for symmetry.
Likewise add H/I wrappers only when there is a concrete supported operation.

Before nesting commands, make the catalogue checker recursive and update
repository-root resolution, Python sibling imports, JS `__dirname`, PowerShell
`$PSScriptRoot`, tests and path/hash consumers. Preserve old CLI paths through
small aliases while callers migrate. Exact historical references resolve old
bytes through verified recovery metadata, never an alias to changed code.

Acceptance: the per-file path map has no collisions; every command is catalogued
recursively; supported callers work from a relocated clean Git checkout; the old
interface remains compatible or has an explicitly recorded retirement.

## Priority 3 — consolidate workflows and archive completed recipes

Consolidate three Delay launchers into one protocol launcher with explicit stages
and the two Trace launchers into their own stages. Preserve manifest/hash checks,
preflight recording-set equality, metric identity, distinct cohort policies and
historical acceptance behavior; common platform plumbing does not merge analyses.

The audit lists 26 completed recipes as archival candidates. The completed
single-fish archive driver has the strongest prior concrete disposition proposal.
A further 14 dated recipes contain helpers to extract before any retirement.
These counts do not authorize a bulk move: preserve each original hash,
version-bound evidence, dependency closure and selected/frozen references; prepare
exact per-file disposition packets and use native verified archive procedures.
Do not delete a source merely because another frozen file has identical bytes.

Generated bytecode is unnecessary for source retention. It can be cleared during
an authorized cleanup and rebuilt by Python; orphan cache files are not supported
commands. This audit preserves them and records their role.

Acceptance: reusable code remains local/tested; completed recipe packets and
historical aliases verify before pruning; immutable requirements and current
states remain unchanged; no pending disposition is silently treated as approved.

## Current status

Audit and per-file recommendations complete; implementation remains proposed.
No script code, source filename, scientific output or selected state was changed
by this audit. SSD publication is tracked in the linked audit record. The prior
study-source migration is committed at `23c99b6`; its 62 relevant tests and the
structure checker pass from an exported committed tree. The independent four
external Figure 1/2 working-copy mismatches and hosted CI billing issue remain.
