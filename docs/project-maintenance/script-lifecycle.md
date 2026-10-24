# Script lifecycle

Keep supported tools in `scripts/`, their shared implementation in `src/` and
tests in `tests/`. Keep exact frozen reproduction snapshots under their scoped
frozen records. Superseded candidate code belongs on verified external storage
after disposition; retention in `scripts/` does not authorize new scientific use.

The [tool catalogue](../../scripts/README.md) links every retained script. The
[17 October inventory](../../records/repository-history/2026-10-17-organization-reconciliation/script-inventory.json)
is the initial dated inventory. The
[23 October assessment](../../records/repository-history/2026-10-23-script-lifecycle/assessment.json)
completes its 14 pending classifications and records decisions for the five
historical candidates plus the later download-repair helper. Both are dated
evidence, not progress boards. The initial status names below describe the
17 October inventory; they do not leave those 14 files unclassified today.

| Recorded status | Meaning |
|---|---|
| `supported-or-dependency` | Current docs, tests, implementation or another retained tool references the file; preserve its dependency closure. |
| `catalogued-reproduction-or-audit` | Documented reproduction/audit entry with no additional active text consumer found. |
| `historical-candidate-retained` | Inspected dated recipe with temporary-ticket assumptions and/or retired styling; retain for provenance pending author disposition. |
| `retained-pending-classification` | No active reference was found; absence of a text match is not proof that removal is safe. |

`align_figure1_fgh_fonts.py`, `layout_figure2_compact_rows.py`,
`move_figure2_block_notes.py`, `readjust_figure2_g.py` and
`style_figure2_placeholders.py` remain historical presentation recipes. The
[disposition recommendations](../../records/repository-history/2026-10-23-script-lifecycle/disposition-proposals.json)
recommend keeping their exact local source because it explains retained
version-bound evidence and compatibility behavior. They are not entry points
for new Nature styling. Other
versioned names can still supply live helpers; do not classify by name or age.

Before moving any candidate, resolve imports, current state/evidence bindings,
frozen dependencies and exact recovery needs. Prepare a concrete scoped
disposition inventory. Execute archival or deletion only under the applicable
author authorization, using [verified storage procedures](storage.md). Preserve
pending originals and dependency closure until then. The organization
reconciliation itself archives and deletes no scripts.

## Completed classification and proposed disposition

`build_figure3_explanation.py` is a source-bound guide builder;
`verify_figure2_g_analysis_guide.cjs` and `verify_figure3_reference_style.py`
provide scoped verification contracts. Their parameterized interfaces and
selected evidence roles warrant retention even without Python importers.

The remaining assessed files are identified as historical comparison, freeze
preparation/publication, scientific reproduction, presentation or guide-repair
recipes. The assessment records each file's reason, hash, source dependencies
and reference candidates. No scientific recipe was executed for classification.

One completed transaction driver, `archive_figure1_singlefish_versions.py`, has
a concrete [archival proposal](../../records/repository-history/2026-10-23-script-lifecycle/proposed-archive.json).
It is specific to the completed 13 October transfer, has no discovered Python
importers or non-inventory record references, and reusable transfer operations
remain in `repository_archive.py`. The proposal is not authorization: keep the
local source until the author chooses that disposition and recovery is verified.

## Whole-folder reassessment

The [24 October audit](../../records/repository-history/2026-10-24-scripts-audit/README.md)
extends the earlier 20-file follow-up to every current tracked script and ignored
cache. It distinguishes maintained actions, shared implementation, historical
reproduction and completed recipes with live helpers. It also identifies storage
and misleading command-name issues. The earlier retention assessments preserve
source/provenance needs; they do not establish that every completed recipe must
remain a permanent top-level command.

[Plan 13](../../Plans/13_SCRIPT_COMMAND_ORGANIZATION.md) owns the proposed repairs,
extractions and per-figure/panel command organization. Those changes remain
unimplemented. Preserve exact scientific definitions and historical bindings;
extract shared code and verify consumers before renaming or retiring a recipe.
