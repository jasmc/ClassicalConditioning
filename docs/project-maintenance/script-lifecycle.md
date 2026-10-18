# Script lifecycle

Keep supported tools in `scripts/`, their shared implementation in `src/` and
tests in `tests/`. Keep exact frozen reproduction snapshots under their scoped
frozen records. Superseded candidate code belongs on verified external storage
after disposition; retention in `scripts/` does not authorize new scientific use.

The [tool catalogue](../../scripts/README.md) links every retained script. The
[17 October inventory](../../records/repository-history/2026-10-17-organization-reconciliation/script-inventory.json)
records hashes, purpose, current/historical reference candidates and retention
recommendations. It is dated evidence, not another mutable progress board.

| Recorded status | Meaning |
|---|---|
| `supported-or-dependency` | Current docs, tests, implementation or another retained tool references the file; preserve its dependency closure. |
| `catalogued-reproduction-or-audit` | Documented reproduction/audit entry with no additional active text consumer found. |
| `historical-candidate-retained` | Inspected dated recipe with temporary-ticket assumptions and/or retired styling; retain for provenance pending author disposition. |
| `retained-pending-classification` | No active reference was found; absence of a text match is not proof that removal is safe. |

`align_figure1_fgh_fonts.py`, `layout_figure2_compact_rows.py`,
`move_figure2_block_notes.py`, `readjust_figure2_g.py` and
`style_figure2_placeholders.py` are historical candidates. They retain dated
evidence dependencies and are not entry points for new Nature styling. Other
versioned names can still supply live helpers; do not classify by name or age.

Before moving any candidate, resolve imports, current state/evidence bindings,
frozen dependencies and exact recovery needs. Prepare a concrete scoped
disposition inventory. Execute archival or deletion only under the applicable
author authorization, using [verified storage procedures](storage.md). Preserve
pending originals and dependency closure until then. The organization
reconciliation itself archives and deletes no scripts.
