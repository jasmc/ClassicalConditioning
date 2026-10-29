# Supported source

Put reusable implementation in the package and entry-point orchestration in
scripts. Package imports must not execute a run, publish state or change global
plotting style. Apply presentation settings inside a scoped render operation;
keep scientific transformations separate from presentation edits. Declare
packaged resources and explicit checkout-dependent execution requirements.
Consolidate shared plumbing only after checking public behavior, receipts and
source bindings. Similar scientific code is not sufficient evidence for a merge.

The package separates ingestion/preprocessing, scientific analysis, operational
helpers, figure rendering and review resources. Frozen implementations remain
under records; installed `*.egg-info` is generated metadata. Repository-relative
orchestration and legacy comparison need the checkout, not just an installed wheel.

Historical rendering uses its recorded settings inside a scoped context. Import
isolation does not authorize migrating saved artwork to another profile. New
presentation follows the centrally owned figure workflow and shared theme.

See [folder ownership](../docs/project-maintenance/structure.md),
[script lifecycle](../docs/project-maintenance/script-lifecycle.md) and
[tests](../tests/README.md).

The public package currently initializes the historical intake batch backend.
The atlas regression measures its own import after that existing initialization;
whole-package backend isolation is not claimed. Its render-only style context
retains the exact historical atlas settings instead of applying a new profile.
