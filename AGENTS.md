# Repository instructions

Author directions recorded on 2026-10-09 through 2026-10-13 remain standing
instructions. For paper figure/panel display, creation, editing, comparison,
verification or freezing, read and use the repository
[paper-figures skill](.agents/skills/paper-figures/SKILL.md). Its procedures live
in the authoritative [figure workflow](docs/paper/figures/shared/iteration-workflow.md).
If skill discovery is unavailable, read that skill file directly.

## Essential protections

- Preserve other work and uncommitted changes. Stay within requested panels;
  do not restart cleanup or recompute scientific definitions as presentation.
- Scientific authority is the current scoped accepted freeze, resolved through
  [Current figures](docs/paper/figures/README.md) and selection records. A newer
  panel selection supersedes an older assembly only within its scope. Frozen
  originals, hashes, source data and reproducibility dependencies are immutable.
- Presentation authority is the scoped `current-state.json` linked by the
  workflow. Keep frozen display and active draft separate. Read saved authority
  before resuming; filename age, previews and historical audits do not select
  current work. Check external storage before loading or regenerating artifacts.
- A figure display follows its recorded `default_display`. Author selection
  on 2026-10-13 makes v10 the sole Figure 1 working SVG and the default draft
  display. Show frozen selections or historical comparisons on explicit request. A panel assembly is not a whole-figure freeze. Identify unfrozen
  scopes, and show the SVG itself rather than only HTML or a raster preview.
- Preserve manual working-copy edits. Snapshot state, working-copy and bound
  input hashes before edits; recheck under the publication lock. Preserve and
  reconcile conflicts. Publish verified artifacts/evidence before the atomic
  state pointer, using supported guarded tools.
- Scientific roles require explicit source/data mappings. Changes to data,
  masks, timing or transformations are scientific draft changes. Nominal styles
  and structural XML IDs do not prove scientific or final-scale consistency.
- Only an explicit freeze request activates the
  [element specification](docs/analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md)
  and [role/style contract](configs/paper-figures/figure-elements.json).
  Prepare a separate candidate, preserve scoped approvals, and pass
  `scripts/freeze_figure.py --check-only` before publishing with `--output`.
  A freeze request supplies authorization; unresolved decisions must fail,
  never acquire invented approval. Ordinary exports do not freeze anything.
- After a successful new freeze, inventory previous versions and dependencies
  for author disposition. Freezing does not authorize archiving or deleting
  them. Preserve all pending versions; execute only approved dispositions and
  verify external copies before pruning. Rule edits/check-only runs do not
  trigger historical cleanup.

## Storage and ownership

Follow [folder policy](docs/project-maintenance/structure.md) and the
[frozen-record layout](records/frozen-analyses/README.md). `Plans/` holds numbered
future work; current guidance belongs under `docs/paper/`,
`docs/running-analysis/` and `docs/project-maintenance/`; evidence belongs under
`records/`. Paper work is figure-led, with shared methods stored once. Supported
code/tests/tools stay local; repository agent skills live under `.agents/skills/`.

Permanent media, HTML reviews, numerical payloads and superseded candidate code
belong on JOAQUIM at explicit destinations. Use `scripts/repository_archive.py`,
archive indexes and relocation resolution; preserve frozen bytes/path aliases.
Superseded freezes retain their figure/panel scope, dated role folders and
`history/`. If storage is unavailable, preserve originals, record pending work
and continue independent code/doc work. Never fabricate a mount, substitute
older artifacts or claim inspection of inaccessible files.

Author request on 2026-10-13: keep `to_implement.md` at the repository root as
working request notes. Its historical copy remains in archive records; it does
not replace numbered plans, the implementation index or scientific authority.
