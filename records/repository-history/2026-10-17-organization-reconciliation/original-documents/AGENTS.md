# Repository instructions

Author directions recorded on 2026-10-09 through 2026-10-13 remain standing
instructions. For paper figure/panel display, creation, editing, comparison,
verification or freezing, read and use the repository
[paper-figures skill](.agents/skills/paper-figures/SKILL.md). Its procedures live
in the authoritative [figure workflow](docs/paper/figures/shared/iteration-workflow.md).
If skill discovery is unavailable, read that skill file directly.

For new paper-figure presentation work, Nature is the selected target; read the
[maintained Nature guidance](docs/paper/figures/shared/nature-style.md).
Use the [shared plotting module](src/classical_conditioning/figures/theme.py)
with one resolved profile per build; pass the same theme to its plotting helpers
and keep Matplotlib rc changes scoped. Explicit selected-draft migration uses the
workflow’s scoped Nature route; remaining renderer/production checks stay open:
retain compatible author rules, validate actual output, and do not infer
compliance from profile selection or restyle artwork just to display it.

Author direction 2026-10-17: retire pre-Nature styling as a choice for new paper
work. Legacy theme defaults and historical contracts are compatibility and
reproduction dependencies only; do not copy their font/size values into new
work or use an unprofiled call as the paper styling entry point. Resolve current
role values through `nature-main`. Preserve compatible author geometry and
scientific mappings. See the [retirement record](records/repository-archive/2026-10-17-retired-styling/README.md)
for verified archived guidance and dependencies retained in place.

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
- Opening a specific figure/panel defaults to self-contained HTML containing
  only that scope’s selected working SVG and associated legend, in the built-in
  browser for visual review and annotation. Notes, provenance displays, other
  figures/panels, comparisons and versions require an explicit request. Follow scoped
  state for the selected version; keep its matching editable SVG as the figure
  artifact and open it on explicit request. Resolve the recorded HTML path;
  stale localhost copies and browser tabs are not current authority. Verify
  the rendered scope before reporting a successful open. Resolve Figure 1's sole
  working version from scoped state, rather than a hard-coded version name.
  Frozen selections/comparisons require an explicit request;
  identify unfrozen scopes and never infer a whole-figure freeze.
- Plural requests to show the main-paper figures use the authenticated ordered
  collection (`scripts/main_paper_figures.py build`, then `display-target`).
  A particular figure/panel request still shows only its own SVG and legend.
  Collection inclusion does not create a scientific freeze.
- Preserve manual working-copy edits. Snapshot state, working-copy and bound
  input hashes before edits; recheck under the publication lock. Preserve and
  reconcile conflicts. Publish verified artifacts/evidence before the atomic
  state pointer, using supported guarded tools.
- “Ensure consistency” requests also update the scoped draft registry and
  version-bound evidence for the named relationships/properties; follow the
  workflow. Preserve scientific meaning and record unresolved coverage.
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

Author exception, 2026-10-15: all Figure 1/2 panel artwork, including unfrozen
panels and existing placeholders, is also available on this Mac at
`/Users/joaquim/Documents/ClassicalConditioning Figures/`. For offline layout
or aesthetic work, read [the local artwork guide](docs/paper/figures/shared/offline-artwork.md)
and the folder's `ALL-PANELS.md`/`workspace.json` before requiring JOAQUIM.
These are verified source snapshots, not automatically synchronized active
drafts. Compare their recorded state/selection hashes with repository authority,
preserve originals and publish requested derivatives through scoped guards.

Author request on 2026-10-13: keep `to_implement.md` at the repository root as
working request notes. Its historical copy remains in archive records; it does
not replace numbered plans, the implementation index or scientific authority.

Author direction 2026-10-15: future paper, supplementary and analysis figures
use invisible figure/panel layout frames with one shared physical inset,
shared outside margins and two panel insets between adjacent contents. Follow
the invisible-frame section of the figure workflow for calibration, ownership,
exact-render evidence and new-freeze checks. The author selected the shared inset on 2026-10-15; use the active policy
value and do not retrofit current drafts without a scoped edit request.
