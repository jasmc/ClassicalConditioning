# Repository instructions

Author directions recorded on 2026-10-09 through 2026-10-12 remain standing
instructions. Detailed procedures are in the [figure workflow](docs/analysis/figures/reviews/figure-iteration-workflow.md).

## Authority and scope

- Preserve other work and uncommitted changes. Continue the requested panels;
  do not restart cleanup, change unrelated figures, or recompute scientific
  definitions as a presentation operation.
- Scientific authority is the current scoped accepted freeze, resolved through
  [Current figures](docs/current-figures.md) and selection records. An older full
  assembly never overrides a newer panel selection. Frozen originals, hashes,
  source data and reproducibility dependencies remain immutable.
- Presentation authority is the scoped `current-state.json` linked by the
  workflow. It separately identifies the frozen display and active draft.
  Other review/audit files are version-bound evidence, not competing pointers.
- Read those records when resuming; do not choose by filename age or assume a
  chat preview is current. Check external storage before loading/regenerating.

## Display and iteration

- "Show Figure 1" defaults to an authenticated editable SVG working copy of
  the current frozen selections. Reuse its recorded copy; open/show the SVG
  itself. An HTML link or raster preview alone is insufficient. If only panels
  are frozen, label the result an assembly of frozen panels, not a whole-figure
  freeze. Identify unfrozen scopes explicitly. A draft/comparison requires an
  explicit request; preserve the active draft when showing the frozen selection.
- Keep SVG vector groups, text, paths and source mappings addressable. Preserve
  authentic embedded raster data. An SVG source editor supports XML edits;
  do not claim graphical drag-and-drop editing unless the chosen app supports it.
- Implement repeatable edits in supported code. Save one self-contained HTML
  review with requested versions, legends and provenance. The author's editable
  SVG preference permits one recorded working SVG per frozen/draft state;
  avoid redundant copies or extra media formats. Use in-memory previews where
  possible and remove task-created temporary render files after verification.
- Measure effective fonts, strokes, tick spacing and coordinate mappings after
  all transforms at final assembly scale. Record unresolved measurements and
  real visual/renderer evidence; nominal settings alone are not verification.
- Before work, snapshot current state, working-copy and bound input hashes.
  Recheck under the figure-state publication lock before saving. On conflicts,
  preserve both editors' work and reconcile; never silently overwrite. Use
  `figure_working_copy.py` and the guarded review generator described in the
  workflow. Publish verified artifacts/evidence before the atomic state pointer.

## Scientific relationships, alignment and legends

- Read the semantic registry as an explicit map from stable element IDs to
  scientific roles, source/data identity, units, transformations and groups.
  Structural XML IDs alone are not scientific classifications. Report partial
  coverage honestly; historical resolved styles do not verify current styles.
- For related panels, record and verify shared fish/cohort, trial/frame IDs,
  time origin, event definitions and raw-to-scaled processing relationships.
  Match data-axis widths/heights, row centres/order, block/trial labels and time
  limits where requested; exclude colourbars from body-size comparisons.
  Use time relative to CS onset when applicable. Check alignment at final scale.
  Treat a change in source data, masks or timing as a scientific draft change,
  record it explicitly, and preserve accepted freezes.
- Create a companion legend with each new panel/figure. During edits, update it
  only when scientific definitions or necessary explanations change. Move
  method details, fish identifiers and explanatory clutter to the legend while
  keeping essential axis labels, units, ticks and scientific marks visible.

## Freeze and previous versions

- Only explicit freeze requests invoke the [element specification](docs/analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md)
  and [versioned roles/styles](configs/paper-figures/figure-elements.json).
  Build a separate candidate with exact sources, scientific element mappings,
  evidence/protection, resolved styles and scoped approved exceptions.
- Run `.venv/bin/python scripts/freeze_figure.py --candidate <candidate.json>
  --check-only`, resolve every issue, then use the same command with
  `--output <new-freeze.json>` for every new panel/figure freeze. The installed
  `classical-conditioning freeze-figure` command is equivalent. The gate checks;
  it does not infer roles, restyle, invent evidence, or freeze normal exports.
- Preserve approved choices within their original scope. Complete independent
  candidate work before asking about unresolved scientific choices/exceptions.
  A freeze request supplies authorization; do not ask redundantly. Record each
  exception's ID, role, property/value, reason, scope and approval evidence.
  Appearance alone is not approval. Noninteractive unresolved cases must fail.
- After a successful new freeze, prepare an inventory of prior versions within
  that scope, with dependencies and recommendations to keep, archive or delete.
  The author reviews and chooses the disposition. A freeze request alone does
  not authorize deleting or archiving earlier versions. Preserve everything
  pending those choices; execute only approved dispositions, verify external
  copies before pruning, preserve frozen bytes and update indexes/navigation.
  A check-only run or rule change does not trigger historical cleanup.

## Storage

- `Plans/` contains numbered future-work plans and its README/index. Governance,
  decisions, reviews, audits and handoff records live under `docs/`. Supported
  code/tests/tools remain local; exact frozen records belong under
  `records/frozen-analyses/figure-NN/panel-x/YYYY-MM-DD-description/` or joint
  `panels-x-y-z/` scopes, using role subfolders and existing naming conventions.
- Permanent media, HTML reviews, numerical data and superseded candidate code
  belong on JOAQUIM, with explicit external destinations. Use
  `scripts/repository_archive.py`, archive indexes and relocation resolution.
  Preserve original frozen bytes/path aliases; never delete a pending original
  before its external copy is verified. See [record layout](records/frozen-analyses/README.md).
- If storage is unavailable, keep originals, record pending work and continue
  independent code/doc work. Do not fabricate a mount, substitute an older
  figure, or claim to inspect an inaccessible artifact.
