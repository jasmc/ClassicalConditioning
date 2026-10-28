# Repository instructions

Standing author directions remain in force. Use the [rules and authority map](docs/project-maintenance/rules.md)
to find their maintained owners. Read task-relevant procedures; do not load full
plans, historical audits or generated catalogues for routine work.

## Scientific purpose and authority

The first paper release covers Figures 1–4 and the evidence needed for their
claims. [Paper figures](docs/paper/figures/README.md) owns their purposes;
[governance](docs/project-maintenance/governance.md), [decisions](docs/paper/decisions.md)
and [shared methods](docs/paper/methods/README.md) govern scientific work.
Resolve relevant definitions and open decisions before changing analyses.
Do not expand the paper scope or treat an implemented method as approved.

- Scientific authority is the accepted freeze for the requested scope, resolved
  through the figure index and exact selection records. A newer panel selection
  supersedes an older assembly only within that scope. Frozen originals, hashes,
  source data and reproduction dependencies are immutable.
- Presentation authority is scoped `current-state.json`. Read it before resuming;
  filenames, timestamps, previews and historical audits do not select work.
  Scientific selection, presentation selection and storage location are separate.
- Preserve units, timing, missingness, eligibility, aggregation and estimands.
  Changing data, masks, transformations or learner definitions is scientific
  work, even inside a plotting command. Do not recompute science as presentation.

## Figure work

For figure/panel display, creation, editing, comparison, legends, verification
or freezing, use the [paper-figures skill](.agents/skills/paper-figures/SKILL.md),
reading it directly if discovery is unavailable. The [iteration workflow](docs/paper/figures/shared/iteration-workflow.md)
is the procedural authority; its task routes identify the sections to read.

- New presentation work uses [Nature guidance](docs/paper/figures/shared/nature-style.md)
  and resolved `nature-main` values. Use the [shared theme](src/classical_conditioning/figures/theme.py)
  with one profile snapshot per build, passed to helpers inside scoped Matplotlib
  rc changes. Pre-Nature defaults/contracts are reproduction dependencies only.
  Preserve compatible author geometry and scientific mappings. Profile selection
  does not certify output; showing artwork does not request restyling.
- A particular figure/panel opens its state-selected working SVG and matching
  scoped legend only, in minimal self-contained HTML in the built-in browser.
  Other scopes, notes, provenance displays, versions, comparisons and frozen
  views require an explicit request. Keep provenance nonvisual and the matching
  editable SVG available on request. Resolve the recorded path and verify the
  rendered scope before claiming a successful open; stale tabs are not authority.
- Plural main-paper display uses `scripts/main_paper_figures.py build`, then
  `display-target`. The ordered collection follows its recorded membership;
  inclusion does not freeze a figure. Identify unfrozen scopes.
- Preserve other work, uncommitted changes and manual SVG edits. Stay within
  requested scopes. Snapshot state, working-copy and bound input hashes before
  edits; recheck under the publication lock. Preserve/reconcile conflicts and
  publish verified artifacts/evidence before the atomic state pointer, using
  supported guarded tools.
- “Ensure consistency” also updates the scoped draft registry and version-bound
  evidence for the named relationships/properties. Scientific roles need explicit
  source/data mappings; XML IDs and matching appearance do not prove scientific
  identity or final-scale consistency. Record unresolved coverage.
- Only an explicit freeze request activates the [element specification](docs/paper/figures/shared/figure-element-specification.md)
  and [role contract](configs/paper-figures/figure-elements.json). Prepare a separate
  candidate and pass `scripts/freeze_figure.py --check-only` before `--output`.
  That request supplies authorization; preserve scoped approvals and never invent
  unresolved decisions. Exports do not freeze. After a successful new freeze,
  inventory earlier versions/dependencies for author disposition; keep pending
  originals. Freezing, rule edits and check-only runs do not authorize cleanup.
- Future figure layout uses invisible frames, the active shared physical inset,
  shared outside margins and two panel insets between adjacent contents. Follow
  the workflow's frame/ownership/render checks. Do not retrofit existing drafts
  without a scoped edit request.

## Storage and ownership

Follow [folder policy](docs/project-maintenance/structure.md), [storage/lifecycle rules](docs/project-maintenance/storage.md#figure-led-storage-and-lifecycle)
and the [frozen-record layout](records/frozen-analyses/README.md). Work is figure-led;
shared methods/material have one home. Computer origin is provenance. Keep one
working home per scoped state and every retained version reachable from SSD
`START-HERE.html` through the generated figure browser. Generate links with
supported tools and inspect bounded reports. No automatic deletion or release
of manual-review holds follows from these rules.

- `Plans/` holds numbered future work; its implementation index alone tracks
  progress. Current guidance belongs in `docs/paper/`, `docs/running-analysis/`
  and `docs/project-maintenance/`; evidence in `records/`; supported code/tests
  stay local. Repository skills belong in `.agents/skills/`.
- Study/discussion/comparison manuscripts and conversation checkpoints live with
  their HTML on JOAQUIM: follow [study-artifact policy](docs/project-maintenance/review-artifacts.md).
  Keep plans, durable instructions, operational docs, approved definitions,
  scoped authority and compact recovery metadata in their owning homes. Resolve
  migrated sources through [study-sources.json](docs/paper/study-sources.json) and
  the supported artifact resolver; a navigation pointer is not the manuscript.
  Use personal `analysis-review-guide` for substantial study artifacts when
  available; do not vendor it into this repository.
- Permanent media, reviews, numerical payloads and superseded candidate code
  belong at explicit JOAQUIM destinations. Use supported archive/migration tools
  and exact relocation resolution; preserve frozen bytes, aliases, saved notes
  and dependencies. Superseded freezes retain scope, dated role folders and
  history. Verify external copies before any authorized pruning.
- Check external storage before loading/regenerating artifacts. If unavailable,
  retain originals, record pending work and continue independent code/doc work.
  Never fabricate a mount, substitute older artifacts or claim inaccessible-file
  inspection. The former Mac `ClassicalConditioning Figures` cache was consolidated
  into SSD figure versions at the author's request on 28 October 2026; see
  [artwork recovery guidance](docs/paper/figures/shared/offline-artwork.md).
  Dated snapshots are not synchronized active drafts. Compare recorded authority
  hashes, preserve originals and publish derivatives through scoped guards.
- Keep root `to_implement.md` as author request notes. It replaces neither
  numbered plans nor scientific authority.
