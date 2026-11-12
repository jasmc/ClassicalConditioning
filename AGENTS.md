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

Use the [paper-figures skill](.agents/skills/paper-figures/SKILL.md), reading it
directly if discovery is unavailable. The [single figure guide](docs/paper/figures/shared/iteration-workflow.md#task-routes)
owns procedures. The [unified contract](configs/paper-figures/figure-elements.json)
owns semantic roles, current Nature styles, rule origins and frame policy.

- Use the smallest task route. Display shows authenticated selected SVG/legend
  only and verifies the actual rendering; it does not restyle. Expanded/frozen/
  comparison views require an explicit request. Plural paper display follows
  recorded collection membership and identifies unfrozen scopes.
- Disposable previews use saved inputs, a matching legend and provisional status.
  Retain requested versions only; no state bootstrap or journal/archive-wide audit.
- New presentation uses one `nature-main` contract snapshot through the shared
  theme inside scoped Matplotlib rc changes. Preserve scientific mappings and
  author geometry; profile lookup alone certifies no output. Future layouts use
  the selected invisible frames/insets; existing drafts need a scoped edit request.
- Preserve manual/uncommitted work. Capture starting state/artifact/input hashes,
  recheck under publication locks, and publish verified artifacts/evidence before
  the atomic state pointer. Reconcile conflicts rather than overwrite them.
- Named consistency work includes scoped registry/evidence updates and explicit
  source/data mappings. Report partial coverage; appearance alone proves no identity.
- Only an explicit freeze request activates the full panel/assembly/letter gate.
  Preserve scoped approvals and pending originals; ask only about unresolved
  choices. Exports do not freeze. Inventory prior versions after success for
  author disposition; freezes, checks and rule edits authorize no other cleanup.

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
