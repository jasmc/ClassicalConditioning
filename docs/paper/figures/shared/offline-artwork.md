# Offline figure artwork on the Mac

Author direction, 2026-10-15: retain local copies of artwork for all panels,
including unfrozen panels, for layout and aesthetic work without JOAQUIM.
This is a scoped exception to SSD-only media storage, outside the repository.

Local folder: `/Users/joaquim/Documents/ClassicalConditioning Figures/`.
Start with `ALL-PANELS.md` and `workspace.json`. The latter links the original
frozen-source `inventory.json` and `snapshots/inventory.json`, each bound by
SHA-256. The [initial frozen copy record](../../../../records/figure-reviews/offline-frozen-artwork-2026-10-15.json)
and [all-panel copy record](../../../../records/figure-reviews/offline-all-panel-artwork-2026-10-15.json)
record the starting selections/state and copied bytes. The package uses about
73 MiB, including fonts, records and both complete SVG/HTML snapshots.

| Local source folder | Selected frozen scopes |
|---|---|
| `frozen/figure-01/` | A, B, C, D, E and the F/G/H row |
| `frozen/figure-02/` | A, D, E and G |

The complete selected presentation snapshots are in `snapshots/figure-01/`
(A–H) and `snapshots/figure-02/` (A–I), each with its exact matching HTML,
legend and source-state snapshot. Figure 2 B/H contain unfrozen artwork; C/F/I
retain the current Pending placeholders. Combined A/B/C and F/G/H SVG groups
are preserved without creating extra standalone panel versions.

Neither whole figure is frozen. Sources were resolved from scoped selections
and presentation state on 2026-10-15; older assemblies do not replace a newer
panel selection. Check saved hashes against current repository state before
reuse; the cache does not update automatically.

## Working from the artwork

The local SVG/HTML files are exact, read-only source mirrors. Their rendering
resources are embedded or internal; DejaVu Sans font files and licence are included under
`fonts/`. Layout/style edits can use the SVGs directly without loading analysis
data. Save requested derivatives under the local `drafts/` folder and preserve
the originals. A derivative is an unfrozen draft. Numerical recalculation and
new scientific freezes may still require original dependencies on JOAQUIM.

This source cache is not an active working version and does not change either
figure's `current-state.json`. When an author requests an offline draft edit,
resolve the intended scope, capture starting source/state hashes, preserve
manual edits and use supported guarded publication. Publish one matching
SVG/HTML pair and its evidence before making it active in scoped state. Do not
regenerate numerical panels merely to change layout. Apply the workflow's
invisible-frame policy only within an authorized edit scope.

`records/selections/`, `records/freezes/`, source-state and legend files are exact dated
snapshots for reference. Paths inside frozen records remain unchanged; they
do not promise that scientific data dependencies have been mirrored locally.
On reconnection, reconcile any active local draft explicitly before updating
an SSD copy. A cached source or old browser page never selects current work.

To check the copies offline, authenticate both inventory hashes in
`workspace.json`, then hash each file listed in those inventories and compare
with its `sha256`. This needs only the local folder. A changed or
missing source must be preserved/reconciled, never silently refreshed.
