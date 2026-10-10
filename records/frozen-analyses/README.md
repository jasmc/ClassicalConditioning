# Frozen records by figure and panel

| Figure | Records |
|---|---|
| [Figure 1](figure-01/README.md) | Panel E; joint F/G/H V12 and accepted historical V1 records |
| [Figure 2](figure-02/README.md) | Panels D/E version B, shared D/E method/code, panel G Historical LogMedian |
| Figures 3–4 | No frozen record in this collection; scientific gates remain open |

## Layout and names

Use `figure-NN/panel-x/YYYY-MM-DD-description/` for individual panels and `figure-NN/panels-x-y-z/YYYY-MM-DD-description/` for shared records. Date labels identify the dated record collection; actual approval timestamps remain in the original manifests. Version numbers use two digits (`version-01`, `version-12`); named revisions use lowercase labels (`version-b`). Joint panel records stay joint rather than being duplicated or presented as independent freezes.

Within each version, use `manifests/`, `settings/`, `code/`, `sources/`, `reviews/` and `notes/` as needed. JSON/Markdown names use lowercase hyphens. Python/JavaScript source names use lowercase underscores to preserve module-name conventions. `freeze.json`, `candidate.json`, `scientific-selection.json` and `source-figure.json` identify their role within an already scoped version folder. Navigation READMEs are separate from immutable original notes.

## Preservation and authority

All 70 relocated frozen files retain their exact bytes and SHA-256. Their embedded filenames, imports, historical paths and freeze IDs are unchanged; directory names are navigation, not new scientific identities. Historical snapshots are not assumed executable from their relocated folders. Reconstruct original paths when needed using the [relocation map](../../docs/maintenance/archive/relocations.json); use supported tools for current execution.

[Machine-readable index](index.json) records every current path, prior path and hash. [Move record](../../docs/maintenance/archive/records-reorganization.json) records this reorganization. [Scoped freeze authority](../../docs/analysis/figures/freezes/README.md) remains separate from the folder layout. Permanent data/media are on JOAQUIM; they were not moved or altered in this reorganization.
