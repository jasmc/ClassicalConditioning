# External storage and archive procedure

Permanent media, HTML reviews, numerical payloads and superseded candidate code belong on JOAQUIM. One self-contained review embeds requested versions, legends and provenance; one recorded working SVG per frozen/draft state is permitted by the figure workflow. Temporary previews are removed after verification.

Editable study/discussion/comparison Markdown also belongs beside its HTML on
JOAQUIM under the [study-artifact policy](review-artifacts.md). Plans remain in
Git; instructions, approved definitions, operating documentation and compact
authority/recovery records retain their repository roles. Existing source and
notebook migrations follow [Plan 12](../../Plans/12_STUDY_ARTIFACT_STORAGE.md).

Use an explicit external destination or `CLASSICAL_CONDITIONING_ARTIFACT_ROOT`; the configured directory must exist outside the repository. Never fabricate a missing mount. `CLASSICAL_CONDITIONING_ARCHIVE_ROOT` can identify a verified archive.

`Raw Data` holds acquisition originals. `Digested Data` holds processed pipeline
projects, including reusable recording/cohort/analysis tables, processing
metadata, quality checks and run-generated figures. `ClassicalConditioning
Outputs` holds figure and paper workspaces: artwork, legends, figure-specific
results, studies and retained versions. Figure work references required
processed datasets through recorded paths and hashes; it does not duplicate
complete data projects. `ClassicalConditioning-Archive` holds preserved
historical material and recovery evidence. Existing frozen and path-bound
dependencies remain protected. The [pipeline directory contract](../running-analysis/outputs-and-provenance.md#directory-contract)
defines the contents of each processed project.

The figure-led destination is `ClassicalConditioning Outputs/figures/figure-NN/`.
Each populated assembly, individual panel or inseparable joint-panel scope owns
`current/`, `versions/` and `studies/` as needed. Shared studies and cross-figure
reviews live under `Outputs/shared/`. Scoped repository state still selects
current work; directory names and dates do not select science. A dated snapshot
or archived source can supply an accepted panel.

Published archives under `ClassicalConditioning-Archive/` preserve `payload/<original-path>` and immutable transfer inventories. Control records are local under [repository archive](../../records/repository-archive/README.md). Resolve old references with `resolve_artifact(reference, expected_sha256=...)`; explicit hashes select exact recovered bytes and mismatches fail.

Storage consolidation can relocate those payloads without editing historical
inventories. Exact `external_moves` and uniform `external_directory_moves` in
the existing relocation index authenticate the new destinations. Strict archive
verification follows exact aliases and still rejects missing or corrupt bytes;
it does not treat a relocation as permission to omit a payload.

For an existing transaction:

```sh
.venv/bin/python scripts/repository_archive.py dry-run --manifest <transfer-inventory.json>
.venv/bin/python scripts/repository_archive.py transfer --manifest <transfer-inventory.json>
.venv/bin/python scripts/repository_archive.py verify --manifest <transfer-inventory.json>
.venv/bin/python scripts/repository_archive.py prune --manifest <transfer-inventory.json>
.venv/bin/python scripts/repository_archive.py check
```

Transfer copies/verifies without removal. Prune rechecks all originals and published bytes first. Missing/corrupt destinations, changed originals and collisions fail. Keep originals until verification; use an independent manifest and destination for new transactions. Never overwrite a published checkpoint or rewrite historical frozen paths.

After a new freeze, prepare a scoped prior-version inventory and recommendations. The author chooses dispositions; the freeze request alone authorizes no archiving/deletion. Preserve shared active dependencies and pending versions. Historical transfers and dated handoffs remain recoverable through records.

`check_repository_structure.py --external` is a read-only archive audit. It
authenticates the completed disposition records bound in the structure policy
and recognizes only their exact deleted path/hash/size triples. Other missing
or corrupt payloads fail. The transfer inventories stay immutable, and
`repository_archive.py verify`, transfer and prune still require every payload;
audit exceptions cannot weaken those mutation safeguards.

## Figure-led storage and lifecycle

Author approved these conventions on 27 October 2026:

- Organize permanent work under Outputs by figure, assembly, individual panel
  or inseparable panel group, with populated `current/`, `versions/` and
  `studies/` roles. Computer origin is provenance, not a parallel workspace.
- Maintain one working home per scoped state. Preserve distinct historical
  versions and their labels, legends and scientific/source bindings. Dates,
  matching pictures and folder placement never select current work.
- Store shared material once and link it from every relevant scope. Preserve
  physical copies required by protected bundles.
- Treat handoffs as temporary packaging. Reconcile unique edits and Git history,
  integrate useful instructions into their single maintained home, and verify
  recovery and consumers before retiring obsolete packaging.
- Establish redundancy through bytes, scientific bindings and dependencies.
  Verify saved notes before retiring reading routes, preserve notebook and
  revision identities, and respect explicit manual-review holds.
- Name movable artwork by figure, panel and version; matching representations
  share a base name. Keep immutable historical filenames and authenticated aliases.
- Always keep every retained figure/panel version reachable from the main
  `ClassicalConditioning Outputs/START-HERE.html`, through its ordered figure,
  panel and version controls. Refresh the supported browser after retaining or
  publishing a version, including versions embedded in HTML or archives.
  Account explicitly for supporting illustrations, aliases, known gaps and
  unresolved ownership. Show one version at a time with its saved context and sources.
- Keep this upkeep token-efficient: use the supported builder and checker to
  generate links from recorded metadata. Do not manually enumerate every version
  in model instructions, replies or the home page. Keep the home page as the
  compact gateway to figure pages; do not embed the complete catalogue in agent
  instructions or routinely read/emit generated HTML. Save verbose tool reports
  to files and inspect bounded summaries and affected entries.

Maintain these procedures here rather than copying them into handoff documents.
They authorize no automatic deletion, scientific approval, restyling or freeze.

## Find reviews on JOAQUIM

Open `ClassicalConditioning Outputs/START-HERE.html`. Figure folders collect
artwork, legends, all distinct saved versions and studies across former computer
sources. The maintained [review library](../paper/figures/review-library.md) owns
curated reading links; its JSON catalogue supplies verified external identities.
The author confirmed on 27 October that no notes or selections had been added.
The former notebook routes were retired after verifying their retained copies
and preserving notebook identities; no browser import/reload is claimed.

The [consolidation record](../../records/repository-history/2026-10-27-storage-consolidation/README.md)
binds the inventory, recovery snapshot, manifest and publication/pruning receipts.
`scripts/maintenance/storage_migration.py` separates inventory, ZIP-member
recovery, exclusive copying, consumer publication and pruning. Unexpected files,
changed bytes, collisions and state-lock conflicts stop the affected operation.
Full inventories and maintenance evidence stay on JOAQUIM. Supported code, tests
and compact recovery pointers remain in the repository.

## Host configuration and verification

Use the existing Python 3.13 environment: `.venv\Scripts\python.exe` on Windows,
`.venv/bin/python` on macOS. `uv sync --locked --extra analysis` prepares the
declared environment when needed. The directly used `threadpoolctl` dependency
is declared and locked. Run from the repository root.

```powershell
$env:CLASSICAL_CONDITIONING_JOAQUIM_ROOT = 'J:\'
.\.venv\Scripts\python.exe scripts/check_repository_structure.py --external
.\.venv\Scripts\python.exe -m unittest tests.test_external_volume_portability tests.test_repository_archive tests.test_repository_structure tests.test_study_sources tests.test_storage_migration tests.test_figure_workspace tests.test_main_paper_figures tests.test_panel_g_review_repairs tests.test_figure_version_browser tests.test_command_organization
```

On a Mac use `export CLASSICAL_CONDITIONING_JOAQUIM_ROOT=/Volumes/JOAQUIM` and
the same arguments with `.venv/bin/python`. Explicit root configuration takes
precedence over discovery and must name an existing external root. File readers
still require the recorded SHA-256; configuring a directory does not authenticate
its contents. `CLASSICAL_CONDITIONING_ARTIFACT_ROOT` retains its separate role as
an existing destination for new outputs; `CLASSICAL_CONDITIONING_ARCHIVE_ROOT`
can select an existing verified archive. They do not select scientific versions.

`scripts/main_paper_figures.py display-target` verifies the recovered collection.
The `recover` command is for an existing storage-only completion receipt and its
hash-authenticated saved builder; it rejects changed sources, rendering logic,
embedded artwork or legends. Supply `--receipt`, `--receipt-sha256`,
`--builder-source` and a new repository `--evidence` path. Existing recovery
evidence is retained. Subsequent requested builds bind their current sources
and retire the obsolete recovery binding while preserving its evidence.

The portability tests exercise every path in `.gitattributes` using disposable
Git workspaces with `core.autocrlf=true`/`core.eol=crlf` and
`core.autocrlf=false`/`core.eol=lf`. They compare checkout bytes with indexed blobs
and check that the original index is unchanged. Do not renormalize protected
files. Independent edits require reconciliation; only indexed, authenticated
line-ending-only differences may be restored.

Analysis run locks use exclusive non-blocking OS locks; figure publication keeps
its existing lock and atomic pointer workflow. Native Windows and macOS process
contention and error release are tested. The [Mac acceptance record](../../records/figure-reviews/authority-reconciliation-20261025/ssd-portability-reconciliation-20261028.json)
records 223 passes and one Windows-only drive-discovery skip; all three symlink
cases ran on macOS. The verification record owns current SSD audit and final
Windows continuation outcomes; actual shared-filesystem cross-machine locking
remains pending. Serial single-host work does not require a cross-machine lock.
Final PC acceptance passed: 223 of 226 tests passed, with three skips for
Windows symlink privilege. The actual SSD audit verified 366 protected files
and 14 archive transactions.
See the verification record's `pc_acceptance` section for the tested commit and
exact artifact identities.
Allocated-space metrics are `null` where `st_blocks` is unavailable; logical size
is not a physical-space measurement.

The SSD was directly verified on the Mac on 28 October: 366 protected files,
10 registered states, 14 archive transactions and 67 reading entries passed.
The [offline-cache retirement](../../records/repository-history/2026-10-28-offline-cache-retirement/README.md)
then consolidated the dated Mac artwork package into retained SSD figure versions
and retired the completed Mac/PC operational handoffs. No local artwork cache
is required for normal work with the SSD attached. Exact original bytes and
source mappings remain recoverable; scoped scientific/presentation selections
are unchanged.

## Drive-wide audit

The [24 October storage audit](../../records/repository-history/2026-10-24-joaquim-storage-audit/README.md)
records current folder sizes, processed-data status, archive integrity and the
original roles of the dated PC/Mac handoff packets. Its full Markdown/HTML explanation
and file inventory live on JOAQUIM. Machine names and old dates do not by
themselves identify disposable data: the Delay baseline still supplies figure
inputs. At that audit's capture, selected Figure 2 G resolved through the old
Mac packet. The later consolidation migrates those exact exports and reconciles
source history before retiring packaging. The audit itself moved or pruned no
folders and is historical evidence rather than current folder authority.

The subsequent [implementation record](../../records/repository-history/2026-10-24-storage-organization/implementation.json)
records the drive-root `README.md` and `START-HERE.html` map and the verified
quarantine of the interrupted intake staging. That transfer preserved eight
original files, including AppleDouble metadata, under the archive quarantine; the
complete published recording was authenticated before staging removal. No
scientific bytes were discarded or capacity reclaimed by this same-drive move.
The root map distinguishes live data, reading artifacts, recovery archives and
dated handoffs without moving path-bound Figure 2 G dependencies.

On 26 October the author explicitly authorized disposal of those eight redundant
staging files after another raw/complete-input hash check, exact partial-tracking
prefix comparison and camera-table equality check. The
[completed disposition](../../records/repository-archive/intake-quarantine-20261024/disposition.json)
records 1,723,733,906 logical bytes removed. Complete intake files and original raw
inputs remain; original transfer inventories and receipts retain their bytes.
The live recovery index no longer advertises the deleted staging, and the existing
external-audit policy recognizes only the disposition's exact path/hash/size
absences. Strict transfer/verify/prune behavior is unchanged.
