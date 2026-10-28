# External storage and archive procedure

Permanent media, HTML reviews, numerical payloads and superseded candidate code belong on JOAQUIM. One self-contained review embeds requested versions, legends and provenance; one recorded working SVG per frozen/draft state is permitted by the figure workflow. Temporary previews are removed after verification.

Editable study/discussion/comparison Markdown also belongs beside its HTML on
JOAQUIM under the [study-artifact policy](review-artifacts.md). Plans remain in
Git; instructions, approved definitions, operating documentation and compact
authority/recovery records retain their repository roles. Existing source and
notebook migrations follow [Plan 12](../../Plans/12_STUDY_ARTIFACT_STORAGE.md).

Use an explicit external destination or `CLASSICAL_CONDITIONING_ARTIFACT_ROOT`; the configured directory must exist outside the repository. Never fabricate a missing mount. `CLASSICAL_CONDITIONING_ARCHIVE_ROOT` can identify a verified archive.

For host portability, `CLASSICAL_CONDITIONING_JOAQUIM_ROOT` may name the
existing mounted volume root. Windows discovers a unique mounted volume labelled
`JOAQUIM`; macOS uses the mounted `/Volumes/JOAQUIM`. `J:` is a historical path
alias and the current observed Windows mount, not a permanent requirement.
Multiple matching volumes fail until an explicit root is configured.
The relocation index translates recorded paths
to the consolidated figure-led layout before file hashes are checked. A missing
volume or mismatched artifact fails; it does not select another version.

The consolidated `ClassicalConditioning Outputs/figures/figure-NN/` tree owns
figure assemblies, panel scopes, working outputs, versions and studies. Shared
cross-figure material lives under `ClassicalConditioning Outputs/shared/`.
The SSD's 2026-10-26 consolidation records and subsequent 2026-10-27
collection refresh retain the original path identities in the relocation index.
A dated snapshot or published archive may supply a current accepted panel;
do not infer supersession from its location. Figure/study controls were reconciled
under publication guards. The main-paper collection now binds the authenticated
final saved wrapper through its completion receipt, original builder and exact
selected SVG/legend bytes. Recovery publishes repository evidence and the state
pointer under the existing locks; it does not rebuild or alter the SSD wrapper.
See the [portability reconciliation](../../records/repository-history/2026-10-28-ssd-portability/README.md).

Published archives under `ClassicalConditioning-Archive/` preserve `payload/<original-path>` and immutable transfer inventories. Control records are local under [repository archive](../../records/repository-archive/README.md). Resolve old references with `resolve_artifact(reference, expected_sha256=...)`; explicit hashes select exact recovered bytes and mismatches fail.

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

## Find reviews on JOAQUIM

Open `ClassicalConditioning Outputs/START-HERE.html`. The maintained [review library](../paper/figures/review-library.md) owns curated reading links; its JSON catalogue supplies verified external identities. The 26–27 October consolidation retired the machine output mirror and dated handoff roots after preserving their unique sources, Git history and recovery dependencies. Resolve old references through the relocation and recovery indexes rather than recreate those roots.

## Figure-led storage and lifecycle

The SSD's 27 October implementation records preserve these author-approved conventions:

- Keep permanent work under `figures/figure-NN/`, by assembly, panel or inseparable panel group, with `current/`, `versions/` and `studies/` roles. Computer origin is provenance.
- Maintain one working home per scoped state. Preserve distinct versions, legends and source bindings. Dates and matching pictures do not select current work.
- Store shared material once under `shared/` and link relevant scopes, retaining copies required by protected bundles.
- Reconcile useful source/Git changes and instructions from handoff packets before retirement. Further pruning requires its own authorization and verified recovery.
- Preserve notebook identities, annotations and manual-review holds. Register retained artwork and embedded versions in the supported browser; record unresolved context rather than invent legends or scientific meaning.

These conventions authorize no automatic deletion, scientific approval, restyling or freeze.

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
its existing lock and atomic pointer workflow. Windows process contention is
tested. Real macOS execution and simultaneous cross-machine shared-filesystem
locking remain pending. Windows symlink tests are reported separately when
creation privilege is unavailable. Allocated-space metrics are `null` where
`st_blocks` is unavailable; logical size is not a physical-space measurement.

## Drive-wide audit

The [24 October storage audit](../../records/repository-history/2026-10-24-joaquim-storage-audit/README.md)
records current folder sizes, processed-data status, archive integrity and the
roles of the dated PC/Mac handoff packets. Its full Markdown/HTML explanation
and file inventory live on JOAQUIM. Machine names and old dates do not by
themselves identify disposable data: the Delay baseline still supplies figure
inputs, and the old Mac handoff's pc-outputs subtree resolves selected Figure 2 G.
Treat dated source packets as handoff/recovery snapshots, not parallel active
repositories; preserve their dirty-copy history and bound exports before any
future consolidation. No folders were moved or pruned by that read-only audit.

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
