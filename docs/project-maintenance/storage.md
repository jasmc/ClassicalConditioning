# External storage and archive procedure

Permanent media, HTML reviews, numerical payloads and superseded candidate code belong on JOAQUIM. One self-contained review embeds requested versions, legends and provenance; one recorded working SVG per frozen/draft state is permitted by the figure workflow. Temporary previews are removed after verification.

Use an explicit external destination or `CLASSICAL_CONDITIONING_ARTIFACT_ROOT`; the configured directory must exist outside the repository. Never fabricate a missing mount. `CLASSICAL_CONDITIONING_ARCHIVE_ROOT` can identify a verified archive.

`ClassicalConditioning Outputs/current/` contains current working outputs; historical artifacts are retained or placed in separately indexed history/archive locations. A dated snapshot or published archive may supply a current accepted panel. Do not infer supersession from its location.

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
