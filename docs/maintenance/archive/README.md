# Repository archive and cleanup

The repository keeps supported implementation/tests/configuration, current documentation/plans/decisions and exact frozen definitions/manifests/code. Permanent figures, visual HTML, numerical payloads and alternative one-off code go to JOAQUIM. The 2026-10-10 transfer is complete: 2,050 files (1,252,400,031 bytes) verified on JOAQUIM, then removed locally. [Completion record](completed-transfer.json) supersedes the immutable inventory’s pending-transfer checkpoint.

## Current records

- [Starting inventory](starting-inventory.json): initial Git state and original hashes; runtime/cache inspection is separate.
- [Per-file dispositions](transfer-inventory.json): role, candidate consumers, freeze dependencies, action, original hashes and canonical duplicate identity.
- [Repository relocations and exact extraction](relocations.json): moved definitions and byte-preserved freeze/code entries.
- [Retained code](retained-code.json): supported tools and immutable snapshots.
- [Review catalogue](REVIEW_CATALOGUE.md): one current consolidated review per lineage, with historical evidence retained separately.
- [Artifact index](ARTIFACT_INDEX.md): logical source paths, archive destinations and SHA-256.
- [Cleanup evidence](CLEANUP_RECORD.md) and [deletions](deletions.json): measured changes and recoverability.

The inventory is a transfer checkpoint. Refresh it before transfer only when sources changed; never discard its mapping after a completed transfer. `starting-inventory.json` is immutable. Archive control records are retained and excluded from recursive self-inventory.

## Historical preparation procedure

Run from the repository with the locked Python environment:

```sh
.venv/bin/python scripts/repository_archive.py dry-run
.venv/bin/python scripts/repository_archive.py check
```

The dry run hashes pending local sources and reports the disconnected mount without creating a destination. The check rejects unaccounted generated payloads; exact inventoried originals are temporarily permitted until verified transfer.

## Transfer procedure for a new inventory

Default destination: `/Volumes/JOAQUIM/ClassicalConditioning-Archive/repository-cleanup/2026-10-10/`.

```sh
.venv/bin/python scripts/repository_archive.py transfer
.venv/bin/python scripts/repository_archive.py verify
.venv/bin/python scripts/repository_archive.py prune
.venv/bin/python scripts/repository_archive.py check
```

Transfer validates the mounted volume name and destination, space, every source hash, copied bytes and embedded HTML/ZIP members. It resumes its matching staging inventory, refuses destination collisions and publishes only a fully verified archive. It does not remove sources. Prune rechecks the published inventory, every archived payload and all remaining source identities before deleting any local file. A changed source prevents removal. An interrupted prune can be resumed against the same published manifest.

Each archive keeps `payload/<original-repository-path>` to preserve bundle-relative links. SHA-256 duplicate groups identify identical content but original bundle paths are preserved; no frozen payload is discarded merely because another copy exists. Existing unrelated SSD files are not overwritten. The full archive is independent of unrelated pre-existing SSD copies.

The relocation resolver maps old repository and Windows references to moved repository definitions, pending originals or verified external bytes. Historical external J:/F: artifacts not present locally remain explicit unavailable evidence until the drive is accessible; no fake authentication is recorded.

## New output destinations

Set `CLASSICAL_CONDITIONING_ARTIFACT_ROOT` to an existing directory outside this repository, or use explicit external CLI output arguments. The default is on mounted JOAQUIM. `CLASSICAL_CONDITIONING_ARCHIVE_ROOT` can identify a verified relocated archive. Missing volumes are not fabricated as directories. Temporary tests/render verification may use system temporary directories.

No scientific recomputation, new freeze, raw-data transfer, restyling or Git-history rewriting is part of cleanup. Old committed payloads remain in Git history.

## Scripts follow-up

The [scripts-only follow-up](scripts-cleanup-20261010/README.md) has its own immutable checkpoint and completion record. Use `--manifest <checkpoint>` for a separate transfer; do not overwrite a published archive or replace the original inventory. The resolver recognizes indexed follow-up inventories.
