# Repository cleanup and deferred external archive

## Goal

Keep supported code, plans, decisions, exact frozen definitions/manifests/code and small external-artifact indexes in the repository. Store figures, visual HTML reviews, numerical payloads and superseded candidate code on JOAQUIM. This work package runs independently of analysis numbering.

## Phase A: prepare without the SSD

Inventory every repository file, consumers and freeze dependencies. Consolidate durable handoff/review content, coordinate plans, preserve frozen byte identities, implement archive resolution and external output destinations, and classify each transfer-bound file. Keep every pending-transfer original locally.

## Phase B: verified transfer

Use the archive tool after JOAQUIM is mounted. Validate destination and space, copy into staging, check SHA-256 and embedded archive contents, publish without overwriting an existing archive, and only then remove verified local payloads. Interrupted copies resume; modified sources require a refreshed inventory. Preserve existing raw data and historical freeze bytes.

## Acceptance

No pending source is removed prematurely. The completed working tree has no generated media, visual review HTML or numerical payloads. Supported tests and archive failure/recovery tests pass; records, approvals and frozen hashes are preserved. Report the folder map, retained code, review catalogue, dispositions and measured space reduction. Progress lives only in the implementation index; the transfer inventory records per-file state.

See [archive instructions](../docs/maintenance/archive/README.md).
