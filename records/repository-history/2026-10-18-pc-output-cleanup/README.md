# Author-requested PC output cleanup, 18 October 2026

Scope: `F:\ClassicalConditioning Outputs` and `F:\Results_refactored` only.
The author requested deletion of old material no longer useful according to
the current repository and JOAQUIM. `completed.json`, when present, records
the completed deletion; `prepared.json` alone does not establish completion.

The initial folder count for Results_refactored included `Processed data/Analyses`:
there are 56 acquisition directories, not 57. All 56 are represented in
JOAQUIM's retained `Digested Data/allDelay-full-v1` project. The earlier
`delay-cohort-v1` project includes legacy and superseded candidate derivatives;
the retained project has the replacement corrected derivative files.
No repository text consumer of the retired processing root was found before
creating these cleanup records.

Preparation checks matching recording/condition identities, raw-source
SHA-256/size provenance, raw presence and size, and matching logical intake
hashes. The actual retained SSD camera, tracking and stimulus Parquet files
are hashed against their retained manifests. Some 2024 Parquet byte hashes
differ across Windows/Mac serialization while logical data hashes agree;
the old and new derivative files are not claimed to be byte duplicates.
The raw text files are not rehashed in this operation.

Every file in the old Figure 2 output directory is preserved exactly in a
new, verified archive before removing its F: copy. Old Delay run metadata,
quality records, logs, and previews are also preserved. Obsolete Parquet
derivatives and abandoned `.incomplete` files are retired after confirming
the supported replacement project and preserved source provenance.

Archive: `J:\ClassicalConditioning-Archive\pc-output-cleanup\2026-10-18\`.
The supported `scripts/repository_archive.py` transfer, verification and pruning
functions are used for preserved payloads. Its Mac-specific mount check is
replaced by an explicit Windows `Get-Volume` identity check requiring the
`JOAQUIM` volume label at J:. No fabricated mounts or rewritten frozen
manifests are used.

Evidence:

- `starting-inventory.json`: original paths, sizes and modification times.
- `retained-source-verification.json`: actual retained intake hashes and IDs.
- `figure2-mirror-verification.json`: exact comparison with the old Mac handoff.
- `*-transfer-inventory.json`: SHA-256 identities of preserved historical files.
- `*-archived-prune.json`: guarded removal reports for archived originals.
- `completed.json`: deleted roots, file count, bytes and completion time.

`cleanup.py` is the scoped transaction implementation. Its apply phase refuses
unexpected roots, reparse points, changed source inventories or corrupt archives.
The active `F:\Digested Data` projects and JOAQUIM's accepted/current paper
artifacts are outside the deletion scope.
