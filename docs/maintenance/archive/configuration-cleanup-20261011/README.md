# Configuration cleanup — 2026-10-11

Four misplaced records moved byte-for-byte: trace preflight and increasing-trace protocol evidence to `docs/analysis/audits/`, past raw-file transfer provenance to `docs/maintenance/transfers/`, and the historical review-variant catalogue to `docs/analysis/figures/reviews/`. [Repository moves](repository-moves.json) preserve paths and hashes. Writers now save new audits under documentation.

The superseded V5 assembly candidate and full technical cohort CSV were copied through `scripts/repository_archive.py`, verified and then removed locally. The [transfer inventory](transfer-inventory.json) and [completed transfer](completed-transfer.json) record both files, their external destination and 16,163 bytes removed. The cohort schema remains documented locally; no raw data moved.

Eight pre-existing pointer discrepancies were repaired by recovering the expected bytes into a new immutable SSD bundle, without changing the pointers or either historical version. Seven came from exact prior SSD copies. The statistics CSV was recovered by restoring CRLF serialization; its expected SHA-256 matches exactly, with no numerical recomputation. [Recovery inventory](hash-recovery-inventory.json), [verification](hash-recovery-verification.json), and [hash-addressed locations](../verified-artifact-locations.json) record provenance and exact resolution. Requests with an expected hash fail if those bytes are unavailable or corrupt; they never silently select a different version.

The active registry and pipeline now agree on all 27 main panels, including Figure 1 H. E/F/G/H and Figure 2 D/E/G point to their existing scoped selections; first-release readiness remains separate. Historical ratio/trace review recipes remain explicitly precursors. Assembly scaffolds identify current overlays and the current assembly tool. No scientific definition, historical frozen byte, style or freeze changed.

[Protected hashes](protected-hashes.json) records selection and frozen-record bytes before cleanup. [Retained configurations](retained-configurations.json) gives the final file inventory. Validation is recorded in [verification](verification.json).
