# Step 02A — Artifact Integrity and Transactional Publication

**Status:** Complete for the implemented package routes; archived 2026-09-15
**Change class:** Behavior-preserving

## Closed scope

The package can publish and reuse implemented intake and analysis artifacts
without silently accepting changed inputs or partially published output sets.
This scope establishes byte-level integrity and explicit route-specific
lineage; it does not establish a general semantic artifact registry.

## Delivered contract

- SHA-256 is calculated over source and derived files. A changed byte changes
  the digest, so callers can detect altered or substituted files.
- Intake writes a source manifest recording the raw camera, tracking, and
  protocol paths, their SHA-256 digests, and the hashes of their lossless
  Parquet derivatives.
- Completed artifact markers and summaries record recipe, recording or
  analysis identity, output hashes, and required upstream hashes.
- Reuse verifies those identities and hashes before downstream stages read an
  artifact.
- Multi-file publications stage output beside its destination, then publish as
  one transaction with rollback on failure. Existing output is rejected unless
  overwrite is explicitly requested.
- Intake writes lossless Zstandard Parquet and verifies its row content after
  a round trip before publication.

## Evidence

- `src/classical_conditioning/artifacts.py` supplies SHA-256 helpers, atomic
  JSON writes, staged publication, rollback, and completed-artifact checks.
- `src/classical_conditioning/intake.py` publishes authenticated source
  manifests and verifies lossless Parquet conversion.
- Existing `tests/test_artifacts.py`, `tests/test_intake.py`, and downstream
  route tests cover failure, rollback, overwrite, and lineage cases.

## Boundary

SHA-256 identifies the exact stored bytes, not the scientific meaning of a
table. Equivalent Parquet files written by different library versions may have
different byte hashes. General schemas, semantic hashes, explicit artifact
identity/resolution, and controlled pickle conversion were intentionally split
into the active Step 02 plan.
