# Schema, Semantic Provenance, and Legacy Conversion

**Status:** Deferred — not required for the current fixture pipeline
**Change class:** Behavior-preserving
**Depends on:** Stable artifact contracts for the route being formalized
**Unlocks:** Cross-environment semantic identity, unambiguous artifact
selection, controlled legacy conversion, and release-grade provenance

## Objective

Add only the general provenance machinery that remains useful once the
scientific contracts are stable. The archived artifact-integrity plan already
covers file integrity and
safe publication for the current package routes.

## Why this is different from file integrity

The implemented integrity layer answers: “Is this the exact file that this
stage previously wrote?”

This plan answers: “What scientific table is this, what does each field mean,
and is it scientifically identical after a safe re-serialization?”

For example, a Parquet writer upgrade can change compression metadata and
therefore a SHA-256 byte digest, even when rows, values, categories, units, and
missingness are unchanged. A schema and semantic hash distinguish that benign
rewrite from a changed scientific result.

## Deferred work packages

### 02.1 Schema contracts

Define versioned schemas only for artifacts that have stabilized in active
analysis: required columns, units, valid categories, keys, allowed missingness,
ordering, and foreign-identity rules. Do not create a registry for provisional
candidate outputs merely because a generic registry is available.

### 02.2 Canonical scientific-table identity

For each adopted schema, define canonical column order, key ordering, dtypes,
categorical order, and missing-value representation. Compute a semantic hash
from that normalized content and schema version. Preserve a separate SHA-256
file hash for storage integrity.

### 02.3 Explicit artifact references and resolution

Introduce a small typed metadata record and resolver only when more than one
valid version of an artifact could be selected. It must resolve an explicit
artifact ID and validate schema, semantic hash, upstream IDs, configuration,
cohort, code, environment, scientific status, and warnings. It is not a
database abstraction and must not guess “latest.”

### 02.4 Controlled legacy pickle conversion

Convert a legacy pickle only when it is needed as an accepted reproduction
reference. Load it in a pinned compatibility environment; validate object
type; write a schema-versioned Parquet table; reload; and compare values,
identities, categories, indexes, and missingness. New package code must not
write pickle.

### 02.5 Dense-array storage decision

If a later accepted tail or imaging stage requires dense arrays, benchmark HDF5
against the simplest alternatives for lossless round trips, size, access,
metadata, and interoperability. Ordinary tables remain Parquet.

## Entry criteria

- The relevant metric, cohort, outcome, or model contract is approved rather
  than still under Gate P, T1, C0, O, or S discussion.
- There is a concrete cross-environment, release, or multi-version use case.
- The responsible scientific owner can approve field meanings and units.

## Validation

- Equivalent canonical tables have the same semantic hash across permitted
  re-serializations; a value, category, unit, key, or missingness change does
  not.
- Schema violations, duplicate keys, invalid foreign identities, and ambiguous
  artifact selection fail explicitly.
- A selected legacy conversion demonstrates exact scientific equivalence, not
  merely that Parquet was written.
- The resolver rejects a mismatched schema, semantic hash, cohort, config, or
  upstream artifact.

## Exit gate

The adopted paper/release artifacts can be identified and reloaded
unambiguously across supported environments, with their scientific meaning and
lineage preserved. This is required before a release that claims durable
cross-environment reproducibility; it is not required for present fixture-only
development.
