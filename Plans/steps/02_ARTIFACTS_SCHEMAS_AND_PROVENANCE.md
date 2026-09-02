# Step 02 — Artifact Formats, Schemas, and Provenance

**Status:** In progress — integrity/publication foundation implemented
**Change class:** Behavior-preserving  
**Depends on:** Steps 00-01  
**Unlocks:** Safe migration away from pickle and explicit stage dependencies

## Objective

Define how every stage saves, validates, identifies, and reloads data without
changing analytical values.

## Format policy

| Content | Canonical | Additional output |
| --- | --- | --- |
| Scientific tables | Parquet with lossless Zstandard compression | CSV review copy where useful |
| Configuration and provenance | JSON | Optional human-authored YAML/TOML input |
| Dense frame × point arrays | HDF5 only after a local benchmark justifies it | NPZ/Zarr comparison candidates |
| Cohort/classification manifests | Parquet | CSV review copy |
| Figures | SVG and PDF | PNG preview |
| Current artifacts | Read-only pickle | Conversion/comparison input only |

All scientific-data compression is lossless. Ingestion does not downcast,
round, quantize, or clip source values.

## Core types

### `ArtifactRef`

```python
@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    artifact_type: str
    path: Path
    format: ArtifactFormat
    schema_version: str
    logical_content_hash: str
    file_hash: str | None
```

### `ArtifactMetadata`

Records:

```text
stage_id
stage_implementation_version
input artifact IDs and hashes
relevant config hash
cohort hash where applicable
code commit
environment ID
scientific status
created time
row/fish/trial counts
warnings
```

### `ArtifactRepository`

A small file-system interface for:

- writing without silent overwrite;
- validating path, hash, and schema;
- resolving an explicit artifact ID within the local project;
- listing the project artifact manifest;
- loading supported formats.

It is not a database abstraction.

## Work packages

### 02.1 Define schema registry

Initial schemas:

```text
source-inventory/1.0
raw-camera/1.0
raw-tracking/1.0
stimulus-events/1.0
processed-samples-legacy/1.0
tail-representation/1.0
frame-activity/1.0
movement-state/1.0
cohort-manifest/1.0
trial-outcomes/1.0
classification-manifest/1.0
statistical-results/1.0
panel-data/1.0
```

Each schema defines:

- required and optional columns;
- dtypes or permitted dtype families;
- units;
- primary/uniqueness keys;
- valid categories;
- allowed missingness;
- time ordering;
- foreign-key-like identity rules;
- normalization for comparison and hashing.

### 02.2 Define canonical table serialization

Before saving:

- reset semantically meaningful indexes into explicit columns;
- use declared column order;
- sort by declared keys when row order is not itself meaningful;
- normalize extension and categorical dtypes;
- preserve category order in metadata;
- preserve exact missingness;
- reject undeclared duplicate keys;
- attach schema version and scientific metadata.

### 02.3 Define stable hashing

Hashing must include:

- schema version;
- canonical column order;
- declared row ordering;
- normalized dtypes;
- explicit missing representation;
- values.

The required artifact identity is a logical-content hash computed from the
schema and normalized scientific content. Specify the algorithm and maintain
cross-version tests.

A separate file-byte hash verifies the particular stored file. It may differ
after legitimate re-serialization of identical logical content by another
Parquet/compression version and therefore must not define cross-environment
scientific identity.

### 02.4 Implement controlled pickle conversion

Conversion procedure:

```text
legacy pickle
-> load in pinned compatibility environment
-> validate expected object type
-> normalize index/categories/sparse columns
-> write schema-versioned Parquet
-> reload Parquet
-> reconstruct comparison representation
-> compare values, missingness, identities, and summaries
-> record source pickle hash
```

Never accept successful file writing as evidence of scientific equivalence.

### 02.5 Compare without writing new pickle

During Track A:

```text
legacy function or frozen existing pickle
    <-> normalized in-memory comparison
    <-> canonical Parquet output
```

New code never writes pickle. Compatibility consumers migrate to Parquet before
their producing stage is accepted.

### 02.6 Define the stable local directory layout

```text
Paper data/
    Raw single fish data/
    Processed data/<fish-id>/
    Quality checks/<fish-id>/
    Tables/
    Models/
    Figures/Publication/
    Figures/PNG/
    Figures/Interactive/
    Metadata/
```

Artifact versions and the central manifest preserve scientific alternatives
without creating a new directory tree for each execution.

### 02.7 Evaluate HDF5 only for later dense arrays

When Step 06 produces dense frame × point × coordinate arrays, benchmark
lossless HDF5 against the simplest viable alternatives using:

- exact round-trip equality;
- file size;
- sequential and sliced read time;
- append/write time;
- chunk access patterns;
- interoperability and metadata clarity.

HDF5 is not used for ordinary tables merely to place many artifacts in one
container.

## Required tests

- Pickle-to-Parquet round-trip for dense, sparse, categorical, indexed, and
  missing-value tables
- Duplicate-key rejection
- Schema mismatch rejection
- Hash mismatch rejection
- Stable hash under permitted row normalization
- Different hash when a scientific value, category, or missingness changes
- Explicit rejection of ambiguous artifact selection
- No silent overwrite
- Unsupported pickle object rejection

## Current implementation evidence

Implemented foundations:

- SHA-256 file hashing for byte-level integrity;
- authenticated intake source-manifest loading;
- same-filesystem transactional multi-artifact publication with rollback;
- race-safe atomic JSON publication using unique same-directory temporary files;
- preflight rejection of empty transactions, duplicate staged/final paths, and
  missing staged artifacts before existing outputs are moved;
- lossless Zstandard Parquet and authenticated JSON/marker publication across
  the implemented intake, legacy, candidate, movement, profile, figure, and
  trace-review routes.

Still required before the Step 02 exit gate:

- versioned schema registry and canonical dtype/category rules;
- logical-content hashing independent of Parquet byte serialization;
- typed `ArtifactRef`, `ArtifactMetadata`, and explicit-ID repository resolver;
- central project artifact manifest;
- controlled read-only pickle-to-Parquet conversion with dense, sparse,
  categorical, indexed, and missingness equivalence tests;
- migration of all current route-specific metadata to the shared contract.

## Exit gate

Representative legacy artifacts can be converted, loaded, and scientifically
compared without value, identity, category, or missingness drift. Every new
canonical artifact has a schema, hash, provenance, and explicit upstream IDs.

## Downstream invalidation

- Serialization-only change: rewrite and revalidate the artifact; analytical
  descendants need not rerun if the logical content hash is unchanged.
- Schema change affecting meaning: new schema version and downstream contract
  validation.
- Value change: new artifact identity and normal dependency invalidation.
