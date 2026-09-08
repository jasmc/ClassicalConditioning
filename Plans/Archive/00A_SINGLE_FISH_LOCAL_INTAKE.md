# Step 00A — First Local Single-Fish Intake

**Status:** Complete  
**Change class:** Behavior-preserving ingestion  
**Depends on:** Minimal Step 00 path and environment decisions  
**Unlocks:** Inspection of the provided acquisition and reader confidence

**Completed:** 2026-08-29  
**Implementation commits:** `fe02193`, `6e3c4dd`  
**Validation:** 14 focused tests passed; complete pilot triplet converted and
losslessly verified; all raw SHA-256 hashes unchanged.

## Objective

Read one immutable acquisition triplet completely on the local machine, verify
the readers using a small local row preview, save losslessly compressed
Parquet mirrors, and produce an acquisition integrity report.

## Input

Exactly one matching:

```text
*_cam.txt
*_mp tail tracking.txt
*_stim control.txt
```

The current pilot is under:

```text
C:\Users\Public\More projects\Paper data\Raw single fish data
```

Raw files are opened read-only and are never moved, renamed, rewritten, or
used as an output location.

## Local-only boundary

- No Databricks or remote compute/storage.
- The complete files are not included in LLM calls.
- Debug previews and all processing remain local.
- Reports and interactive figures are local files.

## Work

1. Create the minimum package/CLI entry point.
2. Pair the three files by recording basename.
3. Read the header and first tens of rows locally to confirm delimiter,
   decimal notation, columns, and parser behavior.
4. Stream the complete camera and tracking files.
5. Validate frame IDs, timestamps, parse completeness, camera/tracking overlap,
   tail columns, non-finite values, and protocol coverage.
6. Write:
   - `camera.parquet`;
   - `tracking.parquet`;
   - `stimulus_events.parquet`;
   - `source_manifest.json`;
   - `acquisition_summary.json`.
7. Use lossless Zstandard compression and preserve parsed numeric precision.
8. Generate compact acquisition-integrity figures.

## Figures

- Camera inter-frame interval, frame-ID differences, effective rate, and drift
- Camera/tracking frame coverage and gaps
- Tail-point value/missingness overview
- Stimulus protocol timeline and acquisition bounds
- Local HTML summary

Figure modes:

- `static`: PNG for this first check
- `interactive`: self-contained local HTML for zoom/hover where useful
- `publication`: not required for this technical intake

## Folder output

```text
Paper data/
|-- Processed data/<fish-id>/
|   |-- camera.parquet
|   |-- tracking.parquet
|   `-- stimulus_events.parquet
|-- Quality checks/<fish-id>/
|   |-- acquisition_report.html
|   |-- acquisition_summary.json
|   `-- figures/
`-- Metadata/
    `-- source_manifest.json
```

## Validation

- Complete row counts reconcile with the source parser.
- First/last frame identities and timestamps reconcile.
- Parquet reload exactly matches the parsed typed representation.
- Compression round trip is exact.
- The raw directory contains no new or modified files.
- The report clearly separates errors, warnings, and observations.

## Exit gate

The full triplet is read locally, lossless Parquet mirrors reload exactly, raw
files remain unchanged, and the acquisition report is sufficient to decide
whether reader or acquisition issues require review before preprocessing.

## Completion evidence

- Camera rows: 8,361,388
- Tracking rows: 8,361,388
- Protocol rows: 173
- Internal camera frame gaps: 0
- Internal tracking frame gaps: 0
- Protocol events outside acquisition: false
- Review note: camera and tracking ranges are offset by one frame at each
  boundary; this requires interpretation before synchronization but does not
  indicate an internal frame gap.
- Canonical outputs:
  - `C:\Users\Public\More projects\Paper data\Processed data\20221115_04`
  - `C:\Users\Public\More projects\Paper data\Quality checks\20221115_04`
  - `C:\Users\Public\More projects\Paper data\Metadata`

## Out of scope

- filtering;
- vigor/activity metrics;
- bout detection;
- trial segmentation;
- scaling;
- exclusions;
- group statistics.
