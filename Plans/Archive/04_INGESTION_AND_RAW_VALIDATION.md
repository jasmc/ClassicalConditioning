# Step 04 — Raw Ingestion, Tracking Audit, and Validation

**Status:** Complete for local fixtures (Gate T0 answered)  
**Change class:** Behavior-preserving readers plus scientific audit  
**Depends on:** Steps 02-03  
**Unlocks:** Legacy preprocessing and new tail representations

**Completed (scoped):** 2026-08-31  
**Archive note:** Detailed plan archived after local-fixture Gate T0 exit.
Broader paper-recording inventory remains when more raw arrive; video audit
deferred.

Progress note: intake/readers/validate-raw/Gate T0/tracking audits **done** for
both local fish. Broader paper-recording inventory when more raw arrive;
video audit deferred.

## Objective

Extract raw readers, preserve current parsed values, and determine exactly what
tail information is present and trustworthy before implementing 2D or
curvature-based metrics.

## Scientific gate T0

Answered from legacy readers, package geometry checks, and owner confirmation
(see `Plans/DECISIONS.md`):

- [x] Raw `angleN` are radians; legacy vigor converts to degrees with
      `* (180/pi)`.
- [x] `angle1..angle14` behave as local intersegment bends (agree with
      XY-derived segment orientation changes on the pilot); `angle15` is a
      terminal placeholder.
- [x] Raw X/Y coordinates are present (`xN`/`yN`).
- [x] Coordinates are treated as tracking-image pixels (not body-axis
      calibrated world units).
- [x] Absolute micrometre calibration is not required for relative activity
      metrics.
- [x] Tail-point spacings are measured from XY (not a fixed segment length).
- [x] Independent body-axis / confidence fields are absent in current files.
- [ ] Point-count stability across all paper recordings still needs a cohort
      inventory sample.
- [x] Angle wrapping uses standard principal-value differences in candidate
      metrics; legacy vigor uses degree differences after conversion.
- [x] Trailing acquisition summary row is dropped by the legacy/package
      readers.
- [ ] Synchronized raw video remains optional/deferred for blinded validation.

Measured 2D speed is therefore interpretable as relative pixel motion, not as
calibrated physical speed. If synchronized video stays unavailable, use the
predeclared non-video scorecard mode rather than ad hoc reweighting after
candidate results are seen.

## Raw schemas

### Camera

```text
recording_id
frame_id
elapsed_time_ms
absolute_time_ms
```

### Tracking

The reader should preserve available source fields before choosing an analysis
subset:

```text
recording_id
frame_id
point angles
point X/Y coordinates, if present
body reference, if present
tracking confidence, if present
source software/version metadata, if recoverable
```

### Stimulus events

```text
recording_id
event_type
source_event_id
start_time_ms
end_time_ms
```

## Work packages

### 04.1 Extract fish and recording identity parser

Use explicit format definitions and specific errors. Test all known naming
conventions. Do not infer scientific cohort from filenames.

### 04.2 Extract camera reader

Preserve current separator and decimal-format support, but replace broad
success-shaped failure with explicit parse errors.

Validate:

- required columns;
- integer and unique frame IDs;
- monotonic timestamps;
- duplicate rows;
- non-finite timestamps;
- recording coverage.

### 04.3 Extract tracking reader

Initially reproduce current selected-angle parsing. Add an audit/full-reader
mode that inventories all available columns without discarding X/Y or
confidence fields.

The audit must not assume every non-frame column is an angle.

### 04.4 Extract protocol reader

Validate:

- known event types or explicitly retained unknown types;
- start before end;
- stable ordering;
- duplicate source events;
- events outside acquisition;
- required CS/US counts for the experiment.

### 04.5 Implement frame-loss diagnostics

During legacy reproduction, preserve the current decision output.

Separately compute candidate corrected diagnostics:

- exact gaps in expected `FrameID` sequence;
- duplicate frame IDs;
- timestamp differences;
- integer-multiple gap estimates;
- jitter distribution;
- disagreement between frame-ID and timestamp evidence.

Do not change inclusion until the frame-loss scientific decision is approved.

### 04.6 Produce the tracking audit

Sample across:

- experiments;
- conditions;
- rigs;
- recording years/software versions;
- normal and problematic fish.

Report:

- field availability;
- point count and spacing;
- angle and coordinate ranges;
- missingness;
- confidence availability;
- body reference availability;
- compatibility with measured or reconstructed tail representation.

### 04.7 Add malformed-input tests

Cover:

- missing columns;
- mixed separators/decimal formats;
- corrupted rows;
- empty file;
- missing final summary row;
- duplicate frames;
- missing frames;
- nonmonotonic timestamps;
- point-count changes;
- unexpected extra fields.

### 04.8 Audit video synchronization

Where video exists, verify:

- recording/fish identity;
- frame-number or timestamp mapping;
- dropped/duplicated video frames;
- offset and drift relative to tracking;
- ability to extract blinded clips without group labels.

Where it does not exist or cannot be synchronized, record that limitation and
activate only the scorecard variant frozen in Step 00.

## APIs

```python
read_camera(source: ArtifactRef, config: RawReaderConfig) -> pd.DataFrame
read_tracking(source: ArtifactRef, config: TrackingReaderConfig) -> pd.DataFrame
read_protocol(source: ArtifactRef, config: ProtocolReaderConfig) -> pd.DataFrame
audit_tracking_schema(paths: Sequence[Path]) -> TrackingAudit
validate_frame_sequence(camera: pd.DataFrame, config: RawValidationConfig) -> FrameReport
```

## Deliverables

- Explicit reader modules and schemas
- Compatibility adapters for current scripts
- Tracking field/semantics audit
- Frame-sequence report
- Raw Parquet artifacts for approved fixtures
- Reader characterization and malformed-file tests

## Exit gate

For the pilot and fixtures, new readers reproduce normalized current reader
outputs while also preserving sufficient raw fields for the approved tail
representation. Gate T0 is resolved or explicitly limits candidate metrics.

## Downstream invalidation

- Reader-only bug that changes values: ingestion onward
- Newly retained but unused column: no invalidation of existing metrics
- Changed angle semantics or coordinate interpretation: tail representation
  onward
- Changed frame-loss decision: QC/cohort and any processing that excludes or
  interpolates affected frames
