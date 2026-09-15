# Step 06 — Corrected Preprocessing, Tail Representation, and Candidate Metrics

**Status:** Archived by user request on 2026-09-15 — candidate implementation retained; Gate P not approved
**Change class:** Scientific-correction candidate development  
**Depends on:** Steps 04-05; Gate T0 decided in [DECISIONS.md](../DECISIONS.md)
**Unlocks:** Metric validation and selection

> **Archive note:** The three-metric candidate implementation, measured-time
> preprocessing, gap masks, base translation, and fixture pipeline are
> retained. This archive does not claim that Gate P, the remaining mathematical
> known-answer checks, or the optional second-fixture legacy regression passed.

## Objective

Develop and freeze corrected frame/timing preprocessing, create one validated
representation of the cleaned tail, and calculate all candidate activity
metrics side by side while retaining legacy point-15 vigor.

This step creates candidates; it does not select the paper metric.

## Candidate metric identities

| Metric ID | Definition | Indicative unit |
| --- | --- | --- |
| `legacy_distal_angular_speed` | Absolute speed of current distal cumulative angle | deg/ms |
| `tail_length_weighted_angular_l1` | Tail-length-weighted mean absolute segment angular speed | rad/ms |
| `whole_tail_xy_mean_speed_normalized` | Tail-length-weighted mean 2D point speed divided by recording median tail length | tail lengths/ms |

These are the active scalar candidates. The unweighted manuscript-described
segment sum is superseded by the weighted angular L1 metric because an
unweighted sum depends on tracking-point number and spacing. Angular RMS is
also superseded because it uses the same angular signal as L1 while emphasizing
large local movements that are not part of the scientific objective.

## Representation branches

### Measured XY

Use only if source coordinates and scale pass gate T0:

```text
measured points
-> base translation
-> body-axis rotation
-> length normalization/calibration
-> validity mask and weights
```

### Reconstructed XY

Use only if angle semantics and segment lengths are validated:

```text
local angle
-> unwrap
-> cumulative orientation
-> integrate segment vectors
-> relative coordinates
```

Store `coordinate_source = reconstructed`. Do not interpret it as measured
physical displacement.

### Angular representation

Retain:

- source local angle;
- cumulative orientation;
- explicit angle wrapping and unwrapping method;
- per-point validity.

## Core types and interfaces

```text
TailRepresentationSpec
TailRepresentationRef
ActivityMetricConfig
ActivityMetricResult
TailRepresentationBuilder
ActivityMetric
```

Example protocol:

```python
class ActivityMetric(Protocol):
    metric_id: str
    implementation_version: str
    output_unit: str

    def calculate(
        self,
        tail: TailArrays,
        config: ActivityMetricConfig,
    ) -> MetricResult:
        ...
```

## Work packages

### 06.1 Develop corrected foundational preprocessing

Define, implement, and validate:

- direct frame-ID gap and duplicate detection;
- timestamp-gap and jitter diagnostics;
- the corrected frame-loss decision rule;
- camera/tracking/protocol synchronization;
- protocol events between or outside observed frames;
- interpolation eligibility and maximum gap;
- treatment of long invalid gaps;
- spatial and temporal filtering, including edge handling;
- preservation of measured timestamps for derivatives.

Reference tests include perfect cadence, one/several/sustained missing frames,
duplicates, jitter, protocol events between frames, events outside acquisition,
constant/impulse/step signals, missing samples, and filtering edges.

Scientific gate P freezes these policies before any corrected candidate result
is accepted. If a policy remains legacy by deliberate decision, its legacy
implementation ID and residual limitation are recorded explicitly.

### 06.2 Build time representation

Retain measured frame intervals. Define:

- handling of jitter;
- maximum interpolation gap;
- invalid long gaps;
- velocity derivative time points;
- whether any uniform grid is analytical or display-only.

Do not assume exactly 700 FPS when measured timing is approved for the
corrected recipe.

### 06.3 Build body-centred representation

Where body reference exists:

- translate tail base to origin;
- align body axis;
- preserve original coordinates;
- test transformations on known rotations/translations.

Where it does not exist, record the limitation and do not invent a correction.

### 06.4 Calculate tail-length weights

Interior point weight:

```text
0.5 * distance(previous, point) + 0.5 * distance(point, next)
```

Endpoints receive half their neighboring interval. Normalize over valid tail
length for frames with a small missing portion.

Define and version:

- distance source;
- reference frame or per-frame weights;
- minimum valid-tail fraction;
- point-density invariance expectations.

### 06.5 Define shared minimal cleaning

Shared operations may include:

- validity mask;
- angle unwrapping;
- rejection of impossible point jumps;
- long-gap invalidation;
- conservative base/body correction.

Shared cleaning must not favor one candidate. Candidate-specific smoothing is
recorded inside each metric configuration.

### 06.6 Implement candidate metrics

For each metric:

- write mathematical definition;
- define input representation;
- define units;
- define edge and missing-point handling;
- define temporal/spatial smoothing;
- define outlier policy;
- implement pure NumPy calculation;
- return value and validity/coverage arrays;
- add synthetic reference tests.

### 06.7 Preserve legacy benchmark

Calculate or join the accepted legacy point-15 metric using its frozen recipe.
Do not recalculate a subtly different approximation and call it legacy.

The retained legacy column is a modern measured-time benchmark of the old
distal cumulative-angle formula; exact historical execution remains archived.

### 06.8 Write frame-activity artifact

Recommended canonical long form:

```text
experiment_id
recording_id
fish_id
frame_id
timestamp_ms
metric_id
metric_implementation_version
metric_parameter_set_id
value
unit
valid_tail_fraction
valid_point_count
quality_flags
```

An optional wide artifact may be derived for performance.

### 06.9 Add mechanistic branches without making them prerequisites

From the common representation, permit later modules for:

- PCA/eigen-tail dynamics;
- rhythmic power and frequency;
- curvature kymographs;
- traveling-wave energy, direction, speed, and coherence.

These are exploratory until separately approved.

## Mathematical known-answer checks

These are not more fish experiments and do not require video. They are tiny,
artificial tail movements for which the expected answer is known before the
code runs. For example, a tail translated rigidly across the image should have
zero base-centred tail motion; an S-bend with opposite segment rotations should
remain visible to angular L1 but cancel in the legacy distal benchmark. These
checks catch arithmetic and coordinate-handling mistakes before candidate
results are inspected.

Already covered by automated tests:

- stationary straight tail;
- rigid translation;
- one moving segment;
- S-bend with opposing segment motion;
- missing distal points;
- point-density normalization;
- missing, duplicate, reversed, and chunk-boundary frame sequences; and
- long measured-time gaps.

Still to add, with the expected direction or exact value written first:

- stationary bent tail and rigid rotation;
- coordinated C-bend and base-to-tip traveling bend;
- an isolated point spike and the adopted outlier response;
- unequal segment spacing; and
- equivalent motion sampled at different frame rates and with ordinary timing
  jitter.

Filter-edge tests are required only if Gate P adopts a temporal or spatial
filter. They are not missing while the current interim recipe keeps both
filters disabled.

## Deliverables

- Tracking/tail representation specification
- Corrected frame-loss, synchronization, interpolation, and filtering
  specification with synthetic reference tests
- Tail representation artifact and schema
- Three frame-level metric implementations including the legacy benchmark
- Candidate parameter-set definitions
- Frame-activity artifact
- Synthetic reference test suite
- Coverage and quality report per metric

## Exit gate

Scientific gate P is approved. All feasible candidate metrics pass their
mathematical reference tests, share a traceable cleaned input, expose explicit
units and validity, and can be calculated independently for one fish or
recording. Infeasible candidates are closed with evidence rather than
approximated silently.

## Pilot progress

Implemented:

```text
e13e161 feat(metrics): add measured-tail candidate activity metrics
dd0cbda feat(comparison): report legacy versus candidate routes
3c52526 feat(analysis): add measured-time temporal profiles
```

Pilot `20221115_04`:

- 8,361,387 camera/tracking overlap frames;
- 8,361,386 valid measured-time derivatives;
- no cross-gap derivatives;
- measured local-bend/coordinate agreement coverage: 100%;
- mean absolute geometry error: approximately `1.9e-5` radians;
- terminal `angle15` confirmed as a zero placeholder;
- lossless float64 candidate artifact written locally.
- compact measured-time temporal profiles generated for all protocol events.

Second local fish `20221116_12` (2026-08-31):

- 8,428,449 overlap frames; 8,428,448 valid derivatives; 100% geometry;
- mean absolute geometry error approximately `2.2e-5` radians;
- movement-state, temporal outcomes v2 (155,700 rows), and trial outcomes
  (865 rows) built with the same recipes as the pilot;
- two-fish runner analysis `candidate-development-twofish-v1` completed.

Implemented candidates:

- tail-length-weighted angular L1 from measured segment orientations;
- tail-length-normalized whole-tail XY mean speed.

## Remaining work, in plain language

### 1. Freeze Gate P: the corrected-frame handling rule

The implementation already detects duplicate/missing frames, preserves the
measured timestamp, invalidates derivatives across a gap, translates the tail
base to zero, and reports protocol timing. What is still needed is an explicit
scientific decision that this conservative rule is final, or a specified and
validated alternative.

In particular, decide and record:

- whether interpolation remains disabled, and if not, which gaps may be
  filled and how;
- whether temporal or spatial filtering remains disabled, and if not, the
  method, window, and edge rule; and
- the maximum timing/frame gap that invalidates a derivative.

It is acceptable to finalize the current “do not interpolate or filter” policy.
Finishing Gate P does not require adding those operations.

### 2. Add the remaining mathematical known-answer checks

Add the five groups listed above under “Mathematical known-answer checks.”
They are small unit tests, not a request for more recordings or scientific
selection. They complete the evidence that the three formulas behave as their
definitions say they should.

### 3. Record smoothing correctly; do not build more smoothing machinery here

Smoothing is already implemented for the **movement/bout detector**, not for
the three frame-level activity metrics. The current detector uses a 10 ms
centered median smoother, and a sensitivity command already compares 0, 10,
and 20 ms without smoothing across tracking gaps.

The unresolved issue is a decision, not a missing feature: smoothing can remove
brief real movements, change the number/duration of detected bouts, and alter
the detector threshold needed to call movement. The local sensitivity result
showed those outputs change across settings. Therefore, do not treat 10 ms as
scientifically approved merely because it is the current default.

This decision belongs to Gate T1 / Step 07. Step 06 needs only to preserve the
setting and its results in artifacts; it does not need another smoothing
implementation.

### 4. Complete or explicitly waive the second-fixture legacy comparison

`20221115_04` has exercised both the legacy path and candidate path. For
`20221116_12`, the corrected candidate path, movement state, temporal outcomes,
and trial outcomes ran, but the historical stages 3/4/5 have not been run and
checked alongside it.

This is an engineering regression check, not a claim that old and corrected
metrics should have equal values. Its purpose is to confirm that both routes
can process the same second recording, preserve recording/trial identities,
and expose any recording-specific integration failure. If the project no
longer needs a second legacy-route regression fixture, record that it is waived
rather than leaving it as an ambiguous requirement.

Held-out metric selection is Step 07 work and is not required to finish the
Step 06 implementation.

Progress (2026-08-31): `corrected-preprocess-v1` writes measured-time
aligned frames with explicit validity/gap masks, base translation, and
protocol timing diagnostics. Interpolation/filtering remain disabled by
policy. `tail-candidate-corrected-v1` calculates the three-metric candidate set
from that artifact and intersects corrected derivative masks; intake-sourced
`tail-candidate-development-v1` is unchanged.

## Downstream invalidation

- Representation change: all dependent metrics
- One metric formula/parameter change: that metric only, then its detector,
  outcomes, comparisons, and figures
- Mechanistic branch change: that branch only
- Legacy benchmark change: legacy equivalence must be reopened
