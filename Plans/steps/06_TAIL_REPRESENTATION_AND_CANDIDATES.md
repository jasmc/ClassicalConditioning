# Step 06 — Corrected Preprocessing, Tail Representation, and Candidate Metrics

**Status:** In progress  
**Change class:** Scientific-correction candidate development  
**Depends on:** Steps 04-05 and scientific gate T0  
**Unlocks:** Metric validation and selection

## Objective

Develop and freeze corrected frame/timing preprocessing, create one validated
representation of the cleaned tail, and calculate all candidate activity
metrics side by side while retaining legacy point-15 vigor.

This step creates candidates; it does not select the paper metric.

## Candidate metric identities

| Metric ID | Definition | Indicative unit |
| --- | --- | --- |
| `legacy_distal_angular_speed` | Absolute speed of current distal cumulative angle | deg/ms |
| `segment_absolute_angular_speed_sum` | Manuscript-described sum of absolute segment angular speeds | deg/ms |
| `all_point_angular_rms` | Tail-length-weighted RMS angular speed across points | deg/ms |
| `whole_tail_xy_rms_speed` | Tail-length-weighted RMS 2D point speed | normalized length/s or calibrated distance/s |
| `whole_tail_xy_mean_speed` | Tail-length-weighted mean 2D point speed | normalized length/s or calibrated distance/s |
| `curvature_rate_rms` | Tail-length-weighted RMS curvature-change rate | declared curvature/time unit |

Each is a distinct scientific quantity, not merely a version of `vigor`.
The manuscript-described segment-speed sum is a required reconciliation
benchmark in addition to the four tail-dynamics candidates and the legacy
point-15 benchmark. Its feasibility and exact interpretation depend on gate T0
confirming whether source angles are local segment angles.

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

### Angular and curvature representation

Retain:

- source local angle;
- cumulative orientation;
- local curvature/bend;
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

Also retain the separately specified manuscript-reconciliation benchmark. Do
not relabel all-point RMS or distal-point speed as the manuscript sum.

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

## Synthetic reference tests

- Stationary straight tail
- Stationary bent tail
- Rigid translation
- Rigid rotation
- One moving segment
- Coordinated C-bend
- S-bend with opposing segment motion
- Base-to-tip traveling bend
- Single-point spike
- Missing distal points
- Different point densities
- Different segment spacings
- Different frame rates and jitter
- Long missing gap

Expected direction and, where possible, exact numerical result must be written
before inspecting candidate outputs.

## Deliverables

- Tracking/tail representation specification
- Corrected frame-loss, synchronization, interpolation, and filtering
  specification with synthetic reference tests
- Tail representation artifact and schema
- Six frame-level metric implementations including the legacy and
  manuscript-reconciliation benchmarks
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

- manuscript segment absolute angular-speed sum from measured segment
  orientations;
- tail-length-weighted angular RMS;
- tail-length-weighted whole-tail XY RMS speed;
- tail-length-weighted whole-tail XY mean speed;
- curvature-change RMS using `d(bend/length)/dt`.

Still required (implementation; fixtures debug only):

- wire movement/temporal/runner onto corrected-source candidates (optional);
- Gate P policies for interpolation and temporal/spatial filtering;
- approved smoothing candidates and validation machinery;
- video/manual bout comparison where feasible (deferred tooling);
- held-out metric selection plumbing (not runnable as science on two fixtures);
- legacy stage-3/4/5 + legacy-candidate outcome comparison for fixture
  `20221116_12` so both fixtures exercise that path.

Progress (2026-08-31): `corrected-preprocess-v1` writes measured-time
aligned frames with explicit validity/gap masks, base translation, and
protocol timing diagnostics. Interpolation/filtering remain disabled by
policy. `tail-candidate-corrected-v1` calculates the five candidate metrics
from that artifact and intersects corrected derivative masks; intake-sourced
`tail-candidate-development-v1` is unchanged.

## Downstream invalidation

- Representation change: all dependent metrics
- One metric formula/parameter change: that metric only, then its detector,
  outcomes, comparisons, and figures
- Mechanistic branch change: that branch only
- Legacy benchmark change: legacy equivalence must be reopened
