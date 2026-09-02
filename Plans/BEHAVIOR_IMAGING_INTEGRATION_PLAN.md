# Behavior and Imaging Integration Plan

**Status:** Deferred - design complete; implementation not authorized  
**Change class:** Behavior-preserving integration first; scientific changes only
after separate approval  
**Primary purpose:** Integrate optional two-photon imaging with the canonical
behavior analysis without creating a second behavior pipeline  
**Implementation dependencies:** I00 may inventory sources independently; I01+
depend on existing migration Steps 02-05, approved behavior recipes, and
canonical behavior identities  
**Does not block:** Completion of the behavior-only migration and releases  

## 1. Authority and relationship to the main migration

This plan defines a later, parallel implementation track for experiments that
contain both behavior and imaging. It extends, but does not replace:

- [the master analysis migration plan](./MASTER_ANALYSIS_MIGRATION_PLAN.md);
- [the final analysis architecture](./ANALYSIS_ARCHITECTURE.md);
- [the artifact and provenance plan](./steps/02_ARTIFACTS_SCHEMAS_AND_PROVENANCE.md);
- [the domain and configuration plan](./Archive/steps/03_DOMAIN_AND_CONFIGURATION.md);
- [the ingestion and validation plan](./Archive/steps/04_INGESTION_AND_RAW_VALIDATION.md).

The behavior-only pipeline remains authoritative for behavior. Imaging
availability must never select a different behavior formula, preprocessing
path, movement detector, outcome definition, cohort rule, or learner method.

The source imaging implementation is in the sibling `analysis_learning`
repository. It is evidence for the intended workflow, not a specification to
copy literally. Its risks and migration disposition are recorded in
[the imaging pipeline critique](../docs/analysis/IMAGING_PIPELINE_CRITIQUE.md).

No implementation status in the main step index changes merely because this
plan exists.

## 2. Non-negotiable scientific and engineering invariants

### 2.1 Behavior-modality independence

For the same raw behavior sources and the same behavior recipe:

```text
behavior_artifacts(with_imaging = false)
    == behavior_artifacts(with_imaging = true)
```

Equality means the same schema version, logical scientific content, missingness,
units, identities, and canonical logical-content hash. Byte equality is required
only where the existing artifact contract promises deterministic bytes.

The imaging branch may read canonical behavior clocks, stimulus events, trial
identities, and approved behavior outcomes. It may not:

- preprocess tail data independently;
- recalculate vigor with imaging-specific helpers;
- reconstruct a separate trial map;
- alter exclusions in behavior-only analyses;
- convert behavior missingness to zero;
- infer protocol semantics from a fish name or directory name;
- write back into canonical behavior artifacts.

### 2.2 Optional modality

Behavior is required for the analyses covered by this repository. Imaging is an
optional recording capability. A missing imaging modality is a valid recording
state, not an error, when the resolved acquisition specification declares a
behavior-only recording.

The following values are distinct:

```text
imaging not acquired
imaging expected but source missing
imaging acquired but intake invalid
imaging intake valid but synchronization invalid
imaging valid but excluded by imaging QC
imaging valid and analyzed
```

They must not collapse into a single null, false, or zero value.

### 2.3 Additive integration

Imaging is added through new artifact types, schemas, stages, recipe IDs, and
QC fields. Existing frozen behavior recipes are not widened or silently
redefined.

### 2.4 Explicit scientific alternatives

The following are separate methods, not interchangeable software versions:

- legacy-equivalent versus corrected behavior;
- pixel response maps versus ROI response tables;
- Suite2p-detected cells versus correlation-grown ROIs;
- within-trial registration versus between-trial/plane alignment;
- behavior-primary versus imaging-eligible cohorts.

Every alternative requires an explicit method ID and scientific status.

### 2.5 Local, immutable, reproducible execution

All source reading, motion correction, feature extraction, statistics, figure
rendering, and artifact storage run locally. Raw sources remain immutable.
New stages use the common transaction, schema, lineage, and no-silent-overwrite
contracts from the main migration.

## 3. Intended outcomes

When complete, the integration track must provide:

1. the same canonical behavior outputs for all experiments;
2. explicit detection and validation of optional imaging sources;
3. an auditable behavior-to-imaging frame synchronization map;
4. versioned motion-corrected image artifacts and frame-quality masks;
5. versioned pixel and ROI response products;
6. a canonical trial-level multimodal table;
7. distinct behavior-primary, imaging-eligible, and sensitivity cohorts;
8. behavior-only, imaging-only, and cross-modal statistical routes;
9. figures generated from frozen panel-data artifacts;
10. an immutable multimodal release that can be reproduced locally.

## 4. Non-goals

This track does not:

- approve a corrected behavior metric;
- choose a learner classifier;
- declare either inherited ROI method scientifically valid;
- copy pickle object graphs or mutable globals into the package;
- use imaging availability as a biological inclusion criterion for the primary
  behavior result;
- infer missing acquisition metadata from file paths in production;
- make notebooks or numbered scripts canonical orchestration;
- require every historical experiment to contain imaging;
- treat registration success as proof that fluorescence measurements are
  biologically valid.

## 5. Source workflow to be migrated

The `analysis_learning` repository contains the following conceptual stages:

```text
camera + protocol + tail + galvo + functional TIFF + anatomy
    -> behavior/imaging join
    -> Suite2p motion correction
    -> frame rejection and templates
    -> pixel response maps
    -> optional HDF5 export
    -> Suite2p cell detection OR correlation-grown ROI analysis
```

Useful concepts to preserve are:

- fish -> plane -> trial acquisition structure;
- galvo-assisted image timing;
- within-trial and cross-trial registration;
- explicit bad-frame masks;
- stimulus-period response maps;
- optional pixel-level and ROI-level routes.

The following implementation mechanisms must be replaced:

- global path modules and wildcard-imported constants;
- `exec(open(...).read())` script chaining;
- experiment detection from directory names;
- hard-coded plane/trial maps;
- summer-time and event timing inferred from fish names;
- raw behavior rereading inside the imaging join;
- mutable pickled object graphs;
- recursive `item_N` HDF5 serialization;
- broad exception handling that silently skips trials;
- inconsistent file names, field case, and image dimension order.

## 6. Final dependency architecture

The complete architecture is diagrammed in
[the central architecture plan](./ANALYSIS_ARCHITECTURE.md). The governing
dependency direction is:

```text
canonical raw behavior artifacts
    -> canonical behavior analysis --------------------------+
    -> behavior/imaging synchronization                      |
                                                               +-> multimodal join
canonical raw imaging artifacts -> imaging analysis ---------+
```

The synchronization stage depends on raw clocks and canonical stimulus/trial
identity. It does not depend on a particular vigor or movement metric.

The multimodal join depends on completed behavior and imaging outcomes. It
cannot mutate either parent.

## 7. Canonical identity model

### 7.1 Stable biological and acquisition identities

The existing Step 03 identity audit remains authoritative. Imaging adds:

```text
acquisition_id
imaging_run_id
plane_id
imaging_frame_id
roi_id
```

Required relationships:

```text
experiment_id
  -> recording_id
      -> fish_id
      -> acquisition_id
          -> imaging_run_id
              -> plane_id
                  -> imaging_frame_id
```

`trial_id` is the stable technical identity assigned by the canonical trial map,
not a plane position or list index. `trial_number` is the experiment-facing
ordinal and does not replace `trial_id`. `event_id` identifies the canonical
alignment event for that trial. `roi_id` is unique within an ROI-set artifact,
not necessarily across recipes.

### 7.2 Join identity

The authoritative trial-outcome key is:

```text
recording_id
trial_id
alignment
```

`experiment_id`, `fish_id`, `trial_number`, and `event_id` are required
consistency/provenance fields on both modality outcomes and must agree before a
join is accepted. They are not fallback keys. `plane_id` and `roi_id` extend the
authoritative key for normalized plane- and ROI-level child tables. Condition,
phase, and learner label are metadata and must agree with their authoritative
manifests.

### 7.3 Time coordinates

Keep these coordinates distinct:

```text
source_absolute_time
source_elapsed_time
behavior_time
imaging_time
stimulus_relative_time
concatenated_analysis_time
```

Concatenated analysis time is derived and must never replace source time.
All time fields declare units, origin, transformation, and uncertainty.

## 8. Capability and configuration model

### 8.1 Recording capabilities

Add an immutable acquisition capability record:

```text
recording_id
behavior_expected
imaging_expected
functional_imaging_expected
anatomy_expected
galvo_sync_expected
imaging_mode
expected_plane_count
expected_trial_count
source_definition
```

Recommended finite `imaging_mode` values:

```text
none
single_plane
multi_plane
unknown_requires_review
```

Capabilities come from explicit experiment/acquisition metadata. Source
discovery confirms them; it does not define them silently.

### 8.2 New resolved configuration types

```text
ImagingAcquisitionConfig
ImagingReaderConfig
ImagingSynchronizationConfig
MotionCorrectionConfig
ImagingFrameQCConfig
PixelResponseConfig
ROIDetectionConfig
ROIResponseConfig
MultimodalJoinConfig
MultimodalCohortConfig
MultimodalStatisticsConfig
```

Each stage declares only the configuration subset that affects it. For example,
a response-map colormap cannot invalidate registration, and changing the
behavior metric cannot invalidate raw imaging intake.

### 8.3 Acquisition order

Plane and trial acquisition order must be supplied as data:

```text
imaging_run_id
source_sequence_index
plane_id
trial_id
expected_start_event_id
repeat_index
```

Path-string branches and fish-specific Python ranges are prohibited in the
canonical route.

## 9. Artifact and schema plan

### 9.1 New artifact types

Add these initial schema families to the Step 02 registry:

```text
imaging-source-inventory/1.0
raw-galvo/1.0
raw-imaging-metadata/1.0
raw-imaging-array/1.0
raw-anatomy-array/1.0
acquisition-order/1.0
behavior-imaging-sync/1.0
motion-correction-shifts/1.0
registered-imaging-array/1.0
imaging-frame-qc/1.0
registration-template/1.0
imaging-plane-position/1.0
pixel-response-map/1.0
roi-set/1.0
roi-fluorescence/1.0
roi-response/1.0
imaging-trial-outcomes/1.0
multimodal-trial-outcomes/1.0
multimodal-cohort-manifest/1.0
multimodal-statistical-results/1.0
multimodal-panel-data/1.0
```

### 9.2 Dense-array policy

Dense arrays need an explicit benchmark under Step 02. The accepted format must
support:

- lossless exact round-trip;
- chunked plane/time access;
- stable dimension names and order;
- coordinates and units;
- compression metadata;
- source and logical-content hashes;
- read-only loading where practical;
- interoperability with NumPy, xarray, and Suite2p adapters.

Candidate formats may include HDF5 and Zarr, but the decision must be based on a
local benchmark. Recursive serialization of arbitrary Python objects is not an
artifact schema.

Required dimension conventions:

```text
functional imaging: time, y, x
anatomy: plane, y, x
ROI fluorescence: imaging_frame_id, roi_id
response maps: plane, response_window, y, x
```

Adapters must transpose source arrays explicitly and record the source order.

### 9.3 Artifact provenance

Imaging metadata extends the common `ArtifactMetadata` contract with:

```text
source image hashes
source galvo hash
acquisition-order artifact ID
synchronization artifact ID
Suite2p version and ops hash
registration recipe ID
frame-QC recipe ID
ROI method and parameter-set ID
array shape, dtype, dimension order, and chunking
excluded frame/trial/plane counts and reasons
```

Third-party library defaults that affect scientific output are resolved into
the recipe rather than omitted from provenance.

## 10. Stage contracts

### I00 - Governance, source inventory, and frozen reference cases

**Purpose:** Establish what imaging data exists and freeze representative source
and legacy outputs before porting any algorithm.

#### Work

1. Inventory all imaging-enabled experiments and behavior-only experiments.
2. Record source types, rigs, acquisition software versions, frame rates, plane
   counts, repeats, and known exceptions.
3. Select representative frozen cases:
   - behavior-only;
   - single-plane imaging;
   - multi-plane imaging;
   - missing or corrupt galvo;
   - dropped/bad image frames;
   - incomplete anatomy;
   - at least one historical special-case acquisition.
4. Run recoverable legacy imaging stages and hash their inputs and outputs.
5. Save review copies of trial maps, frame counts, templates, response maps, and
   ROI outputs.
6. Record stages that cannot run as committed rather than repairing them inside
   the reference run.

#### Deliverables

- imaging source inventory;
- acquisition capability table;
- frozen fixture manifest;
- legacy execution report;
- known-unrecoverable behavior report.

#### Exit gate I0

Representative source families and known exceptions are explicitly covered.
Every reference artifact is tied to immutable source hashes, or its absence is
recorded without inference.

### I01 - Shared contracts and package boundaries

**Depends on:** Main Steps 02-03 and Gate I0.

**Purpose:** Add imaging concepts without changing behavior contracts.

#### Work

1. Extend the schema registry and artifact repository.
2. Add acquisition capabilities and configuration types.
3. Define imaging and multimodal recipe identity.
4. Define dense-array format and dimension convention.
5. Define package namespaces:

```text
classical_conditioning/
    imaging/
        intake.py
        synchronization.py
        registration.py
        frame_qc.py
        response_maps.py
        rois/
        outcomes.py
    multimodal/
        join.py
        cohort.py
        statistics.py
```

6. Add stage registry entries without adding imaging imports to behavior-only
   stage modules.

#### Exit gate I1

A synthetic behavior-only recording and synthetic multimodal recording resolve
valid configurations and artifact plans. The behavior-only plan has no imaging
stage dependencies.

### I02 - Imaging source intake and validation

**Depends on:** I01.

**Purpose:** Convert imaging sources into immutable, explicit artifacts without
scientific transformation.

#### Inputs

- functional TIFF or equivalent image source;
- galvo signal;
- anatomy stack;
- acquisition metadata;
- declared acquisition-order table.

#### Work

1. Match source files to declared recording and imaging-run identities.
2. Read metadata without loading full arrays where possible.
3. Validate image dtype, shape, page/frame count, and readability.
4. Validate galvo schema, sampling order, finite values, and time coverage.
5. Validate anatomy dimensions and plane count.
6. Reject ambiguous multiple matches.
7. Publish source inventory, raw metadata, and canonical dense-array references
   transactionally.
8. Report expected-but-missing and unexpected-present sources separately.

#### Exit gate I2

Every source is either uniquely assigned and authenticated or has a specific
failure status. No path naming convention silently determines experiment,
plane, trial, or modality.

### I03 - Behavior/imaging synchronization

**Depends on:** Main behavior intake and trial map, I02.

**Purpose:** Produce a reusable mapping between behavior, protocol, galvo, and
image frames before motion or response analysis.

#### Required output columns

```text
recording_id
imaging_run_id
plane_id
trial_id
imaging_frame_id
behavior_frame_id
event_id
source_behavior_time_ms
source_imaging_time_ms
stimulus_relative_time_ms
match_status
timing_residual_ms
timing_uncertainty_ms
timestamp_origin
timestamp_synthesized
```

#### Algorithm requirements

1. Detect galvo/image events under an explicit, versioned configuration.
2. Match clocks in both start-order cases:
   - imaging starts before behavior;
   - behavior starts before imaging.
3. Model offset and drift when supported by synchronization evidence.
4. Preserve missing frames and unmatched observations.
5. Permit synthesized timestamps only under an approved recovery rule.
6. Mark every synthesized value and retain uncertainty.
7. Map trials from canonical event IDs and acquisition-order metadata.
8. Never infer US time from a fish identifier.

#### QC

- monotonicity;
- one-to-one and one-to-many cardinality expectations;
- unmatched fraction;
- residual distribution;
- drift;
- expected versus observed frames per trial and plane;
- event-window coverage;
- cross-check against manually reviewed landmarks.

#### Exit gate I3

All frames have an explicit matched, unmatched, or synthesized status. Residual
and coverage thresholds are approved, tested on both clock-start cases, and
passed by the reference fixtures.

### I04 - Registration and motion correction

**Depends on:** I02-I03.

**Purpose:** Wrap Suite2p or another approved engine behind a deterministic
stage contract.

#### Separate transformations

Keep these products separate:

1. within-trial registration;
2. within-plane cross-trial alignment;
3. optional cross-plane/anatomy positioning;
4. final registered data selected for downstream analysis.

#### Work

1. Resolve and serialize every scientifically relevant Suite2p option.
2. Record Suite2p and dependency versions.
3. publish shifts, templates, correlation diagnostics, and registered arrays;
4. define whether background subtraction occurs before registration;
5. preserve raw fluorescence in separate immutable artifacts;
6. explicitly name the registered artifact consumed downstream;
7. support chunked processing and bounded memory;
8. make reruns idempotent and no-overwrite by recipe identity.

#### Exit gate I4

Registration is reproducible from authenticated sources. The final aligned
artifact, rather than an earlier intermediate, is verified as the input to
response and ROI analysis.

### I05 - Imaging frame quality

**Depends on:** I04.

**Purpose:** Separate registration from the scientific decision about usable
frames.

#### Candidate diagnostics

- Suite2p bad-frame flags;
- template correlation;
- displacement magnitude;
- intensity and saturation;
- non-finite pixels;
- edge loss after shifts;
- missing/synthesized synchronization;
- per-trial usable-frame coverage.

The inherited `mean - 2.5 SD` correlation threshold and `th_badframes = 0.7`
remain legacy candidates until validated. They are not adopted as defaults by
copying code.

#### Exit gate I5

A versioned frame-QC mask explains every invalid frame and does not encode
invalid as low activity. Thresholds have positive controls and manual review on
representative recordings.

### I06 - Pixel response maps

**Depends on:** I03-I05 and canonical stimulus windows.

**Purpose:** Produce inspectable trial/plane response maps from valid registered
frames.

#### Work

1. Define pre-CS, CS-US, post-US, and other windows through canonical event
   metadata.
2. Require adequate valid-frame coverage per window.
3. Save raw window means separately from normalized contrasts.
4. Treat the inherited formula

```text
(stimulus_mean - pre_mean) / (pre_mean + soft_threshold)
```

as one named candidate response metric.
5. Validate the soft threshold and Gaussian smoothing independently.
6. Declare whether smoothing is analysis or display-only.
7. Save unsmoothed scientific maps when display smoothing is used.

#### Exit gate I6

Synthetic arrays recover known responses. Reference cases have correct event
windows, valid-frame counts, units, and parity reports against recoverable
legacy outputs.

### I07 - ROI alternatives

**Depends on:** I04-I05; optionally I06.

**Purpose:** Implement ROI methods as distinct, auditable scientific routes.

#### Route A: Suite2p cells

```text
roi_method_id = suite2p-cells-v1
```

Products include ROI geometry, `iscell`, fluorescence, neuropil, optional
deconvolution, and method-specific QC.

#### Route B: Correlation-grown ROIs

```text
roi_method_id = correlation-grown-v1
```

Products include seeds, growth history or sufficient diagnostics, pixel masks,
fluorescence, regressor correlations, and threshold provenance.

#### Shared requirements

- no hard-coded `fs = 30` unless declared and validated for the recording;
- stable `roi_id`;
- separate detection, trace extraction, classification, and response stages;
- no overwriting one route with another;
- no claim that correlation with a task regressor is independent evidence if
  the same data selected the ROI;
- neuropil/background policy declared explicitly;
- manual QC artifacts blinded where practical.

#### Exit gate I7

Each retained ROI method passes synthetic and reference tests and has a
scientific interpretation. If neither method passes, pixel maps may remain the
only approved imaging product.

### I08 - Behavior equivalence under imaging integration

**Depends on:** Main behavior legacy-equivalence stage, I01-I03.

**Purpose:** Prove that adding imaging did not fork behavior.

#### Work

1. Run each selected multimodal fixture through behavior-only orchestration.
2. Run the same fixture through multimodal orchestration.
3. Compare every behavior artifact:
   - schema and recipe;
   - row and trial identity;
   - values and missingness;
   - units and categories;
   - logical-content hash;
   - cohort eligibility.
4. Test both legacy-equivalent and any approved corrected behavior recipe.
5. Fail the multimodal run if an imaging stage attempts to publish behavior
   artifact types.

#### Exit gate I8

All canonical behavior logical hashes are identical between behavior-only and
multimodal execution for the same sources and recipe.

### I09 - Imaging outcomes and multimodal integration

**Depends on:** Main behavior outcomes, I03, and at least one approved imaging
outcome route from I06 or I07; Gate I8.

**Purpose:** Join completed modality products without recalculating them.

#### Imaging trial outcomes

Potential outputs, subject to scientific approval:

- mean response by response window and plane;
- active-pixel area;
- ROI response magnitude;
- responsive-ROI count or fraction;
- trial-to-trial response stability;
- anatomy-relative position;
- imaging coverage and QC measures.

#### Join rule

Behavior trial outcomes are the left/primary table:

```text
behavior_trial_outcomes
LEFT JOIN imaging_trial_outcomes
USING (recording_id, trial_id, alignment)
```

Behavior-only trials remain present. Imaging fields carry explicit availability
and QC status; unavailable values remain missing, never zero. Before joining,
the stage asserts exact agreement of `experiment_id`, `fish_id`, `trial_number`,
and `event_id` for every matched key.

#### Cardinality

Plane- or ROI-level outcomes must either:

- remain in normalized child tables; or
- be aggregated under an explicit, versioned formula before trial-level join.

Accidental one-to-many duplication of behavior outcomes is a hard error.

#### Exit gate I9

Join completeness and cardinality reports pass. Every multimodal value traces
to one behavior outcome artifact, one imaging outcome artifact, one sync
artifact, and one configuration hash.

### I10 - Cohorts, statistics, and sensitivity

**Depends on:** Main Steps 08-10 and I09.

#### Cohort hierarchy

Maintain separate immutable manifests:

1. **Behavior-primary cohort** - determined without imaging availability.
2. **Imaging-acquired cohort** - records declared to include imaging.
3. **Imaging-valid cohort** - imaging-acquired records passing sync,
   registration, frame, and outcome QC.
4. **Multimodal analysis cohort** - imaging-valid records with required
   behavior outcomes.
5. **Sensitivity cohorts** - alternative defensible imaging QC thresholds.

The paper's primary behavior inference uses the behavior-primary cohort unless
an approved estimand explicitly says otherwise.

#### Statistical routes

Keep separate result families:

- behavior-primary inference;
- imaging response inference;
- behavior-imaging association;
- trial-level longitudinal or repeated-measures models;
- learner-stratified imaging, only if the learner plan later passes its own
  non-circularity gate;
- exploratory spatial, ROI, or mechanistic analyses.

Cross-modal models must account for fish, trial, plane, and ROI dependence as
appropriate. Treating frames, pixels, or ROIs as independent animals is
prohibited.

#### Exit gate I10

Estimands, populations, hierarchy, multiplicity, missingness, and sensitivity
rules are predeclared. Model diagnostics and effective sample sizes are saved.

### I11 - Figures, interfaces, and reproducibility

**Depends on:** I06-I10 as applicable.

#### Work

1. Extend the common panel-data contract.
2. Add behavior-imaging synchronization QC figures.
3. Add registration and frame-QC review figures.
4. Add pixel/ROI response figures.
5. Add multimodal trial and fish-level association figures.
6. Reuse the existing publication, PNG, and local interactive modes.
7. Ensure all modes consume identical panel data.
8. Add CLI planning and execution selectors:

```text
--recording
--with-imaging
--imaging-recipe
--roi-method
--dry-run
--explain
```

`--with-imaging` enables declared optional stages; it does not alter the
behavior recipe.

#### Exit gate I11

Figures have semantic IDs, source artifacts, reproduction commands, and the
same numerical panel data across rendering modes.

### I12 - Multimodal release and legacy retirement

**Depends on:** All retained integration stages.

#### Release contents

```text
release.json
source inventories and hashes
resolved behavior and imaging configs
behavior, imaging, and multimodal recipe IDs
schema registry snapshot
synchronization QC
registration and frame-QC summaries
behavior-primary and multimodal cohort manifests
canonical outcome and statistical artifacts
panel data and figures
environment report
known limitations
reproduction command
```

The release must reproduce from immutable sources without executing scripts
from `analysis_learning`.

#### Retirement rule

Legacy imaging scripts become read-only references only after:

- source and output inventories are preserved;
- each retained algorithm has a package replacement;
- parity differences are explained;
- final registered data is demonstrably consumed downstream;
- selected ROI routes have explicit scientific status;
- a clean local reproduction succeeds.

## 11. Migration disposition by inherited component

| Inherited component | Disposition | Reason |
| --- | --- | --- |
| Fish/plane/trial conceptual hierarchy | Adapt | Useful domain structure, but replace mutable object graph |
| Raw source discovery | Rewrite | Global paths and implicit folder conventions are unsafe |
| Camera/tail/protocol readers | Replace with canonical behavior intake | Behavior must have one source of truth |
| Galvo peak extraction | Reimplement and validate | Scientifically useful, but thresholds and recovery need contracts |
| Old trial protocol slicing | Adapt | More complete than the newer join, but identity must come from canonical events |
| New join object construction | Retire | Approximate protocol replication and incomplete behavior slicing |
| Plane/trial path-string maps | Rewrite as data | Silent dataset-specific branching is not reproducible |
| Suite2p motion correction | Wrap and validate | Retain engine; replace orchestration and provenance |
| Correlation frame rejection | Candidate only | Threshold requires validation and sensitivity |
| Pixel response maps | Reimplement as named metrics | Windows, soft threshold, and smoothing need separation |
| Activity map rendering | Rewrite on panel-data contract | Desktop paths and analysis/display coupling |
| Recursive HDF5 serializer | Retire | Not a stable schema |
| Suite2p ROI detection | Reimplement as explicit route | Valid candidate but hard-coded assumptions |
| Correlation-grown ROI method | Reimplement as experimental route | Heuristic and selection-bias risks need explicit status |
| Pickled `Data/Plane/Trial` artifacts | Retire after conversion | Unsafe and schema-implicit |
| Script chaining | Retire | Broken names, hidden state, no stage contracts |

## 12. Validation strategy

### 12.1 Unit and property tests

- source readers reject malformed and ambiguous inputs;
- array dimension adapters preserve exact values;
- synchronization handles both clock start orders;
- drift and offset recovery on synthetic clocks;
- missing/synthesized frame states remain explicit;
- registration adapter consumes the intended final product;
- invalid frames never become zero fluorescence or rest;
- response windows are left/right bounded as specified;
- multimodal joins reject duplicate expansion;
- behavior logical hashes are invariant to imaging enablement.

### 12.2 Golden fixtures

Freeze small, legally and scientifically appropriate subsets for:

- behavior-only;
- single-plane;
- multi-plane;
- missing galvo;
- dropped frames;
- registration failure;
- no valid pre-stimulus frames;
- no detected ROIs;
- multiple ROI routes.

Fixtures must be small enough for normal automated tests. Full recordings remain
external and are used for authenticated local validation.

### 12.3 Legacy parity

Parity is descriptive, not automatic approval. Compare:

- frame/trial/plane counts;
- synchronization offsets;
- templates and shifts;
- good/bad frame masks;
- window means;
- response maps;
- ROI masks and traces;
- trial outcomes.

Classify differences as:

```text
exactly equivalent
numerically equivalent within declared tolerance
intentional correction
source legacy defect
unresolved
```

### 12.4 Scientific validation

- manual synchronization landmark review;
- registration overlay review;
- known-motion and known-response synthetic controls;
- response-window negative controls;
- ROI stability and split-half checks;
- sensitivity to frame-QC thresholds;
- sensitivity to response normalization and smoothing;
- fish-level, not pixel-level, biological replication checks.

### 12.5 End-to-end acceptance

One behavior-only and one multimodal recording must run from immutable sources
to figures under one CLI. The behavior portion of the multimodal run must match
the standalone behavior run logically.

## 13. Failure policy

Stages fail with a specific typed error or publish an explicit reviewed failure
artifact where the workflow requires batch continuation. They do not silently
continue.

Examples:

| Condition | Required behavior |
| --- | --- |
| Imaging not declared | Skip imaging as planned; behavior proceeds |
| Imaging declared but source missing | Fail imaging intake; behavior may proceed with recorded modality failure |
| Ambiguous source match | Fail intake |
| Synchronization residual too high | Block imaging descendants |
| One trial lacks coverage | Mark trial-specific imaging failure; do not drop behavior trial |
| Registration fails one plane | Mark plane failure; apply declared batch policy |
| No valid ROI | Publish valid empty ROI artifact if scientifically meaningful |
| Multimodal join duplicates behavior rows | Fail join |
| Imaging stage changes behavior hash | Fail integration acceptance |

## 14. Execution and storage layout

The artifact manifest remains the authority. A human-readable local layout may
include:

```text
Paper data/
    Raw single fish data/
    Processed data/<recording-id>/
        behavior/
        imaging/
            intake/
            synchronization/
            registration/
            frame_qc/
            responses/
            rois/
        multimodal/
    Quality checks/<recording-id>/
        behavior/
        imaging/
        multimodal/
    Tables/
    Models/
    Figures/
    Metadata/
```

Directory names do not define artifact identity, scientific recipe, or current
version.

## 15. Deferred sequencing and prerequisites

### 15.1 Work that may be prepared now

- preserve the source repository and commit ID;
- inventory imaging-enabled experiments;
- identify acquisition documentation;
- select representative recordings;
- capture legacy outputs that are still runnable;
- add open questions to scientific decision records.

### 15.2 Minimum prerequisites before package implementation starts

These prerequisites apply to I01 and later. They do not prevent the I00 source
inventory and reference-preservation work.

1. Step 02 shared schema, artifact metadata, logical hashing, and resolver are
   accepted.
2. Step 03 recording identity, trial map, and resolved configuration are
   accepted for a pilot imaging experiment.
3. Canonical behavior intake and the frozen legacy behavior recipe are stable.
4. A scientific owner confirms which imaging products are required for the
   final analysis.
5. Source acquisition order and clock semantics are available or explicitly
   declared unrecoverable.
6. Representative local source data is accessible for validation.

### 15.3 Recommended implementation waves

| Wave | Scope | Can proceed when |
| --- | --- | --- |
| A | I00 inventory and reference preservation | Approved to inspect local sources |
| B | I01 contracts and I02 intake | Main Steps 02-03 pass relevant gates |
| C | I03 synchronization | Canonical trial map and clock evidence exist |
| D | I04-I05 registration and frame QC | Intake and synchronization fixtures pass |
| E | I06-I07 pixel and ROI routes | Final registered artifact and QC are stable |
| F | I08-I09 equivalence and integration | Canonical behavior outcomes are stable |
| G | I10-I11 inference and figures | Scientific outcomes and cohorts are approved |
| H | I12 release and retirement | All retained routes pass acceptance |

Waves B-E can be developed against synthetic and frozen pilot artifacts while
the final corrected behavior recipe is unresolved. They cannot publish a final
multimodal paper release until the relevant behavior and scientific gates pass.

## 16. Scientific decision gates

| Gate | Required decision | Blocks |
| --- | --- | --- |
| I0 | Imaging experiment/source inventory and representative fixtures | All imaging implementation |
| I1 | Capability, identity, schema, array format, and dimension contracts | Imaging intake |
| I2 | Clock semantics, galvo detection, offset/drift, recovery, and residual thresholds | Registered trial mapping and response windows |
| I3 | Registration stages, background policy, Suite2p parameters, and final downstream artifact | Response and ROI routes |
| I4 | Frame-quality diagnostics, thresholds, and missing-frame policy | Imaging outcomes |
| I5 | Response windows, normalization, soft threshold, smoothing, and coverage rules | Pixel response claims |
| I6 | Retained ROI methods, neuropil policy, QC, and interpretation | ROI claims |
| I7 | Imaging and multimodal cohorts, estimands, hierarchy, and missingness | Multimodal inference |
| I8 | Figure semantics and release contents | Final multimodal release |

## 17. Definition of complete

The integration track is complete only when:

- behavior-only and imaging-enabled experiments use the same behavior pipeline;
- imaging capability is explicit per recording;
- raw imaging, galvo, anatomy, and acquisition order are authenticated;
- synchronization is a reusable versioned artifact with uncertainty and QC;
- registration stages and their consumed final output are unambiguous;
- invalid frames and unavailable imaging remain distinct from biological zero;
- pixel and retained ROI routes have separate recipe identities;
- behavior artifacts are logically identical with and without imaging;
- trial-level integration has validated keys and cardinality;
- cohort manifests distinguish behavior-primary and imaging-valid populations;
- statistical units respect fish/trial/plane/ROI dependence;
- figures derive from frozen panel data;
- one local command reproduces an immutable multimodal release;
- inherited scripts are no longer required for reproduction.
