# Critique of the Inherited Imaging Pipeline

## Scope

This document reviews the imaging implementation in the sibling
`analysis_learning` repository as a potential source for later integration into
`ClassicalConditioning`.

The review is based on source inspection of `analysis_learning` commit
`dff4603a3595ccc57927ecc4d1d4950af1de5336` on the `main` branch. It does not
quantify the effect of an issue on experimental results, certify biological
conclusions, or substitute for reanalysis of the raw imaging data.

The target integration design is documented in
[the behavior and imaging integration plan](../../Plans/BEHAVIOR_IMAGING_INTEGRATION_PLAN.md).

## Evidence boundary

The main files inspected were:

- [the older join](<../../../analysis_learning/1. Join_all_data.py>);
- [the newer join](<../../../analysis_learning/1. Join_all_data_new.py>);
- [the newer imaging helpers](<../../../analysis_learning/my_functions_imaging_new.py>);
- [the newer motion-correction stage](<../../../analysis_learning/2. Motion_correction_Suite2p_new.py>);
- [the newer pixel-response stage](<../../../analysis_learning/3. Analysis_of_imaging_data_pixels_Suite2p_new.py>);
- [the newer activity-map stage](<../../../analysis_learning/4. Activity_maps_new.py>);
- [the newer HDF5 exporter](<../../../analysis_learning/5. Save_data_as_HDF5_new.py>);
- [the newer Suite2p cell-detection route](<../../../analysis_learning/5. Cell detection_new.py>);
- [the newer correlation-based ROI route](<../../../analysis_learning/6. ROI_analysis_new.py>);
- [the imaging object model](<../../../analysis_learning/my_classes_new.py>);
- [the behavior helpers used in that repository](<../../../analysis_learning/my_functions_behavior.py>).

Line references below describe commit
`dff4603a3595ccc57927ecc4d1d4950af1de5336`. The sibling links open the current
working tree for convenience; use the pinned commit when reproducing the audit
after the source repository changes.

## Severity and disposition labels

### Severity

- **Critical:** Can silently change trial identity, modality alignment, analyzed
  behavior, image data consumed downstream, or biological sample inclusion.
- **High:** Can materially alter fluorescence responses, ROI results, timing,
  uncertainty, or reproducibility.
- **Medium:** Creates substantial maintenance, audit, or portability risk and
  can become scientifically material in some datasets.
- **Low:** Primarily presentation or workflow friction, but still inappropriate
  for a canonical pipeline.

### Disposition

- **Adopt concept:** Preserve the scientific or domain concept.
- **Adapt:** Preserve core logic after making contracts and assumptions explicit.
- **Rewrite:** Reimplement from a specification and tests.
- **Candidate only:** Retain as one method pending scientific validation.
- **Retire:** Do not carry into the canonical route.

## Executive assessment

The source repository contains valuable knowledge about the acquisition and the
intended imaging analysis, but it is not a reliable integrated pipeline as
committed.

The most consequential findings are:

1. imaging rereads and preprocesses raw behavior instead of consuming canonical
   behavior artifacts;
2. imaging is enabled through paths and source presence rather than explicit
   recording capabilities;
3. trial and plane assignment contains hard-coded dataset-specific logic;
4. the newer join loses important trial protocol/behavior slicing present in
   the older join;
5. synchronization reconstructs timestamps under incomplete assumptions and
   lacks explicit uncertainty;
6. downstream response analysis appears to consume an earlier registered image
   product instead of the final between-plane-aligned product;
7. two fundamentally different ROI methods coexist without a canonical choice
   or common contract;
8. orchestration, filenames, field names, and serialization are internally
   inconsistent;
9. broad exception handling can silently omit trials or data;
10. there is no artifact-level provenance, schema validation, or automated
    parity suite.

The recommended strategy is to preserve concepts and frozen reference outputs,
then rewrite the pipeline around the current repository's artifact contracts.

## Critical issues

### C1. Imaging creates a second behavior pipeline

The imaging join reads camera, protocol, and tail files directly:

- `1. Join_all_data_new.py:120-155`
- `1. Join_all_data_new.py:216-230`

It does not consume processed behavior produced by the behavior pipeline. The
behavior repository and the active `ClassicalConditioning` route also differ in
interpolation, filtering, vigor, bout thresholding, scaling, and normalization.

**Risk:** Behavior from imaging-enabled experiments can have different
semantics from behavior-only experiments even when they share raw sources.
Cross-modal effects can then confound biological modality with analysis route.

**Required disposition:** **Rewrite.** Canonical behavior artifacts must be
produced once by `ClassicalConditioning`. Imaging may use their clocks, trial
IDs, and outcomes but may not preprocess tail behavior independently.

### C2. Imaging availability and acquisition type are inferred implicitly

There is no operational `imaging_enabled` capability in the experiment
configuration. Two-photon experiment enum values are present, but the
configuration resolver falls through to `NotImplementedError`:

- `experiment_configuration.py:25-67`
- `experiment_configuration.py:132-170`
- `experiment_configuration.py:391-398`

Actual selection relies on active path modules, expected folders, file presence,
and matching strings in `path_home`:

- `1. Join_all_data_new.py:77-84`
- `1. Join_all_data_new.py:105-125`
- `1. Join_all_data_new.py:176-203`

**Risk:** A renamed folder, partial copy, or unexpected file can silently choose
the wrong path or analysis assumptions.

**Required disposition:** **Rewrite.** Declare capabilities and acquisition
order per recording, then validate discovered sources against that declaration.

### C3. Trial and plane identity is hard-coded from dataset path strings

The newer join assigns plane/trial structure through branches on `path_home`:

- 15-plane, 2-repeat, 4-trial patterns;
- 1-plane, 80-trial patterns;
- a special 4-plane mapping.

See `1. Join_all_data_new.py:176-205`.

The 4-plane branch contains `range(55, 45)` at line 193, which is empty in
Python and is likely an implementation error or unrecorded exception.

**Risk:** Trials can be assigned to the wrong plane while shapes and file writes
still look plausible. Acquisition-order changes cannot be detected reliably.

**Required disposition:** **Rewrite as data.** Use an authenticated
acquisition-order table keyed by imaging run, source sequence, plane, repeat,
and canonical trial/event ID.

### C4. The newer join regresses protocol and trial slicing

The older join constructs per-trial protocol fields such as CS/US beginnings
and endings and slices behavior to trial-relative windows:

- `1. Join_all_data.py:858-892`

The newer join's trial construction is less complete and explicitly describes
its protocol replication as approximate:

- `1. Join_all_data_new.py:362-420`

**Risk:** The structurally cleaner refactor can produce scientifically poorer
trial objects than the older code. Downstream response windows may be absent,
inconsistent, or reconstructed differently.

**Required disposition:** **Adapt old semantics, rewrite implementation.**
Define the trial map once from canonical events and test it against the usable
parts of the older join.

### C5. Synchronization makes untracked timing assumptions

The join:

- retimes behavior after detecting a stable camera interval;
- replaces timestamps with uniform spacing;
- detects galvo peaks using fixed settings;
- backfills missing initial imaging times using the median interval;
- explicitly handles imaging starting before behavior, but not the inverse with
  equivalent normalization.

Evidence:

- `my_functions_imaging_new.py:38-75`
- `1. Join_all_data_new.py:216-223`
- `1. Join_all_data_new.py:252-325`

**Risk:** Offset, drift, dropped frames, or false galvo peaks can move stimulus
windows relative to the image stack. Synthesized timestamps are not represented
as uncertain observations.

**Required disposition:** **Rewrite and validate.** Synchronization must be a
first-class artifact containing match status, residual, uncertainty, source
clock, and synthesized-value flags.

### C6. The final aligned image product is apparently not used downstream

Motion correction produces several binaries, including an intermediate
`registeredBin` and a later `registeredBinFinal`:

- `2. Motion_correction_Suite2p_new.py:283-380`
- `2. Motion_correction_Suite2p_new.py:391-455`

The pixel-response stage reloads `registeredBin`:

- `3. Analysis_of_imaging_data_pixels_Suite2p_new.py:89-112`

**Risk:** Between-trial or between-plane alignment may be calculated but not
propagated into response maps. Users may believe final alignment affects
results when it does not.

**Required disposition:** **Critical correction before parity claims.** Name
each registration product explicitly and test that response and ROI stages
consume the approved final artifact.

### C7. Silent skipping and broad exception handling hide data loss

Examples include broad `except`, `pass`, and `continue` behavior in:

- `1. Join_all_data_new.py:374-390`
- `1. Join_all_data_new.py:426-445`
- `my_functions_behavior.py:149-168`
- `my_functions_behavior.py:535-557`

**Risk:** Missing fish, trials, planes, or malformed values can be omitted
without a complete failure manifest. Resulting sample sizes may differ by
stage.

**Required disposition:** **Rewrite.** Use specific exceptions and explicit
per-recording/trial failure artifacts when batch continuation is intended.

## High-priority issues

### H1. US timing can be inferred from the fish name

When protocol information is absent, response analysis assigns US timing based
on name patterns, for example 9 seconds for delay/control and 13 seconds for
trace:

- `3. Analysis_of_imaging_data_pixels_Suite2p_new.py:68-76`
- `3. Analysis_of_imaging_data_pixels_Suite2p_new.py:138-150`

**Risk:** Naming conventions become hidden scientific metadata. Renaming,
mislabeling, or mixed protocols can shift response windows.

**Disposition:** **Retire.** Require canonical stimulus events or an explicit
reviewed protocol override.

### H2. Summer-time correction is inferred from month in the fish name

The newer join infers a time correction from the fish-name month:

- `1. Join_all_data_new.py:89-95`
- `1. Join_all_data_new.py:249-250`

**Risk:** Clock correction depends on identifier formatting rather than source
timezone metadata and can fail around transitions or renamed data.

**Disposition:** **Rewrite.** Preserve source timestamps, timezone, and applied
offset as explicit metadata.

### H3. Galvo peak detection and timestamp recovery lack validation artifacts

The peak detector and median-interval recovery may be reasonable for known
signals, but the pipeline does not save a complete match/residual/uncertainty
table.

**Risk:** A false peak can shift all later image frames while the merged table
remains syntactically valid.

**Disposition:** **Adapt as a candidate algorithm.** Add synthetic tests,
diagnostic plots, manual landmark review, and approved residual thresholds.

### H4. Background subtraction is fixed and precedes registration

The motion-correction stage subtracts the first percentile from anatomy and
each trial:

- `2. Motion_correction_Suite2p_new.py:133-164`

**Risk:** Dim biological signal and background structure may be altered before
registration. Its effect can vary by trial and plane.

**Disposition:** **Candidate only.** Preserve raw values, make the transformation
an explicit recipe, and validate registration and response sensitivity with and
without it.

### H5. Frame-quality thresholds are heuristic and coupled to registration

The route combines Suite2p bad-frame handling with correlation rejection around
`mean - 2.5 SD` and a configured `th_badframes = 0.7`:

- `2. Motion_correction_Suite2p_new.py:167-177`
- `2. Motion_correction_Suite2p_new.py:252-275`
- `2. Motion_correction_Suite2p_new.py:384-455`

**Risk:** Valid low-correlation responses or invalid but correlated frames may
be misclassified. The resulting missingness is stimulus- or motion-dependent.

**Disposition:** **Separate and validate.** Registration outputs should be
immutable inputs to a distinct frame-QC stage with threshold sensitivity.

### H6. Response normalization has a fixed soft threshold and smoothing

Response maps use:

```text
(stimulus_mean - pre_mean) / (pre_mean + 100)
```

and Gaussian smoothing with `sigma = 2`:

- `3. Analysis_of_imaging_data_pixels_Suite2p_new.py:64-65`
- `3. Analysis_of_imaging_data_pixels_Suite2p_new.py:164-180`

**Risk:** The fixed value 100 disproportionately controls responses in dim
pixels. Analysis smoothing and display smoothing are not clearly separated.
There is no declared neuropil correction in this route.

**Disposition:** **Reimplement as named candidates.** Save raw window means,
unsmoothed contrasts, normalized contrasts, and display products separately.

### H7. Two incompatible ROI paradigms are not governed as alternatives

One route runs Suite2p cell detection and emits `stat`, `iscell`, `F`, `Fneu`,
and optional `spks`:

- `5. Cell detection_new.py:100-114`
- `5. Cell detection_new.py:229-245`

Another creates seeds from correlation maps, grows regions by pixel-trace
correlation, and classifies them using CS/US/learning regressors:

- `6. ROI_analysis_new.py:388-560`

The Suite2p route is not the direct source consumed by the correlation ROI
route.

**Risk:** "ROI analysis" can refer to two different estimands, detection biases,
and output populations. Regressor-based selection and evaluation on the same
data can inflate apparent task responsiveness.

**Disposition:** **Separate candidate routes.** Give each a method ID, schema,
validation plan, and scientific status. Pixel analysis remains valid as an
option if no ROI route is approved.

### H8. Suite2p cell detection hard-codes frame rate and disables registration

The cell-detection route uses `fs = 30.0` and `do_registration = False`:

- `5. Cell detection_new.py:143-160`

**Risk:** A mismatched frame rate affects temporal filters and deconvolution.
Disabling registration assumes prior products are correct and correctly
selected.

**Disposition:** **Rewrite configuration.** Resolve frame rate from authenticated
acquisition metadata and bind cell detection to an explicit registered input.

### H9. Image dimension semantics are inconsistent

The join constructs functional imaging with dimensions described as
`Time (ms), x, y`, while anatomy and downstream operations use combinations of
`plane, y, x` and `[Y, X, T]`:

- `1. Join_all_data_new.py:343-352`
- `1. Join_all_data_new.py:432-442`
- `6. ROI_analysis_new.py:82-106`

**Risk:** Transposition errors can preserve plausible images while changing
orientation and anatomical mapping.

**Disposition:** **Rewrite around one convention:** `time, y, x` for functional
imaging and `plane, y, x` for anatomy, with tested explicit adapters.

### H10. Trial concatenation removes explicit inter-trial gaps

The ROI route appends trials back-to-back and shifts behavior times similarly:

- `6. ROI_analysis_new.py:318-358`

**Risk:** Filters, correlations, and regressors may operate across artificial
trial boundaries unless every downstream algorithm respects boundary masks.

**Disposition:** **Adapt only with segment boundaries.** Prefer trial-aware
operations. If concatenation is required by a library, persist boundaries and
invalidate cross-boundary calculations.

## Medium-priority issues

### M1. Script chaining is broken or stale

The newer motion-correction and response scripts call filenames that do not
match committed files:

- `2. Motion_correction_Suite2p_new.py:510-515`
- `3. Analysis_of_imaging_data_pixels_Suite2p_new.py:186-192`

**Impact:** Execution is effectively manual and working-directory dependent.

**Disposition:** **Retire script chaining.** Use registered package stages and
the common CLI.

### M2. A required import is missing

`my_functions_imaging_new.read_protocol()` uses `Path` without importing it:

- `my_functions_imaging_new.py:8-18`
- `my_functions_imaging_new.py:97-103`

**Impact:** The refactored reader path can fail immediately.

**Disposition:** Do not patch as the migration strategy; replace it with the
canonical protocol reader and retain this as legacy evidence.

### M3. HDF5 writer and ROI loader disagree on file names

The writer creates a name resembling `<fish_ID>_4. Activity_maps_data.h5`, while
the ROI entry point expects `<fish_name>_data.h5`:

- `5. Save_data_as_HDF5_new.py:46-52`
- `6. ROI_analysis_new.py:628-635`

**Impact:** Manual renaming or hidden local convention is required.

**Disposition:** **Rewrite.** Resolve inputs by explicit artifact ID, not file
name guessing.

### M4. Recursive HDF5 serialization is not a stable schema

The newer exporter recursively serializes lists as `item_N`, xarray objects as
`data + coords`, and DataFrames as `values/index/columns`:

- `5. Save_data_as_HDF5_new.py:169-350`

**Risk:** Object layout, reconstruction rules, types, categories, missingness,
and forward compatibility are implicit.

**Disposition:** **Retire.** Define explicit dense-array and table schemas.

### M5. Output roots and active datasets are hard-coded

Examples include `F:\Results (paper)`, `H:\2-P imaging`, and one selected fish
in path modules:

- `my_paths.py:3-12`
- `my_paths_new.py:27-37`
- `1. Join_all_data_new.py:81-84`
- `2. Motion_correction_Suite2p_new.py:76-90`

**Risk:** Results depend on edited source files and local drives, with weak run
provenance.

**Disposition:** **Rewrite.** Resolve source and project roots from explicit
local configuration; keep recording selection in CLI/API arguments.

### M6. Shared constants are imported globally

Column names and scientific constants live in `my_general_variables.py` and are
wildcard-imported across scripts:

- `my_general_variables.py:63-116`
- `my_general_variables.py:158-199`
- `my_general_variables.py:264-289`

**Risk:** Stage dependencies and invalidation are unknowable without code
inspection.

**Disposition:** **Rewrite as stage-scoped immutable configuration.**

### M7. Activity-map output location is unrelated to the run artifact root

The newer activity-map stage writes TIFFs to Desktop:

- `4. Activity_maps_new.py:83-85`
- `4. Activity_maps_new.py:112-113`

**Impact:** Outputs are easy to lose, overwrite, or detach from provenance.

**Disposition:** **Rewrite on the common figure/artifact export contract.**

### M8. Field names and casing drift between stages

The response analysis creates lowercase `cs_us_vs_pre`, while a mapping script
expects uppercase `CS_US_vs_pre`:

- `3. Analysis_of_imaging_data_pixels_Suite2p_new.py:176-180`
- `A1. Mapping responses to the stim.py:236-239`

**Impact:** Downstream scripts can fail or consume stale object versions.

**Disposition:** **Rewrite with registered schemas and reject unknown/missing
fields.**

### M9. Mutable object graphs obscure stage boundaries

Trials are progressively mutated with masks, templates, positions, response
maps, and anatomy:

- `my_classes_new.py:15-31`

**Risk:** It is difficult to determine which stage produced a field, whether it
matches current inputs, or whether a partially processed object is valid.

**Disposition:** **Adopt concept, retire persistence model.** Use immutable
artifact references and normalized tables.

### M10. There is no canonical test or provenance framework

The inherited pipeline does not provide an automated suite that proves:

- exact source assignment;
- correct trial/plane mapping;
- synchronization residual limits;
- array orientation;
- consumption of the final registration product;
- behavior equivalence;
- deterministic recipe identity;
- join cardinality;
- cohort completeness.

**Disposition:** **Rewrite around testable stage contracts.**

## Old versus new implementation assessment

| Area | Older implementation | Newer implementation | Assessment |
| --- | --- | --- | --- |
| Trial protocol fields | Builds explicit CS/US boundaries | Approximate/incomplete replication | Older semantics are more useful |
| Behavior slicing | Explicit trial-relative slices | Less complete | Newer route regresses behavior integration |
| Object structure | More procedural | Cleaner `Data/Plane/Trial` hierarchy | Newer concept is clearer, persistence is still unsafe |
| Paths/config | Hard-coded | Still hard-coded in refactored modules | Not resolved |
| Synchronization | Dataset-specific | Cleaner helper separation | Assumptions remain implicit |
| Motion correction | Multiple versions | More modular Suite2p route | Needs final-product and provenance correction |
| HDF5 | Intended fixed layout in documentation | Generic recursive serializer | Newer route is more flexible but less canonical |
| ROI analysis | Multiple exploratory files | Two clearer alternatives | Still scientifically and contractually unresolved |
| Orchestration | Manual numbered scripts | Script chaining attempted | Chaining is stale/broken |

The correct migration is not "use every `_new.py` file." The older join contains
scientifically important logic that the newer refactor did not preserve.
Reference behavior must be selected field by field, then represented through
new contracts.

## Component disposition summary

| Component | Severity if copied | Disposition |
| --- | --- | --- |
| Raw behavior processing in imaging join | Critical | Retire; consume canonical behavior |
| Explicit fish/plane/trial concept | Low | Adopt concept |
| Source/path discovery | Critical | Rewrite |
| Plane/trial hard-coded mapping | Critical | Rewrite as acquisition-order data |
| Galvo peak detection | High | Adapt as validated candidate |
| Timestamp backfill | High | Candidate only with uncertainty |
| Old protocol/trial slicing | High | Adapt semantics |
| New approximate trial construction | Critical | Retire |
| Suite2p registration engine | Medium | Adapt behind explicit stage |
| First-percentile subtraction | High | Candidate only |
| Correlation frame QC | High | Candidate only |
| Pixel window means | Medium | Adopt concept |
| Soft-threshold normalized response | High | Candidate only |
| Gaussian map smoothing | High | Separate analysis from display |
| Suite2p ROI route | High | Reimplement as one explicit route |
| Correlation-grown ROI route | High | Experimental route only |
| Recursive HDF5 | Medium | Retire |
| Pickled mutable objects | High | Retire after reference conversion |
| Desktop/drive output paths | Medium | Rewrite |
| Script chaining | Medium | Retire |

## Questions requiring scientific or acquisition-owner input

1. Which imaging-enabled experiments and recordings are paper-authoritative?
2. What hardware clock generated camera, galvo, and image timestamps?
3. Are absolute times in a defined timezone, and were daylight-saving
   corrections already applied upstream?
4. What constitutes one imaging frame when planes or repeats are interleaved?
5. Is the trial/plane acquisition order available outside source code?
6. What frame rate applies to each recording and plane?
7. Which registration product was intended for published response maps?
8. Why was first-percentile background subtraction selected?
9. How were `th_badframes`, the correlation cutoff, soft threshold 100, and
   Gaussian sigma 2 chosen?
10. Are pixel response maps, Suite2p cells, correlation-grown ROIs, or more than
    one route required for the final scientific claims?
11. Was neuropil correction intended?
12. Are ROI detection and task-response classification meant to use independent
    data?
13. How should trials with insufficient pre-CS or response frames be handled?
14. Which behavior recipe was intended for imaging-enabled fish?
15. Are there frozen published imaging outputs available for comparison?

## Required evidence before scientific reuse

Before any inherited imaging output becomes paper-authoritative:

- source identities and acquisition order are authenticated;
- synchronization is reviewed and residuals are within approved limits;
- the final registered artifact consumed downstream is verified;
- array orientation is tested;
- frame exclusion is explicit and sensitivity-tested;
- response windows derive from canonical stimulus events;
- normalization and smoothing are separately justified;
- ROI methods are identified and independently validated;
- fish, trial, plane, pixel, and ROI statistical units are handled correctly;
- missing imaging never changes the primary behavior cohort;
- behavior output equality with and without imaging is demonstrated;
- all retained stages have schemas, recipe IDs, provenance, and failure reports.

## Conclusion

`analysis_learning` should be retained as an acquisition-knowledge and
reference-output repository, not imported as a package dependency or treated as
the canonical multimodal pipeline.

Its most reusable assets are the conceptual acquisition hierarchy, the intent
to synchronize galvo and behavior, the Suite2p registration route, explicit
frame masking, and the pixel/ROI analysis ideas. Its orchestration, behavior
fork, identity inference, serialization, configuration, and failure handling
need replacement.

The new implementation should be judged by scientific contracts and frozen
reference comparisons, not by whether it preserves the shape of the legacy
Python objects.
