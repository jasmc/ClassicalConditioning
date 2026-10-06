# Preprocessing audit and proposed contract — 7 October 2026

**Status: NOT CONFIRMED across all experiment routes. No downstream analysis review, cohort reprocessing, or population rerun has been launched.** This report separates source observations, reproducible legacy parity, deliberate repairs, and unresolved policy choices. The original detailed code is an oracle for historical behavior, not a presumption of scientific correctness.

Evidence is in `J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/preprocessing-audit-20261007-v2/`. The first attempted audit folder is empty; it was retained rather than overwritten. Figure-builder scripts, assembly configurations, panels, and frozen assets were not edited.

## Scope and provenance

Inspected the detailed `legacy/modules/my_functions.py`, `my_general_variables.py`, archival `my_experiment_specific_variables.py`, current helper/config/parser code, the preprocessing script, current corrected-frame and candidate-metric kernels, and movement-state support rules. Compared HEAD (bf6739584ca1b85729decea710e684433420205f) with the working helpers; existing acquisition-timing changes were already uncommitted. New changes extend those edits, without reverting unrelated work. The historical detailed modules remain unchanged.

Four full recording source triplets were SHA256-verified against their intake manifests before numerical reading: Delay 20221115_07, Delay Control 20221115_09, 3sTrace Control 20230227_03, and 3sTrace 20230306_01. This authenticates the intake Parquet bytes against the stored manifest; it does not independently rehash every multi-gigabyte raw TXT. Raw tracking endpoint inspection and raw protocol hashes provide additional evidence. Delay artifacts have moved F: → J: while manifests retain F: paths; hashes match. The production strict path validator would reject this relocated project. Do not rewrite manifests to hide this discrepancy; use a separately recorded relocation mapping for a future recipe. Trace still matches the F: manifest paths.

AppleDouble `._...source_manifest.json` sidecars are reported as unreadable in the initial audit JSON. They are not corrupt recording manifests. Selection is deterministic: one full recording per source-project condition, with 20221115_07 explicitly preferred. This is representative full-recording QC, not an all-fish clearance.

## Stage contract and findings

| Stage | Detailed legacy behavior | Current behavior / audit disposition | Proposed requirement |
|---|---|---|---|
| Source parsing | Space/dot camera, tab/comma fallback; tracking angles converted rad→deg | Current intake validates numeric camera parsing, full XY/angles; preserves raw fields and recorded missingness | Preserve raw bytes/fields, record parser decisions and hashes; validate format numerically |
| Startup removal | `skiprows=range(1,14000)` removes **13,999** data rows; then discard before stable reference | Corrected-v1 keeps startup. Comment saying ~60 s is wrong: nominal discard is ~20 s | Version startup policy explicitly; compare full and historical-discard cadence, never silently call the two identical |
| Trailing row | Unconditionally drops final tracking row | Four raw endpoints end in real, camera-matched numeric FrameIDs with 49 fields. Intake correctly retains them | Drop only an identified nonnumeric summary row. Legacy helper repaired; deliberate correction rather than exact reproduction |
| Frame join | Inner ordered FrameID merge, then zero-based FrameID | Corrected streaming searchsorted inner join preserves IDs; 0–2 leading tracking rows unmatched in sampled sources | Require unique monotonic IDs and record unmatched rows on each side; don't confuse inner-join removal with acquisition loss |
| Missing frames | Historical inference from accumulated arrival lag/buffer threshold | Shared estimator separately counts missing exported IDs and buffer capacity evidence | Direct missing-ID check plus distinct capacity evidence; brief arrival delays are not loss counts |
| Cadence/reference | Three stable arrival intervals; original ID-arithmetic indexing and slice mean | Repaired positional/FrameID-span estimator; reference shifts one frame earlier, tiny FPS differences | Keep algorithm/version/anchors in provenance. Reject insufficient stable support. FPS estimate is inferred, not hardware exposure measurement |
| Acquisition clock | Cadence/buffering model exists in original, but downstream clocks include inconsistencies | Corrected-v1 uses camera arrivals as denominator. Repaired legacy script reconstructs both clocks **before tracking join** and protocol labelling | Preserve arrival clock separately; inferred time = stable-reference arrival epoch + FrameID offset × interval. Unknown reference latency remains |
| Resampling | FrameID×700/sourceFPS, slinear values, `arange(first,last)` excludes final endpoint | Repaired legacy helper rebuilds clock at **700 FPS**, not source FPS; corrected-v1 has no interpolation | Keep mapping/grid endpoint policy explicit. Interpolated original-frame number is not an acquired ID. Never bridge tracking gaps without a bounded/versioned policy |
| Protocol assignment | First frame with AbsoluteTime > onset; last <= end; old singleton selection could skip events | Singleton selection repaired; float time retained. Corrected stage only summarizes protocol start placement | Explicit boundary convention, out-of-range events and completeness checks including event ends. Hardware onset latency is unknown; don't claim exact stimulus exposure synchronization |
| Angles | Spatial cumulative sum of local bends, deg; no temporal unwrap | Candidate legacy-distal sums radians and wraps *temporal delta* to ±π | Spatial cumsum is not unwrapping. Wrapped candidate is a deliberate change; investigate real >π differences before claiming legacy identity |
| Spatial filter | Width-3 rolling mean on cumulative angles with sequential edge updates | Refactor had omitted this entirely; repaired to match original exactly | Preserve exact original behavior for reproduction; choose any alternative under a new policy |
| Temporal filter | Centered 10-frame mean at 700 FPS, drop unsupported edges | Corrected-v1 disables it; current detector uses a median smoothing step | These are different recipes. Mean ≠ median. Record windows, edge handling, order and units |
| Raw vigor | Absolute distal cumulative-angle difference ×0.7, deg/ms; helper initializes first value to zero | Candidate uses wrapped delta / arrival interval, rad/ms, explicit validity | Units convert by π/180. First derivative lacks support; missing/invalid remains NaN in a new scientific recipe, rather than inventing zero |
| Derivative validity | Legacy resampling/dropna can obscure original gaps | Corrected-v1 requires consecutive IDs, positive ≤10 ms arrival interval, valid neighboring frames | Clock-dependent validity must use the declared clock; separate XY coverage, finite angle support, tracking gaps, detector support, and movement eligibility |
| Detector envelope | Centered 20-frame max minus 400-frame min; threshold 4 deg/ms | Refactored envelope arithmetic agrees, but its input/filter differs; modern detector uses contiguous support and odd physical-time windows | Preserve support masks. Do not equate a 67%-invalid arrival-clock detector to motion absence |
| Bout rules | Exclude endpoints, merge `beg-next − end <10`, remove `end−beg <40`, peak distal speed ≥1 deg/ms | Refactor omitted amplitude check and took boolean diff (XOR); both repaired. Modern duration-based bounds differ deliberately | Label exact sample convention vs inclusive physical duration. Numeric transitions distinguish beginnings/endings. Never merge across unsupported gaps |
| Trial assignment | Inclusive ±45 s windows; reset every cropped trial to −45 s even if partial | Same inherited bug in helper; repaired to subtract actual onset reference | Keep actual relative time for partial trials and gaps; onset must map to zero |
| Scaled vigor/missingness | Historical cleanup can replace off-bout vigor with zero / bout summaries | Script's quantile scaling uses time <−15 s and groups CS/US with same trial number together | Existing script scaling is **not approved** for the requested metric. Required baseline is [−15,0), eligible moving-bout log medians, frame-weighted 0.5 s means ignoring NaN; missing remains NaN. Keep this as a distinct proposed recipe, not silent reinterpretation of old scaled columns |
| Export/source identity | Pickles often lack complete recipe provenance; some are gzip with .pkl extension | Current recipes freeze settings and artifacts; corrected-v1 preserved | New scientifically changed outputs require new identity and complete source/code/clock provenance. No frozen artifacts rewritten |

## Repairs made in this audit

1. Restored detailed legacy spatial averaging, including its sequential edge calculations. Unsupported widths now fail explicitly.
2. Restored the omitted second bout amplitude threshold.
3. Replaced boolean-difference bout transitions with numeric transitions. Previously endings were labelled as beginnings and `Bout end` stayed false.
4. Preserved numeric final tracking rows rather than unconditionally deleting them; normalize fallback string IDs to integer join keys.
5. Preserved actual partial-trial relative times rather than shifting the onset.
6. Reconstructed camera clocks at their stable camera reference before the tracking inner join. Otherwise a missing tracking row at the reference could re-anchor the model to a buffered first matched arrival.
7. Clarified corrected-preprocessing documentation: it is a frozen **camera-arrival-time** recipe, not measured hardware acquisition timing. Clarified that spatial angle cumsum is not temporal unwrapping.

None of these edits change existing corrected-v1 files. The legacy preprocessing script itself was not executed. Its remaining quantile baseline/scaling behavior is explicitly unresolved and must not be used as the approved Figure 1 scaling contract.

## Full-recording measured timing and join evidence

| Recording | Camera rows | Arrival intervals >10 ms | Arrival intervals <0.5 ms | Inferred FPS | Maximum post-reference delay ms | Tracking rows unmatched to camera | Summed-angle temporal changes >π |
|---|---:|---:|---:|---:|---:|---:|---:|
| Delay 20221115_07 | 8,344,615 | 13,936 | 164,213 | 702.571512147 | 28.144 | 2 | 5 |
| Delay Control 20221115_09 | 8,313,125 | 3,013 | 261,441 | 702.570287820 | 144.425 | 0 | 399 |
| Trace Control 20230227_03 | 8,399,999 | 3,116 | 251,228 | 702.568204252 | 690.003 | 1 | 0 |
| 3sTrace 20230306_01 | 8,390,680 | 3,056 | 271,938 | 702.569126963 | 233.978 | 1 | 0 |

All four have zero missing, duplicate, or reversed exported camera IDs; no buffer-capacity exceedance; no nonfinite angle cells; 94 Cycle and 79 Reinforcer events, all positive-duration and all event starts inside arrival-time acquisition spans. The 79 reinforcers must not be silently reduced to the configured minimum 78. Events ending outside spans were not counted in this initial measurement and remain a completeness check to add before approval. Whole-recording >π counts are observations, not proof of tracking faults or removable wrap artifacts.

The original cadence function was executed independently via AST extraction, with only diagnostic plotting replaced by no-ops. For full Delay 20221115_07 it returns 702.571510826 FPS, reference 430474, no loss; repaired returns 702.571512147, reference 430473. With 13,999 startup rows removed, original returns 702.571522601, reference 443457; repaired returns 702.571523924, reference 443456. All four recordings, both startup variants, agree on no loss. Rate differences are 0.00000039–0.00000133 FPS. This is measured agreement in cadence inference, **not bit-for-bit equivalence of the pipeline or absolute epoch**.

The earlier Figure 1 evidence documents trial 93 missing bins 25→0 and detector coverage ~31%→100% under inferred acquisition timing. That remains a timing-only, versioned review; this audit did not regenerate its figures. Its five-window results cannot establish every fish's or experiment's preprocessing validity.

## Measured spatial-filter comparison

Four first-50,000-tracking-row windows isolate filter arithmetic at a declared uniform grid. They do not include full raw parsing, cadence resampling, or protocol/trial assignment. Repaired filtering agrees with the AST-loaded original at max absolute difference **0.0°** for all four. The omitted spatial filter changed distal angles by max 5.5093°, 4.0820°, 4.6987°, and 4.6847° respectively; median absolute distal-speed differences are 0.0418924, 0.0270150, 0.0464392, and 0.0473294 deg/ms. The original edge update is sequential, so generic symmetric edge padding is not exact reproduction.

## Validation and reproduction

Run with `.venv-trace/Scripts/python.exe`. pytest is absent; unittest is used. **57 targeted tests pass**, covering original-function spatial/bout parity, partial-trial clocks, real-row preservation, acquisition-time repair, current corrected-frame masks, raw parsing, candidate metric semantics and movement support. The initial boolean-transition parity failure was fixed rather than weakening the expectation. Sandbox temp-directory restrictions were resolved by setting `tempfile.tempdir` to a workspace test directory.

Reproduce diagnostics in a fresh output directory; the first script refuses an existing output directory:

```
.venv-trace/Scripts/python.exe scripts/audit_preprocessing_contract.py --output "J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/preprocessing-audit-NEW"
.venv-trace/Scripts/python.exe scripts/audit_preprocessing_routes_and_filters.py "J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/preprocessing-audit-NEW"
.venv-trace/Scripts/python.exe scripts/audit_preprocessing_archival_routes.py "J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/preprocessing-audit-NEW"
.venv-trace/Scripts/python.exe scripts/audit_preprocessing_original_cadence.py "J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/preprocessing-audit-NEW"
.venv-trace/Scripts/python.exe scripts/audit_preprocessing_historical_tables.py "J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/preprocessing-audit-NEW"
.venv-trace/Scripts/python.exe scripts/audit_preprocessing_historical_compression.py "J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/preprocessing-audit-NEW"
```

The original-function oracle compiles selected function bodies only and does not execute legacy module top-level code. Historical pickle reads are limited to trusted local outputs; gzip compression is detected from magic bytes. Stored historical schema/support can be inspected, but missing recipe provenance prevents claiming exact historical reproduction from table shape alone.

## Confirmation boundary

Approve a **new versioned preprocessing contract**, not a blanket declaration that corrected-v1 or every archived route is scientifically correct. Before advancing, settle startup policy; exact legacy mean/filter sequence versus deliberately unfiltered acquired-frame recipe; wrapped versus unwrapped distal differences; tracking-gap interpolation/support; protocol boundary/completeness handling; and the required eligible-bout baseline/binning policy. Finish missing route implementations and condition naming, preserve relocation provenance, and compare sampled saved tables numerically with a fully specified reproduction before claiming historical identity. These are preprocessing tasks and remain inside the current gate. Later fish/cohort reruns require the user's explicit preprocessing confirmation.

## Every experiment route: explicit coverage

Current helper configuration supports 9 of 35 declared routes. Archival source has 38 branches including deprecated routes. Counts below are inventory evidence, not acceptance/exclusion counts. Protocol samples are one available recording per archival branch; full camera/tracking hash checks above cover four recordings in Delay/3sTrace only.

| Route | Current configured | Raw protocol files at archival path | Historical pickles | Raw protocol sample | Archival defect |
|---|---|---:|---:|---|---|
| respToUS | yes | 44 | 44 | parsed |  |
| movingCS_4cond | yes | 132 | 112 | parsed |  |
| only stimulation | yes | 73 | 72 | parsed |  |
| manyDelayTrainingTrials | yes | 4 | 4 | parsed |  |
| _deprecated_delay_trace_protocol | no | 0 | 0 | no sample / placeholder |  |
| allDelay | yes | 0 | 51 | no sample / placeholder |  |
| _deprecated_fixed_trace_protocol | no | 0 | 0 | no sample / placeholder |  |
| _deprecated_pooled_trace_protocol | no | 0 | 0 | no sample / placeholder |  |
| all3sTrace | yes | 0 | 199 | no sample / placeholder |  |
| fixedTraceFullyReinforced | no | 38 | 0 | parsed |  |
| 10sTrace | yes | 85 | 63 | parsed |  |
| all10sTrace | yes | 0 | 63 | no sample / placeholder |  |
| delaymk801 | no | 96 | 96 | parsed |  |
| tracemk801 | no | 72 | 72 | parsed |  |
| delayLongTermSpacedVsMassed-1 | no | 30 | 0 | parsed |  |
| delayLongTermSpacedVsMassed-2 | no | 30 | 30 | parsed |  |
| delayLongTermSpacedVsMassed-all | no | 0 | 30 | no sample / placeholder |  |
| traceLongTermSpacedVsMassed-1 | no | 30 | 29 | parsed |  |
| traceLongTermSpacedVsMassed-2 | no | 29 | 27 | parsed |  |
| traceLongTermSpacedVsMassed-all | no | 0 | 26 | no sample / placeholder |  |
| tracePartialReinforcement | no | 38 | 38 | parsed |  |
| tracepuromycinshort | no | 35 | 33 | parsed |  |
| delayLongTermSpacedpuromycin5mg-1 | no | 77 | 0 | parsed |  |
| delayLongTermSpacedpuromycin5mg-2 | no | 75 | 0 | parsed |  |
| delayLongTermSpacedpuromycin5mg-all | no | 0 | 65 | no sample / placeholder |  |
| delayLongTermSpacedpuromycin10mg-1 | no | 0 | 0 | no sample / placeholder |  |
| delayLongTermSpacedpuromycin10mg-2 | no | 0 | 0 | no sample / placeholder |  |
| delayLongTermSpacedpuromycin10mg-all | no | 0 | 0 | no sample / placeholder |  |
| delayLongTermNew-1 | no | 0 | 0 | no sample / placeholder |  |
| delayLongTermNew-2 | no | 0 | 0 | no sample / placeholder |  |
| delayLongTermNew-3 | no | 0 | 0 | no sample / placeholder |  |
| delayLongTermNew-all | no | 0 | 0 | no sample / placeholder |  |
| 2-P multiple planes top | no | 34 | 33 | parsed | duplicate key: delay |
| 2-P multiple planes bottom | no | 10 | 10 | parsed |  |
| 2-P multiple planes zoom in | no | 21 | 24 | parsed |  |
| 2-P multiple planes ca8 | no | 34 | 33 | parsed | duplicate key: delay |
| 2-P single plane | no | 2 | 2 | parsed |  |
| ca8ablation | yes | 80 | 79 | parsed |  |

Condition-specific current-route samples are recorded in `route_and_filter_comparisons.json`. Many-delay raw files use `delayLong` naming whereas current configuration expects control/delay; neither matched. The `10sTrace` 3sfixedtrace condition had no matching filename in that folder. These are unresolved routing findings, not evidence of absent biology.

Historical table inspection authenticated and read 23 distinct sampled saved tables (23 attempted), including gzip-pickled .pkl files. No files were rewritten. Schema, row count, bout support counts where available, and index names are in the two historical JSON files. These saved tables are not a full numerical reconstruction oracle because exact source/recipe provenance is unavailable.
