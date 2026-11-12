# Paper Input Validation and Reproducibility

## Purpose and boundary

Authenticate the paper-scale inputs and reproducible processing state for the first Figures 1–4 release. Existing pipeline, provenance and fixture infrastructure are implemented; this plan owns the remaining paper-scale evidence, not a fresh implementation from zero. [Plan 02](02_DISCARDING_AND_COHORT_IMPLEMENTATION.md) owns fish inclusion/eligibility policy and the reviewed cohort; [Plan 06](06_ANALYSIS_AND_STATISTICS.md) owns outcome/inference choices.

## Remaining evidence

1. Produce the expected paper recording inventory by experiment/condition, with source paths/hashes and missing, incomplete, ambiguous and unavailable cases. Reconcile all raw triplets against that denominator; keep external source data in place.
2. Identify the exact code revision, locked environment and resolved input/processing configuration for the reference run. Authenticate existing outputs before scheduling any regeneration; record source, configuration, stage and output fingerprints.
3. Validate corrected measured-time preprocessing, synchronization/frame gaps and shared detector behavior across the paper inputs. Use the author-selected tail bend angular speed metric and approved bout-only vigor sampling; detector/smoothing validation does not reopen the selected metric or historical panel freezes.
4. Reconcile protocol/timing evidence through the [stimulus QC implementation design](#stimulus-timing-and-control-randomization-qc). Record coverage and unresolved mappings explicitly; missing inputs or processing failures are not automatically biological exclusions.
5. Deliver one traceable input-validation report and authenticated stage identities to Plan 02 and the outcome/inference owners. Distinguish full paper validation from fixture tests and provisional scientific runs.

## Stimulus timing and control randomization QC

### Scope and implementation boundary

Build Python QC for Delay, fixed 3sTrace, fixed 10sTrace and their configured
controls, per recording and across recordings. Extend existing intake,
verification and run reports. Keep three separate results: logged protocol
delivery, control timing structure, and links to behavioral quality evidence.

The author requested implementation of the reusable descriptive stimulus plot
set on 1 November 2026, including configurable future protocols. Its supported
interface and five complementary views are documented in
[stimulus timing plots](../docs/running-analysis/run-pipeline.md#stimulus-timing-plots).
That request does not authorize the optional inferential route below,
new exclusions, condition relabeling,
behavioral recomputation, frozen-record changes or manuscript edits.
[Plan 02](02_DISCARDING_AND_COHORT_IMPLEMENTATION.md) retains cohort ownership;
[Plan 06](06_ANALYSIS_AND_STATISTICS.md) retains scientific inference ownership.

### Existing code and evidence

| Source | Reuse and remaining work |
|---|---|
| [`config/protocols.py`](../src/classical_conditioning/config/protocols.py) | Shared counts, phases, nominal durations, control constraints and warning policy; do not duplicate these constants in renderers. |
| [`config/experiments.py`](../src/classical_conditioning/config/experiments.py), [`trial_map.py`](../src/classical_conditioning/config/trial_map.py) | Explicit assay/condition aliases, corrected event mappings and fingerprint validation. |
| [`ingestion/protocol_verification.py`](../src/classical_conditioning/ingestion/protocol_verification.py) | Existing numerical evaluator, authenticated-input verification and transactional JSON/HTML publication. Extend this route. |
| [`intake.py`](../src/classical_conditioning/intake.py) | Lossless stimulus table, acquisition evidence and existing `protocol_timeline.png`. |
| [`analysis/candidate_runner.py`](../src/classical_conditioning/analysis/candidate_runner.py), [`pipeline.py`](../src/classical_conditioning/pipeline.py) | Existing routine/resume verification and run summaries; integrate new evidence without coupling it to behavioral computation. |
| [`tests/test_protocol_verification.py`](../tests/test_protocol_verification.py) | Existing fixtures for all assays, exact totals, tolerance boundaries, catches, aliases and stale maps. Extend rather than replace. |
| Legacy `run_plot_protocols()` and `my_functions.plot_protocol()` | Reference for trial rasters and interval views only. Do not import the legacy experiment module: it creates folders and contains superseded settings. |

Bounded read-only inspection on 31 October 2026 found 57 regular Delay stimulus
files and 60 files in the fixed-3sTrace raw folder on JOAQUIM, all with 94 CS and
79 US labels. These file counts do not define a cohort or prove assay provenance.
Delay `20240624_03` also has an `_original` source with 95 CS labels; preserve
both variants and expose their provenance. Delay `20221122_03` contains coarse
scientific-notation epoch timestamps that yield implausible timing despite
correct totals. Use these as real validation cases, without repairing raw data.

For Delay `20221115_04` and `20221115_09`, raw rows equal saved stimulus Parquet
rows and match recorded hashes. Saved trial-outcome timestamps match those
events, but their phase/block labels retain superseded US numbering and their
summaries lack the current trial-map fingerprint. Treat them as historical
alignment evidence, not reusable current outcomes. Their stored Windows paths
also require supported provenance/path resolution before normal authenticated
loading on this Mac; do not rewrite historical manifests to bypass guards.

Real fixed-10sTrace validation was unavailable in the checked raw location.
The generator was not found in the inspected code tree. Resolve its location,
versions, intended schedules, seeds and batch assignments before generator-based
inference. Specifically resolve whether 46 CSs were randomized with four fixed
events, or all 50 were randomized, and whether spacing was measured onset to
onset or offset to onset. Logs alone do not settle those questions.

### 1. Extend numerical delivery verification

- Always report exactly 94 expected CS and 79 expected US events, including
  missing and extra labels. Keep raw counts separate from counts with usable
  timestamps; invalid rows must remain visible in evidence.
- Preserve original row identity and raw timestamp text where needed to expose
  precision loss. Assign observed event numbers by stable time ordering, retain
  ties and anomalies, and distinguish observed IDs from expected protocol slots.
- Use the canonical conditioned pair map: CS 15–64 with training catches
  25, 39, 53 and 59; paired US 19–64. Keep analysis catch CS 65 distinct.
  Report expected-pair latency even when it lies beyond the present proximity
  search. Missing/extra events make correspondence uncertain; do not force a
  plausible remapping. Proximity matches remain additional diagnostic evidence.
- Retain inclusive 200 ms tolerance for CS duration and paired onset/gap.
  Nominal latencies are 9, 13 and 20 s; derive signed offset gaps from measured
  CS offsets and the shared protocol. Retain US duration measurements and
  nominal 50/100/500 ms comparisons without duration-based exclusions.
- Expose phase and viability provenance. Ordinal slicing into expected phase
  sizes is not independent phase-count verification. Do not silently identify
  an arbitrary last pulse as the expected viability event when identity is
  ambiguous. Pulse presence does not establish a behavioral viability response.
- Detect duplicate onsets, overlaps, invalid durations, implausible/coarse
  timestamps, clock discontinuities and acquisition containment problems.
  Preserve legitimate nonchronological log rows; sorting is not a repair.
- Distinguish outer camera bounds from internal frame gaps and coverage of
  event-aligned windows. Consume authenticated saved coverage where available;
  mark unavailable coverage explicitly. Do not rerun behavioral analyses.
- Preserve warn-and-continue for protocol discrepancies. Existing schema and
  integrity failures retain their established failure behavior.

### 2. Add compact automatic plots

The author-selected plot set supersedes the four-output display proposal below:
recording timelines, dual-alignment rasters, ordered timing measurements with
processed residuals, fish-preserving lag summaries, and schedule comparisons.
The reusable implementation separates `config/protocol_timing.py`,
`analysis/protocol_timing.py`, `figures/protocol_timing.py` and
`ingestion/stimulus_timing.py`; the operational guide owns its current settings.
The remaining numerical-verifier and optional inference requirements retain
their separate scope. This plan does not select a randomness null or threshold.
The following original display proposal is retained as design context; its
filenames and module name are superseded by the supported interface above.

Create `src/classical_conditioning/figures/protocol_qc.py` as a renderer of
verified event evidence. Reuse the shared plotting theme and semantic stimulus
colors. Plot functions must not define eligibility, recalculate behavior or
select sources. Initial display settings below are proposed defaults, not
scientific acceptance thresholds.

| Output under each recording's existing `Quality checks` home | Content and diagnosis |
|---|---|
| `figures/protocol_timeline.png` | Extend the existing timeline with onset/offset lanes, phase provenance, both count comparisons, camera bounds/gaps and a distinct final viability marker. Check truncation, gross phase errors and clustering. |
| `figures/protocol_trials.png` | Compact CS-aligned and US-aligned rasters, with all 94/79 observed anchors when complete, actual neighboring onsets/offsets, phase/block separators and nominal/catch annotations. Start with a local window of -30 to +30 s. Check pairing, variability and phase contamination. |
| `figures/protocol_timing.png` | Duration, inter-onset interval, conditioned latency and signed gap versus event number and elapsed time; small distribution/ECDF insets. Separate short, long and viability US roles. Check drift, extreme gaps and isolated anomalies without hiding order in a histogram. |
| `figures/control_contingency.png` | Controls only: all-pairs signed lag summaries per fish, local -30 to +30 s in 1 s bins and wider -300 to +300 s in 5 s bins; exact target-coincidence counts and contributing anchors/exposure. Check forward, backward and recurrent structure. |

Define lag as US onset minus CS onset. Use all 50 control training CS events for
descriptive contingency, without dropping ordinal catch labels. Separate the
46 training USs, short maintenance pulses and viability. Do not use nearest/next
US latency as the primary contingency distribution. For rate plots, expose
observable phase time and window overlap; if phase boundaries are unverified,
show descriptive counts and the limitation instead of an invented denominator.

Optional deeper review: full-resolution event tables; block-specific rasters;
onset/offset ECDFs; arbitrary lag zoom; broader horizons; and detailed source
variant comparisons. Restore the useful legacy views without inheriting its
-40/+47 s range, 4 s bins, shifted numbering or density normalization.

### 3. Add across-fish aggregation and schedule reuse checks

Create `src/classical_conditioning/ingestion/protocol_summary.py` to consume
verified recording reports and an explicit expected recording inventory.

- Add fish-by-check and fish-by-lag plots to the existing run summary. Keep
  missing recordings, failed verification and uncheckable fields visible.
- Separate assays and show fish values grouped by acquisition day and verified
  protocol batch. Retain unknown batch/day fields as unknown; a filename date
  does not prove protocol-batch or clutch identity.
- Summarize per fish first and weight fish equally. Retain individual curves
  and extremes so a pooled mean cannot conceal one problematic recording.
- Record intended US schedule IDs, matched protocol families and provenance.
  Compare session-relative logged US vectors descriptively; matching US
  schedules do not prove common assay membership.
- Fingerprint intended CS schedules where available. For logged schedules,
  rank similarity after removing only a common time offset and show residuals.
  Do not normalize away clock scaling/drift. Exact logged hashes alone miss
  reuse obscured by timestamp jitter. Validate any future near-match threshold
  against known reused schedules before treating it as a classification.
- Show equal-fish and equal-unique-schedule summaries separately. Reused
  schedules are not independent generator draws. Preserve day/batch dependence
  and reuse relationships in any uncertainty calculation. With insufficient
  independent clusters, show stratified values without a population interval.

### 4. Keep control inference optional and explicitly specified

Automatic delivery and descriptive timing QC must work without a generator.
Create `src/classical_conditioning/ingestion/control_randomization.py` only
when implementing the optional inferential route. Missing model provenance
returns `not_checkable`, never a success claim or an improvised null.

| Null model | Implementation contract |
|---|---|
| Original generator: preferred | Hold intended US schedule fixed and reproduce actual sampling distribution, CS count, fixed events/catches, phase boundaries, spacing, rounding, rejection algorithm and batch/reuse relationships. |
| Constrained schedule model: sensitivity | Specify a distribution over complete admissible CS schedules. Constraints alone do not specify that distribution. Clearly distinguish assumed rules from recovered generator rules. |
| Conditional interval jitter: optional | Specify coarse windows and fixed counts; answer a fine-timing question only. State which clustering has been conditioned away. |

Do not default to unrestricted US randomization, circular shifts or trial
shuffling: these can violate the fixed-US design, phase boundaries or spacing.
The 0.5 s onset constraint itself changes the expected lag distribution; a flat
histogram is not the appropriate null expectation.

Retain effect sizes: target coincidences, full signed lag curves, clustering
and block changes. A proposed additional measure compares US incidence inside
the union of specified CS/post-CS windows with the remaining observable phase
time. Biological window lengths require author agreement before inference.

Before inferential implementation, settle the primary statistic, lag range,
effect direction, independent schedule units, uncertainty and test family under
Plan 06. Prefer a simultaneous envelope over the declared lag vector to
pointwise bin tests. Consider Holm adjustment across separate assay tests;
individual-fish inferential flags also need a declared family across fish.
Use reproducible simulations, include the observed configuration in the Monte
Carlo probability calculation, handle ties conservatively and report simulation
uncertainty. A final run of 9,999 simulations is a proposed cost/precision
setting to benchmark, not an approved scientific requirement.

Non-significance cannot establish randomness or absence of learning. A bounded
contingency claim additionally needs an agreed biological margin and horizon.
No such margin, label policy or exclusion threshold is selected by this plan.

Method references for later implementation: [conditional resampling](https://pubmed.ncbi.nlm.nih.gov/22031767/),
[global envelopes](https://arxiv.org/abs/1307.0239), and
[Monte Carlo probability calculation](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
These methods inform the design; they do not approve its application here.

### 5. Integrate artifacts, commands and resume validity

- Extend the existing `verify-protocol` command, evaluator, JSON/HTML evidence,
  intake links and routine/resumed candidate-run reporting. Add batch-summary
  support through the existing pipeline; do not introduce a parallel launcher.
- Publish recording evidence in `Quality checks/<recording-id>/` and run-level
  plots alongside the existing run report. Keep permanent outputs on the
  configured real storage and linked through supported discovery tools. Do not
  add a repository report family or another progress board.
- Bind evidence to recording/experiment identity, selected raw variants and
  hashes, intake hashes, event correspondence, phase/batch provenance, camera
  coverage, shared protocol/configuration and trial-map fingerprints, evaluator
  version and renderer settings. Null outputs additionally bind the complete
  model specification, generator version, seed and simulation count.
- Preserve publication staging and source rechecks. Authenticate once per
  invocation and reuse that verified evidence across plots without weakening
  source-integrity checks. Avoid reading/hashing large tracking tables per plot.
- Invalidate diagnostic evidence on relevant input/configuration/evaluator
  changes; invalidate plots on renderer/display changes. Neither change alone
  invalidates behavioral frames/metrics. Report stale saved analytical mappings
  through existing guards; do not rebuild them from the QC command.
- Update `docs/running-analysis/run-pipeline.md` and `parameters.md` with the
  implemented interface and limitations. Correct the stale minimum-count wording
  to exact 94/79. Record new scientific decisions only after author agreement.

### 6. Validation and completion criteria

Extend the existing verifier tests and add focused renderer, aggregation and
optional-null tests. Cover:

- Complete conditioned/control sessions in all three assays; inclusive timing
  boundaries; missing and extra CS/US events; first-US retention and corrected
  training US 19–64; analysis catch 65; final viability ambiguity.
- Duplicate onsets, overlaps, nonpositive durations, fractional/nonfinite and
  coarse timestamps, clock reset/drift, legitimate log-order interleaving,
  missing internal camera intervals and events outside acquisition.
- Off-target pairs beyond the old search range, phase contamination, unexpected
  pulse durations, and uncertain mappings after missing/extra events.
- Fixed pairing, backward pairing, variable latency, clustering and drift;
  per-fish patterns that cancel in a pooled curve; reused schedules across days;
  matched US schedules; absent provenance and unequal/missing coverage.
- Original/regular raw variants; changed inputs during publication; changed
  experiment context; stale evidence/renderer settings; and resumed runs.

For any implemented null, test its constraints and sampling distribution,
preservation of fixed events and reuse, reproducibility, calibrated false
positive behavior and sensitivity to prespecified timing failures. Do not use
successful fixture generation as evidence of real generator reconstruction.

Validate against authenticated Delay/3sTrace examples and the real source
anomalies above, then obtain independent real fixed-10sTrace coverage. Verify
raw/intake event equality separately from processed event alignment and labels.
Inspect actual rendered plots for onset/offset legibility, sparse events,
clipping, phase boundaries and viability visibility. Benchmark event-only
processing, authentication I/O and simulation cost separately.

Completion requires linked numerical and rendered evidence, explicit coverage
for every expected recording, successful focused checks, demonstrated resume
validity and honest unavailable/uncheckable cases. Synthetic-only 10sTrace
coverage must remain distinct from real validation. Behavioral quality or
learning is linked only through already authenticated evidence and retains its
own eligibility/approval requirements.

## Saved-analysis provenance follow-up

Use the [G/H/I audit's exact source manifest and gaps](../records/scientific-evidence/analyses/figure-02-ghi-statistical-audit.json)
to close original-run reproducibility gaps alongside paper input validation:

- Recover the exact H/I driver and phase-helper revisions still missing from
  supported recovery, plus the G imported fitter/runtime closure. Search by
  recorded identity/hash; extend the existing recovery index only after exact
  verification. Reuse the 86 G artifact aliases and recovered H/I fitter.
- Recover the original producer binding for I's saved bootstrap draw file, if
  available. Its audit snapshot and successful index/percentile replay do not
  establish that historical binding.
- Establish the historical environment/library identities and effective fit
  settings not recorded in saved diagnostics, including optimizer iteration
  defaults. Do not substitute current defaults or rewrite old expected hashes.
- Document exhausted searches and unrecoverable evidence honestly. If an
  independent reproduction is required, prepare it separately on authenticated
  inputs under the owning scientific plan; compare it with the preserved result
  and seek the necessary scientific acceptance. A new run cannot repair the
  identity of the original run retrospectively.

Acceptance for this follow-up is an authenticated source/runtime/producer map,
or explicit reviewed limitations and a separately traceable reproduction route.
It is not another general repository/SSD cleanup audit.

## Acceptance

The expected recording inventory, source/stage identities, code/configuration/environment, timing reconciliation and preprocessing/detector validation evidence are mutually traceable at paper scale. Every unresolved case has a disposition and owner. Existing evidence is reused where authenticated; this plan does not authorize recomputation or refreezing during repository cleanup. Progress is tracked only in the [implementation index](IMPLEMENTATION_STEP_INDEX.md); enduring rules live in [governance](../docs/project-maintenance/governance.md).
