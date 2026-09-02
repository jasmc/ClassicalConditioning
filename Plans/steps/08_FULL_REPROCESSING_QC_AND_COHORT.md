# Step 08 — Full Corrected Reprocessing, QC, and Cohort Freeze

**Status:** In progress (local fixture plumbing)  
**Change class:** Approved scientific correction  
**Depends on:** Steps 05 and 07; scientific gates P, T1, and C0  
**Unlocks:** Corrected paper outcomes and inference

### Local fixture progress (not paper C1)

- [x] `freeze-cohort` for `fixture-two-fish-v1` (`20221115_04`, `20221116_12`)
- [x] Corrected per-recording chain on both fish
- [x] Corrected two-fish metric comparison + runner resume
- [x] Deterministic batch work manifest (`plan-batch` / `batch-work-manifest-v1`)
- [x] Pending/failed/all batch execute (`execute-batch`)
- [ ] Paper-scope QC report and Gate C0/C1 review

## Objective

Apply the frozen corrected preprocessing recipe to every paper-scope recording,
account for all successes and failures, calculate QC from corrected data, and
freeze explicit primary and sensitivity cohorts.

## Preconditions

- [ ] Paper source inventory is frozen.
- [ ] Legacy-equivalent processing passed Step 05.
- [ ] Frame-loss, synchronization, interpolation, filter, and bout policies are
      approved.
- [ ] Tail representation and primary activity recipe passed Step 07.
- [ ] Synthetic and pilot comparisons pass.
- [ ] Technical validity and missing-data policies are approved.
- [ ] Output storage has sufficient capacity and write protection.

Gate C0 approves cohort rules before QC results are reviewed. The resulting
reviewed cohort instance and hash are produced by this step as gate C1.

## Work packages

### 08.1 Plan the batch

Create a deterministic work manifest:

```text
recording_id
source artifact IDs
expected recipe
expected output
status
attempt
failure reason
disposition
```

Support one fish, one condition, one experiment, failed-only resume, and full
paper-scope execution.

### 08.2 Process every recording

For each recording:

1. validate source hashes;
2. ingest and validate raw data;
3. build the selected tail representation;
4. calculate the selected and legacy benchmark metrics;
5. calculate selected movement state and bouts;
6. annotate stimuli and trials;
7. validate processed output;
8. save artifact and provenance atomically;
9. record performance and warnings.

Failed recordings remain in the coverage report. They never disappear from the
denominator silently.

### 08.3 Compare corrected with legacy

Report per recording and experiment:

- retained frames and trials;
- frame-loss decisions;
- tracking-valid fraction;
- activity distributions;
- bout counts and durations;
- baseline and response summaries;
- output coverage;
- processing failures;
- large numerical differences;
- whether cohort-relevant QC changes.

The report explains expected differences from approved corrections and flags
unexpected differences for investigation.

### 08.4 Calculate technical QC

QC dimensions include:

- raw file completeness;
- frame sequence and timestamp quality;
- tracking-point availability;
- valid-tail fraction;
- implausible jumps;
- protocol completeness and timing;
- CS/US alignment;
- baseline recording coverage;
- response-window coverage;
- processing warnings;
- rig/day anomalies.

Technical QC must not depend on whether the fish shows the hypothesized
conditioned suppression.

### 08.5 Generate review artifacts

- Per-recording QC table
- Protocol timing plots
- Tracking/tail representation diagnostics
- Legacy-versus-corrected activity comparison
- Trial coverage heatmaps
- Draft exclusion reasons

Review plots are derived artifacts and never modify or move raw sources.

### 08.6 Draft cohort manifest

Minimum fields:

```text
experiment_id
fish_id
condition_id
technical_valid
technical_exclusion_reason
behavioral_engagement
behavioral_engagement_reason
primary_included
sensitivity_population_ids
review_status
reviewer
reviewed_at
source_qc_artifact_id
```

### 08.7 Separate populations

Define:

- primary intention-to-analyze population;
- strict technical-validity sensitivity population;
- legacy-equivalent cohort;
- legacy behavioral-engagement cohort, if required for comparison;
- any prespecified minimum-coverage population.

Do not make response-window movement a primary inclusion requirement when
reduced movement is the outcome.

### 08.8 Freeze cohort

After review:

- canonicalize rows and categories;
- validate uniqueness;
- save Parquet plus CSV review copy;
- calculate content hash;
- prevent in-place changes;
- issue a new cohort version for later amendments.

### 08.9 Apply cohort by validated join

Use many-to-one validation. Unknown and duplicate fish stop processing and
produce explicit unmatched reports.

## APIs

```python
run_batch(recordings, recipe, selection=None) -> BatchResult
calculate_qc(processed: ArtifactRef, config: QCConfig) -> ArtifactRef
draft_cohort(qc: ArtifactRef, config: CohortConfig) -> CohortManifest
freeze_cohort(reviewed_table: pd.DataFrame, policy_id: str) -> CohortManifest
apply_cohort(data: pd.DataFrame, manifest: CohortManifest) -> pd.DataFrame
```

## Deliverables

- Full corrected per-recording artifacts
- Batch execution and coverage report
- Legacy-versus-corrected difference report
- QC tables and figures
- Draft and approved cohort manifests
- Sample-flow table
- Unmatched/duplicate reports
- Full provenance and environment record

## Exit gate

Every paper-scope recording is accounted for, every failed recording has an
approved disposition, corrected artifacts are the sole input to corrected
cohort construction, and gate C1 freezes the primary/sensitivity cohort hashes.

## Downstream invalidation

- Preprocessing recipe change: repeat full processing and QC
- QC calculation change only: QC and cohort review onward
- Cohort decision change: cohort-dependent outcomes, statistics,
  classification, and figures
- Display-only QC plot change: QC rendering only
