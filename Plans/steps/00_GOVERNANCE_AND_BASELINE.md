# Step 00 — Governance, Paper Scope, and Baseline Freeze

**Status:** Not started  
**Change class:** Observation only; no analytical change  
**Depends on:** None  
**Unlocks:** Every later step

## Objective

Create a complete, immutable description of what the current analysis does,
which data and outputs support the paper, and who can approve scientific and
engineering decisions.

The baseline is a comparison target, not an endorsement of current scientific
behavior.

## Required decisions

- [ ] Name the scientific owner.
- [ ] Name the implementation owner.
- [ ] Decide who approves cohort, metric, model, learner, and figure decisions.
- [ ] Decide the target Python version for the migration.
- [ ] Select the dependency and lockfile approach.
- [ ] Select one test runner.
- [ ] Define where non-Git experimental data and immutable releases reside.
- [ ] Define the paper-scope experiments, conditions, figures, tables, and
      claims.
- [ ] Confirm whether synchronized raw video exists and can be retrieved for
      blinded validation.
- [ ] Define an a priori minimum-fish/power rule for permanent confirmation
      holdout versus nested fish-level cross-fitting.

## Inputs

- Current repository commit and worktree state
- Current numbered scripts and configuration modules
- Raw and processed data locations
- Current manuscript methods, figures, legends, and statistical results
- Current discard/exclusion files
- Existing analysis outputs
- Existing environment(s), including the learner-vigor environment
- Raw video and video-to-tracking synchronization metadata, if available

## Work packages

### 00.1 Create the paper claim inventory

For every manuscript claim, record:

```text
claim_id
claim_text
confirmatory_or_exploratory
experiment_ids
conditions
population/cohort
outcome
analysis window
statistical result
figure/table reference
current source script
current source artifact
```

Unknown sources are blockers, not blanks to silently accept.

### 00.2 Create the recording inventory

Record every paper-scope recording:

```text
recording_id
experiment_id
condition_id
fish_id
day
rig
camera source
tracking source
protocol source
expected trial counts
current processing status
current inclusion status
```

Calculate source hashes once where operationally feasible. If full hashing must
be deferred, mark the identity provisional and record size and modification
time without treating them as equivalent to a cryptographic hash.

### 00.3 Freeze the active current configuration

Capture all effective settings, including values copied or derived across:

- `general_configuration.py`;
- `experiment_configuration.py`;
- top-level constants in each numbered script;
- learner-classifier variants;
- environment variables;
- plotting defaults and fonts;
- implicit filename suffixes and path choices.

The snapshot must preserve inconsistencies between scripts. For example,
different exclusion switches or bootstrap counts are recorded as separate
legacy stage settings.

### 00.4 Run the current baseline

Run the existing workflow unchanged for the agreed representative scope and,
where feasible, the complete current paper scope:

```text
preprocessing
-> QC and protocol plots
-> discard/exclusion workflow
-> fish grouping
-> scaled-vigor data and figures
-> normalized-vigor data and statistics
-> selected learner implementation
-> learner-stratified workflow, if currently used
```

Record exact commands, flags, manual actions, wall time, failures, and output
locations.

### 00.5 Fingerprint outputs

For each table:

- path, format, size, and hash;
- DataFrame shape and index;
- columns, order, dtypes, sparse dtypes, and categorical order;
- exact missing-value mask summary;
- fish, trial, block, and phase counts;
- selected numerical summaries;
- min/max time and expected units.

For each statistical output:

- model input identity;
- formula and reference categories;
- optimizer and random-effects settings;
- row and fish counts;
- coefficients, uncertainty, p-values, corrections, and warnings.

For each figure:

- producing script/function;
- input artifacts;
- canvas size, axes count, labels, units, limits, colors, legend;
- displayed sample sizes;
- output hash and preview.

### 00.6 Select regression fixtures

Select or synthesize:

- control/reference fish;
- apparent learner and non-learner;
- sparse-movement fish;
- tracking-error case;
- missing-frame case;
- catch-trial response;
- low-baseline case;
- opposing tail-section movement case.

Real fixtures require explicit privacy/ethics approval and should be minimal.

### 00.7 Define validation feasibility

Record:

- video availability, format, access, and frame synchronization;
- people/time available for blinded annotation;
- minimum confirmation sample size by condition;
- whether permanent held-out experiments/days are feasible;
- nested fish-level cross-fitting as the fallback when a permanent holdout is
  underpowered;
- a scorecard version and weights for both `video_available` and
  `video_unavailable` modes, frozen before candidate results are inspected.

## Deliverables

```text
baseline/
    baseline-run.json
    paper-claims.csv
    paper-recordings.csv
    source-inventory.csv
    output-inventory.csv
    current-config.json
    environment.json
    cohort-snapshots/
    table-fingerprints.json
    statistical-results.csv
    figure-inventory.csv
    execution-log/
```

These deliverables are generated during implementation; they are not committed
experimental data by default.

## Validation

- [ ] Every paper figure and table has a source path or a recorded blocker.
- [ ] Every paper-scope recording has a unique stable identity.
- [ ] Every active script flag is represented in the configuration snapshot.
- [ ] Baseline outputs are read-only after approval.
- [ ] The environment can be reconstructed or is explicitly documented as
      unreconstructable.
- [ ] Known issues are linked but not corrected during baseline capture.
- [ ] Video-validation and confirmation-partition feasibility are recorded.

## Exit gate

The scientific and implementation owners can trace every important current
paper result to data, code, configuration, cohort, and environment well enough
to compare it with a migrated result.

## Failure conditions

Stop the migration if:

- an important current result cannot be located;
- experiment or fish identity is ambiguous;
- output files were overwritten without a recoverable reference;
- the active learner implementation cannot be established;
- current cohort membership cannot be reconstructed.
