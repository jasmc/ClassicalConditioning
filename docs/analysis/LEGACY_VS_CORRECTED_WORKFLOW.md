# Legacy versus Corrected Analysis Workflow

## Purpose

Every improved analysis stage is implemented beside the frozen previous route.
The improved route never overwrites or masquerades as legacy output.

## Artifact pattern

```text
samples_legacy-v1.parquet
samples_corrected-v1.parquet
legacy-vs-corrected.json
```

Equivalent version pairs are used for downstream trial outcomes, statistics,
classification, and panel data.

## Required comparison notes

For every corrected stage, record:

| Field | Meaning |
| --- | --- |
| Previous behavior | Exact executable legacy operation |
| Identified issue | Code defect, manuscript disagreement, or scientific limitation |
| Corrected behavior | New mathematical/technical definition |
| Decision authority | Scientific decision or engineering correction |
| Expected effect | Which rows, fish, metrics, models, or figures may change |
| Measured effect | Counts, missingness, numerical differences, cohort changes, effects, and uncertainty |
| Manuscript impact | Methods/results/figures requiring revision |

## Current status

### Available

- Lossless source ingestion
- Acquisition integrity report
- Frozen stage-1 `legacy-paper-v1` preprocessing
- Row-aligned Parquet comparison utility
- Measured-XY/measured-time candidate activity table
- Machine-readable legacy-versus-candidate route report
- Measured-time CS/US-aligned total-activity profiles
- Candidate movement probability, fraction moving, conditional intensity, bout
  rate, and duration
- Static PNG, self-contained interactive HTML, and semantic SVG/PDF candidate
  figures from the same profile artifact

Pilot comparison:

```text
legacy final trial samples: 10,836,172
candidate overlapping frames: 8,361,387
candidate valid derivatives: 8,361,386
candidate geometry validation coverage: 100%
```

The row counts are not directly comparable: legacy output duplicates frames
into CS- and US-aligned trial windows, while the candidate table currently has
one row per overlapping acquisition frame.

The pilot raw protocol contains 94 CS and 79 US events. The legacy stage-1
route retains 94 CS and 78 US trials after its startup crop; the measured-time
candidate route represents all 94 CS and 79 US events, including zero-coverage
bins if an event falls outside the available recording. This is an explicit
route difference requiring scientific interpretation, not an automatic claim
that either population should be used for final inference.

The candidate figure is descriptive single-fish evidence. It is useful for
metric validation but cannot select the paper metric or establish a population
learning effect.

### Planned corrected comparisons

| Area | Legacy | Corrected candidate |
| --- | --- | --- |
| Input tracking | Angles only; final row dropped | Preserve measured XY and all valid rows |
| Frame loss | Accumulated timestamp drift | Direct FrameID plus timestamp-gap validation |
| Timing | Uniform 700 FPS interpolation/extrapolation | Measured timing with bounded gap policy |
| Filtering | Temporal mean only | Approved spatial and temporal filtering |
| Activity | Distal cumulative-angle speed | Six explicit candidate metrics (five whole-tail/segment candidates + one legacy-derived benchmark) |
| Bout detection | Primary threshold only | Calibrated versioned detector |
| Rest | Often converted to missing | Total activity plus movement probability and conditional intensity |
| Scaling | Early-baseline P10/P90 | Approved bounded analytical baseline definition |
| Cohort | Stage-specific switches | Immutable primary and sensitivity manifests |
| Uncertainty | Potential row/trial bootstrap | Fish-level or hierarchical bootstrap |
| Statistics | Multiple fragile local models | Prespecified diagnosed longitudinal model |
| Learners | Four incompatible variants | One versioned validation-aware classifier, if retained |

## Rebuilding existing candidate artifacts

Because `legacy_distal_angular_speed_rad_per_ms` is a sixth column in `CANDIDATE_COLUMNS`,
existing `frame_activity_candidates*.parquet` and downstream candidate Parquet tables
created prior to this addition lack the column. Run candidate rebuilds with `--overwrite`
to re-extract all six columns uniformly:

```powershell
python -m classical_conditioning candidate-runner `
  --project-dir <project_dir> `
  --recording-id <recording_id> `
  --analysis-id <analysis_id> `
  --recipe candidate-corrected-runner-v1 `
  --overwrite
```

## Comparison command

```powershell
python -m classical_conditioning compare `
  --left <legacy.parquet> `
  --right <corrected.parquet> `
  --output <comparison.json>
```

The report records:

- input hashes;
- row and column inventory;
- schema and categorical differences;
- null-mask differences;
- exact integer/category mismatches;
- numeric mismatches and maximum/mean absolute differences under declared
  tolerances.
