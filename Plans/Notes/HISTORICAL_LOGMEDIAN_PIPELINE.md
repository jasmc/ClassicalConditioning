# Historical Handwritten LogMedian Pipeline Interpretation

## Source and role

Historical source:

```text
commit f372101e4e1083ab9dade27445625e31a515a96e
file 0_Pipeline_Data_Flow_Description.txt
```

This handwritten description is important context for refactoring. It describes
the intended data flow through preprocessing and a later LogMedian analysis
variant. It is treated as an intent/history source alongside the executable
code, not as proof of exact implemented behavior.

## Described routes

### Shared acquisition and preprocessing

```text
camera + multi-point tracking + stimulus protocol
-> frame/timing validation
-> tracking validation
-> synchronization and interpolation
-> stimulus annotation
-> cumulative tail angles
-> filtering
-> distal vigor
-> bout detection
-> trials and blocks
-> per-fish output
```

### Standard route on `main`

The current `main` scripts use percentile-scaled vigor and ratio-style
normalized vigor in several stages.

### Historical LogMedian route

The branch containing the handwritten description adds a distinct downstream
route:

```text
bout mask
-> right-aligned rolling median
-> downsample
-> log positive vigor
-> subtract pre-stimulus log median
-> grouped heatmaps and line plots
-> trial response-minus-baseline log median
-> summaries and mixed models
```

This route must be preserved and characterized as a separate analysis recipe,
not silently merged into the standard legacy baseline.

## Statements requiring code verification

The handwritten document states or implies behavior that conflicts with the
current code audit:

- spatial filtering is applied;
- a two-stage bout threshold uses the secondary threshold;
- the learning response is described as anticipatory movement rather than the
  suppression framing used elsewhere;
- file exclusion and downstream cohort behavior are simpler than the executable
  scripts indicate.

Therefore, before moving each operation:

1. inspect the exact executable function and active constants;
2. characterize it with a test;
3. compare it with this handwritten intent;
4. classify disagreement as documentation drift, legacy behavior, or an
   approved scientific correction;
5. preserve Standard and LogMedian routes as different recipes where their
   mathematics differ.

## Useful branch work to port later

The following commits contain potentially reusable learner/figure work:

```text
4d6939f Add learner-stratified vigor analysis pipeline
f372101 Add catch-trial timing analysis
```

They will be reviewed and selectively ported during learner Step 11. The entire
branch will not become the refactor base because preceding commits rename or
delete much of the current pipeline and encode a different scientific route.

## Executable verification

The route was verified against commit
`e4fe3f48d66916174f7fcaa4a5a18be5d49431f3`, which introduced:

```text
3_FishGrouping_LogMedian.py
4_ScaledVigorPlotting_LogMedian.py
5_NormalizedVigorPlotting_LogMedian.py
6_LearnersQuantification_WIP_LogMedian.py
```

Verified stage-3 order:

```text
bout-mask vigor
-> stable fish/trial/time order
-> right-aligned 10-frame rolling median
-> retain every tenth row
-> log only positive vigor
-> subtract each trial's median earlier than -15 seconds
```

Stage 4 consumes `_new_logmedian` artifacts, omits the standard heatmap P10/P90
rescaling, and displays baseline-subtracted log-median values directly. Stage 5
uses median response minus median baseline rather than the standard ratio of
arithmetic means.

The package recipe `historical-logmedian-v1` now preserves the verified stage-3
transformation from frozen `legacy-paper-v1` Parquet. It writes lossless Zstandard
Parquet plus authenticated JSON and never creates a new pickle. This is a
legacy-reproduction route, not a corrected or paper-approved analysis.

The executable contract includes the historical pre-transform coercions that
affect results: `Fish` construction from `Day` and `Fish no.`, int32 trial/time
columns, and float32 vigor input and rolling-median output before the float64
log transform. Shared fixtures match an independent transcription of the
historical `process_data` implementation exactly.

## Bounded-memory pilot verification

Pilot `20221115_04` was processed locally from the authenticated
`legacy-paper-v1` Parquet:

```text
10,836,172 source rows
-> 172 contiguous, time-monotonic trial groups
-> maximum 63,001 source rows held as one scientific group
-> 1,083,772 historical-logmedian-v1 rows
```

The final run completed in 32.4 seconds and produced a 111,425,573-byte Zstandard
Parquet with one row group per trial (94 CS and 78 US). The artifact SHA-256 is:

```text
fceb8c9fd8634508f78109045cb4a08076390cae646db74376454bd16ab14473
```

The summary and completion marker authenticate this hash and the frozen input
hash. Independent reference execution on the first CS trial and final US trial
matched the published rows exactly.

Streaming is intentionally rejected when trial groups reappear, trial time
does not strictly increase within a group, or source group order differs from
the historical global sort order. This prevents bounded-memory execution from
silently changing historical row order or making tied-time downsampling
ambiguous.

## Recovered pickle limitation

No historical stage-1 pickle was found locally. Two large pooled artifacts were
found and hashed read-only:

```text
control_CS_new_logmedian.pkl
  5,531,425,570 bytes
  SHA-256 52cbe59775c1568c4e8479d487160d40d4959053c09b88288ade14349335f75b

trace_CS_new_logmedian.pkl
  4,770,233,312 bytes
  SHA-256 25733a18c8e062a04c05a48d1ce1ccf714e5d0e41df3cccdaef4dfd359ec3cd8
```

Their pickle prefixes identify pandas DataFrames and their names identify the
pooled CS LogMedian route. They have no authenticated lineage sidecars, cannot
be inspected with bounded memory, and are not evidence for stage-1 or pilot
equivalence. They remain immutable historical artifacts.
