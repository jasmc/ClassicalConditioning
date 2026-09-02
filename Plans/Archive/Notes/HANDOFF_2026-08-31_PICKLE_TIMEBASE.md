# Handoff — local analysis migration (2026-08-31)

Use this file to start a **new conversation**. Do not reload pickle/raw TXT
into the model; paths and JSON summaries only.

## Context

- **Repo:** `C:\Users\Public\More projects\ClassicalConditioning`
- **Branch:** `refactor/local-analysis-pipeline`
- **Paper data:** `C:\Users\Public\More projects\Paper data`
- **Rule:** never Databricks; keep README updated when CLI/behavior changes;
  atomic semantic commits; pickles stay on disk (metadata/compare JSON only).

## Priority lane

See `Plans/DECISIONS.md`. Immediate path: Steps 03–05 at real data, then
candidate metrics at cohort scale. Learners and imaging deferred.

## What just finished (Step 05 pickle equivalence)

### Fish available locally

| ID | Raw | Intake | `samples_legacy-v1.parquet` | Historical gzip pickle |
| --- | --- | --- | --- | --- |
| `20221115_04` | yes | yes | yes (10,836,172 rows; 94 CS / 78 US) | `Pickle single fish data\20221115_04_…_6dpf.pkl` |
| `20221116_12` | yes (isolate dir) | yes (status REVIEW: camera/tracking frame-range boundary warning) | yes (same row/trial counts) | `Pickle single fish data\20221116_12_…_7dpf.pkl` |

Raw isolate for intake (exactly one triplet per `--input-dir`):

```text
Paper data\Raw single fish data\_intake_20221116_12\
```

Compare reports:

```text
Paper data\Quality checks\20221115_04\legacy-pickle-vs-parquet.json
Paper data\Quality checks\20221116_12\legacy-pickle-vs-parquet.json
```

### Key scientific finding (do not “fix” by matching the pickle)

Legacy uses two rates:

- **expected = 700 Hz** — analysis / trial-time grid (`Trial time (frame) [700 FPS]`).
- **predicted ≈ 702.56 Hz** — measured camera rate.

Interpolate (current `analysis_utils` / package `interpolate_legacy`):

```text
FrameID *= expected / predicted
→ resample onto integer expected-rate times
```

On that grid, absolute acquisition frames (`Original frame number`) must
advance at **`predicted / expected`**. Parquet does this.

Historical pickles advance Original frame at the **reciprocal**
(`expected / predicted`) within each trial. CS onset (`t = 0`) still agrees
(~same acquisition moment). Away from onset, naive trial-time alignment
compares different acquisition moments. Lag grows roughly as:

```text
Δ ≈ t × (expected/predicted − predicted/expected)
```

At trial edge (`|t| ≈ 31500` ≈ 45 s): ~**231 acquisition frames ≈ 0.33 s**.
Near CS: ~0. After rate-warp resampling, CS1 vigor corr recovers to
~**0.996** / ~**0.988** (fish 04 / 12); lag vs that prediction corr ~**0.999**.

**Decision:** keep `legacy-paper-v1` interpolate as written. Do **not** invert
it to reproduce pickle Original-frame slopes. Pickle = historical timebase
inconsistency with current source; Parquet = faithful current interpolate.

Separate quirk (not the pickle warp): after stim marking, AbsoluteTime is
rebuilt with `1000/predicted` ms per row while rows sit on the 700 grid
(dual clock). Trials/vigor use trial time.

Recorded in: `Plans/Archive/steps/05_…`, `docs/analysis/ANALYSIS_ISSUES.md` (H1),
`docs/analysis/CURRENT_IMPLEMENTATION_STATUS.md`, README compare section.

### Tools added

```powershell
uv run classical-conditioning compare-legacy-pickle `
  --pickle "...\Pickle single fish data\<fish>.pkl" `
  --parquet "...\Processed data\<id>\samples_legacy-v1.parquet" `
  --output "...\Quality checks\<id>\legacy-pickle-vs-parquet.json" `
  --overwrite
```

Also: `scripts/inspect_legacy_pickle_metadata.py` (counts/columns only).

Gate T0 (already decided): raw `angleN` = radians; legacy vigor uses degrees;
XY = tracking-image pixels.

## Suggested next work

Superseded by
[HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md](./HANDOFF_2026-08-31_TWOFISH_CANDIDATES.md).
Local fish are **refactor debug fixtures**, not analysis targets. Continue
full package implementation (corrected preprocessing, batch/QC, outcomes,
mixed-effects); re-run fixtures only to debug new code. Do not invert
`legacy-paper-v1` interpolate to match pickles.

## Commands cheat sheet

```powershell
uv sync --frozen --all-extras
uv run python -m unittest discover -s tests

uv run classical-conditioning inventory `
  --input-dir "C:\Users\Public\More projects\Paper data\Raw single fish data" `
  --skip-hashes --inspect-tracking-headers

uv run classical-conditioning intake `
  --input-dir "C:\Users\Public\More projects\Paper data\Raw single fish data\_intake_20221116_12" `
  --project-dir "C:\Users\Public\More projects\Paper data"

uv run classical-conditioning preprocess `
  --project-dir "C:\Users\Public\More projects\Paper data" `
  --recording-id 20221116_12 `
  --recipe legacy-paper-v1 --experiment allDelay
```

## Recent commits (this thread)

```text
fddd791 docs: record pickle timebase warp explains naive trial-time diffs
e5f051b test(preprocessing): cover reciprocal-rate warp vigor recovery
99ae17e feat(cli): print rate-warp vigor agreement from pickle compare
a01019b feat(preprocessing): diagnose pickle reciprocal-rate timebase warp
… earlier: compare-legacy-pickle, metadata script, Original-frame classification
```
