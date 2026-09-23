# Candidate figure guide

Figures are derived views of authenticated artifacts. The routine run renders
every family whose inputs are ready, including both CS and US alignments for
per-fish profiles and all five comparison outcomes. They are exploratory and do
not make an approved scientific claim by themselves. Preserve the source table,
QC summary, and provenance sidecar with each rendered figure.

## Output modes

| Mode | Primary output | Intended use |
| --- | --- | --- |
| `static` | PNG | Fast inspection and routine local review. |
| `publication` | SVG/PDF plus provenance sidecar | High-quality export with explicit source context. |
| `interactive` | Self-contained HTML, profile figures only | Inspection with hover values. |

Static figures go below `Figures/PNG/`; publication figures go below
`Figures/Publication/`; interactive profile figures go below
`Figures/Interactive/`. Cohort-level figures include `Analyses/<analysis-id>`
in their paths.

## Per-recording temporal profiles

Use `figure-candidate-profiles` with one of these selections:

| Figure | Rows | Cell value |
| --- | --- | --- |
| `total-activity-raw` | Three activity metrics | Mean valid metric value per bin, native units. |
| `total-activity-scaled` | Three activity metrics | Historical two-layer per-trial scaled value. |
| `conditional-intensity-raw` | Three activity metrics | Mean metric value only during a shared detected bout. |
| `bout-outcomes` | Three detector outcomes | Movement probability, time moving, or bout rate. |

Example:

```powershell
uv run classical-conditioning figure-candidate-profiles `
  --project-dir "<SAVE>" --recording-id <RECORDING-ID> `
  --trial-type CS --figure total-activity-raw --mode static `
  --recipe candidate-temporal-outcomes-corrected
```

All profile figures use time relative to the selected stimulus event on the
x-axis and trial number on the y-axis. Blank/masked cells are not zero: a bin
is drawn only when its expected-frame coverage reaches the configured gate.
This makes missing or invalid tracking visible rather than presenting it as low
activity.

## How to interpret the four figure families

`total-activity-raw` combines whether a fish moved and how strongly it moved.
`conditional-intensity-raw` isolates the latter by averaging only within the
single shared bout segmentation. A no-bout window is `NaN`, not zero.

`total-activity-scaled` applies a two-layer historical visualization transform:
first, each trial is scaled with pre-onset frame quantiles; then its binned
pre-onset values are scaled and clipped to `[0, 1]`. It is useful for temporal
pattern inspection, not for comparing physical magnitude across metrics.

`bout-outcomes` is metric-free. The detector is run once from its dedicated
distal cumulative-angle source; its movement probability, fraction of valid
time moving, and bout-onset rate are repeated in profile data for convenience
but should not be interpreted as three independent metric measurements.

## Cohort metric comparison

```powershell
uv run classical-conditioning figure-metric-comparison `
  --project-dir "<SAVE>" --analysis-id <ANALYSIS-ID> `
  --trial-type CS --outcome movement-probability --mode static
```

The cohort figure represents one fish-level standardized difference per metric
and condition. Each fish contributes equally; it is descriptive, not
confirmatory inference. Use the matching processed comparison table and QC
summary to check inclusion, baseline/response coverage, and recipe identity.

## Frozen-cohort response and CR profiles

The cohort response commands require an immutable `cohort_id` and one selected
metric. The three ratio commands render selected blocks, all CS trials, or an
event-aligned response/baseline trajectory. Two temporal-profile commands use
the authenticated `Scaled total activity` field:

```powershell
uv run classical-conditioning figure-cohort-catch-profile `
  --project-dir "<SAVE>" --analysis-id <ANALYSIS-ID> `
  --cohort-id <COHORT-ID> --metric <METRIC-ID> --mode static

uv run classical-conditioning figure-cohort-block-profile `
  --project-dir "<SAVE>" --analysis-id <ANALYSIS-ID> `
  --cohort-id <COHORT-ID> --metric <METRIC-ID> --mode static
```

Catch membership comes from the experiment definition; the migrated Delay,
3-second Trace, and 10-second Trace assays use CS 25, 39, 53, 59, and 65. Trial
65 is the first Early Test trial. The catch command pools those trials within
fish. The block command uses every declared CS ten-trial block. Both mask bins
below `--minimum-coverage` (default 0.9), pool trials within fish first, and
then give each fish equal weight in the condition median and IQR.

The selected-block ratio figure uses final Pre-Train trials 10–14, Early Test
65–69, and Late Test 90–94. Its response/baseline ratios are not the same
quantity as the 0–1 scaled catch/block profiles. A signed, baseline-centered
paper Figure 4 representation remains gated on an approved definition and
frozen learner identity; current descriptive plots cannot be relabelled as it.

These are descriptive population figures. Learner-stratified versions remain
gated on the frozen learner-representation manifest and validation mode.

When the reviewed cohort contains one paired condition and its matched
control, the routine run also renders a CS-aligned population heatmap from one
value per fish/trial/time bin. It displays 0–1 scaled total activity across
**all valid frames, including valid zeros**; a second row displays the
contributing-fish fraction and the panel-data Parquet records exact counts.
This is not the movement-conditional vigor estimand in draft Figure 2.

## Colour and comparison rules

- Scaled profiles use one `[0, 1]` colour scale because the transform creates a
  shared display scale.
- Native-unit intensity and bout-outcome rows use separate colour bars because
  their units/ranges differ. Do not compare colour intensity across such rows.
- Compare candidate metrics through the cohort table/figure, where each metric
  is labelled and the fish-level aggregation rule is explicit.

For the underlying scientific definitions, see [metrics and bouts](METRICS_AND_BOUTS.md).
For the full legacy/refactored figure inventory, see
[figure pipelines](figures/FIGURE_PIPELINES.md).

## Detailed heatmap reference

Two candidate figure families are available. Both accept `--mode`: `static`
(PNG), `publication` (SVG/PDF with provenance sidecar), and, for profiles,
`interactive` (self-contained HTML with hover values).

For every per-fish heatmap, the x-axis is time from stimulus onset, from −45 s
to +45 s in 0.5 s bins (180 bins); zero is CS or US onset selected by
`--trial-type`. The y-axis is trial number, and a shaded vertical band marks the
stimulus window. Blank cells are masked, not zero: a bin is drawn only when at
least 90% of its expected frames are usable, where expected frames are bin width
divided by median frame interval. This prevents dropped/invalid tracking from
masquerading as low activity.

Rows in Figures 1–3 are the three genuinely distinct tail-motion metrics:

| Figure | Cell value |
| --- | --- |
| `total-activity-raw` | Mean metric over valid frames in the bin, native units. |
| `total-activity-scaled` | Same quantity after historical two-layer per-trial scaling. |
| `conditional-intensity-raw` | Mean metric only over frames inside a detected bout, native units. |

Conditional intensity separates how hard the animal moved from how often it
moved: approximately, `total activity ≈ fraction of time moving × conditional
intensity`. The scaled display first applies `(v − P10) / (P90 − P10)` to frames
per trial, using samples earlier than −15 s; after binning it applies the same
transform over pre-onset bins and clips to `[0, 1]`. A trial with no usable
pre-baseline window is `NaN`, never self-rescaled.

`bout-outcomes` has no metric dimension. Its rows are movement probability
(moving frames/detector-valid frames), fraction of time moving (moving time/
detector-valid time weighted by `DeltaTimeMs`), and bout-onset rate (onsets ×
60000 / valid time). One detector supplies one segmentation; the profile table
repeats these values across metric rows only for convenient storage, and the
figure reads one copy rather than drawing duplicate data.

Scaled profiles use a shared `[0, 1]` colour bar. Native-unit activity and bout
figures use a colour bar per row because units/ranges differ; within bout
outcomes, proportions are `[0, 1]` and bout rate has its own 99th-percentile
limit. Colour is therefore not comparable between native-unit rows.

Each export records input/source hashes, its reproduction command, and an
artist-level mapping of source column, coverage field, threshold, display scale,
and shared-detector status. The cohort comparison collapses each fish to the
standardized difference `(response − baseline) / baseline SD`; bars are grouped
by metric and condition. It is descriptive only and makes no inferential or
paper-approved claim.
