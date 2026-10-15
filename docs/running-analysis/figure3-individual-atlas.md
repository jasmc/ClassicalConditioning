# Figure 3 exploratory individual-fish atlas

This author-review pilot implements the [individual-fish display plan](../paper/figures/figure-03/individual-fish-display.md).
It uses only `legacy_distal_angular_speed` and keeps the original four classifier
algorithms as comparators. It does not select a paper learner method, validate
individual inference, change cohort membership or freeze Figure 3. Aggregate
Control/Learner/Nonlearner behavioral comparisons remain in Figure 4.

## Resolve the review

The separate [exploratory state](../paper/figures/figure-03/exploratory-state.json)
binds the self-contained `review.html`, editable `overview.svg`, legend, artifact
manifest and exact-render layout evidence. All media and numerical payloads are
under the explicit external destination:

```text
/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure3-learner-exploration
```

The review has four proposed illustrative cases, a complete cross-method
decision matrix, and 57 detailed fish cards. Filters select rule, condition,
decision or fish ID. A failed eligibility check stays **unclassified**; a
control that does not exceed a rule is **reference**. Each card retains the
actual features, score, threshold, directional votes, failure explanations and
protocol-review attributes. No majority-vote label is created.

The initial examples are 20240618_08 (all four rules flag), 20240612_08 (all
four leave unflagged), 20240702_01 (rule disagreement), and 20221123_02 (near
the WIP score threshold). They were chosen after enumerating the full atlas,
are explicitly illustrative, and remain candidates for author review.

## Data and interpretation

The authenticated Figure 2 G report stage `report-7bb6bd520060` supplies the
inventory, schedule, movement detector and bout-only trial outcomes. Its
analysis-state selection and manifest are authenticated before consumption.
Full frame inputs reconstruct all 5,130 scheduled CS5–94 trial rows for 29
Delay and 28 control fish. Every reconstructed baseline/response arithmetic
bout mean matches its authenticated saved outcome (maximum difference zero
in this pilot). `input-lineage.json` records 177 consumed source files.

The heatmap estimator is direct 1 s median log positive bout intensity, minus
the median of defined baseline-bin medians over −15 to 0 s. All fish use
fixed limits −1 to +1, with saturation indicated and numerical values retained.
This display differs deliberately from each classifier's exact transformation;
the WIP trial plot shows `ln(CR/base)` and five-trial medians. Full cards also
show baseline and CR conditional means on the same absolute rad/ms scale.
No-bout intensity is undefined. Moving fractions are shown separately.

`movement.valid` measures **detector-eligible observed time**, not independent
raw tracking completeness. The `Obs B/CR` strip reports this availability; it
must not be read as a coordinate-tracking QC measurement. Measured intensity
is not hidden by an invented 80% coverage threshold. Intensity, moving fraction
and detector availability remain distinct arrays in `heatmaps.npz`.

Actual US timing, CS guides, configured catch trials and original scheduled
trial numbers are retained. Behavioral/protocol flags are attributes, never
implicit exclusions. `classification-manifest.csv` accounts for every
fish × method combination, including per-block/epoch eligibility reasons.

| Original rule | Delay flagged / eligible | Controls flagged / eligible | Unclassified Delay / control |
|---|---:|---:|---:|
| Nominal | 6 / 27 | 1 / 23 | 2 / 5 |
| New | 8 / 26 | 3 / 23 | 3 / 5 |
| Improved | 9 / 26 | 3 / 23 | 3 / 5 |
| WIP | 16 / 26 | 2 / 23 | 3 / 5 |

These fresh results apply the original algorithms to current bout-only inputs;
they do not reproduce the historical pre-policy counts. The original
Improved/WIP runs emitted 3/4 boundary-of-parameter-space convergence warnings.
Logs, settings and original numerical fallbacks are retained. All four default
rules use acquisition and recovery, including late Test. Therefore these
illustrations are same-data explanations, and the Test curves are not held-out
validation. No inferential bands or artificial fitted trajectories are drawn.

Only Delay/control processed inputs were authenticated for this pilot. Raw
Trace folders were present, but an authenticated processed frame/outcome bundle
was unavailable in the checked mounted roots. This is a scope gap; it does not
imply Trace fish were excluded or that their inventory has been reconciled.

## Supported implementation and verification

`scripts/build_figure3_individual_atlas.py` separates `prepare`, `classify`,
`render`, `verify` and `publish`. Use `.venv/bin/python` from the repository
root and an existing directory on mounted JOAQUIM. `prepare` and `classify`
are scientific rebuilds; do not run them to change presentation. `classify`
uses the existing isolated legacy worker and leaves the four originals intact.

For the current review, authenticate without rebuilding:

```sh
.venv/bin/python scripts/build_figure3_individual_atlas.py verify --output-dir '/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure3-learner-exploration'
```

For an authorized presentation change, `render` checks the prior working-output
hashes, snapshots consumed dependencies, renders into a temporary destination,
rechecks dependencies and preserves the previous SVG/HTML pair in external
history. Manual output edits cause failure for reconciliation. New exact-render
browser evidence is required after changing SVG bytes.

`scripts/check_figure3_atlas.cjs` uses the shared layout collector to measure
the overview and all 57 exact card SVGs at 600 dpi. It also checks search,
unclassified filtering, full-inventory reset and page errors. Python
`layout_frames.assess` converts these measurements to `layout-evidence.json`.
Evidence binds the active 1.0 mm policy and exact artifacts; visual review is
still required. The saved browser checks and layout evidence accompany the
artifact manifest. Final source snapshots include the browser checker.

`publish` deep-authenticates all consumed sources and rendering dependencies,
reassesses all 58 exact SVGs against browser evidence, checks the starting state
ticket and publishes only the separate Figure 3 exploratory pointer under the
supported lock. Changes to other figures' pointers are recorded, never
overwritten. The first pilot's starting ticket was captured before creation.

Eight targeted tests cover the existing legacy adapter and atlas semantics,
including no-bout versus unavailable detector time, duration weighting,
unclassified labels and distinct example selection without fabricated agreement.
Review the whole atlas before approving examples or deciding a learner method;
calibration, stability and independent evaluation remain open under Gate L.
