# Classical Conditioning

Analysis of tail tracking from a head-fixed larval-zebrafish conditioning assay. The routine run inventories raw recordings, verifies or creates lossless intake, calculates **three candidate movement metrics** from corrected frames, and renders every descriptive figure whose inputs are available. Candidate metrics and paper panels are not scientific approvals.

For quick review, start with the [numbered analysis guides](docs/analysis/README.md), the [parameter index](docs/analysis/03_ANALYSIS_PARAMETER_INDEX.md), and the [panel-by-panel paper figure provenance](docs/analysis/figures/01_PAPER_PANEL_PROVENANCE.md). The latter covers main Figure 1–4 panels and their supporting analyses and figures.

## Scientific and figure order

1. Inventory and validate raw recordings, make lossless intake, preprocess measured-time tail data, calculate the three exploratory vigor metrics, and apply one shared bout detector.
2. Assess every fish for technical readiness and record exploratory behavior-dependent discard checks after per-fish outcomes exist. Approve a label-independent paper inclusion policy and freeze the reviewed cohort before population or learner claims.
3. Build **Figure 1** from protocol and selected single-fish tail-angle traces, vigor traces, and heatmaps. Prepare its stimulus controls, per-fish metric comparisons, and QC supplements alongside it. Choose one approved metric for final paper panels.
4. Build **Figure 2** from matched frozen cohorts: equal-fish pooled heatmaps, block-level summaries and model contrasts, and trial-level trajectories and onset analysis. Prepare coverage, fish-level trajectories, US-response comparisons, and diagnostics alongside it. When claiming extinction, first define and validate a separate analysis of decline or loss of an established response; the current code does not provide that estimator.
5. Build **Figure 3** from the approved learner representation, fish-level effects, and independent validation, with eligibility and sensitivity data.
6. Build **Figure 4** from the frozen learner manifest and authenticated signed block/catch profiles. Prepare individual catches, movement/coverage, controls, and independently evaluated timing alongside it. Finish the follow-on tail analyses and then freeze one verified analysis release.

The [active plans](Plans/README.md) track decisions and unfinished work. The
[supplementary plan](Plans/10_SUPPLEMENTARY_FIGURES.md) groups supporting
outputs by parent figure; final supplementary numbering is assigned during
paper composition. Optional behavior/imaging integration is a parallel track.

## Set up

Use CPython 3.12 or 3.13. On Windows, install the locked environment:

```powershell
uv sync --frozen --all-extras
uv run python -m unittest discover -s tests
```

Alternatively, install with `python -m pip install -e ".[analysis,interactive]"`. Choose a writable output volume with room for lossless Parquet files. Keep raw acquisition files unchanged. `save_dir` must differ from `raw_dir`; when nested under it, the derived directory must be named `Paper data`.

## Configure one run

Copy [the strict JSON example](configs/example-run.json), edit its paths, then run:

```powershell
uv run classical-conditioning run-pipeline --config configs\example-run.json
```

JSON has no comments; this table documents **every accepted field**. Obsolete route, recipe, `run_intake`, and `run_figures` switches, and all unknown fields, cause a clear error.

| Field | Default | Meaning |
| --- | --- | --- |
| `raw_dir` | required | Read-only directory recursively scanned for camera, tracking, and protocol files. |
| `save_dir` | required | Writable derived-data project. |
| `experiment` | required | Experiment ID: `allDelay`, `all3sTrace`, or `all10sTrace`. |
| `analysis_id` | required | Stable output identity; reruns replace derived files at this identity. |
| `keep_conditions` | all | Condition tokens from filenames, for example `control` and `delay`. |
| `recording_ids` | `null` | Explicit recording IDs; when omitted, use matching inventory records. |
| `overwrite` | `false` | Force rebuilding otherwise reusable derived stages. |
| `continue_on_error` | `true` | Record individual failures and continue other fish; the invocation still reports failure if any required stage fails. |
| `batch_size` | `250000` | Rows per chunk in intake and corrected metric calculation. |
| `figure_mode` | `static` | `static` PNG, or `publication` SVG and PDF with provenance. |
| `show_progress` | `true` | Print progress to stderr. `--quiet` also suppresses it. |
| `cohort_id` | `null` | ID of an existing immutable reviewed cohort; supply together with `metric`. |
| `metric` | `null` | Selected metric for frozen-cohort and learning analysis; supply together with `cohort_id`. |
| `learner_representation_id` | `null` | Reserved identity for a frozen learner representation. Until Gate L is approved, learner figures remain blocked. |
| `assessment_metric` | selected `metric`, otherwise `legacy_distal_angular_speed` | Candidate metric used only for the technical and exploratory discarding assessment; it does not select a paper metric. |
| `technical_policy` | `null` | Path to a reviewed technical policy JSON. Without one, the assessment is a draft evidence audit, not an inclusion decision. |
| `disabled_discard_checks` | `[]` | Named historical checks to disable for an exploratory sensitivity assessment; never changes a reviewed cohort. |

Experiment-specific trial blocks, stimulus timings, catch assignments, and response windows live in [`src/classical_conditioning/config/experiments.py`](src/classical_conditioning/config/experiments.py). The active `all3sTrace` latency is 9 s, whereas the written paper scaffold describes a 13-s expected US; this protocol identity must be reconciled against raw events before paper timing panels are approved. Inspect fully resolved settings and the trial map before a run:

```powershell
uv run classical-conditioning resolve-config --experiment allDelay --project-dir "<SAVE-DIR>"
```

See `classical-conditioning resolve-config --help` for all options. The routine corrected runner identity is `candidate-corrected-runner`. The direct-from-intake calculation is available only as the explicitly selected `candidate-development-runner` benchmark through `candidate-runner`; it is outside `run-pipeline`.

## What the complete run does

1. Inventory **all** recognized raw files and hash their bytes. A raw recording is a matching `*_cam.txt`, `*_mp tail tracking.txt`, and `*_stim control.txt` triplet. Incomplete and ambiguous groups remain visible in the inventory.
2. For each selected fish, verify the source manifest and hashes of all three lossless Parquet files before reuse. A new or changed triplet is ingested transactionally with acquisition QC. Intake statuses `ready`, `incomplete`, and `failed` include reasons in `Metadata/intake_status.json`. An unchanged failed fish is skipped on the next run; after correcting its source or environment, retry it explicitly:

   ```powershell
   uv run classical-conditioning retry-intake --input-dir "<RAW-DIR>" --project-dir "<SAVE-DIR>" --recording-id 20260101_01
   ```

3. For ready fish, run corrected measured-time, gap-aware preprocessing; three frame-level metrics; one shared movement/bout detector; and CS- and US-aligned temporal profiles and per-trial outcomes. A cached stage is reused only when marker, output hashes, recipe, settings, and upstream lineage authenticate. Changed derived outputs are atomically replaced at the same stable paths.
4. Assess technical readiness and exploratory historical discarding rules for **every** inventoried recording after the per-fish outcomes are available, carrying forward intake and candidate-stage failures. Then build the descriptive cohort metric comparison. The run summary links to the authenticated assessment bundle under `Processed data/Discarding/`. This is **not** a scientific inclusion decision. A reviewed cohort is frozen separately with `freeze-cohort` and is never replaced by a routine rerun. See the [discarding assessment guide](./docs/analysis/04_DISCARDING_AND_SELECTION.md) for technical evidence and behavioral rules.
5. As inputs appear, a two-process pool renders intake QC, detector review, five candidate profile families for **CS** and four for **US** per fish (signed log vigor is CS-only), and all five metric-comparison outcomes for **both CS and US**. With a reviewed cohort and selected metric, it also renders five cohort figure families; a matched-control cohort additionally gets a descriptive population heatmap with contributing-fish coverage. Learning-model residual diagnostics follow fitting; the onset figure is rendered only when required diagnostics pass. Every required figure is recorded as `completed`, `failed`, or `blocked` with a reason. Proposed paper panels stay blocked until their scientific gates are met.
6. Always write `Metadata/<analysis_id>_pipeline_run.json`, including after stage failure. A failed run returns a non-zero command status; inspect this summary and the intake ledger before retrying.

The three metric columns are tail-length-weighted angular L1 speed (`rad/ms`), whole-tail XY mean speed (`tail lengths/ms`), and distal angular speed (`rad/ms`). Exact definitions and validity rules are in [`candidate_metric_kernel.py`](src/classical_conditioning/preprocessing/candidate_metric_kernel.py). The detector uses one shared bout segmentation for all three metrics.

### Output locations

| Directory | Contents |
| --- | --- |
| `Processed data/<fish>/` | Three lossless intake Parquets and corrected per-fish tables. |
| `Processed data/Analyses/<analysis_id>/` | Cohort comparison and configured learning-analysis tables. |
| `Processed data/Discarding/` | Authenticated technical and exploratory assessment bundles; their hashes and dispositions are linked from the run summary. |
| `Processed data/Cohorts/<cohort_id>/` | Immutable reviewed cohort manifest. |
| `Quality checks/<fish>/` | Intake report, QC figures, and stage coverage summaries. |
| `Metadata/` | Inventory, source manifest, intake ledger, completion markers, run summary, and resolved configuration. |
| `Figures/PNG/` | Routine static descriptive plots. |
| `Figures/Publication/` | Semantic SVG/PDF and provenance sidecars in publication mode. |

The original pre-refactor source is kept under [`legacy/`](legacy/README.md)
for reference. Intermediate refactors and old plans are available in Git
history. Active outputs have stable names; a new suffix is not stamped for each
run.

## Figures and scientific interpretation

[`configs/paper-figures/behavior-paper.json`](configs/paper-figures/behavior-paper.json) is the proposed Figure 1–4 panel registry. It maps stable IDs to required artifacts, rendering rules, and precise current blocked reasons. The run summary authenticates selected cohort, metric definition, settings, and inventory hashes. Figure 4 now follows the later approved learner-stratified block/catch layout; Figure 3 has no draft image.

Figure 2 requires matched-control **frozen fish cohorts**, not candidate per-fish heatmaps. Its selected blocks are final Pre-Train trials **10–14**, Early Test **65–69**, and Late Test **90–94**. The current descriptive population heatmap shows 0–1 scaled total activity **across all valid frames, including valid zeros**, with a contributing-fish coverage strip and exact counts in panel data. It is not the draft's movement-conditional vigor signal or an approved paper panel. The 10sTrace result is presently inconclusive. Draft significance stars are not reused without approved inference.

The separate paper-review Figure 2 A adapter now pools the same signed,
baseline-centered bout-log-vigor bins used for Figure 1 E and displays them
with `managua_r` at −0.25…+0.25. See the [paper figure freeze record](./docs/analysis/figures/02_PAPER_FIGURE_SPECIFICATION.md)
for the review command and remaining scientific approvals.

Routine catch/block plots show 0–1 scaled activity and are descriptive. The current [Figure 4 learner-stratified route](./docs/analysis/figures/06_FIGURE4_LEARNER_PROFILES.md) instead builds signed, baseline-centered profiles for all three assays from a frozen categorical classifier manifest; its layout must be adapted if Gate L approves a continuous representation. Training catches 11, 25, 39, and 45 correspond to global CS trials 25, 39, 53, and 59. The five-trial pooled catch view also includes global trial 65, the first Test trial. Expected-US guides require verified paired-training events; anticipatory catch suppression is distinct from a direct post-US response.

Static PNG is the routine default. The same renderer can export SVG/PDF with semantic SVG IDs, artist-to-data mappings, physical units, provenance sidecars, and structural checks. See [figure guidance](./docs/analysis/08_CANDIDATE_FIGURES.md) and [draft comparison](./docs/analysis/figures/05_DRAFT_COMPARISON.md).
Publication export also requires a clean Git worktree so its recorded commit contains the rendering code.

## Learning and LME parameters

Use `learning-onset` directly when a prespecified scientific model configuration is needed. Its output is diagnostic until cohort, metric, effect threshold, and model decisions are approved. See the [complete LME parameter reference](./docs/analysis/07_LEARNING_ONSET_LME.md).

| Stage | User-facing parameters |
| --- | --- |
| Identity and data | `--project-dir`, `--cohort-id`, `--analysis-id`, `--metric`, `--outcome` (`total-activity` or `conditional-intensity`), `--alignment` (`CS` or `US`), `--control-condition`, `--test-condition`, `--pretraining-block`, repeated `--late-block`. |
| Eligibility and scale | `--min-baseline-samples`, `--min-response-samples`, `--activity-offset`. |
| Onset criterion | Required `--delta-min`, `--persistence-trials`, `--confidence-level`. |
| Longitudinal LME | `--spline-df`, `--random-effects-formula`, `--optimizer`, `--disable-random-intercept-fallback`, `--skip-categorical-sensitivity`, `--sensitivity-optimizer`, `--skip-random-intercept-sensitivity`. |
| Fish-level uncertainty | `--bootstrap`, `--min-successful-bootstrap`, `--min-bootstrap-success-fraction`, `--permutations`, `--seed`. |
| Publication/rebuild | `--overwrite`; figure commands also take `--mode` (`static`/`publication`) and `--overwrite`. |

The routine config's cohort/metric pair enables a technical fit with packaged defaults. For a non-default effect threshold or sensitivity settings, use `learning-onset` explicitly and inspect its diagnostic artifacts; the paper onset panel remains gated by required diagnostics.

## More detail

- [Current pipeline guide](./docs/analysis/02_CURRENT_PIPELINE.md)
- [Figure generation and draft comparison](./docs/analysis/figures/03_FIGURE_PIPELINES.md)
- [Paper cohort completion plan](Plans/01_COHORT_IMPLEMENTATION.md)
- [Learning-onset completion plan](Plans/03_LEARNING_ONSET_IMPLEMENTATION.md)
- [Learning-onset analysis reference](./docs/analysis/07_LEARNING_ONSET_LME.md)

The active suite in `tests/` protects scientific identities, raw-to-derived provenance, failure handling, and figure semantics.
