# Figure 4 handover for another computer

## State at handover

The Figure 4 analysis and renderer are implemented, with focused tests and a synthetic layout check. **No paper-data Figure 4 has been run.** The current machine did not have the frozen Gate L classifier manifest or the recording volume mounted. A preview made with synthetic curves is not a scientific result.

The implementation is in `src/classical_conditioning/analysis/figure4.py`, `src/classical_conditioning/figures/figure4.py`, the `figure4-analyze` and `figure4-render` CLI routes, and the `figure4` paper-panel registry route. The input contract is in [the Figure 4 guide](./06_FIGURE4_LEARNER_PROFILES.md). Independent response timing is covered by the [supplementary plan](../../../Plans/10_SUPPLEMENTARY_FIGURES.md).

At the time this handover was written, the working branch was `codex/remove-legacy-package` at `3c1bcdb`, **with uncommitted Figure 4 changes**. A checkout of that commit alone is insufficient. Transfer the complete current working tree, or apply [the working-tree patch](../../../handover/figure4-working-tree.patch) to a checkout of `3c1bcdb` (then review and commit it). The patch includes the untracked Figure 4 guide and this handover; the patch itself is a transfer aid and is not included inside itself. On the receiving machine, inspect `git status --short` and verify that `figure4-analyze` and `figure4-render` appear in `python -m classical_conditioning --help`.

```text
git checkout 3c1bcdb
git apply --check "<PATH_TO>/figure4-working-tree.patch"
git apply "<PATH_TO>/figure4-working-tree.patch"
git status --short
```

## Required inputs

Obtain these from the approved workstreams; do not create substitute labels or change hashes to make a run pass:

1. One selected metric: `tail_length_weighted_angular_l1`, `whole_tail_xy_mean_speed_normalized`, or `legacy_distal_angular_speed`.
2. A reviewed cohort ID for each of `allDelay`, `all3sTrace`, and `all10sTrace`, plus the project directory holding each cohort. Each cohort needs authenticated corrected candidate metrics, movement states, temporal profiles, stimulus events, source manifests, and cohort trial outcomes for its fish.
3. The frozen Gate L fish-classification CSV/Parquet table and its same-stem `.manifest.json` sidecar. Its metric and cohort hashes must match the selection above. It must classify every fish in the three cohorts exactly once, including controls and any `Unclassified` fish with reasons.
4. The three `assessment-summary.json` files named in that sidecar, with all five associated assessment Parquet files still present and hash-matching. The selection assessments must use `tail-candidate-corrected` and the selected metric.
5. A writable output project directory and figure output directory. Keep the input project directories available throughout analysis.

**Path-bound artifacts:** source manifests check the absolute paths of intake Parquet files. Several other artifact summaries also record absolute paths. Copying a processed project to a different location or operating system can invalidate it. The supported choices are to mount/copy it at the same absolute paths, or rerun the authenticated intake and corrected stages on the receiving computer and regenerate the reviewed cohort and selection/classifier manifests as needed. Do not hand-edit hashed manifests; reissue them through their owning workstream. Run Figure 4 analysis on the computer where the final project paths will remain, because its saved summary also records absolute paths.

## Setup and dry run

Use CPython 3.12 or 3.13 and `uv` in the transferred repository:

```text
uv sync --frozen --all-extras
uv run python -m unittest tests.test_figure4 -q
uv run python -m classical_conditioning figure4-analyze --help
uv run python -m classical_conditioning figure4-render --help
```

The focused test uses temporary synthetic input and does not validate the paper cohort. If the full project was rebuilt at new paths, authenticate its cohort and selection outputs before running Figure 4. The Figure 4 command will independently verify the input identities and hashes.

## Run analysis first

Replace every angle-bracket placeholder with a path or ID on the receiving machine. These one-line commands work in a terminal without shell-specific continuation syntax. If all three cohorts live under the output project, omit the three `--*-project-dir` options.

```text
uv run python -m classical_conditioning figure4-analyze --project-dir "<OUTPUT_PROJECT>" --analysis-id figure4-v1 --metric "<METRIC>" --delay-cohort-id "<DELAY_COHORT_ID>" --trace3-cohort-id "<TRACE3_COHORT_ID>" --trace10-cohort-id "<TRACE10_COHORT_ID>" --learner-manifest "<CLASSIFIER_TABLE>" --delay-project-dir "<DELAY_PROJECT>" --trace3-project-dir "<TRACE3_PROJECT>" --trace10-project-dir "<TRACE10_PROJECT>"
```

The command writes `trial-bins.parquet`, `fish-bins.parquet`, `group-bins.parquet`, `sample-flow.parquet`, `analysis.json`, and `complete.json` under:

```text
<OUTPUT_PROJECT>/Processed data/Analyses/figure4-v1/figure4/
```

It computes signed bout-log-vigor in 0.5 s bins from −20 to +20 s with each trial's −20 to 0 s baseline and a 0.9 coverage threshold. Missing no-bout signed values stay missing; movement probability is separate. Trials are pooled within fish before the equal-fish median and IQR. Trial 65 belongs to both Test 1 and the pooled catch group. The pooled catch set is 25, 39, 53, 59, 65. Group tables include fish and trial counts per bin, including explicit zero-fish groups.

**Timing gate:** `ExperimentSpec` currently declares a 9 s paired-US latency for 3sTrace, while the paper scaffold refers to 13 s. The analyzer checks recorded `Reinforcer` events for each conditioned fish against the declared latency and stops on disagreement, missing/extra paired US events, or US events in non-US trials. Inspect the authenticated records and obtain an approved correction to the experiment definition or manuscript description. Do not shift a guide or disable this check to force a figure. If the records agree with 9 s, the written 13 s claim still needs reconciliation before a paper release.

## Render saved analysis

Run only after `complete.json` exists and analysis succeeded:

```text
uv run python -m classical_conditioning figure4-render --analysis-summary "<OUTPUT_PROJECT>/Processed data/Analyses/figure4-v1/figure4/analysis.json" --output-dir "<FIGURE_OUTPUT>" --mode static
```

This produces **18 review PNG figures**: per experiment, one ten-row main figure (nine CS blocks plus pooled catches) and five supplementary figures covering individual signed catches, main and individual movement probability, and main and individual contributing-fish coverage. Four classifier groups, their fish counts, fish-IQR bands where at least two fish contribute, CS interval, and verified paired-training expected-US guides are drawn from saved tables. The x-axis is −20…+20 s; a 20 s US sits at the right boundary.

For SVG/PDF publication files, rerun `figure4-render` with `--mode publication` and a separate clean output directory. The publication exporter requires a clean Git working tree and embeds provenance. Do not use `--overwrite` unless intentionally replacing an output with the same analysis ID and recipe.

The integrated command `render-paper-panels --figure-set figure4` also runs analysis then rendering and records `paper-panel-run.json`; use `--plan` first to inspect its two commands. Its output directory basename becomes the analysis ID. The direct commands above are easier to diagnose on the first paper-data run.

## Acceptance checks and delivery

1. Inspect `analysis.json`: selected metric, three cohort IDs/hashes, classifier execution/validation identity, expected-US evidence for all assays, and four table hashes. Confirm `complete.json` binds that summary.
2. Inspect `sample-flow.parquet` for every cohort fish, including unclassified reasons and controls retaining the `reference` role. Check all four plotted strata and zero-fish counts.
3. Inspect the ten main rows and five individual catches per experiment; confirm all 18 PNGs, their `.figure.json` sidecars, and the ±20 s axes. For publication mode, confirm each figure has SVG and PDF outputs and a provenance sidecar.
4. Confirm the three main figures are Delay (4A), 3sTrace (4B), and 10sTrace (4C). Treat the same-data learner curves as descriptive; add no learner-versus-nonlearner significance marks.
5. Save the exact source commit, selected metric, cohort IDs/hashes, classifier manifest and hash, selection assessment hashes, `analysis.json`, `complete.json`, all four tables, `paper-panel-run.json` if used, and figure files/sidecars together with the review record.

If authentication fails, keep the diagnostic and repair the upstream artifact through its owning pipeline. If rendering fails, preserve the saved analysis tables; the renderer reads them and performs no classification or signal recalculation.
