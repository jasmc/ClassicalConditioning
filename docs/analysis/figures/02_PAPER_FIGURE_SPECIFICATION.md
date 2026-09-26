# Behaviour paper figure specification and freeze record

**Version:** working specification 2026-09-24. **Scientific freeze status:** open. This file fixes the intended panel map, the current rendering recipe and the decisions still needed before any panel is labeled final. The [machine panel registry](../../../configs/paper-figures/behavior-paper.json) holds panel IDs and readiness reasons; the [review version comparison](./04_REVIEW_VARIANTS.md) records every generated alternative. The written source plan is the separate paper repository's `Helpers/List of Figures.md`, `Helpers/List of Sup. Figures.md`, and `Helpers/Paper_Scaffold.md`. Those files inform the plan; their drafting notes are not executable instructions. The three supplied draft PNGs demonstrate layout and visual intent, with known differences described in [the draft comparison](./05_DRAFT_COMPARISON.md).

## Shared figure rules

- Show time relative to CS onset, with −20…0 s as the single-fish baseline and the 10-s CS window indicated consistently. The Pre-Train/Train/Test heatmap order is top to bottom, in chronological order.
- Candidate vigor metrics under review: tail-length-weighted angular L1, normalized whole-tail XY mean speed, and legacy distal angular speed. Use **one chosen metric across corresponding manuscript panels** after scientific selection; retain all three as review outputs.
- Figure 1 B compares within-trial CS–US contingencies; C shows the across-session protocol. D is measured tail angle over time, and E is measured frame vigor over time for the *same selected fish and trials*. E's optional orange steps are the exact Figure 1 F/H signed heatmap bins on a separately labeled axis; gaps remain NaN, with finite-run boundaries.
- Single-fish heatmap proposal: moving, valid positive bout frames; log vigor minus each trial's median log vigor from −20≤t<0 s; bout-median then 0.5-s bin mean; `managua_r`, display −0.25…+0.25, NaN distinct. This is the user-selected **review method**, awaiting metric, example and visual approval.
- Pooled heatmap review: calculate the **same signed bout-log-vigor bins** as Figure 1 for every cohort fish, then average one fish value per condition/trial/0.5-s bin. Do not apply fish or pooled P10/P90 scaling. Use `managua_r` with the same −0.25…+0.25 limits; missing bins remain distinct. The Figure 2 supplement shows contributing-fish coverage separately, also with `managua_r` on its own 0…1 fraction scale. This aligns the measure and display, while final cohort and metric approval remain open.
- Figure 2 D–I plot response/pre-CS baseline ratios, which are not heatmap color values. Every inferential label must be generated from saved model outputs and its diagnostic status. Failed influence and unlocalized onset must remain explicit.
- The assay definitions specify conditioned-response windows of 9 s (Delay), 13 s (3sTrace), and 20 s (10sTrace); the current per-trial and metric-comparison builders use those values. Old Trace trial-outcome artifacts made with the former 9 s default must be rebuilt, followed by cohort/model artifacts and ratio figures, before cross-assay Figure 2 interpretation.
- Routine PNGs and publication SVG/PDF need sidecars with source/input hashes, cohort and metric identity, scaling, artist semantics, and any inference table. A displayed draft, a visual match or an exploratory star does not approve a result.

## Main figures

| Figure | Panel plan | Current implementation / freeze gate |
| --- | --- | --- |
| **1. Preparation, protocol and individual behavior** | **A** head-fixed larva, basal illumination, green CS and violet/optovin US. **B** within-trial timing of Delay, 3sTrace, 10sTrace and unpaired control. **C** session phases, trial counts and 50/100-ms US legend. **D** representative tail angle. **E** matching raw vigor trace. **F** Delay fish signed heatmap. **G** 3sTrace fish signed heatmap. **H** unpaired-control fish signed heatmap. | B/C have vector review sources in the [assembly layout](../../../configs/paper-figures/figure1-assembly.json). D/E and a paired Delay/control F/H review are renderable; G has an authenticated exploratory example (`20230307_12`). Artwork, fish, metric and final composition still need approval. The source-protocol audit and active experiment definition agree on a 13 s paired-US onset; every plotted fish still requires protocol verification. |
| **2. Population learning across contingencies** | Columns: Delay/control, 3sTrace/control, 10sTrace/control. **A–C** CS-aligned vigor heatmaps. **D–F** paired fish block ratios, Late Pre-Train 10–14, Early Test 65–69, Late Test 90–94. **G–I** trial-by-trial response ratios with approved model contrasts/onset. Present 10sTrace as inconclusive. | A/D/G Delay review renderers exist; A now pools the Figure 1 signed bins equally across fish. The 3sTrace column has an exploratory cohort and signed heatmap; its ratio panels require the corrected 0–13 s trial outcomes. The 10sTrace column still needs a reviewed cohort and processed data. Pick one metric and approve cohort and inference. Current Delay G LME has a failed influence gate and no localized simultaneous onset. No extinction estimator is implemented; an extinction claim needs a separate prespecified analysis. |
| **3. Which fish learn?** | **A** approved representation workflow, **B** continuous fish effects or scores, **C** uncertainty/stability and threshold calibration if a threshold is approved, **D** class fractions only if categories are approved, **E** all-fish paired change, **F** prespecified example trajectories, **G** all-eligible-fish validation. | Learner representation and validation (Gate L), outcome metric, examples and any classification thresholds are not frozen. Panel composition changes if Gate L rejects categories. No manuscript renderer is approved. |
| **4. Learner-stratified CR profiles** | **A** Delay, **B** 3sTrace, **C** 10sTrace. The current categorical review layout has nine declared ten-trial CS blocks and one pooled-catch row with learner/nonlearner and control classifier groups. Signed log-vigor curves cover −20…+20 s with CS and verified expected-US guides. | Analysis and rendering commands exist for a categorical manifest. Final paper output awaits Gate L, three reviewed cohorts, and authenticated protocol timing; a continuous or model-based Gate L decision requires a different Figure 4 grouping and renderer. Same-data strata are descriptive. Individual catches, movement probability and fish coverage are supplementary. Independent response timing is gated in the [supplementary plan](../../../Plans/10_SUPPLEMENTARY_FIGURES.md). |

## Supplementary figures and data

Build the supporting data and figures alongside their parent main figure. These
are proposed content groups; assign supplementary figure numbers and panel
letters after the analyses and composition are approved. The detailed work is
in [Plan 10](../../../Plans/10_SUPPLEMENTARY_FIGURES.md).

| Parent figure | Supporting content | Current source/status |
| --- | --- | --- |
| **Figure 1** | Optovin/violet unconditioned response, 50/100-ms pulse and artifact controls; violet-only and sham comparisons; full protocol, baseline, bout/metric definitions, and trial availability. | Protocol and per-fish QC/profile outputs exist. Approved controls, artwork, and final cohort audit remain open. |
| **Figure 2** | US-aligned paired/unpaired responses; contributing-fish coverage and QC for every pooled heatmap; individual-fish trajectories and example-selection context; block/trial model diagnostics and sensitivity. | CS/US profiles, Delay coverage review, and trial outcomes exist. Trace cohorts, final metric/cohort, approved inference, and any extinction analysis remain open. |
| **Figure 3** | Full fish-score distribution, eligibility, threshold calibration if used, control false positives, method sensitivity, and held-out or cross-fitted validation. | Gate L is open; no final learner validation artifacts or paper renderer exist. |
| **Figure 4** | Individual catches, movement probability, signed-signal coverage, red-CS/visual controls, and independently evaluated response timing with catch-selection sensitivity. | Catch/block review tools and Figure 4 supplementary renderers exist. Timing estimator, protocol reconciliation, independent evaluation, and visual-control approval remain open. Training catch trials 11/25/39/45 map to global CS 25/39/53/59; the pooled set may also include global 65, first Test. Record the exact set in panel data. |

The earlier condition-wide 0–1 catch/block profiles remain distinct review
outputs; they are not Figure 4 panel data. The [Figure 4 command and manifest
contract](./06_FIGURE4_LEARNER_PROFILES.md) describes the replacement signed
learner-stratified analysis.

## Rendering interface

The integrated production entry point is `classical-conditioning render-paper-panels`, implemented in [`paper_panels.py`](../../../src/classical_conditioning/figures/paper_panels.py) and dispatched by the [CLI](../../../src/classical_conditioning/cli.py). It invokes the versioned SSD figure adapters, verifies their source artifacts through those adapters, and writes `paper-panel-run.json` with the exact commands, settings, artifact sidecars and blocked registry panels. It is **not a test command**. By default it renders current review families for Figure 1 D/E and paired Delay/control F/H, and Figure 2 Delay A/D/G plus separate heatmap coverage. The Figure 1 B/C vector assembly is a separate review route. The Figure 2 inference overlay is opt-in because the saved model has not passed its diagnostic gate.

```bash
MPLCONFIGDIR=/private/tmp/cc-mpl classical-conditioning render-paper-panels \
  --project-dir '/Volumes/JOAQUIM/Digested Data/allDelay-full-v1' \
  --output-dir outputs/paper-figures/review-2026-09-24 \
  --metric tail_length_weighted_angular_l1 --plan

MPLCONFIGDIR=/private/tmp/cc-mpl classical-conditioning render-paper-panels \
  --project-dir '/Volumes/JOAQUIM/Digested Data/allDelay-full-v1' \
  --output-dir outputs/paper-figures/review-2026-09-24 \
  --metric tail_length_weighted_angular_l1
```

`--figure-set figure1` or `figure2-delay` limits the run. `--metric` selects one candidate for all produced images. `--include-inference-review` adds exploratory Figure 2 D/G inferential versions for tail-length-weighted angular L1; it does not approve their marks. `--mode publication` requests SVG/PDF through the supported adapters; the exploratory inference renderer currently exports static only. The signed heatmap adapter currently accepts only the provisional `20221115_07`/`20221115_09` pair. Other figure families deliberately remain blocked in the registry until their scientific inputs exist; this command reports them rather than producing substitutes.

## Conditions for a true paper freeze

1. Record one metric definition/hash, reviewed cohort manifest/hash per comparison, selected fish and global trials, and preprocessing and detector recipe hashes. Verify that Figure 1 F/H and Figure 2 A use the shared signed-bin function and display limits. Fill the registry's currently null authentication fields from the accepted run.
2. Approve the protocol timeline from raw event records, especially the 3sTrace US time, before timing art or expected-US overlays are final.
3. Approve Gate L for Figures 3–4 and the Figure 2 model/diagnostics, contrasts and multiplicity rule. Independent response-timing claims require the evaluation protocol in the [supplementary plan](../../../Plans/10_SUPPLEMENTARY_FIGURES.md). A failed diagnostic does not become significant evidence because a star is visible.
4. Compose final panel groups, captions and supplementary cross references; render PNG, SVG and PDF from the same panel data; inspect visual layout and semantic SVG structure.
5. Store a release manifest with source commit, config/input SHA-256 values, exact render command, panel-data and export hashes, plus reviewer/date. At that point change the relevant registry panel statuses from `blocked` only after the scientific approvals are recorded.

