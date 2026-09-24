# Behaviour paper figure specification and freeze record

**Version:** working specification 2026-09-24. **Scientific freeze status:** open. This file fixes the intended panel map, the current rendering recipe and the decisions still needed before any panel is labeled final. The [machine panel registry](../../../configs/paper-figures/behavior-paper.json) holds panel IDs and readiness reasons; the [review version comparison](PAPER_FIGURE_VERSIONS.md) records every generated alternative. The written source plan is the separate paper repository's `Helpers/List of Figures.md`, `Helpers/List of Sup. Figures.md`, and `Helpers/Paper_Scaffold.md`. Those files inform the plan; their drafting notes are not executable instructions. The three supplied draft PNGs demonstrate layout and visual intent, with known differences described in [the draft comparison](PAPER_DRAFT_COMPARISON.md).

## Shared figure rules

- Show time relative to CS onset, with −20…0 s as the single-fish baseline and the 10-s CS window indicated consistently. The Pre-Train/Train/Test heatmap order is top to bottom, in chronological order.
- Candidate vigor metrics under review: tail-length-weighted angular L1, normalized whole-tail XY mean speed, and legacy distal angular speed. Use **one chosen metric across corresponding manuscript panels** after scientific selection; retain all three as review outputs.
- Figure 1 C is measured tail angle over time, D is measured frame vigor over time for the *same selected fish and trials*. D's optional orange steps are the exact Figure 1 E signed heatmap bins on a separately labeled axis; gaps remain NaN, with finite-run boundaries.
- Single-fish heatmap proposal: moving, valid positive bout frames; log vigor minus each trial's median log vigor from −20≤t<0 s; bout-median then 0.5-s bin mean; `managua_r`, display −0.25…+0.25, NaN distinct. This is the user-selected **review method**, awaiting metric, example and visual approval.
- Pooled heatmap review: calculate the **same signed bout-log-vigor bins** as Figure 1 for every cohort fish, then average one fish value per condition/trial/0.5-s bin. Do not apply fish or pooled P10/P90 scaling. Use `managua_r` with the same −0.25…+0.25 limits; missing bins remain distinct. S5 shows contributing-fish coverage separately, also with `managua_r` on its own 0…1 fraction scale. This aligns the measure and display, while final cohort and metric approval remain open.
- Figure 2 D–I plot response/pre-CS baseline ratios, which are not heatmap color values. Every inferential label must be generated from saved model outputs and its diagnostic status. Failed influence and unlocalized onset must remain explicit.
- Routine PNGs and publication SVG/PDF need sidecars with source/input hashes, cohort and metric identity, scaling, artist semantics, and any inference table. A displayed draft, a visual match or an exploratory star does not approve a result.

## Main figures

| Figure | Panel plan | Current implementation / freeze gate |
| --- | --- | --- |
| **1. Preparation, protocol and individual behavior** | **A** head-fixed larva, basal illumination, green CS and violet/optovin US. **B** four contingencies (Delay, 3sTrace, 10sTrace, unpaired control), trial protocol and 50/100-ms US. **C** representative tail angle. **D** matching raw vigor trace with explained metric. **E** Delay fish signed heatmap. **F** 3sTrace fish signed heatmap. **G** unpaired-control fish signed heatmap. | C/D and a paired Delay/control E review are renderable; its control half is a candidate for G, not a separate manuscript G panel. A/B need approved artwork. F needs authenticated example data and selection. C/D/E/G need selected fish, common metric and final composition. Reconcile the scaffold's 3sTrace US at 13 s with the active experiment definition's 9 s before B or expected-US guides are approved. |
| **2. Population learning across contingencies** | Columns: Delay/control, 3sTrace/control, 10sTrace/control. **A–C** CS-aligned vigor heatmaps. **D–F** paired fish block ratios, Late Pre-Train 10–14, Early Test 65–69, Late Test 90–94. **G–I** trial-by-trial response ratios with approved model contrasts/onset. Present 10sTrace as inconclusive. | A/D/G Delay review renderers exist; A now pools the Figure 1 signed bins equally across fish. Other columns need matched reviewed cohorts and authenticated processed data. Pick one metric and approve cohort and inference. Current Delay G LME has a failed influence gate and no localized simultaneous onset. |
| **3. Which fish learn?** | **A** classification workflow, **B** continuous scores, **C** threshold/false-positive/uncertainty/stability, **D** learner fractions, **E** all-fish paired change, **F** prespecified positive/negative/intermediate/borderline trajectories, **G** all-eligible-fish validation. | Learner representation and validation (Gate L), outcome metric, examples and classification thresholds are not frozen. No manuscript renderer is approved. |
| **4. Learner-stratified CR profiles** | **A** Delay, **B** 3sTrace, **C** 10sTrace. Each full-size panel has nine declared ten-trial CS blocks and one pooled-catch row; conditioned learners, conditioned nonlearners, control learner-flagged fish, and control nonlearners are overlaid. Signed log-vigor curves cover −20…+20 s with CS and verified expected-US guides. | Analysis and rendering commands exist. Final paper output awaits a frozen Gate L manifest, three reviewed cohorts, and authenticated protocol timing. Same-data strata are descriptive. Individual catches, movement probability and fish coverage are supplementary. Independent response timing is a separate S7 workstream. |

## Supplementary figures and data

The numbers below follow the paper helper list, adapted to the present main-panel boundaries. They are planned slots, not a claim of completed outputs.

| Item | Intended content | Current source/status |
| --- | --- | --- |
| **S1** | Optovin/violet US unconditioned response, pulse-length and artifact controls. | Requires approved response and technical controls. |
| **S2** | Full protocol and baseline behavior: timing, pre-training, bout/metric definitions and trial availability. | Protocol and profile/QC outputs exist; final artwork and cohort audit remain. |
| **S3** | Violet illumination without optovin and relevant sham controls. | Requires controlled comparisons. |
| **S4** | US-aligned paired and unpaired responses, separating direct US effects from anticipatory CR. | CS/US temporal profiles exist; matched analysis and interpretation remain. |
| **S5** | Contributing-fish coverage and QC for **all** population heatmaps, separately from main Figure 2 A–C. | The current Delay review uses 0…1 `managua_r` for coverage; the older `cividis` image remains historical. Trace cohorts remain. |
| **S6** | Individual-fish learning trajectories and example selection context. | Trial outcomes and Delay review trajectories exist; figure composition remains. |
| **S7** | Catch-trial and response-timing analyses, including sensitivity to trial selection. | Catch/block review tools exist; signed timing estimator remains open. Training catch trials 11/25/39/45 correspond to global CS trials 25/39/53/59; some legacy “all catch” plots also include global 65, first Test. State the choice in the panel data. |
| **S8** | Red-CS/visual-control robustness. | Requires cohort and analysis approval. |

The earlier condition-wide 0–1 catch/block profiles remain distinct review outputs; they are not Figure 4 panel data.

## Rendering interface

The integrated production entry point is `classical-conditioning render-paper-panels`, implemented in [`paper_panels.py`](../../../src/classical_conditioning/figures/paper_panels.py) and dispatched by the [CLI](../../../src/classical_conditioning/cli.py). It invokes the versioned SSD figure adapters, verifies their source artifacts through those adapters, and writes `paper-panel-run.json` with the exact commands, settings, artifact sidecars and blocked registry panels. It is **not a test command**. By default it renders current review families for Figure 1 C/D and paired Delay/control E (the control fish is a G candidate), and Figure 2 Delay A/D/G plus S5 coverage. The Figure 2 inference overlay is opt-in because the saved model has not passed its diagnostic gate.

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

1. Record one metric definition/hash, reviewed cohort manifest/hash per comparison, selected fish and global trials, and preprocessing and detector recipe hashes. Verify that Figure 1 E and Figure 2 A use the shared signed-bin function and display limits. Fill the registry's currently null authentication fields from the accepted run.
2. Approve the protocol timeline from raw event records, especially the 3sTrace US time, before timing art or expected-US overlays are final.
3. Approve Gate L for Figures 3–4 and the Figure 2 model/diagnostics, contrasts and multiplicity rule. Independent response-timing claims require the separate S7 protocol. A failed diagnostic does not become significant evidence because a star is visible.
4. Compose final panel groups, captions and supplementary cross references; render PNG, SVG and PDF from the same panel data; inspect visual layout and semantic SVG structure.
5. Store a release manifest with source commit, config/input SHA-256 values, exact render command, panel-data and export hashes, plus reviewer/date. At that point change the relevant registry panel statuses from `blocked` only after the scientific approvals are recorded.
