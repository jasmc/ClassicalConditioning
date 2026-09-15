# Paper figure automation plan

**Status:** Proposed
**Parent workstream:** Step 12 — figures, CLI, and notebooks
**Paper source inspected:** `ClassicalConditioningPaper` at commit `e17149e`
**Recommended initial scope:** Behavioral paper, Figures 1–2 and Supplementary
Figures S1–S6; keep Figures 3–4 gated on learner and timing decisions.

## 1. Goal

Build every paper figure from explicit, versioned analysis artifacts with one
command, while making it easy to answer:

- which fish, trials, conditions, metric, detector, windows, and transformations
  are shown;
- which values are data, summaries, model estimates, or annotations;
- why an artifact was rebuilt or reused;
- which code, configuration, and source artifacts produced every panel; and
- whether a display-only change can be made without changing the analysis.

The figure system must never hide scientific calculations inside plotting code.

## 2. Paper-plan inventory and authority

The cloned paper repository currently contains:

- `Helpers/List of Figures.md`: four planned main figures;
- `Helpers/List of Sup. Figures.md`: eight planned supplementary figures;
- `Helpers/Paper.md`: detailed proposed panels and narrative;
- `Helpers/Milestones.md`: alternative paper scopes and additional panel detail;
- `main.tex` and `sections/*.tex`: no active `figure`/`includegraphics` blocks;
- no committed raster or vector figure files.

These sources are not fully synchronized. In particular, the milestone document
uses an alternative supplementary grouping and numbering. Use this precedence:

1. a frozen paper release manifest;
2. the figure registry proposed below;
3. `Helpers/List of Figures.md` and `Helpers/List of Sup. Figures.md` for order;
4. `Helpers/Paper.md` for candidate panel content;
5. `Helpers/Milestones.md` for scope decisions and scientific gates.

Before a publication build, copy the approved registry summary back into the
paper repository so the prose plan and executable figure plan cannot drift.

## 3. Planned paper figures

### Main figures

| Order | Figure | Short purpose | Automation status |
| --- | --- | --- | --- |
| 1 | Experimental setup, behavioral preparation, protocols, and single-fish evidence | Explain the assay and show representative delay-conditioning behavior | Partly automatable; setup and tracking schematics require approved source artwork |
| 2 | Population evidence for delay and trace conditioning | Primary learning result for delay, 3-s trace, and cautious 10-s trace comparisons | Requires population panel builders and frozen cohort/statistics |
| 3 | Learner classification | Continuous learning scores, threshold behavior, uncertainty, and representative heterogeneity | Blocked on Step 11 scientific decision and validation |
| 4 | Conditioned-response dynamics and timing | Catch/test time courses, CS versus expected-US alignment, and timing measures | Blocked on independent timing definitions and non-circular analysis gate |

### Supplementary figures

| Order | Figure | Short purpose | Principal dependency |
| --- | --- | --- | --- |
| S1 | Optovin-evoked responses and baseline activation | Validate the US and post-stimulus response | US-aligned outcomes and dose/pulse metadata |
| S2 | Detailed protocol and baseline maintenance | Show every protocol phase and demonstrate usable baseline movement | Protocol registry and baseline outcome panels |
| S3 | Violet light without optovin | Demonstrate specificity of optovin-mediated reinforcement | No-optovin experiment cohort |
| S4 | US-aligned paired and unpaired responses | Separate anticipatory CR from evoked UR | US-aligned profiles and matched cohorts |
| S5 | Coverage and heatmap QC | Show how many fish/frames contribute to displayed summaries | Coverage panel data from every population figure |
| S6 | Individual learning trajectories | Show heterogeneity without making binary classes the primary evidence | Fish-level trial outcomes |
| S7 | Additional timing and catch-trial analyses | Support response timing and alignment conclusions | Timing-analysis artifacts |
| S8 | Robustness across visual-CS conditions | Compare red-CS-only with pooled red/white-CS results | Frozen sensitivity cohorts |

## 4. Recommended panel blueprint

Panel letters are provisional until the layouts are reviewed. Each row should
become a set of executable panel specifications rather than bespoke script
logic.

### Figure 1

| Panel | Content | Source type |
| --- | --- | --- |
| A | Head-fixed preparation, free tail, CS and US arrangement | Approved manual/vector asset |
| B | Experimental phase and delay/control timing diagram | Protocol metadata renderer |
| C | Tail points and definition of the selected activity metric | Approved schematic plus formula metadata |
| D | Representative US-evoked raw tail/activity trace | Prespecified representative-recording panel data |
| E | Representative delay and control trial heatmaps/traces | Per-recording temporal-profile panel data |

Only delay-conditioning larval data should appear in Figure 1. A separate
decision is required on whether trace protocol diagrams may appear before trace
data are introduced in Figure 2.

### Figure 2

Use a repeated matched-comparison grammar for delay, 3-s trace, and 10-s trace:

| Panel family | Content |
| --- | --- |
| Population heatmap | CS-aligned selected primary outcome by trial and time |
| Learning trajectory | Fish-equal trial or block summary with raw fish visible |
| Primary contrast | Late pre-training versus early test, with estimate and uncertainty |
| Extinction contrast | Early versus late test where prespecified and informative |
| Cross-condition summary | Delay versus trace effect estimates on one comparable scale |

The 10-s trace result must be labeled as a boundary condition unless a frozen
analysis supports a stronger claim.

### Figure 3

| Panel family | Content |
| --- | --- |
| Classifier definition | Inputs, eligible population, continuous score, and threshold derivation |
| Score distributions | Delay, 3-s trace, and matched controls before binarization |
| Threshold uncertainty | Resampling interval and assignment stability |
| Fractions | Classifier-positive fractions plus control false positives |
| Paired changes | Fish-level late-pre-training to early-test changes |
| Examples | Positive, negative, intermediate, and borderline fish |
| Independence check | Primary population result using all eligible fish |

No Figure 3 implementation begins until the continuous measure, threshold,
validation population, and missing-data policy are frozen.

### Figure 4

| Panel family | Content |
| --- | --- |
| Catch/test profiles | Delay and 3-s trace, classifier-positive and all eligible fish |
| Alignment comparison | The same data aligned to CS onset and expected US |
| Acquisition dynamics | Response development by trial or training block |
| Timing estimates | Onset, peak, center of mass, duration, or offset with uncertainty |
| Trace-interval test | Explicit suppression inside the 3-s stimulus-free interval |
| Sensitivity | Alternative timing measure/window fixed before confirmation |

Timing panel data must be computed independently of the learner-classification
window to avoid circular evidence.

## 5. Executable figure registry

Create one versioned JSON configuration, initially:

```text
configs/paper-figures/behavior-paper-v1.json
```

Every panel entry must declare:

```text
paper_id                     behavior-paper-v1
figure_id                    fig-02
panel_id                     fig-02-a
title                        internal descriptive title
renderer_id                  population-temporal-heatmap-v1
panel_data_recipe            population-temporal-panel-v1
input_artifact_ids           exact recipes/artifacts
cohort_id + cohort_hash      biological population
conditions                   ordered IDs
recording/fish selector      explicit or rule-based
trial types and trials       CS/US, catch/test/block membership
metric_id                    one of the three registered metrics
detector_id                  shared detector where relevant
outcome_id                   total activity, movement probability, etc.
alignment                    CS onset, CS offset, US, or expected US
baseline/response windows    references to frozen named windows
aggregation                  frame → trial → fish → condition order
uncertainty                  method, level, cluster unit, seed
statistical_result_ids       precomputed model/contrast artifacts
display transform            raw, log, standardized, scaled
coverage rule                threshold and visual encoding
axes                         labels, units, limits, ticks
annotations                  stimulus windows and approved comparisons
dimensions                   width, height, and intended column span
output modes                 review PNG and publication SVG/PDF
```

Scientific fields must reference named, versioned recipes. The figure file may
override display fields such as dimensions or label wording, but it must not
silently override cohorts, windows, aggregation, exclusions, or statistics.

## 6. Software architecture

```text
analysis artifacts + cohort/model manifests
                |
                v
resolved paper figure specification
                |
                v
versioned panel-data builders
  Parquet: values used by renderers
  JSON: windows, tests, sample sizes, hashes
                |
                v
reusable renderers
  heatmap | trace | paired change | distribution | estimate | schematic
                |
                v
major-figure composer
  panel placement and lettering only
                |
                v
PNG review + SVG/PDF publication + figure sidecar
                |
                v
structural QC + visual regression + release manifest
                |
                v
optional controlled sync into ClassicalConditioningPaper/figures/generated
```

Recommended package layout:

```text
src/classical_conditioning/figures/paper/
    registry.py              load and validate figure specifications
    resolve.py               resolve recipes, cohorts, artifacts, and hashes
    panel_data.py            common typed panel-data interfaces
    builders/                scientific panel-data builders
    renderers/               display-only reusable panel renderers
    compose.py               page layout and panel lettering
    build.py                 orchestration, caching, and manifests
    validate.py              structural and provenance checks
```

The existing `FigureMode`, `FigureProvenance`, semantic SVG, PDF, PNG, and
sidecar machinery should be reused rather than replaced.

## 7. User controls and commands

Add one top-level command family:

```text
classical-conditioning paper-figures plan --spec ...
classical-conditioning paper-figures explain --spec ... --panel fig-02-a
classical-conditioning paper-figures build --spec ... [--figure fig-02]
classical-conditioning paper-figures validate --spec ...
classical-conditioning paper-figures manifest --spec ...
```

Required behavior:

- `plan` is read-only and lists selected panels, resolved input hashes, cache
  status, missing dependencies, and planned outputs;
- `explain` prints the full scientific and display contract for one panel in
  plain language;
- `build` can target the entire paper, one figure, or one panel;
- `validate` checks inputs, structure, labels, dimensions, provenance, and
  approved visual baselines;
- `manifest` records the complete reproducible paper-figure release;
- `--dry-run` performs no writes;
- `--force` is limited to explicitly selected non-release outputs;
- publication releases are immutable and require a clean Git commit.

For review, every figure should also export a human-readable resolved-spec file
and the exact panel-data tables. A scientist must be able to inspect the plotted
numbers without reverse-engineering Matplotlib objects.

## 8. Output contract

```text
Figures/Paper/behavior-paper-v1/
    resolved-spec.json
    build-manifest.json
    panel-data/
        fig-02-a.parquet
        fig-02-a.json
    Review/
        Figure_02.png
        panels/fig-02-a.png
    Publication/
        Figure_02.svg
        Figure_02.pdf
        Figure_02.figure.json
        panels/fig-02-a.svg
```

The paper repository should consume only stable publication paths, for example:

```text
ClassicalConditioningPaper/figures/generated/behavior-paper-v1/Figure_02.pdf
```

Syncing is a separate explicit step. Analysis builds must not overwrite paper
files implicitly.

## 9. Implementation sequence

### Phase 0 — freeze scope and naming

1. Choose Milestone 1 or Milestone 2 as the first paper release.
2. Confirm four-main/eight-supplement order or approve a revised registry.
3. Decide whether Figure 1 contains trace protocol diagrams.
4. Freeze the primary metric, companion outcomes, detector, cohorts, windows,
   and biological replicate.
5. Record decisions in `Plans/DECISIONS.md`.

### Phase 1 — registry and inspectability

1. Implement typed figure/panel specs and JSON validation.
2. Implement `plan`, `explain`, targeted selection, and `--dry-run`.
3. Resolve every scientific field to an artifact and hash.
4. Fail on unknown conditions, trials, metrics, windows, or stale lineage.

Exit: Figure 2 can be completely described without rendering it.

### Phase 2 — panel-data layer

Implement reusable builders in this order:

1. temporal heatmap plus coverage;
2. fish-level trial/block trajectory;
3. paired phase change;
4. condition/model estimate with uncertainty;
5. raw representative trace;
6. protocol timeline;
7. score distribution and classification diagnostics;
8. response-timing estimates.

Exit: every plotted number exists in an inspectable panel-data artifact.

### Phase 3 — first end-to-end paper figure

Build Figure 2 first because it contains the primary scientific claim and
exercises heatmaps, trajectories, contrasts, cohorts, statistics, labels, and
multi-panel composition. Generate review PNGs before publication SVG/PDF.

Exit: Figure 2 rebuilds from a clean checkout with one command and passes all
structural tests.

### Phase 4 — assay and validation figures

1. Build automated data panels for Figure 1 and S1–S6.
2. Add approved setup/tracking assets to a versioned assets directory.
3. Compose manual assets and generated panels through the same manifest.
4. Verify representative-fish selection is prespecified and documented.

Exit: Milestone 1 paper figures build automatically.

### Phase 5 — learner and timing figures

Only after their scientific gates pass:

1. implement Figure 3 and its validation panels;
2. implement Figure 4 with independent timing outcomes;
3. implement S7 timing extensions;
4. implement S8 visual-CS cohort sensitivity.

Exit: Milestone 2 figure release builds automatically without modifying the
Milestone 1 analysis definitions.

### Phase 6 — paper integration and release

1. Validate all figures from a clean analysis commit.
2. Freeze a figure release manifest with environment and input hashes.
3. Explicitly sync publication PDFs into the paper repository.
4. Generate or update LaTeX figure blocks from the registry.
5. Compile the paper and verify every referenced figure and caption mapping.

## 10. Tests and quality gates

### Scientific/structural tests

- exact cohort and cohort hash;
- exact metric, outcome, detector, alignment, and named windows;
- declared frame → trial → fish → condition aggregation order;
- fish is the uncertainty/resampling unit where required;
- sample sizes and exclusions agree with manifests;
- statistics are loaded from model artifacts, never recomputed by renderers;
- heatmap coverage and missingness are visible and tested;
- stable condition order, colors, units, labels, stimulus marks, and panel IDs;
- expected number of panels and output formats;
- renderer changes do not modify panel-data hashes.

### Visual tests

- review PNG regression at pinned backend/font versions;
- no clipped text, legends, labels, or confidence intervals;
- publication dimensions match single/double-column specifications;
- rasterized heatmaps remain embedded correctly in SVG/PDF;
- representative traces remain readable at final print size;
- color remains interpretable in grayscale and for common color-vision deficits.

### Release gate

A figure is paper-authoritative only if:

1. all upstream scientific gates are approved;
2. the worktree and environment are recorded;
3. inputs, panel data, rendered outputs, and sidecars hash successfully;
4. structural and visual QC pass;
5. the resolved spec is reviewed and approved; and
6. the immutable release manifest includes it.

## 11. Immediate next actions

1. Treat the cloned paper lists as planning input, not yet as frozen truth.
2. Resolve the Figure 1 trace-diagram ambiguity and choose Milestone 1 versus 2.
3. Create `behavior-paper-v1.json` with Figure 2 panel placeholders.
4. Inventory which required conditions and experiment metadata are represented
   in the current configuration registry and local data.
5. Implement the read-only registry validator plus `plan` and `explain` first.
6. Build Figure 2 as the first complete automated paper figure.

## 12. Current readiness and concrete blockers

| Requirement | Current state | Consequence |
| --- | --- | --- |
| Delay and matched control | Registered as `allDelay` | Available for panel-data prototyping after cohort review |
| Short-trace experiment | `fixedVsIncreasingTrace` is registered, but its display name, CS duration, US latency, and paper label need explicit reconciliation | Do not label outputs “3-s trace” until protocol identity is validated |
| 10-s trace and matched control | No explicit package experiment specification found | Figure 2 boundary-condition panels cannot yet be built |
| No-optovin/violet-only control | No explicit package experiment specification found | S3 is blocked |
| Red-CS-only sensitivity cohort | No versioned cohort definition found | S8 is blocked |
| Catch-trial identity | `TrialSpec.catch` exists, but current generated trial specifications do not mark catch trials | Figure 4/S7 catch panels are blocked |
| Expected-US alignment | Current trial-outcome summary explicitly says expected-US identities are not emitted | Figure 4 alignment comparison is blocked |
| Learner classification | Planned/deferred; no approved active classifier | Figure 3 is blocked |
| Population inference | Fish bootstrap, permutation, and mixed-effects modules exist | Must be frozen, orchestrated, and referenced by panel data rather than run inside plotting |
| Figure rendering | Per-recording profile heatmaps and cohort metric-comparison bars exist | Useful reusable foundations, but not major-paper compositions |
| Manual figure assets | None are committed in the paper repository | Figure 1 setup/tracking panels need an asset-source decision |
| LaTeX integration | No active figure environments or `includegraphics` references found | Registry-driven figure blocks can be introduced without preserving an existing layout contract |

This readiness check makes Figure 2’s delay/control portion the best first
vertical slice. It can establish the architecture without pretending that the
trace, classifier, timing, and sensitivity dependencies are already resolved.
