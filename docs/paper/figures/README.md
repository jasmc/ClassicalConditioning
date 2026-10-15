# Paper figures

The first paper release covers Figures 1–4. Figure folders are the main workspace; they own panel composition, scientific purpose, legends and unresolved review. Shared methods are linked once.

| Figure | Purpose | Accepted scope |
|---|---|---|
| [Figure 1](figure-01/README.md) | Protocol and individual behavioral examples | A–D, E and F/G/H 1 s medians (2026-10-13); whole assembly not frozen |
| [Figure 2](figure-02/README.md) | Population responses and inference | A pooled 1 s medians (2026-10-14), D/E B and G LogMedian; other panels proposed |
| [Figure 3](figure-03/README.md) | Individual learning and validation | Proposed |
| [Figure 4](figure-04/README.md) | Conditioned-response temporal profiles | Proposed |

- [Iteration and handoff](shared/iteration-workflow.md): SVG working copies, review HTML and guarded state publication.
- [Shared assembly safeguards](shared/assembly-preflight.md)
- [Cross-figure editorial notes](shared/review.md)
- [Exact scientific selections](../../../records/frozen-analyses/selections.md)
- [Methods](../methods/README.md), [decisions](../decisions.md) and [paper-wide review](../review.md)

Current means explicitly selected for a scope, not newest filename. A frozen panel does not freeze its whole assembly. A current presentation draft does not supersede scientific selection. Historical freezes remain accepted evidence within their recorded scope.

The [main-paper collection](main-paper-current-state.json) selects Figures 1 and 2
for one continuous SVG-and-legend HTML. Run `.venv/bin/python
scripts/main_paper_figures.py build` before plural paper-figure display requests;
resolve its exact authenticated HTML with `display-target`. Individual figure
and panel requests keep their scoped views. Inclusion is presentation selection,
not a whole-figure scientific freeze.

Supplementary content is grouped by parent figure and owned by [Plan 09](../../../Plans/09_SUPPLEMENTARY_FIGURES.md). Progress belongs only in the [implementation index](../../../Plans/IMPLEMENTATION_STEP_INDEX.md).

Current F/G/H selection: [1 s direct log-frame medians, managua_r](/Users/joaquim/Documents/Codex/2026-09-15/clo/ClassicalConditioning/configs/paper-figures/selections/figure1-fgh-historical-one-second-managua-freeze-20261013.json). The 2026-10-09 V12 source is superseded within F/G/H and remains preserved; no whole-figure freeze is inferred.

Current E selection: [unbinned raw-baseline-scaled traces, 0–5, current F bands](../../../configs/paper-figures/selections/figure1-panel-e-raw-frame-freeze-20261013.json); E only is frozen.
