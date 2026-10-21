# Figure 1: Protocol and individual behavioral examples

A–D, E and F/G/H 1 s medians (2026-10-13) have accepted scoped freezes. The whole figure is not frozen.

Resolve the sole active working SVG and its matching minimal HTML/legend
from [current presentation state](current-state.json), using
`scripts/figure_working_copy.py display-target`. Display that selected pair for
annotation; open its matching SVG when requested. The whole-figure draft remains
unfrozen. Historical reviews and accepted scientific sources remain preserved.

The active draft's displayed H example is 3sTrace fish **20230419_10**, selected
by the author on 2026-10-20. Its source-bound replacement and rendered review
are recorded in `draft.trace3_example_selection` in current state. This example
is unfrozen; the historical F/G/H freeze retains its original fish identities.

- [Review concerns](review.md)

- [Current presentation state](current-state.json): frozen display and active draft, separately authenticated.
- [Assembly instructions](assembly.md)
- [Editable legend](legend.md)

Scientific authority: [scoped selection index](../../../../records/frozen-analyses/selections.md). Shared definitions: [methods](../../methods/README.md).

## Figure 1

A shows preparation/illumination; B shows the four within-trial contingencies; C shows the session timeline; D shows the selected measured tail-angle traces. Their accepted source bytes come from the retained A–D freeze. Older approval dates alone do not make those panels obsolete.

E’s scoped 2026-10-13 freeze uses unbinned raw-baseline-scaled black traces (0–5) and exact rows from frozen source panel F (Delay, displayed as G). The joint [F/G/H definition](panels-f-g-h/README.md#frozen-scientific-definition) owns their processing, eligibility, normalization and display units. E’s current scientific definition is accepted by its separate scoped freeze; D and whole-figure draft changes remain separately recorded in [state](current-state.json).

## Panel composition

These roles describe the current displayed composition. Frozen F/G/H source
identifiers retain their original order; state records the mapping H→F, F→G,
G→H. Displayed H now uses the author-selected replacement above with the same
normalization and display scale. Exact accepted historical definitions remain
in the linked scope index.

| Panel | Purpose |
|---|---|
| A | head-fixed preparation, CS, basal illumination and optovin US |
| B | within-trial CS–US protocol contingencies: Delay, 3sTrace, 10sTrace and unpaired control |
| C | across-session protocol, phase trial counts and US pulse durations |
| D | representative measured tail angle for selected fish and trials |
| E | matching frozen frame-vigor with exact selected 1 s Delay heatmap rows (display G) on a separate color scale |
| F | unpaired-control single-fish 1 s baseline-centered direct log-frame median heatmap |
| G | Delay single-fish 1 s baseline-centered direct log-frame median heatmap |
| H | 3sTrace single-fish 1 s baseline-centered direct log-frame median heatmap |

Current F/G/H selection: [1 s direct log-frame medians, managua_r](../../../../configs/paper-figures/selections/figure1-fgh-historical-one-second-managua-freeze-20261013.json). The 2026-10-09 V12 source is superseded within F/G/H and remains preserved; no whole-figure freeze is inferred.
## Fixed-10sTrace illustrative candidate review

The author-requested 20 October learner/candidate review is complete in the
[separate comparison scope](trace10-learner-selection/current-state.json).
It recommends **20230309_07** provisionally as an illustration, with matched
heatmaps, trial outcomes, full score/coverage/exclusion table and sensitivity
audits on JOAQUIM. This does not replace selected Figure 1 artwork, validate
binary learner categories or freeze any panel. See the
[method/request handoff](../../../../records/figure-reviews/figure-01/trace10-learner-handoff-20261020.md).
