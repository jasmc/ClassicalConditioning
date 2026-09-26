# Figure 1B condition inset: SVG redesign plan

**Status:** exploratory v4 is selected in the current Figure 1 assembly.
Its SVG is
`J:\ClassicalConditioning Outputs\ORGER-JOAQUIM\outputs\figure1-assembly\schemes\Fig1_PanelB_ConditionTiming_exploratory_v4.svg`.
The standalone composition is `panels/Fig1_PanelB.svg`. Only SVG is a figure
source; PNG is a visual check.

The first exploratory source is
`schemes/Fig1_PanelB_ConditionTiming_exploratory.svg`, with standalone
`panels/Fig1_PanelB_exploratory.svg` and whole-figure
`figure1-exploratory-B.svg`. A second exploratory source,
`schemes/Fig1_PanelB_ConditionTiming_exploratory_v2.svg`, uses filled violet
US-onset dots to match C. Its control row contains **three miniature example
trials**, one US onset before, during, or after the CS. Their positions are
illustrative, not measured frequencies or three US events in one trial. Its
standalone composition is `panels/Fig1_PanelB_exploratory_v2.svg` and its joint
comparison with C is `figure1-exploratory-BC.svg`. All versions are saved on
the SSD. V3 changed Control to one full-size green CS bar with three
alternative US dots, but looked like three US events in one trial. V4 places
each dot on its own example-trial lane, with a full-size green CS bar and a
question-mark timing tick. The original B source remains archived; v4 is
selected in `figure1-assembly.json`.

The remaining sections record the original design analysis and earlier
proposals. The selected v4 implementation is authoritative for this preview.

## Purpose of panel B

B explains the **within-trial temporal contingency** between the green CS and
purple US across the three conditioning assays and their unpaired controls.
Its key comparison is one common 10 s CS versus paired US onset at 9, 13, or
20 s, plus the absence of a fixed CS-relative US onset in control. It is not
a session timeline, a count of trials, a behavioral result, or a specification
of US pulse width. The single control row is a schematic of the unpaired
relationship common to the three assays, not a claim that their control fish
are pooled into one group. Panel C handles phases, trial counts, and the
50/100 ms pulse-length legend.

## Scientific content to retain

The active [`ExperimentSpec`](../../../src/classical_conditioning/config/experiments.py)
defines a 10 s CS in all three assays and paired US onset at 9 s (Delay),
13 s (3sTrace), and 20 s (10sTrace) relative to CS onset. The control
conditions have no fixed US latency in that specification. The raw 3sTrace
audit supports 13 s; a historical provenance note that says 9 s is stale.
Panel C carries the session phases, trial counts, and 50/100 ms US legend.
Panel B should explain contingency timing without repeating C's schedule or
assigning one US pulse length to an assay without evidence.

## Problems in the current B artwork

| Scheme | Current issue |
| --- | --- |
| All four | Separate axes look equivalent, but the right ends do not cover the same depicted event range. Title baselines, time labels, and label spacing vary. |
| Delay | The 9 s US and 10 s CS-end labels nearly collide. The 9 label is vertically offset from other tick labels. |
| 3sTrace | 10 and 13 are close at the present text size. |
| 10sTrace | The 20 s US marker is beyond the drawn arrowed time axis, even though the green CS bar has the same 0–10 scale as the others. Its orange title also fails to distinguish it from 3sTrace in the experiment palette. |
| Control | A purple US arrow appears just before 0 s with a question mark. That placement can be read as a specific pre-CS US onset, while the condition has no fixed latency. |
| Time label | Only one of the four `t (s)` strings renders visibly; the other three use a class with `fill:none`. |

## Recommended design

Rebuild B as a **new, editable SVG** from a small timing specification rather
than moving the old Illustrator objects one by one. Keep Asset 10 unchanged as
the comparison reference.

1. Use four evenly spaced horizontal rows inside B's existing rounded frame:
   Control, Delay, 3 s trace, and 10 s trace. This puts the unpaired reference
   first, followed by the paired conditions in increasing US-latency order.
   Reserve a left label column, a common −5–22 s plot column, and a narrow
   right annotation column. Set all CS onsets on one vertical guide and one
   linear scale. The 0–10 s green CS bar has the same length and position in
   each paired row and each small control example. The negative portion is
   needed to show that an unpaired US can precede CS onset. The shared axis
   sits under the rows, labeled
   "Time relative to CS onset (s)".
2. Draw one filled purple **US onset dot** at 9, 13, or 20 s in the paired
   rows. Keep the 20 s marker within the axis. Retain 0, 10, and 20 as shared
   axis labels. State the exact paired onset and its relationship to CS offset
   in a separate annotation column; the 9/10 numerals therefore cannot
   overlap. The 3 s and 10 s trace gaps remain visible on the common scale.
3. The first proposal gave control one 0–10 s CS bar without a US marker. The
   second proposal makes the intended variability visible: three short
   CS-aligned example rows, each with one filled US dot before, during, or
   after the CS. It states that these are illustrative trials. The negative
   time region has an open-ended axis arrow, avoiding a suggested earliest
   control US time. No single fixed control US latency or empirical onset
   distribution is claimed.
4. Use DejaVu Sans and the existing figure's CS green and US purple. Put an
   exact experiment-palette stripe beside each dark, legible row title:
   Delay magenta, 3sTrace orange, 10sTrace dark brown, and control blue. Use
   one size for row titles, one size for annotations, and consistent line and
   marker dimensions. The purple marks denote **US onset**, not pulse width;
   panel C's 50/100 ms legend covers US duration.
5. Leave A to the left and C below as in the current composite. B adds the
   within-trial contingency comparison; C keeps the across-session schedule
   and short/long US legend. This keeps B useful without duplicating C.

## Implementation and review

- `scripts/build_figure1_panel_b.py` draws B with named SVG groups for each
  condition, guides, CS bars, and US markers. It reads the active experiment
  definitions and derives each x position from one shared scale, so spacing
  cannot drift on later edits. Its default SVG destination is the SSD.
- Both generated sources are saved on the SSD. Preview a version with a
  one-panel source override, then with a whole-figure source override. After
  assessment, selecting it in the main layout is one source-path edit.
- Check in code that the CS covers 0–10 s in all rows, paired US markers land
  at 9/13/20 s, the three control examples lie before, during, and after
  the CS on distinct mini trial rows, and all text
  stays inside the B viewBox. Review the standalone B at its final composite
  size, then inspect A–C together. No rasterized panel should enter the SVG.
