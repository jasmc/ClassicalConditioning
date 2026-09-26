# Figure 1C session schematic: redesign plan

**Status:** exploratory v2 is selected in the current Figure 1 assembly at
`J:\ClassicalConditioning Outputs\ORGER-JOAQUIM\outputs\figure1-assembly\schemes\Fig1_PanelC_SessionProtocol_exploratory_v2.svg`.
It displays four phases and omits the final viability check at the user's
direction. The phase cards have been widened to use the freed space. V1 remains
on the SSD with the check. The protocol specification still records that event;
omitting it from the panel does not change the experiment. The rest of this
document records the earlier redesign rationale and recommendations.

## Scientific job of C, relative to A and B

- **A:** the larva, LEDs, and physical CS/US apparatus.
- **B:** the relative timing of a 10 s CS and the training US in Delay,
  3sTrace, 10sTrace, and an unpaired control.
- **C:** the order and contents of the experimental phases: where CSs, short
  baseline-maintenance USs, long training USs, and catch trials occur. It should explain how the trial types in B fit into
  the full session, without repeating B's 9/13/20 s diagrams.

## Protocol facts to represent

The current paper's `sections/Materialsandmethods.tex` (lines 139–190) gives:

| Phase | CS presentations | US presentations | Function |
| --- | ---: | ---: | --- |
| Priming/habituation | 4 CS-like, including left-side presentations | 15 short, 50 ms | Establish activity and habituate to CS-like light. |
| Pre-training | 10 CS | 3 short, 50 ms, interspersed separately | Measure the unreinforced CS response while maintaining activity. |
| Training | 50 CS | 46 long, 100 ms, paired in conditioned groups; 4 CS-only catch trials | Acquisition and catch-trial assessment. |
| Testing | 30 CS | 14 short, 50 ms, interspersed separately | Measure response without paired long US while maintaining activity. |
| After testing | none | One 500 ms pulse | Health/viability check, separate from learning trials. |

The protocol has 94 CS events in all, counting the four priming CS-like events;
the 78 regular US events are 32 short plus 46 long. The 500 ms final check is a
separate 79th event in the paper's sequence. The active `ExperimentSpec` encodes
the 10 s CS, 50 training CS trials, 46 paired long-US latencies, and 30 testing
CS trials, but `minimum_us_trials=78` is an analysis threshold, not a figure
instruction to omit the final viability pulse. Its four training catch CS
events correspond to global CS IDs 25, 39, 53, and 59; the configuration also
marks first Test CS 65 as a catch for analysis.

The paper's control protocol matches the absolute **US schedule** and total
stimulus exposure of the paired protocol while randomizing **training CS
onsets** independently for each control fish. A single fixed control CS event
train is therefore not representative of all control fish. The figure should
state the rule instead of inventing one universal control raster.

## Problems in the selected artwork

1. `10 CS-only trials` and `30 CS-only trials` can be read as no US in those
   phases, while the artwork depicts violet pulses and the methods specify 3
   and 14 interspersed short USs. The trials are unreinforced CS presentations;
   the separate short pulses are present for baseline activity.
2. `46 CS-US trials` omits the four training CS-only catch trials and obscures
   the total of 50 training CSs.
3. Priming/habituation has no clear count or explanation of its 4 CS-like and
   15 short-US events. The final 500 ms health-check US is absent.
4. The dense tiny event marks, narrow light-purple legend, and large 0–200 min
   axis compete with the phase story. At that scale, 50 ms versus 100 ms cannot
   be shown by pulse width. The alternative supplied timeline uses larger
   arrows, but remains crowded and retains the wording and count problems.
5. The displayed control training sequence looks like one fixed protocol,
   despite per-fish CS randomization. The minute positions should not claim
   exact timing unless they come from a named, authenticated example protocol.
6. C's current typography requires source-specific 7 px legend repairs and
   positioned-text transforms during assembly. A fresh live-text SVG can make
   all labels legible at the main figure's final size.

## Preferred C design

Use a **single horizontal session sequence** with four phase blocks and a
small terminal viability-check marker. The phase blocks need not have widths
proportional to minutes; label the view `session schematic (not to scale)` and
remove the 0–200 min axis unless we generate an explicitly named
representative schedule from verified event logs. Give Training enough width
for the paired/catch distinction. A compact tier for each block should show
the CS count and the relevant US count, purpose, and duration:

`Priming: 4 CS-like + 15 short US → Pre-training: 10 CS + 3 short US → Training:
50 CS = 46 paired long-US trials + 4 CS-only catch trials → Testing: 30 CS +
14 short US → final 500 ms check`.

Use a consistent green CS symbol and one violet US hue with distinguishable
**short** and **long** marker shapes, directly labeled `50 ms` and `100 ms`.
The 500 ms check gets a third distinct terminal marker and does not enter the
conditioning-trial counts. Define short USs as separate, interspersed pulses
that support active baseline behavior; do not position them as if paired with
CSs. Label the Pre-training and Testing CSs `unreinforced` or `no paired long
US` rather than `CS-only` in isolation.

Within the Training block, place a small text cue `trial timing: B`; do not
repeat B's four mini diagrams. A one-line control note below Training should
say `Same long-US times; training CS onsets randomized per control fish`.
If exact protocol rasters are needed, move them to the supplementary detailed
protocol panel, where one named source protocol and the variant/control
schedule can be displayed at readable scale.

Keep DejaVu Sans, editable SVG text, and the same green/violet meanings as A
and B. Give the four phases equal baseline alignment, consistent label size,
and spacing readable at the full Figure 1 print width. Make counts more
prominent than decorative event ticks.

## Figure-level recommendation

Maintain a separate C source during iteration, so B and C can be reviewed
independently. For the final paper, the more economical layout is **one
protocol panel B**: the session sequence as the main part and the four
contingencies as its inset. Then raw tail angle can be C, vigor calculation D,
and the three individual examples E–G, matching the current paper draft's
seven-panel sequence. Keeping separate B and C in the final figure creates an
eight-panel Figure 1 and shifts every downstream paper panel letter. If that
eight-panel structure is ultimately preferred, update the manuscript text,
legend, registry, and figure references together.

## Implementation checks before replacing C

1. Confirm the phase counts, catch-trial placement, pulse lengths, and final
   health check against one authenticated source protocol/event log and the
   active `ExperimentSpec`; reconcile any variant-specific exceptions.
2. The new SSD source is
   `schemes/Fig1_PanelC_SessionProtocol_exploratory_v1.svg`, generated by
   `scripts/build_figure1_panel_c.py` from the validated
   `configs/paper-figures/figure1-session-protocol.json`. The supplied SVGs
   remain untouched references.
3. The standalone comparison is `panels/Fig1_PanelC_exploratory_v1.svg`; the
   A–C comparison with B's second exploratory version is
   `figure1-exploratory-BC.svg`, both on the SSD. Check legibility at final
   figure width, then decide whether to merge B/C for the manuscript layout.
4. Verify all panel C contents are SVG elements and live DejaVu Sans text;
   ensure no label suggests that 50 ms maintenance USs are conditioning
   pairings or that the controls share one randomized CS sequence.

The current exploratory design uses the same filled violet dot for a long-US
onset as B v2. A hollow violet dot denotes a **50 ms short US** only in C;
the control examples in B v2 use three separate miniature CS rows with one
filled long-US dot each. This avoids assigning the hollow symbol two meanings.
The final viability pulse is a violet diamond. The C Training card says
`Trial timing in B` and B v2 says `Training trial`, providing a clear zoom
relationship without a leader line crossing the figure's text.
