# All retained F/G/H versions — 9 October 2026

V3 has no established recipe in this shortlist. V4/V5 and the original frame-centred V7 are excluded because their displayed baseline median is not guaranteed zero. Four-colour V6 remains discarded.

## Version 1 C

Complete-bout eligible log median, including support outside the display crop → repeat on eligible samples → subtract baseline P50 → divide by (P90−P10)/2 → clip to ±1.
Baseline: [−15,0); timepoints carrying complete-bout medians
Colour: Continuous; median-centred P10/P90 width scale
Defined: 269/270
Pros: Fine timing; robust bout summary.
Cons: Within-bout detail lost; more saturation; complete bouts can cross CS.

## Version 1 D

Same summaries and baseline as V1 C; divide centred values by max(P50−P10,P90−P50), then clip to ±1.
Baseline: [−15,0); same as V1 C
Colour: Continuous; larger symmetric denominator
Defined: 269/270
Pros: Less or equal saturation than C.
Cons: Same bout-summary limitations; weaker contrast.

## Version 2 C

Mean eligible original log frames in each 0.5 s bin → subtract baseline-bin P50 → divide by (P90−P10)/2 → clip to ±1.
Baseline: [−15,0); finite bin means
Colour: Continuous; median-centred P10/P90 width scale
Defined: 269/270
Pros: Finer bins; retains direct frame variation.
Cons: Scaling hides physical magnitudes; one undefined trial.

## Version 6 · five colours

Mean eligible log frames in 1 s bins → subtract baseline P50 → assign bands using baseline P25/P45/P55/P75 → show five managua_r colours. Ties enter the upper band.
Baseline: [−20,0); finite bin means
Colour: Five baseline percentile bands; P45–P55 dark centre
Defined: 270/270
Pros: Simple categories; all trials defined.
Cons: Coarser timing; hides within-band magnitude; up to 20 baseline bins.

## Version 6 · softer High

Identical numeric data, bands and layout to Version 6. Only High changes from #ffcf67 to #e09f57; all other colours, including the dark centre, are unchanged.
Baseline: Same as V6
Colour: Same five bands; High sampled at managua_r 0.90
Defined: 270/270
Pros: Less bright High colour.
Cons: Smaller visual separation between Above and High.

## Version 7 · physical scale (earlier)

Subtract original eligible log-frame baseline median → take each bout median within [−20,+20) → repeat on eligible frames → mean in 0.5 s bins → subtract median of finite baseline bins. No percentile amplitude scaling. Colour saturation only beyond ±0.25.
Baseline: [−15,0); median of finite bout-summary bins
Colour: Continuous physical log differences; ±0.25
Defined: 270/270
Pros: Common physical log scale; all trials defined.
Cons: Baseline contrast varies across trials and fish.

## Version 7 · scaled (current)

Use V7 physical centred bins → fit separate lower/upper P10/P90 stretches with a symmetric median-preserving centre → apply to all trial bins. At least 10 finite baseline bins required; sparse/degenerate scaled trials undefined. Scaled exports remain unclipped.
Baseline: [−15,0); median and P10/P90 of finite bins
Colour: Continuous; P10/P90 at ±0.7; limits ±1
Defined: 257/270
Pros: Comparable baseline extents; continuous response headroom.
Cons: Physical magnitude differs for equal colours; 13 sparse Control trials undefined.

## Version 8 · endpoint saturation

Divide Version 7 scaled values by 0.7. Keep the same baseline references, centre width and defined-trial mask. P10/P90 reach colour endpoints; values beyond them have saturated colours.
Baseline: Same as V7
Colour: Continuous; P10/P90 at ±1; tails saturated
Defined: 257/270
Pros: Stronger baseline contrast; full colour range.
Cons: Large responses beyond anchors share endpoint colours.

## Version 9 · discrete Version 7

Start from unchanged Version 7 scaled 0.5 s bout-summary values and its defined-trial mask. Assign fixed bands at -0.5, -0.1, +0.1 and +0.5. Exact threshold ties enter the upper band. Zero lies in the dark central band [-0.1,+0.1). Apply Version 6 managua_r colours with softer High (#e09f57, palette position 0.90); this uses fixed scaled boundaries rather than Version 6 percentile boundaries.
Baseline: [−15,0); Version 7 baseline
Colour: Five V6 colours with softer High; fixed scaled boundaries -0.5/-0.1/+0.1/+0.5
Defined: 257/270
Pros: Bout-summary recipe with simple five-colour categories.
Cons: Within-band differences hidden; sparse V7 trials remain undefined.

## Version 10 · three colours

Use unchanged Version 9 physical and Version 7 scaled values. Merge Low with Below (old classes 1/2), retain Centre (old class 3), and merge Above with High (old classes 4/5). Fixed boundaries are -0.1 and +0.1; exact ties enter the upper band. Cyan #81e7ff represents Low/Below, dark purple #582948 represents Centre, and softer amber #e09f57 represents Above/High. The same 13 sparse Control trials remain undefined.
Baseline: [−15,0); Version 7 baseline
Colour: Three V9 colours; fixed boundaries -0.1/+0.1; dark centre and softer amber
Defined: 257/270
Pros: Simplest display: lower, near-baseline, higher; dark centre.
Cons: Merges moderate and extreme deviations; magnitude detail reduced further.
