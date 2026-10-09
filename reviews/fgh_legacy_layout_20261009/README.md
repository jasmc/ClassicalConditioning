# Legacy presentation update of all four F/G/H versions

This changes presentation only. The four sets of numeric values and baseline definitions are retained and verified against their existing source hashes. Earlier figures are preserved in their original review folders.

- CS onset and offset at 0 and 10 s use broad green strokes (`#0d8136`, 2.4 pt, opacity 0.8), following the uploaded legacy SVG's green boundary bands (2 pt, opacity 0.75).
- Pre/Train/Test labels sit to the left of F, with enough margin to separate them from trial-number ticks.
- G, the 3 s Trace fish 20230310_08, has five black left-pointing arrowheads just outside its right spine, centred on trials 9, 17, 63, 66 and 93.
- Those provisional examples span Pre, early/late Train and early/late Test. Each has at least 79/80 finite displayed bins and 29/30 finite baseline bins in Version 4. They match the legacy phase positions and were not selected as the largest responders. `example_trials.json` records the exact counts and source hash for adapting panel E later.
- Each row retains one shared managua_r colourbar at the far right. C/D rows retain limits ±1; Version 4 retains ±0.25 natural-log units.

`render_layout.py` reads existing CSV values without recalculation, renders all four rows, verifies every SVG cell's value, colour and interval, and checks CS/arrowhead positions. Separate validation records bind each output to its numeric source files. The latest HTML summary and scoped Figure 1 registration point to these figures. Panel E itself is unchanged.
