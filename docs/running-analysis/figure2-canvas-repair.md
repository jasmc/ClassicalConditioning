# Figure 2 canvas-overdraw repair

The selected Figure 2 contains overlapping nested SVG viewports. Panel E's
opaque, full-canvas white path covers D's rightmost points, labels and comparison
marks; F's equivalent path covers E's right edge. This is reproducible SVG
overdraw, confirmed by native before/after rendering, rather than evidence that
the scientific marks or tick labels are absent from their sources.

The supported candidate adapter is
[`repair_canvas.py`](../../scripts/paper/figure_02/assembly/repair_canvas.py).
Its implementation in
[`svg_canvas.py`](../../src/classical_conditioning/figures/svg_canvas.py)
changes only the inline white fill of `matched-E__canvas-E` and
`trace10-F__canvas-F` to `none`. Each selected group must uniquely identify a
single untransformed white rectangular path matching its nearest SVG viewport.
Data-axis backgrounds are ineligible. Every other SVG byte, including data,
text, rasters, statistical marks, coordinates and physical dimensions, is
preserved. It neither recalculates science nor repacks the artwork.

## Saved candidate and evidence

The [candidate review](</Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-02/assembly/versions/figure-02-production-20261025/candidate-v02/review.html>)
and matching editable SVG live on JOAQUIM under
`ClassicalConditioning Outputs/figures/figure-02/assembly/versions/figure-02-production-20261025/`.
That directory holds the pre-edit ticket, exact original history, implementation
snapshot, candidate manifest and native-render review. The
[compact record](../../records/figure-reviews/figure-02/canvas-overdraw-20261025.json)
binds them. The candidate is unselected; selected Figure 2 and the collection
remain unchanged. There is no new freeze or production-PDF certification.

## Preparation and publication boundary

Capture and authenticate state, current SVG/HTML, legend, registry, evidence and
implementation hashes **before** edits. The saved `starting-ticket.json` records
the exact 25 October baseline. To reproduce that baseline into another existing
external directory, use:

```sh
.venv/bin/python scripts/paper/figure_02/assembly/repair_canvas.py \
  --starting-ticket '<external starting-ticket.json>' \
  --output-dir '<new empty existing external directory>'
```

Existing output files are never overwritten. The command verifies the existing
review/caption pair, binds the implementation closure, authenticates scientific
selection records, creates a matching candidate review and measurement page,
and rechecks inputs under the current-state lock. A concurrent manual edit
prevents all candidate publication. The active state is never written.

`candidate-v02/layout-review.html` is the supported exact browser measurement
page. Collect fresh measurements for `candidate.svg`, then use
`scripts/paper/shared/verify_layout_frames.py` with `--browser-report` and a new
`--evidence` destination. An old report cannot be relabelled with a new SVG hash.
Check newly exposed content against owned frames, spacing and alignments;
additional conflicts must remain unresolved rather than trigger automatic
resizing. Native previews do not supply browser frame evidence.

Only after passing exact frame checks and visual review may a guarded publisher
replace the same active SVG/HTML pair, bind current registry/presentation/frame
evidence and atomically update state. Preserve originals and any concurrent
work, then refresh the authenticated collection last. The candidate adapter
deliberately has no selection or publication flag.
