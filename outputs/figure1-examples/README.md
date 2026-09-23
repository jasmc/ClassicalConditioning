# Figure 1 C/D example fish metric comparison

These are visual review exports, generated from the read-only
`/Volumes/JOAQUIM/Digested Data/allDelay-full-v1` processed dataset. They are
not a choice of final vigor metric or final manuscript panels.

- Delay fish: `20221115_07`, the active example in the February 14 and March 24
  historical `2_ExampleFishPlotting.py` revisions.
- Control fish: `20221115_09`, the commented control example in those revisions
  and in the archived normalized-vigor script. Its source manifest identifies
  it as `control`.
- Rows: global CS trials `9, 17, 63, 66, 93`. The historical script labels
  these `5, 13, 59, 62, 89` after subtracting its four-trial offset. Their
  intended stage names are Pre-Train, Early Train, Late Train, Early Test,
  Late Test.
- Each fish has three paired tail-angle/vigor figures. Each vigor figure uses
  one candidate metric; both columns use the same measured frames and trial
  windows. The 0 and 10 s green guides mark CS onset and offset, while purple
  guides mark actual reinforcer events inside the displayed window.
- Every figure has a JSON sidecar with source hashes and a reproduction command.

Run one fish with all three vigor metrics:

```sh
MPLCONFIGDIR=/private/tmp/cc-mpl PYTHONPATH=src ./.venv/bin/python \
  scripts/render_legacy_ssd_example_traces.py \
  --project-dir '/Volumes/JOAQUIM/Digested Data/allDelay-full-v1' \
  --output-dir outputs/figure1-examples \
  --recording-id 20221115_07 --all-metrics \
  --trial 9 --trial 17 --trial 63 --trial 66 --trial 93 \
  --mode static --overwrite
```

Replace the recording ID with `20221115_09` for control. The SSD adapter
checks completion-marker hashes for corrected frames and candidate metrics,
and the intake manifest hash for protocol. It reads only Parquet row groups
overlapping selected windows. The shared Figure 1 renderer creates the plots.
