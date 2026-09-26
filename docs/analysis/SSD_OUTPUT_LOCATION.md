# Review outputs on the raw-data SSD

The repository's `outputs/` directory is mirrored to
`J:\ClassicalConditioning Outputs\ORGER-JOAQUIM\outputs\`. The mirror keeps
the repository paths used by the running 3sTrace jobs working while making
their figures, review tables, logs, and provenance sidecars available on J:.
Each copied file is checked by SHA-256. A file held open or changed during a
copy is retried on the next pass. The active 3sTrace mirror runs every five
minutes and makes a final pass after the rebuild, downstream analysis, and
verified J: to F: transfer finish.

Put outputs from another computer in a sibling folder under
`J:\ClassicalConditioning Outputs\`, named for that computer. Keep its own
`outputs/` tree and sidecars together. Do not merge those files into
`ORGER-JOAQUIM\outputs\`, where names could collide.

For later runs on this computer, refresh the mirror with:

```powershell
pwsh -File scripts/sync-repo-outputs-to-ssd.ps1
```

The full 3sTrace project and its generated `Figures/` directory are under
`F:\Digested Data\all3sTrace-full-v1\`; the J: mirror above is specifically
for the repository's `outputs/` directory. The raw recordings remain under
`J:\Raw Data\`.

The Figure 1 scheme-assembly workflow writes **directly** to
`J:\ClassicalConditioning Outputs\ORGER-JOAQUIM\outputs\figure1-assembly\`.
Its SVG schemes and fonts live in `schemes/` and `fonts/`, while the combined
figure and individual panel previews are generated in that SSD folder. The
layout JSON and assembly code remain in the repository. This direct-write
workflow does not depend on the repository-output mirror.
