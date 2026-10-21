# Fixed 10-second Trace processing launch, 18 October 2026

The author instructed this chat to process the missing fixed-10-second Trace
fish and associated controls, then explicitly selected
`F:\Digested Data\all10sTrace-full-v1` after the concurrent cleanup retired
the previously selected generic output root. This authorizes this new project;
the obsolete `F:\Results_refactored` root was not recreated.

The complete recursive inventory has 63 acquisition day/number identities,
31 trace and 32 control. Fresh raw discovery and protocol preflight resolve
all 63 to unique triplets without changing any D: filename. The four previously
missed suffix variants use supported explicit-source intake. Historical
`10sTrace`, `10sFixedTrace` and `trace` condition spellings publish the canonical
`trace` outcome label. The outcome experiment is `all10sTrace`, response window
[0,20). All original raw aliases and source paths remain in source manifests.

No matching per-fish output was found under the established projects before
launch, including the active 3sTrace project. The 32 historical control
identities also appear in the 3sTrace pickle inventory. Their shared acquisition
identities are retained explicitly; separate assay folder membership does not
make them independent fish. This processing run produces their 20-second
outcomes and does not create a paper cohort or inclusion decision.

`20250715_08` has only 53 CS and 54 US events. The runner retains an explicit
incomplete-protocol flag while allowing supported stages to process the observed
data. Successful artifact creation would not make its acquisition a complete
standard protocol.

The guarded runner is `scripts/process_trace10_additions.py`. It calls supported
per-fish intake, corrected preprocessing, corrected candidate metrics, movement
state, temporal profiles and trial outcomes with `overwrite=False`. It verifies
completed trial outputs and input lineage. Exceptions are logged per fish and
the remaining independent fish continue. An audit hook blocks writes to D:,
historical pickle paths and every pre-existing project file. Initial handoff
hashes and the baseline file states are checked at the end. No shared inventory
or intake ledger is regenerated.

Focused checks passed: 30 existing/intake/condition/outcome tests, followed by
six focused alias/schedule tests including two additional cases for interleaved
control logs and an incomplete trace with a final unreinforced CS. Sandbox temp
permissions required directing test temporary files to the authorized workspace.
No tests or processing wrote to D:.

Run ID: `trace10-additions-20261018-run`. Launched at
2026-10-18T14:36:25Z with launcher PID 19776. The host resource check found no
active raw-processing Python worker, approximately 45 GiB free physical memory,
and more than 8 TiB free on F:. The concurrent Delay completion record existed.

Evidence and live progress are in the project's `Metadata` directory:

- `trace10-additions-20261018-run-launch.json`
- `trace10-additions-20261018-run.json` (full aliases, source candidates,
  raw-line protocol event mapping, incomplete flags and preservation baseline)
- `trace10-additions-20261018-run.md` (all-fish readable audit)
- `trace10-additions-20261018-run.stdout.log`
- `trace10-additions-20261018-run.stderr.log`
- `trace10-additions-20261018-run-events.jsonl` (created as fish finish)
- `trace10-additions-20261018-run-completion.json` (created when the batch ends)

Launch and preflight are verified. The first fish entered intake. Completion
is not claimed by this launch record; use the actual completion JSON and
verified per-fish artifacts before reporting completed counts.
