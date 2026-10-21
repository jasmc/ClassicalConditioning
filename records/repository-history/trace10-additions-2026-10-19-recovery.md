# 10sTrace stage recovery, 19 October 2026

The author asked to process the fish after the earlier batch had stopped.
Existing authorization and destination remain:
`F:\Digested Data\all10sTrace-full-v1`.

The host process check found no active Python processing worker. The original
run log stopped after completion of `20230306_02` and movement-state completion
for `20230306_03`; it had no batch completion record. Original logs, published
artifacts and any staging are preserved.

The first fish's trial outcomes were independently authenticated through the
supported verifier before recovery. They retain canonical `trace` identity,
`all10sTrace`, and [0,20) response windows.

New runner: `scripts/resume_trace10_additions.py`. It requires the exact hash of
a new recovery plan linked to the original immutable run inventory. Recovery
recognizes only a contiguous prefix of wholly published stages. It verifies
completed stages, builds missing stages with `overwrite=False`, and refuses
partially published or conflicting outputs. Each fish runs in a separate child
process, releasing its memory before the next fish. A Windows mutex prevents
concurrent recovery coordinators. Original project files, historical pickle
files and every D: path remain protected against mutation by audit hooks.

Ten focused alias, schedule and recovery tests passed. Tests include preserving
completed stages, refusing partial publication, rejecting orphan downstream
artifacts, and processing an observed incomplete trace without inferring a
complete standard protocol.

The sealed plan has 63 fish and no recovery conflicts. `20230306_02` has six
stages to verify/reuse; `20230306_03` has four stages to verify/reuse and needs
temporal profiles and trial outcomes. The remaining 61 fish have missing stages.
The 32 shared control identities and the incomplete stimulus schedule of
`20250715_08` remain explicit in the original and recovery records.

Run ID: `trace10-additions-20261019-recovery`.
Plan SHA-256:
`510460e33f7717ae5f17082d439785453ae232f6aa2794a3c92a0528a5f4b263`.
Launcher PID: 12024. Started 2026-10-19T19:23:19Z.

Evidence in the project's `Metadata` directory:

- `trace10-additions-20261019-recovery-plan.json`
- `trace10-additions-20261019-recovery-launch.json`
- `trace10-additions-20261019-recovery.stdout.log`
- `trace10-additions-20261019-recovery.stderr.log`
- `trace10-additions-20261019-recovery-events.jsonl`
- `trace10-additions-20261019-recovery-<identity>-result.json` per completed worker
- `trace10-additions-20261019-recovery-completion.json` and `.md` when the batch ends

At the launch checkpoint, the first fish passed the recovery worker's full
stage and raw-source verification, and the second worker was verifying its
four existing stages. No remaining fish is claimed complete here. The final
completion record must establish the batch outcome and preservation checks.
