# Delay additions from the PC inventory, 18 October 2026

Author scope: recursively inventory all pickle-identified fish under
`F:\Results (paper)\2025_delay\Processed data\pkl files\1. Original`,
including `Excluded`; find their raw triplets on D:; process fish missing from
the current analysis. The author explicitly requires adding data to the
existing project, without creating a separate output project. Never modify D:
or replace old processed data.

## Inventory and audit

Author clarification, 18 October: day and numeric fish number are the primary
identity; some suffixes were historically renamed. Preserve aliases and flag
conflicting evidence rather than interpreting suffix changes as another fish.

- 94 fish: 48 Delay and 46 control. Exactly three raw source files were found
  for every fish. The D: scan could not enter protected Recycle Bin and System
  Volume Information directories; all requested fish were found elsewhere.
- Existing project: `F:\Digested Data\allDelay-full-v1`.
- 57 fish already have trial-outcome completion markers. All 57 raw protocol
  SHA-256 values match the existing inventory.
- 37 additions: 19 Delay and 18 control. All are 2022 acquisitions with no
  existing per-fish processing artifacts in this project at launch.
- Raw 2024 `delayNoMK801`/`controlNoMK801` labels correspond to the historical
  pickle labels `delay`/`control`; remaining full recording-name fields agree.
- Several existing controls have isolated near-9-second US coincidences,
  recorded in the audit; these are not a systematic paired-training schedule.
- Existing control **20240624_03** has abnormal CS durations for cycles 1–8:
  48.999, 55.941, 50.231, 50.251, 58.797, 62.641, 64.359 and 62.316 seconds.
  Its raw stimulus hash matches the current inventory. This remains flagged;
  no exclusion or modification was made.

Canonical complete inventory and protocol evidence for this operation:

- `F:\Digested Data\allDelay-full-v1\Metadata\delay-expansion-20261018-discovery.json`
- `F:\Digested Data\allDelay-full-v1\Metadata\delay-additions-20261018-run.csv`
- `F:\Digested Data\allDelay-full-v1\Metadata\delay-additions-20261018-run.md`
- Detailed anomaly/renaming Markdown:
  `F:\Digested Data\allDelay-full-v1\Metadata\delay-anomalies-and-renaming-20261018.md`
  lists all abnormal CS durations, isolated control timing coincidences and
  each historical NoMK801 alias with its raw and pickle filenames.

Earlier audit attempts (`delay-additions-20261018` and
`delay-additions-20261018-checked`) were too strict about historical NoMK801
names and chance control coincidences. Their files are preserved. The `-run`
audit supersedes their classifications and retains the real duration anomaly.

## Processing launch and historical progress

Run uses `scripts/process_delay_additions.py` and the supported individual
intake, corrected preprocessing, corrected candidate metrics, movement state,
temporal profile and trial-outcome stages. Every stage has `overwrite=False`.
The individual-stage route avoids replacing shared inventory/intake ledgers,
old run records and earlier cohort comparisons. A Python audit hook also blocks
D: mutations, changes to original pickle files and replacement/removal of files
that already existed in the project at launch.

Command (existing repository `.venv`, Python 3.13.14):

```powershell
& .\.venv\Scripts\python.exe -u scripts/process_delay_additions.py --discovery 'F:\Digested Data\allDelay-full-v1\Metadata\delay-expansion-20261018-discovery.json' --run-id delay-additions-20261018-run
```

At this record's creation the process is running in execution session **63444**
(Python worker PID **20868**, launcher PID **21800**). First fish
`20221115_01` completed intake and entered corrected preprocessing. Do not
launch the command again while it is running. These are live process references,
not durable completion evidence.

Per-fish results are appended to
`Metadata\delay-additions-20261018-run-events.jsonl`. Final results and the
size/mtime preservation check will be written to
`Metadata\delay-additions-20261018-run-completion.json`. Absence of the final
file means completion has not been established. Confirm all 37 fish completed,
inspect failures, verify the generated stage markers, and reconcile the full
94-fish inventory before reporting success. Existing cohort/paper results
have not been expanded or replaced by this operation.

Subsequent progress: `20221115_01` completed all six processing stages in
579.2 seconds; `20221115_03` then started intake. Full batch completion remains
pending.

## Final completion and verification

The batch exited successfully after 17,666 seconds (4.91 hours). All 37
planned additions completed all six supported stages: 19 Delay and 18 control.
The final membership exactly matches the planned missing-fish list. The run
reported no existing project files changed or removed, based on size/mtime;
this check is not a retrospective full-byte audit of every original file.

The supported `verify_candidate_trial_outcomes` verifier subsequently passed
for all 37 additions, authenticating outcome/coverage hashes, Delay response
window settings and current upstream lineage. All 94 inventoried identities
now have trial-outcome completion markers. The older 57 were preserved and
not reprocessed or newly fully verified. Shared cohort/paper results remain
unchanged.

New final evidence in the existing project's Metadata folder:

- `delay-additions-20261018-run-completion.json`: 37 successful stage results
  and the existing-file preservation check.
- `delay-additions-20261018-final-verification.json`: 37 supported verifier
  passes, zero failures and zero missing markers across the 94-fish inventory.
- `delay-additions-20261018-completion-report.md`: final summary.
- `delay-additions-20261018-acquisition-flags.md`: all acquisition QC warnings
  and per-fish alignment evidence.

Acquisition QC remains a review matter: 31 additions have camera/tracking
boundary-range differences; six have PASS status. Fish `20221118_01` also
has stimulus events outside the camera acquisition interval. These findings
were retained without changing sources or making scientific exclusions.
The pre-existing `20240624_03` long-CS anomaly and historical aliases remain
documented in `delay-anomalies-and-renaming-20261018.md`.

## Separate 3sTrace chat

The author requested a new chat to perform the same full-inventory addition
for 3sTrace. Created local Analysis chat
`01a14e04-8ef6-7501-a3aa-16fdb9ca12fc`, titled
`Process missing 3sTrace fish from full pickle inventory`; it has started work.
Its complete inventory/handoff is
`F:\Digested Data\all3sTrace-full-v1\Metadata\trace3-additions-20261018-reconciled.md`
with companion CSV/JSON. 199 pickle files represent 197 date/number identities;
58 have actual versionless trial markers, leaving 139 without such markers.
These presence counts require full provenance/completion reconciliation.
The handoff flags conflicting control/Delay/Trace aliases for `20251123_03`,
missing raw matches for that ID and `20250714_06`/`20250714_07`, and two stimulus
versions for `20230425_02`. It requires the existing 3sTrace output project,
preserves all old files and D:, and avoids competing heavy processing while
the Delay batch is running.
