# 3sTrace intake review, 18 October 2026

The author requested a review of all work performed in this chat before
continuing the authorized missing-fish analysis. The existing project remains
`F:\Digested Data\all3sTrace-full-v1`; the author forbids a separate project,
raw D: writes, replacement of historical outputs and paper selection changes.

## Current review evidence

### Completed, 19 October 2026

Recovery finished at 19:52:58 Europe/London on 19 October. The independent
completion report was written at 19:53:09 and reports
**completed-and-verified**, with **124 total verified additions**: 123 in the
recovery run plus the previously completed, reverified fish. There are no
failed fish, pending fish, run errors, per-fish final verification errors or
changed pre-existing files recorded by the preservation check.

Authoritative completion evidence is
`F:\Digested Data\all3sTrace-full-v1\Metadata\trace3-additions-20261018-recovery-01-verification.json`
and its companion Markdown report; the worker's `-completion.json` and
`-events.jsonl` remain alongside them. Both background processes have exited.
The 15 unresolved identities remain on hold. The 20 increasing-trace raw
context fish remain flagged separately from fixed-3sTrace and were not added.
This completes the authenticated 124-fish processing selection, not the
unresolved identities or a new paper cohort/freeze.

### Recovery status, 18 October 2026 at 20:35 UTC

The first worker stopped without a completion record after completing
`20230227_01`. No worker from that run was alive at recovery inspection.
The second fish, `20230227_04`, had only an unpublished intake staging
directory containing `camera.parquet`; no per-fish Metadata, Processed data
or Quality checks output was published. The abandoned staging directory,
old lock and original run evidence are preserved in place.

Recovery plan `Metadata\trace3-additions-20261018-recovery-01-plan.json`
has SHA-256
`58f3a28bf7cc951076d08c6795d4a70e9356c310e7193253227bcf290e001297`.
It binds the original plan/events, current source code and abandoned staging
hashes. All 124 originally selected fish were reconciled: one completed fish
passed full supported upstream verification and source identity checks again;
the other 123 had no published outputs. Their raw source states and protocol
hashes match the original plan. All 375 original metadata hashes still match.
The 15 held identities and 20 separately flagged increasing fish remain held
or separate as recorded below.

The detached hidden worker was launched at 20:35:10 UTC (launcher PID 11868,
actual Python worker PID 6544). Its run-start and `20230227_04` intake-start
events were observed at 20:35:17 UTC. This is **running, not completed**.
New evidence uses the `trace3-additions-20261018-recovery-01-` prefix.
Original worker/verifier source snapshots were saved under that prefix before
editing. Recovery uses an exclusive run lock and Windows named mutex, fresh
supported intake staging, `overwrite=False`, and the preservation write guard.
It refuses any fish with published partial artifacts or changed abandoned
staging; it does not delete staging to resume.

A detached read-only verifier was launched at 20:35:34 UTC, bound to the actual
worker process handle. It writes new `-verification.json`/`.md` evidence after
completion or reports an interrupted run. It also rechecks the previously
completed fish and reports the combined verified additions count. No app
notification automation was created. The 16 timing, identity, marker and
completion regression checks passed before launch.

- Detailed independent inventory/protocol review:
  `F:\Digested Data\all3sTrace-full-v1\Metadata\trace3-additions-20261018-reviewed-final.md`
  and its companion JSON.
- Separate increasing-trace evidence:
  `Metadata\trace3-review-20261018-increasing-separated.md` and JSON.
- 2025 control acquisition context:
  `Metadata\trace3-review-20261018-control-context.md` and JSON.
- Supported full upstream integrity verification completed successfully for
  all 58 inventory fish. Evidence: `trace3-additions-20261018-integrity-review`
  `.json`/`.md` in that same Metadata folder; all 375 preserved metadata hashes
  remain unchanged. Raw D: camera/tracking bytes were not rehashed by this
  existing-output check; D: protocol hashes match the prior source lineage.
- Original pre-review script snapshot:
  `Metadata\trace3-review-20261018-original-script.py`.
- Preservation baseline:
  `Metadata\trace3-review-20261018-preservation-baseline.json`.
  All 375 baseline metadata hashes were unchanged at the independent review.

Earlier `trace3-additions-20261018-run*` reports and the initial `reviewed`
report are retained as preliminary evidence. Their PASS/ready counts are
withdrawn; they must not launch processing. The current script is an audit-only
route. A separate reviewed append-only runner now owns new processing below.

## Authorized additions now running

The author said “continue” after the review. The active run is
`trace3-additions-20261018-verified-run`, launched on 18 October at 15:41 London
time, execution session `81356`, worker PID `4112` at launch. Confirm command
identity before relying on that PID. Stage starts/completions and per-fish
results are appended to its `-events.jsonl` file in the existing Metadata
folder. Completion evidence will be `-completion.json` there.

The immutable `-plan.json` has SHA-256
`759078ac5604f3ba01f369fff532899e01cdb884cc3ed703f92aab67b80b100d`;
the companion `-plan.md` records all selected and held identities. The run
contains 124 additions: 52 fixed-3sTrace fish and 72 matched controls. Fifteen
identities remain held: the 11 controls with fixed-10sTrace acquisition context,
the three missing-source identities and the conflicting `20230425_02`.

The supported identity parser now maps only explicit untreated
`traceNoMK801`/`controlNoMK801` aliases to canonical roles for `all3sTrace`.
Original names/paths remain in source manifests. Treated and other-protocol
aliases remain rejected. The concurrent 10sTrace changes are preserved.
All ten scoped parser/integrity/alias regression tests pass. Four completion
evidence tests and the two existing 10sTrace identity compatibility tests also
pass (16 tests total). Thirty-five of the selected additions
use the newly supported untreated alias; the other NoMK801 identity is held
for its conflicting stimulus sources.

`scripts/run_trace3_additions.py` executes the sealed plan with
`overwrite=False`, per-fish source-state/protocol checks and final supported
lineage verification. A project lock prevents a second append run. The Python
audit guard protects D:, historical pickle paths and every pre-existing
project file, and limits F: writes to selected per-fish/staging paths and new
run evidence. Source code hashes are checked at each fish boundary; changes
stop further work for reconciliation. Failed/partial artifacts are preserved.
No shared inventory, ledger or cohort/paper output is replaced.

The Delay worker had completed before this batch began. Host inspection found
about 45 GiB free RAM and idle D:/F: disk samples at preparation. New 3sTrace
processing uses one worker. At launch the first fish was converting raw input;
completion must not be inferred from launch or intake progress.

A separate read-only completion verifier is attached to the exact worker using
a Windows wait handle (session `27756`). It cannot terminate or write to the
worker. It will validate the worker completion record, reauthenticate current
trial-output hashes and source/condition identities, and create new
`-verification.json` / `-verification.md` evidence. A missing completion record,
partial/duplicate results, missing stages, changed originals or failed lineage
checks cannot produce a success status. The verifier never retries processing.

The first fish passed intake, corrected preprocessing and candidate metrics
without a preservation-guard rejection and reached movement-state processing.
No recurring app monitor has been created; the optional 30-minute monitoring
preference was asked asynchronously while work continued.

## Corrections made during review

The earlier route did not recognize the actual completion-marker hash schema;
its recursive path/hash lookup missed `artifact_sha256` and `summary_sha256`.
The replacement uses the supported artifact verifier. The earlier control
check relied on folder wording and an unsupported coincidence threshold.
The independent review retains observed timing and separately checks acquisition
day/directory context. Thirty-six untreated `NoMK801` aliases require explicit
supported downstream mapping; the old processing route would not have supplied
it. The earlier US-duration gate had no established exclusion authority; short
events are now reported descriptively without an invented exclusion decision.

Protocol parsing now retains malformed rows, exact integer timestamps, all
events and source line numbers. Camera/tracking/protocol groups are keyed by
both directory and full recording stem. Every reinforced training CS is checked
for the fixed 13-second onset interval; increasing/other nonfixed protocols are
separated. Tests cover a single nonfixed pair, increasing timing, the 20-second
boundary, unsorted rows, malformed rows, short US events and tampering with the
actual marker hash map or output bytes. All eight pass. Windows sandbox ACLs
prevented the first temporary-directory fixtures from running; the fixtures now
use the repository's supported inherited-ACL staging helper.

The independently configured US analysis groups (18–63) must not be mistaken
for an assertion about which raw US event numbers pair with training CSs.
Observed pair numbering is retained in the review; its uniqueness and continuity
are checked without changing the existing experiment definition.

## Findings and remaining work

- Fresh recursive scan: 199 pickles, 197 date/number identities; no discrepancy
  with the reconciled JSON/CSV pickle paths or the D: raw-candidate paths.
  No D: scan errors were reported.
- 58 identities passed authenticated trial-output/summary and full upstream
  verification. Another project marker belongs
  to `20230803_12`, outside this pickle inventory; it is preserved.
- 139 identities have no per-fish processing artifacts. This is an absence
  count, not a processing-ready cohort.
- Available trace-labeled sources: 91 observed fixed-3sTrace protocols; no
  increasing schedule observed among those files. Unavailable/conflicting
  sources cannot be classified by absence.
- A separate scan of the raw `increasingTrace` subfolder identifies 20 actual
  increasing protocols, with intervals rising from about 10.5 s to 13 s.
  All 20 are outside the 197-identity pickle inventory. They are separately
  listed with paths, timing and hashes and remain ineligible for fixed-3sTrace
  processing. This also checks the parser against actual increasing data.
- Three identities lack D: raw matches: `20250714_06`, `20250714_07`, and
  `20251123_03`. The latter retains three different-sized condition aliases.
  Same-day brown/blue suffix cues for the two July fish point to differently
  numbered raw fish `20250714_02`/`20250714_03`, both observed fixed-10sTrace.
  Those numeric/protocol conflicts do not authorize an identity reassignment.
  The other-year `20221123_03` raw name has a different blue suffix, and
  `20231123_03` belongs to a different treatment/age protocol. Neither is an
  authenticated correction for `20251123_03`. Its three pickles are gzip
  containers; bounded decompression/opcode inspection found no source identity
  before the first large array. No pickle was unpickled or renamed.
- `20230425_02` has two different stimulus files. The working file has 94 CS
  events and a 102 ms CS at trial 17, raw line 38. The preserved original has
  95 CS events, an epoch-zero event and numerous abnormal durations. Both
  protocols and SHA-256 values are recorded; neither was selected or modified.
- Eleven 2025 controls lack a matching fixed-3sTrace acquisition day in the
  current pickle inventory's raw directory; matched-control provenance remains
  open. Inspection of all 27 stimulus files from those acquisition days finds
  fixed-10sTrace recordings in the same raw directory. Their presence in a
  historical 3sTrace pickle folder does not establish matched-3sTrace provenance.
  Their raw protocol acquisition dates match their recorded filename day.
- Ten identities contain recorded US events shorter than 20 ms. Exact event
  numbers, lines and timestamps are documented; this screen does not authorize
  automatic exclusion.
- Existing staging trees for `20230307_12` and `20230309_11` are preserved.

Continue monitoring the authorized run, verify its results and preservation
check, and reconcile any failed/partial fish without overwriting. Keep the 15
held identities and 20 increasing raw-context fish separately documented.
Preserve shared inventories, ledgers, old run metadata and cohort/paper outputs.
Do not interrupt or modify work owned by other chats.
