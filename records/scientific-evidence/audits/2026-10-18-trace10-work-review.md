# Review of fixed 10-second Trace intake work, 18 October 2026

The author requested a review before continuing the authorized inventory,
raw-protocol audit and missing-fish processing task. This record separates
directly checked findings from earlier incomplete conclusions. It is not a
scientific inclusion decision, processing completion record or paper freeze.

## Actions and saved state

- Read repository `AGENTS.md`, the dated PC handoff, folder policy, and the
  preserved 10sTrace inventory Markdown, CSV and JSON. The original combined
  tool output was truncated; later structured reads and direct raw stimulus
  parsing supplied the checked findings below.
- Inspected the supported individual intake, corrected preprocessing,
  candidate metric, movement, temporal-profile and trial-outcome routes.
  Did not edit the active Delay or 3sTrace scripts or shared processing code.
- Searched the three established F: output roots recursively for filenames
  containing the 63 identities and `all10sTrace`. No matching per-fish
  artifacts or completion markers were found. The generic
  `F:\Results_refactored` inventory describes an older 57-fish Delay intake;
  it must be preserved, not regenerated for the 10sTrace additions.
- The author explicitly selected **`F:\Results_refactored`** as the destination
  for the new 10sTrace per-fish outputs. The earlier destination question is
  resolved; no further destination approval is required.
- No intake or downstream processing was launched by this chat. No code edits
  or new per-fish outputs were made before this review. An attempted external
  report-writing command was interrupted; a subsequent explicit existence
  check confirmed that
  `F:\Digested Data\trace10-additions-20261018-audit.md` did not exist.
  Its draft text was not published evidence.
- All D: commands were reads: directory listings, stimulus parsing, small
  camera/tracking endpoint reads and hashing. No D: write, rename or deletion
  command was issued. The preserved handoff files were not replaced.

## Inventory and raw identity corrections

The preserved inventory has 63 pickle files and 63 acquisition day/number
identities: 31 `10strace` and 32 `control`. A complete accessible D: directory
walk found **three matching `.txt` raw files for each of the 63 identities**.
No additional same-ID raw candidate collision was found. Five denied system
directories were reported: four user recycle-bin directories and System Volume
Information. This is an explicit limit on the D: search, not evidence of files
inside those directories.

The four initial apparent misses, `20250714_05`, `_06`, `_07`, `_08`, are
present in `D:\2023 03_3-s vs 10-s fixed trace\Raw data`. Their file suffixes
are concatenated to `5dpf`, for example `...5dpfcam.txt`,
`...5dpfmp tail tracking.txt`, and `...5dpfstim control.txt`. They were missed
by the initial underscore-suffix matcher. Their date/number identity is
unchanged; no corrected date or fish number is needed.

| Identity | Stimulus SHA-256 | Cycle / US | CS duration range (s) | Training near-20 pairings |
|---|---|---|---|---|
| 20250714_05 | d38c6d462b60743a8fb7725a006fc7b724d8bfba2e636a0a066d684da540b2ca | 94 / 79 | 9.997–10.014 | 46 |
| 20250714_06 | 32eeab1cef0d47718bea222d36d90beb51d1080de1d5ad16999e080933d675b7 | 94 / 79 | 9.999–10.040 | 0 |
| 20250714_07 | 7bb76061a24d84d9c842ace50593b9e2b3fba0cca9aec8542a2cf951f27ac34d | 94 / 79 | 9.957–10.025 | 0 |
| 20250714_08 | ef890016b99936e5439b150ccc74ee150afbc55e520ec6c5f61757650103d8dc | 94 / 79 | 9.969–10.003 | 46 |

## Full event checks and corrections

Directly parsed all 63 current D: stimulus files, including the four recovered
suffix variants. Tested all CS events for US starts within 20 ± 0.2 s and
examined the actual CS trial indices of the training associations.

- All 30 complete conditioned recordings have 94 Cycle and 79 Reinforcer
  events, 46 training associations near 20 s, and exactly the unpaired
  training CS trials **25, 39, 53, 59**. No near-20 associations occur outside
  training in these conditioned recordings.
- Training CS numbering is 15–64. US analysis numbering 18–63 is a separate
  numbering system. The interrupted report draft's reference to pairings at
  “trials 18–63” was ambiguous and must not be reused as CS numbering.
- The configured additional CS catch at 65 lies in early test; it is not a
  fifth training catch.
- Controls have zero or one training coincidence near 20 s, rather than a
  repeated 46-event pairing pattern. The isolated coincidences are:

| Control identity | CS trial | US event number | Latency (s) |
|---|---|---|---|
| 20230314_03 | 33 | 35 | 19.938 |
| 20230315_01 | 36 | 43 | 19.895 |
| 20250715_12 | 33 | 35 | 20.022 |
| 20250717_03 | 19 | 22 | 19.810 |

The interrupted report draft incorrectly identified 20.404 s as the last
control's flagged coincidence. The actual flagged coincidence is 19.810 s;
20.404 s is a different latency outside the ±0.2-s screen. Isolated
coincidences do not establish a paired-protocol mismatch.

## Incomplete acquisition: 20250715_08

This recording has 53 CS and 54 US events. There are **39 observed training
CS events**, trials 15–53 inclusive, with 36 pairings near 20 s. Observed
unpaired training CS trials are 25, 39 and 53. The handoff's next-US screen
contains only 38 nonempty latencies because the final CS has no later US;
38 must not be reported as the number of observed training CS events.

The actual source files are under the mixed-assay raw directory:

- `20250715_08_10sTrace_black-2_mitfaMinusMinus,elavl3GCaMP6fEF05_6dpf_cam.txt`
- `20250715_08_10sTrace_black-2_mitfaMinusMinus,elavl3GCaMP6fEF05_6dpf_mp tail tracking.txt`
- `20250715_08_10sTrace_black-2_mitfaMinusMinus,elavl3GCaMP6fEF05_6dpf_stim control.txt`

Camera endpoint inspection gives first/last frame IDs 507629 / 5711889 and
absolute times 1752583829035 / 1752591236560 ms. Tracking endpoints are
507627 / 5711889. The final CS starts at 1752591144589 ms and ends at
1752591154587 ms, about 82 s before the last camera timestamp. This is a
recording with a shortened stimulus schedule and bounded acquisition, not
evidence of a complete 94-CS protocol. Endpoint reads do not verify every
interior frame. Supported intake/downstream behavior still requires review;
the recording must remain explicitly incomplete even if some stages succeed.

## Broader shared-control finding

The intersection of the complete 3sTrace and 10sTrace inventory identity sets
is **all 32 controls**, not merely `20250714_06` and `_07`. There is no
conditioned-identity overlap in these two inventory sets. Historical assay
folder membership therefore cannot establish that these are independent
control acquisitions.

The two initially highlighted duplicate pickle pairs were hashed directly:

| Identity | Size in each assay folder (bytes) | Identical SHA-256 in both folders |
|---|---|---|
| 20250714_06 | 762836747 | 4126629067f3f0e3b56ec79baa258234331e33a73c8f07903b9d53d7e8e1c3b8 |
| 20250714_07 | 752105684 | 56ce98d3ba7312c37476fdb15df819dec7836f62a6567195f5c89dab27d74569 |

For each identity the files are in both
`F:\Results (paper)\2025_3sTrace\Processed data\pkl files\1. Original` and
`F:\Results (paper)\2025_10sTrace\Processed data\pkl files\1. Original`.
These two pairs are proven byte-identical copies, not evidence of two
independent acquisitions. The other 30 pairs have only been compared by
inventory identity at this checkpoint; byte identity is not yet claimed.

The 32 shared control IDs are:

`20230306_03`, `20230306_09`, `20230306_12`, `20230307_02`,
`20230307_05`, `20230307_08`, `20230307_11`, `20230309_06`,
`20230309_12`, `20230310_01`, `20230310_03`, `20230310_09`,
`20230310_12`, `20230314_03`, `20230314_05`, `20230314_08`,
`20230314_10`, `20230315_01`, `20230315_03`, `20250714_01`,
`20250714_04`, `20250714_06`, `20250714_07`, `20250715_02`,
`20250715_04`, `20250715_05`, `20250715_07`, `20250715_10`,
`20250715_12`, `20250716_03`, `20250717_01`, `20250717_03`.

Before processing controls, reconcile shared upstream artifacts and per-assay
outcome windows. Source identity remains the date/number. Shared controls
must not become independent fish through duplicated assay directory labels.

## Code and execution findings still requiring work

- `intake_recording` discovers underscore suffixes only, so the four recovered
  concatenated-suffix triplets need a supported explicit-source route without
  changing D: filenames.
- `condition_from_recording_name` retains the lowercased source token.
  `parse_recording_identity(..., experiment_name='all10sTrace')` currently
  recognizes the configured `10sFixedTrace` source name but does not recognize
  the observed `10sTrace` alias. A declared alias mapping to canonical `trace`
  is required before the affected trial-outcome stages can succeed.
- The routine all10sTrace outcome window is [0,20), as required. No Delay or
  3sTrace window should be silently substituted.
- Default `uv run` initially selected an inaccessible `.venv` interpreter.
  Direct system Python and `.venv-trace\Scripts\python.exe` subsequently
  executed successfully. This did not modify an environment or install packages.
- Host process visibility from the sandbox is limited. An empty local Python
  process listing does not prove the other chats' workers have stopped.
  Existing run event files show Delay progress beyond the dispatch checkpoint;
  resource coordination must use fresh evidence before launching heavy work.

Processing remains pending. Preserve all original inventories, ledgers,
markers and per-fish artifacts; use the authorized generic destination and
per-fish stages with `overwrite=False` when the remaining source/API issues
have been resolved.
