# PC handoff: more fish for Figures 1 and 2

Prepared 18 October 2026. The author confirmed **fixed 10-second trace fish**,
not increasing trace. This is a dated operational handoff, not a scientific
selection or a new freeze.

## Readiness

**Partly ready.** The supported pipeline has definitions for Delay, fixed 3 s
Trace and fixed 10 s Trace. Delay and 3 s raw data are on JOAQUIM. Processing
new fish into separate candidate projects can proceed after PC path, storage
and protocol checks. Reproducing and extending the exact selected Figure 1/2
analyses needs additional work: current paper scripts have fixed paths,
versioned input names and cohort assumptions. A routine pipeline run does not
regenerate the selected paper analyses automatically.

| Item | Checked on the connected SSD on 18 October |
| --- | --- |
| Delay raw data | `Raw Data/allDelay`: 57 complete triplets, 29 Delay + 28 controls; no incomplete groups found. |
| Fixed 3 s raw data | `Raw Data/all3sTtrace` (spelling intentional): 59 complete triplets, 40 trace + 19 controls; four additional incomplete groups. Complete trace protocols classify as fixed 3 s. |
| Incomplete 3 s recordings | `20230227_01`, `20230302_12`, `20230426_08`: camera only. `20230426_10`: stimulus file only. Preserve and report them; no usable triplet yet. |
| Fixed 10 s raw data | No dedicated folder or corresponding triplets found in the SSD's three raw-data folders. Find the PC data before launching. |
| Increasing trace | `Raw Data/incTrace`: 14 complete triplets. This is a separate protocol and must not supply fixed-10 s fish. |
| Existing processed data on SSD | `Digested Data/allDelay-full-v1` exists. No full 3 s or 10 s project found immediately under `Digested Data`. |
| SSD capacity | Approximately 144 GiB available. The 9 October handoff recorded a roughly 240 GiB full 3 s project on the PC; a new full rebuild may not fit. Recheck actual PC output capacity. |
| Current presentation artifacts | Top-level selected Figure 1 and Figure 2 SVG/HTML/legend hashes matched their state records. This was authentication, not a new visual or whole-figure scientific review. |

These raw checks covered filenames, triplet completeness and 3 s protocol
text. They did **not** hash the hundreds of gigabytes of camera/tracking data
or establish technical validity, fish inclusion or availability of new fish.
The SSD currently contains the previously known 57/59 complete recordings;
additional fish still need to be located on the PC or supplied as complete
triplets.

The old SSD handoff reports fixed-10 s raw data on the PC at
`D:\2023 03_3-s vs 10-s fixed trace\Raw data` and the 3 s processed project at
`F:\Digested Data\all3sTrace-full-v1`. These are **historical leads**, not paths
verified from this Mac. The mixed-assay directory requires protocol-based
selection and authenticated matched controls.

## Take the current code with you

The fresh SSD packet is `ClassicalConditioning-PC-20261018/` at the drive root:

- `HANDOFF.md`: this self-contained handout.
- `ClassicalConditioning-source.zip`: the current tracked working files,
  nonignored untracked files and Git metadata, preserving uncommitted changes
  and tracked deletions. Virtual environments, caches and raw/processed datasets
  are excluded. Existing external artwork stays in its recorded SSD locations.
- `handoff-manifest.json`: ZIP hash, source identity and transfer details.
- `source-files.json`: per-file hashes for the snapshot.
- `readiness-audit.json`: dated checks and limits.

Use this packet rather than the older `ClassicalConditioning-Mac-20261009`
checkout, whose selections predate the current 1 s heatmaps, Figure 2 A freeze
and Nature work. Git HEAD at preparation was
`31ee708c73956e59f27d959fe28b072150ca0056`; the uncommitted working files are
also required. A clone of that commit alone is insufficient.

On the PC, install/use Git, `uv` and CPython 3.12 or 3.13. In PowerShell,
substitute the actual SSD letter and a **new, empty local destination**:

```powershell
$ssd = 'J:\'
$packet = Join-Path $ssd 'ClassicalConditioning-PC-20261018'
$manifest = Get-Content (Join-Path $packet 'handoff-manifest.json') -Raw | ConvertFrom-Json
$zip = Join-Path $packet 'ClassicalConditioning-source.zip'
if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifest.archive_sha256) {
    throw 'Source ZIP hash mismatch'
}
$destination = Join-Path $HOME 'Documents\ClassicalConditioning-PC-20261018'
if (Test-Path $destination) { throw 'Choose a new empty destination' }
Expand-Archive -LiteralPath $zip -DestinationPath $destination
Set-Location (Join-Path $destination 'ClassicalConditioning')
git status --short
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-pc'
uv sync --frozen --all-extras --python 3.12
uv run classical-conditioning environment-report
uv run python -m unittest tests.test_configuration tests.test_trace_preparation
```

The dirty Git status is expected: preserve it. Do not reset/clean or copy a
Mac virtual environment to Windows. Run the repository from the PC's local
disk; keep large inputs and permanent outputs on external storage with enough
space. For new paper artwork also verify the browser/Node rendering tools and
Inkscape where required by the chosen exporter.

## Configure processing of additional fish

Use `uv run classical-conditioning run-pipeline --config <JSON>` from the new
checkout. Choose a new analysis ID and output directory for an expanded cohort.
Keep the original `allDelay-full-v1` and prior accepted results intact. All
completed triplets are processing candidates; a reviewed paper cohort is a
separate decision.

Do not use the old Windows launchers unchanged:

- `run-allDelay-full-windows.ps1` and `complete-allDelay-to-lme-windows.ps1`
  expect exactly 57 fish and create a technical all-complete cohort. They are
  unsuitable for an expanded cohort or for choosing publication inclusions.
- `resume-allDelay-full-windows.ps1` has fixed user/drive paths.
- `all3sTrace-full-windows.json` contains a fixed 59-recording selection.
- `run-all3sTrace-full-windows.ps1` has machine-specific defaults, regenerates
  that config and fixes the output name. Prefer a fresh preflight/config.

### Delay

This example writes a new config under a dedicated derived project. Set
`$derived` to an existing output root with sufficient free space; `F:` is only
an example. The raw input can remain on JOAQUIM even if outputs use another
drive. Do not put derived files into the raw folder.

```powershell
$derived = 'F:\Digested Data'
if (-not (Test-Path $derived -PathType Container)) { throw 'Output root unavailable' }
$delayProject = Join-Path $derived 'allDelay-expanded-20261018'
if (Test-Path $delayProject) { throw 'For a first run choose a new project; review existing runs before resuming' }
$delayRaw = Join-Path $ssd 'Raw Data\allDelay'
if (-not (Test-Path $delayRaw -PathType Container)) { throw 'Delay raw folder unavailable' }
New-Item -ItemType Directory -Path (Join-Path $delayProject 'Metadata') | Out-Null
$delayConfig = Join-Path $delayProject 'Metadata\run.json'
[ordered]@{
    raw_dir = $delayRaw
    save_dir = $delayProject
    experiment = 'allDelay'
    analysis_id = 'allDelay-expanded-20261018'
    keep_conditions = @('control', 'delay')
    overwrite = $false
    continue_on_error = $true
    metric = 'legacy_distal_angular_speed'
    assessment_metric = 'legacy_distal_angular_speed'
    figure_mode = 'static'
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $delayConfig -Encoding ascii
uv run classical-conditioning run-pipeline --config $delayConfig
```

Point `raw_dir` to the actual expanded raw dataset first if more Delay fish
live elsewhere. Otherwise this processes only the 57 already on the SSD.
For a safe resume, rerun the final command with the saved config and
`overwrite: false`; reuse requires verified markers/hashes. Review failures
even when `continue_on_error` allows the remaining fish to proceed.

### Fixed 3 s Trace

Use the existing preflight in a **new external configuration directory**, so
it cannot overwrite the checked-in 59-fish selection. It reads protocol timing
and rejects increasing-trace fish. With the SSD's present incomplete groups,
`--allow-incomplete` processes the 59 complete triplets and reports the other
four separately; it does not approve their exclusion from a paper.

```powershell
$traceRun = Join-Path $derived 'trace3-run-20261018'
if (Test-Path $traceRun) { throw 'Choose a new run directory or review the saved run before resuming' }
New-Item -ItemType Directory -Path $traceRun | Out-Null
$traceRaw = Join-Path $ssd 'Raw Data\all3sTtrace'
uv run python scripts/workflows/trace/prepare_runs.py `
    --raw-dir $traceRaw --digested-dir $traceRun `
    --config-dir (Join-Path $traceRun 'configs') `
    --audit-dir (Join-Path $traceRun 'audit') --allow-incomplete
if ($LASTEXITCODE -ne 0) { throw 'Trace preflight blocked the run' }
$traceConfig = Join-Path $traceRun 'configs\all3sTrace-full-windows.json'
uv run classical-conditioning run-pipeline --config $traceConfig
```

The generated project is `$traceRun\all3sTrace-full-v1`; the analysis ID is
`all3sTrace-full`. This output name is generated by the preflight, not a new
scientific version selection. For additional fish, use the expanded raw folder,
regenerate the preflight before launch, and compare planned IDs against the
pipeline ledger after completion. The helper scans a flat folder; recursively
nested new input needs a supported inventory/explicit selection instead.

### Fixed 10 s Trace

**Data/preflight blocked at this handoff.** First locate complete fixed-10 s
triplets and establish their matched control set. Use `experiment: all10sTrace`
only after verifying the fixed protocol: CS duration 10 s, paired US onset
20 s after CS onset (10 s after CS offset). Delay and 3 s paired onsets are
9 s and 13 s respectively. `incTrace` is not an alternative source.

Inventory the located mixed-assay folder outside the raw tree. A quick
`inventory --skip-hashes` can help locate candidates; routine processing must
then generate its normal source-hashed inventory. Select explicit recording
IDs supported by protocol and control provenance, not every `trace` or control
in the mixed folder. Original fixed-10 s filenames may contain `10sFixedTrace`,
which inventory lowercases to `10sfixedtrace`, whereas the canonical experiment
condition is `trace`. A `keep_conditions: ["control", "trace"]` filter alone
can miss those fish. Verify selected/active IDs and canonical outcome conditions
before trusting a run. There is no prepared fixed-10 s all-fish launcher/config
in this packet, and the 3 s preflight accepts 13 s onsets only.

## Carry the same science into Figures 1/2

Read `AGENTS.md`, `.agents/skills/paper-figures/SKILL.md`,
`docs/paper/figures/README.md`, and `records/frozen-analyses/selections.md` in
the extracted checkout before adapting any renderer. Resolve presentation from
each figure's `current-state.json`; use selected scientific records for methods.

| Scope | Definition to preserve when making an expanded-cohort candidate |
| --- | --- |
| Figure 1 F/G/H; Figure 2 A and future B/C | Finite positive valid adjacent detected-bout frame vigor, natural log per original frame, direct medians in 1 s bins; subtract the median of finite baseline-bin medians over [−15,0), each baseline bin one vote. One finite baseline bin suffices. No amplitude scaling or numerical clipping; missing black, native `managua_r` at ±0.25. Population cells average centered fish values with equal fish weight; retain contributing-fish counts. |
| Timing for those heatmaps | Accepted camera-cadence reconstruction, not a silent substitution of measured `AbsoluteTime`. `figures/pooled_log_heatmap.py` is the supported calculation reference; its loss-evidence gate must remain explicit. |
| Figure 1 D/E | D and the black E trace use measured times; E has its own accepted raw-baseline scaling, 0–5 display, and exact selected heatmap rows as color bands. A generic native-vigor trace export does not reproduce E. Preserve the D/E/example-heatmap fish/trial links. |
| Figure 2 D/E | Frozen B: median natural log of positive valid bout frames in response minus baseline, then median eligible trials per fish/block. Baseline [−15,0); response D [0,9), E [0,13). PT10–14, ET65–69, LT90–94; minimum eligible trials D=3/E=1. Accepted family: 36 two-sided exact-sign/Brunner–Munzel-t tests across D/E, Holm36. |
| Figure 2 G | Frozen NativeHistoricalLogMedian response-minus-baseline trajectory and recorded phase-aware LMM/contrasts. Whole-fish bootstrap 5,000, seed 10, pointwise 95% intervals; frozen heavy-tail and Test3 limitations remain. The generic total-activity learning-onset launcher is a different analysis. |
| Fixed 10 s additions | Experiment response window [0,20) exists. Figure 2 C/F/I remain unfinished; neither the exact inferential family for a new F nor a G-style 10 s analysis is approved by the existing freezes. Extending the D/E test family changes multiplicity and needs an explicit scientific candidate. |

The normal corrected pipeline supplies intake, corrected metrics, movement,
profiles, outcomes, assessment and descriptive figures. Its signed profile
figures are **not** proof of reproducing the selected cadence-based heatmaps.
`build_figure2_pooled_heatmaps.py` currently targets Delay, the old cohort,
`-v1` inputs and Mac output paths. Other historical Figure 1/2 recipes also
bind specific sources. Parameterize/reconcile these in supported code for new
projects and reproduce the existing accepted example cells before calculating
expanded Figure 1/2 candidates. Do not launch them unchanged against a new cohort.

New fish change scientific inputs. Retain original accepted freezes; publish
new candidates with cohort IDs, source hashes, timing/metric definitions,
eligibility counts and inference settings. Figure 2 B retains earlier data;
future B/C must follow A's pooled definition. H remains a different exploratory
ratio outcome until an explicit scientific change is made.

## Windows paths and completion checks

Set `CLASSICAL_CONDITIONING_ARTIFACT_ROOT` to an existing external output
directory for tools that use that setting. It chooses **new output paths**;
set `CLASSICAL_CONDITIONING_JOAQUIM_ROOT` to the SSD's actual existing volume
root when overriding discovery. Windows detects a unique mounted `JOAQUIM`
volume; `J:` is the currently observed letter. Recorded Mac/Windows references
resolve through the shared resolver and authenticated consolidation indexes.
Current figure/study bindings have been reconciled under publication guards.
Use the Python 3.13 environment and verification commands in the maintained
[storage guide](../project-maintenance/storage.md#host-configuration-and-verification).
The former machine mirror and dated handoff directories have been consolidated;
their historical names remain recovery aliases, not active output destinations.
The main-paper collection's final saved wrapper has been recovered through its
authenticated completion receipt without rebuilding SSD artwork or HTML.
Mac-side execution and cross-machine locking remain explicitly pending. Never rewrite frozen manifests/hashes,
invent a mount or substitute an older artifact to bypass a missing path.
`mac_ssd_run.py` is the Windows-to-Mac adapter, not a reverse PC adapter.

For each candidate run, inspect `Metadata/<analysis-id>_pipeline_run.json`,
inventory/intake status, per-recording acquisition QC, completion markers and
the discarding assessment. Compare **planned IDs, active IDs, failures and
figure dispositions**. Computational completion is separate from reviewed
inclusion and paper approval. Keep each experiment's matched controls explicit.

New paper presentation uses `nature-main`, the shared theme, scoped rc changes
and current 1 mm invisible-frame policy, followed by exact-render review and
guarded publication. The authenticated assembly pointers select the current
drafts; historical migration notes do not override them. No new freeze is
requested by this handoff.

## Resume instruction for the next PC chat

> Continue Figure 1/2 analysis for additional Delay, fixed 3 s Trace and fixed
> 10 s Trace fish using this handoff and the connected JOAQUIM SSD. Preserve all
> existing dirty files, raw data, current selected artwork and immutable freezes.
> Verify the new source snapshot, actual drive letters, PC-only processed/raw
> datasets, free output space and protocol/control membership first. Inventory
> additional fish and run supported candidate processing into dedicated projects.
> Then reconcile Windows artifact paths and adapt the exact selected Figure 1/2
> scientific recipes to expanded cohorts, reproducing accepted examples first.
> Record unresolved inference choices and missing inputs. Produce reviewable
> candidates; do not silently replace accepted selections or freeze anything.

Local evidence: `records/repository-history/2026-10-18-pc-handoff/readiness-audit.json`.
No real cohort was reprocessed during handoff preparation; no Windows execution
or fixed-10 s data validation was performed on this Mac.
