<#
.SYNOPSIS
Refreshes the fixed-trace preflight and starts its candidate pipeline.

.DESCRIPTION
All derived artifacts go below DigestedDir\all3sTrace-full-v1. The preflight
rejects any increasing-trace protocol or incomplete recording in RawDir, then
selects every trace and control triplet present when this command starts.
#>
param(
    [string]$RawDir = 'J:\Raw Data\all3sTtrace',
    [string]$DigestedDir = 'J:\Digested Data',
    [string]$PythonPath = 'C:\Users\joaquim\.local\bin\python3.12.exe',
    [string]$UvPath = 'C:\Users\joaquim\.local\bin\uv.exe'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$configDir = Join-Path $repo 'configs'
$configPath = Join-Path $configDir 'all3sTrace-full-windows.json'
$env:UV_CACHE_DIR = Join-Path $repo '.uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $repo '.venv-trace'

& $PythonPath (Join-Path $PSScriptRoot 'prepare-trace-runs.py') `
    --raw-dir $RawDir --digested-dir $DigestedDir --config-dir $configDir
if ($LASTEXITCODE -ne 0) {
    throw "Trace preflight failed with exit code $LASTEXITCODE"
}
$planned = @((Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).recording_ids)

& $UvPath run --offline classical-conditioning run-pipeline --config $configPath
if ($LASTEXITCODE -ne 0) {
    throw "Trace pipeline failed with exit code $LASTEXITCODE"
}

# A second preflight catches files added during the run. A complete candidate
# manifest must cover exactly the same fish selected at launch.
& $PythonPath (Join-Path $PSScriptRoot 'prepare-trace-runs.py') `
    --raw-dir $RawDir --digested-dir $DigestedDir --config-dir $configDir
if ($LASTEXITCODE -ne 0) {
    throw 'Raw trace data changed or became incomplete during the pipeline run'
}
$current = @((Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).recording_ids)
if (@(Compare-Object $planned $current).Count -ne 0) {
    throw 'Trace recordings changed during the pipeline run; the output does not cover every current fish'
}
$project = Join-Path $DigestedDir 'all3sTrace-full-v1'
$manifestPath = Join-Path $project 'Metadata\all3sTrace-full-v1-candidate_candidate-corrected-runner-v1_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.status -ne 'complete' -or @(Compare-Object $planned @($manifest.recording_ids)).Count -ne 0) {
    throw 'Candidate manifest is not complete for every selected trace and control fish'
}
Write-Output "Completed fixed-trace pipeline for $($planned.Count) fish: $project"
