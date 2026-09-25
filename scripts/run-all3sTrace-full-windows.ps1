<#
.SYNOPSIS
Refreshes the fixed-trace preflight and starts its candidate pipeline.

.DESCRIPTION
All derived artifacts go below DigestedDir\all3sTrace-full-v1. The preflight
rejects increasing-trace protocols. With AllowIncomplete, it selects every
complete fixed-trace/control triplet and records incomplete groups separately.
#>
param(
    [string]$RawDir = 'J:\Raw Data\all3sTtrace',
    [string]$DigestedDir = 'F:\Digested Data',
    [string]$PythonPath = 'C:\Users\joaquim\.local\bin\python3.12.exe',
    [string]$UvPath = 'C:\Users\joaquim\.local\bin\uv.exe',
    [switch]$AllowIncomplete
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$configDir = Join-Path $repo 'configs'
$configPath = Join-Path $configDir 'all3sTrace-full-windows.json'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $repo '.venv-trace'

# Trace review HTML and the downstream learner/learning-onset fits use the
# project's optional interactive, legacy, and analysis dependencies.
& $UvPath sync --frozen --extra analysis --extra legacy-learners --extra interactive
if ($LASTEXITCODE -ne 0) {
    throw "Dependency sync failed with exit code $LASTEXITCODE"
}

$preflightArgs = @('--raw-dir', $RawDir, '--digested-dir', $DigestedDir, '--config-dir', $configDir)
if ($AllowIncomplete) { $preflightArgs += '--allow-incomplete' }
& $PythonPath (Join-Path $PSScriptRoot 'prepare-trace-runs.py') @preflightArgs
if ($LASTEXITCODE -ne 0) {
    throw "Trace preflight failed with exit code $LASTEXITCODE"
}
$planned = @((Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).recording_ids)

& $UvPath run --offline --no-sync classical-conditioning run-pipeline --config $configPath
if ($LASTEXITCODE -ne 0) {
    throw "Trace pipeline failed with exit code $LASTEXITCODE"
}

# A second preflight catches files added during the run. A complete candidate
# manifest must cover exactly the same fish selected at launch.
& $PythonPath (Join-Path $PSScriptRoot 'prepare-trace-runs.py') @preflightArgs
if ($LASTEXITCODE -ne 0) {
    throw 'Raw trace data changed or became incomplete during the pipeline run'
}
$current = @((Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).recording_ids)
if (@(Compare-Object $planned $current).Count -ne 0) {
    throw 'Trace recordings changed during the pipeline run; the output does not cover every current fish'
}
$project = Join-Path $DigestedDir 'all3sTrace-full-v1'
$summaryPath = Join-Path $project 'Metadata\all3sTrace-full_pipeline_run.json'
$summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
if ($summary.status -ne 'complete' -or
    @(Compare-Object $planned @($summary.recording_ids)).Count -ne 0 -or
    @(Compare-Object $planned @($summary.active_recording_ids)).Count -ne 0 -or
    -not $summary.candidate_runner_status) {
    throw 'Pipeline summary is not complete for every selected trace and control fish'
}
Write-Output "Completed fixed-trace pipeline for $($planned.Count) fish: $project"
