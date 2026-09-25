<#
.SYNOPSIS
Waits for the running 3sTrace pipeline, then completes its downstream review.

.DESCRIPTION
This helper is intended for one overnight run. It writes a small status JSON
and log in the repository's trace-transfer-review output directory. It never
modifies the J: source project.
#>
param(
    [string]$ProjectDir = 'F:\Digested Data\all3sTrace-full-v1',
    [int]$TimeoutHours = 24
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$project = [IO.Path]::GetFullPath($ProjectDir).TrimEnd('\')
$expected = [IO.Path]::GetFullPath('F:\Digested Data\all3sTrace-full-v1').TrimEnd('\')
if (-not $project.Equals($expected, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Watcher is restricted to the F: 3sTrace project: $project"
}
$output = Join-Path $repo 'outputs\trace-transfer-review'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$log = Join-Path $output 'overnight-post.log'
$statusPath = Join-Path $output 'overnight-post-status.json'
$pipelinePath = Join-Path $project 'Metadata\all3sTrace-full_pipeline_run.json'
$deadline = (Get-Date).AddHours($TimeoutHours)

function Save-Status {
    param([string]$Status, [string]$Details)
    [ordered]@{
        status = $Status
        details = $Details
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        pipeline_summary = $pipelinePath
        log = $log
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

Save-Status 'waiting_for_pipeline' 'Waiting for the 59-fish candidate run'
while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $pipelinePath -PathType Leaf) {
        try {
            $pipeline = Get-Content -LiteralPath $pipelinePath -Raw | ConvertFrom-Json
        } catch {
            Start-Sleep -Seconds 10
            continue
        }
        if ($pipeline.status -eq 'failed') {
            Save-Status 'pipeline_failed' (($pipeline.stage_errors | Select-Object -First 3) -join '; ')
            exit 1
        }
        if ($pipeline.status -eq 'complete') {
            if (@($pipeline.recording_ids).Count -ne 59 -or
                @(Compare-Object @($pipeline.recording_ids) @($pipeline.active_recording_ids)).Count -ne 0) {
                Save-Status 'pipeline_mismatch' 'Completed summary does not cover all 59 selected fish'
                exit 1
            }
            Save-Status 'running_downstream' 'Starting cohort, Figures 1-4 and learning-onset review'
            & pwsh -NoProfile -File (Join-Path $PSScriptRoot 'run-3strace-exploratory-post.ps1') -ProjectDir $project *>> $log
            if ($LASTEXITCODE -ne 0) {
                Save-Status 'downstream_failed' "Exit code $LASTEXITCODE; inspect $log"
                exit $LASTEXITCODE
            }
            Save-Status 'downstream_complete' '3sTrace exploratory outputs completed; J: source still preserved'
            exit 0
        }
    }
    Start-Sleep -Seconds 60
}
Save-Status 'timed_out' "No complete pipeline summary within $TimeoutHours hours"
exit 1
