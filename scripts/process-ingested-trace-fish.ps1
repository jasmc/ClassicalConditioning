<#
.SYNOPSIS
Processes every fully ingested 3 s trace or control fish through per-fish trial summaries.

.DESCRIPTION
Waits for an optional intake process to finish, then reads the source manifests
in ProjectDir. Run again after more complete raw triplets have been ingested.
The cohort analysis is deliberately separate because the raw cohort is still
receiving files.
#>
param(
    [string]$ProjectDir = 'J:\Digested Data\all3sTrace-full-v1',
    [int]$IntakePid = 0,
    [string]$UvPath = 'C:\Users\joaquim\.local\bin\uv.exe'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $repo '.uv-cache'
$env:UV_PROJECT_ENVIRONMENT = Join-Path $repo '.venv-trace'
$metadata = Join-Path $ProjectDir 'Metadata'

while ($IntakePid -gt 0 -and (Get-Process -Id $IntakePid -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 15
}

$manifests = @(Get-ChildItem -LiteralPath $metadata -File -Filter '*_source_manifest.json' | Sort-Object Name)
$failures = @()
foreach ($manifest in $manifests) {
    $id = $manifest.Name.Replace('_source_manifest.json', '')
    Write-Output "Processing $id"
    $stages = @(
        @{Marker="${id}_corrected-preprocess_complete.json"; Args=@('preprocess','--experiment','all3sTrace')},
        @{Marker="${id}_candidate-corrected_complete.json"; Args=@('activity-metrics','--recipe','corrected')},
        @{Marker="${id}_movement-candidate-corrected_complete.json"; Args=@('movement-state','--recipe','movement-candidate-corrected')},
        @{Marker="${id}_candidate-temporal-outcomes-corrected_complete.json"; Args=@('temporal-profiles','--recipe','candidate-temporal-outcomes-corrected','--experiment','all3sTrace')}
    )
    foreach ($stage in $stages) {
        if (Test-Path -LiteralPath (Join-Path $metadata $stage.Marker) -PathType Leaf) {
            continue
        }
        $command = @('run','--offline','--no-sync','classical-conditioning',$stage.Args[0],
            '--project-dir',$ProjectDir,'--recording-id',$id) + @($stage.Args | Select-Object -Skip 1)
        & $UvPath @command
        if ($LASTEXITCODE -ne 0) {
            $failures += "${id}: $($stage.Args[0]) (exit $LASTEXITCODE)"
            break
        }
    }
}

Write-Output "Processed $($manifests.Count) ingested fish; failures: $($failures.Count)"
if ($failures.Count) {
    $failures | ForEach-Object { Write-Error $_ }
    exit 1
}
