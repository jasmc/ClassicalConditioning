<#
.SYNOPSIS
Completes the user-requested J: to F: 3sTrace move after full verification.

.DESCRIPTION
Waits for the separate downstream review helper to report completion, checks
the exact 3sTrace project paths and every required authenticated artifact,
then removes only J:\Digested Data\all3sTrace-full-v1. Other J:/F: data is never
included in the deletion target.
#>
param([int]$TimeoutHours = 24)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$statusPath = Join-Path $repo 'outputs\trace-transfer-review\overnight-post-status.json'
$readinessPath = Join-Path $repo 'outputs\trace-transfer-review\move-readiness.json'
$moveStatusPath = Join-Path $repo 'outputs\trace-transfer-review\move-status.json'
$python = Join-Path $repo '.venv-trace\Scripts\python.exe'
$expectedParent = [IO.Path]::GetFullPath('J:\Digested Data').TrimEnd('\')
$expectedSource = [IO.Path]::GetFullPath('J:\Digested Data\all3sTrace-full-v1').TrimEnd('\')
$expectedDestination = [IO.Path]::GetFullPath('F:\Digested Data\all3sTrace-full-v1').TrimEnd('\')

function Save-MoveStatus {
    param([string]$Status, [string]$Details)
    [ordered]@{
        status = $Status
        details = $Details
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        source_project = $expectedSource
        destination_project = $expectedDestination
        readiness_report = $readinessPath
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $moveStatusPath -Encoding utf8
}

Save-MoveStatus 'waiting_for_downstream' 'Awaiting completed 3sTrace review on F:'
$deadline = (Get-Date).AddHours($TimeoutHours)
while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $statusPath -PathType Leaf) {
        $post = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($post.status -in @('pipeline_failed', 'pipeline_mismatch', 'downstream_failed', 'timed_out')) {
            Save-MoveStatus 'blocked_by_review_failure' $post.details
            exit 1
        }
        if ($post.status -eq 'downstream_complete') { break }
    }
    Start-Sleep -Seconds 60
}
if (-not (Test-Path -LiteralPath $statusPath -PathType Leaf) -or
    (Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json).status -ne 'downstream_complete') {
    Save-MoveStatus 'timed_out' 'No completed downstream review within timeout'
    exit 1
}

$parent = (Resolve-Path -LiteralPath $expectedParent).Path.TrimEnd('\')
$source = (Resolve-Path -LiteralPath $expectedSource).Path.TrimEnd('\')
$destination = (Resolve-Path -LiteralPath $expectedDestination).Path.TrimEnd('\')
if (-not $parent.Equals($expectedParent, [StringComparison]::OrdinalIgnoreCase) -or
    -not $source.Equals($expectedSource, [StringComparison]::OrdinalIgnoreCase) -or
    -not $destination.Equals($expectedDestination, [StringComparison]::OrdinalIgnoreCase) -or
    -not $source.StartsWith($parent + '\', [StringComparison]::OrdinalIgnoreCase) -or
    (Get-Item -LiteralPath $source).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    Save-MoveStatus 'unsafe_target' 'Resolved move target differs from exact 3sTrace paths'
    exit 1
}
& $python (Join-Path $PSScriptRoot 'verify_3strace_move_readiness.py') `
    --project-dir $destination --source-dir $source --output $readinessPath
if ($LASTEXITCODE -ne 0) {
    Save-MoveStatus 'verification_failed' 'Move readiness verification rejected source removal'
    exit $LASTEXITCODE
}
$readiness = Get-Content -LiteralPath $readinessPath -Raw | ConvertFrom-Json
if ($readiness.status -ne 'ready_to_remove_j_3strace_project' -or
    $readiness.source_project -ne $source -or
    $readiness.destination_project -ne $destination) {
    Save-MoveStatus 'verification_failed' 'Readiness report identity mismatch'
    exit 1
}

# This is the only destructive operation. All resolved paths above are fixed
# to the one user-named source project, and its F: replacement is authenticated.
Remove-Item -LiteralPath $source -Recurse -Force -ErrorAction Stop
if (Test-Path -LiteralPath $source) {
    Save-MoveStatus 'removal_incomplete' 'J: 3sTrace source still exists'
    exit 1
}
Save-MoveStatus 'move_complete' 'Only the J: 3sTrace derived project was removed after F: verification'
exit 0
