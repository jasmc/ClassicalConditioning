<# Wait for all 59 fish to use the 0–13 s response window, then rebuild dependent outputs. #>
param([int]$TimeoutHours = 24)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$output = Join-Path $repo 'outputs\trace-transfer-review'
$rebuildPath = Join-Path $output 'window13-rebuild-status.json'
$statusPath = Join-Path $output 'window13-post-status.json'
$log = Join-Path $output 'window13-post.log'

function Save-Status {
    param([string]$Status, [string]$Details)
    [ordered]@{
        status = $Status
        details = $Details
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        rebuild_status = $rebuildPath
        log = $log
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

Save-Status 'waiting_for_window13_rebuild' 'Awaiting all 59 authenticated 0–13 s trial outcomes'
$deadline = (Get-Date).AddHours($TimeoutHours)
while ((Get-Date) -lt $deadline) {
    if (Test-Path -LiteralPath $rebuildPath -PathType Leaf) {
        try { $rebuild = Get-Content -LiteralPath $rebuildPath -Raw | ConvertFrom-Json }
        catch { Start-Sleep -Seconds 10; continue }
        if ($rebuild.status -eq 'failed') {
            Save-Status 'rebuild_failed' "$($rebuild.failed) fish failed; inspect fish logs"
            exit 1
        }
        if ($rebuild.status -eq 'complete') {
            if ($rebuild.fish_total -ne 59 -or $rebuild.completed -ne 59 -or $rebuild.failed -ne 0) {
                Save-Status 'rebuild_mismatch' 'Window13 rebuild status did not cover 59 fish'
                exit 1
            }
            Save-Status 'running_downstream' 'Building 0–13 s cohort, Figures 2–4 and learning onset'
            & pwsh -NoProfile -File (Join-Path $PSScriptRoot 'run-3strace-window13-post.ps1') *>> $log
            if ($LASTEXITCODE -ne 0) {
                Save-Status 'downstream_failed' "Exit code $LASTEXITCODE; inspect $log"
                exit $LASTEXITCODE
            }
            Save-Status 'downstream_complete' '0–13 s 3sTrace exploratory outputs completed; J: source preserved'
            exit 0
        }
    }
    Start-Sleep -Seconds 60
}
Save-Status 'timed_out' "No completed 0–13 s rebuild within $TimeoutHours hours"
exit 1
