<# Mirror active review outputs to J: while the current 3sTrace jobs run. #>
param(
    [int[]]$WaitForProcessIds = @(14992, 25796, 26212),
    [int]$IntervalSeconds = 300
)

$ErrorActionPreference = 'Stop'
$syncScript = Join-Path $PSScriptRoot 'sync-repo-outputs-to-ssd.ps1'
$pwsh = Join-Path $PSHOME 'pwsh.exe'
$watchStatus = Join-Path (Join-Path 'J:\ClassicalConditioning Outputs' $env:COMPUTERNAME) 'sync-watch-status.json'
$finalRetries = 0

while ($true) {
    & $pwsh -NoProfile -File $syncScript | Out-Null
    $syncExit = $LASTEXITCODE
    $active = @($WaitForProcessIds | Where-Object {
        $null -ne (Get-Process -Id $_ -ErrorAction SilentlyContinue)
    })
    $done = $active.Count -eq 0 -and $syncExit -eq 0
    $status = [ordered]@{
        status = if ($done) { 'complete' } elseif ($active.Count -eq 0) { 'retrying_final_sync' } else { 'watching' }
        active_process_ids = $active
        last_sync_exit_code = $syncExit
        updated_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    $status | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $watchStatus -Encoding utf8
    if ($done) { break }
    if ($active.Count -eq 0) {
        $finalRetries++
        if ($finalRetries -ge 10) { break }
        Start-Sleep -Seconds 60
    }
    else {
        Start-Sleep -Seconds $IntervalSeconds
    }
}
