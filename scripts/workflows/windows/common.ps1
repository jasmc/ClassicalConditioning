# Shared manifest completion predicate; scientific cohort selection stays in stages.
function Test-CompleteCandidateRun {
    param([string]$ManifestPath, [int]$ExpectedRecordings)
    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { return $false }
    $manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    return ($manifest.status -eq 'complete' -and @($manifest.recording_ids).Count -eq $ExpectedRecordings)
}
