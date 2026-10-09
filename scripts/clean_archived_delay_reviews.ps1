$ErrorActionPreference='Stop'
$taskRepo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskAuditPath=Join-Path $taskRepo 'reviews/delay_versions_cleanup_20261009.json'
$taskAudit=Get-Content -LiteralPath $taskAuditPath -Raw | ConvertFrom-Json
if ($taskAudit.status -ne 'verified-awaiting-cleanup') { throw 'Cleanup is not pending' }
if ((Get-FileHash -LiteralPath $taskAudit.html_path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskAudit.html_sha256) { throw 'Consolidated review changed' }
$taskReviewRoot=[IO.Path]::GetFullPath('F:\ClassicalConditioning Outputs\ORGER-JOAQUIM\outputs\figure2-assembly\row3-trial-ratio-review')
$taskFreezeRoot=[IO.Path]::GetFullPath('F:\ClassicalConditioning Outputs\ORGER-JOAQUIM\outputs\figure2-assembly\frozen\20261009-G-historical-logmedian')
$taskFreezeHashes=@{}
Get-ChildItem -LiteralPath $taskFreezeRoot -Recurse -File | ForEach-Object { $taskFreezeHashes[$_.FullName]=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
$taskKnownFiles=@{}
foreach ($taskRecord in $taskAudit.files) {
    $taskResolved=[IO.Path]::GetFullPath($taskRecord.original_path)
    if (-not $taskResolved.StartsWith($taskReviewRoot+'\',[StringComparison]::OrdinalIgnoreCase)) { throw "Outside review root: $taskResolved" }
    $taskItem=Get-Item -LiteralPath $taskResolved
    if ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse file: $taskResolved" }
    if ((Get-FileHash -LiteralPath $taskResolved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskRecord.sha256) { throw "Review file changed: $taskResolved" }
    if ($taskRecord.canonical_freeze_path -and (Get-FileHash -LiteralPath $taskRecord.canonical_freeze_path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskRecord.sha256) { throw 'Canonical freeze mismatch' }
    $taskKnownFiles[$taskResolved]=$taskRecord
}
# Validate every final absolute directory and every descendant before any deletion.
foreach ($taskDirectory in $taskAudit.directories) {
    $taskResolved=[IO.Path]::GetFullPath($taskDirectory)
    if ([IO.Path]::GetDirectoryName($taskResolved) -ne $taskReviewRoot) { throw "Unexpected cleanup directory: $taskResolved" }
    $taskItems=@(Get-Item -LiteralPath $taskResolved)+@(Get-ChildItem -LiteralPath $taskResolved -Recurse -Force)
    foreach ($taskItem in $taskItems) {
        if ($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse descendant: $($taskItem.FullName)" }
        if (-not $taskItem.PSIsContainer -and -not $taskKnownFiles.ContainsKey($taskItem.FullName)) { throw "Unrecorded file: $($taskItem.FullName)" }
    }
}
$taskRemovedBytes=0L
$taskRemovedCount=0
foreach ($taskDirectory in $taskAudit.directories) {
    Get-ChildItem -LiteralPath $taskDirectory -Recurse -File | ForEach-Object { $taskRemovedBytes+=$_.Length; $taskRemovedCount++ }
    Remove-Item -LiteralPath $taskDirectory -Recurse -Force
}
foreach ($taskFile in $taskFreezeHashes.Keys) {
    if ((Get-FileHash -LiteralPath $taskFile -Algorithm SHA256).Hash -ne $taskFreezeHashes[$taskFile]) { throw "Freeze changed during cleanup: $taskFile" }
}
$taskTemporaryImage=Join-Path $taskFreezeRoot 'visual-review-temporary.png'
$taskRemovedBytes+=(Get-Item -LiteralPath $taskTemporaryImage).Length
$taskRemovedCount++
Remove-Item -LiteralPath $taskTemporaryImage
$taskAudit.status='completed'
$taskAudit | Add-Member -NotePropertyName removed_bytes -NotePropertyValue $taskRemovedBytes -Force
$taskAudit | Add-Member -NotePropertyName removed_file_count -NotePropertyValue $taskRemovedCount -Force
$taskAudit | Add-Member -NotePropertyName freeze_hash_verification -NotePropertyValue 'All selected freeze files unchanged during cleanup; temporary visual-review PNG removed after review embedding' -Force
$taskAudit | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $taskAuditPath -Encoding utf8
Write-Output "Removed $taskRemovedCount stale files, $taskRemovedBytes bytes; selected freeze hashes unchanged."
