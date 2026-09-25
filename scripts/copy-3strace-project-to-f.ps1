<#
.SYNOPSIS
Copies the existing 3sTrace project from J: to F: and verifies every copied file.

.DESCRIPTION
Leaves J: intact. Excludes dot-prefixed intake staging directories, which are
not completed artifacts. The pipeline will rebuild the copied path-bound
artifacts at F: before using them for analysis.
#>
param(
    [string]$SourceProject = 'J:\Digested Data\all3sTrace-full-v1',
    [string]$DestinationProject = 'F:\Digested Data\all3sTrace-full-v1',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
function Get-HashWithRetry {
    param([string]$Path)
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            return (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds 2
        }
    }
}
$source = (Resolve-Path -LiteralPath $SourceProject).Path.TrimEnd('\')
$destination = [System.IO.Path]::GetFullPath($DestinationProject).TrimEnd('\')
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Source project is absent: $source"
}
if ([System.IO.Path]::GetPathRoot($source) -eq [System.IO.Path]::GetPathRoot($destination)) {
    throw 'The source and destination must be on separate drives.'
}
if (Test-Path -LiteralPath $destination) {
    if (@(Get-ChildItem -LiteralPath $destination -Force).Count -gt 0 -and -not $Resume) {
        throw "Destination already contains files; use -Resume to verify and continue: $destination"
    }
}

$sourceFiles = @(
    Get-ChildItem -LiteralPath $source -Directory -Force |
        Where-Object { -not $_.Name.StartsWith('.') } |
        ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Force }
)
if ($sourceFiles.Count -eq 0) { throw 'No completed project files to copy.' }
$snapshot = @{}
foreach ($file in $sourceFiles) {
    if (-not $file.FullName.StartsWith($source + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Computed source escaped the 3sTrace project: $($file.FullName)"
    }
    if ($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        throw "Source contains a linked file; refusing to copy outside data: $($file.FullName)"
    }
    $relative = $file.FullName.Substring($source.Length + 1)
    $snapshot[$relative] = [pscustomobject]@{
        Length = $file.Length
        LastWriteTimeUtc = $file.LastWriteTimeUtc
    }
}

New-Item -ItemType Directory -Path $destination -Force | Out-Null
$verified = @()
foreach ($file in $sourceFiles) {
    $relative = $file.FullName.Substring($source.Length + 1)
    $target = [System.IO.Path]::GetFullPath((Join-Path $destination $relative))
    if (-not $target.StartsWith($destination + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Computed copy target escaped destination: $target"
    }
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
        Copy-Item -LiteralPath $file.FullName -Destination $target -ErrorAction Stop
    }
    $copied = Get-Item -LiteralPath $target
    if ($copied.Length -ne $snapshot[$relative].Length) {
        throw "Copied file length differs: $relative"
    }
}

foreach ($file in $sourceFiles) {
    $relative = $file.FullName.Substring($source.Length + 1)
    $current = Get-Item -LiteralPath $file.FullName
    if ($current.Length -ne $snapshot[$relative].Length -or
        $current.LastWriteTimeUtc -ne $snapshot[$relative].LastWriteTimeUtc) {
        throw "Source changed during copy: $relative"
    }
    $target = Join-Path $destination $relative
    $sourceHash = Get-HashWithRetry $file.FullName
    $targetHash = Get-HashWithRetry $target
    if ($sourceHash -ne $targetHash) {
        throw "SHA-256 verification failed: $relative"
    }
    $verified += [pscustomobject]@{relative_path=$relative; size_bytes=$current.Length; sha256=$sourceHash}
}
$reportDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'outputs\trace-transfer-review'
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$reportPath = Join-Path $reportDir 'verified-copy.json'
[ordered]@{
    source_project = $source
    destination_project = $destination
    generated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    completed_file_count = $verified.Count
    completed_bytes = ($verified | Measure-Object size_bytes -Sum).Sum
    files = $verified
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath -Encoding utf8
Write-Output "Verified copy of $($sourceFiles.Count) completed files to $destination; source preserved at $source"
Write-Output "Transfer report: $reportPath"
