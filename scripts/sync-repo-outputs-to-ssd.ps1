<#
Copy this computer's repository review outputs to the raw-data SSD.

The repository outputs directory remains the live working path. Copies are
written to a machine-specific directory so imports from another computer can
occupy a sibling directory without collisions. This script never deletes
source or destination files.
#>
param(
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$SsdRoot = 'J:\ClassicalConditioning Outputs',
    [string]$Machine = $env:COMPUTERNAME
)

$ErrorActionPreference = 'Stop'
$source = [System.IO.Path]::GetFullPath((Join-Path $Repo 'outputs'))
$ssd = [System.IO.Path]::GetFullPath($SsdRoot)
if (-not $ssd.StartsWith('J:\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "SSD output root must be on J: $ssd"
}
if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Repository outputs directory does not exist: $source"
}
if ($Machine -notmatch '^[A-Za-z0-9_-]+$') {
    throw "Machine name is not a safe directory name: $Machine"
}
$destination = Join-Path (Join-Path $ssd $Machine) 'outputs'
$destination = [System.IO.Path]::GetFullPath($destination)
if (-not $destination.StartsWith(($ssd.TrimEnd('\') + '\'), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination escaped SSD output root: $destination"
}

New-Item -ItemType Directory -Path $destination -Force | Out-Null
$copied = 0
$verified = 0
$unstable = [System.Collections.Generic.List[string]]::new()
$files = @(Get-ChildItem -LiteralPath $source -Recurse -File)
foreach ($file in $files) {
    $relative = [System.IO.Path]::GetRelativePath($source, $file.FullName)
    $target = Join-Path $destination $relative
    $targetDir = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    $temp = "$target.copying-$PID"
    try {
        $before = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        if ((Test-Path -LiteralPath $target -PathType Leaf) -and
            ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $before)) {
            $verified++
            continue
        }

        Copy-Item -LiteralPath $file.FullName -Destination $temp -Force
        $after = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        $copiedHash = (Get-FileHash -LiteralPath $temp -Algorithm SHA256).Hash
        if ($after -ne $before -or $copiedHash -ne $before) {
            $unstable.Add($relative)
            continue
        }
        Move-Item -LiteralPath $temp -Destination $target -Force
        $copied++
        $verified++
    }
    catch {
        # Running analysis processes can hold log files open without sharing.
        # A later sync will pick these up after the writer closes them.
        if (-not $unstable.Contains($relative)) { $unstable.Add($relative) }
    }
    finally {
        if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }
    }
}

$report = [ordered]@{
    source = $source
    destination = $destination
    machine = $Machine
    file_count_at_start = $files.Count
    files_verified = $verified
    files_copied = $copied
    unstable_files_to_retry = @($unstable)
    synced_at_utc = [DateTime]::UtcNow.ToString('o')
}
$reportPath = Join-Path (Split-Path -Parent $destination) 'sync-status.json'
$report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath -Encoding utf8
$report | ConvertTo-Json -Depth 5
if ($unstable.Count -gt 0) { exit 2 }
