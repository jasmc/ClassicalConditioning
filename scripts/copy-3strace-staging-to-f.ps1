<# Copies and verifies the two legacy intake staging folders within the named 3sTrace project. #>
param()

$ErrorActionPreference = 'Stop'
$source = (Resolve-Path -LiteralPath 'J:\Digested Data\all3sTrace-full-v1').Path.TrimEnd('\')
$destination = (Resolve-Path -LiteralPath 'F:\Digested Data\all3sTrace-full-v1').Path.TrimEnd('\')
$expectedSource = [IO.Path]::GetFullPath('J:\Digested Data\all3sTrace-full-v1').TrimEnd('\')
$expectedDestination = [IO.Path]::GetFullPath('F:\Digested Data\all3sTrace-full-v1').TrimEnd('\')
if (-not $source.Equals($expectedSource, [StringComparison]::OrdinalIgnoreCase) -or
    -not $destination.Equals($expectedDestination, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'This copy is restricted to the exact J: and F: 3sTrace projects.'
}
$folders = @(
    '.20230307_12-intake-be77eadf21eb4411b2c33512fd892950',
    '.20230309_11-intake-2fb3ba2619a945f692660c78b09d5466'
)
$records = @()
foreach ($folder in $folders) {
    $sourceFolder = Join-Path $source $folder
    $destinationFolder = Join-Path $destination $folder
    if (-not (Test-Path -LiteralPath $sourceFolder -PathType Container)) {
        throw "Expected J: staging folder is missing: $sourceFolder"
    }
    if ((Get-Item -LiteralPath $sourceFolder).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "J: staging folder is a link: $sourceFolder"
    }
    if (Test-Path -LiteralPath $destinationFolder) {
        throw "F: staging folder already exists: $destinationFolder"
    }
    $files = @(Get-ChildItem -LiteralPath $sourceFolder -Recurse -File -Force)
    foreach ($file in $files) {
        if ($file.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "J: staging file is a link: $($file.FullName)"
        }
        $relative = $file.FullName.Substring($source.Length + 1)
        $target = [IO.Path]::GetFullPath((Join-Path $destination $relative))
        if (-not $target.StartsWith($destination + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Destination escaped the 3sTrace project: $target"
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target -ErrorAction Stop
        $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        $destinationHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ($sourceHash -ne $destinationHash -or $file.Length -ne (Get-Item -LiteralPath $target).Length) {
            throw "Staging file verification failed: $relative"
        }
        $records += [ordered]@{relative_path=$relative;size_bytes=$file.Length;sha256=$sourceHash}
    }
}
$reportPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'outputs\trace-transfer-review\verified-staging-copy.json'
[ordered]@{
    source_project = $source
    destination_project = $destination
    generated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    folder_count = $folders.Count
    file_count = $records.Count
    bytes = ($records | ForEach-Object { $_['size_bytes'] } | Measure-Object -Sum).Sum
    files = $records
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $reportPath -Encoding utf8
Write-Output "Verified $($records.Count) 3sTrace staging files on F:; J: preserved."
