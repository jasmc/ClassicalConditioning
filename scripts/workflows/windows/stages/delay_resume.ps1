param(
    [int]$InventoryProcessId = 0
)

$commandRepo = $PSScriptRoot
while (-not (Test-Path -LiteralPath (Join-Path $commandRepo 'pyproject.toml'))) {
    $parent = Split-Path -Parent $commandRepo
    if (-not $parent -or $parent -eq $commandRepo) { throw 'Repository root unavailable' }
    $commandRepo = $parent
}
$ErrorActionPreference = "Stop"

$uv = "C:\Users\joaquim\.local\bin\uv.exe"
$project = "F:\Digested Data\allDelay-full"
$inventoryPath = Join-Path $project "Metadata\recording_inventory.json"
$configPath = Join-Path $commandRepo "configs\allDelay-full-windows.json"

if ($InventoryProcessId -gt 0) {
    $inventoryProcess = Get-Process -Id $InventoryProcessId -ErrorAction SilentlyContinue
    if ($null -ne $inventoryProcess) {
        Wait-Process -Id $InventoryProcessId
    }
}

if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
    throw "The hash inventory did not complete: $inventoryPath is missing."
}

$inventory = Get-Content -LiteralPath $inventoryPath -Raw | ConvertFrom-Json
if (-not $inventory.source_hashes_included) {
    throw "The inventory at $inventoryPath does not contain source SHA-256 hashes."
}

& $uv run classical-conditioning run-pipeline --config $configPath
exit $LASTEXITCODE
