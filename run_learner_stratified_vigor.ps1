param(
    [string]$TracePath = "C:\Users\Public\More projects\3sTrace_CS_new_logmedian.pkl",
    [string]$ControlPath = "C:\Users\Public\More projects\control_CS_new_logmedian.pkl",
    [string]$OutputPath = "$PSScriptRoot\build\learner_stratified\all3sTrace_full",
    [int]$BootstrapIterations = 1000,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$VenvPath = Join-Path $PSScriptRoot ".venv-learner-vigor"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"
$LogDirectory = Join-Path $OutputPath "logs"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$LogPath = Join-Path $LogDirectory "pipeline-$Timestamp.log"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "Creating isolated Python environment at $VenvPath"
    python -m venv $VenvPath
}

Write-Host "Installing the pinned analysis environment"
& $PythonPath -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot "requirements-learner-vigor.txt")

$arguments = @(
    (Join-Path $PSScriptRoot "learner_stratified_vigor_pipeline.py"),
    "--trace", $TracePath,
    "--control", $ControlPath,
    "--output", $OutputPath,
    "--bootstrap-iterations", $BootstrapIterations
)
if ($Force) {
    $arguments += "--force"
}

$env:MPLBACKEND = "Agg"
Write-Host "Running the full learner-stratified vigor pipeline"
Write-Host "Full output log: $LogPath"

function ConvertTo-CmdArgument([string]$Value) {
    return '"' + $Value.Replace('"', '""') + '"'
}

$CommandParts = @($PythonPath) + $arguments
$CommandLine = (($CommandParts | ForEach-Object { ConvertTo-CmdArgument $_ }) -join " ") + " 2>&1"
& $env:ComSpec /d /c $CommandLine | Tee-Object -LiteralPath $LogPath
$PipelineExitCode = $LASTEXITCODE

# Tee-Object uses UTF-16 in Windows PowerShell 5; normalize for portable logs.
$LogLines = Get-Content -LiteralPath $LogPath
[System.IO.File]::WriteAllLines(
    $LogPath,
    [string[]]$LogLines,
    [System.Text.UTF8Encoding]::new($false)
)

if ($PipelineExitCode -ne 0) {
    Write-Host ""
    Write-Host "Pipeline failed. Full Python traceback is preserved at:"
    Write-Host $LogPath
    throw "Learner-stratified vigor pipeline failed with exit code $PipelineExitCode."
}

Write-Host "Completed. Results: $OutputPath"
