# Scoped platform dispatcher. Original stage parameters and checks are preserved.
param(
    [ValidateSet('pipeline', 'process-ingested')]
    [string]$Stage = 'pipeline'
)
$ErrorActionPreference = 'Stop'
$stages = @{
    'pipeline' = 'trace_pipeline.ps1'
    'process-ingested' = 'trace_process_ingested.ps1'
}
$command = Join-Path (Join-Path $PSScriptRoot 'stages') $stages[$Stage]
& $command @args
exit $LASTEXITCODE
