# Scoped platform dispatcher. Original stage parameters and checks are preserved.
param(
    [ValidateSet('pipeline-analysis', 'resume', 'complete-analysis')]
    [string]$Stage = 'pipeline-analysis'
)
$ErrorActionPreference = 'Stop'
$stages = @{
    'pipeline-analysis' = 'delay_pipeline_analysis.ps1'
    'resume' = 'delay_resume.ps1'
    'complete-analysis' = 'delay_complete_analysis.ps1'
}
$command = Join-Path (Join-Path $PSScriptRoot 'stages') $stages[$Stage]
& $command @args
exit $LASTEXITCODE
