<# Complete the isolated 53-fish legacy-metric Figure 2/3/4 preview. #>
param([int]$TimeoutHours = 12)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv-trace\Scripts\python.exe'
$project = 'F:\Digested Data\all3sTrace-full-v1'
$preview = Join-Path $repo 'outputs\trace-legacy-preview\20260925-window13'
$statusPath = Join-Path $preview 'preview-status.json'
$log = Join-Path $preview 'preview-post.log'
$cohort = 'all3sTrace-window13-legacy-20260925-53fish'
$metric = 'legacy_distal_angular_speed'
$figure2 = 'figure2-3strace-window13-legacy-53fish'
$figure3 = 'figure3-3strace-window13-legacy-53fish'
$figure4 = 'figure4-3strace-window13-legacy-53fish'
$assessment = Join-Path $project 'Processed data\Discarding\all3sTrace-window13-legacy-53fish\assessment-summary.json'
$comparison = Join-Path $project "Processed data\Analyses\$figure3\legacy-distal"
$labels = Join-Path $project "Processed data\Analyses\$figure4\learner-labels.csv"
$analysis = Join-Path $project "Processed data\Analyses\$figure4\figure4\analysis.json"
$figureRoot = Join-Path $project 'Figures\PNG\Analyses'
$figure4Dir = Join-Path $figureRoot $figure4
$figure2Heatmap = Join-Path $figureRoot "$figure2\signed-heatmap\figure-2B-3strace_signed_${metric}.png"
$deadline = (Get-Date).AddHours($TimeoutHours)

function Save-Status {
    param([string]$Stage, [string]$Details)
    [ordered]@{
        status = $Stage
        details = $Details
        cohort_id = $cohort
        fish_count = 53
        metric_id = $metric
        classifier_version = 'legacy-wip'
        response_window_s = @(0, 13)
        updated_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        log = $log
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    Save-Status 'running' $Name
    "Starting $Name" | Add-Content -LiteralPath $log
    & $python @Arguments *>> $log
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
    "Completed $Name" | Add-Content -LiteralPath $log
}

try {
    Save-Status 'waiting_for_assessment' 'Legacy-metric assessment is verifying 53 corrected fish'
    while (-not (Test-Path -LiteralPath $assessment -PathType Leaf)) {
        if ((Get-Date) -ge $deadline) { throw 'Timed out waiting for legacy-metric assessment' }
        Start-Sleep -Seconds 30
    }
    if (-not (Test-Path -LiteralPath $labels -PathType Leaf)) {
        Invoke-Step 'legacy-metric WIP labels' @((Join-Path $PSScriptRoot 'finalize_3strace_exploratory.py'),
            'classify', '--project-dir', $project, '--cohort-id', $cohort,
            '--comparison-dir', $comparison, '--analysis-id', $figure4,
            '--assessment-summary', $assessment, '--metric', $metric,
            '--classifier-execution-id', 'legacy-wip-3strace-distal-window13-53fish')
    }
    if (-not (Test-Path -LiteralPath $analysis -PathType Leaf)) {
        Invoke-Step 'legacy-metric Figure 4 panel data' @('-m', 'classical_conditioning',
            'figure4-analyze', '--project-dir', $project, '--analysis-id', $figure4,
            '--metric', $metric, '--trace3-cohort-id', $cohort,
            '--learner-manifest', $labels)
    }
    $figure4Kinds = @('figure-4', 'supplement-single-catches', 'supplement-movement',
        'supplement-coverage', 'supplement-single-catch-movement',
        'supplement-single-catch-coverage')
    $missing = @($figure4Kinds | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $figure4Dir "all3sTrace\${_}_${metric}.png") -PathType Leaf)
    })
    if ($missing.Count -gt 0) {
        Invoke-Step 'legacy-metric Figure 4 renders' @('-m', 'classical_conditioning',
            'figure4-render', '--analysis-summary', $analysis,
            '--output-dir', $figure4Dir, '--overwrite')
    }
    Save-Status 'waiting_for_figure2b' 'Figure 4 rendered; awaiting signed Figure 2B heatmap'
    while (-not (Test-Path -LiteralPath $figure2Heatmap -PathType Leaf)) {
        if ((Get-Date) -ge $deadline) { throw 'Timed out waiting for signed Figure 2B heatmap' }
        Start-Sleep -Seconds 30
    }
    foreach ($figureId in @($figure2, $figure3, $figure4)) {
        $source = Join-Path $figureRoot $figureId
        $destination = Join-Path $preview "figures\$figureId"
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Copy-Item -Path (Join-Path $source '*') -Destination $destination -Recurse -Force
    }
    Save-Status 'complete' 'Legacy-metric Figures 2, 3, and 4 rendered for 53 corrected fish'
}
catch {
    Save-Status 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath $log
    exit 1
}
