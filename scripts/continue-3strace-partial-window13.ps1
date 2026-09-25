<# Finish the immutable 36-fish 0–13 s preview without touching full-run IDs. #>
param([int]$TimeoutHours = 12)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv-trace\Scripts\python.exe'
$project = 'F:\Digested Data\all3sTrace-full-v1'
$preview = Join-Path $repo 'outputs\trace-partial-preview\20260925-window13'
$log = Join-Path $preview 'preview-post.log'
$statusPath = Join-Path $preview 'preview-status.json'
$cohort = 'all3sTrace-window13-partial-20260925-36fish'
$metric = 'tail_length_weighted_angular_l1'
$figure4 = 'figure4-3strace-window13-partial-36fish'
$learning = 'all3sTrace-window13-partial-36fish-learning-onset'
$assessment = Join-Path $project 'Processed data\Discarding\all3sTrace-window13-partial-36fish\assessment-summary.json'
$comparison = Join-Path $project 'Processed data\Analyses\figure3-3strace-window13-partial-36fish\legacy-tail-l1'
$labels = Join-Path $project "Processed data\Analyses\$figure4\learner-labels.csv"
$figure4Summary = Join-Path $project "Processed data\Analyses\$figure4\figure4\analysis.json"
$figure4Dir = Join-Path $project "Figures\PNG\Analyses\$figure4"
$learningMarker = Join-Path $project "Metadata\${learning}_learning-onset_complete.json"
$learningDir = Join-Path $project "Figures\PNG\Analyses\$learning"
$deadline = (Get-Date).AddHours($TimeoutHours)

function Save-Status {
    param([string]$Stage, [string]$Details)
    [ordered]@{
        status = $Stage
        details = $Details
        cohort_id = $cohort
        fish_count = 36
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
    Save-Status 'waiting_for_assessment' 'Preview assessment is verifying 36 corrected fish'
    while (-not (Test-Path -LiteralPath $assessment -PathType Leaf)) {
        if ((Get-Date) -ge $deadline) { throw 'Timed out waiting for preview assessment' }
        Start-Sleep -Seconds 30
    }
    if (-not (Test-Path -LiteralPath $labels -PathType Leaf)) {
        Invoke-Step 'provisional WIP labels' @((Join-Path $PSScriptRoot 'finalize_3strace_exploratory.py'),
            'classify', '--project-dir', $project, '--cohort-id', $cohort,
            '--comparison-dir', $comparison, '--analysis-id', $figure4,
            '--assessment-summary', $assessment,
            '--classifier-execution-id', 'legacy-wip-3strace-tail-l1-window13-partial-36fish')
    }
    if (-not (Test-Path -LiteralPath $figure4Summary -PathType Leaf)) {
        Invoke-Step 'Figure 4 partial panel data' @('-m', 'classical_conditioning',
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
        Invoke-Step 'Figure 4 partial renders' @('-m', 'classical_conditioning',
            'figure4-render', '--analysis-summary', $figure4Summary,
            '--output-dir', $figure4Dir, '--overwrite')
    }

    Save-Status 'waiting_for_learning' 'Figure 4 rendered; awaiting partial learning-onset model'
    while (-not (Test-Path -LiteralPath $learningMarker -PathType Leaf)) {
        if ((Get-Date) -ge $deadline) { throw 'Timed out waiting for partial learning-onset model' }
        Start-Sleep -Seconds 30
    }
    if (-not (Test-Path -LiteralPath (Join-Path $learningDir 'learning-diagnostics.png') -PathType Leaf)) {
        Invoke-Step 'partial learning diagnostics' @('-m', 'classical_conditioning',
            'figure-learning-diagnostics', '--project-dir', $project,
            '--analysis-id', $learning, '--mode', 'static')
    }
    if (-not (Test-Path -LiteralPath (Join-Path $learningDir 'learning-onset.png') -PathType Leaf)) {
        Invoke-Step 'partial learning-onset review' @('-m', 'classical_conditioning',
            'figure-learning-onset', '--project-dir', $project,
            '--analysis-id', $learning, '--mode', 'static', '--allow-unaccepted')
    }
    Save-Status 'complete' 'Partial Figures 2–4 and learning-onset review complete'
}
catch {
    Save-Status 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath $log
    exit 1
}
