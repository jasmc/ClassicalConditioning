<# Rebuild the exploratory 3sTrace outputs that depend on the 0–13 s response window. #>
param(
    [string]$ProjectDir = 'F:\Digested Data\all3sTrace-full-v1',
    [string]$PythonPath = 'C:\Users\joaquim\Documents\ClassicalConditioning\.venv-trace\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$project = [IO.Path]::GetFullPath($ProjectDir).TrimEnd('\')
$expected = [IO.Path]::GetFullPath('F:\Digested Data\all3sTrace-full-v1').TrimEnd('\')
if (-not $project.Equals($expected, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'This rebuild is restricted to the named F: 3sTrace project.'
}
$ids = @((Get-Content -LiteralPath (Join-Path $project 'Metadata\all3sTrace-full_pipeline_run.json') -Raw |
    ConvertFrom-Json).recording_ids)
if ($ids.Count -ne 59) { throw 'Expected the frozen 59-fish 3sTrace candidate set.' }
foreach ($id in $ids) {
    $summary = Get-Content -LiteralPath (Join-Path $project "Quality checks\$id\candidate-trial-outcomes-corrected_summary.json") -Raw | ConvertFrom-Json
    if ($summary.experiment -ne 'all3sTrace' -or
        @($summary.config.response_window_s).Count -ne 2 -or
        [double]$summary.config.response_window_s[0] -ne 0 -or
        [double]$summary.config.response_window_s[1] -ne 13) {
        throw "$id still has an obsolete candidate response window."
    }
}
$cohort = 'all3sTrace-full-exploratory'
$metric = 'tail_length_weighted_angular_l1'
$figure2Id = 'figure2-3strace-window13'
$figure3Id = 'figure3-3strace-window13'
$figure4Id = 'figure4-3strace-window13'
$learningId = 'all3sTrace-full-learning-onset-window13'
$assessmentId = 'all3sTrace-full-window13'

function Invoke-Python {
    param([string]$Stage, [string[]]$Arguments)
    Write-Output "Starting $Stage"
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Stage failed with exit code $LASTEXITCODE" }
    Write-Output "Completed $Stage"
}

# The current loader rejects the old 0–9 s source setting even when its cohort
# table hashes still match; only republish when the authenticated load fails.
& $PythonPath -c 'import sys; from pathlib import Path; from classical_conditioning.analysis.cohort_outcomes import load_cohort_trial_outcomes; load_cohort_trial_outcomes(Path(sys.argv[1]), sys.argv[2])' $project $cohort 2>$null
if ($LASTEXITCODE -ne 0) {
    Invoke-Python '0–13 s cohort trial outcomes' @('-m', 'classical_conditioning',
        'build-cohort-trial-outcomes', '--project-dir', $project, '--cohort-id', $cohort,
        '--metric-recipe', 'tail-candidate-corrected', '--overwrite')
}

$assessment = Join-Path $project "Processed data\Discarding\$assessmentId\assessment-summary.json"
if (-not (Test-Path -LiteralPath $assessment -PathType Leaf)) {
    $fishArgs = @()
    foreach ($id in $ids) { $fishArgs += @('--recording-id', $id) }
    Invoke-Python '0–13 s exploratory assessment' (@('-m', 'classical_conditioning',
        'assess-discarding', '--raw-dir', 'J:\Raw Data\all3sTtrace',
        '--project-dir', $project, '--analysis-id', $assessmentId,
        '--experiment', 'all3sTrace', '--metric', $metric,
        '--metric-recipe', 'tail-candidate-corrected') + $fishArgs)
}

$slug = $metric.Replace('_', '-')
$figure2Dir = Join-Path $project "Figures\PNG\Analyses\$figure2Id"
$figure2e = Join-Path $figure2Dir "cohort-selected-block-ratio_${slug}_total-activity.png"
if (-not (Test-Path -LiteralPath $figure2e -PathType Leaf)) {
    Invoke-Python 'Figure 2E block ratios, 0–13 s' @('-m', 'classical_conditioning',
        'figure-cohort-selected-block-ratio', '--project-dir', $project,
        '--analysis-id', $figure2Id, '--cohort-id', $cohort,
        '--metric', $metric, '--outcome', 'total-activity')
}
$figure2h = Join-Path $figure2Dir "cohort-trial-ratio_${slug}_total-activity.png"
if (-not (Test-Path -LiteralPath $figure2h -PathType Leaf)) {
    Invoke-Python 'Figure 2H trial ratios, 0–13 s' @('-m', 'classical_conditioning',
        'figure-cohort-trial-ratio', '--project-dir', $project,
        '--analysis-id', $figure2Id, '--cohort-id', $cohort,
        '--metric', $metric, '--outcome', 'total-activity')
}

$cohortPath = Join-Path $project "Processed data\Cohorts\$cohort\cohort-manifest.parquet"
$outcomesPath = Join-Path $project "Processed data\Cohorts\$cohort\cohort-trial-outcomes.parquet"
$comparisonDir = Join-Path $project "Processed data\Analyses\$figure3Id\legacy-tail-l1"
if (-not (Test-Path -LiteralPath (Join-Path $comparisonDir 'comparison.json') -PathType Leaf)) {
    Invoke-Python 'Figure 3 historical learner comparison, 0–13 s' @('-m',
        'classical_conditioning', 'compare-legacy-learners', '--cohort', $cohortPath,
        '--trial-outcomes', $outcomesPath, '--output-dir', $comparisonDir,
        '--metric', $metric)
}
if (-not (Test-Path -LiteralPath (Join-Path $comparisonDir 'figures\figures.json') -PathType Leaf)) {
    Invoke-Python 'Figure 3 historical plots, 0–13 s' @('-m', 'classical_conditioning',
        'render-legacy-learner-figures', '--comparison-dir', $comparisonDir)
}
$figure3Dir = Join-Path $project "Figures\PNG\Analyses\$figure3Id"
if (-not (Test-Path -LiteralPath (Join-Path $figure3Dir "figure-3-3strace-review_${metric}.png") -PathType Leaf)) {
    Invoke-Python 'Figure 3 exploratory review, 0–13 s' @((Join-Path $repo 'scripts\render_figure3_3strace_review.py'),
        '--project-dir', $project, '--cohort-id', $cohort, '--comparison-dir', $comparisonDir,
        '--output-dir', $figure3Dir)
}

$labels = Join-Path $project "Processed data\Analyses\$figure4Id\learner-labels.csv"
if (-not (Test-Path -LiteralPath $labels -PathType Leaf)) {
    Invoke-Python '0–13 s provisional legacy-wip labels' @((Join-Path $repo 'scripts\finalize_3strace_exploratory.py'),
        'classify', '--project-dir', $project, '--cohort-id', $cohort,
        '--comparison-dir', $comparisonDir, '--analysis-id', $figure4Id,
        '--assessment-summary', $assessment,
        '--classifier-execution-id', 'legacy-wip-3strace-tail-l1-window13-exploratory')
}
$figure4Summary = Join-Path $project "Processed data\Analyses\$figure4Id\figure4\analysis.json"
if (-not (Test-Path -LiteralPath $figure4Summary -PathType Leaf)) {
    Invoke-Python 'Figure 4B learner profiles, 0–13 s labels' @('-m', 'classical_conditioning',
        'figure4-analyze', '--project-dir', $project, '--analysis-id', $figure4Id,
        '--metric', $metric, '--trace3-cohort-id', $cohort, '--learner-manifest', $labels)
}
$figure4Dir = Join-Path $project "Figures\PNG\Analyses\$figure4Id"
$figure4Kinds = @('figure-4', 'supplement-single-catches', 'supplement-movement',
    'supplement-coverage', 'supplement-single-catch-movement',
    'supplement-single-catch-coverage')
$figure4Missing = @($figure4Kinds | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $figure4Dir "all3sTrace\${_}_${metric}.png") -PathType Leaf)
})
if ($figure4Missing.Count -gt 0) {
    Invoke-Python 'Figure 4B renders, 0–13 s labels' @('-m', 'classical_conditioning',
        'figure4-render', '--analysis-summary', $figure4Summary,
        '--output-dir', $figure4Dir, '--overwrite')
}

$learningMarker = Join-Path $project "Metadata\${learningId}_learning-onset_complete.json"
if (-not (Test-Path -LiteralPath $learningMarker -PathType Leaf)) {
    Invoke-Python '3sTrace learning onset, 0–13 s' @('-m', 'classical_conditioning',
        'learning-onset', '--project-dir', $project, '--cohort-id', $cohort,
        '--analysis-id', $learningId, '--metric', $metric, '--outcome', 'total-activity',
        '--test-condition', 'trace', '--delta-min', '0', '--bootstrap', '499',
        '--permutations', '9999')
}
$learningDir = Join-Path $project "Figures\PNG\Analyses\$learningId"
if (-not (Test-Path -LiteralPath (Join-Path $learningDir 'learning-diagnostics.png') -PathType Leaf)) {
    Invoke-Python '3sTrace learning diagnostics, 0–13 s' @('-m', 'classical_conditioning',
        'figure-learning-diagnostics', '--project-dir', $project,
        '--analysis-id', $learningId, '--mode', 'static')
}
if (-not (Test-Path -LiteralPath (Join-Path $learningDir 'learning-onset.png') -PathType Leaf)) {
    Invoke-Python '3sTrace learning-onset review, 0–13 s' @('-m', 'classical_conditioning',
        'figure-learning-onset', '--project-dir', $project,
        '--analysis-id', $learningId, '--mode', 'static', '--allow-unaccepted')
}
Write-Output "Completed the 0–13 s 3sTrace dependent analysis on F:"
