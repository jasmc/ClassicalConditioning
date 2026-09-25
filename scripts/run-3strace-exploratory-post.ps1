<#
.SYNOPSIS
Completes the exploratory cohort, Figure 1/2, learner, and Figure 4 stages.

.DESCRIPTION
Run after all3sTrace-full_pipeline_run.json reports complete. Every output is
under the one fixed 3sTrace F: project. Existing completed stage markers allow
continuation after interruption without changing frozen identities.
#>
param(
    [string]$ProjectDir = 'F:\Digested Data\all3sTrace-full-v1',
    [string]$PythonPath = 'C:\Users\joaquim\Documents\ClassicalConditioning\.venv-trace\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$project = [System.IO.Path]::GetFullPath($ProjectDir).TrimEnd('\')
$expected = [System.IO.Path]::GetFullPath('F:\Digested Data\all3sTrace-full-v1').TrimEnd('\')
if (-not $project.Equals($expected, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "This review runner is restricted to the F: 3sTrace project: $project"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python missing: $PythonPath" }
$cohortId = 'all3sTrace-full-exploratory'
$analysisId = 'all3sTrace-full'
$metric = 'tail_length_weighted_angular_l1'
$pipeline = Join-Path $project 'Metadata\all3sTrace-full_pipeline_run.json'
$state = Get-Content -LiteralPath $pipeline -Raw | ConvertFrom-Json
if ($state.status -ne 'complete' -or @($state.recording_ids).Count -ne 59 -or
    @(Compare-Object @($state.recording_ids) @($state.active_recording_ids)).Count -ne 0) {
    throw 'The complete 59-fish candidate pipeline must finish before downstream review.'
}

function Invoke-Python {
    param([string]$Stage, [string[]]$Arguments)
    Write-Output "Starting $Stage"
    & $PythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Stage failed with exit code $LASTEXITCODE" }
    Write-Output "Completed $Stage"
}

$cohortMarker = Join-Path $project "Metadata\${cohortId}_cohort-manifest_complete.json"
if (-not (Test-Path -LiteralPath $cohortMarker -PathType Leaf)) {
    Invoke-Python 'exploratory cohort freeze' @((Join-Path $repo 'scripts\finalize_3strace_exploratory.py'),
        'freeze', '--project-dir', $project, '--cohort-id', $cohortId)
}
$outcomesMarker = Join-Path $project "Metadata\${cohortId}_cohort-trial-outcomes_complete.json"
if (-not (Test-Path -LiteralPath $outcomesMarker -PathType Leaf)) {
    Invoke-Python 'cohort trial outcomes' @('-m', 'classical_conditioning', 'build-cohort-trial-outcomes',
        '--project-dir', $project, '--cohort-id', $cohortId, '--metric-recipe', 'tail-candidate-corrected')
}

$figure1 = Join-Path $project "Figures\PNG\20230307_12\figure-1-F-3strace_${metric}.png"
if (-not (Test-Path -LiteralPath $figure1 -PathType Leaf)) {
    Invoke-Python 'Figure 1F example' @((Join-Path $repo 'scripts\render_figure1_3strace_example.py'),
        '--project-dir', $project, '--recording-id', '20230307_12', '--metric', $metric)
}
$figure2Dir = Join-Path $project 'Figures\PNG\Analyses\figure2-3strace-review'
$figure2 = Join-Path $figure2Dir "figure-2B-3strace_signed_${metric}.png"
$figure2Coverage = Join-Path $figure2Dir "figure-2B-3strace_coverage_${metric}.png"
if (-not (Test-Path -LiteralPath $figure2 -PathType Leaf) -or
    -not (Test-Path -LiteralPath $figure2Coverage -PathType Leaf)) {
    Invoke-Python 'Figure 2B signed profile' @((Join-Path $repo 'scripts\render_figure2_3strace_signed_review.py'),
        '--project-dir', $project, '--cohort-id', $cohortId, '--metric', $metric,
        '--overwrite')
}
$figureAnalysis = 'figure2-3strace-exploratory'
$slug = $metric.Replace('_', '-')
$figure2e = Join-Path $project "Figures\PNG\Analyses\$figureAnalysis\cohort-selected-block-ratio_${slug}_total-activity.png"
if (-not (Test-Path -LiteralPath $figure2e -PathType Leaf)) {
    Invoke-Python 'Figure 2E block ratios' @('-m', 'classical_conditioning', 'figure-cohort-selected-block-ratio',
        '--project-dir', $project, '--analysis-id', $figureAnalysis, '--cohort-id', $cohortId,
        '--metric', $metric, '--outcome', 'total-activity')
}
$figure2h = Join-Path $project "Figures\PNG\Analyses\$figureAnalysis\cohort-trial-ratio_${slug}_total-activity.png"
if (-not (Test-Path -LiteralPath $figure2h -PathType Leaf)) {
    Invoke-Python 'Figure 2H trial ratios' @('-m', 'classical_conditioning', 'figure-cohort-trial-ratio',
        '--project-dir', $project, '--analysis-id', $figureAnalysis, '--cohort-id', $cohortId,
        '--metric', $metric, '--outcome', 'total-activity')
}

$comparisonDir = Join-Path $project 'Processed data\Analyses\figure3-3strace-exploratory\legacy-tail-l1'
$comparisonJson = Join-Path $comparisonDir 'comparison.json'
if (-not (Test-Path -LiteralPath $comparisonJson -PathType Leaf)) {
    $cohortPath = Join-Path $project "Processed data\Cohorts\$cohortId\cohort-manifest.parquet"
    $outcomesPath = Join-Path $project "Processed data\Cohorts\$cohortId\cohort-trial-outcomes.parquet"
    Invoke-Python 'Figure 3 historical learner comparison' @('-m', 'classical_conditioning',
        'compare-legacy-learners', '--cohort', $cohortPath, '--trial-outcomes', $outcomesPath,
        '--output-dir', $comparisonDir, '--metric', $metric)
}
$legacyFigures = Join-Path $comparisonDir 'figures\figures.json'
if (-not (Test-Path -LiteralPath $legacyFigures -PathType Leaf)) {
    Invoke-Python 'Figure 3 historical learner plots' @('-m', 'classical_conditioning',
        'render-legacy-learner-figures', '--comparison-dir', $comparisonDir)
}
$figure3 = Join-Path $project "Figures\PNG\Analyses\figure3-3strace-exploratory\figure-3-3strace-review_${metric}.png"
if (-not (Test-Path -LiteralPath $figure3 -PathType Leaf)) {
    Invoke-Python 'Figure 3 exploratory review' @((Join-Path $repo 'scripts\render_figure3_3strace_review.py'),
        '--project-dir', $project, '--cohort-id', $cohortId, '--comparison-dir', $comparisonDir)
}

$labels = Join-Path $project 'Processed data\Analyses\figure4-3strace-exploratory\learner-labels.csv'
if (-not (Test-Path -LiteralPath $labels -PathType Leaf)) {
    Invoke-Python 'provisional legacy-wip labels' @((Join-Path $repo 'scripts\finalize_3strace_exploratory.py'),
        'classify', '--project-dir', $project, '--comparison-dir', $comparisonDir,
        '--cohort-id', $cohortId)
}
$figure4Summary = Join-Path $project 'Processed data\Analyses\figure4-3strace-exploratory\figure4\analysis.json'
if (-not (Test-Path -LiteralPath $figure4Summary -PathType Leaf)) {
    Invoke-Python 'Figure 4B learner profiles' @('-m', 'classical_conditioning', 'figure4-analyze',
        '--project-dir', $project, '--analysis-id', 'figure4-3strace-exploratory',
        '--metric', $metric, '--trace3-cohort-id', $cohortId, '--learner-manifest', $labels)
}
$figure4Dir = Join-Path $project 'Figures\PNG\Analyses\figure4-3strace-exploratory'
$figure4Expected = @('figure-4', 'supplement-single-catches', 'supplement-movement',
    'supplement-coverage', 'supplement-single-catch-movement', 'supplement-single-catch-coverage')
$figure4Missing = @($figure4Expected | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $figure4Dir "all3sTrace\${_}_${metric}.png") -PathType Leaf)
})
if ($figure4Missing.Count -gt 0) {
    Invoke-Python 'Figure 4B and companion renders' @('-m', 'classical_conditioning', 'figure4-render',
        '--analysis-summary', $figure4Summary, '--output-dir', $figure4Dir, '--overwrite')
}

$learningId = 'all3sTrace-full-learning-onset'
$learningMarker = Join-Path $project "Metadata\${learningId}_learning-onset_complete.json"
if (-not (Test-Path -LiteralPath $learningMarker -PathType Leaf)) {
    Invoke-Python '3sTrace learning onset' @('-m', 'classical_conditioning', 'learning-onset',
        '--project-dir', $project, '--cohort-id', $cohortId, '--analysis-id', $learningId,
        '--metric', $metric, '--outcome', 'total-activity', '--test-condition', 'trace',
        '--delta-min', '0', '--bootstrap', '499', '--permutations', '9999')
}
$diagnostics = Join-Path $project "Figures\PNG\Analyses\$learningId\learning-diagnostics.png"
if (-not (Test-Path -LiteralPath $diagnostics -PathType Leaf)) {
    Invoke-Python '3sTrace learning diagnostics' @('-m', 'classical_conditioning',
        'figure-learning-diagnostics', '--project-dir', $project, '--analysis-id', $learningId,
        '--mode', 'static')
}
$learningFigure = Join-Path $project "Figures\PNG\Analyses\$learningId\learning-onset.png"
if (-not (Test-Path -LiteralPath $learningFigure -PathType Leaf)) {
    Invoke-Python '3sTrace learning-onset review' @('-m', 'classical_conditioning',
        'figure-learning-onset', '--project-dir', $project, '--analysis-id', $learningId,
        '--mode', 'static', '--allow-unaccepted')
}
Write-Output "Completed exploratory 3sTrace downstream review in $project"
