<#
.SYNOPSIS
Runs the resumable allDelay technical workflow from immutable raw files through LME analysis.

.DESCRIPTION
This is the single-command Windows entry point for the all-complete technical
analysis. It writes only below ProjectDir, preserves every completed artifact,
and can be invoked again after an interruption. The generated cohort is
explicitly labelled as a technical all-complete cohort, not a publication cohort.
The learning figure is rendered only when required diagnostics pass; residual
diagnostics are rendered first.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$RawDir,

    [Parameter(Mandatory = $true)]
    [string]$ProjectDir,

    [string]$UvPath = "uv",
    [string]$AnalysisId = "allDelay-full",
    [string]$CohortId = "allDelay-full",
    [string]$LearningAnalysisId = "allDelay-full-learning-onset",
    [string]$MetricId = "tail_length_weighted_angular_l1"
)

$ErrorActionPreference = "Stop"

$raw = [IO.Path]::GetFullPath($RawDir)
$project = [IO.Path]::GetFullPath($ProjectDir)
$metadata = Join-Path $project "Metadata"
$inventoryPath = Join-Path $metadata "recording_inventory.json"
$environmentPath = Join-Path $metadata "environment.json"
$configPath = Join-Path $metadata "allDelay-full-pipeline-config.json"
$cohortPath = Join-Path $metadata "${CohortId}-cohort-review.csv"
$runnerManifest = Join-Path $metadata "${AnalysisId}-candidate_candidate-corrected-runner_manifest.json"
$cohortMarker = Join-Path $metadata "${CohortId}_cohort-manifest_complete.json"
$outcomesMarker = Join-Path $metadata "${CohortId}_cohort-trial-outcomes_complete.json"
$lmeMarker = Join-Path $metadata "${LearningAnalysisId}_learning-onset_complete.json"
$lmeSummary = Join-Path $project "Quality checks\Analyses\$LearningAnalysisId\learning-onset_summary.json"
$figureDirectory = Join-Path $project "Figures\PNG\Analyses\$LearningAnalysisId"
$learningFigure = Join-Path $figureDirectory "learning-onset.png"
$diagnosticsFigure = Join-Path $figureDirectory "learning-diagnostics.png"

function Invoke-WorkflowCommand {
    param([string[]]$Arguments)
    & $UvPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $UvPath $($Arguments -join ' ')"
    }
}

function Test-FullCandidateRun {
    if (-not (Test-Path -LiteralPath $runnerManifest -PathType Leaf)) {
        return $false
    }
    $manifest = Get-Content -LiteralPath $runnerManifest -Raw | ConvertFrom-Json
    return ($manifest.status -eq "complete" -and @($manifest.recording_ids).Count -eq 57)
}

New-Item -ItemType Directory -Path $metadata -Force | Out-Null

if (-not (Test-Path -LiteralPath $environmentPath -PathType Leaf)) {
    Invoke-WorkflowCommand @("run", "classical-conditioning", "environment-report", "--output", $environmentPath)
}

if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
    Invoke-WorkflowCommand @("run", "classical-conditioning", "inventory", "--input-dir", $raw, "--output", $inventoryPath, "--inspect-tracking-headers")
}

$inventory = Get-Content -LiteralPath $inventoryPath -Raw | ConvertFrom-Json
if (-not $inventory.source_hashes_included) {
    throw "Inventory does not include source SHA-256 hashes: $inventoryPath"
}
$records = @($inventory.records | Where-Object {
    $_.status -eq "COMPLETE" -and $_.condition_id -in @("control", "delay")
} | Sort-Object recording_id)
if ($records.Count -ne 57) {
    throw "Expected 57 complete control/delay recordings, found $($records.Count)."
}

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    [ordered]@{
        raw_dir = $raw
        save_dir = $project
        experiment = "allDelay"
        analysis_id = $AnalysisId
        keep_conditions = @("control", "delay")
        overwrite = $false
        continue_on_error = $true
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $configPath -Encoding utf8
}

if (-not (Test-FullCandidateRun)) {
    Invoke-WorkflowCommand @("run", "classical-conditioning", "run-pipeline", "--config", $configPath)
}
if (-not (Test-FullCandidateRun)) {
    throw "Candidate pipeline did not produce a verified 57-recording manifest."
}

if (-not (Test-Path -LiteralPath $cohortMarker -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $cohortPath -PathType Leaf)) {
        $reviewedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
        $rows = foreach ($record in $records) {
            [PSCustomObject]@{
                experiment_id = "allDelay"
                recording_id = $record.recording_id
                fish_id = $record.recording_id
                condition_id = $record.condition_id
                technical_valid = $true
                technical_exclusion_reason = ""
                behavioral_engagement = ""
                behavioral_engagement_reason = ""
                primary_included = $true
                sensitivity_population_ids = "[]"
                review_status = "approved"
                reviewer = "Codex-technical-all-complete"
                reviewed_at = $reviewedAt
                source_qc_artifact_id = "Metadata/$($record.recording_id)_source_manifest.json"
            }
        }
        $rows | Export-Csv -LiteralPath $cohortPath -NoTypeInformation -Encoding utf8
    }
    Invoke-WorkflowCommand @("run", "classical-conditioning", "freeze-cohort", "--project-dir", $project, "--input", $cohortPath, "--cohort-id", $CohortId, "--policy-id", "all-complete-triplets")
}

if (-not (Test-Path -LiteralPath $outcomesMarker -PathType Leaf)) {
    Invoke-WorkflowCommand @("run", "classical-conditioning", "build-cohort-trial-outcomes", "--project-dir", $project, "--cohort-id", $CohortId, "--metric-recipe", "tail-candidate-corrected")
}

if (Test-Path -LiteralPath $lmeMarker -PathType Leaf) {
    if (-not (Test-Path -LiteralPath $lmeSummary -PathType Leaf)) {
        throw "Learning analysis marker exists without its summary: $lmeSummary"
    }
    $savedAnalysis = Get-Content -LiteralPath $lmeSummary -Raw | ConvertFrom-Json
    if ($savedAnalysis.config.metric_id -ne $MetricId) {
        throw "Learning analysis '$LearningAnalysisId' uses metric '$($savedAnalysis.config.metric_id)', but '$MetricId' was requested. Choose a new LearningAnalysisId for the new metric."
    }
}

if (-not (Test-Path -LiteralPath $lmeMarker -PathType Leaf)) {
    Invoke-WorkflowCommand @("run", "classical-conditioning", "learning-onset", "--project-dir", $project, "--cohort-id", $CohortId, "--analysis-id", $LearningAnalysisId, "--metric", $MetricId, "--outcome", "total-activity", "--test-condition", "delay", "--delta-min", "0", "--bootstrap", "499", "--permutations", "9999")
}

if (-not (Test-Path -LiteralPath $diagnosticsFigure -PathType Leaf)) {
    Invoke-WorkflowCommand @("run", "classical-conditioning", "figure-learning-diagnostics", "--project-dir", $project, "--analysis-id", $LearningAnalysisId, "--mode", "static")
}
if (-not (Test-Path -LiteralPath $learningFigure -PathType Leaf)) {
    Invoke-WorkflowCommand @("run", "classical-conditioning", "figure-learning-onset", "--project-dir", $project, "--analysis-id", $LearningAnalysisId, "--mode", "static", "--allow-unaccepted")
}

Write-Output "Learning-onset figure: $learningFigure"
Write-Output "Learning diagnostics figure: $diagnosticsFigure"
