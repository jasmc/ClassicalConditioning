param(
    [int]$PipelineProcessId = 0,
    [string]$MetricId = "tail_length_weighted_angular_l1"
)

$ErrorActionPreference = "Stop"

$uv = "C:\Users\joaquim\.local\bin\uv.exe"
$project = "F:\Digested Data\allDelay-full-v1"
$inventoryPath = Join-Path $project "Metadata\recording_inventory.json"
$configPath = Join-Path $PSScriptRoot "..\configs\allDelay-full-windows.json"
$cohortPath = Join-Path $PSScriptRoot "..\configs\allDelay-full-v1-cohort.csv"
$cohortId = "allDelay-full-v1"
$analysisId = "allDelay-full-learning-onset-v1"
$runnerManifest = Join-Path $project "Metadata\allDelay-full-v1-candidate_candidate-corrected-runner-v1_manifest.json"

function Test-FullCandidateRun {
    if (-not (Test-Path -LiteralPath $runnerManifest -PathType Leaf)) {
        return $false
    }
    $manifest = Get-Content -LiteralPath $runnerManifest -Raw | ConvertFrom-Json
    return ($manifest.status -eq "complete" -and @($manifest.recording_ids).Count -eq 57)
}

if ($PipelineProcessId -gt 0) {
    $pipelineProcess = Get-Process -Id $PipelineProcessId -ErrorAction SilentlyContinue
    if ($null -ne $pipelineProcess) {
        Wait-Process -Id $PipelineProcessId
    }
}

for ($attempt = 1; $attempt -le 4 -and -not (Test-FullCandidateRun); $attempt++) {
    Write-Output "Resuming candidate pipeline (attempt $attempt of 4)."
    & $uv run classical-conditioning run-pipeline --config $configPath
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Pipeline attempt $attempt exited with code $LASTEXITCODE."
    }
}

if (-not (Test-FullCandidateRun)) {
    throw "Candidate pipeline did not produce a verified 57-recording manifest."
}

if (-not (Test-Path -LiteralPath $inventoryPath -PathType Leaf)) {
    throw "Required source inventory is missing: $inventoryPath"
}

if (-not (Test-Path -LiteralPath $cohortPath -PathType Leaf)) {
    $inventory = Get-Content -LiteralPath $inventoryPath -Raw | ConvertFrom-Json
    $records = @($inventory.records | Where-Object {
        $_.status -eq "COMPLETE" -and $_.condition_id -in @("control", "delay")
    } | Sort-Object recording_id)
    if ($records.Count -ne 57) {
        throw "Expected 57 complete control/delay recordings, found $($records.Count)."
    }
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

& $uv run classical-conditioning freeze-cohort --project-dir $project --input $cohortPath --cohort-id $cohortId --policy-id all-complete-triplets-v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $uv run classical-conditioning build-cohort-trial-outcomes --project-dir $project --cohort-id $cohortId --metric-recipe tail-candidate-corrected-v1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $uv run classical-conditioning learning-onset --project-dir $project --cohort-id $cohortId --analysis-id $analysisId --metric $MetricId --outcome total-activity --test-condition delay --delta-min 0 --bootstrap 499 --permutations 9999
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $uv run classical-conditioning figure-learning-diagnostics --project-dir $project --analysis-id $analysisId --mode static
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $uv run classical-conditioning figure-learning-onset --project-dir $project --analysis-id $analysisId --mode static
exit $LASTEXITCODE
