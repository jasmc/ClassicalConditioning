#!/usr/bin/env bash
# Run or resume the complete 10-fish refactor smoke test from this repository.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="/Volumes/JOAQUIM/Digested Data/allDelay-refactor-smoke-10fish"
cohort_id="allDelay-refactor-smoke-10fish-v1"
analysis_id="allDelay-refactor-smoke-learning-onset-v1"
python_bin="$root_dir/.venv/bin/python"

cd "$root_dir"
"$python_bin" -m classical_conditioning run-pipeline --config configs/allDelay-refactor-smoke-10fish.json

for fish_id in \
  20240607_01 20240607_02 20240610_07 20240610_08 20240612_07 \
  20240612_08 20240613_01 20240613_02 20240618_01 20240618_02; do
  for figure_id in \
    total-activity-raw total-activity-scaled conditional-intensity-raw bout-outcomes; do
    "$python_bin" -m classical_conditioning figure-candidate-profiles \
      --project-dir "$project_dir" --recording-id "$fish_id" \
      --trial-type CS --figure "$figure_id" --mode static \
      --recipe candidate-temporal-outcomes-corrected-v3 --overwrite
  done
done

for outcome_id in \
  total-activity movement-probability fraction-time-moving conditional-intensity bout-rate; do
  "$python_bin" -m classical_conditioning figure-metric-comparison \
    --project-dir "$project_dir" \
    --analysis-id allDelay-refactor-smoke-10fish-v1-candidate \
    --trial-type CS --outcome "$outcome_id" \
    --recipe candidate-metric-comparison-corrected-v1 \
    --mode static --overwrite
done

if [[ ! -f "$project_dir/Metadata/${cohort_id}_cohort-manifest-v1_complete.json" ]]; then
  "$python_bin" -m classical_conditioning freeze-cohort \
    --project-dir "$project_dir" \
    --input configs/allDelay-refactor-smoke-10fish-cohort.csv \
    --cohort-id "$cohort_id" \
    --policy-id technical-refactor-smoke-v1
fi

"$python_bin" -m classical_conditioning build-cohort-trial-outcomes \
  --project-dir "$project_dir" --cohort-id "$cohort_id" \
  --metric-recipe tail-candidate-corrected-v1 --overwrite
"$python_bin" -m classical_conditioning learning-onset \
  --project-dir "$project_dir" --cohort-id "$cohort_id" \
  --analysis-id "$analysis_id" --metric tail_length_weighted_angular_l1 \
  --outcome total-activity --test-condition delay --delta-min 0 \
  --bootstrap 199 --permutations 999 --overwrite
"$python_bin" -m classical_conditioning figure-learning-onset \
  --project-dir "$project_dir" --analysis-id "$analysis_id" --mode static --overwrite
"$python_bin" -m classical_conditioning figure-learning-diagnostics \
  --project-dir "$project_dir" --analysis-id "$analysis_id" --mode static --overwrite
