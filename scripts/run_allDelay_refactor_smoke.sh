#!/usr/bin/env bash
# Run an allDelay smoke configuration through the same mandatory routine pipeline.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s <run-config.json>\n' "$0" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="$1"
exec "$root_dir/.venv/bin/python" -m classical_conditioning run-pipeline --config "$config_path"
