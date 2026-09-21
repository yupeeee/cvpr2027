#!/usr/bin/env bash
# Execute from the repository so relative config/artifact paths are consistent.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then
        cat <<'HELP'
Usage: ./run_pilot.sh [experiment arguments]

Run prepare, collect, fit, intervene, controls, diagnose, policy, and plot in order.
Defaults: --config configs/demo.json --device auto
Valid saved artifacts are always reused; missing or damaged outputs are repaired.
Use --overwrite to rerun the workflow and replace outputs for this run ID.
Arguments are forwarded unchanged to every stage and override these defaults.
Use --dry-run for configuration checks without models, network access, or CUDA.
Pretrained weights must be local/cached unless --no-local-files-only is supplied.
Use --device cpu for CPU execution or CUDA_VISIBLE_DEVICES to limit GPUs.
Progress combines all devices; controls uses one total and ETA for its whole stage.
Use --no-progress to hide bars or --progress-mininterval SECONDS to tune updates.
Set PYTHON to a Python executable path to select the environment.
Relative paths are resolved from this script's repository directory.

Examples:
  ./run_pilot.sh --run-id pilot01 --no-local-files-only
  ./run_pilot.sh --config configs/demo.json --device auto --dry-run
  PYTHON=/path/to/env/bin/python ./run_pilot.sh --run-id pilot01

List all experimental arguments: python -m exps.prepare --help
HELP
        exit 0
    fi
done

python_executable="${PYTHON:-python}"
args=(--config configs/demo.json --device auto "$@")
stages=(prepare collect fit intervene controls diagnose policy plot)
for index in "${!stages[@]}"; do
    stage="${stages[index]}"
    printf '\n[%d/%d] Running exps.%s\n' "$((index + 1))" "${#stages[@]}" "$stage"
    "$python_executable" -m "exps.$stage" "${args[@]}"
done
