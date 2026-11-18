#!/bin/bash
# Run on the login node from the project root: bash scripts/submit_cluster_long.sh
set -euo pipefail
: "${TURBINE_LONG_RUN_ROOT:=runs/metax_long_$(date +%Y%m%d_%H%M%S)_$$}"
export TURBINE_LONG_RUN_ROOT
# Slurm opens output/error before the job script starts.
mkdir -p logs
array_job="$(sbatch --parsable --export=ALL scripts/slurm_train_all_metax.sh)"
array_id="${array_job%%;*}"
if ! [[ "$array_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected sbatch result: $array_job" >&2
  exit 1
fi
printf 'Training array: %s\nResults: %s\n' "$array_id" "$TURBINE_LONG_RUN_ROOT"
if ! summary_job="$(sbatch --parsable --export=ALL --dependency="afterany:$array_id" scripts/slurm_compare_metax.sh)"; then
  echo "Training array is submitted, but automatic summary submission failed." >&2
  echo "After the array finishes, export TURBINE_LONG_RUN_ROOT=$TURBINE_LONG_RUN_ROOT and submit scripts/slurm_compare_metax.sh." >&2
  exit 1
fi
printf 'Summary job: %s\n' "$summary_job"
