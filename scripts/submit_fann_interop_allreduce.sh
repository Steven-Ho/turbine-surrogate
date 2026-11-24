#!/bin/bash
# Two independent exclusive two-node allocations; submit without dependency.
set -euo pipefail
mkdir -p logs
root="${FANN_CHECKS_ROOT:-runs/fann_interop_allreduce_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$root")";mkdir "$root"
export FANN_INTEROP_OUTPUT="$root/interop"
job=$(sbatch --parsable --export=ALL --exclusive "$@" scripts/slurm_fann_interop_compare.sh);job=${job%%;*}
printf 'interop\t%s\n' "$job" >> "$root/jobs.tsv"
export FANN_ALLREDUCE_OUTPUT="$root/allreduce"
job=$(sbatch --parsable --export=ALL --exclusive "$@" scripts/slurm_fann_allreduce.sh);job=${job%%;*}
printf 'allreduce\t%s\n' "$job" >> "$root/jobs.tsv"
printf 'Independent checks submitted: %s\n' "$root"
cat "$root/jobs.tsv"
