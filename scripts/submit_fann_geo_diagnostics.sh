#!/bin/bash
# Submit sequential diagnostics, avoiding interference between comparisons.
set -euo pipefail
mkdir -p logs
suite="${FANN_DIAG_ROOT:-runs/fann_geo_diag_suite_$(date +%Y%m%d_%H%M%S)}"
export FANN_DIAG_OUTPUT="$suite/two_nodes_offload_on"
export FANN_DIAG_GPUS_PER_NODE=8 FANN_DIAG_OFFLOAD=on FANN_DIAG_SWAPS=2 FANN_DIAG_COMPARE_OFFLOAD=0
first=$(sbatch --parsable --export=ALL "$@" scripts/slurm_fann_geo_diagnostic.sh)
first=${first%%;*}
export FANN_DIAG_OUTPUT="$suite/single_gpu"
export FANN_DIAG_GPUS_PER_NODE=1 FANN_DIAG_OFFLOAD=on FANN_DIAG_SWAPS=1 FANN_DIAG_COMPARE_OFFLOAD=1
second=$(sbatch --parsable --export=ALL --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=128G --dependency="afterok:$first" "$@" scripts/slurm_fann_geo_diagnostic.sh)
second=${second%%;*}
printf 'Diagnostic suite: %s\n2-node exchange: %s\nSingle GPU offload on/off (same allocation): %s\n' "$suite" "$first" "$second"
