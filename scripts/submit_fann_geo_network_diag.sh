#!/bin/bash
# Stage timings plus NCCL/MCCL transport logs; no network routing overrides.
set -euo pipefail
mkdir -p logs
export FANN_SCALING_DIAGNOSTIC=1
export FANN_NETWORK_DIAG_STEPS=5
export FANN_SCALING_WARMUP_CASES=16 FANN_SCALING_MEASURE_CASES=64
export FANN_SCALING_2NODE_OUTPUT="${FANN_NETWORK_DIAG_OUTPUT:-runs/fann_geo_network_diag_$(date +%Y%m%d_%H%M%S)}"
unset FANN_SCALING_REFERENCE_REPORT
sbatch --export=ALL --job-name=fann_geo_network_diag \
 --output=logs/fann_geo_network_diag_%j.out --error=logs/fann_geo_network_diag_%j.err \
 "$@" scripts/slurm_fann_geo_scaling_2nodes.sh
