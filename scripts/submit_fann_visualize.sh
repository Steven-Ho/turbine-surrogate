#!/bin/bash
set -euo pipefail
mkdir -p logs
exec sbatch scripts/slurm_fann_visualize.sh "$@"
