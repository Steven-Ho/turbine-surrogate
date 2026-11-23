#!/bin/bash
# All four configurations share one node allocation and run sequentially.
set -euo pipefail
mkdir -p logs
sbatch --export=ALL "$@" scripts/slurm_fann_geo_scaling.sh
