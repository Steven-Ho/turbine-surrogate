#!/bin/bash
set -euo pipefail
mkdir -p logs
sbatch --export=ALL "$@" scripts/slurm_fann_geo_scaling_2nodes.sh
