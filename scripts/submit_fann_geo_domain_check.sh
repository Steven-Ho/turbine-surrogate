#!/bin/bash
set -euo pipefail
mkdir -p logs
exec sbatch --exclusive --export=ALL "$@" scripts/slurm_fann_geo_domain_check.sh
