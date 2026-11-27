#!/bin/bash
set -euo pipefail
mkdir -p logs
exec sbatch scripts/slurm_cr15_prepare.sh "$@"
