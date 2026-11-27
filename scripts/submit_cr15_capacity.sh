#!/bin/bash
# Two largest TRAIN cases, one full case/update; no validation/test access.
set -euo pipefail
mkdir -p logs
export CR15_RUN_ROOT="${CR15_CAPACITY_ROOT:-runs/cr15_geo_base_capacity_$(date +%Y%m%d_%H%M%S)}"
export CR15_SCALE=base CR15_AMP=bf16 CR15_BATCH_CASES=1
export CR15_EPOCHS=400 CR15_EPOCH_BUDGET=0 CR15_MAX_UPDATES=2 CR15_LARGEST_FIRST=1
export CR15_GPUS_PER_NODE=1 CR15_RESUME=0
exec sbatch --nodes=1 --gres=gpu:1 --time=01:00:00 --export=ALL \
 --job-name=cr15_capacity --output=logs/cr15_capacity_%j.out --error=logs/cr15_capacity_%j.err \
 "$@" scripts/slurm_cr15_train.sh
