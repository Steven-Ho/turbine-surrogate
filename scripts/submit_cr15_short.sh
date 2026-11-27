#!/bin/bash
# Single GPU, accumulate sixteen cases/update, five complete train/val epochs.
set -euo pipefail
mkdir -p logs
export CR15_RUN_ROOT="${CR15_SHORT_ROOT:-runs/cr15_geo_base_short_$(date +%Y%m%d_%H%M%S)}"
export CR15_SCALE=base CR15_AMP=bf16 CR15_BATCH_CASES=16
export CR15_EPOCHS=400 CR15_EPOCH_BUDGET=5 CR15_LARGEST_FIRST=0
export CR15_GPUS_PER_NODE=1 CR15_RESUME=0
unset CR15_MAX_UPDATES
exec sbatch --nodes=1 --gres=gpu:1 --time=04:00:00 --export=ALL \
 --job-name=cr15_short --output=logs/cr15_short_%j.out --error=logs/cr15_short_%j.err \
 "$@" scripts/slurm_cr15_train.sh
