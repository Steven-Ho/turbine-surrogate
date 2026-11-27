#!/bin/bash
# Two nodes x eight DDP ranks; one complete epoch at global batch <=16.
set -euo pipefail
mkdir -p logs
export CR15_RUN_ROOT="${CR15_TWO_NODE_ROOT:-runs/cr15_geo_base_2nodes_1epoch_$(date +%Y%m%d_%H%M%S)}"
export CR15_SCALE=base CR15_AMP=bf16 CR15_BATCH_CASES=16
export CR15_EPOCHS=400 CR15_EPOCH_BUDGET=1 CR15_LARGEST_FIRST=0
export CR15_GPUS_PER_NODE=8 CR15_RESUME=0 CR15_LEARNING_RATE=0.0001
export CR15_LR_PATIENCE=20 CR15_LR_THRESHOLD=0.001 CR15_LR_COOLDOWN=5
unset CR15_MAX_UPDATES
exec sbatch --exclusive --nodes=2 --ntasks-per-node=1 --gres=gpu:8 \
 --cpus-per-task=32 --mem=64G --time=02:00:00 --export=ALL \
 --job-name=cr15_2n1ep --output=logs/cr15_2n1ep_%j.out --error=logs/cr15_2n1ep_%j.err \
 "$@" scripts/slurm_cr15_train.sh
