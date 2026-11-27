#!/bin/bash
# Fresh LR=2e-4 control: 400 epochs, exclusive two nodes x eight GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export CR15_RUN_ROOT="${CR15_LR2_RUN_ROOT:-runs/cr15_geo_base_lr2_2nodes_400_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ -e "$CR15_RUN_ROOT/GeoTransolver/last.pt" || -e "$CR15_RUN_ROOT/GeoTransolver/best.pt" ]]; then
  echo "Refusing to overwrite an existing experiment: $CR15_RUN_ROOT" >&2
  exit 1
fi
export CR15_PREPARED="${CR15_PREPARED:-$PWD/runs/cr15_prepared_v1}"
export CR15_MODEL=GeoTransolver CR15_SCALE=base CR15_AMP=bf16
export CR15_BATCH_CASES=16 CR15_GPUS_PER_NODE=8
export CR15_EPOCHS=400 CR15_EPOCH_BUDGET=0 CR15_LARGEST_FIRST=0
export CR15_RESUME=0 CR15_CAPACITY_CHECK=0 CR15_LEARNING_RATE=0.0002
export CR15_LR_PATIENCE=20 CR15_LR_THRESHOLD=0.001 CR15_LR_COOLDOWN=5
unset CR15_MAX_UPDATES CUDA_VISIBLE_DEVICES
printf 'CR15 fresh LR=2e-4: %s; target=400 epochs; exclusive 2 nodes x 8 GPUs; global batch=16\n' "$CR15_RUN_ROOT"
exec sbatch --exclusive --nodes=2 --ntasks-per-node=1 --gres=gpu:8 \
 --cpus-per-task=32 --mem=64G --time=04:00:00 --export=ALL \
 --job-name=cr15_geo_lr2_long --output=logs/cr15_geo_lr2_long_%j.out \
 --error=logs/cr15_geo_lr2_long_%j.err "$@" scripts/slurm_cr15_train.sh
