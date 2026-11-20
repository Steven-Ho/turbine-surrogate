#!/bin/bash
# Run from the project root: bash scripts/submit_fann_geo_amp.sh
set -euo pipefail
mkdir -p logs
export FANN_SPLITS="${FANN_SPLITS:-runs/fann_prepared_20261119_202344/splits.json}"
export FANN_RUN_ROOT="${FANN_RUN_ROOT:-runs/fann_geo_bf16_2nodes_$(date +%Y%m%d_%H%M%S)}"
export FANN_AMP=bf16
export FANN_EPOCHS="${FANN_EPOCHS:-400}"
export FANN_LOG_EVERY="${FANN_LOG_EVERY:-1}"
export FANN_DDP_GPUS_PER_NODE=8
export FANN_THREADS="${FANN_THREADS:-4}"
# Fresh run for comparison; precision changes cannot resume FP32 trajectories.
export FANN_RESUME=0
printf 'GeoTransolver BF16 AMP: 2 nodes x 8 GPUs; output=%s\n' "$FANN_RUN_ROOT"
sbatch --array=6 --nodes=2 --ntasks-per-node=1 --gres=gpu:8 \
  --cpus-per-task=32 --mem=768G --signal=B:USR1@900 --export=ALL \
  "$@" scripts/slurm_fann_ddp.sh
