#!/bin/bash
# One-node throughput control for the two-node no-offload baseline.
set -euo pipefail
mkdir -p logs
export FANN_SPLITS="${FANN_SPLITS:-runs/fann_prepared_20261119_202344/splits.json}"
export FANN_RUN_ROOT="${FANN_SINGLE_NODE_RUN_ROOT:-runs/fann_geo_bf16_no_offload_1node_$(date +%Y%m%d_%H%M%S)}"
export FANN_DDP_GPUS_PER_NODE=8
export FANN_DDP_SAMPLER=random
export FANN_AMP=bf16 FANN_SCALE=small
export TURBINE_GEO_ACTIVATION_OFFLOAD=0
export TURBINE_GEO_NODE_CHUNK=32768
export TURBINE_GEO_MEMORY_CACHE_MB=512
export TURBINE_GEO_MEMORY_CACHE=32
export TURBINE_GEO_QUERY_BACKEND=kdtree TURBINE_GEO_REFERENCE_MAX_POINTS=0
export FANN_PROFILE_STEPS=0 TURBINE_FANN_PROFILE_STEPS=0
export FANN_LOG_EVERY=1 FANN_THREADS=4 FANN_RESUME=0
export FANN_EPOCHS=400 FANN_DDP_CHECKPOINT_EVERY=10
# 250 cases: 32 batches/epoch; 40 updates process 250+64=314 cases,
# matching the earlier 16-GPU, 20-update baseline's case count.
export FANN_MAX_UPDATES="${FANN_SINGLE_NODE_MAX_UPDATES:-40}"
printf 'Single-node Geo control: 1 node x 8 GPUs, BF16, offload=off, sampler=random, cache=512MiB/module, max_updates=%s, output=%s\n' "$FANN_MAX_UPDATES" "$FANN_RUN_ROOT"
sbatch --array=6 --nodes=1 --ntasks-per-node=1 --gres=gpu:8 \
  --cpus-per-task=32 --mem=768G --signal=B:USR1@900 --export=ALL \
  "$@" scripts/slurm_fann_ddp.sh
