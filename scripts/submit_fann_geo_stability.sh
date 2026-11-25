#!/bin/bash
# Exclusive 16-GPU stability run; stop only after complete validation epochs.
set -euo pipefail
mkdir -p logs
nodes="${FANN_STABILITY_NODES:-2}"
if [[ ! "$nodes" =~ ^[1-9][0-9]*$ ]]; then echo 'FANN_STABILITY_NODES must be positive' >&2; exit 2; fi
export FANN_RUN_ROOT="${FANN_STABILITY_RUN_ROOT:-runs/fann_geo_stability_$(date +%Y%m%d_%H%M%S)}"
export FANN_MAX_EPOCHS_PER_RUN="${FANN_STABILITY_EPOCHS:-5}"
if [[ ! "$FANN_MAX_EPOCHS_PER_RUN" =~ ^[1-9][0-9]*$ ]]; then echo 'FANN_STABILITY_EPOCHS must be a positive integer' >&2; exit 2; fi
# A formal long run disables the per-launch short-run budget.
if [[ "${FANN_STABILITY_LONG_RUN:-0}" == 1 ]]; then export FANN_MAX_EPOCHS_PER_RUN=0; fi
# Keep the checkpoint's full training budget so this run can resume to 400 later.
export FANN_EPOCHS=400 FANN_RESUME="${FANN_STABILITY_RESUME:-0}" FANN_DDP_GPUS_PER_NODE=8
export FANN_DOMAIN_SIZE="${FANN_STABILITY_DOMAIN_SIZE:-1}"
export FANN_BETA1="${FANN_STABILITY_BETA1:-0.9}" FANN_BETA2="${FANN_STABILITY_BETA2:-0.999}"
export FANN_LEARNING_RATE="${FANN_STABILITY_LEARNING_RATE:-0.0001}"
export FANN_AMP=bf16 FANN_SCALE=small FANN_DDP_SAMPLER=random
export TURBINE_GEO_ACTIVATION_OFFLOAD=0 TURBINE_GEO_NODE_CHUNK=32768
export TURBINE_GEO_MEMORY_CACHE_MB=512 TURBINE_GEO_MEMORY_CACHE=32
export TURBINE_GEO_QUERY_BACKEND=kdtree TURBINE_GEO_REFERENCE_MAX_POINTS=0
export FANN_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# The already-fast 1973 default-interop setting is the control.
unset FANN_INTEROP_THREADS
export FANN_PROFILE_STEPS=0 TURBINE_FANN_PROFILE_STEPS=0 FANN_LOG_EVERY=1
export FANN_DDP_CHECKPOINT_EVERY=10
unset FANN_MAX_UPDATES FANN_STEPS FANN_COMM_DIAGNOSTICS_DIR
unset NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_DEBUG_FILE MCCL_DEBUG MCCL_DEBUG_SUBSYS MCCL_DEBUG_FILE
unset TORCH_DISTRIBUTED_DEBUG TORCH_CPP_LOG_LEVEL
# Resource-only snapshots do not enable library logs or per-step GPU sync.
export FANN_RUNTIME_SNAPSHOT_DIR="$FANN_RUN_ROOT/GeoTransolver/runtime"
printf 'Exclusive Geo stability: %s nodes x 8 GPUs, epoch_budget=%s (0=to total 400), output=%s\n' "$nodes" "$FANN_MAX_EPOCHS_PER_RUN" "$FANN_RUN_ROOT"
printf 'Learning rate: %s (resume=%s, no warmup)\n' "$FANN_LEARNING_RATE" "$FANN_RESUME"
sbatch --exclusive --array=6 --nodes="$nodes" --ntasks-per-node=1 --gres=gpu:8 \
 --cpus-per-task=32 --mem=768G --signal=B:USR1@900 --export=ALL \
 --job-name=fann_geo_stability --output=logs/fann_geo_stability_%A_%a.out --error=logs/fann_geo_stability_%A_%a.err \
 "$@" scripts/slurm_fann_ddp.sh
