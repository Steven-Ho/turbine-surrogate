#!/bin/bash
#SBATCH --job-name=fann_ddp_heavy
#SBATCH --array=3,6
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/fann_ddp_heavy_%A_%a.out
#SBATCH --error=logs/fann_ddp_heavy_%A_%a.err
# CPU-offloaded models: 8 ranks/node on the confirmed ~1TiB RAM nodes.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export FANN_DDP_GPUS_PER_NODE="${FANN_HEAVY_GPUS_PER_NODE:-8}"
export FANN_DDP_CHECKPOINT_EVERY="${FANN_DDP_CHECKPOINT_EVERY:-1}"
source scripts/slurm_fann_ddp.sh
