#!/bin/bash
#SBATCH --job-name=fann_model_parallel
#SBATCH --array=2,3,4,6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --signal=TERM@120
#SBATCH --output=fann_%A_%a.out
# One process owns multiple GPUs. Do not launch torchrun or one task per GPU.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export TURBINE_FANN_PIPELINE_GPUS="${TURBINE_FANN_PIPELINE_GPUS:-2}"
source scripts/slurm_fann_array.sh
