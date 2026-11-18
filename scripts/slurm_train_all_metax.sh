#!/bin/bash
#SBATCH --job-name=turbine_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-6%7
#SBATCH --get-user-env
#SBATCH --output=logs/turbine_long_%A_%a.out
#SBATCH --error=logs/turbine_long_%A_%a.err

set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export TFAI_PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# GPU identifiers may be remapped by the site runtime. Do not derive a
# CUDA_VISIBLE_DEVICES mask from SLURM_JOB_GPUS/SLURM_STEP_GPUS here.
echo "node=$(hostname) job=${SLURM_JOB_ID:-unknown} array_task=${SLURM_ARRAY_TASK_ID:-unknown}"
echo "batch CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>} SLURM_STEP_GPUS=${SLURM_STEP_GPUS-<unset>}"
models=(POD UNet GCN Graph_UNet Graph_Transformer Transolver_3 GeoTransolver)
index="${SLURM_ARRAY_TASK_ID:?Use sbatch array submission}"
if (( index < 0 || index >= ${#models[@]} )); then exit 2; fi
run_root="${TURBINE_LONG_RUN_ROOT:-runs/metax_long_${SLURM_ARRAY_JOB_ID}}"
echo "model=${models[$index]} results=$run_root/${models[$index]}"
# Let Slurm create a one-GPU job step and supply its device environment.
runner_args=(--model "${models[$index]}" --run-root "$run_root")
exec srun --ntasks=1 --gres=gpu:1 python scripts/run_cluster_long.py "${runner_args[@]}"
