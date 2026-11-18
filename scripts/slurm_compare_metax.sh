#!/bin/bash
#SBATCH --job-name=turbine_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --get-user-env
#SBATCH --output=logs/turbine_compare_%j.out
#SBATCH --error=logs/turbine_compare_%j.err

set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export TFAI_PROJECT_ROOT="$PWD" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec srun --ntasks=1 --gres=gpu:1 python scripts/compare_cluster_long.py --run-root "${TURBINE_LONG_RUN_ROOT:?Set the run root used by the training array}"
