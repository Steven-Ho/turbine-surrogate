#!/bin/bash
#SBATCH --job-name=fann_models
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=fann_%A_%a.out
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TURBINE_GEO_QUERY_BACKEND=kdtree
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/fann_neighbor_cache}"
: "${FANN_SPLITS:?Set FANN_SPLITS to prepared splits.json}"
exec srun --ntasks=1 python scripts/run_fann_split.py --splits "$FANN_SPLITS" --index "$SLURM_ARRAY_TASK_ID" --steps "${FANN_STEPS:-20000}" --output "runs/fann_${SLURM_ARRAY_JOB_ID}"
