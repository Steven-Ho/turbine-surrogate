#!/bin/bash
#SBATCH --job-name=fann_models
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --signal=TERM@120
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
# Never launch dense cdist reference kernels on MetaX, including small tie cases.
export TURBINE_GEO_REFERENCE_MAX_POINTS="${TURBINE_GEO_REFERENCE_MAX_POINTS:-0}"
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/fann_neighbor_cache}"
: "${FANN_SPLITS:?Set FANN_SPLITS to prepared splits.json}"
if [[ -n "${FANN_STEPS:-}" ]]; then
  echo "FANN_STEPS belongs to the old baseline; unset it and use FANN_EPOCHS." >&2
  exit 2
fi
resume_args=()
if [[ "${FANN_RESUME:-0}" == 1 ]]; then resume_args+=(--resume); fi
exec srun --ntasks=1 python scripts/run_fann_split.py --splits "$FANN_SPLITS" --index "$SLURM_ARRAY_TASK_ID" --epochs "${FANN_EPOCHS:-400}" --domain "${FANN_DOMAIN:-full}" --grid "${FANN_GRID:-32}" --rank "${FANN_POD_RANK:-32}" --output "${FANN_RUN_ROOT:-runs/fann_formal_${SLURM_ARRAY_JOB_ID}}" "${resume_args[@]}"
