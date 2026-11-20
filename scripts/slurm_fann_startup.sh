#!/bin/bash
#SBATCH --job-name=fann_startup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/fann_startup_%j.out
#SBATCH --error=logs/fann_startup_%j.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${FANN_SPLITS:?Set FANN_SPLITS to the prepared splits.json}"
extra=()
if [[ "${FANN_STARTUP_FORCE:-0}" == 1 ]]; then
  extra+=(--force)
elif [[ -n "${FANN_STARTUP_FROM_CHECKPOINT:-}" ]]; then
  extra+=(--from-checkpoint "$FANN_STARTUP_FROM_CHECKPOINT")
fi
exec python scripts/prepare_fann_startup.py --splits "$FANN_SPLITS" --workers "${FANN_STARTUP_WORKERS:-4}" "${extra[@]}"
