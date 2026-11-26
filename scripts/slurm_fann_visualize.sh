#!/bin/bash
#SBATCH --job-name=fann_visualize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fann_visualize_%j.out
#SBATCH --error=logs/fann_visualize_%j.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
export CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fann_mpl_${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$MPLCONFIGDIR"
exec python scripts/summarize_fann_run.py "$@"
