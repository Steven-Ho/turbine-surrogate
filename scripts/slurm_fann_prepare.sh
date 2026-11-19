#!/bin/bash
#SBATCH --job-name=fann_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=fann_prepare_%j.out
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/prepare_fann_cluster.py --source "${FANN_SOURCE:-$HOME/Dataset/faNN}" --output "${FANN_PREPARED:-$PWD/runs/fann_prepared_${SLURM_JOB_ID}}" --cells "${FANN_CELLS:-10000}" --selection "${FANN_DOMAIN:-full}"
