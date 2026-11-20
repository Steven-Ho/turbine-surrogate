#!/bin/bash
#SBATCH --job-name=gpu_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --get-user-env
#SBATCH --output=logs/gpu_probe_%j.out
#SBATCH --error=logs/gpu_probe_%j.err

# Submit from the project root. Extra arguments are passed to probe_gpu.py.
set -eo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Submit using sbatch scripts/slurm_gpu_probe.sh from the project root." >&2
  exit 2
fi
module purge
module load "${PROBE_MODULE:-conda-py31010/3.10.10}"
# Platform Python already provides vendor PyTorch. Override if needed.
set -u
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
exec srun --ntasks=1 "${PROBE_PYTHON:-python}" scripts/probe_gpu.py "$@"
