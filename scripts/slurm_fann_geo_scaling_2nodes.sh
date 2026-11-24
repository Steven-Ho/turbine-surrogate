#!/bin/bash
#SBATCH --job-name=fann_geo_scaling16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --output=logs/fann_geo_scaling16_%j.out
#SBATCH --error=logs/fann_geo_scaling16_%j.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/fann_defaults.sh
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/fann_neighbor_cache}"
mapfile -t scaling_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
output="${FANN_SCALING_2NODE_OUTPUT:-runs/fann_geo_scaling16_${SLURM_JOB_ID}}"
extra=()
if [[ -n "${FANN_SCALING_REFERENCE_REPORT:-}" ]]; then extra+=(--reference-report "$FANN_SCALING_REFERENCE_REPORT"); fi
printf 'Two-node controlled scaling: nodes=2 GPUs/node=8, output=%s\n' "$output"
# The controller runs in the batch shell; its child srun launches one agent/node.
# Do not wrap the controller in a one-task srun, which would hold task resources.
python scripts/run_fann_geo_scaling.py --splits "$FANN_SPLITS" --output "$output" \
 --nodes 2 --gpu-counts 16 --master-addr "${scaling_hosts[0]}" \
 --master-port "$((15000 + SLURM_JOB_ID % 30000))" \
 --warmup-cases "${FANN_SCALING_WARMUP_CASES:-16}" \
 --measure-cases "${FANN_SCALING_MEASURE_CASES:-64}" "${extra[@]}"
