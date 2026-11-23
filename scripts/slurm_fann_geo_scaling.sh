#!/bin/bash
#SBATCH --job-name=fann_geo_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --output=logs/fann_geo_scaling_%j.out
#SBATCH --error=logs/fann_geo_scaling_%j.err
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
output="${FANN_SCALING_OUTPUT:-runs/fann_geo_scaling_${SLURM_JOB_ID}}"
read -r -a counts <<< "${FANN_SCALING_COUNTS:-1 2 4 8}"
printf 'Single-node scaling: process_counts=%s output=%s\n' "${counts[*]}" "$output"
srun --export=ALL --ntasks=1 --kill-on-bad-exit=1 \
 python scripts/run_fann_geo_scaling.py --splits "$FANN_SPLITS" --output "$output" \
 --gpu-counts "${counts[@]}" --warmup-cases "${FANN_SCALING_WARMUP_CASES:-16}" --measure-cases "${FANN_SCALING_MEASURE_CASES:-64}"
