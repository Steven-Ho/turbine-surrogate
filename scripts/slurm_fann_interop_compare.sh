#!/bin/bash
#SBATCH --job-name=fann_interop
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --output=logs/fann_interop_%j.out
#SBATCH --error=logs/fann_interop_%j.err
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
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
output="${FANN_INTEROP_OUTPUT:-runs/fann_interop_${SLURM_JOB_ID}}"
port_offset=0
for mode in default 1; do
 if [[ "$mode" == default ]]; then unset FANN_INTEROP_THREADS; else export FANN_INTEROP_THREADS=1; fi
 python scripts/run_fann_geo_scaling.py --splits "$FANN_SPLITS" --output "$output/interop_$mode" \
  --nodes 2 --gpu-counts 16 --master-addr "${hosts[0]}" --master-port "$((15000 + SLURM_JOB_ID % 20000 + port_offset))" \
  --warmup-cases 16 --measure-cases 64 --profile-steps 5 --comm-diagnostics
 port_offset=$((port_offset+1))
done
