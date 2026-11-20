#!/bin/bash
#SBATCH --job-name=fann_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/fann_cache_%j.out
#SBATCH --error=logs/fann_cache_%j.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/fann_defaults.sh
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TURBINE_FANN_TREE_WORKERS="${TURBINE_FANN_TREE_WORKERS:-$OMP_NUM_THREADS}"
export TURBINE_GEO_TREE_WORKERS="${TURBINE_GEO_TREE_WORKERS:-$OMP_NUM_THREADS}"
export TURBINE_GEO_QUERY_BACKEND=kdtree
export TURBINE_GEO_REFERENCE_MAX_POINTS="${TURBINE_GEO_REFERENCE_MAX_POINTS:-0}"
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/fann_neighbor_cache}"
export TURBINE_FANN_GRAPH_CACHE="${TURBINE_FANN_GRAPH_CACHE:-$PWD/runs/fann_graph_topology_cache_v1}"
export TURBINE_FANN_UNET_CACHE="${TURBINE_FANN_UNET_CACHE:-$PWD/runs/fann_unet_mapping_cache_v1}"
export TURBINE_FANN_POD_CACHE="${TURBINE_FANN_POD_CACHE:-$PWD/runs/fann_pod_cache_v1}"
extra=()
if [[ -n "${FANN_POD_CACHE_CHECKPOINT:-}" ]]; then extra+=(--pod-checkpoint "$FANN_POD_CACHE_CHECKPOINT"); fi
if [[ -n "${FANN_CACHE_MODELS:-}" ]]; then read -r -a selected_models <<< "$FANN_CACHE_MODELS"; extra+=(--models "${selected_models[@]}"); fi
if [[ -n "${FANN_CACHE_OWNER_NODES:-}" ]]; then read -r -a owners <<< "$FANN_CACHE_OWNER_NODES"; extra+=(--owner-nodes "${owners[@]}"); fi
printf 'faNN static cache preparation: splits=%s report=%s\n' "$FANN_SPLITS" "${FANN_CACHE_OUTPUT:-runs/fann_cache_precompute_${SLURM_JOB_ID}}"
exec python scripts/precompute_fann_caches.py --splits "$FANN_SPLITS" --device cuda \
  --output "${FANN_CACHE_OUTPUT:-runs/fann_cache_precompute_${SLURM_JOB_ID}}" --scale "${FANN_SCALE:-small}" \
  --grid "${FANN_GRID:-32}" --rank "${FANN_POD_RANK:-32}" --owner-nodes "${FANN_GRAPH_OWNER_NODES:-32768}" \
  --graph-edge-chunk "${FANN_GRAPH_EDGE_CHUNK:-262144}" --workers "${FANN_STARTUP_WORKERS:-4}" --threads "${SLURM_CPUS_PER_TASK:-8}" "${extra[@]}"
