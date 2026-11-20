#!/bin/bash
#SBATCH --job-name=fann_models
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --signal=TERM@120
#SBATCH --output=logs/fann_%A_%a.out
#SBATCH --error=logs/fann_%A_%a.err
set -eo pipefail
printf 'faNN job=%s task=%s host=%s started=%s\n' "${SLURM_JOB_ID:-unknown}" "${SLURM_ARRAY_TASK_ID:-unknown}" "$(hostname)" "$(date -Is)"
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TURBINE_FANN_TRANSOLVER_CHUNK_SIZE="${TURBINE_FANN_TRANSOLVER_CHUNK_SIZE:-8192}"
export TURBINE_FANN_TRANSOLVER_CHECKPOINT="${TURBINE_FANN_TRANSOLVER_CHECKPOINT:-1}"
export TURBINE_FANN_UNET_CACHE="${TURBINE_FANN_UNET_CACHE:-$PWD/runs/fann_unet_mapping_cache_v1}"
export TURBINE_GEO_QUERY_BACKEND=kdtree
export TURBINE_GEO_ACTIVATION_OFFLOAD="${TURBINE_GEO_ACTIVATION_OFFLOAD:-1}"
export TURBINE_GEO_NODE_CHUNK="${TURBINE_GEO_NODE_CHUNK:-32768}"
export TURBINE_FANN_PIPELINE_GPUS="${TURBINE_FANN_PIPELINE_GPUS:-1}"
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
if [[ -n "${FANN_MAX_UPDATES:-}" ]]; then resume_args+=(--max-updates "$FANN_MAX_UPDATES"); fi
printf 'faNN launching: splits=%s run_root=%s pipeline_gpus=%s\n' "$FANN_SPLITS" "${FANN_RUN_ROOT:-runs/fann_formal_${SLURM_ARRAY_JOB_ID}}" "$TURBINE_FANN_PIPELINE_GPUS"
exec srun --ntasks=1 python scripts/run_fann_split.py --splits "$FANN_SPLITS" --index "$SLURM_ARRAY_TASK_ID" --epochs "${FANN_EPOCHS:-400}" --domain "${FANN_DOMAIN:-full}" --graph-execution "${FANN_GRAPH_EXECUTION:-bounded}" --graph-owner-nodes "${FANN_GRAPH_OWNER_NODES:-32768}" --graph-edge-chunk "${FANN_GRAPH_EDGE_CHUNK:-262144}" --grid "${FANN_GRID:-32}" --rank "${FANN_POD_RANK:-32}" --output "${FANN_RUN_ROOT:-runs/fann_formal_${SLURM_ARRAY_JOB_ID}}" "${resume_args[@]}"
