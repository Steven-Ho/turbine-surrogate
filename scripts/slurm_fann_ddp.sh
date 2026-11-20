#!/bin/bash
#SBATCH --job-name=fann_ddp
#SBATCH --array=1,2,4,5
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/fann_ddp_%A_%a.out
#SBATCH --error=logs/fann_ddp_%A_%a.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/fann_defaults.sh
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${FANN_THREADS:-4}"
export TURBINE_FANN_PIPELINE_GPUS=1
export TURBINE_FANN_STARTUP_CACHE="${TURBINE_FANN_STARTUP_CACHE:-$PWD/runs/fann_startup_cache_v1}"
export FANN_STARTUP_WORKERS="${FANN_STARTUP_WORKERS:-1}"
export TURBINE_FANN_UNET_CACHE="${TURBINE_FANN_UNET_CACHE:-$PWD/runs/fann_unet_mapping_cache_v1}"
export TURBINE_FANN_TRANSOLVER_CHUNK_SIZE="${TURBINE_FANN_TRANSOLVER_CHUNK_SIZE:-32768}"
export TURBINE_FANN_TRANSOLVER_CHECKPOINT="${TURBINE_FANN_TRANSOLVER_CHECKPOINT:-1}"
export TURBINE_GEO_QUERY_BACKEND="${TURBINE_GEO_QUERY_BACKEND:-kdtree}"
export TURBINE_GEO_REFERENCE_MAX_POINTS="${TURBINE_GEO_REFERENCE_MAX_POINTS:-0}"
export TURBINE_GEO_ACTIVATION_OFFLOAD="${TURBINE_GEO_ACTIVATION_OFFLOAD:-1}"
export TURBINE_GEO_NODE_CHUNK="${TURBINE_GEO_NODE_CHUNK:-32768}"
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/fann_neighbor_cache}"
export TURBINE_FANN_GRAPH_CACHE="${TURBINE_FANN_GRAPH_CACHE:-$PWD/runs/fann_graph_topology_cache_v1}"
: "${FANN_SPLITS:?Set FANN_SPLITS to prepared splits.json}"
: "${FANN_RUN_ROOT:?Set FANN_RUN_ROOT to the result root}"
if [[ -n "${FANN_STEPS:-}" ]]; then echo 'Unset FANN_STEPS; use FANN_EPOCHS.' >&2; exit 2; fi
case "${SLURM_ARRAY_TASK_ID:?}" in
  1) model=UNet;;
  2) model=GCN;;
  3) model=Graph_UNet;;
  4) model=Graph_Transformer;;
  6) model=GeoTransolver;;
  5) model=Transolver_3;;
  *) echo 'DDP supports neural-model array indices 1-6' >&2; exit 2;;
esac
mapfile -t ddp_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${FANN_MASTER_ADDR:-${ddp_hosts[0]}}"
export MASTER_PORT="${FANN_MASTER_PORT:-$((15000 + ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} % 30000 + SLURM_ARRAY_TASK_ID))}"
export FANN_DDP_GPUS_PER_NODE="${FANN_DDP_GPUS_PER_NODE:-8}"
export FANN_DDP_STOP_FILE="$FANN_RUN_ROOT/.ddp_stop_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$FANN_RUN_ROOT" logs
wait_interrupted=0
trap 'touch "$FANN_DDP_STOP_FILE"; wait_interrupted=1' USR1
extra=()
if [[ "${FANN_RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
if [[ -n "${FANN_MAX_UPDATES:-}" ]]; then extra+=(--max-updates "$FANN_MAX_UPDATES"); fi
printf 'DDP model=%s nodes=%s GPUs/node=%s master=%s:%s\n' "$model" "$SLURM_NNODES" "$FANN_DDP_GPUS_PER_NODE" "$MASTER_ADDR" "$MASTER_PORT"
# One torchrun agent per node; each starts one worker per local GPU.
srun --export=ALL --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -c 'exec python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$FANN_DDP_GPUS_PER_NODE" --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" scripts/train_fann_ddp.py "$@"' \
  fann-ddp --models "$model" --splits "$FANN_SPLITS" --output "$FANN_RUN_ROOT" --domain "${FANN_DOMAIN:-full}" \
  --epochs "${FANN_EPOCHS:-400}" --threads "${FANN_THREADS:-4}" --grid "${FANN_GRID:-32}" --rank "${FANN_POD_RANK:-32}" \
  --backend "${FANN_DDP_BACKEND:-nccl}" --timeout-minutes "${FANN_DDP_TIMEOUT_MINUTES:-120}" --checkpoint-every "${FANN_DDP_CHECKPOINT_EVERY:-10}" \
  --graph-execution "${FANN_GRAPH_EXECUTION:-bounded}" --graph-owner-nodes "${FANN_GRAPH_OWNER_NODES:-32768}" \
  --graph-edge-chunk "${FANN_GRAPH_EDGE_CHUNK:-262144}" --graph-cache-mb "${FANN_GRAPH_CACHE_MB:-1024}" \
  --stop-file "$FANN_DDP_STOP_FILE" "${extra[@]}" &
launcher_pid=$!
set +e
while true; do
  wait_interrupted=0
  wait "$launcher_pid"
  launch_status=$?
  # A pre-walltime USR1 interrupts wait, but the workers poll the stop file
  # and save at a complete global-batch boundary before exiting normally.
  if ((wait_interrupted==0)); then break; fi
done
exit "$launch_status"
