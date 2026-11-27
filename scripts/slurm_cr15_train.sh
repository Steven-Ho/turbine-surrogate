#!/bin/bash
#SBATCH --job-name=cr15_geo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/cr15_geo_%j.out
#SBATCH --error=logs/cr15_geo_%j.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TURBINE_GEO_QUERY_BACKEND=kdtree TURBINE_GEO_REFERENCE_MAX_POINTS=0
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/cr15_neighbor_cache}"
export TURBINE_GEO_MEMORY_CACHE_MB=512 TURBINE_GEO_MEMORY_CACHE=32
: "${CR15_RUN_ROOT:?Set CR15_RUN_ROOT to an independent experiment directory}"
export CR15_PREPARED="${CR15_PREPARED:-$PWD/runs/cr15_prepared_v1}"
mkdir -p "$CR15_RUN_ROOT" logs
stop_file="$CR15_RUN_ROOT/.stop_${SLURM_JOB_ID}"
trap 'touch "$stop_file"; wait_interrupted=1' USR1
extra=()
if [[ "${CR15_RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
if [[ -n "${CR15_MAX_UPDATES:-}" ]]; then extra+=(--max-updates "$CR15_MAX_UPDATES"); fi
if [[ "${CR15_LARGEST_FIRST:-0}" == 1 ]]; then extra+=(--largest-first); fi
if [[ "${SLURM_NNODES:-1}" -gt 1 ]]; then
  mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  export MASTER_ADDR="${hosts[0]}" MASTER_PORT="$((15000+SLURM_JOB_ID%30000))"
  launch=(srun --export=ALL --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1
    bash -c 'exec python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="${CR15_GPUS_PER_NODE:-1}" --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$@"' cr15)
elif [[ "${CR15_GPUS_PER_NODE:-1}" -gt 1 ]]; then
  launch=(srun --export=ALL --ntasks=1 --kill-on-bad-exit=1 python -m torch.distributed.run --standalone --nproc_per_node="$CR15_GPUS_PER_NODE")
else
  launch=(srun --export=ALL --ntasks=1 --kill-on-bad-exit=1 python)
fi
"${launch[@]}" scripts/train_cr15.py --prepared "$CR15_PREPARED" --output "$CR15_RUN_ROOT" \
 --scale "${CR15_SCALE:-base}" --amp "${CR15_AMP:-bf16}" --batch-cases "${CR15_BATCH_CASES:-16}" \
 --epochs "${CR15_EPOCHS:-400}" --epoch-budget "${CR15_EPOCH_BUDGET:-5}" --threads 4 \
 --learning-rate "${CR15_LEARNING_RATE:-0.0001}" --chunk 32768 \
 --lr-patience "${CR15_LR_PATIENCE:-20}" --lr-threshold "${CR15_LR_THRESHOLD:-0.001}" \
 --lr-cooldown "${CR15_LR_COOLDOWN:-5}" --log-every 1 --checkpoint-every 10 \
 --stop-file "$stop_file" "${extra[@]}" "$@" &
launcher=$!
set +e
while true; do
 wait_interrupted=0
 wait "$launcher"
 result=$?
 if ((wait_interrupted==0)); then break; fi
done
exit "$result"
