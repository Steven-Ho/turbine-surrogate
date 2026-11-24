#!/bin/bash
#SBATCH --job-name=fann_allreduce
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/fann_allreduce_%j.out
#SBATCH --error=logs/fann_allreduce_%j.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 FANN_INTEROP_THREADS=1
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
output="${FANN_ALLREDUCE_OUTPUT:-runs/fann_allreduce_${SLURM_JOB_ID}}"
for nodes in 1 2; do
 world=$((nodes*8));export AR_NODES="$nodes" AR_MASTER="${hosts[0]}" AR_PORT="$((15000 + SLURM_JOB_ID % 20000 + nodes))"
 case "$output" in
  /*) export FANN_COMM_DIAGNOSTICS_DIR="$output/gpus_$world/communication";;
  *) export FANN_COMM_DIAGNOSTICS_DIR="$PWD/$output/gpus_$world/communication";;
 esac
 extra=()
 if [[ "$nodes" == 1 ]]; then extra+=(--nodelist "${hosts[0]}"); fi
 srun --export=ALL --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 --kill-on-bad-exit=1 "${extra[@]}" \
  bash -c 'exec python -m torch.distributed.run --nnodes="$AR_NODES" --nproc_per_node=8 --node_rank="$SLURM_PROCID" --master_addr="$AR_MASTER" --master_port="$AR_PORT" scripts/benchmark_fann_allreduce.py "$@"' \
  fann-allreduce --output "$output/gpus_$world" --warmup 5 --iterations "${FANN_ALLREDUCE_ITERATIONS:-20}"
done
