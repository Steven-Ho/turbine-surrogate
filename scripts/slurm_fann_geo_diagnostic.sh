#!/bin/bash
#SBATCH --job-name=fann_geo_diag
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --output=logs/fann_geo_diag_%j.out
#SBATCH --error=logs/fann_geo_diag_%j.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/fann_defaults.sh
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${FANN_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TURBINE_FANN_PIPELINE_GPUS=1
export TURBINE_GEO_QUERY_BACKEND=kdtree TURBINE_GEO_REFERENCE_MAX_POINTS=0
export TURBINE_GEO_NODE_CHUNK="${TURBINE_GEO_NODE_CHUNK:-32768}"
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/fann_neighbor_cache}"
mapfile -t diag_hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${diag_hosts[0]}" MASTER_PORT="$((15000 + SLURM_JOB_ID % 30000))"
export FANN_DIAG_GPUS_PER_NODE="${FANN_DIAG_GPUS_PER_NODE:-8}"
output="${FANN_DIAG_OUTPUT:-runs/fann_geo_diagnostic_${SLURM_JOB_ID}}"
extra=()
if [[ -n "${FANN_DIAG_CASE_B:-}" ]]; then extra+=(--case-b "$FANN_DIAG_CASE_B"); fi
if [[ -n "${FANN_DIAG_CHECKPOINT:-}" ]]; then extra+=(--checkpoint "$FANN_DIAG_CHECKPOINT"); fi
printf 'Geo fixed-weight diagnostic nodes=%s GPUs/node=%s offload=%s output=%s\n' "$SLURM_NNODES" "$FANN_DIAG_GPUS_PER_NODE" "${FANN_DIAG_OFFLOAD:-on}" "$output"
run_probe() {
 local probe_offload="$1" probe_output="$2"
 srun --export=ALL --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
 bash -c 'exec python -m torch.distributed.run --nnodes="$SLURM_NNODES" --nproc_per_node="$FANN_DIAG_GPUS_PER_NODE" --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" scripts/diagnose_fann_geo.py "$@"' \
 fann-geo-diag --splits "$FANN_SPLITS" --output "$probe_output" --amp "${FANN_AMP:-bf16}" \
 --threads "$OMP_NUM_THREADS" --scale "${FANN_SCALE:-small}" --case-a "${FANN_DIAG_CASE_A:-0}" \
 --offload "$probe_offload" --swaps "${FANN_DIAG_SWAPS:-2}" "${extra[@]}"

}
if [[ "${FANN_DIAG_COMPARE_OFFLOAD:-0}" == 1 ]]; then
 run_probe on "$output/offload_on"
 run_probe off "$output/offload_off"
else
 run_probe "${FANN_DIAG_OFFLOAD:-on}" "$output"
fi
