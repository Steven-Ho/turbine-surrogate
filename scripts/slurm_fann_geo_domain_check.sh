#!/bin/bash
#SBATCH --job-name=fann_geo_domaincheck
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=768G
#SBATCH --output=logs/fann_geo_domaincheck_%j.out
#SBATCH --error=logs/fann_geo_domaincheck_%j.err
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/fann_defaults.sh
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TURBINE_GEO_QUERY_BACKEND=kdtree TURBINE_GEO_REFERENCE_MAX_POINTS=0
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/fann_neighbor_cache}"
export TURBINE_GEO_ACTIVATION_OFFLOAD=0 TURBINE_GEO_NODE_CHUNK=32768
export TURBINE_FANN_PIPELINE_GPUS=1
unset FANN_INTEROP_THREADS FANN_MAX_EPOCHS_PER_RUN FANN_MAX_UPDATES FANN_STEPS
unset FANN_COMM_DIAGNOSTICS_DIR FANN_RUNTIME_SNAPSHOT_DIR
export FANN_PROFILE_STEPS=0 TURBINE_FANN_PROFILE_STEPS=0
output="${FANN_DOMAIN_CHECK_ROOT:-runs/fann_geo_domaincheck_${SLURM_JOB_ID}}"
# All four GPUs cooperate on ONE small case, first FP32 then BF16. Stop on failure.
srun --export=ALL --ntasks=1 --kill-on-bad-exit=1 \
 python -m torch.distributed.run --standalone --nproc_per_node=4 \
 scripts/verify_fann_geo_domain.py --output "$output/capability"
# Then compare matching full-case/update budgets at global batch=1:
# one GPU per case versus four GPUs cooperating on that same case.
srun --export=ALL --ntasks=1 --kill-on-bad-exit=1 \
 python scripts/run_fann_geo_scaling.py --splits "$FANN_SPLITS" --output "$output/scaling" \
 --gpu-counts 1 4 --domain-sizes 1 4 \
 --warmup-cases "${FANN_DOMAIN_WARMUP_CASES:-2}" --measure-cases "${FANN_DOMAIN_MEASURE_CASES:-8}"
