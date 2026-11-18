#!/bin/bash
#SBATCH --job-name=geo_optimized
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --get-user-env
#SBATCH --output=logs/geo_optimized_%j.out
#SBATCH --error=logs/geo_optimized_%j.err

set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export TFAI_PROJECT_ROOT="$PWD" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TURBINE_GEO_QUERY_BACKEND=kdtree
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/geo_neighbor_cache_kdtree_v1}"
: "${TURBINE_SOURCE_RUN:?Export the existing seven-model result directory}"
export TURBINE_LONG_RUN_ROOT="${TURBINE_LONG_RUN_ROOT:-runs/metax_geo_kdtree_${SLURM_JOB_ID}}"
if [ -e "$TURBINE_LONG_RUN_ROOT/GeoTransolver/status.json" ] || [ -d "$TURBINE_LONG_RUN_ROOT/GeoTransolver/train" ]; then
  echo "Existing Geo result detected. Unset TURBINE_LONG_RUN_ROOT or choose a new path." >&2
  exit 2
fi
mkdir -p "$TURBINE_LONG_RUN_ROOT"
python - <<'PYTHON_METADATA'
import os,json,hashlib
from pathlib import Path
root=Path(os.environ['TFAI_PROJECT_ROOT'])
files=['tfai/models/_vendor/geotransolver2/ball_query.py','tfai/models/_vendor/geotransolver2/spatial_neighbors.py']
value={'query_backend':os.environ['TURBINE_GEO_QUERY_BACKEND'],'cache_directory':os.environ['TURBINE_GEO_CACHE_DIR'],
       'source_run':os.environ['TURBINE_SOURCE_RUN'],'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}}
(Path(os.environ['TURBINE_LONG_RUN_ROOT'])/'geo_query_backend.json').write_text(json.dumps(value,indent=2)+'\n')
PYTHON_METADATA
# Cache construction is a separate process and timer; it never updates weights.
if [ "${GEO_OPT_PRECOMPUTE:-1}" = 1 ]; then
  srun --ntasks=1 --gres=gpu:1 python scripts/precompute_geo_neighbors.py \
    --source-run "$TURBINE_SOURCE_RUN" --record "$TURBINE_LONG_RUN_ROOT/neighbor_precompute.json"
fi
printf 'GeoTransolver backend=kdtree cache=%s results=%s\n' "$TURBINE_GEO_CACHE_DIR" "$TURBINE_LONG_RUN_ROOT"
exec srun --ntasks=1 --gres=gpu:1 python scripts/run_cluster_long.py \
  --model GeoTransolver --run-root "$TURBINE_LONG_RUN_ROOT"
