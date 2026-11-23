#!/bin/bash
# Geometry preferred sampler plus bounded CPU neighbor caches; isolated experiment.
set -euo pipefail
export FANN_BASELINE_RUN_ROOT="${FANN_AFFINITY_RUN_ROOT:-runs/fann_geo_affinity_2nodes_$(date +%Y%m%d_%H%M%S)}"
export FANN_DDP_SAMPLER=geometry_affinity
export TURBINE_GEO_MEMORY_CACHE_MB="${FANN_AFFINITY_CACHE_MB:-4096}"
exec bash scripts/submit_fann_geo_no_offload.sh "$@"
