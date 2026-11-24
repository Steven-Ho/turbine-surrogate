#!/bin/bash
# Exclusive 48-GPU short run with the established stable settings.
set -euo pipefail
export FANN_STABILITY_NODES=6
export FANN_STABILITY_RUN_ROOT="${FANN_SIX_NODE_RUN_ROOT:-runs/fann_geo_stability_6nodes_$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/submit_fann_geo_stability.sh \
 --job-name=fann_geo_stability48 \
 --output=logs/fann_geo_stability48_%A_%a.out --error=logs/fann_geo_stability48_%A_%a.err "$@"
