#!/bin/bash
# Fresh 20-epoch Adam beta comparison against the lr=1.7e-4 control.
set -euo pipefail
export FANN_STABILITY_NODES=6 FANN_STABILITY_EPOCHS=20 FANN_STABILITY_RESUME=0
export FANN_STABILITY_LEARNING_RATE=0.00017
export FANN_STABILITY_BETA1=0.7 FANN_STABILITY_BETA2=0.997
export FANN_STABILITY_RUN_ROOT="${FANN_SIX_NODE_BETA_RUN_ROOT:-runs/fann_geo_stability_6nodes_beta_$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/submit_fann_geo_stability.sh \
 --job-name=fann_geo_beta48 \
 --output=logs/fann_geo_beta48_%A_%a.out --error=logs/fann_geo_beta48_%A_%a.err "$@"
