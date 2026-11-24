#!/bin/bash
# Compare against the six-node 1e-4 control; change only the learning rate.
set -euo pipefail
export FANN_STABILITY_NODES=6 FANN_STABILITY_EPOCHS=5
export FANN_STABILITY_LEARNING_RATE=0.00017
export FANN_STABILITY_RUN_ROOT="${FANN_SIX_NODE_LR17_RUN_ROOT:-runs/fann_geo_stability_6nodes_lr17_$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/submit_fann_geo_stability.sh \
 --job-name=fann_geo_lr17_48 \
 --output=logs/fann_geo_lr17_48_%A_%a.out --error=logs/fann_geo_lr17_48_%A_%a.err "$@"
