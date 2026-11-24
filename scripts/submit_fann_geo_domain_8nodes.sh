#!/bin/bash
# 64 physical ranks = 16 case replicas x 4 within-node domain ranks.
# Run the single-node domain check before allocating eight nodes.
set -euo pipefail
export FANN_STABILITY_NODES=8 FANN_STABILITY_DOMAIN_SIZE=4
export FANN_STABILITY_EPOCHS="${FANN_DOMAIN_EPOCHS:-5}" FANN_STABILITY_RESUME=0
export FANN_STABILITY_LEARNING_RATE=0.0001 FANN_STABILITY_BETA1=0.9 FANN_STABILITY_BETA2=0.999
export FANN_STABILITY_RUN_ROOT="${FANN_DOMAIN_RUN_ROOT:-runs/fann_geo_domain4_8nodes_$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/submit_fann_geo_stability.sh \
 --job-name=fann_geo_dp16dom4 \
 --output=logs/fann_geo_dp16dom4_%A_%a.out --error=logs/fann_geo_dp16dom4_%A_%a.err "$@"
