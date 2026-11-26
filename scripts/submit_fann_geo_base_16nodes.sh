#!/bin/bash
# Base model, sixteen case replicas; 8 within-node GPUs per case.
set -euo pipefail
export FANN_STABILITY_NODES=16 FANN_STABILITY_DOMAIN_SIZE=8
export FANN_STABILITY_SCALE=base FANN_STABILITY_EPOCHS=5
export FANN_STABILITY_RESUME=0 FANN_STABILITY_LONG_RUN=0
export FANN_STABILITY_LEARNING_RATE=0.0001 FANN_STABILITY_BETA1=0.9 FANN_STABILITY_BETA2=0.999
export FANN_STABILITY_RUN_ROOT="${FANN_BASE_16_RUN_ROOT:-runs/fann_geo_base_16nodes_dom8_$(date +%Y%m%d_%H%M%S)}"
exec bash scripts/submit_fann_geo_stability.sh \
 --time=04:00:00 --job-name=fann_geo_base16 \
 --output=logs/fann_geo_base16_%A_%a.out --error=logs/fann_geo_base16_%A_%a.err "$@"
