#!/bin/bash
# Short, fresh two-node BF16 baseline: only CPU activation offload is changed.
set -euo pipefail
export FANN_RUN_ROOT="${FANN_BASELINE_RUN_ROOT:-runs/fann_geo_bf16_no_offload_2nodes_$(date +%Y%m%d_%H%M%S)}"
export TURBINE_GEO_ACTIVATION_OFFLOAD=0
export TURBINE_GEO_NODE_CHUNK=32768
export FANN_SCALE=small
# Synchronized diagnostics alter throughput. Disable both timing switches.
export FANN_PROFILE_STEPS=0
export TURBINE_FANN_PROFILE_STEPS=0
export FANN_LOG_EVERY=1
# Explicit empty FANN_BASELINE_MAX_UPDATES means full training to the epoch limit.
export FANN_MAX_UPDATES="${FANN_BASELINE_MAX_UPDATES-20}"
export FANN_EPOCHS="${FANN_EPOCHS:-400}"
printf 'No-offload baseline: BF16, 2 nodes x 8 GPUs, chunk=32768, max_updates=%s, output=%s\n' "${FANN_MAX_UPDATES:-unlimited}" "$FANN_RUN_ROOT"
exec bash scripts/submit_fann_geo_amp.sh "$@"
