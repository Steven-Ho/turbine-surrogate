#!/bin/bash
# Queue the five-epoch experiment only if the capacity probe exits successfully.
set -euo pipefail
capacity=$(bash scripts/submit_cr15_capacity.sh --parsable "$@")
capacity_id="${capacity%%;*}"
if [[ ! "$capacity_id" =~ ^[0-9]+$ ]]; then echo "Unexpected capacity job ID: $capacity" >&2; exit 2; fi
short=$(bash scripts/submit_cr15_short.sh --parsable --dependency="afterok:$capacity_id" "$@")
printf 'CR15 capacity job: %s\nCR15 short training job: %s (after capacity succeeds)\n' "$capacity" "$short"
