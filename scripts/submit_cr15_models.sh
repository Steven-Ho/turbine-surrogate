#!/bin/bash
# No PyTorch import or accelerator runtime is required on the login node.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/submit_cr15_models.py "$@"
