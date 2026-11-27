#!/bin/bash
# No PyTorch/accelerator runtime import on the login node.
set -euo pipefail
exec python3 scripts/submit_cr15_2nodes_long.py "$@"
