#!/bin/bash
# Resume base/16 nodes from the completed five-epoch short run.
# No Torch or accelerator runtime is imported on the login node.
set -euo pipefail
exec python3 scripts/submit_fann_geo_domain_long.py --preset base16 "$@"
