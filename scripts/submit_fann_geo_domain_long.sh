#!/bin/bash
# Standard-library-only preflight; no Torch import on login nodes.
set -euo pipefail
exec python3 scripts/submit_fann_geo_domain_long.py "$@"
