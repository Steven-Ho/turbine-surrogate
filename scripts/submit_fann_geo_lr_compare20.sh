#!/bin/bash
# Both jobs can run concurrently when twelve exclusive nodes are available.
set -euo pipefail
python_bin="${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/python"
if [[ ! -x "$python_bin" ]]; then python_bin=python3; fi
exec "$python_bin" scripts/submit_fann_geo_lr_compare20.py "$@"
