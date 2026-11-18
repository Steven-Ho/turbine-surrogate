#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS
# Run after loading the cluster module and activating pde-fm-metax.
python - <<'PY'
import sys, platform
assert sys.version_info[:2] == (3, 10), sys.version
assert platform.machine() == 'x86_64', platform.machine()
import torch, torchvision
print('Vendor stack:', torch.__version__, torch.__file__, torchvision.__version__)
PY
# Install the complete frozen non-vendor set without dependency resolution.
# This guarantees pip cannot download/replace vendor GPU packages.
python -m pip install --no-index --no-deps wheelhouse/*.whl
python - <<'PY'
import hydra, h5py, numpy, timm, the_well, wandb
import torch
print('Offline dependencies imported; torch:', torch.__version__, torch.__file__)
PY
