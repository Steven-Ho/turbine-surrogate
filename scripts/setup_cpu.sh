#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3.11}" -m venv .venv
.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install torch-geometric==2.6.1 -r requirements-qiyuan.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/prepare_rotor37.py
