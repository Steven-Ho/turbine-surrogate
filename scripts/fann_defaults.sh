#!/bin/bash
# Source from the project root. An explicit FANN_SPLITS overrides this default.
# An explicitly empty value requests fresh preparation in submit_fann_formal.sh.
export FANN_SPLITS="${FANN_SPLITS-$PWD/runs/fann_prepared_20261119_202344/splits.json}"
