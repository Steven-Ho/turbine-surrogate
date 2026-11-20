#!/bin/bash
# Submit preparation (if needed), seven model jobs, and an after-any collector.
set -euo pipefail
# Slurm opens output/error before the job script starts.
mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
export FANN_RUN_ROOT="${FANN_RUN_ROOT:-$PWD/runs/fann_formal_$stamp}"
export FANN_DOMAIN="${FANN_DOMAIN:-full}"
export FANN_EPOCHS="${FANN_EPOCHS:-400}"
if [[ -n "${FANN_STEPS:-}" ]]; then
  echo "Unset FANN_STEPS; the formal workflow uses FANN_EPOCHS." >&2; exit 2
fi
dependency=()
if [[ -z "${FANN_SPLITS:-}" ]]; then
  export FANN_PREPARED="${FANN_PREPARED:-$PWD/runs/fann_prepared_$stamp}"
  export FANN_SPLITS="$FANN_PREPARED/splits.json"
  preparation=$(sbatch --parsable scripts/slurm_fann_prepare.sh)
  preparation=${preparation%%;*}
  dependency=(--dependency="afterok:$preparation")
  echo "Preparation: $preparation"
elif [[ ! -f "$FANN_SPLITS" ]]; then
  echo "FANN_SPLITS does not exist: $FANN_SPLITS" >&2; exit 2
fi
startup=$(sbatch --parsable "${dependency[@]}" scripts/slurm_fann_startup.sh)
startup=${startup%%;*}
echo "Startup registration: $startup"
array=$(sbatch --parsable --dependency="afterok:$startup" scripts/slurm_fann_array.sh)
array=${array%%;*}
summary=$(sbatch --parsable --dependency="afterany:$array" scripts/slurm_fann_collect.sh)
printf 'Array: %s\nSummary: %s\nResults: %s\n' "$array" "$summary" "$FANN_RUN_ROOT"
