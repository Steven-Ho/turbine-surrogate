#!/bin/bash
# Independent exclusive nodes: do not run these groups concurrently on one node.
set -euo pipefail
mkdir -p logs
suite="${FANN_SCALING_PARALLEL_ROOT:-runs/fann_geo_scaling_parallel_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$suite")"
mkdir "$suite"
for count in 1 2 4 8; do
 export FANN_SCALING_COUNTS="$count"
 export FANN_SCALING_OUTPUT="$suite/parts/${count}gpu"
 job=$(sbatch --parsable --export=ALL --exclusive --nodes=1 --gres=gpu:8 \
   --cpus-per-task=32 --mem=768G --job-name="fann_geo_${count}gpu" \
   "$@" scripts/slurm_fann_geo_scaling.sh)
 job=${job%%;*}
 printf '%s\t%s\n' "$count" "$job" >> "$suite/jobs.tsv"
 printf '%s-GPU test submitted: job=%s\n' "$count" "$job"
done
printf 'Parallel suite: %s\nAfter completion: python3 scripts/summarize_fann_geo_scaling_parallel.py --root "%s"\n' "$suite" "$suite"
