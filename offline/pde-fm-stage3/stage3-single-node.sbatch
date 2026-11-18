#!/bin/bash
#SBATCH --job-name=pde-fm-stage3
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=pde-fm-stage3-%j.out
#SBATCH --error=pde-fm-stage3-%j.err

set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${PDE_FM_VENV:-$HOME/venvs/pde-fm-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from pde-fm root}"

export PDE_FM_DATA_DIR="${PDE_FM_DATA_DIR:-$HOME/Dataset/the_well}"
export PDE_FM_GPUS_PER_NODE="${PDE_FM_GPUS_PER_NODE:-2}"
export PDE_FM_MODE="${PDE_FM_MODE:-fresh}"
export PDE_FM_STAGE3_DIR="$PWD/stage3"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

case "$PDE_FM_MODE" in
  fresh)
    export PDE_FM_RUN_ROOT="${PDE_FM_RUN_ROOT:-$PWD/runs/pde-fm/stage3-${SLURM_JOB_ID}}"
    test ! -e "$PDE_FM_RUN_ROOT" || { echo 'Use a new run root for fresh mode'; exit 2; }
    mkdir -p "$PDE_FM_RUN_ROOT"
    printf '%s\n' "$PDE_FM_GPUS_PER_NODE" > "$PDE_FM_RUN_ROOT/gpu-count.txt"
    ;;
  resume)
    : "${PDE_FM_RUN_ROOT:?Set the previous fresh run root}"
    export PDE_FM_RUN_ROOT
    test -f "$PDE_FM_RUN_ROOT/fresh.ok"
    test "$(cat "$PDE_FM_RUN_ROOT/gpu-count.txt")" = "$PDE_FM_GPUS_PER_NODE"
    test ! -e "$PDE_FM_RUN_ROOT/resume.ok"
    test -f "$PDE_FM_RUN_ROOT/training/checkpoints/last/metadata.pt"
    ;;
  *) echo 'PDE_FM_MODE must be fresh or resume'; exit 2 ;;
esac
case "$PDE_FM_RUN_ROOT" in /*) ;; *) echo 'PDE_FM_RUN_ROOT must be absolute'; exit 2 ;; esac
for split in train valid test; do
  test -d "$PDE_FM_DATA_DIR/active_matter/data/$split"
done
for file in pde_fm_stage3.yaml launch_stage3.py; do
  test -f "$PDE_FM_STAGE3_DIR/$file"
done

export PDE_FM_JOB_DIR="$PDE_FM_RUN_ROOT/jobs/$SLURM_JOB_ID"
mkdir -p "$PDE_FM_JOB_DIR"
cp "$PDE_FM_STAGE3_DIR/"{pde_fm_stage3.yaml,launch_stage3.py,stage3-single-node.sbatch} "$PDE_FM_JOB_DIR/"
git rev-parse HEAD > "$PDE_FM_JOB_DIR/git-commit.txt"
git status --short > "$PDE_FM_JOB_DIR/git-status.txt"
python -m pip freeze > "$PDE_FM_JOB_DIR/environment.txt"
scontrol show job "$SLURM_JOB_ID" > "$PDE_FM_JOB_DIR/slurm-job.txt"
echo "node=$(hostname) mode=$PDE_FM_MODE GPUs=$PDE_FM_GPUS_PER_NODE"
echo "PDE_FM_RUN_ROOT=$PDE_FM_RUN_ROOT"

args=(--config-dir "$PDE_FM_JOB_DIR" --config-name pde_fm_stage3)
if [ "$PDE_FM_MODE" = resume ]; then
  args+=(auto_resume=true trainer.max_epoch=3)
else
  args+=(auto_resume=false trainer.max_epoch=2)
fi
# Save the composed input configuration before launching all ranks.
# Composition does not initialize a GPU or distributed process group.
python -m walrus.train "${args[@]}" --cfg job > "$PDE_FM_JOB_DIR/composed-config.yaml"

srun --ntasks=1 --gres="gpu:$PDE_FM_GPUS_PER_NODE" --kill-on-bad-exit=1 \
  python -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="$PDE_FM_GPUS_PER_NODE" \
    --log-dir="$PDE_FM_JOB_DIR/ranks" --redirects=3 --tee=3 \
    "$PDE_FM_JOB_DIR/launch_stage3.py" "${args[@]}" \
  2>&1 | tee "$PDE_FM_JOB_DIR/console.log"

# Verify epoch metadata and nonempty optimizer state without loading on GPU.
srun --ntasks=1 --gres="gpu:$PDE_FM_GPUS_PER_NODE" python - <<'PY'
import os
from pathlib import Path
import torch
root = Path(os.environ['PDE_FM_RUN_ROOT']) / 'training/checkpoints'
expected = 3 if os.environ['PDE_FM_MODE'] == 'resume' else 2
last = root / 'last'
metadata = torch.load(last / 'metadata.pt', map_location='cpu', weights_only=False)
assert metadata['epoch'] == expected, metadata
assert last.resolve() == (root / f'step_{expected}').resolve()
checkpoint = torch.load(last / 'full_checkpoint.pt', map_location='cpu', weights_only=False)
assert checkpoint['app']['model']
assert checkpoint['app']['optimizer']['state'], 'Empty optimizer state'
print('PDE_FM_CHECKPOINT_OK', metadata)
PY

# Require resume messages and the expected next epoch in the continuation log.
if [ "$PDE_FM_MODE" = resume ]; then
  grep -q 'Resume from checkpoint' "$PDE_FM_JOB_DIR/console.log"
  grep -q 'Epoch 3/3: starting training' "$PDE_FM_JOB_DIR/console.log"
  if grep -q 'Epoch 1/3: starting training' "$PDE_FM_JOB_DIR/console.log"; then
    echo 'Unexpected restart from epoch 1'; exit 2
  fi
fi
printf '%s\n' "$SLURM_JOB_ID" > "$PDE_FM_RUN_ROOT/$PDE_FM_MODE.ok"
echo "PDE_FM_STAGE3_OK mode=$PDE_FM_MODE"
