#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=768G
#SBATCH --get-user-env
#SBATCH --output=logs/train_gpu_%j.out
#SBATCH --error=logs/train_gpu_%j.err

set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from the extracted project root}"
# Load after module commands, before any Python process.
TURBINE_OPENGL_RUNTIME="${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}"
if [ ! -f "$TURBINE_OPENGL_RUNTIME/load.sh" ]; then
  echo "Missing OpenGL runtime: $TURBINE_OPENGL_RUNTIME/load.sh" >&2
  echo "Extract turbine-opengl-jammy.tar.gz in the project root first." >&2
  exit 2
fi
source "$TURBINE_OPENGL_RUNTIME/load.sh"
export TFAI_PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
python -c 'import vtk, torch; print("VTK:", vtk.vtkVersion.GetVTKVersion()); assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
python scripts/prepare_rotor37.py
# Single-process training on the first visible GPU; this is not DDP.
python scripts/train.py --config-name "${TURBINE_CONFIG:-train_smoke}" general.device=cuda:0 "$@"
