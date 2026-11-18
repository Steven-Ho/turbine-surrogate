#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=768G
#SBATCH --get-user-env
#SBATCH --output=vtk_library_check_%j.out
#SBATCH --error=vtk_library_check_%j.err

set -eo pipefail
module purge
module load conda-py31010/3.10.10

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Please submit this file with sbatch, not bash on a login node."
  exit 2
fi

echo "=== Job and node ==="
echo "job_id=$SLURM_JOB_ID node=$(hostname)"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
module list 2>&1 || true

echo "=== OpenGL library cache ==="
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p 2>/dev/null | grep -E 'libGL\.so\.1|libGLX|libGLdispatch' || true
elif [ -x /sbin/ldconfig ]; then
  /sbin/ldconfig -p 2>/dev/null | grep -E 'libGL\.so\.1|libGLX|libGLdispatch' || true
else
  echo "ldconfig command not found"
fi

echo "=== Existing OpenGL libraries and symlinks ==="
# Search read-only; bounded to avoid scanning a large shared filesystem forever.
# find -H follows command-line directory symlinks such as /lib -> /usr/lib.
for directory in /usr/lib /usr/lib64 /lib /lib64 /opt/hpcc-3.5.3; do
  if [ -d "$directory" ]; then
    echo "Searching $directory"
    if command -v timeout >/dev/null 2>&1; then
      timeout 45s find -H "$directory" -maxdepth 8 \
        \( -name 'libGL.so*' -o -name 'libGLX.so*' -o -name 'libGLdispatch.so*' \) \
        -ls 2>&1 || true
    else
      find -H "$directory" -maxdepth 8 \
        \( -name 'libGL.so*' -o -name 'libGLX.so*' -o -name 'libGLdispatch.so*' \) \
        -ls 2>&1 || true
    fi
  fi
done

echo "=== Virtual environment ==="
TURBINE_VENV="${TURBINE_VENV:-$HOME/venvs/turbine-metax}"
if [ ! -f "$TURBINE_VENV/bin/activate" ]; then
  echo "Missing virtual environment: $TURBINE_VENV"
  exit 2
fi
source "$TURBINE_VENV/bin/activate"
export PYTHONUNBUFFERED=1

python - <<'PY'
import ctypes
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

print('Python:', sys.executable, flush=True)
print('Virtual environment:', sys.prefix, flush=True)
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', '<unset>'), flush=True)

print('\n=== Direct libGL load ===', flush=True)
try:
    ctypes.CDLL('libGL.so.1')
    print('PASS libGL.so.1 can be loaded', flush=True)
except OSError as exc:
    print('FAIL libGL.so.1:', exc, flush=True)

print('\n=== Installed VTK and its dynamic library dependencies ===', flush=True)
try:
    dist = metadata.distribution('vtk')
    print('VTK version:', dist.version, flush=True)
    library_dir = Path(dist.locate_file('vtkmodules'))
    print('VTK library directory:', library_dir, flush=True)
    if shutil.which('ldd'):
        libraries = sorted(set(library_dir.glob('libvtkRenderingOpenGL2*.so*')) |
                           set(library_dir.glob('vtkRenderingOpenGL2*.so')) |
                           set(library_dir.glob('vtkWebCore*.so')))
        for library in libraries:
            print('\nldd:', library, flush=True)
            try:
                result = subprocess.run(['ldd', str(library)], capture_output=True,
                                        text=True, timeout=30)
                print(result.stdout, flush=True)
                if result.stderr:
                    print(result.stderr, flush=True)
            except Exception as exc:
                print('ldd failed:', exc, flush=True)
    else:
        print('ldd not found', flush=True)
except metadata.PackageNotFoundError:
    print('VTK package is not installed', flush=True)

print('\n=== VTK import test ===', flush=True)
try:
    import vtk
    print('PASS VTK import:', vtk.vtkVersion.GetVTKVersion(), flush=True)
except Exception:
    print('FAIL VTK import', flush=True)
    traceback.print_exc(file=sys.stdout)
    print('\nNo libraries were installed and no environment paths were changed.', flush=True)
    sys.exit(1)
print('\nNo libraries were installed and no environment paths were changed.', flush=True)
PY
