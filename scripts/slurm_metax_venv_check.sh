#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=768G
#SBATCH --get-user-env
#SBATCH --output=logs/venv_check_%j.out
#SBATCH --error=logs/venv_check_%j.err

set -eo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Submit with sbatch on a GPU node; do not run with bash on the login node." >&2
  exit 2
fi

module purge
module load conda-py31010/3.10.10

export PYTHONUNBUFFERED=1
TURBINE_VENV="${TURBINE_VENV:-$HOME/venvs/turbine-metax}"
if [ ! -f "$TURBINE_VENV/bin/activate" ]; then
  echo "Virtual environment missing: $TURBINE_VENV/bin/activate" >&2
  echo "Set TURBINE_VENV to the existing environment path before submission." >&2
  exit 2
fi

echo "=== Before virtual environment activation ==="
echo "node=$(hostname)"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
python - <<'BASELINE'
import sys
import torch
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, torch.__file__)
print("GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Base environment cannot access GPU; check node allocation and module")
BASELINE
export PROBE_BASE_TORCH_FILE="$(python -c 'import os, torch; print(os.path.realpath(torch.__file__))')"
export PROBE_BASE_TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"

source "$TURBINE_VENV/bin/activate"
set -u
# Always inspect the activated environment's Python.
PROBE_PYTHON="$TURBINE_VENV/bin/python"
export PROBE_EXPECTED_VENV="$TURBINE_VENV"
echo "=== After virtual environment activation ==="
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
echo "job_id=${SLURM_JOB_ID:-none} node=$(hostname) started=$(date -Is)"
echo "submit_dir=${SLURM_SUBMIT_DIR:-unknown} working_dir=$PWD"
if type module >/dev/null 2>&1; then module list 2>&1 || true; fi
if command -v ht-smi >/dev/null 2>&1; then
  ht-smi || true
else
  echo "ht-smi not found in PATH"
fi
command -v "$PROBE_PYTHON"
"$PROBE_PYTHON" -m pip --version || true

"$PROBE_PYTHON" - <<'PY'
import importlib
import importlib.metadata as metadata
import os
import platform
import site
import sys
import traceback

failures = []

def check(name, function):
    try:
        function()
        print(f"PASS {name}")
    except Exception as exc:
        failures.append(name)
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stdout)

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Architecture:", platform.machine())
print("Prefix:", sys.prefix, "Base prefix:", sys.base_prefix)
print("User site enabled:", site.ENABLE_USER_SITE)
for key in ("CONDA_PREFIX", "VIRTUAL_ENV", "MACA_PATH", "CUDA_VISIBLE_DEVICES"):
    print(f"{key}={os.environ.get(key, '<unset>')}")

def torch_info():
    global torch
    import torch
    print("torch:", torch.__version__, torch.__file__)
    print("torch.version.cuda:", torch.version.cuda)
    print("GPU available:", torch.cuda.is_available())
    print("Visible GPU count:", torch.cuda.device_count())
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is unavailable; verify allocation and example environment")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name}, memory={p.total_memory / 2**30:.2f} GiB")
    print("Distributed available:", torch.distributed.is_available())
    print("NCCL API available:", torch.distributed.is_nccl_available())

def verify_venv():
    assert os.path.realpath(sys.prefix) == os.path.realpath(os.environ["PROBE_EXPECTED_VENV"]), "Wrong Python environment"
    assert sys.prefix != sys.base_prefix, "Python is not running in a venv"
    import torch
    assert os.path.realpath(torch.__file__) == os.environ["PROBE_BASE_TORCH_FILE"], "PyTorch path changed after activation"
    assert torch.__version__ == os.environ["PROBE_BASE_TORCH_VERSION"], "PyTorch version changed after activation"

check("Virtual environment inherits platform PyTorch", verify_venv)
check("PyTorch and GPU visibility", torch_info)

def matmul_backward():
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item()

def fft_backward():
    x = torch.randn(2, 16, 16, device="cuda", requires_grad=True)
    spectrum = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho")
    y = torch.fft.irfftn(spectrum, s=(16, 16), dim=(-2, -1), norm="ortho")
    torch.testing.assert_close(y, x, atol=1e-5, rtol=1e-4)
    y.square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(x.grad).all().item()

def attention_backward():
    q = torch.randn(2, 2, 16, 32, device="cuda", requires_grad=True)
    torch.nn.functional.scaled_dot_product_attention(q, q, q).square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(q.grad).all().item()

def amp_backward(dtype):
    layer = torch.nn.Linear(32, 32).cuda()
    x = torch.randn(16, 32, device="cuda")
    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = layer(x).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(layer.weight.grad).all().item()

if "PyTorch and GPU visibility" not in failures:
    check("FP32 matmul and backward", matmul_backward)
    check("FFT complex operations and backward", fft_backward)
    check("SDPA and backward", attention_backward)
    check("FP16 autocast and backward", lambda: amp_backward(torch.float16))
    check("BF16 autocast and backward", lambda: amp_backward(torch.bfloat16))

print("\nProject dependency imports (missing packages are not installed automatically):")
dependencies = {
    "numpy": "numpy", "pandas": "pandas", "pyyaml": "yaml",
    "matplotlib": "matplotlib", "hydra-core": "hydra", "h5py": "h5py",
    "vtk": "vtk", "scipy": "scipy", "scikit-learn": "sklearn",
    "tqdm": "tqdm", "einops": "einops", "torch-geometric": "torch_geometric",
}
for distribution, module in dependencies.items():
    def inspect(distribution=distribution, module=module):
        imported = importlib.import_module(module)
        print(distribution, metadata.version(distribution), getattr(imported, "__file__", None))
    check(f"dependency {distribution}", inspect)

print("\nOptional/vendor packages (absence alone is not a failure):")
for name in ("torchvision", "torchaudio", "triton", "torch-scatter", "torch-sparse", "pyg-lib"):
    try:
        print(name, metadata.version(name))
    except metadata.PackageNotFoundError:
        print(name, "NOT INSTALLED")

print("\nSummary:", "ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
for name in failures:
    print(" -", name)
print("No packages were installed. No network access was attempted.")
print("This does not validate DDP communication, custom Triton kernels, or full training.")
sys.exit(1 if failures else 0)
PY
