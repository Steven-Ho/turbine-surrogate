Target: Linux x86_64, CPython 3.10, glibc >= 2.28; MetaX/Mars vendor PyTorch retained.
Check compute-node glibc using: getconf GNU_LIBC_VERSION
Build environment: existing HPCC Python 3.10.10 / Ubuntu 22.04 container.

Upload the archive and project source separately. Extract this archive into a
persistent directory. Load the cluster module and activate the existing venv:

module purge
module load conda-py31010/3.10.10
source ~/venvs/turbine-metax/bin/activate

From this extracted directory:
sha256sum -c SHA256SUMS
python -m pip install --dry-run --no-index --find-links=wheelhouse -r requirements-metax-offline.txt
python -m pip install --no-index --find-links=wheelhouse -r requirements-metax-offline.txt
python -m pip install --no-index --find-links=wheelhouse 'setuptools>=68' wheel

Then from the project source directory:
python -m pip install --no-index --no-deps --no-build-isolation -e .
sbatch scripts/slurm_metax_venv_check.sh

No torch, torchvision, torchaudio, triton, NVIDIA libraries, or PyG binary
extensions are included. PyG model execution and DDP must be tested on the
cluster. The clean offline verification environment intentionally has no torch;
verification covers dependency installation and imports other than PyG/torch.
resolved-packages.txt records all versions chosen during the clean install.

System-library prerequisite: VTK 9.2.6 imports require OpenGL/X11 runtime
libraries, including libGL.so.1. These are OS libraries, not pip dependencies.
The original minimal verification image lacked libGL.so.1; the final offline
verification mounted local OpenGL/X11 libraries read-only. They are NOT shipped
in wheelhouse. If the cluster VTK import reports missing libGL.so.1 or another
system .so, use a platform-provided runtime/module or ask the administrator.
Do not replace the working vendor GPU driver or runtime to fix a VTK import.
