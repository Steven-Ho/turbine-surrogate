Ubuntu 22.04 (Jammy), x86_64 user-space OpenGL runtime for VTK imports

Sources: Ubuntu packages libgl1, libglx0, libglvnd0.
https://packages.ubuntu.com/jammy/libgl1
https://packages.ubuntu.com/jammy/libglx0
https://packages.ubuntu.com/jammy/libglvnd0

From the project root, extract:
tar -xzf /path/to/turbine-opengl-jammy.tar.gz
sha256sum -c turbine-opengl-jammy/SHA256SUMS
sbatch turbine-opengl-jammy/slurm_vtk_with_opengl.sh

For every existing training/check job, insert this AFTER module load and
before running Python (use an absolute path if the working directory differs):
source /absolute/path/to/turbine-surrogate/turbine-opengl-jammy/load.sh

Example:
module purge
module load conda-py31010/3.10.10
source "$HOME/venvs/turbine-metax/bin/activate"
source "$HOME/Projects/turbine-surrogate/turbine-opengl-jammy/load.sh"
python scripts/train.py --config-name train_smoke

This bundle adds its lib directory to LD_LIBRARY_PATH in the current shell.
It does not install system packages or change the vendor PyTorch installation.
The cluster's existing X11/xcb/libc libraries are required, as shown in the
provided diagnostic log. No renderer or GPU driver is included or validated.
This addresses VTK import/data IO; it is not an OpenGL rendering stack.

Check the test log vtk_opengl_JOBID.out for ALL CHECKS PASSED.
If another shared library is missing, preserve the exact error for diagnosis.
