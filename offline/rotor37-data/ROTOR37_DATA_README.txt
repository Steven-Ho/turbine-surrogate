Rotor37 training data archive

Includes exactly the 1000 VTU files referenced by
exp_rotor37_GCN/data/raw/samples.csv in the cluster code package.
No checkpoints, processed data, predictions, or training results are included.

After transferring the archive and its .sha256 file, check:
sha256sum -c turbine-surrogate-rotor37-data.tar.gz.sha256

From the extracted turbine-surrogate PROJECT ROOT:
tar -xzf /path/to/turbine-surrogate-rotor37-data.tar.gz
sha256sum -c DATA_ROTOR37_SHA256SUMS

The files will be at data/rotor37/sample_*.vtu.
Do not extract inside data/rotor37 (that would create nested data/rotor37).
The original sample table is in the CODE archive, not this DATA archive.
Then submit scripts/slurm_train_metax.sh from the project root. Its preparation
step creates the fixed train/validation/test split tables before training.
