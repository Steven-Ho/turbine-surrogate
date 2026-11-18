# Rotor37: Transolver-3 and GeoTransolver

## Implementations and environment

- Transolver-3 official source: https://github.com/thuml/Transolver-3 , commit `ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37`. The unmodified numerical implementation is vendored under `tfai/models/_vendor/transolver3`; `UPSTREAM.json` records its SHA256. Upstream at this commit did not contain a LICENSE file; no new license is asserted here.
- GeoTransolver official implementation: `physicsnemo.experimental.models.geotransolver.GeoTransolver`, released in `nvidia-physicsnemo==2.0.0`, Apache-2.0. This uses official GALE, global context and multiscale local geometry features; it is not a renamed older Transolver.
- PyTorch remains `2.5.1+cu124`. PhysicsNemo 2.0.0 supports this Torch version; newer PhysicsNemo releases require a newer Torch and were not installed. `warp-lang==1.9.0` retains the API expected by PhysicsNemo 2.0.0. `use_te=False` avoids requiring optional Transformer Engine.
- Exact installed environment: `requirements-transolver.lock.txt`. GPU execution requires access outside the filesystem sandbox. Run `PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python ...` from the project root.

## Benchmark contract

Both models use the existing 800 training / 100 validation / 100 held-out test cases, seed 42, learning rate 1e-4, batch size 1, 20 epochs (16000 optimizer updates). All 29773 nodes of each case are retained; graph partitions are rejected to preserve global attention. The preprocessing cache is hardlinked from the established benchmark with independent metadata. No POD projection or target statistics are introduced into model inputs.

Inputs are normalized reference mesh xyz and constant case conditions Omega and P. Output is normalized nodal Pressure, inverted into Pa for testing. Transolver-3 receives xyz and functional conditions, uses official checkpointing and globally coupled chunks of at most 8192 nodes. GeoTransolver receives xyz as spatial inputs and geometric context, and Omega/P as a global embedding. Its official multiscale local geometry features use radii 0.05 and 0.25, and 8/32 neighbors in normalized coordinates. Models use 3 layers, 64 base hidden features, 4 heads and 32 slices. GeoTransolver adds its local feature channels, so parameter counts differ.

This benchmark deliberately keeps the prior five-model reference-geometry input protocol. Actual case coordinates vary slightly and are preserved when writing predictions, but case-specific changing geometry is not supplied to these networks. Consequently it tests condition-to-pressure prediction on Rotor37, not generalization to new blade shapes, blade counts, angles, holes or mesh topologies. Predictions of velocity/temperature require corresponding training labels and additional output channels. Single-seed 20-epoch comparisons do not establish convergence or universal architecture superiority.

## Reproduce

```
PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/run_benchmark.py --phase probe --models Transolver_3 GeoTransolver --run-root runs/transolver_probes
PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/run_benchmark.py --phase train --epochs 20 --models Transolver_3 GeoTransolver --run-root runs/transolver_comparison20
PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/compare_transolvers.py
```

The training phase also runs the 100-case independent test. Best checkpoints contain evaluation source/output paths. Due to limited project disk space, full VTU test outputs are under `/tmp/turbine-transolver-evaluation` via project symlinks. Report generation saves compact complete nodal prediction NPZs under `reports/transolver_comparison20` so the numerical predictions remain in the project. Checkpoints, logs, exact configuration and comparison metrics also remain there. A cold cache rebuild requires additional disk space.

## Checks

Adapter tests check official source identity, chunked/full parity, finite checkpointed backward gradients, absence of target leakage, and actual official GeoTransolver multiscale forward/backward with condition sensitivity on GPU. The report asserts complete 16000-step training, 20 validation epochs, matching original train/validation splits, all 100 held-out cases, finite predictions, exact per-source coordinates and topology. Metrics use raw Pressure on all 2,977,300 nodes with the existing training-derived MAPE threshold. Existing five-model metrics are reused with the baseline report hash recorded.
