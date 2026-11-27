# Rotor37 saved-prediction pressure evaluation

CPU-only evaluation of the existing 100 held-out cases; no retraining. Training-only pressure standard deviation: 35201.23971404369.

| Model | Mean case relative L2 (%) | Pooled relative L2 (%) | Pressure RMSE (export scale) | Jump RMSE / sigma | Strongest 1% jump RMSE / sigma |
|---|---:|---:|---:|---:|---:|
| GeoTransolver | 1.602281 | 1.626620 | 1900.9144 | 0.023420 | 0.081551 |
| Transolver_3 | 2.163109 | 2.191111 | 2560.5951 | 0.027515 | 0.087896 |
| POD | 3.142640 | 3.210433 | 3751.8046 | 0.019007 | 0.062063 |

Same metric formulas as NVIDIA, different input/target setup and split algorithm (Python random vs NumPy); test not validation; no SOTA ranking.

Historical exports label pressure as Pa; upstream official example states source units are undeclared.

See metrics.json for definitions, prediction hashes, sample membership, confidence intervals and paired comparisons.

## Interpretation

GeoTransolver has the lowest field RMSE and mean relative L2 among these three saved models. POD has the lowest pressure-jump error, including the strongest 1% of true jumps. Thus lower global field error does not imply sharper or better-located shocks. These edge metrics identify a weakness but do not by themselves distinguish smoothing from shock displacement.

Historical reports used pooled relative L2 (GeoTransolver 1.626620%); the NVIDIA convention first computes each case's relative L2 and then averages it (1.602281%). Neither is MAPE.

NVIDIA's published three-seed validation reference reports pressure relative L2 0.457%, jump error 0.0055 and strongest-1% jump error 0.0107. Our numerical values are higher, but this is not a matched benchmark: different splits, different train-derived sigma, geometry inputs, output targets, decoder and training recipe. No performance ratio or rank should be inferred.
