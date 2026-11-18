# turbine-surrogate

从 `/data/hesc/qiyuan-dist` 提取的 Rotor37 代理模型项目，预测工况 `Omega、P` 对应的节点 `Pressure` 场。对应源码为 `tfai/models/gcn.py`；当前 GCN 实际采用 MeshGraphNet 消息传递结构。

## 数据与代码

- `data/rotor37/`：1000 个原始 VTU 文件及原目录附件，全部为独立副本。
- `exp_rotor37_GCN/data/raw/`：上游样本表。
- `exp_rotor37_GCN/model_rotor37_GCN/`：上游权重及训练记录，仅保留作参考；本项目运行使用新训练的权重。
- `tfai/`、`scripts/`：完整代理模型、数据处理、训练及评估源码。
- `SOURCE_MANIFEST.json`：初次复制时逐文件验证的源文件 SHA-256 清单；后续配置调整不会重写该来源记录。
- `README.upstream.md`、`configs/*_upstream.yaml`：上游说明与原始训练、评估模板。

Rotor37 通常称为压气机转子算例；当前源目录只确认找到这套转子数据，未据此将其改称涡轮数据。
数据、模型、虚拟环境及 `runs/` 运行产物保存在磁盘，已被 Git 忽略。没有复制上游的预处理缓存和旧评估输出。

## 环境

当前独立环境为 `.venv`：Python 3.11、PyTorch 2.5.1+cu124、PyTorch Geometric 2.6.1、VTK 9.2.6、NumPy 1.26.4。
CUDA PyTorch 及其运行依赖从本机已有环境复制为独立文件，没有连接到其他项目 site-packages；来源见 `verification/local_torch_bootstrap.json`。其他依赖经 pip 安装，源码以 editable 方式安装。
`requirements-linux-cu124.lock` 保存实际依赖版本；`uv.lock` 是上游原始文件，本项目使用 pip 工作流。

在项目目录执行以下命令（需要沙箱外 GPU 访问权限；当前机器为 RTX 3090）：

```bash
cd /data/hesc/Projects/turbine-surrogate
source .venv/bin/activate
python -m pip check
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

在其他机器重新建立 CUDA 12.4 环境：

```bash
PYTHON=/path/to/python3.11 bash scripts/setup_cuda.sh
```

也提供 `scripts/setup_cpu.sh`；可通过 `general.device=cpu` 显式选择 CPU。若下载较慢，可为普通 PyPI 依赖设置 `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`。

## 划分与运行

```bash
python scripts/prepare_rotor37.py
```

固定 seed=42，先留出 100 个独立测试样本，另外 900 个由训练入口分为 800 个训练样本和 100 个验证样本。所有归一化参数由训练集计算。上游原 `test.csv` 的 3 个样本与 `samples.csv` 重叠，因此不使用该表作为本项目独立测试集。

完整训练（默认 100 epochs，完整网格、不分块，自动选择 GPU）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg python scripts/train.py
```

一个 epoch 的全量流程验证，以及独立测试：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg python scripts/train.py training.epochs=1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg python scripts/evaluate.py
python scripts/verify_smoke.py --full
```

快速流程验证使用 6 个训练、2 个验证、2 个独立测试样本，保持完整的 29773 节点网格和 GCN small 结构：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg python scripts/train.py --config-name train_smoke
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg python scripts/evaluate.py --config-name evaluate_smoke
python scripts/verify_smoke.py
```

快速验证只更新 2 个 optimizer steps，用于验证可运行性，不代表预测精度。全量一个 epoch 也只是流程验证，模型精度需继续训练评估。
运行 `prepare_rotor37.py` 会重建原始样本表，使用固定种子保持划分不变；运行训练默认从头训练并覆盖该实验模型，使用不同的 `general.experiment` 并同时指定 `general.train_samples=runs/rotor37/data/raw/samples.csv` 可保存新实验。

## 输出

完整实验在 `runs/rotor37/`，快速实验在 `runs/rotor37_smoke/`：

- `train/best_model.pt`：最佳验证模型，独立评估结果也写入此文件。
- `train/last_checkpoint.pt`：完整续训检查点。
- `train/train.log`：预处理、优化器步数、训练和验证损失。
- `evaluate/test/`：独立测试指标、散点表和图。
- 测试预测 VTU 文件由上游评估链路生成，具体路径可查 `evaluate/test/evaluate.log`。
- `verification/smoke_report.json`、`verification/full_report.json`：划分互斥性、训练步数、有限权重和实际测试指标的核验结果。

训练和测试入口均使用本项目路径，不依赖 `/data/hesc/qiyuan-dist` 中的运行数据。

## 本项目修复

全量训练的训练目标超过 2^24 个值，上游 `torch.quantile` 在计算 MAPE 的 p95 尺度时会报错。`tfai/data/data_pipeline.py` 对大数组改用 NumPy 精确线性分位数计算，不抽样；小数组保留原行为。回归测试覆盖大数组和非有限值过滤：

```bash
OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
```

## 已完成的验证（2026-11-06）

- 独立 CUDA 环境可用，`pip check` 无缺失依赖；GPU 运行证据见 `verification/environment.json`。
- 大数组分位数计算的两项回归测试通过。
- 完整数据覆盖：800 训练、100 验证、100 独立测试，三组互斥；全部样本路径有效。
- GCN small 实际完成一个 epoch、800 个优化器步骤，训练损失 0.012330，验证损失 0.013686。
- 100 个测试样本的预测和目标 VTU 全部写出，无失败。
- 一轮训练后独立测试：MAPE 18.162%，R² 0.239962，RMSE 30705.939。这是可运行性验证结果，尚未完成精度调优。
- 可审阅结果见 `verification/full_report.json`、`verification/smoke_report.json`、`verification/asset_audit.json`。

## 五模型多轮训练对比

统一实验协议、运行和评分方式见 [BENCHMARK.md](BENCHMARK.md)。
正式实验使用 POD、UNet、GCN、Graph_UNet、Graph_Transformer，同一 800/100/100 划分、20 epochs，并在原始 VTU 节点上统一评分。

```bash
python scripts/benchmark_status.py
python scripts/compare_models.py --epochs 20
```

最终报告输出至 `reports/comparison20/REPORT.md`。训练中请先查看状态，避免重复运行调度脚本覆盖任务。

## 五模型 20 轮实测结果（2026-11-06）

所有模型已实际完成 20 轮训练、16000 个优化器步骤，并对同一 100 个独立测试工况完成测试。下表在每个样本自己的原始网格节点上统一评分；完整图表与协议见 [对比报告](reports/comparison20/REPORT.md)。

| 模型 | MAPE (%) | RMSE (Pa) | R² |
|---|---:|---:|---:|
| POD | 2.876 | 3980.6 | 0.9872 |
| GCN | 12.511 | 20111.0 | 0.6740 |
| Graph_Transformer | 16.414 | 23201.8 | 0.5661 |
| Graph_UNet | 16.571 | 23373.7 | 0.5596 |
| UNet | 20.201 | 26040.9 | 0.4534 |

训练集平均场基线 MAPE 为 3.957%，本轮仅 POD 优于该基线。此为单种子、相同样本遍历预算的首轮比较，尚未做逐模型调参；数据存在坐标变化，本轮采用相同节点拓扑上的参考几何训练设定，不代表任意几何泛化能力。

POD 验证集泄漏、U-Net 曲面采样/反归一化/背景输入、逐样本几何写回及逆插值缓存问题均已修复，14 项回归测试通过。有效 U-Net 实验位于 `runs/comparison20_surface/UNet`，旧无效运行明确排除。

## 静子叶片全流场样例

已从 `/data/hesc/测试数据/case4` 提取网格与边界条件到节点 `u,v,w,p,T` 的单工况样例（5503387 节点、16074294 体单元）。输入/标签分离、原始网格对齐和多输出图模型接口已验证；尚无跨工况泛化模型。见 [样例报告与建模方案](reports/stator_case/REPORT.md)。

## Transolver-3 / GeoTransolver

新增两个官方模型的 Rotor37 适配、环境版本和复现命令见 [docs/transolver_rotor37.md](docs/transolver_rotor37.md)。扩展后的实际依赖锁文件为 `requirements-transolver.lock.txt`；原 `requirements-linux-cu124.lock` 对应此前五模型环境。

训练入口分别使用 `--config-name train_rotor37_transolver_3` 和 `--config-name train_rotor37_geotransolver`。统一训练及测试由 `scripts/run_benchmark.py --phase train --epochs 20 --models Transolver_3 GeoTransolver --run-root runs/transolver_comparison20` 执行；完成后运行 `scripts/compare_transolvers.py`，生成 [七模型比较报告](reports/transolver_comparison20/REPORT.md)。

长训练、自动降学习率、200→400 轮延长策略及 POD 对照见 [长训练方案](docs/transolver_long_training.md)，入口为 `scripts/run_long_training.py`。
