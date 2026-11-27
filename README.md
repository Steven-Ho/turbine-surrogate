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

## faNN 正式训练

faNN 的 epoch 训练、验证选优、调度早停、完整续训、独立测试及 Slurm 提交见 [正式训练说明](docs/fann_formal_training.md)。一键入口为 `bash scripts/submit_fann_formal.sh`。默认 full 数据模式使用全部节点；神经模型支持多节点数据并行，GeoTransolver另提供实验性网格域并行（见正式训练说明中的4卡检查及8节点入口），X203域并行显存与性能仍需集群实测。旧 `run_fann.py --smoke` 保留为局部接口调试入口。


## CR15 MAP数据准备（第1–3步）

入口`scripts/prepare_cr15_map.py`使用已完成的`scripts/audit_cr15_map.py`报告，生成独立于faNN的CR15五场缓存；此处尚不接入训练，也不改动既定FANN_SPLITS。CPU依赖仅NumPy和标准库，不导入Torch/VTK，不需要GPU。

集群运行时，`--root`沿用审计时的数据根目录，`--audit`为该次审计的输出目录（包含summary.json、eligible_A.csv）。使用新输出目录：

```bash
git pull
bash scripts/submit_cr15_prepare.sh \
  --root /实际路径/CR15_stator_400_MAP_results \
  --audit /实际路径/已通过的审计报告目录 \
  --output runs/cr15_prepared_v1 --workers 2
```

提交单节点CPU任务，4CPU、32G主存、6小时上限，日志logs/cr15_prepare_<job>.out/.err。若不经Slurm，可在有NumPy的CPU环境中直接运行同参数的`python scripts/prepare_cr15_map.py ...`。无需重新运行全库审计：读取原审计的source SHA256作为身份记录，不重复计算源文件哈希；转换过程仍检查当前读取的记录长度、形状、有限值、正压力/温度及通道声明，避免生成错位或损坏缓存。

首轮仅A级，保持原train/validation/test/baseline划分。本机同版清单402例包含A级250例：train199、valid23、test26、baseline2；B级152例单独列入excluded_B.csv，不转换、不混入真值训练。基准2例单独转换但不进入训练/归一化。最终数量始终以集群实际清单和原审计一致性为准，不凭文件名或训练目录重新随机划分。

输出包括：

- eligible_A.csv / excluded_B.csv：仅身份、原划分及设计/网格质量元数据。
- diagnostics_only.csv：原清单全部列，仅供诊断；其中已求解流量、压比、实际出口压力/温度不得作为输入。
- samples/<sample>.npz：原始物理坐标float32、五场targets、输出名/单位、固定边界向量、块形状/偏移/节点编号、原始整数标记及其存在掩码。无Density、无伪造wall_distance、无把整数标记自动解释为边界/有效节点掩码。
- metadata/<sample>.json：审计源身份、精度转换误差、块布局、目标统计及必要提示。
- splits.json：相对缓存根目录的train/valid/test/baseline列表；不改变原validation成员。
- train_statistics.json：仅训练集坐标界限和五场均值/标准差，缓存本身保留物理量，不提前标准化。
- contract.json / plan.json / summary.json：字段、边界、来源与完成状态。全A级完成时cache_ready=true；不代表已实现下一步CR15训练适配。

通道映射来自`configs/datasets/cr15_map_v1.json`：MAP八通道中取[1,2,3,0,4]得到u/v/w/静压/静温，不将湍流黏度、转捩量或比热比当密度。完整坐标必须来自审计确认的ofmeshplt.xyz，明确拒绝inmeshcfd.xyz。按原块序拼接，各块i最快、再j再k；节点ID为这一拼接顺序的零基编号，可结合块偏移/维数还原(i,j,k)。接口重复坐标和原始整数标记原样保留，不自行过滤或推断接口连接。MAP输出坐标按米记录，不套用CAD毫米缩放。

固定边界向量为进口总压2500000Pa、进口总温1850K、进口气流角0度、湍流强度0.014、出口参考静压1314400Pa、转速0rpm；来源为交付README的统一求解设置，不能用清单里的实际ps_out_pa代替参考边界。叶片数42作为固定元数据。此数据主要检验几何变化，不能据此声称覆盖可变边界条件。本机无逐例原生边界输入文件，因此来源标记为数据集声明，不伪称已逐例解析验证。标签标记quasi_steady_not_convergence_certified。

中断后复用已生成缓存：

```bash
bash scripts/submit_cr15_prepare.sh \
  --root /实际路径/CR15_stator_400_MAP_results \
  --audit /实际路径/已通过的审计报告目录 \
  --output runs/cr15_prepared_v1 --workers 2 --resume
```

续跑对相同contract/审计源身份/split的完整单例直接复用，不重读原始网格/场；不完整单例重新转换。输出被标记failed/partial时不能作为完整训练缓存。可用`--samples CR15S_0001 CR15W_0001`在独立目录先做小样例，随后去掉--samples并--resume补齐A级；或用`--plan-only`只生成选择计划（不强制审计路径，不产出网格/场缓存）。路径迁移支持重复`--path-map OLD=NEW`，完整审计与manifest仍须对应同一数据集。

本机真实402行清单的选择计划已生成于runs/cr15_preparation_plan_20261127；CR15S_0001、CR15W_0001的真实八场解析与独立审计逐块范围一致。本机缺ofmeshplt.xyz，未声称完成真实网格/末场配对转换；合成测试覆盖多块大小端格式、映射、输入网格拒绝、审计split一致性、原始标记保留、无求解量输入、train-only统计和缓存续跑。本机代表旧可视化metadata哈希不同于当前末场，故图/统计不作为标签来源，转换记录此类提示但以原审计指定末场为准。
