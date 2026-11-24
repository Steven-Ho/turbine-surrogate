# faNN 正式训练与整域数据入口

本流程补齐了监督学习实验协议：按 epoch 训练、完整验证、验证选优、学习率调度、早停、原子检查点、优化器/随机数/数据位置恢复、冻结后的独立测试和七模型汇总。默认申请每模型一张 GPU，七个数组任务各自训练一个模型。

**整域数据模式已实现，但不是多 GPU 域并行实现。** `FANN_DOMAIN=full` 会使用所有 9,806,346 节点、9,429,120 六面体，不裁成 512 单元，也不静默降采样。是否能在 X203 单卡 64 GiB 上完成各架构的前向/反向，必须由集群实测确认。旧 dense 图路径的整图边激活可能超出单卡容量；新版 bounded 图执行见文末。GeoTransolver 邻域与注意力激活仍可能超出单卡容量；OOM 会将任务标记为 failed，不会换模型、换 CPU 或把整域结果改称局部结果。本轮未实现跨 GPU 图分区/halo 和跨 GPU 全局切片通信。

## 数据准备

要求原始发布布局 `~/Dataset/faNN/data/hf_train-*.parquet` 和 `tree_constant_part.pkl`，不是 Arrow save_to_disk 布局。准备环境需要兼容 NumPy1 的 `pyarrow==19.0.1`，可在已激活的厂商环境执行 `python -m pip install -r requirements-fann.txt`（离线集群需提前备好 wheel）。只从有标签的 hf_train 准备样本；官方 89 个 hf_test 没有公开输出，不能直接计算误差。

- `full`：每样本保存完整实际坐标、壁面距离、六目标场、转速/出口压力、GeometryNumber。公共拓扑单独存为 `shared_mesh.npz`，各样本引用同一文件，避免复制数百份连接关系。公共拓扑校验源 sidecar SHA256。原始节点/单元编号均保留。
- `structured_patches`：显式选择采样域实验，在公共结构索引的各网格块、三个叶高区间分别选择局部块，固定 seed/cell IDs，不使用预测目标选择位置。每个小块保留真实六面体连接。默认 10000 单元、可设置 1..10000；多个块合在一起不是完整叶道，误差只代表这些块，且块边界存在上下文截断。
- `first`：仅为旧调试协议保留；正式入口默认拒绝旧数据。原先的 512 单元 NPZ 不含壁面距离/新域元数据，需要重新准备；只有手动声明 legacy 才能用于旧协议验证。

按 GeometryNumber 固定 seed42 将有标签样本分到约80/10/10的训练/验证/内部测试几何组；三组几何不能交叉。同一几何的所有工况进入同一组。公共气体/入口参数保存在 `fixed_conditions`，若它们变化但没有对应模型输入则拒绝混合训练。

输出：

```
runs/fann_prepared_时间戳/
  shared_mesh.npz              # full 模式共用
  hf_train-..._0.npz           # 各样本
  selected_cell_ids.npy        # 仅采样模式
  splits.json
```

NPZ 与 shared_mesh 引用目前使用绝对路径。移动已准备数据后需更新引用和 splits，再启动新实验；续训严格拒绝数据签名或划分变化。

## 集群一键提交

先 `git pull --ff-only origin master`，从项目根目录提交。原有厂商环境和 `turbine-opengl-jammy/load.sh` 仍需可用；可通过 TURBINE_VENV / TURBINE_OPENGL_RUNTIME 指定路径。不要覆盖厂商 PyTorch。

```
unset FANN_STEPS FANN_SPLITS FANN_PREPARED FANN_RUN_ROOT FANN_RESUME
export FANN_SOURCE="$HOME/Dataset/faNN"
export FANN_DOMAIN=full
export FANN_EPOCHS=400
bash scripts/submit_fann_formal.sh
```

脚本返回准备、数组和汇总作业号，以及固定的结果根目录；自动依赖为：准备成功 → 一次性 CPU 数据登记 → 七训练任务 → 无论任务成功与否都执行汇总。准备失败时训练不运行。汇总不会把缺失或失败模型当作成功。

若已有**新版**数据，可跳过准备：

```
export FANN_SPLITS="$PWD/runs/fann_prepared_实际时间戳/splits.json"
export FANN_RUN_ROOT="$PWD/runs/fann_full_experiment_01"
export FANN_DOMAIN=full
bash scripts/submit_fann_formal.sh
```

若希望先验证可训练性和集群算子，可显式使用 `FANN_DOMAIN=structured_patches`、`FANN_CELLS=10000`，重新准备后运行；这不能作为整域训练性能或精度的证据。固定数据和模型后，应先测单步显存/吞吐再申请长期作业。

## 训练协议

- 每个 epoch 以 seed+epoch 的可重建随机顺序遍历全部训练样本，每样本一次优化器更新，batch=1；不再使用 FANN_STEPS。旧变量非空时提交脚本直接报错，避免把更新次数误当轮数。
- 一次只构造/保留一个算例的 GPU 输入，不预先把全部算例放进显存。CPU 数据 LRU 默认1个样本；归一化逐样本累计，不拼接全部目标场。
- 输入为实际 xyz、壁面距离、RotatingVelocityX、OutletPressure；六个输出为绝对参考系 VelocityX/Y/Z、静压、静温、Density。不使用 MassFlow、Efficiency 等求解后标量。
- 目标均值/标准差只由训练集拟合，逐通道标准化；训练与选择指标为六通道标准化 MSE，避免 Pa 单位主导速度/密度。几何以训练集共同原点、等比例长度归一化。壁面距离除以同一长度尺度。
- Adam，初始 lr=1e-4；ReduceLROnPlateau，factor=.5、patience=8、相对有效改善阈值.005、cooldown=2、min_lr=1e-6。每个完整验证周期 step 一次并保存调度状态。
- 默认最多400轮；每轮验证完整验证集，任何严格更低的验证 MSE 都保存 best。连续60轮没有新最优值则 early stop。梯度范数上限1，发现非有限损失/梯度直接报错。
- `last.pt` 包含模型、Adam、调度器、训练归一化、POD基底、CPU/CUDA/Python/NumPy RNG、当前epoch/样本位置/部分损失累计、数据签名与配置；临时写入后原子替换。SIGTERM/SIGINT 在安全步骤边界保存，验证期间也会轮询退出。
- 训练结束后重新加载 best，然后计算完整验证与内部测试集的六变量物理单位 RMSE/MAE/R²，以及壁面距离小于1mm节点的局部RMSE。POD 截断与 UNet 映射误差均计入原始目标；指标为节点加权，不是体积加权，也未报告守恒残差。
- 测试集的内容完整性/几何归属在首次数据登记时核验，后续数据不变时复用登记，但测试指标不参与训练、学习率、选模或早停。改变测试标签不会改变训练所得最佳权重（有回归检查）。

## 七模型的整域处理差别

- POD：采用磁盘 memmap 和快照 Gram 矩阵拟合训练集基底，不把所有快照堆进 RAM；默认32模态。整域约295个训练上限样本需要约77GB的快照/基底临时空间（实际按训练集大小计算），额外留20%余量。基底通过公共节点ID对应回所有节点。几何输入先按公共节点顺序池化为256个区间的坐标/壁面距离描述符，避免把约4000万维几何直接输入全连接层；该表示是独立 POD 基线，需评估表达能力。推理和续训直接重载保存基底；新开训练也复用匹配训练数据、归一化和模态数的磁盘基底缓存，不重新拟合。
- UNet：完整节点几何映射到规则体素网格，三线性解码回全部原始节点并计算损失；网格分辨率默认32，可通过 FANN_GRID 设置64/128等。默认分辨率仅为可运行基线，不保证解析近壁细节；该映射不是保守重网格化。原始节点输出是全域，但内部表示有分辨率瓶颈。
- 三种图网络：使用完整真实六面体边，不随机删点后沿用旧连接。默认 bounded：GCN/Graph_Transformer 使用有限感受野 owner+halo 分区、完整算例累计梯度；Graph_UNet 保留全局TopK，CPU邻接扩展及分批稀疏乘法。可使用节点内按层模型并行进一步分摊参数及激活；不是跨节点域并行。dense仅供小图对照。
- Transolver-3：整域按8192节点分块，所有块贡献共同物理切片，保持全域耦合；训练使用官方梯度检查点，但并不自动跨GPU。
- GeoTransolver：采用已移植的 portable_torch / KDTree 查询路径，以实际xyz/壁面距离为局部输入、几何点集和全局工况为上下文。整域采用节点分块的全局切片聚合、块级梯度重计算和 CPU 激活卸载，可配置节点内按层多卡执行；与NVIDIA Warp版本的整域性能不能直接等同。邻域缓存默认目录由 TURBINE_GEO_CACHE_DIR 控制，千万节点邻域缓存可能很大，应计入磁盘规划。

## 整域单步容量检查

在集群已准备完整多样本数据后，可先单独测一个模型：

```
python scripts/train_fann.py --splits "$FANN_SPLITS" --models GeoTransolver \
  --domain full --device cuda --epochs 400 --max-updates 1 --output runs/fann_full_capacity_check
```

这会从训练集执行一个完整算例更新，然后保存中断检查点并以130退出，不作测试集评估；不是正式完成结果。若有OOM/稀疏算子错误会明确失败。该检查仍需要先从训练集拟合归一化，POD还需要拟合基底，因此不是纯粹一秒钟启动的测速。GPU算子需要预热和多步稳定计时后才能外推总耗时。

## 输出与恢复

```
runs/fann_formal_时间戳/
  POD/
    status.json
    last.pt
    best.pt
    history.json
    report.json
    predictions/valid/prediction_00000.npz
    predictions/test/prediction_00000.npz
    pod_scratch/basis.f32
  UNet/ ...
  GCN/ ...
  Graph_UNet/ ...
  Graph_Transformer/ ...
  Transolver_3/ ...
  GeoTransolver/ ...
  comparison.json
  metrics.csv
  REPORT.md
```

不再出现双层同名模型目录。每个 prediction 保存原节点ID、六物理量值、变量名和来源路径。日志为 `fann_数组作业号_索引.out`；epoch、训练/验证loss、学习率、早停计数逐轮打印。Slurm 提交申请每任务1 GPU/4 CPU/64G主存，没有宣称跨GPU训练。

恢复已停止的任务：

```
export FANN_SPLITS="原准备目录/splits.json"
export FANN_RUN_ROOT="原结果根目录"
export FANN_DOMAIN=full
export FANN_EPOCHS=400
export FANN_RESUME=1
bash scripts/submit_fann_formal.sh
```

数据/划分、模型超参数、优化设置变化会拒绝恢复；预算可延长。没有产生 last.pt 的初始化失败不能用 --resume，先诊断并使用新输出目录。也不要在原进程仍运行时重复提交，运行锁会拒绝同时写入同一模型目录。

独立冻结评估无需训练数据或重新拟合基底：

```
python scripts/evaluate_fann.py --checkpoint runs/实际结果/POD/best.pt \
  --splits runs/实际准备/splits.json --split test --device cuda --output runs/frozen_pod_test
```

已完成任务恢复到相同预算时，可复用与 best 哈希匹配的测试结果；显式 evaluate_fann.py 使用新目录重新评估。环境、检查点SHA和数据签名写入报告。

## 已做验证与尚未验证

- CPU 完整协议检查：七模型训练/选优/测试、首轮前中断的精确恢复、优化器与调度恢复、早停、几何与内容泄漏防护、冻结 POD 不再拟合、测试标签不影响权重、流式基底和 Slurm 依赖。
- RTX3090 GPU：七模型在合成四几何小数据上通过正式训练/检查点/独立测试链路；这不是 faNN 精度或千万节点显存测试。
- 真实 faNN 单样本：跨块采样导出和全部981万节点/942万单元整域导出均已实际执行，保留壁面距离及真实拓扑。
- X203 集群上的全域前向/反向、稀疏算子和容量/吞吐尚未验证。本机无法访问集群数据或提交集群作业；当前没有多真实几何的本地 faNN 泛化结果。

## 2026-11-19 大点云邻域查询修复

`fann_1793_6.out` 的失败发生在 KDTree 近等距/半径边界回退到全点 `torch.cdist` 的路径，而不是主 KDTree 搜索。981万点×256查询的一块距离矩阵约10GB，且可能超过设备后端的内核启动维度限制。避免该启动错误后，原来的全点回退仍可能有不可接受的二次复杂度。

新实现的候选临时数组按32768条查询分块；大点云不再进入GPU dense参考计算。CPU KDTree候选按输入精度重算距离并作半径过滤，等距候选采用KDTree顺序，不承诺与旧GPU topk的等距排列逐位一致。小点云默认仍保留原参考路径以供兼容性检查；沐曦Slurm脚本默认 `TURBINE_GEO_REFERENCE_MAX_POINTS=0`，彻底关闭dense回退。模型参数布局不变。

缓存schema更新为v2，旧缓存不会被误用；内存缓存增加每个查询模块512MiB的字节上限，超出时只使用设备最近一次缓存与磁盘缓存。可设置 TURBINE_GEO_TREE_QUERY_CHUNK、TURBINE_GEO_MEMORY_CACHE_MB 调整CPU查询临时数组和缓存上限。新邻域策略记录在GeoTransolver检查点配置，策略变化会拒绝精确续训；发生在首次更新前的旧失败任务应从新输出目录重跑，而非 --resume。

这是对当前 cdist 错误的修复，不是千万节点整域激活显存优化或多卡域并行。仍需在X203上重新验证完整前向/反向的容量与吞吐。

## 图模型 2/3/4 显存不足的修复

真实 faNN 公共网格产生 58,075,160 条有向边。一个边×64通道FP32张量恰约13.85GiB，与日志失败申请大小一致；仅约18MiB reserved-but-unallocated 并非显存碎片的证据。降低batch（已经是1）或设置expandable_segments不是这次容量问题的主要解法。

新默认 `FANN_GRAPH_EXECUTION=bounded`：

- **GCN / Graph_Transformer**：CPU构造公共CSR图，owner按节点编号分组。halo分别取模型的5层/3层（small）的完整感受野，所有owner覆盖整个算例且不重复。每分区反向只计owner损失，除以完整算例目标元素总数；全部分区梯度累加完成后才clip并step一次。验证/测试将所有owner预测按原序组回完整节点场。没有改变数据范围或优化器更新次数；浮点累加次序会有微小差异。
- **Graph_UNet**：保留原模型的全局TopK选择、全局邻接平方、池化/反池化和权重。邻接扩展/归一化留在CPU，固定稀疏线性算子的前向和转置反向以边块传输到GPU，避免保留全图E×hidden消息。全局节点特征仍在GPU；CPU邻接扩展可能非常大且较慢，必须先做单步容量测试。不能保证每个几何或训练阶段都装入64GiB GPU/64G主存。

参数：`FANN_GRAPH_OWNER_NODES` 默认32768（首次集群试验建议8192）；`FANN_GRAPH_EDGE_CHUNK` 默认262144。CPU拓扑/halo产物复用目录为 `TURBINE_FANN_GRAPH_CACHE`，默认 `runs/fann_graph_topology_cache_v1`。进程内分区LRU默认1GiB，通过直接训练命令的 `--graph-cache-mb` 调整。没有把完整图边表和所有算例一次性搬到GPU。bounded图模型初始化后和每个完整算例更新后都会保存last.pt，避免长整域epoch在作业超时后丢失整轮；分区中途不保存半次优化器更新。

先只重试失败的三模型，并保留原预处理：

```
git pull --ff-only origin master
export FANN_SPLITS="$PWD/runs/fann_prepared_20261119_202344/splits.json"
export FANN_RUN_ROOT="$PWD/runs/fann_graph_capacity_v1"
export FANN_DOMAIN=full
export FANN_GRAPH_EXECUTION=bounded
export FANN_GRAPH_OWNER_NODES=8192
export FANN_MAX_UPDATES=1
unset FANN_RESUME FANN_STEPS
sbatch --array=2,4 scripts/slurm_fann_array.sh
# Graph_UNet将压力部分转移到主存，首次可单独申请较多主存；仍需实测：
sbatch --array=3 --mem=128G scripts/slurm_fann_array.sh
```

一个完整算例更新后会保存last.pt，并以130退出，日志写明安全停止和GPU峰值；这是主动容量测试的预期退出，不是OOM。若希望降低更小分区的峰值，可继续减小owner数；halo占比/计算和I/O会增加。

只有三者容量测试均通过后，保持相同数据、模型和graph设置续训：

```
unset FANN_MAX_UPDATES
export FANN_RESUME=1
export FANN_EPOCHS=400
sbatch --array=2,4 scripts/slurm_fann_array.sh
sbatch --array=3 --mem=128G scripts/slurm_fann_array.sh
```

新bounded与旧dense配置不混作精确续训；旧任务首次更新前失败的，从新结果目录开始。已有其他模型结果不覆盖。需要最终七模型汇总时，应将匹配同一数据签名的模型结果收集到同一个根目录，再运行compare_fann.py；不同实验不要混用。

验证：小网格CPU/GPU原始/低显存输出与参数梯度对照；全节点owner覆盖；常量稀疏算子转置梯度；正式训练恢复和分区中途SIGTERM不产生半次优化器更新。真实981万节点网格上，8192个owner的首分区在RTX3090上完成前向反向：GCN峰值约2.82GiB，Graph_Transformer约0.69GiB。这个测量只覆盖首分区，不能等同完整算例或X203峰值保证。

后续又在同一个真实整域样本上完成更大范围验证（单样本归一化，仅容量/梯度验证，不是泛化精度）：

| 模型 | 验证范围 | RTX3090峰值显存 | 单次耗时 |
|---|---|---:|---:|
| GCN | 全部owner、完整反向及1次Adam更新；owner=32768 | 11.79GiB | 96.6s |
| Graph_Transformer | 全部owner、完整反向及1次Adam更新；owner=32768 | 2.93GiB | 32.6s |
| Graph_UNet | 全域前向/反向，梯度有限 | 17.78GiB | 393.3s |

Graph_UNet进程峰值主存约35.4GiB；第三层扩展约17.1亿边。CPU全局稀疏计算很慢，因此这条路径只解决容量，不代表适合直接投入400轮或满足三天训练目标。不同几何、训练状态和X203后端仍须实测，128G主存为首次试验的保守申请，非最低硬性要求。浮点累加和接近TopK阈值的节点排序可能与原PyG路径有细微差别，不承诺逐位一致。

若在新根目录只重跑了图模型，可显式复用旧根目录中其他已完成模型的结果：

```
python scripts/compare_fann.py --root "$FANN_RUN_ROOT" \
  --reference-root "$PWD/runs/fann_formal_20261119_202344"
```

仅在新根目录不存在某模型目录时才引用旧结果；新任务failed/interrupted不会被旧成功结果掩盖。比较会检查数据签名、域范围及best.pt哈希，记录每个模型实际结果目录。Slurm汇总也可通过FANN_REFERENCE_ROOT传入旧根目录。


## GeoTransolver 显存修复与节点内多卡

GeoTransolver 的邻居查询优化只处理了空间搜索，并不能避免多层整域激活占满显存。
faNN 正式训练现在默认 `TURBINE_GEO_NODE_CHUNK=32768`、`TURBINE_GEO_ACTIVATION_OFFLOAD=1`：
逐块累积全体节点的切片分子/分母（仅加一次归一化常数），所有块使用同一组全局状态；
局部几何处理、节点 MLP 也分块，并在反向时重算块内激活。保存的激活放在普通 CPU 主存中，避免锁页内存耗尽。
没有裁剪计算域，也没有缩小 hidden/layers/slices 或改变六通道损失。
分块改变浮点加法顺序，因此与旧实现可能有微小舍入差异，不能承诺逐位相同。
设置 `TURBINE_GEO_NODE_CHUNK=0` 可恢复原始未分块路径，仅用于小样例对照。
卸载可通过 `TURBINE_GEO_ACTIVATION_OFFLOAD=0` 禁用，显存占用会增加。
这些选项只在 faNN 路径默认启用，Rotor37 的既有默认行为保持。

新增 `TURBINE_FANN_PIPELINE_GPUS`（默认1），支持 GCN、Graph_UNet、Graph_Transformer、GeoTransolver。
同一进程将网络层放到不同 GPU，自动迁移中间张量，梯度跨设备返回，仍为完整算例一次 Adam 更新。
它分摊单算例的参数和激活显存，**不是 DDP、跨节点域并行或并发微批流水线**，不保证卡数翻倍就加速。
GCN/Graph_Transformer 仍结合 owner+halo；Graph_UNet 仍保留全局 TopK，CPU 邻接扩展的主存和耗时不会因多卡自动下降。
不要同时使用 torchrun 或启动每 GPU 一个进程。

集群先只重试编号6，使用两卡、128G CPU 主存：

```bash
git pull --ff-only origin master
# FANN_SPLITS 沿用已经准备的整域 splits.json
export FANN_DOMAIN=full
export FANN_RUN_ROOT="$PWD/runs/fann_multigpu_capacity_v1"
export TURBINE_FANN_PIPELINE_GPUS=2
export FANN_MAX_UPDATES=1
unset FANN_RESUME FANN_STEPS
sbatch --array=6 scripts/slurm_fann_multigpu_array.sh
```

需要同时验证四个模型时用 `--array=2,3,4,6`；每个数组元素分别申请两卡，不是四个任务共享两卡。
需要四卡时同时修改申请和进程可见卡数：

```bash
export TURBINE_FANN_PIPELINE_GPUS=4
sbatch --array=6 --gres=gpu:4 scripts/slurm_fann_multigpu_array.sh
```

`--max-updates 1` 执行一个完整算例更新，保存 last.pt，以130主动退出。
通过后使用相同运行根目录和数据，`unset FANN_MAX_UPDATES; export FANN_RESUME=1 FANN_EPOCHS=400`，再提交相同脚本。
检查点记录实际设备、激活卸载和块大小。单卡/多卡参数结构相同，可读取原权重与优化器状态；
改变设备布局或切片聚合方式的续训不能承诺逐位复现原轨迹。owner 节点数等既有不可变配置仍需匹配。
若旧作业从未完成一个保存点，没有 last.pt，则使用新的运行目录重新开始。

本机真实约980万节点单样例（单样例归一化，仅容量验证）已经通过 GeoTransolver 单次前向、反向和 Adam 更新：
RTX3090 峰值 GPU 分配约18.21 GiB、CPU 进程峰值约58.65 GiB。
不是 X203 吞吐或最终预测精度证据；集群仍应先做容量检查。

双卡真实整域验证进一步通过连续两次 Adam 更新和一次完整推理：每卡 GPU 峰值约18.21/14.05 GiB，
CPU 进程峰值约59.22 GiB。第二次热缓存算例更新约29.26秒；这些数值不能直接外推 X203 吞吐或400轮耗时。
四模型双卡小样例输出/梯度/优化器对照及正式中断续训、冻结测试均通过；相关回归45项通过。
详情见 `verification/fann_geo_memory_multigpu.json`。其他三种图模型的双卡验证使用小样例，
不要把它描述成三个模型都已经完成真实整域双卡长期训练。


## Slurm 日志位置

所有作业脚本的标准输出和错误均写入项目 `logs/` 目录：
训练为 `logs/fann_ARRAYID_TASKID.out` / `.err`，准备和汇总使用各自作业号。
`logs/.gitkeep` 保证拉取代码后目录存在，提交入口也会在 sbatch 前创建目录。
在集群迁移旧的项目根目录日志可执行：

```bash
mkdir -p logs
find . -maxdepth 1 -type f \( -name '*.out' -o -name '*.err' \) -exec mv -n -t logs -- {} +
```

已提交或正在运行的作业不会因 git pull 自动修改原定日志路径；以上命令可移动现有日志。

## UNet / Transolver-3 吞吐调优

UNet 默认缓存体素中心到原始网格最近节点的映射，目录为
`TURBINE_FANN_UNET_CACHE`（默认 `runs/fann_unet_mapping_cache_v1`）。
缓存只包含几何映射，键包含完整归一化坐标及体素查询位置；不同网格、归一化或分辨率自动分开。
不缓存目标场，不影响几何分组，不改变输入、插值方法、损失或梯度。
首次遇到某个几何仍需建树，后续样本和 epoch 复用。设置该变量为空字符串可禁用。
本机980万节点样例的映射构建约19.18秒，命中约0.12秒；这只是映射阶段，不能作为整轮加速比。

Transolver-3 原默认块大小8192现在可通过
`TURBINE_FANN_TRANSOLVER_CHUNK_SIZE` 调整。64GiB卡可先比较8192与32768，
显存允许时再测65536。所有块仍累积到共同全局切片，每个完整算例仍一次参数更新。
大块可减少 Python、算子启动和重计算调用次数，但会增加瞬时工作区；不能保证所有后端都加速。
`TURBINE_FANN_TRANSOLVER_CHECKPOINT=1` 默认保留重计算；设0可以测关闭重计算的速度与显存，
但本机未验证千万节点无重计算容量，不建议未经单步检查就长期运行。
改变分块会改变浮点求和顺序，续训读取相同参数和 Adam 状态，但不能保证逐位相同。

`TURBINE_FANN_PROFILE_STEPS=3` 会同步设备并记录本次启动前3次参数更新的
数据加载、输入准备、前向、目标与损失、反向及优化器耗时，日志标识为 `step_profile`。
这是诊断计时，会引入同步开销；默认0关闭。加载时间包括 NPZ 解压与完整性检查。
该计时不覆盖启动时的数据检查/归一化，也不包含整轮验证，不能把各步之和当成整轮耗时。

等原作业停止且 `last.pt` 可用后，在原 `FANN_SPLITS` 和 `FANN_RUN_ROOT` 下可测试续训：

```bash
export FANN_RESUME=1
export FANN_MAX_UPDATES=3
export TURBINE_FANN_PROFILE_STEPS=3
export TURBINE_FANN_TRANSOLVER_CHUNK_SIZE=32768
export TURBINE_FANN_TRANSOLVER_CHECKPOINT=1
export TURBINE_FANN_PIPELINE_GPUS=1
sbatch --array=1,5 scripts/slurm_fann_array.sh
```

3次更新后保存并以130退出。首轮缓存可能未热，UNet还应比较后续重复几何的命中时间。
检查日志耗时和显存后，`unset FANN_MAX_UPDATES TURBINE_FANN_PROFILE_STEPS`，保持
`FANN_RESUME=1` 并提交同一脚本继续。不要在旧作业仍占用同一输出目录时重复提交。
更新代码不会改变已经运行的 Python 进程。

编号1、5使用后文的跨算例DDP入口扩展吞吐；当前六种神经网络均支持多节点数据并行。
仅增加 Slurm 申请卡数不能切换并行方式，需要使用相应的DDP脚本。

本机 RTX3090 的百万节点前缀吞吐探针（不是整域训练）：8192与32768分块的第二步耗时约
2.10/0.55秒，峰值GPU分配2.78/2.87 GiB，前两步损失一致。约3.8倍仅适用于该本机探针，
不能作为 X203 整域 epoch 的加速承诺。本机24GiB卡无法容纳这两个配置的千万节点完整模型；
X203需按上述方式实测。验证记录见 `verification/fann_speed_probe.json`。


## 数据不变时，后续不再重复校验

正式训练、续训和 `evaluate_fann.py` 现默认共用
`TURBINE_FANN_STARTUP_CACHE`（默认 `runs/fann_startup_cache_v1`）。
首次登记一次性完成 train/valid/test 全部样本的完整校验、签名和几何隔离检查，
并在读取训练样本时同步计算归一化统计，不再为归一化重读一遍训练集。
统计始终只使用 train，合并顺序与原实现一致；样本内容签名和旧检查点兼容。

后续程序命中登记后：

- 不重新读取全库做校验，不重新拟合归一化；恢复训练优先使用检查点统计。
- 样本实际加载跳过重复全量哈希、节点ID唯一性和大数组有限值检查；仍需解压读取当前算例。
- 只比较路径、文件大小、mtime/ctime、inode 等轻量元数据，包含 splits、所有样本和共享网格。
  新增/替换样本、改变划分或共享网格会使登记失效，自动重新完整登记。
- 活跃作业读取样本时信任启动登记，不再检查文件身份或mtime/ctime等元数据；训练、验证、测试共用此读取路径。数据保持不变由使用者保证。
- 多模型同时启动时使用文件锁，只由一个进程建立登记，其余复用；缓存原子写入。

这基于“准备后的文件保持不变”的使用约定；元数据检查不是重新做全库内容校验。
首次由新版代码启动、且没有登记或可复用检查点时，仍需要一次完整读取。
如所有作业直接一起启动，会等待同一个登记任务；建议事先在 CPU 上完成，避免占着 GPU 等待。

### 复用当前已经验证过的数据

若数据和划分从现有 `last.pt` 保存以来一直未变，可明确指定任意同一划分模型的检查点，
直接导入已有校验清单和训练归一化，不再完整扫描这批数据：

```bash
git pull --ff-only origin master
# FANN_SPLITS 保持为当前同一份 splits.json
export FANN_STARTUP_FROM_CHECKPOINT="/实际路径/某模型/last.pt"
sbatch scripts/slurm_fann_startup.sh
```

这是对“检查点之后数据未变”的显式声明。程序检查划分路径/顺序和原清单签名，但首次导入不会
重新哈希文件内容来证明这一声明。数据已变化或不确定时，不使用该参数，执行一次完整登记。
登记完成后 `unset FANN_STARTUP_FROM_CHECKPOINT`；之后训练、续训和测试自动命中缓存。
若已有登记检测到文件变化，即使传入旧检查点，也强制重新扫描，不能用旧统计覆盖新数据。

本次准备任务不需要 GPU，默认申请8个CPU、64G主存，使用4个读取进程。
增加 CPU 需要配合进程数才会有效：可先比较 `FANN_STARTUP_WORKERS=2` 和4，
再视存储带宽考虑8；不要让多个GPU任务各自启动大量读取进程争用共享存储。
并行读取时每个worker只返回小型统计摘要，按原始样本顺序合并，不把整套字段传回主进程。
同时，每个worker会保留解压后的样本和共享拓扑，因此进程数也会增加主存需求。

### 新数据或主动复查

```bash
unset FANN_STARTUP_FROM_CHECKPOINT
export FANN_STARTUP_FORCE=1
sbatch scripts/slurm_fann_startup.sh
# 提交后清除，避免后续准备任务被强制重扫
unset FANN_STARTUP_FORCE
```

`submit_fann_formal.sh` 已自动加入 CPU 登记依赖，流程为准备→登记→训练→汇总。
手动提交训练数组时先等待上述 CPU 作业成功，或使用 `--dependency=afterok:登记作业号`。
缓存不会消除模型初始化、首次CPU邻接图构建、分区缓存或首个几何的邻居搜索；这些是独立开销。
缓存路径和进度见 `logs/fann_startup_作业号.out`，日志会明确报告 cache hit、登记或失效原因。

相关回归41项通过；详见 `verification/fann_startup_cache.json`。已经运行的进程不会自动采用新缓存，更新对新启动的程序生效。

## 六种神经网络的跨节点 DDP

UNet、GCN、Graph_UNet、Graph_Transformer、Transolver-3 和 GeoTransolver 均有单进程每GPU、跨节点的数据并行入口：
`train_fann_ddp.py` 和 `slurm_fann_ddp.sh`。每张卡处理一个完整算例，随后同步梯度，
因此每卡仍需容纳完整算例。此入口独立于编号2/3/4/6的按层拆分；DDP脚本固定
`TURBINE_FANN_PIPELINE_GPUS=1`，不把这两种并行方式混用。

轻量入口默认每个数组元素申请 **2节点×每节点8卡**，默认数组为1、2、4、5。
CPU卸载较重的3、6使用后文独立脚本，默认每节点8个进程并申请768G主存（节点主存已确认约1TiB）。
两个模型各自申请这份资源，不是共享同一份16卡分配。
一个节点启动一个 torchrun agent，每个agent再启动本节点各GPU的worker。
所有节点必须能读取同一项目、准备数据、启动缓存及结果目录，并使用同一厂商PyTorch环境。
后端默认 `FANN_DDP_BACKEND=nccl`，需要集群的沐曦PyTorch实际提供对应分布式后端和跨节点通信支持。
这里本机测试使用的是 NVIDIA NCCL，不能代替 X203 的互联和后端验证。
不要把 CUDA/NCCL 软件栈替换进现有沐曦环境；接口是否可用以集群安装为准。

### 先做小规模跨节点检查

沿用已登记数据的 FANN_SPLITS；使用独立结果目录，先测两节点、每节点两卡：

```bash
git pull --ff-only origin master
export FANN_RUN_ROOT="$PWD/runs/fann_ddp_probe_01"
export FANN_DDP_GPUS_PER_NODE=2
export FANN_MAX_UPDATES=1
unset FANN_RESUME FANN_STEPS
sbatch --array=1,5 --nodes=2 --gres=gpu:2 --cpus-per-task=8 scripts/slurm_fann_ddp.sh
```

一张卡一份完整算例，一个全局更新最多处理4个不同算例。安全检查停止后所有worker以0退出，
日志标记 `DDP interrupted` / `Safe DDP stop`，`last.pt` 与 `status.json` 保存中断位置。
**这里的0是主动检查停止，不代表正式训练完成；与单卡入口的130约定不同。**
真正算子错误、OOM或通信错误会导致torchrun/srun非零退出。

日志为 `logs/fann_ddp_ARRAYID_TASKID.out` / `.err`。
`MASTER_ADDR` 取第一个分配节点，端口按数组号和任务号生成；可用
`FANN_MASTER_ADDR` / `FANN_MASTER_PORT` 覆盖。网络接口及通信环境变量按集群管理员配置，
脚本不硬编码网卡名称，也不强制关闭高速互联。

### 正式训练或迁移现有单卡检查点

小规模检查通过后，选择结果根目录并恢复8卡/节点：

```bash
unset FANN_MAX_UPDATES
export FANN_DDP_GPUS_PER_NODE=8
export FANN_EPOCHS=400
export FANN_RUN_ROOT="/原来包含UNet或Transolver_3目录的运行根目录"
export FANN_RESUME=1
sbatch --array=1,5 scripts/slurm_fann_ddp.sh
```

原运行目录必须有对应模型的 `last.pt`，已完成过验证的模型还需 `best.pt`。
如果两模型原本属于不同根目录，分别设置目录并分别提交 `--array=1` 和 `--array=5`。
原作业仍运行时不能向相同目录重复提交；rank0持有与单卡入口相同的 `run.lock`。
新建训练使用新目录并取消 FANN_RESUME。不要把小规模检查的结果误认为正式完整结果。

### 训练、指标和恢复语义

- 每轮以 seed+epoch 生成全局算例顺序，每个真实训练算例恰好处理一次。
  不使用会补齐重复样本的采样器，也不丢弃末尾不足一批的数据。
- 一次全局更新最多合并 world_size 个完整算例，梯度按各算例实际节点/通道元素数加权。
  最后一批空闲rank执行极小的零梯度占位前向，不加载额外真实样本，BatchNorm处于eval模式。
- 有效batch随卡数变化：250个训练算例、16卡时，每轮约16次Adam更新，而原单卡是250次。
  学习率默认仍1e-4，不自动按卡数扩大。400轮的收敛轨迹不能当作单卡轨迹的精确加速重放。
- UNet使用本地普通BatchNorm，并按标准DDP行为广播rank0的运行统计；未启用SyncBatchNorm。
  验证/测试前再次同步全部buffer，确保各rank评估相同冻结模型；精度变化仍需实测。
- 验证和测试按算例分配给各rank，不补齐重复样本。汇总全局SSE、绝对误差、真值和/平方和、
  节点数及近壁统计，再计算RMSE/MAE/R²，不能简单平均各rank的RMSE。
- 所有rank用同一个完整验证集指标调整学习率、选最优模型和早停。
  测试指标仅在训练结束、加载验证最优权重之后计算，不参与模型选择。
- 只有rank0写模型/优化器/调度器/历史/状态，检查点保留每rank RNG、全局算例位置、
  未完成epoch的累计损失、world_size和归一化。轻量入口默认每10次全局更新及每轮保存last；重主存入口默认每次更新保存last；
  `FANN_DDP_CHECKPOINT_EVERY` 可调整间隔。
- 权重键不带 `module.`，与现有冻结评估入口兼容。相同world_size、数据和配置下支持批次边界恢复；
  测试验证CPU路径的权重、Adam状态和历史精确一致。迁移单卡检查点或改变world_size时，
  读取原权重和优化器状态，但有效batch改变，不能承诺后续轨迹逐位一致。
- 最终预测按rank/原始样本索引存入 `predictions/{valid,test}/rank_XXXXX/case_XXXXX/`，
  每个文件保留原节点ID和source。`report.json` 仍是全局指标，可用于现有compare_fann汇总。

### 协调停止

Slurm轻量入口提前300秒、重主存入口提前900秒向batch shell发送USR1，脚本写共享停止标记；各rank在完整全局批次或验证算例轮次
边界共同停止并保存，避免直接终止torchrun agent导致其他节点仍在通信中。
手动请求安全停止可在提交节点执行：

```bash
touch "$FANN_RUN_ROOT/UNet/STOP"
# 或 Transolver_3/STOP
```

确认保存并退出后，删除对应STOP文件，再以FANN_RESUME=1提交。
直接scancel、节点故障或强制kill不保证有时间完成新检查点，最多回到最近一次已保存更新。
如果最终冻结评估期间停止，保留训练结束时的last，不会把best权重与末次Adam状态混写。

本机已通过两进程Gloo、双GPU NCCL和双torchrun agent控制流程测试；双agent仍位于同一物理主机，
不构成真实X203跨节点验证。测试使用小样例，未验证X203整域显存、网络吞吐或多节点加速倍数。

原DDP入口的验证见 `verification/fann_ddp.json`；新增四模型的验证见 `verification/fann_graph_ddp.json`。

DDP脚本默认 Transolver 分块32768并保留梯度重计算；如单步显存不足，可将 `TURBINE_FANN_TRANSOLVER_CHUNK_SIZE=8192` 后重试独立容量目录。


## 编号2、3、4、6的多节点数据并行

四个模型保留已有的单卡整域显存优化：GCN/Graph_Transformer 使用 owner+halo，
Graph_UNet 使用CPU全局邻接扩展与分块稀疏运算，GeoTransolver 使用全局切片、重计算和CPU激活卸载。
新增并行只在不同算例之间分配工作；每卡仍处理完整算例，不把同一算例的几何切片当作独立数据样本。

| 模型 | 梯度同步方式 | 入口 |
|---|---|---|
| GCN、Graph_Transformer | 本卡所有owner分块累计完成，再按全局节点数加权归约 | slurm_fann_ddp.sh，编号2、4 |
| Graph_UNet | 保留全局池化和自定义稀疏前反向；完整算例结束后归约梯度 | slurm_fann_ddp_heavy.sh，编号3 |
| GeoTransolver | 原生PyTorch DistributedDataParallel，保留重计算与CPU卸载 | slurm_fann_ddp_heavy.sh，编号6 |

前三种图模型不是逐分块调用原生DDP封装，而是执行数学等价的完整算例数据并行：
各rank先累计本地梯度，按同一个全局分母缩放，然后all_reduce并平均梯度，最后统一裁剪和Adam更新。
这样不同rank即使节点数、owner分区数不同，也只在共同批次边界进行相同次数的通信。
全局均未使用的参数保持grad=None，末尾无算例的rank贡献零梯度，不读取额外真实样本。
owner分区中途收到停止请求时，所有rank共同丢弃未完成全局批次的梯度，恢复批次前RNG后保存，
不会把半个算例的梯度写入Adam状态。全局验证、选模、测试及检查点约定与上节相同。

### 推荐先运行的容量/通信检查

沿用当前FANN_SPLITS和已建立的数据登记，不需要重做全库检查：

```bash
git pull --ff-only origin master
export FANN_RUN_ROOT="$PWD/runs/fann_ddp_2346_probe_01"
export FANN_MAX_UPDATES=1
unset FANN_RESUME FANN_STEPS

# GCN、Graph_Transformer：先两节点、每节点两卡
export FANN_DDP_GPUS_PER_NODE=2
sbatch --array=2,4 --nodes=2 --gres=gpu:2 --cpus-per-task=8 scripts/slurm_fann_ddp.sh

# Graph_UNet、GeoTransolver：两节点、每节点八卡，每节点768G主存
sbatch --array=3,6 scripts/slurm_fann_ddp_heavy.sh
```

重主存入口会用 `FANN_HEAVY_GPUS_PER_NODE`（默认8）覆盖进程数，不受之前残留的
`FANN_DDP_GPUS_PER_NODE=2` 影响。节点已确认 RealMemory=1031774，约1007.6GiB。重主存入口现默认8进程、8卡、32CPU和768G主存。
这是根据本机峰值估计的首次容量检查预算，不是所有几何的保证值。若要缩小检查规模，
设置 FANN_HEAVY_GPUS_PER_NODE=2 并以 `--gres=gpu:2 --cpus-per-task=8 --mem=192G` 提交，
进程数和GPU申请必须同时匹配。

本机约980万节点测试的单进程主存峰值：Graph_UNet约35.4GiB、GeoTransolver约59GiB。
因此不能把8个GeoTransolver进程直接放入128G主存申请：仅进程峰值合计就约472GiB，
还需系统和I/O余量。轻量入口的默认128G不是重主存模型的8进程配置。
在真实X203数据上观察新增日志 `DDP capacity per rank` 的gpu_peak_GiB和process_peak_RSS_GiB，
按节点汇总各rank的进程峰值并留余量，再选择并发数。进程峰值求和是保守估计，不等于同一时刻的系统内存读数。

两种入口均会在一次完整全局更新后主动停止、保存last.pt并以0退出，状态为interrupted。
先确认通信、GPU容量和CPU主存均正常；正式运行取消FANN_MAX_UPDATES，设置FANN_RESUME=1并使用相同输出目录，
或者在原有单卡/按层多卡检查点目录继续。增加GPU进程数会改变全局batch，不能承诺后续轨迹逐位相同。

迁移旧检查点时，保留原 FANN_GRAPH_OWNER_NODES、FANN_GRAPH_EDGE_CHUNK 和图执行模式，
GeoTransolver的查询后端/reference策略也需相同。配置不一致会明确拒绝恢复；不要通过更换数据登记来绕过。
例如旧owner=8192，就在提交前继续设置 `export FANN_GRAPH_OWNER_NODES=8192`。
单节点按层多卡的权重键相同，可以载入；DDP会将每份完整模型放在一个GPU上，TURBINE_FANN_PIPELINE_GPUS固定1。

日志：轻量入口为 `logs/fann_ddp_ARRAYID_TASKID.out/.err`，重主存入口为
`logs/fann_ddp_heavy_ARRAYID_TASKID.out/.err`。已有运行进程不受git pull影响，旧进程停止后再在同一目录续训。

本机双GPU、双torchrun agent（同一物理主机）已通过四模型小样例的完整训练/验证/测试。
额外的GeoTransolver双GPU容量检查使用同一个真实980万节点样例在两个rank上重复，
完成了前反向和一次Adam更新，每卡峰值约18.21GiB、每进程主存约58.93GiB；
这是容量证据，不是两个不同几何的正式DDP训练或X203跨节点吞吐证据。

当前55项相关检查通过；新增验证记录见 `verification/fann_graph_ddp.json`。节点已确认约1007.6GiB主存，因此重主存入口更新为每节点8进程、768G主存；实际空闲及排队情况仍取决于AllocMem等当前分配。

768G申请约占已确认节点配置主存的76.2%，保留约239.6GiB给未分配资源；作业内部余量需以单步峰值确认。8个Graph_UNet进程的本机峰值合计约283GiB，8个GeoTransolver进程约472GiB。已经提交的作业不会因代码更新自动改变资源申请。


## 固定使用当前数据划分

用户确认的默认划分为 `runs/fann_prepared_20261119_202344/splits.json`。
`scripts/fann_defaults.sh` 为正式提交入口、单卡/按层多卡、DDP及CPU登记脚本提供统一默认值，
路径相对于项目根目录展开。未设置 FANN_SPLITS 时自动沿用该文件；显式设置其他路径仍可覆盖。
后续命令无需重复 export，数据登记缓存也继续复用同一划分。
`submit_fann_formal.sh` 若要重新准备新数据，需显式传入空值，
例如 `FANN_SPLITS='' bash scripts/submit_fann_formal.sh`；仅unset会恢复上述固定默认值。
已有运行进程继续使用提交时的环境，更新后的默认值对新提交生效。


## 一次预构建七模型静态缓存

在集群项目根目录更新代码后运行：

```bash
git pull
sbatch scripts/slurm_fann_precompute.sh
```

默认沿用 `runs/fann_prepared_20261119_202344/splits.json`，处理全部七模型，配置为 small、UNet grid=32、POD rank=32、图分区 owner=32768。脚本申请单节点、1张GPU、8个CPU和128G主存。GPU用于生成与后续GPU训练一致的查询坐标及设备相关缓存键；这不是训练任务。本机GPU忙时无需运行GPU验证，直接把全量预构建提交至X203。

可在提交前设置以下可选项；训练时应沿用同一模型配置：

```bash
# 同时准备旧的8192分区与当前32768分区
export FANN_CACHE_OWNER_NODES="8192 32768"
# 可选：只准备指定模型；不设置则全部七个
export FANN_CACHE_MODELS="UNet GeoTransolver"
# 可选：从匹配数据、归一化及rank的旧POD检查点导入基底
export FANN_POD_CACHE_CHECKPOINT=runs/实际旧结果/POD/last.pt
```

`FANN_SCALE`、`FANN_GRID`、`FANN_POD_RANK`、`FANN_GRAPH_OWNER_NODES`控制预构建配置。改变scale可能改变图层数和halo深度，需准备对应变体。上述示例变量只在需要时设置。

输出记录在 `runs/fann_cache_precompute_作业号/`：

- `summary.csv`：各模型耗时、任务数、构建数和命中数。
- `report.json`：状态、配置、共享登记/几何索引/锁等待耗时、总耗时和缓存大小。
- `timings.jsonl`：每个几何或分区的耗时、命中状态和文件路径，保留历次执行记录。
- Slurm日志：`logs/fann_cache_作业号.out`、`.err`。

确认作业成功、report状态为complete后，再提交原有正式训练脚本。后续新开训练自动读取以下共享目录，无需重复运行预构建脚本：

| 模型 | 磁盘缓存 |
|---|---|
| POD | `runs/fann_pod_cache_v1`：仅训练集拟合的基底及节点ID |
| UNet | `runs/fann_unet_mapping_cache_v1`：各唯一几何的最近节点映射 |
| GCN、Graph_Transformer | `runs/fann_graph_topology_cache_v1`：共享拓扑及各owner/halo分区 |
| Graph_UNet | 同一图目录：基础归一化邻接及第一层固定平方邻接 |
| Transolver_3 | 无额外可预构建的模型静态缓存，共用数据登记与归一化 |
| GeoTransolver | `runs/fann_neighbor_cache`：各唯一几何的两类邻域索引 |

另有 `runs/fann_cache_work_v1` 保存几何去重索引及预构建锁。相同拓扑跨模型共享，相同坐标跨样本共享，不按算例重复构建。数据、配置或邻域后端改变会使用对应的新缓存键。运行环境与缓存目录必须由各训练节点共享；Geo缓存包含设备类型及软件版本信息，应在目标X203环境中预构建。CPU模式适合开发检查，不能保证生成的Geo/UNet缓存直接匹配GPU训练。

磁盘需预留空间：980万节点下，Geo两类原始int32索引约1.46GiB/唯一几何，75个唯一几何约110GiB；实际数量和估计值会写入报告。POD冷构建还需要快照临时空间。已完成的缓存以原子方式写入，中断后重新提交会复用完成项；POD单键锁避免并发重复拟合。

预构建不能消除样本读取/解压、缓存读取和设备传输、参数初始化及DDP通信初始化。Graph_UNet更深层池化邻接依赖当前权重，仍需逐次计算。已有运行进程不受更新影响。

本地验证覆盖冷/热缓存、训练自动命中、POD并发构建、旧检查点导入和DDP兼容性。用户要求跳过GPU验证后，余下检查全部隐藏GPU并在CPU运行；完整数据预构建和X203性能以集群报告为准。


## 带时间戳的batch进度日志

七模型普通训练与六模型DDP训练默认每10个batch打印进度，每轮第一个和最后一个batch也打印。使用 `export FANN_LOG_EVERY=5` 可改为每5个batch；该变量必须为正整数。它只控制日志，不改变训练配置或旧检查点的续训兼容性。DDP仅rank 0打印常规进度，一个batch代表一次全局优化更新。普通训练一个batch对应一个完整算例，图模型内部owner分块不计为batch。

训练入口、模型初始化、图执行和epoch汇总日志使用带UTC偏移的北京时间，例如 `[2026-11-20T18:30:00+08:00]`。普通batch日志包含epoch、batch、全局step、当前MSE、累计训练MSE、学习率、batch耗时和当前模型所用GPU的已分配显存之和。DDP保留全局batch loss及rank0步骤耗时。耗时不增加额外GPU同步；单步很慢时仍需等待该batch完成才打印，无定时心跳。验证完成后继续打印epoch汇总。

GPU繁忙时只需在集群 `git pull` 后新启动或安全续训；已运行进程不会自动改变日志行为。


## GeoTransolver：2节点BF16 AMP对比

```bash
git pull
bash scripts/submit_fann_geo_amp.sh
```

提交2节点×8卡，每节点768G主存、32 CPU，启用 `FANN_AMP=bf16`，默认400轮，每个batch打印时间。输出默认新建 `runs/fann_geo_bf16_2nodes_时间/GeoTransolver/`，仍沿用既定FANN_SPLITS和已有几何缓存。可在命令末尾追加Slurm参数，例如 `bash scripts/submit_fann_geo_amp.sh --partition=实际分区`。

短程吞吐对比可先 `export FANN_MAX_UPDATES=20`，以20次全局更新为界安全保存并停止；正式400轮前 `unset FANN_MAX_UPDATES`。若先前已设置FANN_RUN_ROOT，请改为独立空目录，避免覆盖FP32实验。新脚本强制新开训练；续训BF16实验应使用原DDP脚本并设置FANN_AMP=bf16、FANN_RESUME=1及对应FANN_RUN_ROOT。

BF16 autocast覆盖Geo网络前向，参数/梯度/Adam保持FP32，坐标和邻域缓存策略不变，输出转FP32后计算损失；分块切片的跨块累计保持FP32。检查点记录amp配置，拒绝把FP32检查点当作BF16精确续训。评估和冻结检查点加载沿用其精度设置。当前仅支持GeoTransolver的off/bf16，不支持FP16开关。

FP32对照继续使用原DDP脚本并 `export FANN_AMP=off`，其他设置、节点数、训练数据、随机种子、chunk及CPU卸载应相同。对比预热后连续batch的rank0_seconds及验证误差，不用首步外推。BF16暂未在X203验证，须关注算子支持、非有限值和验证精度；本地验证只使用CPU，未占用GPU。


## 移除训练过程中的文件元数据核查

针对集群已确认准备数据内容不变、但活跃作业触发Registered faNN data changed的情况，SampleStore不再在读取前后比较文件元数据，包括缓存命中及重新加载。所有使用该读取器的模型、普通/DDP训练、验证及测试均生效。启动阶段仍检查登记是否可复用，实际文件缺失或无法读取仍正常报错。无需重建几何、POD、图或邻域缓存。

集群git pull后，设置原FANN_RUN_ROOT、FANN_RESUME=1、FANN_AMP=off，按原2节点16卡配置重新提交scripts/slurm_fann_ddp.sh的array=6作业，可从1827的last.pt继续。不要使用强制新开训练的submit_fann_geo_amp.sh恢复该FP32作业。


## DDP分阶段与跨rank计时

默认关闭。提交前设置 `export FANN_PROFILE_STEPS=20`，会对本次启动后前20个全局batch计时（续训也重新计数）；可结合 `FANN_MAX_UPDATES=20` 做短程测试。此开关不改变检查点训练配置。开启后，检查点和验证/最终测试阶段也计时；所有rank必须使用相同开关。

```bash
export FANN_PROFILE_STEPS=20
export FANN_MAX_UPDATES=20
export FANN_LOG_EVERY=1
bash scripts/submit_fann_geo_amp.sh
```

rank0打印带时间戳的DDP timing日志，并在结果模型目录追加 `timings.jsonl`。每条记录包含各rank主机、阶段秒数、是否实际处理样本、显存已分配量及进程生命周期峰值；summary给出各阶段min/mean/max及最慢rank编号。train_step记录step/epoch、load_sample、prepare_input、forward、target_loss_or_idle、loss_collectives、backward_with_ddp_sync、clip_optimizer和step_total等。

原生DDP的梯度通信与反向重叠，因此backward_with_ddp_sync包含反向计算、梯度通信和等待，不能解释成纯GPU计算时间。GCN/Graph_Transformer分区反向仍记录为input_forward_backward_combined，随后独立记录manual_gradient_sync；Graph_UNet输入与前向合并计入forward。CPU激活卸载/恢复和梯度重计算包含在相应forward/backward内。短尾batch的空闲rank仍执行必要DDP操作，需结合active标记解释，不能把其阶段耗时与完整样本直接比较。

checkpoint事件包含RNG收集、rank0序列化写盘及各rank屏障等待；evaluation事件包含该rank的加载/推理/指标统计及最终集体通信，最终评估还包含预测写盘。step_total不包含后续性能报告收集和检查点；它包含本步常规日志及记账开销。不同阶段的最大值不能相加冒充某一rank总耗时，使用step_total的max判断慢rank。

计时会在阶段边界同步GPU，并额外收集各rank记录，可能改变流水重叠和吞吐，仅用于诊断。20步后的常规训练不再添加batch阶段同步；若仍开启计时，后续检查点/验证事件继续记录。正式吞吐复测应 `unset FANN_PROFILE_STEPS`，正式完整训练还需 `unset FANN_MAX_UPDATES`。CPU路径已验证，X203性能须集群实测。


## Geo：区分算例、缓存、节点和CPU卸载差异

```bash
git pull
bash scripts/submit_fann_geo_diagnostics.sh
```

提交两个串行依赖的独立诊断作业，覆盖以下三组配置，避免作业间相互争抢资源：

1. 双节点×8卡，BF16、CPU卸载开启。每个rank固定权重执行10次前向/反向：节点0按A/A/B/B/A、节点1按B/B/A/A/B，再交换两节点分配重复。没有优化器更新，不写训练检查点。
2. 单节点单卡、卸载开启，执行A/A/B/B/A。
3. 在第2项同一作业、同一节点与GPU分配中启动新进程，关闭卸载，执行相同序列。仅最后这项可能因完整激活超出显存而OOM，错误写入该诊断目录及Slurm日志，不影响训练结果。

默认自动从训练集中选取坐标确实不同的两个算例，A从第0例开始。可设置FANN_DIAG_CASE_A和FANN_DIAG_CASE_B指定训练集索引。选择过程只比较坐标，不以目标场难易选择样本。GPU查询缓存按既有目录读取，不清除磁盘或OS缓存；每次清空SampleStore的单样本LRU，使各步都测量样本读取。A/A用于观察立即重复的几何设备缓存，最后返回A用于观察淘汰和重读。

默认新建 `runs/fann_geo_diag_suite_时间/`，各子目录包含protocol.json、completion.json、timings.jsonl和summary.csv；失败时记录failure_rank_编号.json。Slurm输出在logs/fann_geo_diag_作业号.out/.err。completion.json校验运行前后参数SHA256完全一致；诊断不改变原训练权重或优化器。CSV列出每个rank的样本、节点、loss、加载/前向/反向耗时、邻域查询耗时、缓存命中/构建次数及显存/主存。

可先设置 `export FANN_DIAG_CHECKPOINT=原结果目录/GeoTransolver/last.pt`。诊断从该只读检查点获取数据清单、归一化、架构和参数，显式使用既有数据、不执行启动全库登记；BF16与卸载设置由诊断参数控制。未指定检查点则使用既有启动登记及seed42初始化，登记失效时仍可能全量扫描。检查点应与当前准备数据匹配。不要将FANN_DIAG_ROOT设为已有结果目录。

单卡开启/关闭卸载在同一Slurm分配中顺序执行，使用独立进程和独立结果子目录，避免比较时更换节点；第二次执行仍可能受OS/磁盘热缓存影响，应比较序列中的重复步骤。套件脚本末尾可传partition等公共参数，不要传nodes/gres等覆盖实验设计的资源参数。

训练自身也已补充跟踪：开启FANN_PROFILE_STEPS后，train_step各rank明细记录source、geo.queries及host_RSS_GiB。queries包含module、radius、邻居数、完整几何缓存键、各级缓存命中、构建次数、缓存get耗时和构建/写盘耗时。query总时间还包含坐标传回CPU、哈希、索引转换/传输和设备缓存相等检查，不能直接等同磁盘读时间。forward_query_seconds是forward的子集，不应重复相加。backward中的重计算查询另行标记。

判读：慢速跟着样本/几何走，查缓存与几何；跟着节点走，查节点资源；A首次慢、立即重复快，查缓存；16进程很慢、单进程快，查资源争用；卸载关闭且显存可容纳时明显变快，查CPU卸载路径。计时含GPU同步开销，最终吞吐仍需关闭诊断后测量。固定序列可能放大共享缓存争用，是诊断压力测试，不等同随机训练吞吐。


## 关闭CPU卸载的2节点BF16基线

```bash
git pull
bash scripts/submit_fann_geo_no_offload.sh
```

固定GeoTransolver small、BF16、2节点×8卡、节点块32768、CPU激活卸载关闭；仍保留原有梯度重计算与磁盘缓存策略，每节点32 CPU、768G主存。默认新开独立结果目录 `runs/fann_geo_bf16_no_offload_2nodes_时间/GeoTransolver/`，忽略可能残留的FANN_RUN_ROOT；可用FANN_BASELINE_RUN_ROOT指定新的独立目录。保留既定FANN_SPLITS。

默认只做20次全局真实训练更新：250个训练算例、16卡时，为完整1轮含验证加第2轮4个batch，随后保存last.pt安全停止。启动日志应显示amp=bf16、geo_activation_offload=False、geo_node_chunk=32768。无需重新预计算缓存。

该基线强制关闭FANN_PROFILE_STEPS和TURBINE_FANN_PROFILE_STEPS，避免诊断同步开销影响吞吐；仍逐batch打印时间戳、loss及rank0_seconds，结束汇报各rank显存和主存峰值。原生DDP通信与反向仍正常执行。主存申请暂保持不变，以便只改变卸载设置。

若20步无OOM且性能稳定，再用新目录运行完整预算：

```bash
export FANN_BASELINE_MAX_UPDATES=""
bash scripts/submit_fann_geo_no_offload.sh
```

本脚本是新开训练入口；续训已保存的无卸载基线时用原slurm_fann_ddp.sh，设置对应FANN_RUN_ROOT、FANN_RESUME=1、FANN_AMP=bf16、TURBINE_GEO_ACTIVATION_OFFLOAD=0、TURBINE_GEO_NODE_CHUNK=32768，并取消FANN_MAX_UPDATES/FANN_PROFILE_STEPS。不要直接用本入口续训，因为它通过AMP提交器固定新开模式。单卡两例无卸载诊断可运行不保证所有训练算例均不OOM，先以此20步基线确认容量。


## GeoTransolver几何优先采样与有界缓存实验

```bash
git pull
bash scripts/submit_fann_geo_affinity.sh
```

新入口采用2节点×8卡、BF16、CPU卸载关闭、节点块32768和4GiB/邻域模块/进程的CPU LRU，默认20次真实训练更新。同步诊断计时关闭，输出独立的 `runs/fann_geo_affinity_2nodes_时间/GeoTransolver/`。它与1965的4GiB缓存实验主要区别是采样顺序；不要期待同一步loss与随机采样相同。可用FANN_AFFINITY_RUN_ROOT指定新的目录、FANN_AFFINITY_CACHE_MB调整缓存、FANN_BASELINE_MAX_UPDATES调整更新上限。显式空FANN_BASELINE_MAX_UPDATES请求完整epoch预算。

采样方式：以完整原始坐标哈希确定几何，复用预计算的geometry索引；不依赖GeometryNumber相同的假设。根据训练集中每种几何的样本数量，将几何组均衡分配到固定优先rank。每轮打乱各rank内的几何组顺序及组内工况顺序，组内样本连续访问以争取设备缓存命中。若某rank的队列先耗尽，则从剩余最长队列转移样本，确保除最后一批外所有rank都工作。每个样本每轮仅使用一次，不补齐重复、不丢弃尾部，全局batch数量仍为ceil(N/world)，损失仍按节点元素数加权。

这不是严格的固定rank分区，也不保证各几何实际计算时间相等；均衡的是样本数量。较大的几何组可能部分转移给其他rank。更新批次组成发生变化，属于新的训练实验，应重新检查验证精度。默认随机采样保持不变，可通过torchrun入口 --sampler geometry_affinity 或环境变量FANN_DDP_SAMPLER=geometry_affinity显式启用，仅支持GeoTransolver DDP。

启动时输出geometry affinity enabled。结果目录中的geometry_affinity.json记录优先rank和几何分组；sampling/rank_编号.jsonl逐步记录样本路径、几何键、优先rank、实际rank、active和各级邻域cache_delta。在检查点及结束时写入，避免逐batch额外同步GPU。device_cache_hits包含同一前向内的复用，解释时应结合连续样本几何；空闲rank的dummy forward也可能产生缓存事件，以active字段区分。

已有几何索引直接读取；不存在时会首次读取各样本坐标建立索引，不读取目标场作几何分组。几何键和采样schema进入检查点，支持同一world_size下的中断精确恢复；不允许把随机采样检查点作为几何模式精确续训。更换world_size会重分配优先rank，延用原有非精确轨迹提示。恢复本实验仍使用slurm_fann_ddp.sh并显式设置原FANN_RUN_ROOT、FANN_RESUME=1、FANN_AMP=bf16、FANN_DDP_SAMPLER=geometry_affinity、TURBINE_GEO_ACTIVATION_OFFLOAD=0及匹配缓存配置。

本地仅做CPU验证：不重不漏、批次数与负载平衡、极端几何组、调度可重现、BF16两进程短尾训练、中断前后参数/优化器/调度器一致，以及原随机模式梯度参考。集群收益需以实际batch耗时和sampling记录判定。


## 单节点8卡吞吐对照

```bash
git pull
bash scripts/submit_fann_geo_single_node.sh
```

固定单节点8卡、BF16、CPU卸载关闭、随机采样、512MiB/模块/进程的CPU LRU、32768节点分块、4 CPU线程/进程；保留梯度重计算，每节点仍申请32 CPU、768G主存。脚本显式覆盖可能残留的几何采样、4GiB缓存、诊断计时和resume设置。独立结果目录为 `runs/fann_geo_bf16_no_offload_1node_时间/GeoTransolver/`；可用FANN_SINGLE_NODE_RUN_ROOT指定新目录。

默认40步：250例、8卡时一轮32步，40步处理250+64=314例，与1964的16卡20步相同，并都完成一次验证。开始状态均为seed42新模型，但全局batch从16变为8，优化器更新数不同，不应比较逐步loss是否相等。维持原学习率不自动缩放。比较每秒算例数、同314例的总墙钟时间及GPU资源消耗，不能只比较epoch步数。

若只做更短测试，可设置FANN_SINGLE_NODE_MAX_UPDATES=20，此时只训练160例，尚未完成一轮，不能与双节点20步的总时间直接比较。启动日志应显示world_size=8、amp=bf16、geo_activation_offload=False、geo_node_chunk=32768，并且不出现geometry affinity enabled。

判断扩展：设单节点稳定步骤平均t8、双节点平均t16，则吞吐扩展比约2*t8/t16，双节点相对线性扩展效率约t8/t16；尾批、验证和保存时间另外计入总工期。最终是否值得增加节点，应兼顾工期和GPU小时。缓存/节点运行状态仍可能不同，必要时重复对照。


## 同节点1/2/4/8卡扩展诊断

```bash
git pull
bash scripts/submit_fann_geo_scaling.sh
```

一次申请1节点8卡、32 CPU、768G主存，在同一节点分配中依次运行1、2、4、8个训练进程，其余卡在相应阶段空闲。各配置独立新开seed42训练，固定BF16、关闭CPU卸载、随机采样、512MiB CPU缓存、32768节点分块和每进程4线程。OpenMP、MKL和OpenBLAS线程均明确限制为4。不会同时运行各组，也不覆盖原结果。每组结束后进程退出，再启动下一组。

默认各组均使用同一首轮随机样本序列的前80例：16例预热，64例计量。对应1卡80步、2卡40步、4卡20步、8卡10步。计量样本数相同；全局batch和参数更新轨迹不同，不能按同一步loss比较。预算总和必须不超过训练集大小，并且预热/计量预算均须被各卡数整除，以避免跨轮或尾批改变对照样本。此诊断不运行验证，关闭中途定期检查点，仍保存初始和结束状态；计量排除启动、预热及最终保存时间。

输出默认 `runs/fann_geo_scaling_作业号/`：report.json记录状态和结果，summary.csv列出卡数、计量例数、实际墙钟耗时、cases/s、单卡相对加速比和效率。计量墙钟区间从最后一个预热batch日志到最后一个计量batch日志，包含正常训练步骤间开销；mean_rank0_seconds为日志步骤耗时的平均值。各gpus_卡数子目录包含console.log、steps.json及GeoTransolver检查点。Slurm日志为logs/fann_geo_scaling_作业号.out/.err。

自定义测量量：提交前设置FANN_SCALING_WARMUP_CASES和FANN_SCALING_MEASURE_CASES，默认16/64，必须为正且被8整除。FANN_SCALING_OUTPUT可指定新的输出目录；目录已存在会拒绝覆盖。需要更短测试可用16/32，但8卡计量仅4步，统计稳定性较弱。套件逐步输出日志，不需要全量重新预计算缓存。

相对加速比=该卡数cases/s÷单卡cases/s；效率=加速比÷卡数。若4/8卡吞吐明显偏离线性，应结合console.log中的主存/显存峰值检查节点内争用和DDP；该实验本身不把纯GPU计算与通信分离。后运行的配置可能受益于OS文件缓存，同一节点也可能存在其他资源负载，必要时重复或用Python入口 --gpu-counts 8 4 2 1反序验证（保持预算满足整除）。此脚本默认关闭同步阶段诊断，避免计时改变重叠行为。

CPU小样例已验证1/2进程预算与报告逻辑；未在本机运行GPU诊断。集群实测后提供summary.csv、report.json和Slurm .out即可比较扩展效率。


## 四节点并行扩展诊断

```bash
git pull
bash scripts/submit_fann_geo_scaling_parallel.sh
```

无依赖提交四个独立作业，每个申请一个独占8卡节点，在其中分别启动1、2、4、8个训练进程。其余卡保持空闲，以隔离同节点CPU/GPU资源争用；因此占用四个8卡节点，实际15卡参与训练。资源充足时各组并发，整体工期接近最慢一组；排队时间另计。原同节点顺序入口保持不变。

预热/计量预算和训练设置与顺序版相同，默认16例预热+64例计量。提交器创建独立 `runs/fann_geo_scaling_parallel_时间/`，jobs.tsv记录卡数和作业号，parts/1gpu、2gpu、4gpu、8gpu分别保存该组report.json和内部gpus_卡数训练目录。FANN_SCALING_PARALLEL_ROOT可指定新的套件目录，不允许覆盖。

完成后在项目根目录汇总（仅标准库，可在登录节点执行）：

```bash
python3 scripts/summarize_fann_geo_scaling_parallel.py --root runs/fann_geo_scaling_parallel_实际时间
```

生成顶层summary.csv和report.json，重算相对单卡加速比与扩展效率。可在部分完成时运行，状态会标为incomplete并列出尚未完成的卡数。汇总检查数据签名、预热/计量预算、Torch版本及GPU型号一致，拒绝混合不一致结果。失败组请查看对应日志，汇总不会把未完成组当成成功。

独占节点避免组间CPU/GPU争用，但不同节点性能可能不同，共享存储仍有并发读压力，不能将并行版所有差异都解释为纯计算扩展。报告保留host、gpu_name、job_id，必要时以原同节点顺序版复核。公共partition等参数可追加到提交器，不要覆盖exclusive、gres、nodes等隔离资源设置。


## 对照1970的2节点16卡固定配置吞吐测试

```bash
git pull
# 可选：指向已完成的1970串行报告，自动计算8→16卡扩展
export FANN_SCALING_REFERENCE_REPORT=runs/fann_geo_scaling_1970/report.json
bash scripts/submit_fann_geo_scaling_2nodes.sh
```

采用2节点×8卡、每节点32 CPU和768G主存，OpenMP/MKL/OpenBLAS均4线程/进程。复用run_fann_geo_scaling.py的全部训练设置：BF16、CPU卸载关闭、随机采样、512MiB缓存、32768节点分块、seed42、原学习率、独立初始权重，关闭同步阶段诊断及中途检查点。预热16例、计量64例，与1970覆盖相同随机样本区间；16卡时共5次全局更新，其中1次预热、4次计量，之后安全保存退出，不运行验证。全局batch不同会改变参数更新轨迹，不能据此比较逐步loss。

Python控制器直接在batch进程运行，由它启动srun的两个节点代理，避免嵌套srun资源占用。每节点仅一个torchrun代理，每代理8个worker，总world_size应为16；master地址取首节点，端口按作业号区分。

结果默认 `runs/fann_geo_scaling16_作业号/`，包含report.json、summary.csv及gpus_16中的console.log、steps.json和GeoTransolver检查点。Slurm日志为logs/fann_geo_scaling16_作业号.out/.err。FANN_SCALING_2NODE_OUTPUT可指定新的独立目录。无需设置旧FANN_RUN_ROOT或resume变量，也不重新预计算几何缓存。

参考报告可省略，届时只报告16卡吞吐，之后手动与1970比较。若提供参考，必须数据签名、预算、Torch版本和GPU型号匹配；report中的speedup_vs_8gpu为16卡吞吐÷参考8卡吞吐，efficiency_vs_8gpu为该加速比÷2。若参考还含1卡结果，也报告相对单卡的总加速比。

默认仅4个计量batch，适合先筛查跨节点退化；更长测量应给单/双节点都设置相同更大预算，并用新单节点报告比较，不能与不同预算的1970自动混算。已在CPU上模拟双Slurm代理、真实torchrun会合与训练，尚未在本机GPU或X203跨节点验证。
