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

- POD：采用磁盘 memmap 和快照 Gram 矩阵拟合训练集基底，不把所有快照堆进 RAM；默认32模态。整域约295个训练上限样本需要约77GB的快照/基底临时空间（实际按训练集大小计算），额外留20%余量。基底通过公共节点ID对应回所有节点。几何输入先按公共节点顺序池化为256个区间的坐标/壁面距离描述符，避免把约4000万维几何直接输入全连接层；该表示是独立 POD 基线，需评估表达能力。推理和续训直接重载保存基底，不重新拟合。
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

编号1、5当前尚未接入节点内按层多卡脚本。仅多申请 GPU 不会自动加速它们。
若单步容量已满足，跨算例的数据并行是进一步提升 epoch 吞吐的方向；这需要实现梯度同步、
跨rank指标归约和一致的保存/恢复协议，并评估有效batch变化，不能直接套用现有四模型的按层拆分。

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
- 活跃作业读取前后也检查文件身份；检测到登记后文件被改动会停止，避免混入新数据。
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
