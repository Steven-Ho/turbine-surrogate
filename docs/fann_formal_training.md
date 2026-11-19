# faNN 正式训练与整域数据入口

本流程补齐了监督学习实验协议：按 epoch 训练、完整验证、验证选优、学习率调度、早停、原子检查点、优化器/随机数/数据位置恢复、冻结后的独立测试和七模型汇总。默认申请每模型一张 GPU，七个数组任务各自训练一个模型。

**整域数据模式已实现，但不是多 GPU 域并行实现。** `FANN_DOMAIN=full` 会使用所有 9,806,346 节点、9,429,120 六面体，不裁成 512 单元，也不静默降采样。是否能在 X203 单卡 64 GiB 上完成各架构的前向/反向，必须由集群实测确认。GCN/Graph_UNet/Graph_Transformer 的全图中间张量、GeoTransolver 邻域与注意力激活可能超出单卡容量；OOM 会将任务标记为 failed，不会换模型、换 CPU 或把整域结果改称局部结果。本轮未实现跨 GPU 图分区/halo 和跨 GPU 全局切片通信。

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

脚本返回准备、数组和汇总作业号，以及固定的结果根目录；自动依赖为：准备成功 → 七训练任务 → 无论任务成功与否都执行汇总。准备失败时训练不运行。汇总不会把缺失或失败模型当作成功。

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
- 测试集的内容完整性/几何归属在开始时核验，但测试指标不参与训练、学习率、选模或早停。改变测试标签不会改变训练所得最佳权重（有回归检查）。

## 七模型的整域处理差别

- POD：采用磁盘 memmap 和快照 Gram 矩阵拟合训练集基底，不把所有快照堆进 RAM；默认32模态。整域约295个训练上限样本需要约77GB的快照/基底临时空间（实际按训练集大小计算），额外留20%余量。基底通过公共节点ID对应回所有节点。几何输入先按公共节点顺序池化为256个区间的坐标/壁面距离描述符，避免把约4000万维几何直接输入全连接层；该表示是独立 POD 基线，需评估表达能力。推理和续训直接重载保存基底，不重新拟合。
- UNet：完整节点几何映射到规则体素网格，三线性解码回全部原始节点并计算损失；网格分辨率默认32，可通过 FANN_GRID 设置64/128等。默认分辨率仅为可运行基线，不保证解析近壁细节；该映射不是保守重网格化。原始节点输出是全域，但内部表示有分辨率瓶颈。
- 三种图网络：使用完整真实六面体边，不随机删点后沿用旧连接；当前全图位于单卡，暂不具备跨卡图分区。
- Transolver-3：整域按8192节点分块，所有块贡献共同物理切片，保持全域耦合；训练使用官方梯度检查点，但并不自动跨GPU。
- GeoTransolver：采用已移植的 portable_torch / KDTree 查询路径，以实际xyz/壁面距离为局部输入、几何点集和全局工况为上下文。当前整域仍在单卡执行；与NVIDIA Warp版本的整域性能不能直接等同。GPU缓存默认目录由 TURBINE_GEO_CACHE_DIR 控制，千万节点邻域缓存可能很大，应计入磁盘规划。

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
