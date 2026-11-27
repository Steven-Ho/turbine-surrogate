# CR15 六个基线的正式训练

复用 `runs/cr15_prepared_v1`：199 train / 23 valid / 26 test / 2 baseline，只有 A 级算例；五场 u/v/w/p/T；固定六维边界条件。所有模型输入真实算例几何，禁止把求解后的性能诊断作为输入。

```bash
git pull
bash scripts/submit_cr15_models.sh --dry-run
bash scripts/submit_cr15_models.sh
```

提交六个独立容量检查及六个 afterok 正式训练作业。六组可由 Slurm 并行调度。容量检查对最大训练算例执行一次完整前向、反向和优化更新（1 卡、batch=1），不跑验证和测试；POD 在此阶段建立可复用的训练基。容量检查通过只证明最大节点算例的该次运行可行，不是所有几何峰值显存的保证。失败时对应长训不会运行，应检查日志并取消不再需要的依赖作业。

| 模型 | 单节点 GPU 数 | 规模 | 精度 | CPU/节点 | 主存/节点 | 默认时限 |
|---|---:|---|---|---:|---:|---|
| POD | 1 | base MLP / rank32 | FP32 | 8 | 64 GiB | 12h |
| UNet | 2 | small / 64³ | FP32 | 16 | 64 GiB | 12h |
| GCN | 4 | small | FP32 | 32 | 128 GiB | 24h |
| Graph_UNet | 4 | small | FP32 | 32 | 128 GiB | 24h |
| Graph_Transformer | 4 | small | FP32 | 32 | 128 GiB | 24h |
| Transolver_3 | 8 | base | BF16 | 32 | 128 GiB | 12h |

资源配置是待集群验证的起点，并非实测最优。这里按算法开销选择不同规模，属于资源适配基线，不是等参数量的架构对比。GeoTransolver 2节点16卡耗时2h23的反馈用于避免给所有基线都申请多节点，但不同算法不能据此直接推算耗时。DDP 每卡处理一个完整算例，减少卡数通过梯度累积保持全局 batch<=16，尾批为7例；增加 DDP 卡数不会分摊单个算例的显存。图模型使用下述有界执行节省单卡显存。

统一 Adam 1e-4、betas=(.9,.999)，节点加权标准化 MSE、clip1、Plateau patience20 / rel-threshold .001 / cooldown5、early stop60。目标400轮，不是强制跑满；只根据验证选最优，最终冻结 best 后输出 valid/test/baseline 的五场 RMSE、MAE、R² 和原节点预测。UNet 使用 rank-local BatchNorm，不承诺改变卡数后的数值轨迹完全相同。

```bash
# 先短训5轮，目标总轮数仍为400，之后从同一目录续训
bash scripts/submit_cr15_models.sh --epoch-budget 5 --run-root runs/cr15_baselines_short
bash scripts/submit_cr15_models.sh --resume --run-root runs/cr15_baselines_short
# 单个模型或资源覆盖
bash scripts/submit_cr15_models.sh --models GCN Graph_Transformer --gpus 8 --time-limit 24:00:00
# 只提交单模型
bash scripts/submit_cr15_models.sh --models Transolver_3
```

可通过 `--prepared`、`--partition`、`--account` 指定路径和分区。每次提交清除遗留 CR15 实验变量，防止继承 GeoTransolver 的 epoch-budget/max-updates/resume 等设置。改变规模、精度、网格或缓存路径需要与保存配置一致，不能任意改变后续训。

输出：`runs/cr15_six_models_<时间>/<模型>/`，含 configuration.json、history.json、last.pt、best.pt、report.json、predictions/{valid,test,baseline}；所有 Slurm .out/.err 在 `logs/cr15_<模型>_{capacity,train}_<jobid>.*`；时间戳日志每次优化更新打印一次，仅 rank0。共享磁盘缓存 `runs/cr15_model_cache_v1/` 可跨实验复用，缓存只由准备数据身份、训练数据及几何构造；不重新扫描原始 MAP 文件。

## 模型表示与边界

- POD：先把各训练场按最近点投影到第一个训练算例的参考网格，再以磁盘流式快照法计算均值/32模态；MLP 输入当前几何投影后256段平均坐标与6条件，输出系数；重建后按最近点返回当前算例原节点。所有目标只在训练集拟合基，valid/test 只用于评价。不同网格的映射是近似表示，会产生投影误差，不等于严格节点一一对应的 POD。
- UNet：当前几何、条件和距最近网格点距离嵌入 64³ Cartesian 网格，再回采样到全部原始节点；存在体素分辨率和插值误差，输出仍为全域五场。
- 图模型：从 MAP 的 i-fastest block_shapes / block_offsets 生成实际 HEXA_8 边；对精确重合节点加连接并保留原输出编号。没有虚构周期性、壁面或冷却孔标记；非重合块接口和周期边界未自动补连，属于当前基线限制。
- GCN / Graph_Transformer：逐 owner 节点块、包含完整层数感受野的 halo，checkpoint 前向，再拼接每个 owner 一次的全节点预测。没有独立裁块后丢弃全场目标。
- Graph_UNet：保留全局 TopK pooling，复用已有 CPU 稀疏邻接扩展及有界稀疏乘法；CPU 常量图计算不是激活卸载，仍可能成为瓶颈。
- Transolver-3：9输入通道、5输出；真实坐标+6边界作为节点特征，使用原全局切片注意力和分块/checkpoint实现。

UNet 和图模型默认 FP32；Transolver-3 默认 BF16（CPU 小样例已通过，X203 支持由容量作业实测）。UNet 的 CPU BF16 卷积检查耗时过长，因此不以它作为默认精度。

本机验证使用 CPU 合成多块/变节点数数据；没有宣称在本机完整 CR15 数据或 X203 上完成六模型容量实测。

## GeoTransolver 大学习率对照

```bash
bash scripts/submit_cr15_geo_lr2_long.sh
```

从头训练，独占2节点×8卡，base/BF16/global batch16，初始LR=2e-4，目标400轮；不设置100轮或更新次数的中途停止预算。保持原Adam betas、随机种子42、Plateau和early stop60；若触发早停，可能在400轮前结束。默认独立目录 `runs/cr15_geo_base_lr2_2nodes_400_<时间>_<PID>/GeoTransolver/`，日志 `logs/cr15_geo_lr2_long_<jobid>.out/.err`。可用 `CR15_PREPARED` 指定准备数据，用 `CR15_LR2_RUN_ROOT` 指定新实验目录；已有检查点时拒绝覆盖。默认4小时Slurm时限，可追加 `--time=08:00:00` 等调度参数。不会继承旧实验的resume、max-updates、100轮预算或其他模型名称。
