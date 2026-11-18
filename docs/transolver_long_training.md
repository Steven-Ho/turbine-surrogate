# Rotor37 长训练方案

目标：在统一原始测试节点上，让 Transolver-3 / GeoTransolver 的 Pressure 预测精度显著超过 POD。增加 epoch 是基于当前未收敛曲线的实验策略，不能保证达到目标。

## 训练协议

- 保留原 800 / 100 / 100 工况划分、训练集归一化、参考几何输入和全部 29773 节点，不重新抽样。只用 800 个训练工况更新模型，100 个验证工况调度与选检查点。
- 从原 20 轮最后检查点继续，恢复模型、Adam 动量、数据顺序和随机数状态；新实验目录 `runs/transolver_long`，原 20 轮实验保持不变。原验证最优模型复制为初始 best，避免续训恶化丢失旧最佳值。
- 第一阶段：总计 200 epoch（包含已有 20 轮）。每轮验证完整 100 个验证工况，batch=1，MSE，初始学习率沿用 1e-4。
- 学习率：验证损失连续 8 次没有至少 0.5% 的相对改善，乘 0.5；冷却 2 次验证，最低 1e-6。使用 PyTorch ReduceLROnPlateau。每轮保存调度状态、优化器状态、last 和验证最优 best。旧恒定学习率检查点升级时，以旧 best 初始化调度器，不回放过去的降学习率。
- 早停：连续 60 次完整验证未出现任何新最优值则停止，不因未达目标无限运行。
- 第二阶段：200 轮完成后，若最后 40 轮的最优验证损失比前 160 轮的全局最优至少改善 2%，自动延长至总计 400 轮；否则结束。已经早停的模型不延长。400 轮为本流程上限。
- 两个新模型不改架构、不改损失空间，便于检验额外训练的收益。第三张 GPU 同样续训 POD，形成更强对照。GPU 0=Transolver-3，1=GeoTransolver，2=POD。
- 所有选择与延长判断都不读取独立测试指标。训练结束后，按验证损失选定的 best 测试全部 100 个历史保留工况一次，再汇总结果。

## 精度目标及比较

原 20 轮 POD：RMSE 3980.56 Pa，MAPE 2.876%。预先约定有意义的目标为 RMSE 至少降低 20%，即约 3184.45 Pa，并检验其相对长训练 POD 的改进。

最终目标判定同时要求：RMSE <= 原 POD 的 80%；与长训练 POD 比较，按测试工况配对 bootstrap（2000 次，seed=42）的 RMSE 比值 95% 区间上界 <= 0.8。MAPE、MAE、R²、每工况指标和已完成/最佳 epoch 同时报告。置信区间只反映工况抽样不确定性，不代替多训练种子验证。

若未达到目标，自动报告“尚未达到”，不把增加 epoch 等同于成功。若验证已进入平台，下一轮应诊断表示能力、平均场残差学习或条件融合，而非无限追加 epoch。该后续架构/目标调整尚未启动。

本轮已看过历史测试结果，因此它仍是历史保留集，不是新增盲测集；后续需要新增仿真工况或独立数据验证泛化。更长训练的模型与原 20 轮模型预算不同，不能据此直接作架构优劣结论。本轮仍不验证变叶型、变孔位泛化，也不预测 u/v/w/T。

## 运行与恢复

```
PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/run_long_training.py --run-root runs/transolver_long
```

同一命令可在进程已停止后恢复；不要在现有训练仍运行时重复启动。运行状态见 `runs/transolver_long/status.json`，实际逐轮进度、loss 和学习率见各模型的 `train/train.log`；每阶段控制台日志为 `train_200_console.log` / `train_400_console.log`。流程自动生成 `reports/transolver_long/REPORT.md`、CSV、JSON、完整预测 NPZ 和学习曲线。

```
PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/run_long_training.py --probe --run-root runs/transolver_long_probe
PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
```

probe 从原检查点运行两个真实完整网格更新，每次完整验证，以强制平台条件验证降学习率和保存/恢复，不接触独立测试。probe 是一次性验证，重复验证请使用新的 probe 目录。

原样本缓存通过硬链接共享，元数据独立复制。迁移检查点的路径指纹前，核对所有 sample 缓存为相同文件、划分表字节相同、样本顺序相同，并验证原 checkpoint 指纹与源数据匹配；迁移记录保存在 `verified_cache_relocation.json`。原始 checkpoint SHA256 另存于 `continuation_source.json`。

因项目磁盘空间有限，中期完整 checkpoint 副本保存在 `/tmp/turbine-long-milestones`，大体积 VTU 输出保存在 `/tmp/turbine-long-evaluation`；最终 best/last、训练日志、报告及完整压缩预测保存在项目目录。临时文件可从数据及检查点重建。

按前 20 轮吞吐粗估，Transolver-3 续训到总计 200 轮约需 3.5 小时，GeoTransolver 约需 2.3 小时；若延长至 400 轮，分别约 7.5 小时和 4.8 小时。早停、系统负载和 I/O 会改变实际耗时。

## 验证与实现

- 可恢复调度实现：`tfai/training/optimization.py`；训练器每轮验证后 step，并在 checkpoint 中保存/恢复状态。
- 修复 GPU checkpoint 加载把 CPU RNG ByteTensor 映射到 CUDA 的问题：续训 checkpoint 先加载到 CPU，再由 model/optimizer 的 load_state_dict 放到对应设备。
- 真实三个模型恢复验证：`verification/long_training_probe.log` 与 `runs/transolver_long_probe/status.json`。
- 自动延长策略及工况配对统计有独立回归检查；原模型适配、几何写回、旧指标计算检查继续保留。
