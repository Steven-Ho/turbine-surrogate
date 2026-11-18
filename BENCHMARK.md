# 五模型 Rotor37 对比实验

五模型为 POD、UNet、GCN（MeshGraphNet）、Graph_UNet、Graph_Transformer。
正式实验采用相同 seed=42、800/100/100 划分、batch_size=1、学习率 1e-4、scale=small、20 epochs（16000 个优化器步骤），从头训练，由各模型验证集损失选择最佳检查点。测试数据不参与模型选择。
这是同样本遍历预算的首轮对比；不是等参数量、等算力或多种子统计实验。

## 运行

```bash
cd /data/hesc/Projects/turbine-surrogate
source .venv/bin/activate
# 需要沙箱外的 GPU 权限，调度使用 GPU 0、1、2
python scripts/run_benchmark.py --phase probe
python scripts/run_benchmark.py --phase train --epochs 20
python scripts/compare_models.py --epochs 20
```

调度状态：`runs/comparison20/status.json`，每个模型的配置、日志和检查点独立保存。
`run_benchmark.py` 的调用会重新执行所选模型；请先确认现有任务已结束再运行，避免覆盖正在运行的实验。
排错时可用 `--models POD UNet` 仅重跑指定模型，其他结果状态会保留。

三个图模型使用同一套完整图样本（29773 个节点），样本文件为硬链接、缓存元数据各自独立；训练要求缓存命中，避免错误重算或覆盖共享样本。
POD 只用训练集的 800 个场快照拟合模态，验证集只投影编码；保存拟合样本编号以审计。
UNet 用上游 ImageVoxelProcessor，最长轴 base_resolution=50；独立测试时将预测投影回原始 VTU 节点。

## 统一评分

五模型原生评估空间不同，不能直接用内部 loss 或处理器内部 NRMSE 排名。
`compare_models.py` 读取每个模型写出的预测 VTU 与原始真实 VTU，验证节点顺序与坐标一致，对相同 100 个测试工况的 2977300 个原始节点重新评分。
统一输出 MAE、RMSE、NRMSE（除以全部真实压力极差）、MAPE、相对 L2 和 R²；MAPE 阈值固定为训练集绝对压力 p95 的 0.001 倍。
POD 降维误差、UNet 网格化和插值误差均计入这些指标。

产物保存在 `reports/comparison20/`：`REPORT.md`、`comparison.json`、`metrics.csv`、`per_sample_metrics.csv`，以及统一节点指标图、训练曲线、同一工况的压力场和逐工况误差分布图。
结果生成前会核验所有模型完成 20 轮训练、800/100/100 相同互斥划分、100 个独立预测文件及 POD 模态训练范围。

## 针对实验修复的问题

- POD 模态拟合限制在训练集，消除原实现使用验证快照拟合的泄漏。
- 内容相同的样本表不反复重写，防止缓存失效。
- 图缓存允许相同内容的样本表跨实验目录使用；源数据、表内容、预处理配置和划分变化仍使缓存失效。
- 复用先前的大数组精确 p95 计算修复。

```bash
OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
```

## 曲面 U-Net 修复与最终实验路径

原体单元包含判定在 Rotor37 曲面上只保留了 2 个体素，单样本评估写回也错误地沿空间轴做反归一化。原 `runs/comparison20/UNet` 已标记为无效运行，不进入最终结果；其权重和日志保留，可重建缓存已清理。

修复后采用半个体素对角线距离的曲面带采样，当前 base_resolution=50 对应 4761 个有效体素，参考训练几何的全部原始节点距有效体素均在该采样半径内；测试几何的偏移与超出参考采样带的节点比例单独记录。背景工况输入置零，防止未归一化的物理参数混入网络输入；单样本和批量写回都恢复到 Pa。
新 U-Net 运行保持相同 20 epochs、数据划分和学习率，保存于 `runs/comparison20_surface/UNet`。`run_overrides.json` 明确指定此有效实验，状态和评分脚本会读取该映射。

```bash
python scripts/run_benchmark.py --phase train --epochs 20 --models UNet --gpu 2 --run-root runs/comparison20_surface
```

由于项目分区空间有限，新 U-Net 的可重建预处理缓存经 `data/processed` 链接放置在 `/tmp/turbine-surrogate-unet-surface20/processed`，模型、日志、测试输出与报告仍在项目内。缓存被清理后可重新预处理，不影响已保存检查点的独立测试。

报告还给出仅由训练集拟合的平均压力场参考值，区分工况学习效果与固定平均场先验；该参考不是第六个训练模型。

## 几何与节点对应关系

实测样本保持相同节点数和逐单元节点编号拓扑，但坐标存在毫米量级变化。本轮使用上游的参考几何训练约定：输入工况为 Omega、P，POD 预测共同节点编号上的压力；图模型使用参考几何特征，UNet 在参考体素网格上学习。不是新拓扑或任意新几何泛化实验。

评估时优先使用各样本真实源文件作为写回模板；UNet 按该样本真实坐标重新构建逆插值映射。评分严格核对输出坐标与各自原始网格完全相同、逐单元拓扑一致，再对相同节点编号的原始压力评分。报告记录坐标变化和 U-Net 测试节点在参考采样带外的比例，这部分误差不会排除。

若继续研究几何泛化，应另设显式变几何输入和几何划分实验，不能把本轮工况对比直接解释为该能力。

重跑默认 20 轮实验时，调度器也会读取 `run_overrides.json`，因此会使用修复后的 U-Net 目录并同步状态。可先用 `--dry-run` 查看路径，避免误用历史无效模型：

```bash
python scripts/run_benchmark.py --phase train --epochs 20 --dry-run
```

新建 U-Net 实验时调度器自动将预处理缓存放入 `/tmp`；现有临时缓存目标被删除时会自动重建目录。检查点与正式测试数据继续保存在项目内。

所有模型的训练与测试已完成，结果和检查点校验清单见 `reports/comparison20/REPORT.md` 与 `run_manifest.json`。仅重新测试已有最佳模型（不会重训）：

```bash
python scripts/run_benchmark.py --phase evaluate --epochs 20
python scripts/compare_models.py --epochs 20
```
