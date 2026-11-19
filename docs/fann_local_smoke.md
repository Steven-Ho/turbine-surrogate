# faNN 单样本格式检查与调试

2026-11-19 开发前已执行 git pull --ff-only origin master，由 cde587d 快进到 cab8141。

本机输入位于 data/fann_small，仅一个有标签样本与公共网格。不会展开整个数据集。

脚本 scripts/smoke_fann.py：
- 核验 download_manifest.json 中所有文件 SHA256。
- 使用仅允许 NumPy 数组所需全局对象的 Unpickler 读取公共结构。
- 检查 26 列数据长度和有限性、9429120 个八节点单元连接索引范围。
- 取前512个真实六面体，保留原节点/单元ID，重映射为零基局部连接。
- 输入仅归一化坐标、转速、出口压力；输出为 u/v/w/p/T/rho。
- CPU上运行现有GCN small的三次前向/反向/Adam更新，检查输出、梯度、参数变化。

本次结果见 verification/fann_smoke/report.json，子网格见同目录 submesh.npz。
节点1230，单元512，有向边5734；三次更新前loss分别约0.866、0.747、0.643。
仅为接口检查，不评估泛化、守恒、网格质量或预测精度；所取局部区域不代表全域。
尚未适配全部七模型，也没有进行完整网格训练或GPU验证。

依赖：项目环境 + pyarrow==19.0.1（兼容当前NumPy1.26）。本次将其临时安装到
/tmp/fann-inspect-deps，未修改项目环境。重跑时指定新的输出目录以避免覆盖：

```bash
PYTHONPATH=/tmp/fann-inspect-deps:$PWD .venv/bin/python scripts/smoke_fann.py \
  --output verification/fann_smoke_repeat
```

/tmp依赖清理后可重新使用 pip --target 安装，或在独立开发环境安装该版本。
正式多样本适配需要拟合训练集归一化、明确几何划分、补齐公共工况和边界信息。
不得将 MassFlow、Efficiency 等求解后结果直接作为训练输入。
