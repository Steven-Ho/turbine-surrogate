# faNN 七模型适配

本入口独立于 Rotor37，未更改原七模型和训练器。适配目标为具有对应节点的 faNN
三维六面体子域；统一输出 VelocityX/Y/Z、Pressure、Temperature、Density。
当前实现是可运行的子域比较基线，不是981万节点整域训练实现。

## 数据与模型协议

- 用相同的前 N 个六面体提取每个样本，保留真实单元边、原节点和单元ID。
  不以随机删点后的旧连接构图。该局部区域不代表整个叶道，所得误差不能称全场误差。
- 输入：三维实际坐标、RotatingVelocityX、OutletPressure。入口总压、总温和气体属性
  在当前faNN发布中是公共常量，此入口不将其作为变化条件；不支持这些条件变化的外推。
  不输入质量流量、效率等求解后标量。暂未加入墙面距离和边界标签。
- 三维坐标使用训练集共同原点与等比例长度缩放，避免每个样本单独缩放改变几何比例。
  目标均值/标准差仅从训练样本拟合，设有每通道尺度下限。
- POD：用训练目标快照SVD拟合最多32阶基底，由现有POD MLP从展平几何坐标和工况预测系数。
  要求节点ID顺序和拓扑一致。这与Rotor37原来的POD输入协议不同；单样本只能验证接口。
- UNet：使用现有small三维UNet；在包围盒32³规则网格上编码最近节点的坐标、工况、
  最近节点距离，再以三线性插值回到原节点。损失只在实际子域节点上计算。
  这是最近邻几何延拓，不是保守体素化；没有把整个包围盒宣称为流体区域。
- GCN/Graph_UNet/Graph_Transformer：原模型，使用真实六面体边；GCN额外使用位移及边长。
- Transolver-3：复用原分块状态聚合实现，xyz+两个工况输入、六输出。
- GeoTransolver：复用portable_torch，KDTree邻域。归一化半径0.05/0.25、邻居8/32沿用既有设置，
  后续精度实验需调参；子域边界截断的上下文与完整域不同。

## 本机

当前原始样本位于 data/fann_small。需要 pyarrow==19.0.1 与原项目运行依赖。
本机临时依赖安装在 /tmp/fann-inspect-deps；该目录被清理后需重新安装。

```bash
PYTHONPATH=/tmp/fann-inspect-deps:$PWD .venv/bin/python scripts/prepare_fann.py \
  --source data/fann_small --parquet data/hf_train-00144-of-00151.parquet \
  --cells 512 --output data/fann_small/prepared/case_0144.npz

PYTHONPATH=$PWD .venv/bin/python scripts/run_fann.py \
  --train data/fann_small/prepared/case_0144.npz \
  --smoke --steps 3 --output runs/fann_local_smoke
```

输出目录必须不存在。已有 prepared/case_0144.npz 可直接复用，不必重新提取。
单个Parquet可通过--row指定样本；所有样本应采用相同cells。为了限制本机内存，
准备工具将cells限制为10000。它只读取指定行组和必要列，不复制完整数据集。

## 集群

上传新增源码、脚本以及 prepared NPZ 即可进行GPU接口验证，不需要上传917 MB公共结构：

```bash
sbatch scripts/slurm_fann.sh \
  --train data/fann_small/prepared/case_0144.npz --smoke --steps 3
```

默认顺序运行七模型，申请1 GPU/4 CPU/32G主存，使用原厂商环境。
输出 runs/fann_JOBID，日志 fann_JOBID.out/.err。可通过--models指定模型子集。
尚未在沐曦实际执行；特别是Graph_UNet稀疏算子需要集群验证，不会自动替换模型或跳过失败。

多样本训练示例（以下case_A/B/C须先实际准备，当前只有一个样本）：

```bash
sbatch scripts/slurm_fann.sh --train prepared/case_A.npz prepared/case_B.npz \
  --valid prepared/case_C.npz --steps 1000
```

--steps是每个模型总更新次数，样本按输入顺序循环；不是epoch，也没有early stopping。
没有valid时必须显式--smoke。训练/验证相同路径或重复样本内容会被拒绝。
仍须由使用者按原GeometryNumber设计几何划分；本入口不会把不同工况自动视为不同几何。
正式报告需要多个样本/种子、同几何工况与跨几何分开评估。

## 输出与已验证内容

- report.json：每模型loss、训练墙钟、六变量物理单位RMSE/MAE、错误及运行环境。
- MODEL/prediction_i.npz：原节点ID顺序的六变量物理值，已检查保存/读取一致。
- 非smoke模式保存MODEL/checkpoint.pt，包含模型权重（含POD基底）和训练归一化。
  这是推理权重，不含优化器状态，不能用于精确断点续训。
- smoke模式仅在内存中严格重载state_dict并对照eval输出，避免保存七份大权重。

本机verification/fann_seven_smoke/report.json：7/7模型通过，CPU、FP32、1230节点、512单元，
每模型3次更新；检查形状/有限值/梯度/参数变化/权重重载/物理量回映射/预测文件读取。
POD基底用该训练样本拟合，故训练误差不是泛化指标。

尚未完成：981万节点整域的训练/推理调度、跨块上下文和守恒评估、跨几何统计实验。
不要把当前子域接口通过等同于全域效果已验证。

## 集群原始数据路径与七任务提交

默认原始数据为 ~/Dataset/faNN，要求包含 data/hf_train-*.parquet、tree_constant_part.pkl。
仅支持原始Parquet发布布局，不是datasets.save_to_disk的Arrow布局。
先在虚拟环境安装兼容 NumPy1 的 pyarrow==19.0.1（离线集群需预备wheel）。

```bash
sbatch scripts/slurm_fann_prepare.sh
# 准备作业完成后，将作业号替换为实际值：
export FANN_SPLITS="$PWD/runs/fann_prepared_准备作业号/splits.json"
export FANN_STEPS=20000
sbatch scripts/slurm_fann_array.sh
```

预处理默认512单元，可提交前设置FANN_CELLS（最大10000）。按GeometryNumber固定seed=42，
约80/10/10划分train/valid/test，测试集不参与训练或验证。当前训练入口只评估valid，
独立test仍需后续增加冻结检查点评估。七个数组任务各使用一张GPU。
输出 runs/fann_数组作业号/模型名/模型名/，包含权重、验证预测和报告。
本入口为固定步数的子域基线，不含early stopping和断点续训；不是完整叶道正式全场训练。
