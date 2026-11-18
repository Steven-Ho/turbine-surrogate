# 发动机静子叶片：边界条件与网格到全流场的样例抽象

已从 `/data/hesc/测试数据/case4` 导出实际可读取的单工况样例，未修改原始文件。输入网格来自原始 CGNS，输出标签来自同一算例最终保存的 VTK 结果；逐节点坐标和逐体单元的节点集合、类型已验证一致。CGNS 与 VTK 的局部角点约定不同，导出使用合法 VTK 单元顺序。

## 计算问题

\[
F_\theta(\text{mesh},\text{boundary conditions},\text{fixed physics})
\longrightarrow [u_i,v_i,w_i,p_i,T_i]_{i=1}^{N}
\]

- `N=5503387` 个节点、`16074294` 个体单元，混合四面体、三棱柱和金字塔。
- 3 个流体域：`fluid_in`、`fluid_out`、`fluid_rotor`。末者虽然名字含 rotor，求解配置的转速均为 0，边界包含静子叶片与冷却孔。
- 26 个边界或接口面，直接从 CGNS 的具名面单元抽取节点编号，不使用无约束最近邻贴标签。
- 标签是已有的 **PointData**，无需从单元值人为插值：`Velocity` 三分量、`Static_Pressure`、`Temperature`。因此这是流体体网格上的五个场，不只是叶片表面，也不是一个性能标量。
- 速度选静止/绝对参考系的 `Velocity`，压力选静压，温度选静温；总压/总温不是同一个输出量。
- 固定物理设置包括可压缩理想气体、能量方程、SST k-omega、材料参数。改变这些设置时，应一并作为输入或分别建数据集。

## 已提取的主要工况

| 边界 | 已知输入 | 本例值 |
|---|---|---:|
| 主流入口 inlet | 总压 p0 | 2807720 Pa |
| 主流入口 inlet | 总温 T0 | 1920.79 K |
| 出口 outlet | 静压 p | 1649470 Pa |
| 冷却入口 cold_in_up | 质量流量、温度 | 0.18 kg/s、850 K |
| 冷却入口 cold_in_down | 质量流量、温度 | 0.18 kg/s、850 K |

解析以实际 `AIflow.ctl` 为主，原始 GUI 的 `MassFlowRate` 字段用于交叉核对冷却入口的质量流量含义。全部边界、材料和接口配置保存在 `boundary_conditions.json` 以及 HDF5 输入元数据中，不仅保留表中七个参数。

必须保留边界类型和数值是否定义的掩码：

- 入口总压、总温不能直接当作节点静压、静温强制钳制。
- 质量流量入口的 `u=v=w=0` 配置字段不是三个速度零值约束；实际条件是法向质量流量。
- 壁面 `temp=Neumann,t=0` 不是壁温 0 K。
- `per_*` 边界在本例中实际声明为 `symm`，`periodicPairs` 为空，不能仅凭名字构造周期配对。
- 两对域间接口须保留双侧面及配对关系。不能把接口的真实求解压力、速度当作推理时已知的边界值，否则会泄漏答案。

## 数据产物

`data/stator_case_000/sample.h5` 约 300 MiB，已保存在项目内，SHA-256 见同目录 `manifest.json`。原始数 GB 的结果无需再复制。

```text
inputs/
  mesh/
    positions             [N,3]
    cell_connectivity     [flattened corner indices]
    cell_offsets          [C+1]
    cell_types            [C]
    node_region_ids       [N]
  boundaries/<domain>/<patch>/
    node_ids
    face_connectivity
    face_offsets
  boundary_flags          [N] 多位标志，保留角点上的多边界归属
  global_bc_values        [7]
  attributes: 参数名称、完整类型化边界条件、固定物理设置

targets/
  uvwpT                   [N,5], float32
```

输入只包含几何、拓扑、域标识和已知边界/物理参数；压力、速度、温度结果以及残差均不进入输入。原文件的 MPI 分区编号 `HPC_Region` 也不是物理域输入标签。

复现与接口检查：

```bash
cd /data/hesc/Projects/turbine-surrogate
source .venv/bin/activate
PYTHONUTF8=1 python scripts/prepare_stator_case.py
# GPU 访问需在沙箱外执行
PYTHONUTF8=1 OMP_NUM_THREADS=4 python scripts/smoke_stator_graph.py
PYTHONUTF8=1 OMP_NUM_THREADS=4 python -m unittest discover -s tests -v
```

已经在完整 `fluid_in` 子域上进行一次接口/梯度检查：44652 个节点、145526 个体单元、443894 条有向网格边，16 通道网格/边界输入，输出 `[44652,5]`，CUDA 前向与反向梯度均为有限值。
**这不是经过训练并验证精度的全域代理模型**。该检查仅证明样例可接入现有 MeshGraphNet 多输出接口；全域分区、跨域耦合和跨工况训练尚未完成。

## 建模路线

1. 固定当前几何，围绕主流 p0/T0、出口静压、两路冷气流量和温度设计物理上可行的工况组合。范围需由设计工况确定，不能任意扰动后保留旧解作为新标签。
2. 每组边界条件重新运行 CFD，保留网格、完整输入配置和对应的合格解。固定材料/湍流模型，或将变化一并编码；不要遗漏影响解的输入。
3. 按完整工况先划分训练、验证、测试集，然后再分区。一个工况的不同节点、三个域和三个结果副本不是独立工况，不能据此构造泛化测试。
4. 使用支持网格的条件图模型作为起点：节点坐标与域/边界标志、边相对坐标/长度、分类型边界值与掩码、全局工况上下文，输出五通道场。现有 GCN 的多输出接口已验证可用，但局部少层消息传递不能保证学会全域稳态耦合。
5. 完整网格规模较大，应增加分区与重叠 halo、多尺度/粗图全局传播、域间接口消息交换。接口可能非共形，不能简单把相近节点合并；周期约束仅在源配置明确给出时使用。
6. 各物理量分别归一化并评估 MAE/相对 L2；同时检查质量流量、能量平衡、出口总压损失和壁面边界约束。高精度最终测试必须回到全部原始节点，而不是只在抽样节点上评分。

图网络方法的基础参考：[Pfaff et al., Learning Mesh-Based Simulation with Graph Networks](https://arxiv.org/abs/2010.03409)。这里的静态多场代理、边界编码和多域分块是针对本项目的工程方案，不是该论文对本算例的性能保证。

## 当前数据的边界

- 目前找到的是 **1 个保存工况的最终迭代快照**。目录中的 setup/solve、总网格/分域文件、CFD/FEM 分支不能作为多个独立流场样本；6000 行残差也不是 6000 个全场解。
- 五个目标通道均无 NaN/Inf，但有 16 个节点达到 200 K、2 个达到 5000 K，恰好为配置限值；最后的动量与湍流残差也需要结合求解器定义审查。保留原始值、标记质量疑点，不擅自删除，也不把有限数值等同于已收敛。
- 当前样例覆盖 **流体域**。若还要预测叶片金属内部温度或结构应力/位移，需要另行接入固体热分析/FEM 数据与耦合接口，不能用流体温度冒充固体温度。
- 因为只有一个工况，本样例明确标记 `eligible_for_generalization_benchmark=false`。可以做读写、形状、边界映射和单样本拟合检查，不能报告新边界条件或新几何的预测精度。
- 当前找到的求解启动脚本指向 Windows AICFD 安装路径，尚未验证本机能自动批量调用该求解器；批量产样需要补齐这一运行环节。
