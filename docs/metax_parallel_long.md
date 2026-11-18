# 七模型沐曦集群长训练

从项目根目录解压 turbine-cluster-long-addons.tar.gz；保留现有源码、数据、虚拟环境和 OpenGL 运行库。
补充包包括 Slurm 作业、执行/汇总程序，以及从本机真实检查点导出的七模型冻结配置和计时基线。

## GeoTransolver 源码适配

本机原路径依赖 nvidia-physicsnemo==2.0.0（要求 Python >=3.11）和 Warp 查询算子，不能直接补入当前 Python 3.10 环境。
补充包已包含该版本 GeoTransolver 所需模块的源码子集（保留许可证与原始文件 SHA256），不需要安装整个 PhysicsNeMo、Warp 或 jaxtyping。
网络结构、参数名称和张量形状保持不变，Transformer Engine 保持关闭；长训脚本自动设置 model.geotransolver_backend=portable_torch。
本机原配置默认仍走 physicsnemo 路径，不会改动历史实验的后端。

几何查询改用官方 PyTorch 参考后端的语义：半径内最近 K 点、包含半径边界、缺失邻居补零；分块控制显存。
参考几何不变时缓存索引，每次仍从当前输入张量 gather，保留输入梯度；坐标或设备变化时重新查询。
原始 PhysicsNeMo 明确不保证 Warp 与 Torch 的截断邻居集合/顺序完全一致。因此保留网络并不代表与本机 Warp 路径逐位等价；
性能比较要同时注明查询后端和索引缓存变化，不能全部归因于 X203 硬件。
本机已对照官方 Torch 后端验证参数键、同权重前向输出、输入梯度和参数梯度；X203 真机仍需提交后验证。
Graph_UNet 和 GeoTransolver 均会先做小图 GPU 前向/反向，检查实际后端算子；失败单独记录，不会伪装成完成。

官方包信息：https://pypi.org/project/nvidia-physicsnemo/2.0.0/
源码来源与差异：tfai/models/_vendor/geotransolver2/UPSTREAM.json。

## 训练协议

七模型顺序：POD、UNet、GCN、Graph_UNet、Graph_Transformer、Transolver_3、GeoTransolver。
使用原 1000 个 Rotor37 样本；seed=42，800/100/100 划分，batch_size=1，small，初始 LR=1e-4。
每个模型独立从头运行 20 epoch 固定 LR，然后恢复模型、优化器、随机数和采样器状态续训至 200 epoch。
续训启用 ReduceLROnPlateau：factor=.5、patience=8、threshold=.005、cooldown=2、min_lr=1e-6；
早停 patience=60。满 200 epoch 且第 161–200 epoch 最佳验证损失较之前改善超过 2%，才延长至 400。
这与本机已有三模型长训调度一致；其余四模型是采用同一协议的新长训实验。
只有 POD、Transolver_3、GeoTransolver 有本机长训基线，其他四个只有 20 epoch 基线。
UNet 使用已经修复的曲面采样配置。图模型使用完整网格，不分块。
第一阶段在各自目录生成预处理缓存；续训必须命中该缓存。所有目录写在持久存储，不指向 /tmp。

## 提交

确保依赖和数据已就位，OpenGL 包位于项目根目录 turbine-opengl-jammy/。
在登录节点、项目根目录执行：

```bash
bash scripts/submit_cluster_long.sh
```

它提交训练数组和 afterany 依赖的自动汇总作业。无需在登录节点导入 torch。
默认最多并行 7 个模型，每个任务 1 GPU、4 CPU、128 GiB 主机内存；其余任务排队，
实际并发数受集群资源影响；若仅有两张空闲卡，则两个运行、其余排队。不是单个模型跨卡 DDP，也不是七个模型共享一张卡。
如需改变并行数，修改 slurm_train_all_metax.sh 中 --array=0-6%7 的 %7（例如改为 %2）。
若平台需要账号、分区或更长时间上限，在两个 Slurm 文件中按平台要求设置。
任务尊重 Slurm 的 GPU 可见性；若不能确定独占的单卡绑定，会预检失败，避免多个任务抢同一卡。

自定义目录（每次新实验请使用不同名字）：

```bash
export TURBINE_LONG_RUN_ROOT=runs/metax_long_trial1
export TURBINE_VENV="$HOME/venvs/turbine-metax"
# 只有 OpenGL 放在其他位置才需要设置：
# export TURBINE_OPENGL_RUNTIME=/absolute/path/to/turbine-opengl-jammy
bash scripts/submit_cluster_long.sh
```

也可单独提交数组：`sbatch scripts/slurm_train_all_metax.sh`。
默认保存到 runs/metax_long_<数组作业号>；数组结束后自行设置该路径并提交 slurm_compare_metax.sh。
失败后不会自动覆盖同名实验；使用新目录重跑。不要同时修改运行中的冻结配置。

## 保存的产物

每模型独立目录：status.json（含失败原因、每阶段实际轮数/学习率/耗时）、environment.json、
configs/stage_*.yaml、input/ 和 data_split.json、train_*_console.log、train_*.log、
training_history.csv（验证周期的训练损失、验证损失、LR、step、时间戳）、
train/best_model.pt、train/last_checkpoint.pt、train/loss_curve.png、
milestones/20、200、400 下的检查点，以及 evaluate/、evaluate_console.log。
首次预处理可能较慢，各任务会独立生成缓存，避免并发写共享缓存。
每完成一个阶段都会导出历史 CSV；运行中可查看训练日志和最新检查点。

自动汇总：REPORT.md、comparison.json、timing_comparison.csv、metrics_original_nodes.csv。
汇总在部分任务失败时仍生成，并返回非零退出码；不会用缺失/失败任务构造“七模型成功”的结果。
精度统一在相同 100 个独立测试样本的原始节点上评分，核验坐标和拓扑，包含 POD/UNet 表示误差。

## 耗时比较口径

training_seconds 读取训练器原生日志，与本机相同，包含验证、数据加载和训练内检查点写入，
不含前置预处理。pipeline_train_seconds 包括整个训练子进程和预处理，应单独看。
本机部分模型使用预热缓存，集群第一阶段是冷缓存，因此不能把总流程时间差都归因于 GPU。
吞吐比 = 本机每新增 epoch 秒数 / 集群每新增 epoch 秒数；大于 1 表示集群更快。
早停或延长分支不同会明确标记预算不相同；其吞吐比只作参考，不能直接比总训练时间。
GeoTransolver 的纯 Torch 查询和索引缓存变化也会影响吞吐，汇总将标注模型后端。
本机为 RTX 3090 / PyTorch 2.5.1 / Python 3.11；集群为 X203 / 厂商 PyTorch 2.6 / Python 3.10，
因此测得的是硬件、后端实现和软件环境的综合差异，不是严格隔离的 GPU 硬件性能比。

## GeoTransolver 单独使用

长训脚本已自动选择 portable_torch。若直接使用原训练入口，需要显式选择：

```bash
python scripts/train.py --config-name train_rotor37_geotransolver \
  ++model.geotransolver_backend=portable_torch training.require_preprocess_cache=false
```

先按原说明准备样本表。此命令仍应在已加载 OpenGL/虚拟环境的 GPU 作业中执行。
旧的本机 PhysicsNeMo 路径保留不变；无需安装新的 wheel，已有 torch/einops 足够运行模型子模块。

## GPU 步骤与调度日志修正

训练任务现在通过 srun --ntasks=1 --gres=gpu:1 启动，不从全局 GPU 编号推导 CUDA_VISIBLE_DEVICES。
Slurm 的 .out/.err 保存在项目根目录 logs/；提交入口会在 sbatch 前创建它。
启动诊断同时写入各模型目录的 gpu_preflight.json 和 ht-smi.txt。
需要从头重训时，先取消旧训练数组及旧汇总作业，然后 unset TURBINE_LONG_RUN_ROOT 再提交。
详见 docs/gpu_binding_fix.md。
