# 集群几何查询与 CPU 数量诊断

两份 .sh 都自带 Python 诊断程序，不需额外上传 Python 文件或安装新包。
上传到项目 scripts/ 下；依赖现有虚拟环境、OpenGL 运行库和已生成的训练缓存。
所有输出写入新的 runs/diagnostics/，读取原训练配置/缓存，不覆盖原检查点或训练目录。

## 提交

在 SSH 登录终端、项目根目录执行：

```bash
mkdir -p logs
export TURBINE_SOURCE_RUN=runs/metax_long_20261111_100428_4166078
geo_job=$(sbatch --parsable scripts/slurm_geo_neighbor_probe.sh)
sbatch --dependency="afterany:${geo_job%%;*}" scripts/slurm_cpu_scaling_probe.sh
```

将 TURBINE_SOURCE_RUN 换成待诊断的七模型结果目录。第二个作业等待第一个结束，避免两次诊断相互干扰。
两份脚本各申请 1 GPU；几何脚本申请 4 CPU，CPU 脚本申请 8 CPU。Slurm 有空闲资源后才会启动。
保留 srun 的设备分配，不手工设置 CUDA_VISIBLE_DEVICES。
虚拟环境/OpenGL 不在默认位置时，设置 TURBINE_VENV / TURBINE_OPENGL_RUNTIME。

## 1. GeoTransolver 几何邻居

文件：slurm_geo_neighbor_probe.sh。
默认读取前三个实际训练样本，测试两个半径 .05/.25 及邻居数 8/32。
分别测量首次 A、同值新张量 A、不同样本 B/C、切回 A 的查询时间及命中/失效计数。
再用同一序列运行完整 GeoTransolver 前向/反向，记录实际查询调用耗时占比；不执行优化器更新。
单块 cdist/topk 分开计时，帮助定位距离计算还是选邻居慢。

结果：runs/diagnostics/geo_neighbors_JOBID/：

- geometry_differences.json：不同样本坐标是否相同及最大差异。
- neighbor_query_results.csv：独立邻居查询的命中/失效与总耗时。
- kernel_block_radius_*.json：一个距离块的 cdist/topk 时间，不能直接当完整训练耗时。
- model_results.csv、query_events.csv：全模型前向/反向与各层索引查询耗时。
- REPORT.md、environment.json、status.json、geo_probe.py。

如果不同坐标样本反复失效、查询占满模型时间，就支持“逐样本 O(N²) 重算是主要瓶颈”。
如果命中时仍慢，或查询只占很小比例，则需要调查模型其他算子。
同步计时和包装函数有剖析开销，这些数字不是正式训练吞吐分数。
可调 GEO_PROBE_SAMPLES（默认3，至少2）、GEO_PROBE_CHUNK（默认256）。

## 2. CPU 数量对 Transolver_3 稳定训练步骤的影响

文件：slurm_cpu_scaling_probe.sh。
同一个 Slurm 作业、同一张 GPU，依次限制为 4 -> 8 -> 8 -> 4 个逻辑 CPU，避免不同 GPU 的干扰。
每次运行新 Python 进程，在导入 torch 前设置亲和性和线程环境。
默认 OMP/MKL/PyTorch 线程数都固定为 4，与原本机/集群训练一致，主要测试允许使用的 CPU 数量。
可另提交一轮 CPU_PROBE_THREADS=0，使线程数随 4/8 CPU 成比例变化；那是额外改变线程上限的对照。

模型为原 Transolver_3 small，使用现有完整网格训练样本、相同初始权重、seed42、Adam、原学习率和 batch_size=1。
默认前32个样本、每条件预热10步后计时100步、每个CPU条件重复两次。

两种模式：

- disk：每步通过原 FullSampleDataset 读取/反序列化 .pt，再使用原 collate_gcn_batch；相当于 num_workers=0 的串行加载语义。
- memory：预载相同 CPU 样本，仍做同样的拼批和 CPU->GPU 传输，排除每步文件读取/反序列化。

全部条件先预热同一组文件的页缓存。每种模式都重置到相同权重与优化器初态，样本顺序相同。
计时分为数据加载、CPU->GPU 传输和前向/反向/Adam 更新，并记录进程 CPU 时间/墙钟时间、实际亲和性及 CPU 拓扑。
测试使用与训练相同的数据集/拼批函数，但不包含完整 DataLoader 框架、验证、检查点和逐步日志开销；不是完整 epoch 端到端测试。

结果：runs/diagnostics/cpu_scaling_JOBID/：

- summary.csv、REPORT.md：4/8 CPU 的每步时间、相对速度和有效 CPU 使用量。
- all_results.json、repeat*_results.json：各次重复的明细、环境、实际线程数与最终 loss。
- repeat*_*_steps.csv：逐步加载/传输/计算时间。
- allocation.json、sample_manifest.json、*_spec.json：分配、初始权重校验值和固定样本清单。
- *.log：各子进程输出；initial_weights.pt 仅为新诊断模型的共同初态，不是原训练权重。

8核持续更快才支持增加 CPU；只有 disk 慢而 memory 快时，优先检查读取/反序列化。
低有效 CPU 使用量不排除单线程瓶颈，多核也不能加速串行工作。
并发 I/O、NUMA 和频率等因素仍可能影响结果；最好选择相对空闲时段运行。

可调参数（提交前 export）：
CPU_PROBE_COUNTS=4,8；CPU_PROBE_THREADS=4（0表示按核数变化）；
CPU_PROBE_REPEATS=2；CPU_PROBE_SAMPLES=32；CPU_PROBE_WARMUP=10；CPU_PROBE_STEPS=100。
如增加到16 CPU，需同时通过 sbatch --cpus-per-task=16 增加申请，并设置 CPU_PROBE_COUNTS=4,8,16。

## 日志与下载

Slurm 输出位于 logs/geo_neighbors_JOBID.out/.err 和 logs/cpu_scaling_JOBID.out/.err。
诊断结束后下载对应 runs/diagnostics/ 目录，连同 Slurm 日志一起分析；不要只下载 .out/.err。
本机已停止试验；提交版仅完成静态检查，不使用本机并发 GPU 环境下的时间作为性能基线。
