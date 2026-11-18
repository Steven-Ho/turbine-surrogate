# 集群 GPU 硬件与性能核实

从项目根目录提交，使用平台沐曦 PyTorch 环境，无需安装额外包：

```bash
sbatch scripts/slurm_gpu_probe.sh
# 同一节点申请两卡，依次测试每张可见卡：
sbatch --gres=gpu:2 scripts/slurm_gpu_probe.sh
# 只采集信息，不运行性能负载：
sbatch scripts/slurm_gpu_probe.sh --inventory-only
```

默认申请 1 张 GPU、4 个 CPU、16 GiB 主存，15 分钟时限。按集群要求附加
`--partition`、`--account` 等 sbatch 参数。脚本只测试当前作业可见的 GPU，
不改写 CUDA_VISIBLE_DEVICES。多卡测试是逐卡运行，不测集合通信或跨节点网络。
需要覆盖多个节点时，在各目标节点分别提交（例如 `sbatch --nodelist=节点名 ...`）。

默认加载 `conda-py31010/3.10.10`。可在提交前设置 `PROBE_MODULE` 和
`PROBE_PYTHON`（Python 可执行文件的绝对路径）。已分配的交互计算节点或 NVIDIA
环境也可以直接运行：

```bash
python scripts/probe_gpu.py
python scripts/probe_gpu.py --devices 0 --buffer-mib 512 --matrix-size 4096
python scripts/probe_gpu.py --inventory-only
```

输出默认位于 `runs/diagnostics/gpu_<主机>_<UTC时间>/`，可用 `--output 新目录` 指定。
控制台摘要见 `gpu_probe_<jobid>.out`，详细结果为：

- `report.json`：GPU 型号、总显存/初始空闲显存、PyTorch 和作业环境、每项实测结果及错误。
- `inventory.json`：`ht-smi`/`nvidia-smi` 原始信息；NVIDIA 额外保存详细查询及拓扑。
- `telemetry.jsonl`：测试期间约每秒一次的管理工具原始读数，含测试阶段和时间。
  可核对工具提供的核心/显存时钟、温度、功耗、利用率及 PCIe 状态。
  工具输出可能包含整个节点的卡，需按 UUID/PCI 总线号及作业分配匹配，不能将物理索引
  直接等同于 PyTorch 逻辑索引。短测试不保证每阶段都有采样。

沐曦各版本 `ht-smi` 支持的字段不同，因此保留默认输出和 `--help`，不猜测专有参数。
若默认输出缺少频率、总线宽度等字段，表示本次没有采集到，不能视为 0；
可依据保存的帮助信息进一步查询。PyTorch 属性中不存在的字段同样不填造。

## 测量口径

| 项目 | 方法与单位 |
| --- | --- |
| 显存容量 | 字节数，控制台换算 GiB（2^30 字节） |
| 显存有效带宽 | GPU 内复制；读+写计为 `2 × buffer_bytes / time`，十进制 GB/s |
| CPU → GPU / GPU → CPU | 固定页主存、单方向复制；`buffer_bytes / time`，十进制 GB/s |
| FP32 / FP16 / BF16 算力 | 方阵 GEMM；`2 × N³ / time`，十进制 TFLOP/s，禁用 TF32 |
| 波动 | 每批平均延迟的中位数、最小值、最大值及全部样本 |

默认每项预热 5 次，随后 5 批、每批 20 次，使用 GPU Event 计时并同步。
显存复制默认单缓冲区 256 MiB，最多使用初始可用显存的 10%/缓冲区。
若缓冲区可装入缓存，结果可能含缓存收益，可增大 `--buffer-mib` 对照。
传输测试使用固定页主存；NUMA 绑定及 CPU 拓扑会影响结果。
矩阵默认 N=4096，三个矩阵超过当时空闲显存的 50% 时报告错误，需减小尺寸。
复制检查全部元素，GEMM 检查有限值并对照 CPU FP64 计算的 8×8 子块。
不支持某精度或内存不足时保留该项错误并继续，最终退出码为 1；全部完成为 0。
没有 GPU 或 PyTorch 时也保留管理工具信息和失败原因。

这些结果是当前环境下的有效吞吐量，不等于规格表理论峰值，也不构成全显存故障扫描。
比较不同节点时保持参数和环境一致，优先使用独占空闲 GPU；结合负载期间时钟、
温度、功耗及其他进程占用解读差异。脚本不设置自动“达标”阈值。
