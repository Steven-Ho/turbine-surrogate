# 阶段 3：单节点多卡训练与恢复

对应 `docs/x203_full_pretraining_plan.md` 阶段 3。先两卡，再单节点八卡；双节点属于阶段 6，使用其他脚本。

## 文件放置

将本目录上传到集群 `~/Projects/pde-fm/stage3/`。目录内须包含配置、launcher、sbatch 脚本。

## 两卡首次训练

从项目根目录提交：

```bash
cd ~/Projects/pde-fm
sbatch --export=ALL,PDE_FM_MODE=fresh,PDE_FM_GPUS_PER_NODE=2 stage3/stage3-single-node.sbatch
```

默认数据 `~/Dataset/the_well`，venv `~/venvs/pde-fm-metax`。
作业输出打印绝对 `PDE_FM_RUN_ROOT`，默认 `runs/pde-fm/stage3-<首次作业号>`。
配置：active_matter，tiny Focused Linear + Wavelet + Fourier + Muon，FP16 AMP，
每 rank batch=2，累积=2，每 rank 采样预算 max_samples=32，训练至 epoch 2。
两卡名义有效 batch=8。验证/rollout 范围刻意缩小，仅作功能验收。

## 第二个作业：恢复到 epoch 3

首次作业完成且有 `PDE_FM_STAGE3_OK mode=fresh` 后，将首次作业号代入：

```bash
export PDE_FM_RUN_ROOT="$HOME/Projects/pde-fm/runs/pde-fm/stage3-首次作业号"
sbatch --export=ALL,PDE_FM_MODE=resume,PDE_FM_GPUS_PER_NODE=2 stage3/stage3-single-node.sbatch
```

续训须使用同一根目录和相同 GPU 数。脚本会检查 fresh.ok、旧检查点、恢复消息和
`Epoch 3/3: starting training`。会继续写入原 training/，请勿与另一作业同时使用该目录。
恢复作业失败后，请先检查实际检查点进度，再决定重试方式；不要删除成功标记强行重跑。

## 验收

- 两个作业全部 rank 均输出 `PDE_FM_RANK_TRAIN_OK`，损失有限，无 collective 卡死或运行时错误。
- 首次输出 `PDE_FM_STAGE3_OK mode=fresh`，保存 step_2。
- 恢复日志明确加载检查点并从 epoch 3 开始，最终输出 `PDE_FM_STAGE3_OK mode=resume`，保存 step_3。
- 两次均有 `PDE_FM_CHECKPOINT_OK`，epoch 正确且模型和优化器状态非空。
- 保留 .out/.err 与整个 run root，提供结果后按 V004 等编号写入验证记录。

这些检查证明基本分布式训练和模型/优化器恢复流程；不证明跨 rank 参数逐项一致、
与单卡数值等价、完整随机数/采样器/scaler 恢复或收敛质量。
现有调度器按 max_epoch 和恢复 epoch 重建，本次将上限 2 改为 3，不能用于严格连续训练等价性测试。

## 扩展到八卡

两卡训练及恢复通过后，使用全新目录执行：

```bash
unset PDE_FM_RUN_ROOT
sbatch --gres=gpu:8 --cpus-per-task=16 --mem=128G \
  --export=ALL,PDE_FM_MODE=fresh,PDE_FM_GPUS_PER_NODE=8 stage3/stage3-single-node.sbatch
```

恢复时设置对应八卡 run root，并保持相同资源申请和 `PDE_FM_GPUS_PER_NODE=8`。
八卡名义有效 batch=32，不能直接与两卡结果比较精度/强扩展效率。
默认 OMP/MKL 每 rank 1 线程、data_workers=0，适合排错，尚未调优。

## 实现说明

`distribution=ddp` 在当前项目实际为 FSDP NO_SHARD，并保留完整参数副本；Muon 不走 FSDP/HSDP 分片。
launcher 在设备网格初始化前设置 local GPU，避免多个进程先初始化同一默认设备。
Hydra 配置从 stage3 加载，引用仓库已有配置组，不需要修改 Python 包名或安装元数据。
配置及脚本按作业复制到 jobs/<jobid>/，同时保存源码提交、环境和输入合成配置。
本地仅完成配置合成、Shell/Python 语法检查；集群训练结果待验证。
