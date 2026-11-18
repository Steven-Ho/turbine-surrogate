# 沐曦集群代码包

此包包含 tfai 源码、configs、训练/评估/预处理 Python 脚本、环境检查脚本、
原始 Rotor37 样本表和安装元数据。数据、权重、历史结果和 wheel 包需单独传输。
原 README 中的 NVIDIA/CUDA 环境安装步骤不适用于此集群；沿用已验证的厂商 PyTorch。

## 目录与数据

解压后将 Rotor37 的 sample_*.vtu 放到 turbine-surrogate/data/rotor37/ 下。
也可从项目根目录执行：

```bash
mkdir -p data
ln -s /实际数据目录/rotor37 data/rotor37
```

样本表 exp_rotor37_GCN/data/raw/samples.csv 已包含。预处理脚本会核验表中全部
1000 个样本是否存在，并生成 runs/rotor37 和 runs/rotor37_smoke 的划分表。
本包不包含定子数据；定子相关脚本须另行提供其输入数据。

## 安装与校验

先按离线 wheel 包说明，补齐 ~/venvs/turbine-metax 中的依赖。
在项目根目录执行：

```bash
module purge
module load conda-py31010/3.10.10
source ~/venvs/turbine-metax/bin/activate
python -m pip install --no-index --no-deps --no-build-isolation -e .
sha256sum -c CODE_SHA256SUMS
sbatch scripts/slurm_metax_venv_check.sh
```

## OpenGL 运行库

在项目根目录解压 turbine-opengl-jammy.tar.gz，得到 turbine-opengl-jammy/load.sh。
训练脚本会在 module load 之后自动加载它，并在训练前检查 VTK 导入。
若运行库位于其他位置，提交前设置 TURBINE_OPENGL_RUNTIME 为该目录的绝对路径。
运行库需另外上传，不包含在此代码包内。

## 训练

从解压后的项目根目录提交：

```bash
sbatch scripts/slurm_train_metax.sh
```

默认 train_smoke 配置只执行两个优化器步骤，验证训练流程；它会写入
runs/rotor37_smoke。该脚本沿用平台已成功的两卡资源申请，训练仅使用第一张卡，
尚不支持双卡 DDP；不是双卡性能测试。平台允许时可将 --gres=gpu:2 改为 gpu:1。
已有同名实验输出时请先备份，训练可能覆盖该实验文件。

完整训练：

```bash
export TURBINE_CONFIG=train_config
sbatch scripts/slurm_train_metax.sh
```

默认完整训练 100 epochs，写入 runs/rotor37。自定义虚拟环境可设置 TURBINE_VENV。

包内保留全部模型源码和配置，但这不表示每个可选模型在沐曦平台已验证。
当前离线 wheel 包不含 GeoTransolver 的 nvidia-physicsnemo 依赖，也不含
PyG 二进制扩展；Graph_UNet 等模型若要求额外算子，需要平台适配后单独验证。
本地只验证代码包完整性和训练配置加载，GPU 训练必须在集群验证。

## 七模型并行长训练

在项目根目录执行 `bash scripts/submit_cluster_long.sh`。
七个独立单卡数组任务按 20 -> 200 -> 条件延长 400 epoch 协议运行，随后自动汇总。
GeoTransolver 已加入 portable_torch 源码后端，长训程序自动选择，不需要 PhysicsNeMo/Warp wheel。
完整参数、输出与本机计时比较口径见 docs/metax_parallel_long.md。
