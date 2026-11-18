# GeoTransolver 空间索引与持久邻居缓存补丁

## 更新范围

- tfai/models/_vendor/geotransolver2/ball_query.py：默认 kdtree 后端，保留原 torch 查询后端，原类名/参数布局/梯度 gather 不变。
- spatial_neighbors.py（同目录，新增）：SciPy cKDTree、几何内容哈希、有限容量内存缓存及原子磁盘缓存。
- UPSTREAM.json：记录算法版本和原文件 SHA256。
- tests/test_geo_spatial_cache.py：仅小规模 CPU 正确性检查。
- scripts/slurm_verify_geo_query_opt.sh：集群完整节点索引、模型输出和参数梯度对照。
- scripts/precompute_geo_neighbors.py：读取原训练/验证缓存，预计算邻居，不更新权重。
- scripts/slurm_train_geo_optimized.sh：单独从头训练优化后的 Geo，沿用 20->200->条件400 协议。

无需重新上传 PyTorch、wheel、OpenGL、原始 VTU 或其他模型源码。SciPy 已在原环境中可用。
其他六个模型不调用这份 Geo 查询，不必因本补丁重新训练。

## 原理及保持一致的边界

旧实现每次切换几何都会做两次全点对距离计算，缓存仅覆盖最近样本。
新实现用精确 cKDTree（eps=0）提供近邻候选，并保留半径包含边界、最近 K 和缺失补零。
距离近似并列或靠近半径边界时，以原来大小的 Torch 查询块回退，保留旧 topk 的顺序/精度。
若此类歧义很多，首次建缓存仍可能较慢；验证报告会列出 ambiguity_rows/reference_fallback_chunks。

每组真实 points/queries 坐标均参与哈希，不能把不同几何当成同一份索引。
键还包括 radius、K、dtype、Torch版本、GPU类型和分块大小，防止错误复用。
CPU内存缓存默认每层32份，磁盘缓存跨样本/epoch/进程复用；切回 A 不再强制重新搜索。
索引和有效性离散选择缓存后，每次仍从当前输入张量 gather，保持输入梯度；缓存不进入 state_dict。
文件通过临时文件+原子替换写入，读取时核验 ZIP CRC、键/shape/dtype/范围；坏缓存重新生成。

## 更新前

先停止旧 GeoTransolver 作业，避免下一阶段新 Python 子进程突然加载新代码，造成同一实验混用两版。
不要取消其他模型。示例 `scancel 数组作业号_6`，请用 squeue 确认实际作业号。
旧模型权重/运行目录保留，不会被补丁或新训练覆盖。

## 上传与验证

把压缩包放到项目根目录，执行：

```bash
mkdir -p logs
tar -xzf turbine-geo-query-opt-patch.tar.gz
sha256sum -c GEO_QUERY_OPT_SHA256SUMS
export TURBINE_SOURCE_RUN=runs/metax_long_20261111_100428_4166078
sbatch scripts/slurm_verify_geo_query_opt.sh
```

TURBINE_SOURCE_RUN 指向现有七模型结果目录，要求 GeoTransolver/data/processed 已完成预处理。
验证会读取两个真实完整样本，比较全部节点的两个半径索引/邻居坐标，再比较同权重模型输出和参数梯度。
它会创建/复用邻居缓存，但不会执行优化器更新，不修改旧训练缓存或权重。
本机只完成小规模 CPU 正确性检查，没有执行 GPU/完整模型试验；X203 需通过此作业验证。

查看 logs/geo_opt_verify_JOBID.out/.err，末尾应为 ALL CHECKS PASSED。
详细结果在 runs/diagnostics/geo_opt_verify_JOBID/：query_validation.json、cache_reuse_*.json、model_validation.json。
出现索引/数值不一致时，验证作业会失败；先保留日志排查，暂不提交优化长训。

## 验证通过后单独重训 Geo

```bash
unset TURBINE_LONG_RUN_ROOT
sbatch scripts/slurm_train_geo_optimized.sh
```

该作业申请 1 GPU、4 CPU、128G，日志写到 logs/geo_optimized_JOBID.out/.err。
默认先为原 train/valid 共900个样本生成缓存（单独进程、单独计时），再执行原 20->200->条件400 长训。
新结果为 runs/metax_geo_kdtree_JOBID/GeoTransolver，原七模型目录不覆盖。
父目录的 neighbor_precompute.json 保存预计算时间/缓存统计；geo_query_backend.json 保存算法和代码指纹。
首次缓存成本与后续训练循环时间应分别报告，不能把“热缓存前向反向”直接当端到端速度。

已有完整缓存时会复用；如需省略预计算，提交前 `export GEO_OPT_PRECOMPUTE=0`，训练中会按需建缓存。
原预处理与新训练归一化结果若不同，内容哈希会失效并自动重建，不错误复用。
首次建缓存是否明显加速及正式步骤吞吐，必须由集群报告确认，不能保证先前估算的倍数。

## 缓存位置、空间及回退

默认持久缓存为项目 runs/geo_neighbor_cache_kdtree_v1，可通过 TURBINE_GEO_CACHE_DIR 指定绝对路径。
900样本、29773节点、8+32邻居使用 int32 索引，磁盘约4 GiB，加少量文件头。
未压缩以避免每步解压CPU开销；默认每层32项的CPU内存缓存约数百MB以内。
验证与训练应使用相同缓存目录（且为计算节点可访问的持久存储）。缓存可重建，删除不会删除模型结果。
如果目录不可写或空间不足，建缓存会明确报错；请选择有足够空间的位置。

设置 TURBINE_GEO_QUERY_BACKEND=torch 可回退旧查询；新优化训练脚本会显式选择 kdtree。
测试代码可通过构造参数 backend='torch' 作对照；无需卸载或重装任何厂商包。
精确回退命令应在普通训练入口使用，或在优化训练脚本中更改 export 后端。
