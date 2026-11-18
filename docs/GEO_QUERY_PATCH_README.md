GeoTransolver 查询优化补丁。完整说明见 docs/geo_query_optimization.md。
先停止旧 Geo 作业，保留其他模型与已有结果；解压后验证 GEO_QUERY_OPT_SHA256SUMS。
先运行 scripts/slurm_verify_geo_query_opt.sh；看到 ALL CHECKS PASSED 后，再运行 scripts/slurm_train_geo_optimized.sh。
必须设置 TURBINE_SOURCE_RUN 为现有七模型结果目录；先 mkdir -p logs。
本机仅做小规模 CPU 正确性检查，没有运行 GPU 试验。加速倍数等待 X203 实测。
