# GeoTransolver 本机与集群差异核查（2026-11-12）

本次仅加载本机检查点到CPU并分析日志、CSV和配置，未使用GPU，未运行模型前向、训练或推理。

## 已确认

- 对比本机20轮与400轮检查点内config、集群stage_20/stage_400.yaml：网络类型GeoTransolver、small规模、in_channels=5、out_channels=1、seed=42、deterministic=false、batch=1、损失配置及长训调度配置未发现改变。配置中的model.geotransolver_backend存在差异：本机默认原版PhysicsNeMo/Warp，集群显式portable_torch。设备选择为auto与cuda:0，文件路径也不同。
- 首20轮本机require_preprocess_cache=true，集群false：本机复用已准备缓存，集群重新预处理。这一开关本身不是数据变换参数差异，但归一化数值和处理结果是否完全一致，必须继续比较实际产物。
- 集群900样本pool与本机effective_train+effective_valid样本集合一致；独立测试100样本集合已在前一报告核验一致。不能由此证明最终800/100内部成员及顺序一致，集群effective表未在现有日志包中提供。本机实际effective表已导出。
- 首20轮两边学习率均固定1e-4。后续学习率下降时机不同，且集群第2次之后总体较晚降到较低学习率。

## 学习率变化（验证结束后的新值，作用于随后训练）

|降至|本机轮数|集群轮数|
|---|---:|---:|
|5e-5|70|70|
|2.5e-5|114|128|
|1.25e-5|145|146|
|6.25e-6|161|171|
|3.125e-6|182|194|
|1.5625e-6|195|209|
|1e-6|209|229|

学习率变化可影响后续优化，但并非最初差异的唯一来源：两边还未发生学习率下降时，第1轮验证loss已为0.012152（本机）与0.013349（集群）；第20轮为0.001260与0.000753。第70轮接近（0.000319与0.000314），之后又出现差异。不能简单将最终12.32%的RMSE改善归因为“集群多用20轮较大学习率”。

## 验证曲线取点

|轮数|本机验证MSE|集群验证MSE|
|---|---:|---:|
|1|0.012152|0.013349|
|20|0.001260|0.000753|
|70|0.000319|0.000314|
|100|0.000121|0.000106|
|128|0.000081|0.000080|
|200|0.000058|0.000045|
|400|0.000056|0.000041|

以上日志仅6位小数。不同归一化参数未排除前，不把两个归一化MSE的比值当作严格物理误差比；统一Pa单位测试RMSE已确认1900.91→1666.61。

## 归一化及逐工况：现有资料的边界

本机checkpoint.norm_params已导出local_norm_params.json。Pressure采用min_max，训练目标min=8079.0087890625 Pa、max=268580.03125 Pa，mean约111134.890559、std约35201.240453。坐标、输入参数及MAPE尺度等完整值在JSON中。

集群日志只记录了norm_params.pt保存位置，不包含实际数值；现有runs.zip、汇总CSV和comparison.json均不能替代该文件/检查点中的norm_params。因此当前不能确认两边归一化参数一致，也没有证据显示它们不同。

本机100个测试工况指标已从既有报告提取到local_per_sample_metrics.csv。集群只有整体汇总，无法判断改善是否覆盖多数工况，还是主要来自少数高误差工况；无法进行工况配对bootstrap。整体指标下降不能推出每个工况都改善。

## 已准备补充导出脚本

slurm_export_geo_audit.sh读取已有集群checkpoint及预测VTU，不训练、不推理：

- checkpoint_config.json：训练实际检查点配置；
- checkpoint_norm_params.json / disk_norm_params.json：检查点与磁盘归一化参数；
- effective_splits.json：最终train/valid/test成员、顺序及表内参数，源目录前缀归一到文件名；
- per_sample_metrics.csv：沿用既有统一节点指标函数和同一MAPE阈值，对100个已保存预测计算指标；
- metadata.json：检查点步数、精确验证loss、训练/验证manifest、环境及总体评估指标。

上传scripts/slurm_export_geo_audit.sh，从项目根目录mkdir -p logs后sbatch提交。成功日志显示PASS Geo audit export及Download路径，输出runs/metax_geo_kdtree_1455/geo_audit_作业号.tar.gz。脚本申请1GPU以沿用已验证的厂商运行库环境，但Python检查点显式map_location=cpu，分析不执行GPU算子。

拿到这一小包后，可以完成最终800/100成员和顺序、归一化逐字段数值比较、100工况配对误差及bootstrap。当前结论是：主要超参数未发现变化，查询后端与优化轨迹确有差异；归一化、内部划分及逐工况分布仍待数据，不应提前归因于某一个算子。
