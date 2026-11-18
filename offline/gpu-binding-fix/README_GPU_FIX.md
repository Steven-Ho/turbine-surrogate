# GPU 作业绑定和日志修正

覆盖 scripts/slurm_train_all_metax.sh、run_cluster_long.py、submit_cluster_long.sh、
slurm_compare_metax.sh 和 slurm_train_metax.sh。

移除从 SLURM_JOB_GPUS/SLURM_STEP_GPUS 手工生成 CUDA_VISIBLE_DEVICES 的逻辑，
改用 srun --ntasks=1 --gres=gpu:1 启动训练步骤，并保留调度器提供的设备环境。
仍然是每模型 1 GPU，最多同时 7 模型；不把所有任务固定到物理 GPU 0。
这修复了原脚本的潜在编号混用问题，但集群真机绑定仍需重新提交后验证。

先取消原训练数组和其关联汇总作业（用实际作业号替换）：
scancel ORIGINAL_ARRAY_JOB_ID ORIGINAL_SUMMARY_JOB_ID

从项目根目录解压补丁：
tar -xzf /path/to/turbine-gpu-binding-fix.tar.gz
sha256sum -c GPU_BINDING_FIX_SHA256SUMS

全部七模型重新训练，避免继承旧结果路径：
unset TURBINE_LONG_RUN_ROOT
bash scripts/submit_cluster_long.sh

提交入口在 sbatch 前创建 logs/。若直接 sbatch，请先 mkdir -p logs。
新的调度日志：logs/turbine_long_ARRAYID_TASKID.out / .err；
汇总日志：logs/turbine_compare_JOBID.out / .err。
单模型训练日志：logs/train_gpu_JOBID.out / .err。
模型详细训练输出仍保存在各自结果目录的 train_20_console.log、
train_200_console.log、train_400_console.log，以及 train/、training_history.csv 等。

启动时会输出节点、GPU 环境及阶段开始/结束信息。每模型还会保存
GPU 诊断 gpu_preflight.json 与 ht-smi.txt。若仍失败，请提供这些文件和 logs/ 对应日志。
原任务 .out/.err 为空不等于没运行，旧版本将训练输出重定向到模型目录。

Slurm 官方说明：
https://slurm.schedmd.com/gres.html
https://slurm.schedmd.com/srun.html
