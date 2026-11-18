pde-fm X203 离线依赖包
目标：Linux x86_64 / CPython 3.10 / glibc >= 2.28。
可用 getconf GNU_LIBC_VERSION 检查节点 glibc。保留集群 torch 2.6.0+mars3.5.3.9。
The Well 固定源码提交：ad50de0e8861380d7c4d62c0503c654710f026e7。
不含项目源码、数据、权重，以及 torch/torchvision/torchaudio/Triton/NVIDIA 库。
源码从 GitHub 单独 clone/pull；本包替代在线依赖下载。

1. 上传 pde-fm-x203-py310-wheels.tar.gz 到集群并解压。
2. 在已分配 GPU 的计算节点执行以下步骤，避免登录节点缺少 libhcruntime.so：

module purge
module load conda-py31010/3.10.10
python -m venv --system-site-packages ~/venvs/pde-fm-metax
source ~/venvs/pde-fm-metax/bin/activate
cd /解压路径/pde-fm-x203
bash install.sh
cd ~/Projects/pde-fm
python -m pip install --no-index --no-deps --no-build-isolation -e .
python -c 'import walrus.train, torch; print(torch.__version__, torch.cuda.is_available())'

若环境已经创建，跳过 python -m venv，直接激活。Slurm 探针/安装任务显式申请
--mem=16G 或按实际需要增加，不要依赖集群默认内存。
install.sh 先检查厂商 torch 和 torchvision 可导入；缺少 torchvision 时必须向
平台取得与厂商 torch 匹配的版本，不要安装官方 CUDA wheel。
安装全部 wheel 使用 --no-deps，不访问网络，也不替换厂商 GPU 包。
requirements-pde-fm-offline.txt 用于记录顶层依赖，不要无约束重新解析安装。
resolved-packages.txt 记录实际 wheel 版本，BUILD_INFO.json 记录来源。
在干净 Python 3.10 环境完成离线安装与非 GPU 依赖检查；X203 算子、AMP、Muon、
FFT、完整短训及检查点恢复需在集群验证。本地验证不能替代真机测试。

项目 pyproject.toml 仍固定 torch 2.5.1 等原开发版本，所以 pip check 可能报告
厂商版本与项目元数据不一致。应逐条检查其他依赖错误，不能忽略所有报告。
