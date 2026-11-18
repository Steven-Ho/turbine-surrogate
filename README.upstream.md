# qiyuan-tfai

稳态场 AI 预测 Python 包，支持 `POD`、`UNet`、`GCN`、`Graph_UNet`、`Graph_Transformer`。安装到 Python 环境后，通过脚本完成训练、评估和预测。

## 1. 安装

要求 Python `>=3.8.1,<3.14`。以下命令均在 `qiyuan-dist` 目录执行。

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

先安装与目标机器 CPU/CUDA 环境匹配的 PyTorch，以及与其兼容的 `torch-geometric`。下面的 PyTorch 命令请根据目标 CUDA 环境调整：

```bash
python -m pip install --upgrade pip
python -m pip install torch torch-geometric
python -m pip install -r requirements-qiyuan.txt 
python -m pip install --no-deps dist/qiyuan_tfai-0.1.0-py3-none-any.whl --break-system-packages --force-reinstall
```

脚本通过当前 Python 环境导入 `tfai`，不会将源码目录添加到 `sys.path`。请使用安装 wheel 的同一环境运行脚本，并避免通过 `PYTHONPATH` 指向本地源码。

## 2. 配置

脚本默认分别加载以下文件：

- `configs/train_config.yaml`：训练配置。
- `configs/evaluate_config.yaml`：评估配置。
- `configs/predict_config.yaml`：预测配置。

编辑配置文件，或使用 Hydra 的 `键=值` 命令行参数覆盖配置。配置从脚本旁的 `../configs` 加载；数据、模型和输出的相对路径默认以 `qiyuan-dist` 为基准。可以通过环境变量 `TFAI_PROJECT_ROOT` 指定数据项目根目录，该变量不改变配置文件加载位置。

三份配置从仓库顶层 `configs` 复制而来。运行前请确认样本路径、输出变量、模型类型和实验目录；评估模板的实验目录与训练模板不同，下面统一覆盖为 `exp_demo`。本发行包仅支持上述五种模型的稳态场任务。

## 3. 训练

准备样本清单，在 `configs/train_config.yaml` 中设置 `general.train_samples`、`task.variables`、`model.type` 和训练参数，然后执行：

```bash
python scripts/train.py general.experiment=exp_demo model.type=POD
```

默认读取 `exp_demo/data/raw/samples.txt`，预处理数据写入 `exp_demo/data/processed`。最佳模型保存到 `exp_demo/train/best_model.pt`，续训检查点保存到 `exp_demo/train/last_checkpoint.pt`。

覆盖样本路径或训练轮数：

```bash
python scripts/train.py general.experiment=exp_demo general.train_samples=exp_demo/data/raw/samples.csv training.epochs=100
```

## 4. 评估

训练完成后执行：

```bash
python scripts/evaluate.py evaluate.experiment=exp_demo evaluate.split=valid
```

`evaluate.checkpoint_path` 留空时使用实验目录下的 `train/best_model.pt`。`evaluate.split` 可选 `train`、`valid`、`test`，需要模型中存在对应的数据划分。结果默认写入 `exp_demo/evaluate/valid`。

指定其他模型文件：

```bash
python scripts/evaluate.py evaluate.experiment=exp_demo evaluate.checkpoint_path=exp_demo/train/best_model.pt
```

## 5. 预测

准备预测样本清单，编辑 `configs/predict_config.yaml`，再执行：

```bash
python scripts/predict.py general.experiment=exp_demo
```

默认加载 `exp_demo/train/best_model.pt`，读取 `exp_demo/predict/data/raw/predict_samples.txt`，结果输出到 `exp_demo/predict/results`。

指定其他样本清单和模型文件（模板未声明 `prediction.checkpoint_path`，新增键使用 `+`）：

```bash
python scripts/predict.py general.experiment=exp_demo prediction.predict_samples=exp_demo/predict/data/raw/predict_samples.csv +prediction.checkpoint_path=exp_demo/train/best_model.pt
```

交付模型时，应同时保留 checkpoint 所需的样本资源和结果写回使用的文件。

## 6. 查看配置与重新打包

以下命令解析配置并退出，不启动训练、评估或预测：

```bash
python scripts/train.py --cfg job --resolve
python scripts/evaluate.py --cfg job --resolve
python scripts/predict.py --cfg job --resolve
```

重新生成 wheel 和源码包：

```bash
python -m pip install build
python -m build --outdir dist
```

源码包包含 `scripts` 和顶层 `configs`；wheel 安装 `tfai` 及其包内配置。部署时，将 `scripts`、`configs` 与实验数据目录放在一起，并在运行环境中安装 wheel；也可以解压源码包取得脚本和配置。
