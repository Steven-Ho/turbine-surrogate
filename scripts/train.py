import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(
    os.environ.get("TFAI_PROJECT_ROOT", Path(__file__).resolve().parent.parent)
).resolve()
os.environ.setdefault("TFAI_PROJECT_ROOT", str(PROJECT_ROOT))

try:
    from tfai.mains import read_train_samples, run_train, split_train_test  # noqa: E402
    from tfai.utils.runtime_paths import resolve_runtime_paths  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != "tfai":
        raise
    sys.path.insert(0, str(PROJECT_ROOT))
    from tfai.mains import read_train_samples, run_train, split_train_test  # noqa: E402
    from tfai.utils.runtime_paths import resolve_runtime_paths  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """将命令行配置适配为后端包训练接口。"""
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("训练配置必须解析为字典")

    # Hydra 会切换运行目录，因此先相对项目根目录解析训练相关路径。
    config = resolve_runtime_paths(config, project_root=str(PROJECT_ROOT))
    general_cfg = config["general"]
    task_cfg = config["task"]

    samples_all, _ = read_train_samples(
        train_samples=general_cfg["train_samples"],
        is_geometry_varying=bool(task_cfg["is_geometry_varying"]),
        mode=str(task_cfg["mode"]),
        config=config,
    )
    #samples_train, _ = split_train_test(samples_all=samples_all, config=config)

    run_train(
        config=config,
        samples_train=samples_all,
        experiment_dir=general_cfg["experiment"],
        model_type=config["model"]["type"],
        train_params={},
    )


if __name__ == "__main__":
    main()
