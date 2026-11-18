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
    from tfai.mains import load_model_for_prediction, run_predict  # noqa: E402
    from tfai.utils.path_utils import resolve_path_from_base  # noqa: E402
    from tfai.utils.runtime_paths import resolve_runtime_paths  # noqa: E402
except ModuleNotFoundError as exc:
    if exc.name != "tfai":
        raise
    sys.path.insert(0, str(PROJECT_ROOT))
    from tfai.mains import load_model_for_prediction, run_predict  # noqa: E402
    from tfai.utils.path_utils import resolve_path_from_base  # noqa: E402
    from tfai.utils.runtime_paths import resolve_runtime_paths  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="predict_config")
def main(cfg: DictConfig) -> None:
    """将命令行配置适配为后端包预测接口。"""
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("预测配置必须解析为字典")

    config = resolve_runtime_paths(config, project_root=str(PROJECT_ROOT))
    general_cfg = config["general"]
    prediction_cfg = config["prediction"]
    experiment_dir = resolve_path_from_base(
        str(PROJECT_ROOT),
        str(general_cfg["experiment"]),
    )
    checkpoint_path = str(prediction_cfg.get("checkpoint_path") or "").strip()
    if not checkpoint_path:
        checkpoint_path = os.path.join(experiment_dir, "train", "best_model.pt")
    else:
        checkpoint_path = resolve_path_from_base(str(PROJECT_ROOT), checkpoint_path)

    model_bundle = load_model_for_prediction(
        model_path=checkpoint_path,
        config=config,
    )
    run_predict(
        config=config,
        model_bundle=model_bundle,
        run_dir=experiment_dir,
    )


if __name__ == "__main__":
    main()
