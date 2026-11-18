from __future__ import annotations

import argparse
import importlib
# Imports `json` for reuse in this module.
import json
import platform
import sys
import tempfile
# Imports `contextmanager` from `contextlib` for use in this module.
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path
# Imports `Any, Dict, Iterator` from `typing` for use in this module.
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import patch


REQUIRED_DEPENDENCIES: List[Tuple[str, str]] = [
    ("torch", "torch"),
    # Completes the multi-line `collection` arguments or elements.
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("matplotlib", "matplotlib"),
    # Completes the multi-line `collection` arguments or elements.
    ("hydra", "hydra-core"),
    ("h5py", "h5py"),
    ("vtk", "vtk"),
    ("scipy", "scipy"),
    # Completes the multi-line `collection` arguments or elements.
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("einops", "einops"),
]

# Initializes the `OPTIONAL_DEPENDENCIES` as a collection for values from this processing stage.
OPTIONAL_DEPENDENCIES: List[Tuple[str, str]] = [
    ("torch_geometric", "torch-geometric"),
]

ROOT_EXPORT_NAMES: List[str] = [
    # Completes the multi-line `collection` arguments or elements.
    "load_train_config",
    "load_evaluate_config",
    "load_predict_config",
    "read_train_samples",
    # Completes the multi-line `collection` arguments or elements.
    "apply_variable_selection",
    "split_train_test",
    "run_train",
    "run_evaluate",
    # Completes the multi-line `collection` arguments or elements.
    "load_evaluate_results",
    "load_model_for_prediction",
    "run_predict",
]


# Defines `_sanitize_for_json` for  sanitize for json logic.
def _sanitize_for_json(value: Any):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        # Returns `str(value)` to the caller for the next stage.
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        # Returns `[_sanitize_for_json(item) for item in value]` to the caller for the next stage.
        return [_sanitize_for_json(item) for item in value]
    return repr(value)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Calls `path.write_text` to perform the operation required at this stage.
    path.write_text(
        json.dumps(_sanitize_for_json(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Defines `_get_distribution_version` for  get distribution version logic.
def _get_distribution_version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        # Returns `"unknown"` to the caller for the next stage.
        return "unknown"


def _check_dependency(import_name: str, dist_name: str, required: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "import_name": import_name,
        # Completes the multi-line `collection` arguments or elements.
        "dist_name": dist_name,
        "required": required,
    }
    try:
        # Stores the `module` for use by the remaining steps in this flow.
        module = importlib.import_module(import_name)
        result["status"] = "ok"
        result["version"] = _get_distribution_version(dist_name)
        result["module_path"] = getattr(module, "__file__", None)
    # Continues the surrounding statement or collection definition.
    except Exception as exc:  # pragma: no cover - 依赖缺失由安装环境决定
        result["status"] = "missing"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# Defines `_is_relative_to` for  is relative to logic.
def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    # Continues the surrounding statement or collection definition.
    except ValueError:
        return False


@contextmanager
def _temporary_sys_path(path: Path | None) -> Iterator[None]:
    # Checks `path is None` before entering this conditional branch.
    if path is None:
        yield
        return

    resolved = str(path.resolve())
    # Stores the `inserted` for use by the remaining steps in this flow.
    inserted = False
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        inserted = True
    # Keeps the operation and its recovery path within the same exception boundary.
    try:
        yield
    finally:
        if inserted and resolved in sys.path:
            # Calls `sys.path.remove` to perform the operation required at this stage.
            sys.path.remove(resolved)


def _resolve_report_base_dir(report_base_dir: Path | None) -> Path:
    if report_base_dir is not None:
        return Path(report_base_dir).resolve()
    # Returns `(Path.cwd() / "reports" / "installed_package_smoke").resolve()` to the caller for the next stage.
    return (Path.cwd() / "reports" / "installed_package_smoke").resolve()


def _infer_repo_root_from_module_file(module_file: Path) -> Path | None:
    """根据模块文件位置推断当前是否运行在仓库源码树内。"""

    for candidate in module_file.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "tfai").is_dir():
            # Returns `candidate` to the caller for the next stage.
            return candidate
    return None


def _detect_import_source(
    tfai_file: Path,
    # Continues the surrounding statement or collection definition.
    project_root: Path | None,
) -> Tuple[str, Path | None]:
    """识别 tfai 是从已安装包导入还是从仓库源码导入。"""

    candidate_roots: List[Path] = []
    if project_root is not None:
        # Adds the current value to `candidate_roots` for later aggregation.
        candidate_roots.append(project_root)

    inferred_repo_root = _infer_repo_root_from_module_file(Path(__file__).resolve())
    if inferred_repo_root is not None and inferred_repo_root not in candidate_roots:
        candidate_roots.append(inferred_repo_root)

    # Visits each item in `candidate_roots` and applies the current processing step.
    for candidate_root in candidate_roots:
        if _is_relative_to(tfai_file, candidate_root):
            return "local_source", candidate_root
    return "installed_package", None


# 执行 tfai 安装后自检。
def run_smoke_check(
    output_dir: Path | None = None,
    project_root: Path | None = None,
    strict_dependency_check: bool = True,
    # Continues the surrounding statement or collection definition.
    report_base_dir: Path | None = None,
) -> Dict[str, Any]:
    """执行 tfai 安装后自检。"""

    resolved_report_base_dir = _resolve_report_base_dir(report_base_dir)
    report_dir = (
        # Completes the multi-line `Path(output_dir).resolve` arguments or elements.
        Path(output_dir).resolve()
        if output_dir is not None
        else (resolved_report_base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    )
    # Calls `report_dir.mkdir` to perform the operation required at this stage.
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "summary.json"

    local_project_root = Path(project_root).resolve() if project_root is not None else None
    mode = "local_source" if local_project_root is not None else "installed_package"
    # Stores the `dependency_policy` for use by the remaining steps in this flow.
    dependency_policy = "strict" if strict_dependency_check else "warning_only"

    errors: List[str] = []
    warnings: List[str] = []

    dependency_results: List[Dict[str, Any]] = []
    # Visits each item in `REQUIRED_DEPENDENCIES` and applies the current processing step.
    for import_name, dist_name in REQUIRED_DEPENDENCIES:
        result = _check_dependency(import_name, dist_name, required=True)
        dependency_results.append(result)
        if result["status"] != "ok":
            # Stores the `message` for use by the remaining steps in this flow.
            message = f"必需依赖导入失败: {dist_name} -> {result['error']}"
            if strict_dependency_check:
                errors.append(message)
            else:
                # Adds the current value to `warnings` for later aggregation.
                warnings.append(message)

    for import_name, dist_name in OPTIONAL_DEPENDENCIES:
        result = _check_dependency(import_name, dist_name, required=False)
        dependency_results.append(result)
        # Checks `result["status"] != "ok"` before entering this conditional branch.
        if result["status"] != "ok":
            warnings.append(f"可选依赖未就绪: {dist_name} -> {result['error']}")

    tfai_info: Dict[str, Any] = {}
    api_check: Dict[str, Any] = {}
    # Initializes the `config_check` as a collection for values from this processing stage.
    config_check: Dict[str, Any] = {}
    split_check: Dict[str, Any] = {}
    api_input_examples: Dict[str, Any] = {}

    try:
        # Scopes resource use so the context manager can release it on exit.
        with _temporary_sys_path(local_project_root):
            import tfai
            from tfai import (
                apply_variable_selection,
                # Continues the surrounding statement or collection definition.
                load_evaluate_config,
                load_model_for_prediction,
                load_predict_config,
                load_train_config,
                # Continues the surrounding statement or collection definition.
                run_predict,
                split_train_test,
            )

        tfai_file = Path(tfai.__file__).resolve()
        # Stores the `(import_source, detected_source_root)` for use by the remaining steps in this flow.
        import_source, detected_source_root = _detect_import_source(
            tfai_file=tfai_file,
            project_root=local_project_root,
        )
        # Checks `import_source == "local_source"` before entering this conditional branch.
        if import_source == "local_source":
            warnings.append("当前使用本地源码模式导入 tfai，本次结果用于仓库内回归验证。")
            if local_project_root is None:
                warnings.append(
                    # Completes the multi-line `warnings.append` arguments or elements.
                    "未显式传入 --project-root，但检测到当前从仓库源码导入 tfai；"
                    "若要验证已安装包，请离开仓库目录或改用隔离环境执行。"
                )

        tfai_info = {
            # Completes the multi-line `str` arguments or elements.
            "module_path": str(tfai_file),
            "version": _get_distribution_version("tfai"),
            "import_source": import_source,
            "source_root": str(detected_source_root) if detected_source_root is not None else None,
        # Completes the multi-line `collection` arguments or elements.
        }

        api_check["root_exports"] = {
            name: callable(getattr(tfai, name, None))
            for name in ROOT_EXPORT_NAMES
        # Continues the surrounding statement or collection definition.
        }

        with tempfile.TemporaryDirectory(prefix="tfai_smoke_") as tmp_dir:
            experiment_dir = Path(tmp_dir) / "exp_smoke"
            train_config = load_train_config()
            # Stores the `evaluate_config` for use by the remaining steps in this flow.
            evaluate_config = load_evaluate_config()
            predict_config = load_predict_config()

            samples_all = [
                {"order": 101, "params": {"velocity": 5.0, "heatPower": 100.0, "unused": 1.0}},
                # Completes the multi-line `collection` arguments or elements.
                {"order": 102, "params": {"velocity": 6.0, "heatPower": 120.0, "unused": 2.0}},
                {"order": 103, "params": {"velocity": 7.0, "heatPower": 140.0, "unused": 3.0}},
            ]
            samples_all = apply_variable_selection(
                # Completes the multi-line `apply_variable_selection` arguments or elements.
                config=train_config,
                samples_all=samples_all,
                input_variables=["velocity", "heatPower"],
                output_variables=["Pressure", "Velocity"],
            # Completes the multi-line `apply_variable_selection` arguments or elements.
            )
            if any("unused" in item["params"] for item in samples_all):
                raise RuntimeError("变量选择未移除未选输入变量")
            train_config["general"]["seed"] = 7
            # Stores the `(samples_train, samples_test)` for use by the remaining steps in this flow.
            samples_train, samples_test = split_train_test(
                samples_all=samples_all,
                config=train_config,
                test_ratio=0.34,
            # Completes the multi-line `split_train_test` arguments or elements.
            )

            config_check = {
                "requested_train_experiment": str(experiment_dir),
                "train_experiment": train_config["general"]["experiment"],
                # Completes the multi-line `collection` arguments or elements.
                "train_samples": train_config["general"]["train_samples"],
                "evaluate_split": evaluate_config["evaluate"]["split"],
                "predict_batch_size": predict_config["prediction"]["batch_size"],
                "predict_experiment": predict_config["general"]["experiment"],
                # Completes the multi-line `collection` arguments or elements.
                "selected_input_variables": train_config["task"]["input_variables"],
                "selected_output_variables": train_config["task"]["output_variables"],
            }
            split_check = {
                # Completes the multi-line `len` arguments or elements.
                "sample_count": len(samples_all),
                "train_count": len(samples_train),
                "test_count": len(samples_test),
                "train_orders": [item["order"] for item in samples_train],
                # Completes the multi-line `collection` arguments or elements.
                "test_orders": [item["order"] for item in samples_test],
            }
            predict_contract_check: Dict[str, Any] = {}

            class SmokePredictor:
                # Defines `__init__` for   init   logic.
                def __init__(
                    self,
                    checkpoint_path,
                    predict_config=None,
                    # Continues the surrounding statement or collection definition.
                    logger=None,
                    defer_reader_initialization=False,
                    **kwargs,
                ):
                    # Stores the checkpoint path on the instance so other methods can reuse it.
                    self.checkpoint_path = checkpoint_path
                    self.config = predict_config or {}
                    self.logger = logger
                    self.defer_reader_initialization = defer_reader_initialization
                    # Stores the `batch_size` on the instance so other methods can reuse it.
                    self.batch_size = int(self.config.get("prediction", {}).get("batch_size", 8))
                    self.is_geometry_varying = True
                    self.last_writeback_report = {
                        "strict": False,
                        # Completes the multi-line `collection` arguments or elements.
                        "status": "success",
                        "expected_count": 0,
                        "success_count": 0,
                        "failed_count": 0,
                        # Completes the multi-line `collection` arguments or elements.
                        "failures": [],
                    }

                def predict(self, samples_csv, processed_dir, output_dir=None):
                    from tfai.readers.sample_map import SampleMap

                    # Stores the `output_root` for use by the remaining steps in this flow.
                    output_root = Path(output_dir or processed_dir)
                    output_root.mkdir(parents=True, exist_ok=True)
                    sample_map = SampleMap(samples_csv)
                    result_files: List[str] = []
                    # Visits each item in `sample_map.get_all_samples()` and applies the current processing step.
                    for sample_id in sample_map.get_all_samples():
                        output_path = output_root / f"smoke_predicted_{sample_id}.vtu"
                        output_path.write_text(f"smoke-{sample_id}", encoding="utf-8")
                        result_files.append(str(output_path))
                    # Stores the `last_writeback_report` on the instance so other methods can reuse it.
                    self.last_writeback_report = {
                        "strict": False,
                        "status": "success",
                        "expected_count": len(result_files),
                        # Completes the multi-line `len` arguments or elements.
                        "success_count": len(result_files),
                        "failed_count": 0,
                        "failures": [],
                    }
                    # Returns `result_files` to the caller for the next stage.
                    return result_files

            lazy_checkpoint_path = experiment_dir / "train" / "best_model.pt"
            lazy_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            lazy_checkpoint = {
                # Completes the multi-line `collection` arguments or elements.
                "config": {
                    "general": {
                        "experiment": str(experiment_dir),
                        "train_samples": str(experiment_dir / "data" / "raw" / "predict_samples_missing.csv"),
                    # Completes the multi-line `collection` arguments or elements.
                    },
                    "task": {
                        "type": "field",
                        "mode": "steady",
                        # Completes the multi-line `collection` arguments or elements.
                        "grid_type": "non-cartesian",
                        "is_geometry_varying": True,
                        "variables": ["Velocity"],
                        "effective_variables": ["velocity"],
                    # Completes the multi-line `collection` arguments or elements.
                    },
                    "model": {
                        "type": "UNet",
                        "in_channels": 1,
                        # Completes the multi-line `collection` arguments or elements.
                        "out_channels": 1,
                        "img_size": 16,
                    },
                    "prediction": {},
                # Completes the multi-line `collection` arguments or elements.
                },
                "model_state_dict": {},
                "extra_info": {"job_id": "smoke-lazy-predict"},
            }
            # Imports `torch` for reuse in this module.
            import torch

            torch.save(lazy_checkpoint, lazy_checkpoint_path)

            with patch("tfai.mains.model_load.Predictor", SmokePredictor):
                model_info = load_model_for_prediction(
                    # Completes the multi-line `str` arguments or elements.
                    model_path=str(lazy_checkpoint_path),
                    config=load_predict_config(),
                )

            predict_contract_check = {
                # Completes the multi-line `bool` arguments or elements.
                "defer_reader_initialization": bool(
                    getattr(model_info["model_instance"], "defer_reader_initialization", False)
                ),
                "train_samples": model_info["model_instance"].config.get("general", {}).get("train_samples", None),
                # Completes the multi-line `collection` arguments or elements.
                "parameter_schema": model_info["parameter_schema"],
                "prediction_context": model_info["prediction_context"],
            }

            predict_source_dir = experiment_dir / "predict_sources"
            # Calls `predict_source_dir.mkdir` to perform the operation required at this stage.
            predict_source_dir.mkdir(parents=True, exist_ok=True)
            predict_source_1 = predict_source_dir / "case_201.vtu"
            predict_source_2 = predict_source_dir / "case_202.vtu"
            predict_source_1.write_text("smoke-case-201", encoding="utf-8")
            # Calls `predict_source_2.write_text` to perform the operation required at this stage.
            predict_source_2.write_text("smoke-case-202", encoding="utf-8")

            predict_result = run_predict(
                config=load_predict_config(),
                model_bundle=model_info,
                # Completes the multi-line `str` arguments or elements.
                run_dir=str(experiment_dir / "predict_request"),
                samples_predict=[
                    {
                        "order": 201,
                        # Completes the multi-line `collection` arguments or elements.
                        "params": {"velocity": 8.0, "heatPower": 160.0},
                        "source_file": str(predict_source_1),
                    },
                    {
                        # Completes the multi-line `collection` arguments or elements.
                        "order": 202,
                        "params": {"velocity": 9.0, "heatPower": 170.0},
                        "source_file": str(predict_source_2),
                    },
                # Completes the multi-line `collection` arguments or elements.
                ],
            )
            predict_contract_check["run_predict"] = {
                "sample_count": predict_result["sample_count"],
                # Completes the multi-line `collection` arguments or elements.
                "result_files": predict_result["result_files"],
                "samples_csv": predict_result["samples_csv"],
            }

            api_input_examples = {
                # Completes the multi-line `collection` arguments or elements.
                "load_train_config": {},
                "load_evaluate_config": {},
                "load_predict_config": {},
                "read_train_samples": {
                    # Completes the multi-line `collection` arguments or elements.
                    "train_samples": "E:/data/train_samples.csv",
                    "is_geometry_varying": True,
                    "mode": "steady",
                    "config": "load_train_config() 返回的配置对象",
                # Completes the multi-line `collection` arguments or elements.
                },
                "apply_variable_selection": {
                    "config": "load_train_config() 返回并经 read_train_samples(...) 更新后的配置对象",
                    "samples_all": "read_train_samples(...) 返回的 samples_all",
                    # Completes the multi-line `collection` arguments or elements.
                    "input_variables": ["velocity", "heatPower"],
                    "output_variables": ["Pressure", "Velocity"],
                },
                "split_train_test": {
                    # Completes the multi-line `collection` arguments or elements.
                    "samples_all": samples_all,
                    "test_ratio": 0.34,
                },
                "run_train": {
                    # Completes the multi-line `collection` arguments or elements.
                    "config": "load_train_config() 返回并完成变量选择后的配置对象",
                    "samples_train": samples_train,
                    "experiment_dir": str(experiment_dir),
                    "model_type": "POD",
                    # Completes the multi-line `collection` arguments or elements.
                    "train_params": {
                        "device": "auto",
                        "model_scale": "small",
                        "norm_type": "min_max",
                        # Completes the multi-line `collection` arguments or elements.
                        "base_resolution": 50,
                        "graph_max_edges_per_node": 6,
                        "batch_size": 1,
                        "epochs": 2000,
                        # Completes the multi-line `collection` arguments or elements.
                        "learning_rate": 1e-4,
                        "valid_ratio": 0.1,
                        "loss_function": "mse",
                        "patience": 100,
                        # Completes the multi-line `collection` arguments or elements.
                        "graph_use_partition": False,
                        "graph_partition_count": 8,
                    },
                    "extra_info": {
                        # Completes the multi-line `collection` arguments or elements.
                        "job_id": "train-job-001",
                        "created_by": "backend-service",
                    },
                    "progress_callback": "后端注册的可选回调函数",
                    # Completes the multi-line `collection` arguments or elements.
                    "terminate_training_checker": "后端注册的可选停止检查函数，返回 True 时提前结束训练并继续自动评估",
                },
                "run_evaluate": {
                    "config": "load_evaluate_config() 返回的配置对象",
                    # Completes the multi-line `str` arguments or elements.
                    "checkpoint_path": str(experiment_dir / "train" / "best_model.pt"),
                    "run_dir": str(experiment_dir / "evaluate_request"),
                    "samples_evaluate": [
                        {
                            # Completes the multi-line `collection` arguments or elements.
                            "order": 101,
                            "params": {"velocity": 5.0, "heatPower": 100.0},
                            "source_file": "E:/data/case_101/result.vtu",
                            "processed_path": str(experiment_dir / "data" / "processed" / "sample_101.pt"),
                        # Completes the multi-line `collection` arguments or elements.
                        }
                    ],
                    "extra_info": {
                        "job_id": "evaluate-job-003",
                        # Completes the multi-line `collection` arguments or elements.
                        "created_by": "backend-service",
                    },
                },
                "load_evaluate_results": {
                    # Completes the multi-line `str` arguments or elements.
                    "model_path": str(experiment_dir / "train" / "best_model.pt"),
                },
                "load_model_for_prediction": {
                    "model_path": str(experiment_dir / "train" / "best_model.pt"),
                    # Completes the multi-line `collection` arguments or elements.
                    "config": "load_predict_config() 返回的配置对象",
                },
                "run_predict": {
                    "config": "load_predict_config() 返回的配置对象",
                    # Completes the multi-line `collection` arguments or elements.
                    "model_bundle": "load_model_for_prediction(...) 的返回值",
                    "run_dir": str(experiment_dir / "predict_request"),
                    "samples_predict": [
                        {
                            # Completes the multi-line `collection` arguments or elements.
                            "order": 201,
                            "params": {"velocity": 8.0, "heatPower": 160.0},
                            "source_file": "E:/data/case_201/result.vtu",
                        }
                    # Completes the multi-line `collection` arguments or elements.
                    ],
                },
            }
            api_check["predict_contract"] = predict_contract_check
    # Continues the surrounding statement or collection definition.
    except Exception as exc:  # pragma: no cover - 失败路径主要由安装环境决定
        errors.append(f"tfai 包基础调用失败: {type(exc).__name__}: {exc}")

    summary: Dict[str, Any] = {
        "passed": len(errors) == 0,
        # Completes the multi-line `collection` arguments or elements.
        "mode": mode,
        "dependency_policy": dependency_policy,
        "report_dir": str(report_dir),
        "report_path": str(report_path),
        # Completes the multi-line `collection` arguments or elements.
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
        # Completes the multi-line `collection` arguments or elements.
        },
        "dependencies": dependency_results,
        "tfai": tfai_info,
        "api_check": api_check,
        # Completes the multi-line `collection` arguments or elements.
        "config_check": config_check,
        "split_check": split_check,
        "api_input_examples": api_input_examples,
        "warnings": warnings,
        # Completes the multi-line `collection` arguments or elements.
        "errors": errors,
    }
    _write_json(report_path, summary)
    return summary


# Defines `main` for main logic.
def main(argv: List[str] | None = None, report_base_dir: Path | None = None) -> int:
    parser = argparse.ArgumentParser(description="tfai 安装后基础烟测脚本")
    parser.add_argument(
        "--output-dir",
        # Completes the multi-line `parser.add_argument` arguments or elements.
        type=Path,
        default=None,
        help="测试报告输出目录。默认写入 reports/installed_package_smoke/<时间戳>/",
    )
    # Calls `parser.add_argument` to perform the operation required at this stage.
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        # Completes the multi-line `parser.add_argument` arguments or elements.
        help="本地源码模式下使用的仓库根目录。仅用于仓库内回归验证，不建议安装后使用。",
    )
    parser.add_argument(
        "--warning-only-dependencies",
        # Completes the multi-line `parser.add_argument` arguments or elements.
        action="store_true",
        help="将必需依赖缺失从失败降级为警告，便于仓库内验证脚本逻辑。",
    )
    args = parser.parse_args(argv)

    # Stores the `summary` for use by the remaining steps in this flow.
    summary = run_smoke_check(
        output_dir=args.output_dir,
        project_root=args.project_root,
        strict_dependency_check=not args.warning_only_dependencies,
        # Completes the multi-line `run_smoke_check` arguments or elements.
        report_base_dir=report_base_dir,
    )
    print(json.dumps(_sanitize_for_json(summary), ensure_ascii=False, indent=2))
    print(f"\n报告已写入: {summary['report_path']}")
    # Returns `0 if summary["passed"] else 1` to the caller for the next stage.
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
