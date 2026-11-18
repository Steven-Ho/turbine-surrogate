from typing import Dict, Tuple
import csv
from typing import Any, List, Optional
# Imports `torch` for reuse in this module.
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
# Imports `DataLoader` from `tfai.dataset` for use in this module.
from tfai.dataset import DataLoader
from tfai.dataset.data_loader import collate_transolver_batch
from tfai.factories.processor_factory import ProcessorFactory
import os
# Imports `matplotlib.pyplot` for reuse in this module.
import matplotlib.pyplot as plt
from tfai.utils.torch_utils import resolve_torch_device


def _get_metric_decimals(metric_name: str) -> int:
    return 3 if str(metric_name).lower() in {"mape", "relative_l2"} else 6


# Defines `_round_metric_value` for  round metric value logic.
def _round_metric_value(metric_name: str, metric_value):
    if isinstance(metric_value, (float, np.floating)) and np.isnan(metric_value):
        return metric_value
    return float(np.round(float(metric_value), _get_metric_decimals(metric_name)))

# 计算 MAE、MSE、RMSE、NRMSE、MAPE、相对 L2 和 R2 指标。
def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    mape_target_threshold: float = 1e-3,
    mape_valid_mask: Optional[np.ndarray] = None,
# Continues the surrounding statement or collection definition.
) -> Dict:
    """计算 MAE、MSE、RMSE、NRMSE、MAPE、相对 L2 和 R2 指标。

    Args:
        predictions: 预测值数组
        targets: 目标值数组
        mape_target_threshold: 计算MAPE时保留的目标值绝对值阈值

    Returns:
        包含完整评估指标的字典
    """
    if len(targets) > 0:
        errors = predictions - targets

        # 计算MAE
        mae = np.mean(np.abs(errors))

        # 计算MSE / RMSE
        # Stores the `mse` for use by the remaining steps in this flow.
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)

        # 计算MAPE，忽略目标值绝对值过小的点，避免小分母放大误差。
        flat_predictions = np.asarray(predictions).reshape(-1)
        flat_targets = np.asarray(targets).reshape(-1)
        valid_mask = (
            np.asarray(mape_valid_mask, dtype=bool).reshape(-1)
            if mape_valid_mask is not None
            else np.abs(flat_targets) >= mape_target_threshold
        )
        if valid_mask.shape[0] != flat_targets.shape[0]:
            raise ValueError("MAPE 有效点掩码长度必须与目标值数量一致")
        if mape_valid_mask is None and mape_target_threshold <= 0:
            valid_mask &= np.abs(flat_targets) > 0
        valid_count = int(np.sum(valid_mask))
        # Checks `valid_count > 0` before entering this conditional branch.
        if valid_count > 0:
            valid_predictions = flat_predictions[valid_mask]
            valid_targets = flat_targets[valid_mask]
            mape = np.mean(np.abs((valid_predictions - valid_targets) / valid_targets)) * 100
        # Continues the surrounding statement or collection definition.
        else:
            mape = np.nan
        valid_ratio = valid_count / int(np.asarray(targets).size)

        # 计算R²
        if np.var(targets) > 0:
            r2 = 1 - mse / np.var(targets)
        # Continues the surrounding statement or collection definition.
        else:
            r2 = 1.0

        # 计算NRMSE，使用目标值范围进行归一化；常量目标场景下该指标无定义。
        target_range = float(np.max(targets) - np.min(targets))
        if target_range > 0:
            # Derives the `nrmse` from the current values for downstream processing.
            nrmse = rmse / target_range
        elif np.isclose(rmse, 0.0):
            nrmse = 0.0
        else:
            # Stores the `nrmse` for use by the remaining steps in this flow.
            nrmse = np.nan

        # 相对 L2 以百分比返回，与 MAPE 的返回口径保持一致。
        target_l2_norm = float(np.linalg.norm(targets.reshape(-1)))
        error_l2_norm = float(np.linalg.norm(errors.reshape(-1)))
        if target_l2_norm > 0:
            # Derives the `relative_l2` from the current values for downstream processing.
            relative_l2 = error_l2_norm / target_l2_norm * 100
        elif np.isclose(error_l2_norm, 0.0):
            relative_l2 = 0.0
        else:
            # Stores the `relative_l2` for use by the remaining steps in this flow.
            relative_l2 = np.nan
    else:
        mae = 0.0
        mse = 0.0
        # Stores the `rmse` for use by the remaining steps in this flow.
        rmse = 0.0
        nrmse = np.nan
        mape = np.nan
        relative_l2 = np.nan
        # Stores the `r2` for use by the remaining steps in this flow.
        r2 = 1.0
        valid_count = 0
        valid_ratio = 0.0

    metrics = {
        'mae': mae,
        'mse': mse,
        # Completes the multi-line `collection` arguments or elements.
        'rmse': rmse,
        'nrmse': nrmse,
        'mape': mape,
        'mape_valid_count': valid_count,
        'mape_valid_ratio': valid_ratio,
        'relative_l2': relative_l2,
        # Completes the multi-line `collection` arguments or elements.
        'r2': r2
    }
    rounded_metrics = {
        metric_name: _round_metric_value(metric_name, metric_value)
        # Completes the multi-line `collection` arguments or elements.
        for metric_name, metric_value in metrics.items()
    }
    rounded_metrics["mape_valid_count"] = valid_count
    rounded_metrics["mape_valid_ratio"] = float(np.round(valid_ratio, 6))
    return rounded_metrics

class Evaluator:
    MAX_SCATTER_POINTS = 10_000

    # 初始化评估器
    def __init__(self, model: nn.Module, config: Dict):
        """初始化评估器

        Args:
            model: 模型实例
            config: 评估配置参数
        """
        self.model = model
        self.config = config
        # 批量大小
        self.batch_size = config.get('evaluate', {}).get(
            # Completes the multi-line `config.get('evaluate', {}).get` arguments or elements.
            'batch_size',
            config.get('training', {}).get('batch_size', 1),
        )
        self.mape_target_threshold = 1e-3
        evaluate_cfg = config.get("evaluate", {}) or {}
        self.mape_relative_threshold = float(evaluate_cfg.get("mape_relative_threshold", 1e-3))
        norm_params = config.get("__eval_norm_params__") or {}
        mape_stats = norm_params.get("mape_target_scale") or {}
        self.mape_target_scales = dict(mape_stats.get("by_variable") or {})

        # 设置设备
        # Stores the compute device on the instance so other methods can reuse it.
        self.device = resolve_torch_device(self.config)
        self.model.to(self.device)

        # 获取任务参数
        task_config = config['task']
        self.task_type = task_config['type']
        # Stores the `mode` on the instance so other methods can reuse it.
        self.mode = task_config['mode']
        self.model_type = config['model']['type']
        self.grid_type = task_config.get('grid_type')
        self.processor = ProcessorFactory.get_processor(
            # Completes the multi-line `ProcessorFactory.get_processor` arguments or elements.
            self.task_type,
            self.model_type,
            self.config,
        )
        # Stores the `set_norm_params` for use by the remaining steps in this flow.
        set_norm_params = getattr(self.processor, "set_normalization_params", None)
        if callable(set_norm_params):
            set_norm_params(self.config.get("__eval_norm_params__"))
        attach_processor = getattr(self.model, "attach_processor", None)
        # Checks `callable(attach_processor)` before entering this conditional branch.
        if callable(attach_processor):
            attach_processor(self.processor)
        self.last_merged_payload = None
        self.last_scatter_data = None

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _infer_batch_sample_count(inputs, targets) -> int:
        if isinstance(targets, list):
            return len(targets)

        # Returns early when `isinstance(targets, torch.Tensor) and targets.ndim >= 1` makes the remaining work inapplicable.
        if isinstance(targets, torch.Tensor) and targets.ndim >= 1:
            return int(targets.shape[0])

        if isinstance(inputs, torch.Tensor) and inputs.ndim >= 1:
            return int(inputs.shape[0])

        # Returns `0` to the caller for the next stage.
        return 0

    @staticmethod
    def _build_batch_sample_meta(
        dataset: Dataset,
        # Continues the surrounding statement or collection definition.
        sample_offset: int,
        sample_count: int,
    ) -> Tuple[List[Dict], List[int]]:
        if sample_count <= 0 or not hasattr(dataset, "get_sample_meta"):
            # Returns `[], []` to the caller for the next stage.
            return [], []

        sample_metas: List[Dict] = []
        sample_ids: List[int] = []
        max_index = len(dataset)
        # Stores the `end_index` for use by the remaining steps in this flow.
        end_index = min(sample_offset + sample_count, max_index)
        for dataset_index in range(sample_offset, end_index):
            meta = dataset.get_sample_meta(dataset_index) or {}
            sample_metas.append(meta)
            # Adds the current value to `sample_ids` for later aggregation.
            sample_ids.append(int(meta.get("sample_id", dataset_index)))
        return sample_metas, sample_ids

    def _calculate_metrics_by_variable(self, predictions_by_var: Dict[str, np.ndarray], targets_by_var: Dict[str, np.ndarray]) -> Dict:
        """按物理量分别计算指标，并给出overall指标。"""
        by_variable = {}
        # Initializes the `all_pred_parts` as a collection for values from this processing stage.
        all_pred_parts = []
        all_tgt_parts = []
        all_mape_masks = []

        for var_name, pred_arr in predictions_by_var.items():
            if var_name not in targets_by_var:
                # Handles the data or control-flow case represented by this statement.
                continue
            tgt_arr = targets_by_var[var_name]
            pred_flat = np.asarray(pred_arr).reshape(-1)
            tgt_flat = np.asarray(tgt_arr).reshape(-1)
            # Validates `pred_flat.shape[0] != tgt_flat.shape[0]` and raises `ValueError` when the input cannot be processed safely.
            if pred_flat.shape[0] != tgt_flat.shape[0]:
                raise ValueError(f"物理量{var_name}评估时预测与目标长度不一致: pred={pred_arr.shape}, target={tgt_arr.shape}")

            by_variable[var_name] = calculate_metrics(
                pred_flat,
                # Completes the multi-line `calculate_metrics` arguments or elements.
                tgt_flat,
                mape_target_threshold=self._mape_threshold_for_variable(var_name),
            )
            all_mape_masks.append(self._mape_valid_mask(var_name, tgt_flat))
            all_pred_parts.append(pred_flat)
            # Adds the current value to `all_tgt_parts` for later aggregation.
            all_tgt_parts.append(tgt_flat)

        if not all_pred_parts:
            return {
                'overall': calculate_metrics(
                    # Completes the multi-line `np.array` arguments or elements.
                    np.array([]),
                    np.array([]),
                    mape_target_threshold=self._mape_threshold_for_variable(None),
                ),
                # Completes the multi-line `collection` arguments or elements.
                'by_variable': {},
            }

        overall_pred = np.concatenate(all_pred_parts, axis=0)
        overall_tgt = np.concatenate(all_tgt_parts, axis=0)
        # Returns `` to the caller for the next stage.
        return {
            'overall': calculate_metrics(
                overall_pred,
                overall_tgt,
                mape_target_threshold=self.mape_target_threshold,
                mape_valid_mask=np.concatenate(all_mape_masks),
            ),
            'by_variable': by_variable
        }

    def _mape_threshold_for_variable(self, var_name: Optional[str]) -> float:
        """Use training p95 scale when present; old checkpoints retain abs threshold."""
        if var_name is None and len(self.mape_target_scales) == 1:
            scale = next(iter(self.mape_target_scales.values()))
        else:
            scale = self.mape_target_scales.get(str(var_name))
        if scale is None:
            return self.mape_target_threshold
        return max(0.0, self.mape_relative_threshold * float(scale))

    def _mape_valid_mask(self, var_name: Optional[str], targets: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets).reshape(-1)
        threshold = self._mape_threshold_for_variable(var_name)
        if var_name is not None and str(var_name) in self.mape_target_scales and threshold <= 0:
            return np.abs(targets) > 0
        return np.abs(targets) >= threshold

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _format_metric_value(metric_name: str, metric_value) -> str:
        if isinstance(metric_value, (float, np.floating)) and np.isnan(metric_value):
            return "N/A"
        if str(metric_name).lower() == "mape_valid_count":
            return str(int(metric_value))
        # Returns `f"{float(metric_value):.{_get_metric_decimals(metric_name)}f}"` to the caller for the next stage.
        return f"{float(metric_value):.{_get_metric_decimals(metric_name)}f}"

    @classmethod
    def _format_metric_display(cls, metric_name: str, metric_value) -> str:
        """按指标单位格式化用于展示的数值。"""
        value_text = cls._format_metric_value(metric_name, metric_value)
        # Returns early when `` makes the remaining work inapplicable.
        if str(metric_name).lower() in {"mape", "relative_l2"} and value_text != "N/A":
            return f"{value_text}%"
        return value_text

    def _build_scatter_path(self, save_path: str, suffix: str) -> str:
        """根据基础路径构建按物理量拆分的散点图文件名。"""
        # Returns early when `not save_path` makes the remaining work inapplicable.
        if not save_path:
            return None
        root, ext = os.path.splitext(save_path)
        safe_suffix = ''.join(ch if ch.isalnum() or ch in ['_', '-'] else '_' for ch in suffix)
        # Returns `f"{root}_{safe_suffix}{ext or '.png'}"` to the caller for the next stage.
        return f"{root}_{safe_suffix}{ext or '.png'}"

    def _get_ordered_scatter_variables(self, scatter_by_var: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
        """按 effective_variables 优先级返回散点变量顺序。"""
        effective_vars = list(self.config.get('task', {}).get('effective_variables') or [])
        ordered_vars = [name for name in effective_vars if name in scatter_by_var]
        # Adds the current value to `ordered_vars` for later aggregation.
        ordered_vars.extend([name for name in scatter_by_var.keys() if name not in ordered_vars])
        return ordered_vars

    @staticmethod
    def _sample_scatter_arrays(
        # Continues the surrounding statement or collection definition.
        predictions: np.ndarray,
        targets: np.ndarray,
        max_points: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对超大规模散点进行稳定采样，避免返回结果过大。"""
        # Stores the `flat_predictions` for use by the remaining steps in this flow.
        flat_predictions = np.asarray(predictions).reshape(-1)
        flat_targets = np.asarray(targets).reshape(-1)

        if flat_predictions.shape[0] != flat_targets.shape[0]:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "构建散点返回数据时预测与目标长度不一致: "
                f"pred={flat_predictions.shape}, target={flat_targets.shape}"
            )

        if flat_predictions.size <= max_points:
            # Returns `flat_predictions, flat_targets` to the caller for the next stage.
            return flat_predictions, flat_targets

        rng = np.random.default_rng(42)
        sample_indices = np.sort(
            rng.choice(flat_predictions.size, size=max_points, replace=False)
        # Completes the multi-line `np.sort` arguments or elements.
        )
        return flat_predictions[sample_indices], flat_targets[sample_indices]

    def _build_scatter_series(
        self,
        # Continues the surrounding statement or collection definition.
        predictions: np.ndarray,
        targets: np.ndarray,
        max_points: int,
    ) -> Dict[str, List[float]]:
        """将散点数据转换为适合后端返回的内存结构。"""
        # Stores the `(sampled_predictions, sampled_targets)` for use by the remaining steps in this flow.
        sampled_predictions, sampled_targets = self._sample_scatter_arrays(
            predictions=predictions,
            targets=targets,
            max_points=max_points,
        # Completes the multi-line `self._sample_scatter_arrays` arguments or elements.
        )
        return {
            "predictions": sampled_predictions.tolist(),
            "targets": sampled_targets.tolist(),
        # Completes the multi-line `collection` arguments or elements.
        }

    def build_scatter_data(
        self,
        merged_payload: Dict[str, Any],
        # Continues the surrounding statement or collection definition.
        max_points: int = None,
    ) -> Dict[str, Any]:
        """根据评估聚合结果构建采样后的散点返回数据。"""
        effective_max_points = int(max_points or self.MAX_SCATTER_POINTS)
        scatter_by_var = merged_payload.get("scatter_by_var", {}) or {}
        # Reads `overall_predictions` from the mapping into the `overall_predictions` with its declared fallback.
        overall_predictions = merged_payload.get("overall_predictions", np.array([]))
        overall_targets = merged_payload.get("overall_targets", np.array([]))

        scatter_data = {
            "overall": self._build_scatter_series(
                # Completes the multi-line `self._build_scatter_series` arguments or elements.
                predictions=overall_predictions,
                targets=overall_targets,
                max_points=effective_max_points,
            ),
            # Completes the multi-line `collection` arguments or elements.
            "by_variable": {},
        }

        for var_name in self._get_ordered_scatter_variables(scatter_by_var):
            payload = scatter_by_var.get(var_name, {})
            # Stores the `scatter_data['by_variable'][var_name]` for use by the remaining steps in this flow.
            scatter_data["by_variable"][var_name] = self._build_scatter_series(
                predictions=payload.get("predictions", np.array([])),
                targets=payload.get("targets", np.array([])),
                max_points=effective_max_points,
            # Completes the multi-line `self._build_scatter_series` arguments or elements.
            )

        return scatter_data

    def _export_scatter_csv(
        self,
        # Continues the surrounding statement or collection definition.
        scatter_by_var: Dict,
        overall_predictions: np.ndarray,
        overall_targets: np.ndarray,
        save_path: str = None,
    # Continues the surrounding statement or collection definition.
    ) -> str:
        """导出前端绘图使用的全量散点数据。"""
        if not save_path:
            return None

        save_dir = os.path.dirname(save_path)
        # Checks `save_dir` before entering this conditional branch.
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        effective_vars = list(self.config.get('task', {}).get('effective_variables') or [])
        ordered_vars = [name for name in effective_vars if name in scatter_by_var]
        # Adds the current value to `ordered_vars` for later aggregation.
        ordered_vars.extend([name for name in scatter_by_var.keys() if name not in ordered_vars])

        with open(save_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['variable', 'target', 'prediction'])

            # Checks `ordered_vars` before entering this conditional branch.
            if ordered_vars:
                for var_name in ordered_vars:
                    payload = scatter_by_var.get(var_name, {})
                    predictions = np.asarray(payload.get('predictions', np.array([]))).reshape(-1)
                    # Stores the target tensors for use by the remaining steps in this flow.
                    targets = np.asarray(payload.get('targets', np.array([]))).reshape(-1)
                    if predictions.shape[0] != targets.shape[0]:
                        raise ValueError(
                            f"物理量{var_name}导出散点数据时预测与目标长度不一致: "
                            # Completes the multi-line `ValueError` arguments or elements.
                            f"pred={predictions.shape}, target={targets.shape}"
                        )

                    for target_value, prediction_value in zip(targets.tolist(), predictions.tolist()):
                        writer.writerow([var_name, target_value, prediction_value])
            # Continues the surrounding statement or collection definition.
            else:
                flat_predictions = np.asarray(overall_predictions).reshape(-1)
                flat_targets = np.asarray(overall_targets).reshape(-1)
                if flat_predictions.shape[0] != flat_targets.shape[0]:
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"导出overall散点数据时预测与目标长度不一致: "
                        f"pred={flat_predictions.shape}, target={flat_targets.shape}"
                    )

                # Visits each item in `zip(flat_targets.tolist(), flat_predictions.tolist())` and applies the current processing step.
                for target_value, prediction_value in zip(flat_targets.tolist(), flat_predictions.tolist()):
                    writer.writerow(['overall', target_value, prediction_value])

        return save_path

    def build_evaluation_log_text(
        # Continues the surrounding statement or collection definition.
        self,
        metrics: Dict,
        scatter_save_path: str = "",
        scatter_csv_path: str = "",
    # Continues the surrounding statement or collection definition.
    ) -> str:
        """构建评估摘要文本。"""
        lines = ["评估结果:"]

        if isinstance(metrics, dict) and 'overall' in metrics:
            lines.append("Overall:")
            # Visits each item in `metrics.get('overall', {}).items()` and applies the current processing step.
            for metric_name, metric_value in metrics.get('overall', {}).items():
                lines.append(
                    f"  {metric_name}: {self._format_metric_display(metric_name, metric_value)}"
                )

            # Reads `by_variable` from the mapping into the `by_var` with its declared fallback.
            by_var = metrics.get('by_variable', {})
            if by_var:
                lines.append("By Variable:")
                for var_name, var_metrics in by_var.items():
                    # Stores the `metric_str` for use by the remaining steps in this flow.
                    metric_str = ", ".join(
                        [
                            f"{k}={self._format_metric_display(k, v)}"
                            for k, v in var_metrics.items()
                        # Completes the multi-line `', '.join` arguments or elements.
                        ]
                    )
                    lines.append(f"  {var_name}: {metric_str}")
        elif isinstance(metrics, dict):
            # Visits each item in `metrics.items()` and applies the current processing step.
            for metric_name, metric_value in metrics.items():
                lines.append(
                    f"{metric_name}: {self._format_metric_display(metric_name, metric_value)}"
                )
        # Continues the surrounding statement or collection definition.
        else:
            lines.append(str(metrics))

        if scatter_save_path:
            lines.append(f"预测值与目标值的散点图已保存至: {scatter_save_path}")
        # Checks `scatter_csv_path` before entering this conditional branch.
        if scatter_csv_path:
            lines.append(f"前端绘图散点数据已保存至: {scatter_csv_path}")

        return "\n".join(lines) + "\n"

    def write_evaluation_log(
        # Continues the surrounding statement or collection definition.
        self,
        evaluate_dir: str,
        metrics: Dict,
        scatter_save_path: str = "",
        # Continues the surrounding statement or collection definition.
        scatter_csv_path: str = "",
    ) -> str:
        """将评估摘要写入 evaluate.log。"""
        os.makedirs(evaluate_dir, exist_ok=True)
        evaluate_log_path = os.path.join(evaluate_dir, 'evaluate.log')
        # Stores the `log_text` for use by the remaining steps in this flow.
        log_text = self.build_evaluation_log_text(
            metrics=metrics,
            scatter_save_path=scatter_save_path,
            scatter_csv_path=scatter_csv_path,
        # Completes the multi-line `self.build_evaluation_log_text` arguments or elements.
        )
        with open(evaluate_log_path, 'w', encoding='utf-8') as f:
            f.write(log_text)
        return evaluate_log_path

    # 将评估摘要打印到训练日志。
    def log_evaluation_summary(
        self,
        logger,
        metrics: Dict,
        # Continues the surrounding statement or collection definition.
        scatter_save_path: str = "",
        scatter_csv_path: str = "",
        evaluate_log_path: str = "",
    ) -> None:
        """将评估摘要打印到训练日志。"""
        # Stores the `summary_text` for use by the remaining steps in this flow.
        summary_text = self.build_evaluation_log_text(
            metrics=metrics,
            scatter_save_path=scatter_save_path,
            scatter_csv_path=scatter_csv_path,
        # Completes the multi-line `self.build_evaluation_log_text` arguments or elements.
        )
        for line in summary_text.strip().splitlines():
            logger.info(line)
        if evaluate_log_path:
            # Records the current execution state to make progress and failures traceable.
            logger.info(f"评估日志已保存至: {evaluate_log_path}")
        
    
    def evaluate(
        self,
        test_dataset: Dataset,
        # Continues the surrounding statement or collection definition.
        plot_scatter: bool = True,
        scatter_save_path: str = None,
        scatter_csv_path: str = None,
    ) -> tuple:
        """执行评估流程

        Args:
            test_dataset: 测试数据集
            plot_scatter: 是否绘制散点图，默认为True
            scatter_save_path: 散点图保存路径，默认为None
            scatter_csv_path: 前端绘图数据CSV保存路径，默认为None

        Returns:
            tuple: (评估指标字典, 预测数组, 目标数组)
        """

        # 创建数据加载器
        # Stores the `test_loader` for use by the remaining steps in this flow.
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 设置模型为评估模式
        self.model.eval()

        batch_payloads = []
        sample_offset = 0
        # Stores the `is_graph_dataset` for use by the remaining steps in this flow.
        is_graph_dataset = getattr(test_dataset, "data_kind", "") == "graph"

        with torch.no_grad():
            for inputs, targets in test_loader:
                batch_sample_metas: List[Dict] = []
                # Initializes the `batch_sample_ids` as a collection for values from this processing stage.
                batch_sample_ids: List[int] = []
                if not is_graph_dataset:
                    batch_sample_count = self._infer_batch_sample_count(inputs, targets)
                    batch_sample_metas, batch_sample_ids = self._build_batch_sample_meta(
                        # Completes the multi-line `self._build_batch_sample_meta` arguments or elements.
                        dataset=test_dataset,
                        sample_offset=sample_offset,
                        sample_count=batch_sample_count,
                    )
                    # Stores the `sample_offset` for use by the remaining steps in this flow.
                    sample_offset += len(batch_sample_ids)

                inputs = inputs.to(self.device)
                
                # 处理targets设备转移
                if isinstance(targets, list):
                    # 变长targets（来自GraphProcessor）
                    targets = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in targets]
                    # 如果是列表，需要合并成单个tensor
                    # Stores the target tensors for use by the remaining steps in this flow.
                    targets = torch.cat(targets, dim=0) if all(isinstance(t, torch.Tensor) for t in targets) else targets
                elif isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)

                # 前向传播
                outputs = self.model(inputs)

                # Stores the `(pred_payload, target_payload)` for use by the remaining steps in this flow.
                pred_payload, target_payload = self.processor.prepare_eval_payload(
                    inputs=inputs,
                    outputs=outputs,
                    targets=targets,
                # Completes the multi-line `self.processor.prepare_eval_payload` arguments or elements.
                )
                payload = self.processor.collect_eval_batch(
                    inputs=inputs,
                    outputs=pred_payload,
                    # Completes the multi-line `self.processor.collect_eval_batch` arguments or elements.
                    targets=target_payload,
                )
                if batch_sample_ids and "sample_ids" not in payload:
                    payload["sample_ids"] = batch_sample_ids
                # Checks `batch_sample_metas and "sample_metas" not in payload` before entering this conditional branch.
                if batch_sample_metas and "sample_metas" not in payload:
                    payload["sample_metas"] = batch_sample_metas
                batch_payloads.append(payload)

        merged = self.processor.merge_eval_batches(batch_payloads, self.config)
        # Stores the `last_merged_payload` on the instance so other methods can reuse it.
        self.last_merged_payload = merged
        self.last_scatter_data = self.build_scatter_data(merged)
        scatter_by_var = merged.get('scatter_by_var', {})

        if scatter_by_var:
            # Builds the `predictions_by_var` using the comprehension's filter and mapping rules.
            predictions_by_var = {
                var_name: np.asarray(payload['predictions']).reshape(-1)
                for var_name, payload in scatter_by_var.items()
            }
            # Builds the `targets_by_var` using the comprehension's filter and mapping rules.
            targets_by_var = {
                var_name: np.asarray(payload['targets']).reshape(-1)
                for var_name, payload in scatter_by_var.items()
            }
            # Stores the `metrics` for use by the remaining steps in this flow.
            metrics = self._calculate_metrics_by_variable(predictions_by_var, targets_by_var)
        else:
            overall_predictions = np.asarray(merged.get('overall_predictions', np.array([]))).reshape(-1)
            overall_targets = np.asarray(merged.get('overall_targets', np.array([]))).reshape(-1)
            # Initializes the `metrics` as a collection for values from this processing stage.
            metrics = {
                'overall': calculate_metrics(
                    overall_predictions,
                    overall_targets,
                    # Completes the multi-line `calculate_metrics` arguments or elements.
                    mape_target_threshold=self.mape_target_threshold,
                ),
                'by_variable': {},
            }

        # Stores the `plot_predictions` for use by the remaining steps in this flow.
        plot_predictions = np.asarray(merged.get('overall_predictions', np.array([]))).reshape(-1)
        plot_targets = np.asarray(merged.get('overall_targets', np.array([]))).reshape(-1)

        self._export_scatter_csv(
            scatter_by_var=scatter_by_var,
            # Completes the multi-line `self._export_scatter_csv` arguments or elements.
            overall_predictions=plot_predictions,
            overall_targets=plot_targets,
            save_path=scatter_csv_path,
        )

        # 绘制散点图
        # Checks `plot_scatter` before entering this conditional branch.
        if plot_scatter:
            # overall散点图
            self.plot_scatter(
                plot_predictions,
                plot_targets,
                # Completes the multi-line `self._build_scatter_path` arguments or elements.
                self._build_scatter_path(scatter_save_path, 'overall') if scatter_save_path else None,
                metrics.get('overall', metrics)
            )

            # 按物理量散点图
            for var_name, var_payload in scatter_by_var.items():
                # Calls `self.plot_scatter` to perform the operation required at this stage.
                self.plot_scatter(
                    np.asarray(var_payload['predictions']).reshape(-1),
                    np.asarray(var_payload['targets']).reshape(-1),
                    self._build_scatter_path(scatter_save_path, var_name) if scatter_save_path else None,
                    # Completes the multi-line `metrics.get('by_variable', {}).get` arguments or elements.
                    metrics.get('by_variable', {}).get(var_name, None)
                )

        return (
            metrics,
            # Completes the multi-line `merged.get` arguments or elements.
            merged.get('denormalized_predictions', merged.get('raw_predictions')),
            merged.get('denormalized_targets', merged.get('raw_targets')),
        )

    def evaluate_autoregressive_full_rollout(
        # Continues the surrounding statement or collection definition.
        self,
        test_dataset: Dataset,
        plot_scatter: bool = True,
        scatter_save_path: str = None,
        # Continues the surrounding statement or collection definition.
        scatter_csv_path: str = None,
    ) -> tuple:
        """逐条执行全长度自由滚动，避免不同序列长度的 padding 参与指标。"""
        rollout_fn = getattr(self.processor, "predict_autoregressive_rollout", None)
        if not callable(rollout_fn):
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError("当前 processor 不支持全长度 autoregressive 评估")

        self.model.eval()
        batch_payloads = []
        with torch.no_grad():
            # Visits each item in `range(len(test_dataset))` and applies the current processing step.
            for dataset_index in range(len(test_dataset)):
                inputs, targets = test_dataset[dataset_index]
                meta = test_dataset.get_sample_meta(dataset_index) if hasattr(test_dataset, "get_sample_meta") else {}
                rollout_steps = int((meta or {}).get("rollout_steps", 0))
                # Validates `rollout_steps <= 0` and raises `ValueError` when the input cannot be processed safely.
                if rollout_steps <= 0:
                    raise ValueError("全长度 autoregressive 评估样本缺少有效 rollout_steps")

                if getattr(test_dataset, "data_kind", "tensor") == "set":
                    inputs, targets = collate_transolver_batch([(inputs, targets)])
                    # Stores the model inputs for use by the remaining steps in this flow.
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                elif isinstance(inputs, dict):
                    inputs = {
                        # Completes the multi-line `isinstance` arguments or elements.
                        key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                        for key, value in inputs.items()
                    }
                    inputs = {
                        # Completes the multi-line `isinstance` arguments or elements.
                        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
                        for key, value in inputs.items()
                    }
                else:
                    # Stores the model inputs for use by the remaining steps in this flow.
                    inputs = inputs.to(self.device)
                    if getattr(test_dataset, "data_kind", "tensor") != "graph":
                        inputs = inputs.unsqueeze(0)
                if getattr(test_dataset, "data_kind", "tensor") != "set":
                    # Stores the target tensors for use by the remaining steps in this flow.
                    targets = targets.to(self.device)
                if getattr(test_dataset, "data_kind", "tensor") not in {"graph", "set"}:
                    targets = targets.unsqueeze(0)

                outputs = rollout_fn(
                    # Completes the multi-line `rollout_fn` arguments or elements.
                    model=self.model,
                    inputs=inputs,
                    rollout_steps=rollout_steps,
                )
                # Validates `` and raises `ValueError` when the input cannot be processed safely.
                if not isinstance(outputs, torch.Tensor) or outputs.shape != targets.shape:
                    raise ValueError(
                        "全长度 autoregressive 预测与目标形状不一致："
                        f"pred={getattr(outputs, 'shape', None)}, target={getattr(targets, 'shape', None)}"
                    # Completes the multi-line `ValueError` arguments or elements.
                    )
                payload = self.processor.collect_eval_batch(
                    inputs=inputs,
                    outputs=outputs,
                    # Completes the multi-line `self.processor.collect_eval_batch` arguments or elements.
                    targets=targets,
                )
                sample_id = int((meta or {}).get("sample_id", dataset_index))
                payload["sample_ids"] = [sample_id]
                # Initializes the `payload['sample_metas']` as a collection for values from this processing stage.
                payload["sample_metas"] = [meta or {}]
                batch_payloads.append(payload)

        merged = self.processor.merge_eval_batches(batch_payloads, self.config)
        self.last_merged_payload = merged
        # Stores the `last_scatter_data` on the instance so other methods can reuse it.
        self.last_scatter_data = self.build_scatter_data(merged)
        scatter_by_var = merged.get('scatter_by_var', {})
        if scatter_by_var:
            metrics = self._calculate_metrics_by_variable(
                # Completes the multi-line `self._calculate_metrics_by_variable` arguments or elements.
                {
                    name: np.asarray(payload['predictions']).reshape(-1)
                    for name, payload in scatter_by_var.items()
                },
                # Completes the multi-line `self._calculate_metrics_by_variable` arguments or elements.
                {
                    name: np.asarray(payload['targets']).reshape(-1)
                    for name, payload in scatter_by_var.items()
                },
            # Completes the multi-line `self._calculate_metrics_by_variable` arguments or elements.
            )
        else:
            metrics = {
                'overall': calculate_metrics(
                    # Completes the multi-line `np.asarray(merged.get('overall_predictions', np.array([]))).reshape` arguments or elements.
                    np.asarray(merged.get('overall_predictions', np.array([]))).reshape(-1),
                    np.asarray(merged.get('overall_targets', np.array([]))).reshape(-1),
                    mape_target_threshold=self._mape_threshold_for_variable(None),
                ),
                # Completes the multi-line `collection` arguments or elements.
                'by_variable': {},
            }
        metrics['evaluation_scope'] = 'full_autoregressive_rollout'

        plot_predictions = np.asarray(merged.get('overall_predictions', np.array([]))).reshape(-1)
        # Stores the `plot_targets` for use by the remaining steps in this flow.
        plot_targets = np.asarray(merged.get('overall_targets', np.array([]))).reshape(-1)
        self._export_scatter_csv(
            scatter_by_var=scatter_by_var,
            overall_predictions=plot_predictions,
            # Completes the multi-line `self._export_scatter_csv` arguments or elements.
            overall_targets=plot_targets,
            save_path=scatter_csv_path,
        )
        if plot_scatter:
            # Calls `self.plot_scatter` to perform the operation required at this stage.
            self.plot_scatter(
                plot_predictions,
                plot_targets,
                self._build_scatter_path(scatter_save_path, 'overall') if scatter_save_path else None,
                # Completes the multi-line `metrics.get` arguments or elements.
                metrics.get('overall', metrics),
            )
            for var_name in self._get_ordered_scatter_variables(scatter_by_var):
                var_payload = scatter_by_var[var_name]
                # Calls `self.plot_scatter` to perform the operation required at this stage.
                self.plot_scatter(
                    np.asarray(var_payload['predictions']).reshape(-1),
                    np.asarray(var_payload['targets']).reshape(-1),
                    self._build_scatter_path(scatter_save_path, var_name) if scatter_save_path else None,
                    # Completes the multi-line `metrics.get('by_variable', {}).get` arguments or elements.
                    metrics.get('by_variable', {}).get(var_name, None),
                )
        return (
            metrics,
            # Completes the multi-line `merged.get` arguments or elements.
            merged.get('denormalized_predictions', merged.get('raw_predictions')),
            merged.get('denormalized_targets', merged.get('raw_targets')),
        )
    
    def plot_scatter(self, predictions: np.ndarray, targets: np.ndarray, save_path: str = None, metrics: Dict = None) -> None:
        """绘制预测值与目标值的散点图

        Args:
            predictions: 预测值数组
            targets: 目标值数组
            save_path: 保存路径，默认为None（不保存）
            metrics: 评估指标字典，默认为None
        """

        # 转换并展平
        # Stores the `predictions` for use by the remaining steps in this flow.
        predictions = np.asarray(predictions).reshape(-1)
        targets = np.asarray(targets).reshape(-1)

        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(
                # Completes the multi-line `ValueError` arguments or elements.
                "绘制散点图时预测与目标长度不一致: "
                f"pred={predictions.shape}, target={targets.shape}"
            )

        if predictions.size == 0:
            # Ends this function and returns control to its caller.
            return

        # 误差指标仍基于全量数据；仅在绘图时对超大规模点集进行随机采样
        max_plot_points = 5_000_000
        if predictions.size > max_plot_points:
            rng = np.random.default_rng(42)
            # Stores the `sample_indices` for use by the remaining steps in this flow.
            sample_indices = rng.choice(predictions.size, size=max_plot_points, replace=False)
            predictions = predictions[sample_indices]
            targets = targets[sample_indices]

        # 创建散点图
        plt.figure(figsize=(12, 10))
        # Calls `plt.scatter` to perform the operation required at this stage.
        plt.scatter(targets, predictions, alpha=0.5, color='blue')

        # 添加对角线（理想情况：预测值等于目标值）
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2)

        # 设置图表属性，增大字号
        # Calls `plt.xlabel` to perform the operation required at this stage.
        plt.xlabel('Target', fontsize=26)
        plt.ylabel('Prediction', fontsize=26)
        plt.title('Scatter Plot of Predictions vs Targets', fontsize=26, fontweight='bold')
        plt.xticks(fontsize=24)
        # Calls `plt.yticks` to perform the operation required at this stage.
        plt.yticks(fontsize=24)
        plt.tick_params(axis='both', direction='in')
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        # Visits each item in `ax.spines.values()` and applies the current processing step.
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # 在图中添加误差指标
        if metrics:
            # 构建指标文本
            metric_text = "Evaluation Metrics:\n"
            # Visits each item in `metrics.items()` and applies the current processing step.
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (float, np.floating)) and np.isnan(metric_value):
                    metric_text += f"{metric_name.upper()}: N/A\n"
                    continue
                # Checks `metric_name.lower() in {'mape', 'relative_l2'}` before entering this conditional branch.
                if metric_name.lower() in {'mape', 'relative_l2'}:
                    # 百分比指标添加%符号。
                    metric_text += (
                        f"{metric_name.upper()}: "
                        f"{self._format_metric_value(metric_name, metric_value)}%\n"
                    # Continues the surrounding statement or collection definition.
                    )
                elif metric_name.lower() == 'r2':
                    # R2的2显示为上标
                    metric_text += f"R²: {self._format_metric_value(metric_name, metric_value)}\n"
                else:
                    # Stores the `metric_text` for use by the remaining steps in this flow.
                    metric_text += (
                        f"{metric_name.upper()}: "
                        f"{self._format_metric_value(metric_name, metric_value)}\n"
                    )
            
            # 使用坐标轴相对坐标，确保指标文本始终位于图框内
            # Calls `ax.text` to perform the operation required at this stage.
            ax.text(
                0.03,
                0.97,
                metric_text,
                # Completes the multi-line `ax.text` arguments or elements.
                transform=ax.transAxes,
                fontsize=24,
                linespacing=1.5,
                ha='left',
                # Completes the multi-line `ax.text` arguments or elements.
                va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.4),
            )

        # 保存或显示图表
        if save_path:
            # 确保保存目录存在
            # Stores the `save_dir` for use by the remaining steps in this flow.
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # print(f"Scatter plot saved to: {save_path}")
        # Continues the surrounding statement or collection definition.
        else:
            plt.show()

        plt.close()
