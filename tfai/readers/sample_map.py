import csv
import logging
from pathlib import Path
# Imports `Dict, List, Optional` from `typing` for use in this module.
from typing import Dict, List, Optional

from ..utils.logger import Logger


def _get_logger() -> logging.Logger:
    return Logger.get_first_configured_logger(
        # Completes the multi-line `collection` arguments or elements.
        ["train", "predict", "evaluate"],
        fallback_name=__name__,
    )


class SampleMap:
    """样本映射类，用于维护样本编号与样本信息。"""

    # 初始化样本映射。
    def __init__(self, sample_table_path: Optional[str] = None):
        """初始化样本映射。"""
        self.sample_map: Dict[int, Dict] = {}
        if sample_table_path:
            self.load_sample_table(sample_table_path)

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _detect_delimiter(sample_table_path: Path) -> str:
        """自动检测分隔符，失败时根据扩展名回退。"""
        default_delimiter = "," if sample_table_path.suffix.lower() == ".csv" else "\t"

        try:
            # Scopes resource use so the context manager can release it on exit.
            with open(sample_table_path, "r", encoding="utf-8-sig", newline="") as f:
                sample_text = f.read(4096)
            if not sample_text.strip():
                return default_delimiter

            # Stores the `dialect` for use by the remaining steps in this flow.
            dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t;|")
            return dialect.delimiter
        except Exception:
            return default_delimiter

    # 从 CSV/TXT 文件加载样本表。
    def load_sample_table(self, sample_table_path: str) -> None:
        """从 CSV/TXT 文件加载样本表。"""
        sample_table_path = Path(sample_table_path)
        if not sample_table_path.exists():
            raise FileNotFoundError(f"样本表文件不存在: {sample_table_path}")

        # Stores the `delimiter` for use by the remaining steps in this flow.
        delimiter = self._detect_delimiter(sample_table_path)

        with open(sample_table_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Keeps the operation and its recovery path within the same exception boundary.
                try:
                    order_value = None
                    for key in row:
                        if key.lower() == "order":
                            # Selects the required item from `row` as the `order_value`.
                            order_value = row[key]
                            break

                    if not order_value:
                        continue

                    # Converts the `order` to `int` so later checks use a consistent type.
                    order = int(order_value)
                    sample_info = {}

                    for key, value in row.items():
                        if key.lower() == "order":
                            # Handles the data or control-flow case represented by this statement.
                            continue
                        if key.lower() in {"file", "file_path", "source_file", "resolved_path"}:
                            sample_info["file_path"] = value
                            continue

                        # Keeps the operation and its recovery path within the same exception boundary.
                        try:
                            if "." in value:
                                sample_info[key] = float(value)
                            else:
                                # Converts the `sample_info[key]` to `int` so later checks use a consistent type.
                                sample_info[key] = int(value)
                        except ValueError:
                            sample_info[key] = value

                    self.sample_map[order] = sample_info
                # Continues the surrounding statement or collection definition.
                except Exception as exc:
                    _get_logger().warning("解析样本表行失败: row=%s reason=%s", row, str(exc))

    def get_sample_info(self, sample_order: int) -> Optional[Dict]:
        """根据样本编号获取样本信息。"""
        return self.sample_map.get(sample_order)

    # 获取所有样本编号。
    def get_all_samples(self) -> List[int]:
        """获取所有样本编号。"""
        return list(self.sample_map.keys())

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.sample_map)

    # 根据样本编号获取文件路径。
    def get_file_path(self, sample_order: int) -> Optional[str]:
        """根据样本编号获取文件路径。"""
        sample_info = self.get_sample_info(sample_order)
        if sample_info:
            return sample_info.get("file_path")
        # Returns `None` to the caller for the next stage.
        return None

    def get_all_file_paths(self) -> List[str]:
        """获取所有样本文件路径。"""
        file_paths = []
        for sample_id in self.get_all_samples():
            # Stores the file path for use by the remaining steps in this flow.
            file_path = self.get_file_path(sample_id)
            if file_path:
                file_paths.append(file_path)
        return file_paths

    # 获取样本数量。
    def get_sample_count(self) -> int:
        """获取样本数量。"""
        return len(self)

    def has_sample(self, sample_order: int) -> bool:
        """检查是否存在指定样本编号。"""
        return sample_order in self.sample_map
