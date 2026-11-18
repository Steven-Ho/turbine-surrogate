"""样本契约相关的通用工具。"""

import json
import os
from typing import Dict


DEFAULT_SOURCE_DIRNAME = "default_source"
DEFAULT_SOURCE_META_FILENAME = "default_source_meta.json"


def resolve_sample_source_path(sample_item: Dict) -> str:
    """解析统一样本契约中的原始样本绝对路径。"""
    source_candidates = [
        sample_item.get("source_file"),
        sample_item.get("resolved_path"),
        sample_item.get("file_path"),
    ]
    for value in source_candidates:
        if value is None:
            continue
        path_text = str(value).strip()
        if path_text:
            return os.path.abspath(path_text)
    return ""


def load_packaged_default_source_meta(train_dir: str) -> Dict:
    """读取 train 目录中的默认预测源元数据。"""
    meta_path = os.path.join(
        os.path.abspath(str(train_dir)),
        DEFAULT_SOURCE_META_FILENAME,
    )
    if not os.path.exists(meta_path):
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            payload = json.load(meta_file)
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def resolve_packaged_default_source_file(train_dir: str) -> str:
    """解析 train 目录中已打包的默认预测源入口绝对路径。"""
    meta = load_packaged_default_source_meta(train_dir)
    entry_relpath = str(meta.get("entry_relpath") or "").strip()
    if not entry_relpath:
        return ""

    candidate = os.path.abspath(os.path.join(os.path.abspath(str(train_dir)), entry_relpath))
    if os.path.exists(candidate):
        return candidate
    return ""
