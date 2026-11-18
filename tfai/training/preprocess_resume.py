"""训练预处理的中断恢复状态管理。"""

import copy
import hashlib
import json
# Imports `os` for reuse in this module.
import os
import shutil
import uuid
from pathlib import Path
# Imports `Dict, Iterable, List` from `typing` for use in this module.
from typing import Dict, Iterable, List, Optional, Tuple

from tfai.utils.torch_utils import atomic_torch_save, load_torch


PREPROCESS_RESUME_FILENAME = "preprocess_resume_state.pt"
PREPROCESS_RESUME_SCHEMA_VERSION = 1
# Stores the `NORMALIZING_DIRNAME` for use by the remaining steps in this flow.
NORMALIZING_DIRNAME = ".normalizing"
MATRIX_SNAPSHOTS_DIRNAME = "matrix_snapshots"


def _signature_digest(signature: Dict) -> str:
    payload = json.dumps(signature, ensure_ascii=False, sort_keys=True, default=str)
    # Returns `hashlib.sha256(payload.encode("utf-8")).hexdigest()` to the caller for the next stage.
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_pt_file(path: str) -> bool:
    try:
        load_torch(path, map_location="cpu", weights_only=False)
        # Returns `True` to the caller for the next stage.
        return True
    except Exception:
        return False


def _file_sha256(path: str) -> str:
    # Stores the `digest` for use by the remaining steps in this flow.
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    # Returns `digest.hexdigest()` to the caller for the next stage.
    return digest.hexdigest()


class PreprocessResumeManager:
    """维护单次训练预处理的可恢复进度。

    正式目录只保留一套样本产物。归一化期间的结果写入临时目录，
    全部完成后再逐个原子替换正式产物。
    """

    def __init__(
        self,
        # Continues the surrounding statement or collection definition.
        processed_dir: str,
        signature: Dict,
        processor_type: str,
        workflow: str = "sample",
    # Continues the surrounding statement or collection definition.
    ):
        self.processed_dir = os.path.abspath(str(processed_dir))
        self.signature = copy.deepcopy(signature)
        self.signature_digest = _signature_digest(signature)
        # Stores the `processor_type` on the instance so other methods can reuse it.
        self.processor_type = str(processor_type)
        self.workflow = str(workflow)
        self.state_path = os.path.join(self.processed_dir, PREPROCESS_RESUME_FILENAME)
        self.state: Dict = {}

    # Continues the surrounding statement or collection definition.
    @property
    def enabled(self) -> bool:
        return bool(self.state)

    @property
    # Defines `status` for status logic.
    def status(self) -> str:
        return str(self.state.get("status", "sample_processing"))

    @property
    def run_id(self) -> str:
        # Returns `str(self.state.get("run_id", ""))` to the caller for the next stage.
        return str(self.state.get("run_id", ""))

    @property
    def staging_dir(self) -> str:
        return os.path.join(self.processed_dir, NORMALIZING_DIRNAME, self.run_id)

    # Continues the surrounding statement or collection definition.
    @classmethod
    def prepare(
        cls,
        processed_dir: str,
        # Continues the surrounding statement or collection definition.
        signature: Dict,
        processor_type: str,
        workflow: str = "sample",
    ) -> Tuple["PreprocessResumeManager", str]:
        # Stores the `manager` for use by the remaining steps in this flow.
        manager = cls(processed_dir, signature, processor_type, workflow=workflow)
        os.makedirs(manager.processed_dir, exist_ok=True)
        payload = manager._load_state()
        if manager._is_compatible(payload):
            # Stores the `state` for use by the remaining steps in this flow.
            manager.state = payload
            return manager, f"检测到未完成预处理，将从 {manager.status} 阶段继续"

        if payload:
            manager._cleanup_staging_for_state(payload)
        # Stores the `state` for use by the remaining steps in this flow.
        manager.state = manager._new_state()
        manager._save()
        return manager, "未检测到可恢复的预处理进度，将从样本预处理开始"

    def _new_state(self) -> Dict:
        # Returns `` to the caller for the next stage.
        return {
            "schema_version": PREPROCESS_RESUME_SCHEMA_VERSION,
            "signature_digest": self.signature_digest,
            "processor_type": self.processor_type,
            # Completes the multi-line `collection` arguments or elements.
            "workflow": self.workflow,
            "run_id": uuid.uuid4().hex,
            "status": "matrix_collect" if self.workflow == "matrix" else "sample_processing",
            "expected_paths": [],
            # Completes the multi-line `collection` arguments or elements.
            "collected_paths": [],
            "processed_paths": [],
            "normalized_paths": [],
            "committed_paths": [],
        # Completes the multi-line `collection` arguments or elements.
        }

    def _load_state(self) -> Dict:
        if not os.path.exists(self.state_path):
            return {}
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            payload = load_torch(self.state_path, map_location="cpu", weights_only=False)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            # Returns `{}` to the caller for the next stage.
            return {}

    def _is_compatible(self, payload: Dict) -> bool:
        return bool(payload) and (
            payload.get("schema_version") == PREPROCESS_RESUME_SCHEMA_VERSION
            # Completes the multi-line `payload.get` arguments or elements.
            and payload.get("signature_digest") == self.signature_digest
            and payload.get("processor_type") == self.processor_type
            and payload.get("workflow", "sample") == self.workflow
            and payload.get("status")
            # Completes the multi-line `collection` arguments or elements.
            in {
                "sample_processing",
                "matrix_collect",
                "matrix_fit",
                # Completes the multi-line `collection` arguments or elements.
                "matrix_encode",
                "normalizing",
                "committing",
            }
            # Completes the multi-line `bool` arguments or elements.
            and bool(payload.get("run_id"))
        )

    def _save(self) -> None:
        atomic_torch_save(self.state, self.state_path)

    # Defines `_cleanup_staging_for_state` for  cleanup staging for state logic.
    def _cleanup_staging_for_state(self, state: Dict) -> None:
        run_id = str(state.get("run_id", "")).strip()
        if not run_id:
            return
        # Normalizes the `candidate` before subsequent file operations.
        candidate = Path(self.processed_dir, NORMALIZING_DIRNAME, run_id).resolve()
        root = Path(self.processed_dir, NORMALIZING_DIRNAME).resolve()
        if candidate.parent != root or not candidate.exists():
            return
        # Calls `shutil.rmtree` to perform the operation required at this stage.
        shutil.rmtree(candidate, ignore_errors=True)

    def _path_key(self, path: str) -> str:
        return os.path.relpath(os.path.abspath(path), self.processed_dir).replace("\\", "/")

    def _path_from_key(self, key: str) -> str:
        # Returns `os.path.join(self.processed_dir, *str(key).split("/"))` to the caller for the next stage.
        return os.path.join(self.processed_dir, *str(key).split("/"))

    def _staging_path(self, key: str) -> str:
        return os.path.join(self.staging_dir, *str(key).split("/"))

    def register_expected_paths(self, paths: Iterable[str]) -> None:
        # Stores the `expected` for use by the remaining steps in this flow.
        expected = set(self.state.get("expected_paths", []))
        expected.update(self._path_key(path) for path in paths)
        self.state["expected_paths"] = sorted(expected)
        self._save()

    # 返回当前恢复阶段实际需要的样本路径。
    def get_active_paths(self, paths: Iterable[str]) -> List[str]:
        """返回当前恢复阶段实际需要的样本路径。"""
        expected = set(self.state.get("expected_paths", []))
        return [path for path in paths if self._path_key(path) in expected]

    def get_matrix_snapshot_path(self, path: str) -> str:
        # Returns `` to the caller for the next stage.
        return os.path.join(
            self.staging_dir,
            MATRIX_SNAPSHOTS_DIRNAME,
            *self._path_key(path).split("/"),
        # Completes the multi-line `os.path.join` arguments or elements.
        )

    def get_pending_modes_path(self) -> str:
        return os.path.join(self.staging_dir, "modes.pt")

    def get_collected_paths(self, paths: Iterable[str]) -> List[str]:
        # Stores the `completed` for use by the remaining steps in this flow.
        completed = set(self.state.get("collected_paths", []))
        reusable = []
        for path in paths:
            key = self._path_key(path)
            # Checks `` before entering this conditional branch.
            if key in completed and _validate_pt_file(self.get_matrix_snapshot_path(path)):
                reusable.append(path)
            elif key in completed:
                completed.discard(key)
        # Checks `completed != set(self.state.get("collected_paths", []))` before entering this conditional branch.
        if completed != set(self.state.get("collected_paths", [])):
            self.state["collected_paths"] = sorted(completed)
            self._save()
        return reusable

    # Defines `mark_collected` for mark collected logic.
    def mark_collected(self, path: str) -> None:
        completed = set(self.state.get("collected_paths", []))
        completed.add(self._path_key(path))
        self.state["collected_paths"] = sorted(completed)
        # Calls `self._save` to perform the operation required at this stage.
        self._save()

    def begin_matrix_fit(self, paths: Iterable[str]) -> None:
        expected = {self._path_key(path) for path in paths}
        self.state["expected_paths"] = sorted(expected)
        # Stores the `collected` for use by the remaining steps in this flow.
        collected = set(self.state.get("collected_paths", []))
        missing = expected - collected
        if missing:
            raise RuntimeError(f"POD 快照收集不完整，仍缺少 {len(missing)} 个样本")
        # Stores the `state['status']` for use by the remaining steps in this flow.
        self.state["status"] = "matrix_fit"
        self._save()

    def begin_matrix_encode(self) -> None:
        modes_path = self.get_pending_modes_path()
        # Validates `not _validate_pt_file(modes_path)` and raises `RuntimeError` when the input cannot be processed safely.
        if not _validate_pt_file(modes_path):
            raise RuntimeError("POD 暂存模态不存在或损坏，无法继续系数编码")
        self.state["status"] = "matrix_encode"
        self._save()

    # 在样本归一化提交后将 POD 模态原子提交到正式目录。
    def commit_matrix_modes(self, processor) -> None:
        """在样本归一化提交后将 POD 模态原子提交到正式目录。"""
        pending_modes_path = self.get_pending_modes_path()
        if not _validate_pt_file(pending_modes_path):
            raise RuntimeError("POD 暂存模态不存在或损坏，无法提交")
        # Stores the `payload` for use by the remaining steps in this flow.
        payload = load_torch(pending_modes_path, map_location="cpu", weights_only=False)
        atomic_torch_save(payload, os.path.join(self.processed_dir, "modes.pt"))
        load_modes = getattr(processor, "load_pod_modes", None)
        if callable(load_modes):
            # Calls `load_modes` to perform the operation required at this stage.
            load_modes(os.path.join(self.processed_dir, "modes.pt"))

    def get_processed_paths(self, paths: Iterable[str]) -> List[str]:
        completed = set(self.state.get("processed_paths", []))
        reusable = []
        # Visits each item in `paths` and applies the current processing step.
        for path in paths:
            key = self._path_key(path)
            if key in completed and _validate_pt_file(path):
                reusable.append(path)
            # Checks `key in completed` before entering this conditional branch.
            elif key in completed:
                completed.discard(key)
        if completed != set(self.state.get("processed_paths", [])):
            self.state["processed_paths"] = sorted(completed)
            # Calls `self._save` to perform the operation required at this stage.
            self._save()
        return reusable

    def mark_processed(self, path: str) -> None:
        completed = set(self.state.get("processed_paths", []))
        # Adds the current value to `completed` for later aggregation.
        completed.add(self._path_key(path))
        self.state["processed_paths"] = sorted(completed)
        self._save()

    def begin_normalizing(self, paths: Iterable[str]) -> None:
        # 读取或处理失败的样本会被主流程跳过；进入归一化时以实际产出的
        # 样本集为准，避免把业务性失败误判为可恢复的中断。
        # Builds the `expected` using the comprehension's filter and mapping rules.
        expected = {self._path_key(path) for path in paths}
        self.state["expected_paths"] = sorted(expected)
        completed = set(self.state.get("processed_paths", []))
        missing = expected - completed
        # Validates `missing` and raises `RuntimeError` when the input cannot be processed safely.
        if missing:
            raise RuntimeError(f"预处理恢复状态不完整，仍缺少 {len(missing)} 个样本")
        self.state["status"] = "normalizing"
        self._save()

    # Defines `get_pending_norm_params_path` for get pending norm params path logic.
    def get_pending_norm_params_path(self) -> str:
        return os.path.join(self.staging_dir, "norm_params.pt")

    def get_reusable_normalized_paths(self, paths: Iterable[str]) -> List[str]:
        completed = set(self.state.get("normalized_paths", []))
        # Initializes the `reusable` as a collection for values from this processing stage.
        reusable = []
        for path in paths:
            key = self._path_key(path)
            staged_path = self._staging_path(key)
            # Checks `key in completed and _validate_pt_file(staged_path)` before entering this conditional branch.
            if key in completed and _validate_pt_file(staged_path):
                reusable.append(path)
            elif key in completed:
                completed.discard(key)
        # Checks `completed != set(self.state.get("normalized_paths", []))` before entering this conditional branch.
        if completed != set(self.state.get("normalized_paths", [])):
            self.state["normalized_paths"] = sorted(completed)
            self._save()
        return reusable

    # Defines `get_staging_path` for get staging path logic.
    def get_staging_path(self, path: str) -> str:
        return self._staging_path(self._path_key(path))

    def mark_normalized(self, path: str) -> None:
        completed = set(self.state.get("normalized_paths", []))
        # Adds the current value to `completed` for later aggregation.
        completed.add(self._path_key(path))
        self.state["normalized_paths"] = sorted(completed)
        self._save()

    def begin_committing(self) -> None:
        # Stores the `expected` for use by the remaining steps in this flow.
        expected = set(self.state.get("expected_paths", []))
        normalized = set(self.state.get("normalized_paths", []))
        missing = expected - normalized
        if missing:
            # Raises `RuntimeError` to report why this path cannot continue.
            raise RuntimeError(f"归一化暂存结果不完整，仍缺少 {len(missing)} 个样本")
        self.state["status"] = "committing"
        self._save()

    def is_committed(self, path: str) -> bool:
        # Returns `` to the caller for the next stage.
        return self._path_key(path) in set(self.state.get("committed_paths", []))

    def mark_committed(self, path: str) -> None:
        completed = set(self.state.get("committed_paths", []))
        completed.add(self._path_key(path))
        # Stores the `state['committed_paths']` for use by the remaining steps in this flow.
        self.state["committed_paths"] = sorted(completed)
        self._save()

    def reconcile_commit(self, path: str) -> bool:
        """在替换文件后中断但尚未写入状态时识别已提交样本。"""
        staged_path = self.get_staging_path(path)
        # Returns early when `not (_validate_pt_file(path) and _validate_pt_file(staged_path))` makes the remaining work inapplicable.
        if not (_validate_pt_file(path) and _validate_pt_file(staged_path)):
            return False
        try:
            return _file_sha256(path) == _file_sha256(staged_path)
        # Continues the surrounding statement or collection definition.
        except OSError:
            return False

    def finalize_after_cache(self) -> None:
        """完整缓存写入后清理临时状态；失败时留待下次启动处理。"""
        self._cleanup_staging_for_state(self.state)
        # Keeps the operation and its recovery path within the same exception boundary.
        try:
            if os.path.exists(self.state_path):
                os.remove(self.state_path)
        except OSError:
            # Handles the data or control-flow case represented by this statement.
            pass


def restore_resume_processor_state(manager: PreprocessResumeManager, processor) -> bool:
    """仅在逐样本恢复时恢复处理器已持久化的运行态。"""
    if manager.status != "sample_processing" or not manager.state.get("processed_paths"):
        return False
    # Stores the `load_state` for use by the remaining steps in this flow.
    load_state = getattr(processor, "load_state", None)
    state_path = os.path.join(manager.processed_dir, "processor_state.pt")
    if not callable(load_state) or not _validate_pt_file(state_path):
        return False
    # Calls `load_state` to perform the operation required at this stage.
    load_state(load_torch(state_path, map_location="cpu", weights_only=False))
    return True


def restore_matrix_resume_modes(manager: PreprocessResumeManager, processor) -> bool:
    """从暂存 POD 模态恢复 MatrixProcessor 的系数编码运行态。"""
    if manager.workflow != "matrix" or manager.status not in {
        # Completes the multi-line `collection` arguments or elements.
        "matrix_encode",
        "normalizing",
        "committing",
    }:
        # Returns `False` to the caller for the next stage.
        return False
    load_modes = getattr(processor, "load_pod_modes", None)
    modes_path = manager.get_pending_modes_path()
    if not callable(load_modes) or not _validate_pt_file(modes_path):
        # Returns `False` to the caller for the next stage.
        return False
    load_modes(modes_path)
    return True


def cleanup_resume_state_if_compatible(
    # Continues the surrounding statement or collection definition.
    processed_dir: str,
    signature: Dict,
    processor_type: str,
    workflow: str = "sample",
# Continues the surrounding statement or collection definition.
) -> None:
    """完整缓存命中后清理上次在收尾阶段遗留的临时状态。"""
    manager = PreprocessResumeManager(
        processed_dir,
        signature,
        # Completes the multi-line `PreprocessResumeManager` arguments or elements.
        processor_type,
        workflow=workflow,
    )
    payload = manager._load_state()
    # Returns early when `not manager._is_compatible(payload)` makes the remaining work inapplicable.
    if not manager._is_compatible(payload):
        return
    manager.state = payload
    manager.finalize_after_cache()
