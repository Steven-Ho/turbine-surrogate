from typing import Any, Dict, Optional, Union

import pandas as pd


class BaseData:
    """统一的原始样本容器，连接 reader 与 processor。

    数据协议说明：
    1. task_type
       - "field"：`data` 表示一个场样本，或一个已经按时间聚合好的场时序样本。
       - "scalar"：`data` 表示一个标量表格样本，或一个标量时序样本。

    2. mode
       - "steady"：单个稳态样本。
       - "temporal"：时序样本，可能已经带时间轴，也可能按时间步列表存放。

    3. data 的约定
       - field + steady:
         `data` 通常是 `Dict[str, Any]`，常见字段包括：
         - `PointCoordinates`: 点坐标，形状通常为 `(N, 3)`
         - `CellCoordinates`: 单元中心坐标，形状通常为 `(M, 3)`
         - `ElementConnectivity`: 网格拓扑连接关系，
           通常可视为长度为 `M` 的单元列表；
           规则单元时可近似看作 `(M, k)`，
           非规则单元时常为 `np.ndarray(dtype=object, shape=(M,))`
         - `Fields`: 物理场字典
         - 以及 reader 额外补充的元信息
       - field + temporal:
         `data` 仍通常是 `Dict[str, Any]`，但其中字段值可能已经带时间维，
         或者是 reader 聚合后的时序结构。
       - scalar + steady:
         `data` 是一个 `pandas.DataFrame`。
         在 reader 刚读入时，它可以是一整张标量表；
         进入 steady 预处理后，通常会被按行拆分成单行 `DataFrame`，
         即一个样本对应表中的一行。
       - scalar + temporal:
         `data` 可以是一个 `pandas.DataFrame`，也可以是 `List[pandas.DataFrame]`。
         当按时间步列表存放时，`is_multi_time_step == True`。

    4. point_or_cell
       - 仅对 field 任务有意义，用来标记场值附着在 point 还是 cell 上。
       - 当 `point_or_cell == "point"` 时，`Fields` 中单字段通常满足：
         - steady: `(N,)`
         - temporal: `(T, N)`
       - 当 `point_or_cell == "cell"` 时，`Fields` 中单字段通常满足：
         - steady: `(M,)`
         - temporal: `(T, M)`

    5. sample_map
       - 可选的样本级元信息。
       - field 任务中通常来自样本表；
       - scalar 任务中可以为空。
    """

    # Defines `__init__` for   init   logic.
    def __init__(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], list],
        task_type: str,
        # Continues the surrounding statement or collection definition.
        mode: str,
        point_or_cell: str = "point",
        sample_map: Optional[Dict] = None,
    ):
        # 原始数据载体。
        # field 任务下通常是字典，内部常见字段有：
        # - PointCoordinates: (N, 3)
        # - CellCoordinates: (M, 3)
        # - ElementConnectivity: 长度为 M 的单元连接列表，或近似 (M, k)
        # - Fields: Dict[str, np.ndarray]
        #   * point + steady: 单字段通常为 (N,)
        #   * cell  + steady: 单字段通常为 (M,)
        #   * point + temporal: 单字段通常为 (T, N)
        #   * cell  + temporal: 单字段通常为 (T, M)
        # Reader 如果已经把多分量字段拆分成标量分量，
        # 则 BaseData 中单个 Fields[key] 通常仍是一维或二维数组。
        # scalar 任务下通常是 DataFrame，或由多个 DataFrame 组成的时间步列表。
        # Stores the current data on the instance so other methods can reuse it.
        self.data = data
        # 任务类型："field" / "scalar"。
        self.task_type = task_type
        # 任务模式："steady" / "temporal"。
        self.mode = mode
        # 场值位置标识，仅 field 任务使用："point" / "cell"。
        self.point_or_cell = point_or_cell
        # 样本级元信息，可来自样本表，scalar 任务可为空。
        # Stores the `sample_map` on the instance so other methods can reuse it.
        self.sample_map = sample_map

    def get_data(self) -> Any:
        """Return raw stored data."""
        return self.data

    def get_scalar_data(self) -> Any:
        """Return scalar task data in its original tabular form."""
        # Returns `self.data` to the caller for the next stage.
        return self.data

    def get_sample_map(self) -> Optional[Dict]:
        """Return the sample metadata mapping."""
        return self.sample_map

    def get_task_type(self) -> str:
        """Return task type: scalar or field."""
        # Returns `self.task_type` to the caller for the next stage.
        return self.task_type

    def get_mode(self) -> str:
        """Return task mode: steady or temporal."""
        return self.mode

    def get_point_or_cell(self) -> str:
        """Return field location representation: point or cell."""
        # Returns `self.point_or_cell` to the caller for the next stage.
        return self.point_or_cell

    def get_mesh_dimension(self) -> Optional[int]:
        """Return mesh topological dimension when the reader provides it."""
        if not isinstance(self.data, dict):
            return None
        # Reads `MeshDimension` from the mapping into the `value`.
        value = self.data.get("MeshDimension")
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            # Returns `None` to the caller for the next stage.
            return None

    def get_geometry_regime(self) -> Optional[str]:
        """返回几何体制标记，例如 static_geometry / displaced_static_topology。"""
        if not isinstance(self.data, dict):
            return None
        # Reads `GeometryRegime` from the mapping into the `geometry_regime`.
        geometry_regime = self.data.get("GeometryRegime")
        if geometry_regime:
            return str(geometry_regime)
        geometry_motion = self.data.get("GeometryMotion")
        # Returns early when `` makes the remaining work inapplicable.
        if isinstance(geometry_motion, dict) and geometry_motion.get("geometry_regime"):
            return str(geometry_motion.get("geometry_regime"))
        sample_map = self.sample_map or {}
        if isinstance(sample_map, dict) and sample_map.get("geometry_regime"):
            # Returns `str(sample_map.get("geometry_regime"))` to the caller for the next stage.
            return str(sample_map.get("geometry_regime"))
        return None

    @property
    def is_multi_time_step(self) -> bool:
        """Whether scalar temporal data is represented as multiple time-step tables."""
        # Returns `isinstance(self.data, list)` to the caller for the next stage.
        return isinstance(self.data, list)
