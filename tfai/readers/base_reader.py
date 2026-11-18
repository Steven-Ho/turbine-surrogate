from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ..data.base_data import BaseData

# Defines `BaseReader` to group related data and behavior.
class BaseReader(ABC):
    def __init__(self, config: Dict):
        """初始化读取器
        
        Args:
            config: 读取器配置参数
        """
        self.config = config
    
    @abstractmethod
    # 读取并解析文件，返回约定的数据类型
    def read(self, file_path: str, variables: Optional[List[str]] = None) -> BaseData:
        """读取并解析文件，返回约定的数据类型
        
        Args:
            file_path: 文件路径
            variables: 所需的物理量名称列表，None表示读取所有物理量
            
        Returns:
            解析后的数据
        """
        pass

    def resolve_variable_manifest(
        self,
        # Continues the surrounding statement or collection definition.
        file_path: str,
        logical_variables: List[str],
        sample_order: Optional[int] = None,
    ) -> Dict[str, object]:
        """Resolve logical variables into effective variables for training/prediction consistency.

        Field readers should override this method. The base implementation keeps a
        pass-through behavior for scalar-style readers.
        """
        # Returns `` to the caller for the next stage.
        return {
            'logical_variables': list(logical_variables),
            'effective_variables': list(logical_variables),
            'point_or_cell': self.config.get('task', {}).get('point_or_cell'),
        # Completes the multi-line `collection` arguments or elements.
        }
    
