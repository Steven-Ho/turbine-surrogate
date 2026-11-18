"""
This file contains the methods to read the mesh infos from the CGNS file.
"""
import logging
import sys
from typing import List, Optional, Tuple

# Imports `numpy` for reuse in this module.
import numpy as np

from . import cgnskeywords

logger = logging.getLogger(__name__)  # module specific logger


def get_corner_node_count(elem_type: int) -> Optional[int]:
    """
    返回单元族对应的一阶角点数量。
    对高阶 Lagrange 单元，默认取其角点子集用于几何近似。
    """
    # Returns early when `elem_type in cgnskeywords.ElementType0D` makes the remaining work inapplicable.
    if elem_type in cgnskeywords.ElementType0D:
        return 1
    if elem_type in cgnskeywords.ElementType1D:
        return 2
    # Returns early when `elem_type in cgnskeywords.ElementType_tri` makes the remaining work inapplicable.
    if elem_type in cgnskeywords.ElementType_tri:
        return 3
    if elem_type in cgnskeywords.ElementType_quad:
        return 4
    # Returns early when `elem_type in cgnskeywords.ElementType_tetra` makes the remaining work inapplicable.
    if elem_type in cgnskeywords.ElementType_tetra:
        return 4
    if elem_type in cgnskeywords.ElementType_pyra:
        return 5
    # Returns early when `elem_type in cgnskeywords.ElementType_penta` makes the remaining work inapplicable.
    if elem_type in cgnskeywords.ElementType_penta:
        return 6
    if elem_type in cgnskeywords.ElementType_hexa:
        return 8
    # Returns `None` to the caller for the next stage.
    return None


def infer_volume_corner_node_count_from_size(node_count: int) -> Optional[int]:
    """按 3D 体单元常见节点数启发式推断角点数量。"""
    tetra_sizes = {
        cgnskeywords.ElementTypeNPE_l[elem_type]
        # Continues the surrounding statement or collection definition.
        for elem_type in cgnskeywords.ElementType_tetra
    }
    pyra_sizes = {
        cgnskeywords.ElementTypeNPE_l[elem_type]
        # Continues the surrounding statement or collection definition.
        for elem_type in cgnskeywords.ElementType_pyra
    }
    penta_sizes = {
        cgnskeywords.ElementTypeNPE_l[elem_type]
        # Continues the surrounding statement or collection definition.
        for elem_type in cgnskeywords.ElementType_penta
    }
    hexa_sizes = {
        cgnskeywords.ElementTypeNPE_l[elem_type]
        # Continues the surrounding statement or collection definition.
        for elem_type in cgnskeywords.ElementType_hexa
    }

    if node_count in tetra_sizes:
        return 4
    # Returns early when `node_count in pyra_sizes` makes the remaining work inapplicable.
    if node_count in pyra_sizes:
        return 5
    if node_count in penta_sizes:
        return 6
    # Returns early when `node_count in hexa_sizes` makes the remaining work inapplicable.
    if node_count in hexa_sizes:
        return 8
    return None


def extract_corner_nodes(
    # Continues the surrounding statement or collection definition.
    nodes: np.ndarray,
    elem_type: Optional[int] = None,
    prefer_volume: bool = False,
) -> np.ndarray:
    """
    提取单元的角点节点。

    - 若提供 elem_type，则按 CGNS 单元族精确提取；
    - 否则在 prefer_volume=True 时，按常见 3D 体单元节点数启发式提取；
    - 其余情况回退为原始节点列表。
    """
    # Stores the `node_array` for use by the remaining steps in this flow.
    node_array = np.asarray(nodes, dtype=np.int64).reshape(-1)
    if node_array.size == 0:
        return node_array

    corner_count = None
    # Checks `elem_type is not None` before entering this conditional branch.
    if elem_type is not None:
        corner_count = get_corner_node_count(int(elem_type))
    elif prefer_volume:
        corner_count = infer_volume_corner_node_count_from_size(int(node_array.size))

    # Returns early when `` makes the remaining work inapplicable.
    if corner_count is None or corner_count <= 0 or corner_count >= node_array.size:
        return node_array
    return node_array[:corner_count]


def read_mixed_mesh(connectivity: np.ndarray) -> set:
    """
    This function reads the mixed mesh from the connectivity array.
    The connectivity array is a 1D array: each segment contains first the
    element type integer, followed by the node index.
    """

    # Stores the `edges` for use by the remaining steps in this flow.
    edges = set()
    start_idx = 0

    while start_idx < len(connectivity):
        # first determine the element type
        elem_type = connectivity[start_idx]
        # first check if the elem_type is supoorted
        # Checks `elem_type not in cgnskeywords.ElementType_Supported` before entering this conditional branch.
        if elem_type not in cgnskeywords.ElementType_Supported:
            emsg = f"Element type {elem_type} is not supported. "
            logger.error(emsg)
            raise ValueError(emsg)
        # look up for the element type
        # elem_type_s = cgnskeywords.ElementType_[elem_type]
        # look up for the npe (number of points per element)
        # npe = cgnskeywords.ElementTypeNPE[elem_type_s]
        # Selects the required item from `cgnskeywords.ElementTypeNPE_l` as the `npe`.
        npe = cgnskeywords.ElementTypeNPE_l[elem_type]

        # get the end index, which points to the element_type of the next element
        end_idx = start_idx + npe + 1
        # get the nodes
        nodes = connectivity[start_idx+1:end_idx]
        # add the edges to the set
        edges.update(get_edges(elem_type, nodes))
        # update the start index
        # Stores the `start_idx` for use by the remaining steps in this flow.
        start_idx = end_idx

    return edges


def _get_corner_edge_pairs(elem_type: int) -> List[Tuple[int, int]]:
    """返回单元角点层面的边连接模板。"""
    if elem_type in cgnskeywords.ElementType0D:
        # Returns `[]` to the caller for the next stage.
        return []
    if elem_type in cgnskeywords.ElementType1D:
        return [(0, 1)]
    if elem_type in cgnskeywords.ElementType_tri:
        # Returns `[(0, 1), (1, 2), (2, 0)]` to the caller for the next stage.
        return [(0, 1), (1, 2), (2, 0)]
    if elem_type in cgnskeywords.ElementType_quad:
        return [(0, 1), (1, 2), (2, 3), (3, 0)]
    if elem_type in cgnskeywords.ElementType_tetra:
        # Returns `[(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]` to the caller for the next stage.
        return [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    if elem_type in cgnskeywords.ElementType_pyra:
        return [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
    if elem_type in cgnskeywords.ElementType_penta:
        # Returns `` to the caller for the next stage.
        return [(0, 1), (1, 2), (2, 0), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (5, 3)]
    if elem_type in cgnskeywords.ElementType_hexa:
        return [
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Completes the multi-line `collection` arguments or elements.
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (7, 4),
        ]

    emsg = f"Element type {elem_type} is not supported. "
    # Records the current execution state to make progress and failures traceable.
    logger.error(emsg)
    raise ValueError(emsg)


def _build_edges_from_corner_pairs(
    elem_type: int,
    # Continues the surrounding statement or collection definition.
    nodes: np.ndarray,
    bidirectional: bool = False,
) -> set:
    """按单元族角点模板生成边集合。"""
    corner_nodes = extract_corner_nodes(nodes, elem_type=elem_type)
    # Stores the `edge_pairs` for use by the remaining steps in this flow.
    edge_pairs = _get_corner_edge_pairs(elem_type)
    edges = set()

    for start_idx, end_idx in edge_pairs:
        if start_idx >= len(corner_nodes) or end_idx >= len(corner_nodes):
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = (
                f"Element type {elem_type} corner count is insufficient for edge extraction: "
                f"{len(corner_nodes)}"
            )
            # Records the current execution state to make progress and failures traceable.
            logger.error(emsg)
            raise ValueError(emsg)
        start_node = int(corner_nodes[start_idx])
        end_node = int(corner_nodes[end_idx])
        # Adds the current value to `edges` for later aggregation.
        edges.add((start_node, end_node))
        if bidirectional:
            edges.add((end_node, start_node))

    return edges


# This function returns the edges of the element.
def get_edges(elem_type: int, nodes: np.ndarray) -> set:
    """
    This function returns the edges of the element.
    """
    return _build_edges_from_corner_pairs(elem_type, nodes, bidirectional=False)


def get_edges_bidirectional(elem_type: int, nodes: np.ndarray) -> set:
    """
    This function returns the edges of the element.
    Sometimes the edges are bidirectional, we need add both directions to the set.
    """
    return _build_edges_from_corner_pairs(elem_type, nodes, bidirectional=True)


# Get the edges of the graph including the self-loops.
def get_edges_selflooped(elem_type: int, nodes: np.ndarray) -> set:
    """
    Get the edges of the graph including the self-loops.
    """
    # edges = get_edges(elem_type, nodes)
    edges = get_edges_bidirectional(elem_type, nodes)
    for node in nodes:
        edges.add((node, node))

    # Returns `edges` to the caller for the next stage.
    return edges


def main():
    pass


if __name__ == "__main__":
    # Calls `main` to perform the operation required at this stage.
    main()
