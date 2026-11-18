""" This file is for the data manipulation of the CGNS file, similar to
    the filters often used in Paraview.
"""

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Imports `numpy` for reuse in this module.
import numpy as np
# scipy modules for interpolation, depend on scipy >= 1.7.0
from scipy.interpolate import RBFInterpolator as RBF
from scipy.spatial import cKDTree

from . import cgnskeywords
# Imports `CGNSNodeType_S, Node` from `tf_cgns` for use in this module.
from .tf_cgns import CGNSNodeType_S, Node

logger = logging.getLogger(__name__)  # module specific logger


class BaseFilter(object):
    """ The base class for the filter. """

    def __init__(self, zone: Node) -> None:
        """
        Args:
            zone (Node): the zone node
        """
        # Stores the `zone` on the instance so other methods can reuse it.
        self.zone = zone

    def apply(self, required_fields: Optional[List[str]] = None) -> bool:
        """
        Apply the filter to the zone.

        Args:
            required_fields (List[str]): the required fields,
                e.g. ["Density", "Velocity", "Pressure"]

        Returns:
            bool: True if the filter is applied successfully, False otherwise.
        """
        raise NotImplementedError("The apply method is not implemented. ")


class CellToNodeFilter(BaseFilter):
    """ The filter for the cell to node conversion. """

    # Args:
    def __init__(self, zone: Node) -> None:
        """
        Args:
            zone (Node): the zone node
        """
        super().__init__(zone)

    def apply(self, required_fields: Optional[List[str]] = None) -> bool:
        """
        Apply the filter to the zone.

        Args:
            required_fields (List[str]): the required fields,
                e.g. ["Density", "Velocity", "Pressure"]

        Returns:
            bool: True if the filter is applied successfully, False otherwise.
        """
        # get the mesh connectivity from the zone
        connectivity = self.zone.get_mesh_connectivity()
        # Selects the required item from `connectivity["Elem2Node"]` as the `elem2node`.
        elem2node = connectivity["Elem2Node"]["connectivity"]
        # get the mesh size from the zone
        nnode = self.zone.get_mesh_size()["VertexSize"]

        # get the solution node
        sol_node = self.zone.get_solution_node()

        # check if the solutions are cell centered
        if sol_node.children["GridLocation"].data != \
           np.bytes_(b"CellCenter"):
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = "The solution is not cell centered. "
            logger.error(emsg)
            raise ValueError(emsg)

        # delete the GridLocation node and PointRange node if they exist
        for node_keys in ["GridLocation", "PointRange"]:
            # Checks `node_keys in sol_node.children.keys()` before entering this conditional branch.
            if node_keys in sol_node.children.keys():
                del sol_node.children[node_keys]

        # delete fields that are not required
        if required_fields is not None:
            for field in list(sol_node.children.keys()):
                # Checks `field not in required_fields` before entering this conditional branch.
                if field not in required_fields:
                    del sol_node.children[field]
            # check if the required fields are in the solution node
            for field in required_fields:
                if field not in sol_node.children.keys():
                    # Stores the `emsg` for use by the remaining steps in this flow.
                    emsg = f"Field {field} is not in the solution node. "
                    logger.error(emsg)
                    raise ValueError(emsg)
        else:
            # Stores the `required_fields` for use by the remaining steps in this flow.
            required_fields = list(sol_node.children.keys())

        # initialize the node data
        node_data = {f: np.zeros([2, nnode]) for f in required_fields}

        # loop over the required fields
        for field in required_fields:
            # get the filed node first
            field_node = sol_node.children[field]
            # get the data
            # Stores the `field_data` for use by the remaining steps in this flow.
            field_data = field_node.data

            # loop over the elements
            for ielem in range(len(elem2node)):
                # get the nodes of the element
                nodes = elem2node[ielem]
                # convert the cgns node index to python node index
                nodes = nodes - 1
                # update the node data
                # Selects the required item from `field_data` as the `node_data[field][0, nodes]`.
                node_data[field][0, nodes] += field_data[ielem]
                node_data[field][1, nodes] += 1

            # compute the mean
            node_data[field] = node_data[field][0, :] / node_data[field][1, :]

        # replace fields in the solution node
        for field in required_fields:
            # Selects the required item from `node_data` as the current data.
            sol_node.children[field].data = node_data[field]

        return True


class CellToNodeConverter(object):

    def __init__(self, coords: np.ndarray, connectivity: np.ndarray,
                 # Continues the surrounding statement or collection definition.
                 method: str = "mean"):
        """"
        Converter for cell data to node(vertex) data conversion.

        Args:
            coords (np.ndarray): coordinates of the nodes, shape (nnode, ndim)
            connectivity (np.ndarray): connectivity array, shape = ncell + \sum_{i=1}^{ncell} npe_i,
                note that the connectivity array mark the node index starting from 1.

        Returns:
            None
        """
        self.coords = coords
        self.connectivity = connectivity
        self.coords_cell = None  # coordinates of the cell centers
        # Stores the `node_info` on the instance so other methods can reuse it.
        self.node_info = None    # node info, a list of list, each list contains the cell indices that share the node
        self.tree = None         # kdTree struct if complex interpolation is required
        try:
            self.convertor = getattr(self, f"_convert_{method}")
        # Continues the surrounding statement or collection definition.
        except AttributeError:
            emsg = f"Method {method} is not supported. "
            logger.error(emsg)
            raise ValueError(emsg)

    # Preprocess the data.
    def _preprocess(self):
        """
        Preprocess the data.
        """
        # initialize the coords_cell as a empty list
        self.coords_cell = []
        # initialize the node_info as a list of list, each list contains the cell indices
        # that share the node
        self.node_info = [[] for _ in range(len(self.coords))]

        # loop over the cells in the connectivity array
        start_index = 0
        # Stores the `i_cell` for use by the remaining steps in this flow.
        i_cell = 0
        while start_index < len(self.connectivity):
            # first determine the element type
            elem_type = self.connectivity[start_index]
            # check if the elem_type is supported
            if elem_type not in cgnskeywords.ElementType_Supported:
                # Stores the `emsg` for use by the remaining steps in this flow.
                emsg = f"Element type {elem_type} is not supported. "
                logger.error(emsg)
                raise ValueError(emsg)

            # look up for the npe (number of points per element)
            npe = cgnskeywords.ElementTypeNPE_l[elem_type]

            # get the end index, which points to the element_type of the next element
            # Derives the `end_index` from the current values for downstream processing.
            end_index = start_index + npe + 1

            # get the nodes
            nodes = self.connectivity[start_index+1:end_index]

            # minus -1 since the node index starts from 1
            nodes = nodes - 1

            # update the node_info, add cell index to the node_info list
            for node in nodes:
                # Adds the current value to `self.node_info[node]` for later aggregation.
                self.node_info[node].append(i_cell)

            # compute the center of the cell
            self.coords_cell.append(np.mean(self.coords[nodes], axis=0))

            # update the start index
            start_index = end_index

            # update the cell index
            i_cell += 1

        # Stores the `coords_cell` on the instance so other methods can reuse it.
        self.coords_cell = np.array(self.coords_cell)
        # self.node_info = np.array(self.node_info)

    def _convert_mean(self, data_cell: np.ndarray) -> np.ndarray:
        """
        Convert the given data from cell data to node data, using simple averaging.

        Args:
            data_cell (np.ndarray): cell data, shape (ncell, ndim)

        Returns:
            np.ndarray: node data, shape (nnode, ndim)
        """

        # preprocess if not done yet
        if self.coords_cell is None or self.node_info is None:
            self._preprocess()

        # initialize the node data
        # Stores the `nnode` for use by the remaining steps in this flow.
        nnode = len(self.node_info)

        if len(data_cell.shape) > 1:
            emsg = f"Only scalar data is supported. "
            logger.error(emsg)
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(emsg)

        data_node = np.zeros(nnode)

        # loop over the nodes and compute the average
        for i_node in range(nnode):
            # get the cell indices that share the node
            cell_indices = self.node_info[i_node]
            # compute the average
            # Stores the `data_node[i_node]` for use by the remaining steps in this flow.
            data_node[i_node] = np.mean(data_cell[cell_indices], axis=0)

        return data_node

    def _convert_inverse_dist(self, data_cell: np.ndarray) -> np.ndarray:
        """
        Convert the given data from cell data to node data, using inverse distance weighting.

        Args:
            data_cell (np.ndarray): cell data, shape (ncell, ndim)

        Returns:
            np.ndarray: node data, shape (nnode, ndim)
        """

        # preprocess if not done yet
        if self.coords_cell is None or self.node_info is None:
            # Calls `self._preprocess` to perform the operation required at this stage.
            self._preprocess()

        # initialize the node data
        nnode = len(self.node_info)

        if len(data_cell.shape) > 1:
            emsg = f"Only scalar data is supported. "
            # Records the current execution state to make progress and failures traceable.
            logger.error(emsg)
            raise ValueError(emsg)

        data_node = np.zeros(nnode)

        # loop over the nodes and do the interpolation
        for i_node in range(nnode):
            # get the cell indices that share the node
            # Selects the required item from `self.node_info` as the `cell_indices`.
            cell_indices = self.node_info[i_node]
            # compute the inverse distance
            dist = np.linalg.norm(self.coords_cell[cell_indices] - self.coords[i_node], axis=1)
            # compute the weight
            weight = 1.0 / dist
            # normalize the weight
            weight = weight / np.sum(weight)
            # interpolate the data
            # Stores the `data_node[i_node]` for use by the remaining steps in this flow.
            data_node[i_node] = np.sum(data_cell[cell_indices] * weight, axis=0)

        return data_node

    def _convert_nearest(self, data_cell: np.ndarray) -> np.ndarray:
        """
        Convert the given data from cell data to node data, using nearest neighbors
        instead of the cells that share the node.
        In this method, the nearest neighbor searching is done using the kdTree.
        The interpolation is done using the radial basis function.
        """

        # preprocess if not done yet
        if self.coords_cell is None or self.node_info is None:
            # Calls `self._preprocess` to perform the operation required at this stage.
            self._preprocess()

        # build the kdTree if not done yet
        if self.tree is None:
            self.tree = cKDTree(self.coords_cell)

        # initialize the node data
        nnode = len(self.node_info)

        # Checks `len(data_cell.shape) > 1` before entering this conditional branch.
        if len(data_cell.shape) > 1:
            emsg = f"Only scalar data is supported. "
            logger.error(emsg)
            raise ValueError(emsg)

        # Stores the `data_node` for use by the remaining steps in this flow.
        data_node = np.zeros(nnode)

        # loop over the nodes and do the interpolation
        for i_node in range(nnode):
            # find the nearest neighbors
            _, cell_indices = self.tree.query(self.coords[i_node], k=10)
            # compute the radial basis function
            rbf = RBF(self.coords_cell[cell_indices], data_cell[cell_indices], kernel="linear")
            # interpolate the data
            # Stores the `data_node[i_node]` for use by the remaining steps in this flow.
            data_node[i_node] = rbf(self.coords[i_node].reshape(1, -1))

        return data_node

    def convert(self, data_cell: np.ndarray) -> np.ndarray:
        """
        A wrapper for the conversion function.
        """

        return self.convertor(data_cell)


# Extract the cell center coordinates from the node coordinates and the
def extract_cell_center(coords_node: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """
    Extract the cell center coordinates from the node coordinates and the
    connectivity array.

    Args:
        coords_node (np.ndarray): coordinates of the nodes, shape (nnode, ndim)
        connectivity (np.ndarray): connectivity array, shape = \sum_{i=1}^{ncell} npe_i,
            note that the connectivity array mark the node index starting from 1.

    Returns:
        np.ndarray: cell center coordinates, shape (ncell, ndim)
    """

    coords_cell = []
    start_idx = 0

    while start_idx < len(connectivity):
        # first determine the element type
        # Selects the required item from `connectivity` as the `elem_type`.
        elem_type = connectivity[start_idx]
        # check if the elem_type is supported
        if elem_type not in cgnskeywords.ElementType_Supported:
            emsg = f"Element type {elem_type} is not supported. "
            logger.error(emsg)
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(emsg)

        # look up for the npe (number of points per element)
        npe = cgnskeywords.ElementTypeNPE_l[elem_type]

        # get the end index, which points to the element_type of the next element
        end_idx = start_idx + npe + 1

        # get the nodes
        nodes = connectivity[start_idx+1:end_idx]

        # compute the center of the cell
        # Adds the current value to `coords_cell` for later aggregation.
        coords_cell.append(np.mean(coords_node[nodes-1], axis=0))

        # update the start index
        start_idx = end_idx

    return np.array(coords_cell)


def main():

    # Handles the data or control-flow case represented by this statement.
    pass


if __name__ == "__main__":
    main()
