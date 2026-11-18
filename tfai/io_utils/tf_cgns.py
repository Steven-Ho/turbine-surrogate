"""
This file contains the CGNSData class, which is used to read and write
CGNS files. The hierarchy of the data is stored in a tree structure, with
each node containing a list of children nodes.
"""

import logging
import sys
import time
# Imports `Dict, List, Optional` from `typing` for use in this module.
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from . import cgnskeywords
from . import cgnsmesh

# get a module-specific logger
# Stores the run logger for use by the remaining steps in this flow.
logger = logging.getLogger(__name__)


class CGNSNodeType_E:
    """
    An enum to store the node types of the CGNS data.
    """
    Root = 0
    CGNSBase = 1
    # Stores the `CGNSLibraryVersion` for use by the remaining steps in this flow.
    CGNSLibraryVersion = 1
    BaseIterativeData = 2
    Zone = 2
    Elements = 3
    # Stores the `GridCoordinates` for use by the remaining steps in this flow.
    GridCoordinates = 3
    FlowSolution = 3
    ZoneBC = 3
    BC = 4
    # types below may have different levels in the same CGNS file
    # Stores the `GridLocation` for use by the remaining steps in this flow.
    GridLocation = max(4, 5)
    DataArray = max(4, 5)
    IndexRange = max(4, 5)


class CGNSNodeType_S:
    """
    CGNS node types in string format.
    """
    # Stores the `Root` for use by the remaining steps in this flow.
    Root = "Root Node of HDF5 File"
    CGNSBase = "CGNSBase_t"
    CGNSLibraryVersion = "CGNSLibraryVersion_t"
    BaseIterativeData = "BaseIterativeData_t"
    # Stores the `Zone` for use by the remaining steps in this flow.
    Zone = "Zone_t"
    Elements = "Elements_t"
    GridCoordinates = "GridCoordinates_t"
    FlowSolution = "FlowSolution_t"
    # Stores the `ZoneBC` for use by the remaining steps in this flow.
    ZoneBC = "ZoneBC_t"
    BC = "BC_t"
    # types below may have different levels in the same CGNS file
    #FlowSolution节点下的GridLocation、各种场和PointRange等的标签。结果场的标签为DataArray_t
    GridLocation = "GridLocation_t"
    DataArray = "DataArray_t"
    # Stores the `IndexRange` for use by the remaining steps in this flow.
    IndexRange = "IndexRange_t"


# map the string format to the enum format
CGNSNodeType_S2E = {  # type: Dict[str, CGNSNodeType_E]
    n: getattr(CGNSNodeType_E, n)\
    for n in CGNSNodeType_S.__dict__.keys() if not n.startswith("__")
}


# the CGNS boundary condition types enumerate
# An enum to store the boundary condition types of the CGNS data.
class CGNSBCType_E:
    """
    An enum to store the boundary condition types of the CGNS data.
    """
    # interior
    BCInterior = 0
    # weak boundaries
    BCSymmetryPlane = 1
    BCOutflow = 2
    # Stores the `BCTunnelOutflow` for use by the remaining steps in this flow.
    BCTunnelOutflow = 2
    # strong boundaries
    BCWall = 3
    BCInflow = 4
    # tower boundaries
    Tower_BCWall = 3
    # Stores the `Tower_BCInflow` for use by the remaining steps in this flow.
    Tower_BCInflow = 5  # same as Inflow, but is tied to the tower
    # chimney boundaries
    Chimney_BCWall = 3
    Chimney_BCInflow = 6  # same as Inflow, but is tied to the chimney
    # interface boundaries
    Tower_BCInterface_In = 10
    # Stores the `Tower_BCInterface_Out` for use by the remaining steps in this flow.
    Tower_BCInterface_Out = 11


# the CGNS boundary condition types string
class CGNSBCType_S:
    """
    CGNS boundary condition types in string format.
    """
    # interior
    BCInterior = "BCInterior"
    # weak boundaries
    BCSymmetryPlane = "BCSymmetryPlane"
    # Stores the `BCOutflow` for use by the remaining steps in this flow.
    BCOutflow = "BCOutflow"
    BCTunnelOutflow = "BCTunnelOutflow"
    # strong boundaries
    BCWall = "BCWall"
    BCInflow = "BCInflow"
    # tower boundaries
    # Stores the `Tower_BCWall` for use by the remaining steps in this flow.
    Tower_BCWall = "Tower_BCWall"
    Tower_BCInflow = "Tower_BCInflow"  # same as Inflow, but is tied to the tower
    # chimney boundaries
    Chimney_BCWall = "Chimney_BCWall"
    Chimney_BCInflow = "Chimney_BCInflow"  # same as Inflow, but is tied to the chimney
    # interface boundaries
    # Stores the `Tower_BCInterface_In` for use by the remaining steps in this flow.
    Tower_BCInterface_In = "Tower_BCInterface_In"
    Tower_BCInterface_Out = "Tower_BCInterface_Out"


# map the string format to the enum format
CGNSBCType_S2E = {  # type: Dict[str, CGNSBCType_E]
    n: getattr(CGNSBCType_E, n)\
    for n in CGNSBCType_S.__dict__.keys() if not n.startswith("__")
# Continues the surrounding statement or collection definition.
}


class Node(object):
    """
    Node class to hold the hdf5 implementation of the CGNS data in
    a tree structure. The tree structure is used to represent the
    hierarchical structure of the CGNS data. The tree structure is
    implemented using a dictionary of children nodes. The children
    nodes are stored in a dictionary, with the key being the name of
    the node and the value being the node itself.
    """

    def __init__(self, name: str, label: str, flags: Optional[int] = None,
                 data: Optional[np.ndarray] = None,
                 # Continues the surrounding statement or collection definition.
                 children: Optional[dict] = None) -> None:
        """
        Initialize the node class.

        Args:
            name (str): The name of the node.
            label (str): The label of the node, usually can be interpreted
                as the data type of the current node. For example, the
                root node often has the label  'Root Node of HDF5 File',
                the 'CGNSBase_t' often associates with the base node,
                'GridCoordinates_t' with the mesh node coordinates,
                'Elements_t' with the surface/volume meshes.
            flags (int, optional): The flags of the node.
                Right now, the root node has value None,
                while others has 1 by default.
            data (np.ndarray, optional): The data of the node, store as
                a numpy array. The data is stored in the data attribute
                of the node. The data is stored in the Fortran order in
                the original hdf5 format. However, the data is transposed
                when read in, see io module for details.
            children (dict, optional): The children of the node, stored
                as a dictionary. The key is the name of the child node,
                while the value is the child node itself.
        """
        self.name = name
        self.label = label
        self.flags = flags  # currently, none denotes the root node
        # Stores the current data on the instance so other methods can reuse it.
        self.data = data
        self.children = children

    # ----------------------------------------------------------------------- #
    # The following functions are used for node info printing                 #
    # ----------------------------------------------------------------------- #

    def __repr__(self) -> str:
        """
        Return a simple representation of the node.
        """
        out_string = f"Node name: {self.name}\n"
        # Stores the `out_string` for use by the remaining steps in this flow.
        out_string += f"Node label: {self.label}\n"
        out_string += f"Node flags: {self.flags}\n"
        out_string += f"Node data: {self.data}\n"
        if self.children is not None:
            # Stores the `out_string` for use by the remaining steps in this flow.
            out_string += f"Node children: {self.children.keys()}\n"

        return out_string

    def _print_child(self, prefix: Optional[str] = "",
                     indent: Optional[int] = 2) -> None:
        """
        Print the representation of all children in the current node,
        whose behavior mimics the tree command in linux.

        Args:
            prefix (str, optional): The prefix appended to the tree repr.
            indent (int, optional): The size of the indentation.
        """
        # Stores the `top_vline` for use by the remaining steps in this flow.
        top_vline = u"\u251C"  # ├
        bot_vline = u"\u2514"  # └
        indent_vline = u"\u2502"  # │
        hline = u"\u2500"  # ─

        # Checks `self.children is not None` before entering this conditional branch.
        if self.children is not None:
            n_child = len(self.children)
            for idx, (cname, child) in enumerate(self.children.items()):
                is_bottom = (idx == n_child-1)
                # Stores the `(vline, vvline)` for use by the remaining steps in this flow.
                vline, vvline = (bot_vline, " ") if is_bottom else \
                    (top_vline, indent_vline)
                # print the current child
                print(prefix + vline + indent*hline + f"{cname}")
                # recursively print the child's tree
                child._print_child(prefix + vvline + indent*" ", indent)

    def print_tree(self, indent: Optional[int] = 2) -> None:
        """
        Print the tree representation of the node.

        Args:
            indent (int, optional): The size of the indentation.
        """
        # Calls `print` to perform the operation required at this stage.
        print(self.name)
        self._print_child(indent=indent)

    # ---------------------------------------------------------------------- #
    # The following functions are used for node insertion, deletion and      #
    # searching, which are building blocks for tree node look-up and         #
    # manipulations.                                                         #
    # ---------------------------------------------------------------------- #
    def add_child(self, child: 'Node') -> None:
        """
        Add a child to the current node.

        Args:
            child (Node): The child node to be added.
        """

        # store the child in the dict of children
        if self.children is None:
            # Stores the `children` on the instance so other methods can reuse it.
            self.children = dict()

        if child.name in self.children:
            emsg = f"Child {child.name} already exists in node {self.name}."
            logger.error(emsg)
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(emsg)
        else:
            self.children[child.name] = child

    # def insert_node function and point it to the add_child function
    def insert_node(self, node: 'Node') -> None:
        """
        Insert a node to the current node. This function is an alias
        of the add_child function.

        Args:
            node (Node): The node to be inserted.
        """
        # Calls `self.add_child` to perform the operation required at this stage.
        self.add_child(node)

    def _search_node_by_name(self, name: str) -> Union['Node', None]:
        """
        Search for the node with the given name.
        This function assumes that the node name is unique.

        Args:
            name (str): The name of the node to be searched.

        Returns:
            node (Node): The node with the given name if found, None else.
        """
        # check if the current node is the one we are looking for
        if self.name == name:
            return self
        # Continues the surrounding statement or collection definition.
        else:
            # search for the node in the children
            if self.children is not None:
                for cname, child in self.children.items():
                    node = child._search_node_by_name(name)
                    # Returns early when `node is not None` makes the remaining work inapplicable.
                    if node is not None:
                        return node

        # if the node is not found, return None
        return None

    def get_node_by_name(self, name: str) -> Union['Node', None]:
        """ Get the node with the given name.
        This function wraps the _search_node_by_name function and
        warns the user if the node is not found.

        Args:
            name (str): The name of the node to be searched.

        Returns:
            node (Node): The node with the given name if found, None else.
        """
        # Stores the `node` for use by the remaining steps in this flow.
        node = self._search_node_by_name(name)
        if node is None:
            emsg = f"Node {name} not found in the tree."
            logger.warning(emsg)
            # Returns `None` to the caller for the next stage.
            return None
        else:
            return node

    def _search_node_by_label(self, label: str) -> List['Node']:
        """
        Search for the node with the given label.
        The main difference between ::meth::`_search_node_by_name' and
        ::meth::`_search_node_by_label' is that the node name is usually
        the unique identifier of the node, while the node label is not.
        For a given label, there can be multiple nodes with the same label,
        e.g., all boundary elements have the same label, Elements_t.

        Args:
            label (str): The label of the node to be searched.

        Returns:
            nodes (List[Node]): A list of nodes with the given label, or
                an empty list if no node is found.
        """
        # for a given label, there can be multiple nodes, so we return a list
        # Stores the `nodes` for use by the remaining steps in this flow.
        nodes = list()
        # check if the current node is the one we are looking for
        if self.label == label:
            nodes.append(self)  # do not return yet, search for other nodes
        else:
            # search for the node in the children
            # Checks `self.children is not None` before entering this conditional branch.
            if self.children is not None:
                for cname, child in self.children.items():
                    nodes.extend(child._search_node_by_label(label))

        return nodes

    # Get the node with the given label.
    def get_node_by_label(self, label: str, sort: Optional[bool] = False)\
            -> Union[List['Node'], None]:
        """
        Get the node with the given label.
        The node label is not unique, this function is a wrapper of the
        ::meth::`_search_node_by_label' function, which returns a list of
        nodes with the given label.

        Args:
            label (str): The label of the node to be searched.
            sort (bool, optional): Whether to sort the nodes by their names.

        Returns:
            nodes (List[Node]): A list of nodes with the given label.
        """
        nodes = self._search_node_by_label(label)
        if len(nodes) == 0:
            emsg = f"Node with label {label} not found in the tree."
            # Records the current execution state to make progress and failures traceable.
            logger.warning(emsg)
            return None
        else:
            if sort:
                # Calls `nodes.sort` to perform the operation required at this stage.
                nodes.sort(key=lambda x: x.name)
            return nodes

    def _delete_node_by_name(self, name: str) -> Union[str, None]:
        """
        Delete a node from the tree by name, recursively.

        Args:
            name (str): The name of the node to be deleted.

        Returns:
            Union[str, None]: The name of the deleted node, or None if the
                node is not found.
        """
        # search for the node in the children
        if self.children is not None:
            # Visits each item in `self.children.items()` and applies the current processing step.
            for cname, child in self.children.items():
                if cname == name:
                    del self.children[cname]
                    return cname
                # Continues the surrounding statement or collection definition.
                else:
                    node_name = child.delete_node_by_name(name)
                    if node_name is not None:
                        return node_name
        # Continues the surrounding statement or collection definition.
        else:
            return None

    def delete_node_by_name(self, name: str) -> Union[str, None]:
        """
        Delete a node from the tree by name.

        Args:
            name (str): The name of the node to be deleted.

        Returns:
            Union[str, None]: The name of the deleted node, or None if the
                node is not found.
        """
        # check if we are doing self-deletion
        if self.name == name:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = f"We are currently at node {name}, cannot delete it."\
                " Only its children can be deleted."
            logger.warning(emsg)
            return None
        # delete the node from its children tree
        node_name = self._delete_node_by_name(name)
        # Checks `node_name is None` before entering this conditional branch.
        if node_name is None:
            emsg = f"Try to delete Node {name}, but not found in the tree."
            logger.warning(emsg)
            return None
        # Continues the surrounding statement or collection definition.
        else:
            logger.info(f"Node {name} deleted from the tree.")
            return node_name

    # delete nodes from the tree by label, recursively
    def _delete_node_by_label(self, label: str) -> List[str]:
        """
        Delete nodes from the tree by label, recursively.

        Args:
            label (str): The label of the node to be deleted.

        Returns:
            List[str]: The names of the deleted nodes, or an empty list if
                the node is not found.
        """
        # given a label, there can be multiple nodes
        # Stores the `node_names` for use by the remaining steps in this flow.
        node_names = list()
        # search for the node in the children
        if self.children is not None:
            for cname, child in self.children.items():
                if child.label == label:
                    # Handles the data or control-flow case represented by this statement.
                    del self.children[cname]
                    node_names.append(cname)
                else:
                    node_names.extend(child.delete_node_by_label(label))

        # Returns `node_names` to the caller for the next stage.
        return node_names

    def delete_node_by_label(self, label: str) -> Union[List[str], None]:
        """
        Delete nodes from the tree by label.

        Args:
            label (str): The label of the node to be deleted.

        Returns:
            Union[List[str], None]: The names of the deleted nodes, or None if
                the node is not found.
        """
        # check if we are doing self-deletion
        if self.label == label:
            emsg = f"The current root node has label {label}, "\
                "cannot delete it. Only its children can be deleted."
            # Records the current execution state to make progress and failures traceable.
            logger.warning(emsg)
            return None
        # delete the node from its children tree
        node_names = self._delete_node_by_label(label)
        if len(node_names) == 0:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = f"Try to delete Node with label {label}"\
                " but not found in the tree."
            logger.warning(emsg)
            return None
        else:
            # Records the current execution state to make progress and failures traceable.
            logger.info(f"Nodes {node_names} with label {label}"
                        " deleted from the tree.")
            return node_names

    def get_num_nodes(self) -> int:
        """
        Get the number of nodes in the tree.

        Returns:
            int: The number of nodes in the tree.
        """
        # Stores the `num_nodes` for use by the remaining steps in this flow.
        num_nodes = 1
        if self.children is not None:
            for cname, child in self.children.items():
                num_nodes += child.get_num_nodes()

        # Returns `num_nodes` to the caller for the next stage.
        return num_nodes

    def get_num_nodes_by_label(self, label: str) -> int:
        """
        Get the number of nodes with the given label in the tree.

        Args:
            label (str): The label of the node to be counted.

        Returns:
            int: The number of nodes with the given label in the tree.
        """
        num_nodes = 0
        if self.label == label:
            # Stores the `num_nodes` for use by the remaining steps in this flow.
            num_nodes += 1
        if self.children is not None:
            for cname, child in self.children.items():
                num_nodes += child.get_num_nodes_by_label(label)

        # Returns `num_nodes` to the caller for the next stage.
        return num_nodes

    def is_node_instance(self, node_type: str) -> bool:
        """
        Check if the current node is an instance of the given node type.

        Args:
            node_type (str): The node type to be checked.

        Returns:
            bool: True if the current node is an instance of the given node
                type, False otherwise.
        """

        return self.label.strip() == node_type.strip()

    # ---------------------------------------------------------------------- #
    # The following functions are used to extract the basic info of the      #
    # root node, including dimensional infos (physical dimensions, number    #
    # of nodes, cells, faces, etc.), zone type, time infos. Many of these    #
    # are aso useful for sanity checks.                                      #
    # ---------------------------------------------------------------------- #

    def get_cgns_version(self) -> Union[np.float32, None]:
        """
        Get the CGNS library version.
        Since version 3.4, a new array ElementStartOffset has been added to
        describe the Element_t structure for MIXED, NGON_n, NFACE_n element
        types. See
        `https://cgns.github.io/CGNS_docs_current/sids/changes.html`_ for
        more details.
        This function checks if the CGNS library version is 3.4 or
        higher. If not, a warning message will be printed.

        Returns:
            Union[np.float32, None]: The CGNS library version, or None if
                the version node is not found.

        Raises:
            NotImplementedError: If called by an invalid node.
            ValueError: If the CGNS library version node is not found.
            ValueError: If the CGNS library version is 3.4 or higher.
        """
        # Checks `` before entering this conditional branch.
        if self.is_node_instance(CGNSNodeType_S.Root) or\
           self.is_node_instance(CGNSNodeType_S.CGNSLibraryVersion):
            pass
        else:
            raise NotImplementedError

        # search for the version node based on the label
        # Stores the `version_node` for use by the remaining steps in this flow.
        version_node = self.get_node_by_label(
            CGNSNodeType_S.CGNSLibraryVersion)
        if version_node is None:
            emsg = "CGNSLibraryVersion_t node not found."
            # Records the current execution state to make progress and failures traceable.
            logger.error(emsg)
            raise ValueError(emsg)
        # get the version, by default a real4 number, i.e. np.float32
        version = version_node[0].data[0]
        # check if the version is 3.4 or higher
        if version >= 3.4:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = f"CGNS library version {version} is higher than 3.4."\
                " Currently we only support lower versions."
            logger.error(emsg)
            raise ValueError(emsg)
        else:
            # Records the current execution state to make progress and failures traceable.
            logger.debug(f"CGNS library version {version}.")

        return version

    def get_base_node(self) -> Union['Node', None]:
        """
        Get the CGNSBase_t node, which is the root node of the tree.

        Returns:
            Union['Node', None]: The CGNSBase_t node, or None if not found.

        Raises:
            NotImplementedError: If called by an invalid node.
            ValueError: If the CGNSBase_t node is not found.
            ValueError: If more than one CGNSBase_t node is found.
        """
        # search for the base node based on the label
        if self.is_node_instance(CGNSNodeType_S.Root) or\
           self.is_node_instance(CGNSNodeType_S.CGNSBase):
            # Handles the data or control-flow case represented by this statement.
            pass
        else:
            raise NotImplementedError

        base_node = self.get_node_by_label(CGNSNodeType_S.CGNSBase)
        # Checks `base_node is None` before entering this conditional branch.
        if base_node is None:
            emsg = "CGNSBase_t node not found."
            logger.error(emsg)
            raise ValueError(emsg)
        # Checks `len(base_node) > 1` before entering this conditional branch.
        if len(base_node) > 1:
            emsg = "More than one CGNSBase_t node found."
            logger.error(emsg)
            raise ValueError(emsg)

        # Returns `base_node[0]` to the caller for the next stage.
        return base_node[0]

    def get_dimension_info(self) -> Dict[str, int]:
        """
        Extracts the dimension information from the node CGNSBase_t.
        The dimension information includes the
        IndexDimension, CellDimension, PhysicalDimension.
            IndexDimension: the dimension of the index space,
                which is tied to the mesh type, structured or unstructured.
            CellDimension: the dimension of the cell space, which is tied
                to the cell dimension, e.g., volume, surface, line, point.
            PhysicalDimension: the dimension of the physical space,
                which is tied to the physical dimension, e.g., 3D, 2D, 1D, 0D.
        Only CellDimension and the PhysicalDimension can be directly extracted
        from the CGNSBase_t node. The IndexDimension is determined by the
        mesh type, structured or unstructured.

        Returns:
            Dict[str, int]: The dimension information.
        """
        # find the base node
        base_node = self.get_base_node()
        # get the dimension information
        dim_info = {}
        # Stores the current data for use by the remaining steps in this flow.
        data = base_node.data
        dim_info["CellDimension"] = data[0]
        dim_info["PhysicalDimension"] = data[1]

        return dim_info

    # Check if the simulation is steady state or not, accessed from the
    def check_steady_state(self) -> bool:
        """
        Check if the simulation is steady state or not, accessed from the
        BaseIterativeData_t node. See
        `https://cgns.github.io/CGNS_docs_current/sids/timedep.html#BaseIterativeData`_
        for more details.

        Returns:
            bool: True if the simulation is steady state, False otherwise.
        """

        # find the base node
        base_node = self.get_base_node()
        # find the BaseIterativeData_t node
        iter_list = base_node.get_node_by_label("BaseIterativeData_t")[0]
        # get the data
        data = iter_list.data
        # check if the simulation is steady state
        # Stores the `steady` for use by the remaining steps in this flow.
        steady = (data[0] == 1)

        return steady

    def ensure_single_zone(self) -> None:
        """
        Ensure the CGNS file contains only a single zone. If not,
        we only keep one zone with the name 'fluid',
        and remove all other zones.
        """

        if self.is_node_instance(CGNSNodeType_S.Root) or\
           self.is_node_instance(CGNSNodeType_S.CGNSBase):
            # Handles the data or control-flow case represented by this statement.
            pass
        else:
            raise NotImplementedError

        # get the base node first
        base_node = self.get_base_node()
        # get the zone list
        # Stores the `zone_list` for use by the remaining steps in this flow.
        zone_list = self.get_node_by_label(CGNSNodeType_S.Zone)

        # check if the CGNS file contains only a single zone
        if len(zone_list) > 1:
            for zone in zone_list:
                if zone.name != "fluid":
                    # Calls `base_node.children.pop` to perform the operation required at this stage.
                    base_node.children.pop(zone.name)

        return None

    def get_zone_node(self) -> 'Node':
        """
        Get the Zone_t node, which is the root node of the zone.
        This function also checks if the CGNS file contains only a single
        zone. Currently we only support a single zone.

        Returns:
            Union['Node', None]: The Zone_t node, or None if not found.

        Raises:
            NotImplementedError: If called by an invalid node.
            ValueError: If the CGNS file contains no zone or more than
                one zone.
        """

        # find the base node
        if self.is_node_instance(CGNSNodeType_S.Root) or\
           self.is_node_instance(CGNSNodeType_S.CGNSBase) or\
           self.is_node_instance(CGNSNodeType_S.Zone):
            # Handles the data or control-flow case represented by this statement.
            pass
        else:
            raise NotImplementedError

        # find the Zone_t node
        zone_list = self.get_node_by_label(CGNSNodeType_S.Zone)
        # check if the CGNS file contains only a single zone
        # Checks `zone_list is None` before entering this conditional branch.
        if zone_list is None:
            emsg = "Zone_t node not found."
            logger.error(emsg)
            raise ValueError(emsg)
        # Checks `len(zone_list) > 1` before entering this conditional branch.
        if len(zone_list) > 1:
            emsg = f"The CGNS file contains {len(zone_list)} zones."\
                " Currently we only support a single zone."
            logger.error(emsg)
            raise NotImplementedError(emsg)

        # Returns `zone_list[0]` to the caller for the next stage.
        return zone_list[0]

    def get_zone_nodes(self) -> 'Node':
        """
        Get the Zone_t node, which is the root node of the zone.
        This function also checks if the CGNS file contains only a single
        zone. Currently we only support a single zone.

        Returns:
            Union['Node', None]: The Zone_t node, or None if not found.

        Raises:
            NotImplementedError: If called by an invalid node.
            ValueError: If the CGNS file contains no zone or more than
                one zone.
        """

        # find the base node
        if self.is_node_instance(CGNSNodeType_S.Root) or\
           self.is_node_instance(CGNSNodeType_S.CGNSBase) or\
           self.is_node_instance(CGNSNodeType_S.Zone):
            pass
        # Continues the surrounding statement or collection definition.
        else:
            raise NotImplementedError

        # find the Zone_t node
        zone_list = self.get_node_by_label(CGNSNodeType_S.Zone)
        # check if the CGNS file contains only a single zone
        if zone_list is None:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = "Zone_t node not found."
            logger.error(emsg)
            raise ValueError(emsg)

        return zone_list

    # Extracts the zone type from the node ZoneType_t.
    def get_zone_type(self) -> str:
        """
        Extracts the zone type from the node ZoneType_t.
        The zone type is either structured or unstructured.
        Currently we only support a single unstructured zone.

        Returns:
            str: The zone type.

        Raises:
            NotImplementedError: If called by an invalid node.
        """

        if not self.is_node_instance(CGNSNodeType_S.Zone):
            raise NotImplementedError

        zone_node = self.get_zone_node()
        # find the ZoneType_t node
        # Selects the required item from `zone_node.get_node_by_label("ZoneType_t")` as the `zone_type_node`.
        zone_type_node = zone_node.get_node_by_label("ZoneType_t")[0]
        # get the zone type
        zone_type = zone_type_node.data.decode().rstrip("\x00")
        return zone_type

    def check_zone_type(self) -> bool:
        """
        Check if the zone is structured or unstructured. Currently we only
        support a single unstructured zone.

        Returns:
            bool: True if the CGNS file contains only one unstructured zone,
                False otherwise.
        """
        # Stores the `supported` for use by the remaining steps in this flow.
        supported = self.get_zone_type() == "Unstructured"

        return supported

    def pre_check(self) -> bool:
        """
        Sanity check if the CGNS file is supported by the current
        implementation.

        Raises:
            NotImplementedError: If the solution is unsteady.
            NotImplementedError: If the zone is structured.

        Returns:
            bool: True if the file is supported, False otherwise.
        """
        # get the CGNS version
        if not self.is_node_instance(CGNSNodeType_S.Root):
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError

        cgns_version = self.get_cgns_version()
        # check if the solution is steady
        steady = self.get_base_node().check_steady_state()
        if not steady:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = "Unsteady solution is not supported."
            logger.error(emsg)
            raise NotImplementedError(emsg)
        # check if the zone is structured or unstructured
        unstructured = self.get_zone_node().check_zone_type()
        # Checks `not unstructured` before entering this conditional branch.
        if not unstructured:
            emsg = "Structured zone is not supported."
            logger.error(emsg)
            raise NotImplementedError(emsg)

        # Returns `steady and unstructured` to the caller for the next stage.
        return steady and unstructured

    # ----------------------------------------------------------------------- #
    # The following functions are used to extract the data from the Zone_t    #
    # node. The Zone_t node is the root node of the zone. The zone            #
    # information includes the zone type, the zone name, the volume mesh,     #
    # the surface mesh, the boundary conditions, and the solution.            #
    # ----------------------------------------------------------------------- #

    # ============================ #
    # 0. Zone information section  #
    # ============================ #

    def get_zone_data(self) -> Dict[str, np.int64]:
        """
        This function extracts the data from the zone node.
        The direct data of the zone node includes the VertexSize, the
        CellSize, and the VertexSizeBoundary.
        VertexSize is the number of vertices defining "the grid" or
        the domain (i.e., without rind points);
        CellSize is the number of cells on the interior of the domain.
        VertexSizeBoundary is the number of vertices defining the boundary,
        when the mesh nodes are sorted, otherwise it is defaulted to 0.
        See `https://cgns.github.io/CGNS_docs_current/sids/cgnsbase.html`_
        for more details.

        Based on our practice, the cell size is sometimes ambiguous.
        For example, Fluent writes the cell size as the number of
        volume cells plus the number of surface cells. However, the
        StarCCM+ writes the cell size as the number of volume cells
        only. Moreover, users should be extremely careful when the
        StarCMM+ uses mixed type mesh. Possible outcome is Star
        only records the number of interior cells, whose indices are,
        however, not the first n cells. In this case, the solution
        node is agnostic to the cell indices. Thus the solution might
        be falsely associated to the wrong cells, e.g., some face cells
        that are indexed as the first n cells. We recommend users to
        use the polyhedral mesh in StarCCM+ to avoid this issue.

        Returns:
            Dict[str, np.int64]: The zone data.

        Raises:
            NotImplementedError: If called by an invalid node.
            NotImplementedError: If the zone is structured. Or if the zone
                data is not in the correct format.
        """

        if not self.is_node_instance(CGNSNodeType_S.Zone):
            raise NotImplementedError

        # get the zone data
        # Stores the current data for use by the remaining steps in this flow.
        data = self.data
        # In general, the data attribute of the Zone_t node is a 2D array,
        # with the first dimension being the IndexDimension, which is 1
        # for unstructured mesh and CellDimension for structured mesh.
        # The second dimension records the VertexSize, CellSize, and
        # VertexSizeBoundary. The data is stored in the Fortran order
        # in the original hdf5 format. However, the data is transposed
        # when read in, see io module for details.

        # Different CGNS writers may expose the unstructured size triplet as
        # either a row vector [[VertexSize, CellSize, VertexSizeBoundary]]
        # or a column vector [[VertexSize], [CellSize], [VertexSizeBoundary]].
        if data.shape == (1, 3):
            normalized = data[0]
        elif data.shape == (3, 1):
            # Selects the required item from `data` as the `normalized`.
            normalized = data[:, 0]
        else:
            emsg = f"Only general unstructured mesh is supported."
            logger.error(emsg)
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError(emsg)

        zone_info = {}
        zone_info["VertexSize"] = normalized[0]
        zone_info["CellSize"] = normalized[1]
        # Selects the required item from `normalized` as the `zone_info['VertexSizeBoundary']`.
        zone_info["VertexSizeBoundary"] = normalized[2]

        return zone_info

    # ================ #
    # 1. Mesh section  #
    # ================ #

    def get_mesh_size(self) -> Dict[str, np.int64]:
        """
        This function is an alias of get_zone_data() for convenience.

        Returns:
            Dict[str, np.int64]: The zone data.
        """

        return self.get_zone_data()

    # Get the node GridCoordinates_t, which contains the coordinates data.
    def get_coords_node(self) -> Union['Node', None]:
        """
        Get the node GridCoordinates_t, which contains the coordinates data.
        Ideally, this function should only be called on a supported node,
        which has passed the check_base_all() function. We do not check
        the node here for simplicity.

        Returns:
            Union['Node', None]: The GridCoordinates_t node, or None if not
                found.

        Raises:
            NotImplementedError: if called by an invalid node.
            ValueError: if the GridCoordinates_t node is not found.
            NotImplementedError: if the more than one GridCoordinates_t
                node is found.
        """

        if not self.is_node_instance(CGNSNodeType_S.Zone):
            raise NotImplementedError

        # find the GridCoordinates_t node
        mesh_node_list = self.get_node_by_label(CGNSNodeType_S.GridCoordinates)

        # Checks `mesh_node_list is None` before entering this conditional branch.
        if mesh_node_list is None:
            emsg = "GridCoordinates_t node not found."
            logger.error(emsg)
            raise ValueError(emsg)

        # Checks `len(mesh_node_list) > 1` before entering this conditional branch.
        if len(mesh_node_list) > 1:
            emsg = f"The current Zone {self.name} contains"\
                " {len(mesh_node_list)} mesh nodes."\
                " Currently we only support a single mesh node."
            logger.error(emsg)
            raise NotImplementedError(emsg)

        # Selects the required item from `mesh_node_list` as the `mesh_node`.
        mesh_node = mesh_node_list[0]

        return mesh_node

    def get_mesh_coords(self) -> np.ndarray:
        """
        Get the mesh coordinates as a numpy array.

        Returns:
            np.ndarray: The mesh coordinates.

        Shapes:
            - **output** (*np.ndarray*): (n, 3), where n is the number of
                vertices.
        """

        if self.is_node_instance(CGNSNodeType_S.Zone):
            # Stores the `node` for use by the remaining steps in this flow.
            node = self.get_coords_node()
        elif self.is_node_instance(CGNSNodeType_S.GridCoordinates):
            node = self
        else:
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError

        # extract the x, y, z coordinates
        x = node.children["CoordinateX"].data
        y = node.children["CoordinateY"].data
        z_node = node.children.get("CoordinateZ")
        # Checks `z_node is None` before entering this conditional branch.
        if z_node is None:
            z = np.zeros_like(np.asarray(x, dtype=np.float64))
        else:
            z = z_node.data

        # combine the coordinates into a single array
        # Stores the `coords` for use by the remaining steps in this flow.
        coords = np.stack((x, y, z), axis=1)

        return coords

    # ======================== #
    # 2. Connectivity section  #
    # ======================== #

    def get_mesh_connectivity_mixed(self) -> np.ndarray:
        """
        Extracts the mesh connectivity from the mixed mesh, which is
        stored in the Elements_t node with element type MIXED.

        Returns:
            np.ndarray: The mesh connectivity.
        """
        if not self.is_node_instance(CGNSNodeType_S.Zone):
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError

        entity_list = self.get_node_by_label(CGNSNodeType_S.Elements) or []
        mixed_sections = [
            section
            # Continues the surrounding statement or collection definition.
            for section in entity_list
            if int(section.data[0]) == cgnskeywords.ElementType["MIXED"]
        ]
        if not mixed_sections:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = "No MIXED mesh section found."
            logger.error(emsg)
            raise ValueError(emsg)

        return self._build_standard_mesh_connectivity(mixed_sections)

    # Return the number of connectivity entries for a fixed element.
    def _lookup_element_npe(self, elem_type: int) -> int:
        """Return the number of connectivity entries for a fixed element."""
        if elem_type < 0 or elem_type >= len(cgnskeywords.ElementTypeNPE_l):
            raise ValueError(f"Unsupported element type id {elem_type}")

        npe = int(cgnskeywords.ElementTypeNPE_l[elem_type])
        # Checks `npe <= 0` before entering this conditional branch.
        if npe <= 0:
            elem_name = cgnskeywords.ElementType_l[elem_type]
            raise ValueError(f"Element type {elem_name} does not have a fixed positive node count")
        return npe

    # Infer the topological dimension of a concrete CGNS element type.
    def _get_element_dimension(self, elem_type: int) -> int:
        """Infer the topological dimension of a concrete CGNS element type."""
        if elem_type == cgnskeywords.ElementType["MIXED"]:
            raise ValueError("MIXED section dimension must be inferred from its contained elements")
        if elem_type == cgnskeywords.ElementType["NFACE_n"]:
            # Returns `3` to the caller for the next stage.
            return 3
        if elem_type == cgnskeywords.ElementType["NGON_n"]:
            return 2
        if elem_type in cgnskeywords.ElementType0D:
            # Returns `0` to the caller for the next stage.
            return 0
        if elem_type in cgnskeywords.ElementType1D:
            return 1
        if elem_type in cgnskeywords.ElementType_tri or elem_type in cgnskeywords.ElementType_quad:
            # Returns `2` to the caller for the next stage.
            return 2
        if (
            elem_type in cgnskeywords.ElementType_tetra
            or elem_type in cgnskeywords.ElementType_pyra
            # Continues the surrounding statement or collection definition.
            or elem_type in cgnskeywords.ElementType_penta
            or elem_type in cgnskeywords.ElementType_hexa
        ):
            return 3

        # Stores the `elem_name` for use by the remaining steps in this flow.
        elem_name = cgnskeywords.ElementType_l[elem_type] if 0 <= elem_type < len(cgnskeywords.ElementType_l) else elem_type
        raise ValueError(f"Unsupported element dimension inference for {elem_name}")

    def _infer_section_dimension(self, section_node: 'Node') -> int:
        """Infer the effective topological dimension of an Elements_t section."""
        elem_type = int(section_node.data[0])
        # Returns early when `elem_type != cgnskeywords.ElementType["MIXED"]` makes the remaining work inapplicable.
        if elem_type != cgnskeywords.ElementType["MIXED"]:
            return self._get_element_dimension(elem_type)

        _, element_type_ids = self._parse_mixed_section_connectivity(section_node)
        if len(element_type_ids) == 0:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"MIXED:{section_node.name} does not contain any element payload")
        return max(self._get_element_dimension(int(type_id)) for type_id in element_type_ids)

    @staticmethod
    def _get_section_start_id(section_node: 'Node') -> int:
        """Return the first CGNS element id of a section for stable ordering."""
        # Stores the `range_node` for use by the remaining steps in this flow.
        range_node = section_node.children.get("ElementRange") if getattr(section_node, "children", None) else None
        if range_node is None or range_node.data is None:
            return sys.maxsize
        data = np.asarray(range_node.data).reshape(-1)
        # Returns `int(data[0]) if data.size > 0 else sys.maxsize` to the caller for the next stage.
        return int(data[0]) if data.size > 0 else sys.maxsize

    @staticmethod
    def _get_section_expected_count(section_node: 'Node') -> Optional[int]:
        """Return the number of elements declared by ElementRange if present."""
        range_node = section_node.children.get("ElementRange") if getattr(section_node, "children", None) else None
        # Returns early when `range_node is None or range_node.data is None` makes the remaining work inapplicable.
        if range_node is None or range_node.data is None:
            return None
        data = np.asarray(range_node.data).reshape(-1)
        if data.size < 2:
            # Returns `None` to the caller for the next stage.
            return None
        start_id = int(data[0])
        end_id = int(data[1])
        if end_id < start_id:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"Invalid ElementRange [{start_id}, {end_id}] in section {section_node.name}")
        return end_id - start_id + 1

    @staticmethod
    def _finalize_elem2node_payload(
        # Continues the surrounding statement or collection definition.
        connectivity_rows: List[np.ndarray],
        element_type_ids: List[int],
        mesh_format: str,
    ) -> Dict[str, object]:
        """Normalize parsed connectivity into the shared payload schema."""
        # Stores the `connectivity` for use by the remaining steps in this flow.
        connectivity = np.empty(len(connectivity_rows), dtype=object)
        for idx, row in enumerate(connectivity_rows):
            connectivity[idx] = np.asarray(row, dtype=np.int64).reshape(-1)

        return {
            # Completes the multi-line `collection` arguments or elements.
            "Elem2Node": {
                "offset": None,
                "connectivity": connectivity,
            },
            # Completes the multi-line `np.asarray` arguments or elements.
            "ElementTypeIds": np.asarray(element_type_ids, dtype=np.int32),
            "MeshFormat": mesh_format,
        }

    def _parse_fixed_section_connectivity(
        # Continues the surrounding statement or collection definition.
        self,
        section_node: 'Node',
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Parse a fixed-type Elements_t section into per-cell node lists."""
        elem_type = int(section_node.data[0])
        # Stores the `npe` for use by the remaining steps in this flow.
        npe = self._lookup_element_npe(elem_type)
        conn = np.asarray(section_node.children["ElementConnectivity"].data, dtype=np.int64).reshape(-1)
        expected_count = self._get_section_expected_count(section_node)

        rows: List[np.ndarray] = []
        # Reads `ElementStartOffset` from the mapping into the `start_offset_node`.
        start_offset_node = section_node.children.get("ElementStartOffset")
        if start_offset_node is not None and start_offset_node.data is not None:
            offsets = self._normalize_element_start_offset(
                start_offset_node.data,
                # Completes the multi-line `len` arguments or elements.
                len(conn),
                f"FIXED:{section_node.name}",
            )
            for i in range(offsets.size - 1):
                # Converts the `s` to `int` so later checks use a consistent type.
                s = int(offsets[i])
                e = int(offsets[i + 1])
                if e - s != npe:
                    raise ValueError(
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"FIXED:{section_node.name}: element {i} expects {npe} nodes, got {e - s}"
                    )
                rows.append(conn[s:e].copy())
        else:
            # Validates `len(conn) % npe != 0` and raises `ValueError` when the input cannot be processed safely.
            if len(conn) % npe != 0:
                raise ValueError(
                    f"FIXED:{section_node.name}: connectivity size {len(conn)} is not divisible by npe {npe}"
                )
            # Visits each item in `range(0, len(conn), npe)` and applies the current processing step.
            for start_index in range(0, len(conn), npe):
                rows.append(conn[start_index:start_index + npe].copy())

        if expected_count is not None and expected_count != len(rows):
            logger.warning(
                # Completes the multi-line `logger.warning` arguments or elements.
                "FIXED section %s ElementRange count (%s) != parsed element count (%s), using parsed count",
                section_node.name,
                expected_count,
                len(rows),
            # Completes the multi-line `logger.warning` arguments or elements.
            )

        return rows, [elem_type] * len(rows)

    def _parse_mixed_section_connectivity(
        self,
        # Continues the surrounding statement or collection definition.
        section_node: 'Node',
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Parse a MIXED Elements_t section into per-cell node lists."""
        conn = np.asarray(section_node.children["ElementConnectivity"].data, dtype=np.int64).reshape(-1)
        expected_count = self._get_section_expected_count(section_node)
        # Initializes the `rows` as a collection for values from this processing stage.
        rows: List[np.ndarray] = []
        element_type_ids: List[int] = []

        start_offset_node = section_node.children.get("ElementStartOffset")
        if start_offset_node is not None and start_offset_node.data is not None:
            # Stores the `offsets` for use by the remaining steps in this flow.
            offsets = self._normalize_element_start_offset(
                start_offset_node.data,
                len(conn),
                f"MIXED:{section_node.name}",
            # Completes the multi-line `self._normalize_element_start_offset` arguments or elements.
            )
            for i in range(offsets.size - 1):
                s = int(offsets[i])
                e = int(offsets[i + 1])
                # Validates `e - s < 2` and raises `ValueError` when the input cannot be processed safely.
                if e - s < 2:
                    raise ValueError(f"MIXED:{section_node.name}: invalid element payload length {e - s}")
                elem_type = int(conn[s])
                npe = self._lookup_element_npe(elem_type)
                # Validates `e - s != npe + 1` and raises `ValueError` when the input cannot be processed safely.
                if e - s != npe + 1:
                    raise ValueError(
                        f"MIXED:{section_node.name}: element {i} expects payload {npe + 1}, got {e - s}"
                    )
                # Adds the current value to `rows` for later aggregation.
                rows.append(conn[s + 1:e].copy())
                element_type_ids.append(elem_type)
        else:
            start_index = 0
            # Repeats the current step while `start_index < len(conn)` remains true.
            while start_index < len(conn):
                elem_type = int(conn[start_index])
                npe = self._lookup_element_npe(elem_type)
                end_index = start_index + npe + 1
                # Validates `end_index > len(conn)` and raises `ValueError` when the input cannot be processed safely.
                if end_index > len(conn):
                    raise ValueError(
                        f"MIXED:{section_node.name}: connectivity overflow at [{start_index}, {end_index})"
                    )
                # Adds the current value to `rows` for later aggregation.
                rows.append(conn[start_index + 1:end_index].copy())
                element_type_ids.append(elem_type)
                start_index = end_index

        if expected_count is not None and expected_count != len(rows):
            # Records the current execution state to make progress and failures traceable.
            logger.warning(
                "MIXED section %s ElementRange count (%s) != parsed element count (%s), using parsed count",
                section_node.name,
                expected_count,
                # Completes the multi-line `len` arguments or elements.
                len(rows),
            )

        return rows, element_type_ids

    def _build_standard_mesh_connectivity(
        # Continues the surrounding statement or collection definition.
        self,
        sections: List['Node'],
    ) -> Dict[str, object]:
        """Build Elem2Node payload from fixed and/or MIXED volume sections."""
        ordered_sections = sorted(sections, key=self._get_section_start_id)
        # Initializes the `connectivity_rows` as a collection for values from this processing stage.
        connectivity_rows: List[np.ndarray] = []
        element_type_ids: List[int] = []
        mesh_format = "fixed"
        parsed_sections = []

        # Visits each item in `ordered_sections` and applies the current processing step.
        for section in ordered_sections:
            elem_type = int(section.data[0])
            if elem_type == cgnskeywords.ElementType["MIXED"]:
                mesh_format = "mixed"
                # Stores the `(rows, type_ids)` for use by the remaining steps in this flow.
                rows, type_ids = self._parse_mixed_section_connectivity(section)
            else:
                rows, type_ids = self._parse_fixed_section_connectivity(section)

            parsed_sections.append((rows, type_ids))

        # Stores the `cell_dimension` for use by the remaining steps in this flow.
        cell_dimension = max(
            self._get_element_dimension(int(type_id))
            for _, type_ids in parsed_sections
            for type_id in type_ids
        # Completes the multi-line `max` arguments or elements.
        )
        for rows, type_ids in parsed_sections:
            for row, type_id in zip(rows, type_ids):
                if self._get_element_dimension(int(type_id)) != cell_dimension:
                    # Handles the data or control-flow case represented by this statement.
                    continue
                connectivity_rows.append(row)
                element_type_ids.append(type_id)

        expected_cell_count = int(self.get_mesh_size()["CellSize"])
        # Checks `expected_cell_count != len(connectivity_rows)` before entering this conditional branch.
        if expected_cell_count != len(connectivity_rows):
            logger.warning(
                "The number of parsed volume cells %s does not match Zone CellSize %s.",
                len(connectivity_rows),
                # Completes the multi-line `logger.warning` arguments or elements.
                expected_cell_count,
            )

        return self._finalize_elem2node_payload(
            connectivity_rows=connectivity_rows,
            # Completes the multi-line `self._finalize_elem2node_payload` arguments or elements.
            element_type_ids=element_type_ids,
            mesh_format=mesh_format,
        )

    def _select_interior_element_sections(
        # Continues the surrounding statement or collection definition.
        self,
        entity_list: List['Node'],
    ) -> List['Node']:
        """筛选当前 zone 中与体/面维度一致的内部单元节。"""
        cell_dimension = max(self._infer_section_dimension(section) for section in entity_list)
        # Initializes the `interior_sections` as a collection for values from this processing stage.
        interior_sections = []

        for section in entity_list:
            elem_type = int(section.data[0])
            if elem_type == cgnskeywords.ElementType["NGON_n"]:
                # Handles the data or control-flow case represented by this statement.
                continue
            if elem_type == cgnskeywords.ElementType["MIXED"]:
                interior_sections.append(section)
                continue

            # Stores the `elem_dimension` for use by the remaining steps in this flow.
            elem_dimension = self._get_element_dimension(elem_type)
            if elem_dimension == cell_dimension:
                interior_sections.append(section)

        return interior_sections

    # 从 fixed / mixed 单元节提取网格边。
    def _extract_edges_from_standard_sections(
        self,
        sections: List['Node'],
        bidirectional: bool = True,
    # Continues the surrounding statement or collection definition.
    ) -> set:
        """从 fixed / mixed 单元节提取网格边。"""
        ordered_sections = sorted(sections, key=self._get_section_start_id)
        edges = set()
        edge_func = cgnsmesh.get_edges_bidirectional if bidirectional else cgnsmesh.get_edges

        # Visits each item in `ordered_sections` and applies the current processing step.
        for section in ordered_sections:
            elem_type = int(section.data[0])
            if elem_type == cgnskeywords.ElementType["MIXED"]:
                rows, element_type_ids = self._parse_mixed_section_connectivity(section)
            # Continues the surrounding statement or collection definition.
            else:
                rows, element_type_ids = self._parse_fixed_section_connectivity(section)

            for row, row_type in zip(rows, element_type_ids):
                edges.update(edge_func(int(row_type), row))

        # Returns `edges` to the caller for the next stage.
        return edges

    @staticmethod
    def _face_id_to_local_idx(face_id: int,
                              face_ranges: List[Tuple[int, int, int]]) -> int:
        """Map absolute CGNS face id to local row index in face2node offset."""
        # Visits each item in `face_ranges` and applies the current processing step.
        for start_id, end_id, local_start in face_ranges:
            if start_id <= face_id <= end_id:
                return local_start + (face_id - start_id)

        raise KeyError(f"Face id {face_id} is out of supported NGON ranges: {face_ranges}")

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _normalize_element_start_offset(raw_offset: np.ndarray,
                                        conn_size: int,
                                        section_name: str) -> np.ndarray:
        """Normalize ElementStartOffset to zero-based indexing."""
        # Stores the `offset` for use by the remaining steps in this flow.
        offset = np.asarray(raw_offset, dtype=np.int64).reshape(-1)
        if offset.size < 2:
            raise ValueError(f"{section_name}: ElementStartOffset size must be >= 2")

        if offset[0] == 0:
            # Stores the `normalized` for use by the remaining steps in this flow.
            normalized = offset
        elif offset[0] == 1:
            normalized = offset - 1
        else:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"{section_name}: unsupported ElementStartOffset start value {offset[0]}"
            )

        if np.any(np.diff(normalized) < 0):
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(f"{section_name}: ElementStartOffset is not monotonic")
        if normalized[0] != 0:
            raise ValueError(f"{section_name}: normalized ElementStartOffset must start from 0")
        if normalized[-1] > conn_size:
            # Raises `ValueError` to report why this path cannot continue.
            raise ValueError(
                f"{section_name}: ElementStartOffset tail {normalized[-1]} exceeds connectivity size {conn_size}"
            )
        if normalized[-1] < conn_size:
            # Records the current execution state to make progress and failures traceable.
            logger.warning(
                "%s: connectivity has trailing entries (%s > %s)",
                section_name,
                conn_size,
                # Completes the multi-line `logger.warning` arguments or elements.
                normalized[-1],
            )

        return normalized

    def _parse_section_slices(self, section_node: 'Node', section_kind: str) -> List[Tuple[int, int]]:
        """Parse variable-length connectivity slices from an Elements_t section.

        Returns zero-based [start, end) slices over ElementConnectivity.
        """
        # Stores the `conn` for use by the remaining steps in this flow.
        conn = section_node.children["ElementConnectivity"].data
        section_name = f"{section_kind}:{section_node.name}"

        start_offset_node = section_node.children.get("ElementStartOffset")
        if start_offset_node is not None and start_offset_node.data is not None:
            # Stores the `offsets` for use by the remaining steps in this flow.
            offsets = self._normalize_element_start_offset(
                start_offset_node.data,
                len(conn),
                section_name,
            # Completes the multi-line `self._normalize_element_start_offset` arguments or elements.
            )
            slices = []
            for i in range(offsets.size - 1):
                s = int(offsets[i])
                # Converts the `e` to `int` so later checks use a consistent type.
                e = int(offsets[i + 1])
                if e <= s:
                    raise ValueError(f"{section_name}: invalid slice [{s}, {e})")
                slices.append((s, e))
            # Returns `slices` to the caller for the next stage.
            return slices

        # Legacy fallback: packed as [n, values...]
        slices = []
        start_index = 0
        while start_index < len(conn):
            # Converts the `nitem` to `int` so later checks use a consistent type.
            nitem = int(conn[start_index])
            if nitem <= 0:
                raise ValueError(f"{section_name}: invalid packed count {nitem}")
            s = start_index + 1
            # Derives the `e` from the current values for downstream processing.
            e = s + nitem
            if e > len(conn):
                raise ValueError(
                    f"{section_name}: packed connectivity overflow at [{s}, {e}) with size {len(conn)}"
                # Completes the multi-line `ValueError` arguments or elements.
                )
            slices.append((s, e))
            start_index = e

        return slices

    # Build face2node from NGON elements without assuming face ids start at 1.
    def _build_face2node_poly(self, boundary_elements: List['Node']) -> Dict[str, object]:
        """Build face2node from NGON elements without assuming face ids start at 1."""
        boundary_elements.sort(key=lambda x: x.children["ElementRange"].data[0])

        bc_connectivity = np.concatenate(
            [e.children["ElementConnectivity"].data for e in boundary_elements]
        # Completes the multi-line `np.concatenate` arguments or elements.
        )

        face_ranges: List[Tuple[int, int, int]] = []
        section_slices = []
        total_faces = 0
        # Stores the `conn_base` for use by the remaining steps in this flow.
        conn_base = 0
        for elem in boundary_elements:
            conn = elem.children["ElementConnectivity"].data
            slices = self._parse_section_slices(elem, section_kind="NGON")
            # Adds the current value to `section_slices` for later aggregation.
            section_slices.append((conn_base, slices))

            start_id, end_id = elem.children["ElementRange"].data
            start_id = int(start_id)
            end_id = int(end_id)
            # Validates `end_id < start_id` and raises `ValueError` when the input cannot be processed safely.
            if end_id < start_id:
                raise ValueError(f"Invalid NGON ElementRange: [{start_id}, {end_id}]")

            nface_section = len(slices)
            expected = end_id - start_id + 1
            # Checks `expected != nface_section` before entering this conditional branch.
            if expected != nface_section:
                logger.warning(
                    "NGON section %s ElementRange count (%s) != parsed face count (%s), using parsed count",
                    elem.name,
                    # Completes the multi-line `logger.warning` arguments or elements.
                    expected,
                    nface_section,
                )

            face_ranges.append((start_id, start_id + nface_section - 1, total_faces))
            # Stores the `total_faces` for use by the remaining steps in this flow.
            total_faces += nface_section
            conn_base += len(conn)

        # Offset layout uses a zero-based start index that can be passed
        # directly into NumPy slicing over the flattened connectivity array.
        # This works for both storage styles:
        #   1. packed legacy layout [n, values...], where the slice starts
        #      after the leading count;
        #   2. ElementStartOffset layout, where the slice starts at the
        #      offset value itself.
        #   offset[i, 1] -> number of nodes in this face
        offset = np.zeros((total_faces, 2), dtype=np.int64)

        local_face_idx = 0
        # Visits each item in `section_slices` and applies the current processing step.
        for conn_base, slices in section_slices:
            for s, e in slices:
                nnode = e - s
                offset[local_face_idx, 0] = conn_base + s
                # Stores the `offset[local_face_idx, 1]` for use by the remaining steps in this flow.
                offset[local_face_idx, 1] = nnode
                local_face_idx += 1

        if local_face_idx != total_faces:
            raise ValueError(f"NGON face assembly mismatch: {local_face_idx} != {total_faces}")

        # Returns `` to the caller for the next stage.
        return {
            "offset": offset,
            "connectivity": bc_connectivity,
            "face_ranges": face_ranges,
        # Completes the multi-line `collection` arguments or elements.
        }

    def _build_elem2face_poly(self, interior_elements: List['Node']) -> Dict[str, object]:
        """Build elem2face from one or more NFACE sections."""
        interior_elements.sort(key=lambda x: x.children["ElementRange"].data[0])

        connectivity_parts = [
            # Completes the multi-line `np.asarray(e.children['ElementConnectivity'].data, dtype=np.int64).reshape` arguments or elements.
            np.asarray(e.children["ElementConnectivity"].data, dtype=np.int64).reshape(-1)
            for e in interior_elements
        ]
        connectivity = np.concatenate(connectivity_parts)

        # Initializes the `section_slices` as a collection for values from this processing stage.
        section_slices = []
        total_elems = 0
        conn_base = 0
        for elem in interior_elements:
            # Stores the `conn` for use by the remaining steps in this flow.
            conn = np.asarray(elem.children["ElementConnectivity"].data, dtype=np.int64).reshape(-1)
            slices = self._parse_section_slices(elem, section_kind="NFACE")
            section_slices.append((conn_base, slices))

            start_id, end_id = elem.children["ElementRange"].data
            # Converts the `start_id` to `int` so later checks use a consistent type.
            start_id = int(start_id)
            end_id = int(end_id)
            if end_id < start_id:
                raise ValueError(f"Invalid NFACE ElementRange: [{start_id}, {end_id}]")

            # Stores the `nelem_section` for use by the remaining steps in this flow.
            nelem_section = len(slices)
            expected = end_id - start_id + 1
            if expected != nelem_section:
                logger.warning(
                    # Completes the multi-line `logger.warning` arguments or elements.
                    "NFACE section %s ElementRange count (%s) != parsed element count (%s), using parsed count",
                    elem.name,
                    expected,
                    nelem_section,
                # Completes the multi-line `logger.warning` arguments or elements.
                )

            total_elems += nelem_section
            conn_base += len(conn)

        expected_cell_count = int(self.get_mesh_size()["CellSize"])
        # Checks `total_elems != expected_cell_count` before entering this conditional branch.
        if total_elems != expected_cell_count:
            logger.warning(
                "The number of parsed volume cells %s does not match Zone CellSize %s.",
                total_elems,
                # Completes the multi-line `logger.warning` arguments or elements.
                expected_cell_count,
            )

        offset = np.zeros((total_elems, 2), dtype=np.int64)
        local_elem_idx = 0
        # Visits each item in `section_slices` and applies the current processing step.
        for conn_base, slices in section_slices:
            for s, e in slices:
                offset[local_elem_idx, 0] = conn_base + s
                offset[local_elem_idx, 1] = e - s
                # Stores the `local_elem_idx` for use by the remaining steps in this flow.
                local_elem_idx += 1

        if local_elem_idx != total_elems:
            raise ValueError(f"NFACE element assembly mismatch: {local_elem_idx} != {total_elems}")

        return {
            # Completes the multi-line `collection` arguments or elements.
            "offset": offset,
            "connectivity": connectivity,
        }

    def get_mesh_connectivity_poly(self) -> np.ndarray:
        """
        Extracts the mesh connectivity from the polyhedral mesh, which
        is stored in the Elements_t node with element type NGON_n and NFACE_n.

        Returns:
            np.ndarray: The mesh connectivity.
        """

        # find all the mesh entities
        # Stores the `entity_list` for use by the remaining steps in this flow.
        entity_list = self.get_node_by_label(CGNSNodeType_S.Elements)

        # extract the interior and boundary connectivity
        interior_elements = [e for e in entity_list if
                             e.data[0] == cgnskeywords.ElementType["NFACE_n"]]
        boundary_elements = [e for e in entity_list if
                             # Continues the surrounding statement or collection definition.
                             e.data[0] == cgnskeywords.ElementType["NGON_n"]]

        if len(interior_elements) == 0:
            emsg = f"No interior mesh entity found."
            logger.error(emsg)
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError(emsg)

        if len(boundary_elements) == 0:
            emsg = f"No boundary mesh entity found."
            logger.error(emsg)
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError(emsg)

        face2node = self._build_face2node_poly(boundary_elements)

        # extract elem2face
        elem2face = self._build_elem2face_poly(interior_elements)
        nelem = elem2face["offset"].shape[0]

        # extract the elem2node connectivity
        # there are several implementation options here
        # 0. 2d indexing, fixed length array of variable length array
        #   a. use python set and then convert to numpy array
        #   b. use numpy concatenate and unique function
        # 1. 1d indexing, fixed length array, preallocated
        # 2. 1d indexing, dynamically extended memory array

        # 1d indexing is compatible with the current implementation of
        # other connectivity with offset array, such as face2node and
        # elem2face. However, it is difficult to estimate the size of
        # the array for option 1.
        # For example, some NGON_n elements (face cells) may have more than
        #  16 nodes, while the rest only have typical 4, 5, 6 nodes.
        # On the other hand, option 2 is easy to implement, but it is
        # the slowest one.

        # 2d indexing is not compatible with the current implementation,
        # but it is easy to implement, with similar performance as option 1.
        # Therefore, we choose option 2d indexing with option 1.a for now,
        # since set is faster than concatenate and unique.

        # Stores the `ts` for use by the remaining steps in this flow.
        ts = time.time()
        offset = None  # we do not need offset by using 2d indexing
        connectivity = np.empty(nelem, dtype=object)
        # loop over elements
        for ielem in range(nelem):
            # get the offset of the element faces, in numpy index
            # Selects the required item from `elem2face["offset"]` as the `(elem_sface, elem_nface)`.
            elem_sface, elem_nface = elem2face["offset"][ielem, :]
            # get the face indices, in cgns index
            elem_faces = elem2face["connectivity"][elem_sface:
                                                   elem_sface + elem_nface]

            # ignore face orientation and keep absolute CGNS face ids.
            elem_faces = np.abs(elem_faces)

            # initialize the element nodes as a set
            # Stores the `elem_nodes` for use by the remaining steps in this flow.
            elem_nodes = set()

            # loop over faces
            for iface in elem_faces:
                local_face_idx = self._face_id_to_local_idx(
                    int(iface), face2node["face_ranges"]
                # Completes the multi-line `self._face_id_to_local_idx` arguments or elements.
                )
                # get the offset of the face nodes
                face_snode, face_nnode = face2node["offset"][local_face_idx, :]
                # get the node indices, in cgns index
                face_nodes = face2node["connectivity"][face_snode:
                                                       face_snode + face_nnode]
                # convert to numpy index
                # face_nodes = self._cgns_idx_to_numpy_idx(face_nodes)

                # add the face nodes to the element nodes, keep the cgns index
                # Adds the current value to `elem_nodes` for later aggregation.
                elem_nodes.update(face_nodes)

            # update the connectivity, also in cgns index
            connectivity[ielem] = np.asarray(list(elem_nodes))

        te = time.time()
        logger.debug("Extracting elem2node connectivity takes" +
                    # Completes the multi-line `logger.debug` arguments or elements.
                    f" {te-ts:.2f} s.")

        # use the numpy concatenate and unique
        # ts = time.time()
        # offset = None  # we do not need offset by using 2d indexing
        # connectivity = np.empty(nelem, dtype=object)
        # # loop over elements
        # for ielem in range(nelem):
        #     # get the offset of the element faces, in numpy index
        #     elem_sface, elem_nface = elem2face["offset"][ielem, :]
        #     # get the face indices, in cgns index
        #     elem_faces = elem2face["connectivity"][elem_sface:
        #                                            elem_sface + elem_nface]

        #     # ignore the face orientation
        #     elem_faces = np.abs(elem_faces)
        #     # convert to numpy index
        #     elem_faces = self._cgns_idx_to_numpy_idx(elem_faces)

        #     # initialize the element nodes array
        #     elem_nodes = np.array([], dtype=np.int64)

        #     # loop over faces
        #     for iface in elem_faces:
        #         # get the offset of the face nodes, in numpy index
        #         face_snode, face_nnode = face2node["offset"][iface, :]
        #         # get the node indices, in cgns index
        #         face_nodes = face2node["connectivity"][face_snode:
        #                                                face_snode + face_nnode]
        #         # convert to numpy index
        #         # face_nodes = self._cgns_idx_to_numpy_idx(face_nodes)

        #         # add the face nodes to the element nodes, keep the cgns index
        #         elem_nodes = np.concatenate((elem_nodes, face_nodes))

        #     # update the connectivity, also in cgns index
        #     connectivity[ielem] = elem_nodes

        # te = time.time()
        # logger.info("Extracting elem2node connectivity takes" +
        #             f" {te-ts:.2f} s.")

        # # use fixed length preallocated array
        # ts = time.time()
        # npe_max = np.max(face2node["offset"][:, 1]) * 2
        # offset = np.zeros((nelem, 2), dtype=np.int64)
        # connectivity = np.zeros(nelem*npe_max, dtype=np.int64)
        # start_index = 0

        # for ielem in range(nelem):
        #     # get the offset of the element faces, in numpy index
        #     elem_sface, elem_nface = elem2face["offset"][ielem, :]
        #     # get the face indices, in cgns index
        #     elem_faces = elem2face["connectivity"][elem_sface:
        #                                            elem_sface + elem_nface]

        #     # ignore the face orientation
        #     elem_faces = np.abs(elem_faces)
        #     # convert to numpy index
        #     elem_faces = self._cgns_idx_to_numpy_idx(elem_faces)

        #     # initialize the element nodes array
        #     # elem_nodes = np.array([], dtype=np.int64)
        #     elem_nodes = set()

        #     # loop over faces
        #     for iface in elem_faces:
        #         # get the offset of the face nodes, in numpy index
        #         face_snode, face_nnode = face2node["offset"][iface, :]
        #         # get the node indices, in cgns index
        #         face_nodes = face2node["connectivity"][face_snode:
        #                                                face_snode + face_nnode]
        #         # convert to numpy index
        #         # face_nodes = self._cgns_idx_to_numpy_idx(face_nodes)

        #         # add the face nodes to the element nodes, keep the cgns index
        #         # elem_nodes = np.concatenate((elem_nodes, face_nodes))
        #         elem_nodes.update(face_nodes)

        #     # remove the duplicated nodes
        #     # elem_nodes = np.unique(elem_nodes)

        #     # get the number of nodes in the element
        #     elem_nnode = len(elem_nodes)

        #     # update the offset
        #     offset[ielem, 0] = start_index
        #     offset[ielem, 1] = elem_nnode

        #     # update the connectivity, also in cgns index
        #     # connectivity[start_index: start_index + elem_nnode] = elem_nodes
        #     connectivity[start_index: start_index + elem_nnode] =\
        #         np.asarray(list(elem_nodes))

        #     # update the start index
        #     start_index += elem_nnode

        # te = time.time()
        # logger.info("Extracting elem2node connectivity takes" +
        #                f" {te-ts:.2f} s.")

        elem2node = {"offset": offset,
                     "connectivity": connectivity}

        return {"Face2Node": face2node,
                # Completes the multi-line `collection` arguments or elements.
                "Elem2Face": elem2face,
                "Elem2Node": elem2node,
                "ElementTypeIds": np.full(nelem, cgnskeywords.ElementType["NFACE_n"], dtype=np.int32),
                "MeshFormat": "poly"}

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _numpy_idx_to_cgns_idx(idx: np.ndarray) -> np.ndarray:
        """
        Convert the numpy index to CGNS index.

        Args:
            idx (np.ndarray): The numpy index.

        Returns:
            np.ndarray: The CGNS index.
        """
        return idx + 1

    @staticmethod
    # Convert the CGNS index to numpy index.
    def _cgns_idx_to_numpy_idx(idx: np.ndarray) -> np.ndarray:
        """
        Convert the CGNS index to numpy index.

        Args:
            idx (np.ndarray): The CGNS index.

        Returns:
            np.ndarray: The numpy index.
        """
        return idx - 1

    def get_mesh_connectivity(self) -> np.ndarray:
        """
        Extract the mesh connectivity from the Elements_t node.

        Returns:
            np.ndarray: The mesh connectivity.
        """

        if not self.is_node_instance(CGNSNodeType_S.Zone):
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError

        # find the Elements_t node
        entity_list = self.get_node_by_label(CGNSNodeType_S.Elements) or []
        if len(entity_list) == 0:
            return self.get_structured_mesh_connectivity()

        # Stores the `has_poly_volume` for use by the remaining steps in this flow.
        has_poly_volume = any(
            int(section.data[0]) == cgnskeywords.ElementType["NFACE_n"]
            for section in entity_list
        )
        # Returns early when `has_poly_volume` makes the remaining work inapplicable.
        if has_poly_volume:
            return self.get_mesh_connectivity_poly()

        interior_sections = self._select_interior_element_sections(entity_list)
        if len(interior_sections) == 0:
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError("No supported volume Elements_t section found in the current zone.")

        return self._build_standard_mesh_connectivity(interior_sections)

    def get_structured_vertex_dimensions(self) -> np.ndarray:
        """获取二维或三维结构网格的顶点逻辑尺寸。"""
        if not self.is_node_instance(CGNSNodeType_S.Zone):
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError

        zone_data = np.asarray(self.data, dtype=np.int64)
        if zone_data.ndim != 2 or min(zone_data.shape) < 2:
            raise ValueError("当前 Zone 不包含二维或三维结构网格尺寸")

        # Visits each item in `(3, 2)` and applies the current processing step.
        for dimension in (3, 2):
            candidates = []
            if zone_data.shape[0] >= 3 and zone_data.shape[1] >= dimension:
                candidates.append((zone_data[0, :dimension], zone_data[1, :dimension]))
            # Checks `zone_data.shape[1] >= 3 and zone_data.shape[0] >= dimension` before entering this conditional branch.
            if zone_data.shape[1] >= 3 and zone_data.shape[0] >= dimension:
                candidates.append((zone_data[:dimension, 0], zone_data[:dimension, 1]))
            for vertex_dims, cell_dims in candidates:
                if np.all(vertex_dims >= 2) and np.array_equal(cell_dims, vertex_dims - 1):
                    # Returns `np.asarray(vertex_dims, dtype=np.int64)` to the caller for the next stage.
                    return np.asarray(vertex_dims, dtype=np.int64)

        raise ValueError("仅支持 CellSize 等于 VertexSize - 1 的二维或三维结构网格")

    def get_structured_mesh_connectivity(self) -> dict:
        """根据 Zone_t 逻辑尺寸生成二维 QUAD_4 或三维 HEXA_8 连接。"""
        vertex_dims = self.get_structured_vertex_dimensions()
        # Stores the `(ni, nj)` for use by the remaining steps in this flow.
        ni, nj = (int(value) for value in vertex_dims[:2])
        cells = []
        if len(vertex_dims) == 2:
            def point_id(i: int, j: int) -> int:
                # Returns `i + ni * j + 1` to the caller for the next stage.
                return i + ni * j + 1

            for j in range(nj - 1):
                for i in range(ni - 1):
                    cells.append(
                        # Completes the multi-line `np.asarray` arguments or elements.
                        np.asarray(
                            [
                                point_id(i, j), point_id(i + 1, j),
                                point_id(i + 1, j + 1), point_id(i, j + 1),
                            # Completes the multi-line `collection` arguments or elements.
                            ],
                            dtype=np.int64,
                        )
                    )
            # Selects the required item from `cgnskeywords.ElementType` as the `element_type`.
            element_type = cgnskeywords.ElementType["QUAD_4"]
        else:
            nk = int(vertex_dims[2])

            def point_id(i: int, j: int, k: int) -> int:
                # Returns `i + ni * (j + nj * k) + 1` to the caller for the next stage.
                return i + ni * (j + nj * k) + 1

            for k in range(nk - 1):
                for j in range(nj - 1):
                    for i in range(ni - 1):
                        # Adds the current value to `cells` for later aggregation.
                        cells.append(
                            np.asarray(
                                [
                                    point_id(i, j, k), point_id(i + 1, j, k),
                                    # Completes the multi-line `point_id` arguments or elements.
                                    point_id(i + 1, j + 1, k), point_id(i, j + 1, k),
                                    point_id(i, j, k + 1), point_id(i + 1, j, k + 1),
                                    point_id(i + 1, j + 1, k + 1), point_id(i, j + 1, k + 1),
                                ],
                                # Completes the multi-line `np.asarray` arguments or elements.
                                dtype=np.int64,
                            )
                        )
            element_type = cgnskeywords.ElementType["HEXA_8"]

        # Returns `` to the caller for the next stage.
        return {
            "Elem2Node": {"connectivity": np.asarray(cells, dtype=object)},
            "ElementTypeIds": np.full(
                len(cells),
                # Completes the multi-line `np.full` arguments or elements.
                element_type,
                dtype=np.int32,
            ),
        }

    # Extract the edges from MIXED elements.
    def get_mesh_edges_mixed(self, bidirectional: Optional[bool] = True) -> set:
        """
        Extract the edges from MIXED elements.
        The native CGNS index (1-based) is preserved.

        Args:
            bidirectional (Optional[bool], optional): Whether to add the
                reversed edges. Defaults to True.

        Returns:
            set: The edges.
        """

        if not self.is_node_instance(CGNSNodeType_S.Zone):
            raise NotImplementedError

        entity_list = self.get_node_by_label(CGNSNodeType_S.Elements) or []
        # Builds the `mixed_sections` using the comprehension's filter and mapping rules.
        mixed_sections = [
            section
            for section in entity_list
            if int(section.data[0]) == cgnskeywords.ElementType["MIXED"]
        # Continues the surrounding statement or collection definition.
        ]
        if len(mixed_sections) == 0:
            raise ValueError("No MIXED Elements_t section found in the current zone.")

        return self._extract_edges_from_standard_sections(
            # Completes the multi-line `self._extract_edges_from_standard_sections` arguments or elements.
            mixed_sections,
            bidirectional=bool(bidirectional),
        )

    def get_mesh_edges_poly(self, bidirectional: Optional[bool] = True) -> set:
        """
        Extract the edges from NGON_n elements.
        The native CGNS index (1-based) is preserved.

        Args:
            bidirectional (Optional[bool], optional): Whether to add the
                reversed edges. Defaults to True.

        Returns:
            set: The edges.
        """
        # Validates `not self.is_node_instance(CGNSNodeType_S.Zone)` and raises `exception` when the input cannot be processed safely.
        if not self.is_node_instance(CGNSNodeType_S.Zone):
            raise NotImplementedError

        # find all mesh entities
        entity_list = self.get_node_by_label(CGNSNodeType_S.Elements)

        # find the NGON_n elements, i.e., face groups of polygons
        face_groups = [e for e in entity_list
                       # Continues the surrounding statement or collection definition.
                       if e.data[0] == cgnskeywords.ElementType["NGON_n"]]

        # sort the face groups
        face_groups.sort(key=lambda x: x.children["ElementRange"].data[0])

        # get total number of face elements across possibly non-1-based ranges
        nface = int(sum(
            int(g.children["ElementRange"].data[1]) - int(g.children["ElementRange"].data[0]) + 1
            # Completes the multi-line `int` arguments or elements.
            for g in face_groups
        ))

        edges = set()
        face_count = 0
        # loop over face groups
        # Visits each item in `face_groups` and applies the current processing step.
        for face_group in face_groups:
            # get the element range
            sface, eface = face_group.children["ElementRange"].data
            # get the connectivity
            connectivity = face_group.children["ElementConnectivity"].data

            # extract the edges
            iface = 0
            # Visits each item in `self._parse_section_slices(face_group, section_kind="NGON")` and applies the current processing step.
            for s, e in self._parse_section_slices(face_group, section_kind="NGON"):
                nodes = connectivity[s:e]
                nnode = len(nodes)
                # get the edges
                for inode in range(nnode):
                    # Adds the current value to `edges` for later aggregation.
                    edges.add((nodes[inode], nodes[(inode + 1) % nnode]))
                    # add the reversed edge if bidirectional is True
                    if bidirectional:
                        edges.add((nodes[(inode + 1) % nnode], nodes[inode]))
                # update the face index
                iface += 1
            # check if the face index is correct
            # Checks the internal invariant `iface == eface - sface + 1` before continuing.
            assert iface == eface - sface + 1

            # update the face count
            face_count += iface

        # check if the face count is correct
        assert face_count == nface

        return edges

    # Extract the mesh edges.
    def get_mesh_edges(self, bidirectional: Optional[bool] = True) -> set:
        """
        Extract the mesh edges.

        Args:
            bidirectional (Optional[bool], optional): Whether to add the
                reversed edges. Defaults to True.

        Returns:
            set: The mesh edges.

        Raises:
            NotImplementedError: If called on a non-zone node.
        """

        if not self.is_node_instance(CGNSNodeType_S.Zone):
            raise NotImplementedError

        entity_list = self.get_node_by_label(CGNSNodeType_S.Elements) or []
        # Validates `len(entity_list) == 0` and raises `ValueError` when the input cannot be processed safely.
        if len(entity_list) == 0:
            raise ValueError("No Elements_t section found in the current zone.")

        has_poly_volume = any(
            int(section.data[0]) == cgnskeywords.ElementType["NFACE_n"]
            # Completes the multi-line `any` arguments or elements.
            for section in entity_list
        )
        if has_poly_volume:
            return self.get_mesh_edges_poly(bidirectional)

        # Stores the `interior_sections` for use by the remaining steps in this flow.
        interior_sections = self._select_interior_element_sections(entity_list)
        if len(interior_sections) == 0:
            raise NotImplementedError("No supported volume Elements_t section found in the current zone.")

        return self._extract_edges_from_standard_sections(
            # Completes the multi-line `self._extract_edges_from_standard_sections` arguments or elements.
            interior_sections,
            bidirectional=bool(bidirectional),
        )

    # extract the mesh connectivity from the node Connectivity_t
    # can be called from the root node or from the Zone_t node
    # Ideally, this function should only be called from the Zone_t node.
    # def get_mesh_connectivity(self) -> np.ndarray:
    #     """
    #     This function extracts the mesh connectivity from the node with the
    #     label Elements_t.
    #     Currently, only mixed elements are supported.
    #     """

    #     # first find the node with the label "Elements_t"
    #     domain_list = self.get_node_by_label("Elements_t")
    #     edges = set()
    #     for domain in domain_list:
    #         elem_connectivity = domain.get_node_by_name("ElementConnectivity").data
    #         edges.update(cgnsmesh.read_mixed_mesh(elem_connectivity))
    #
    #     return edges

    # ==================== #
    # 3. Boundary section  #
    # ==================== #

    def get_boundary_conditions(
        # Continues the surrounding statement or collection definition.
        self,
        family_bc_types: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Extracts the boundary conditions from the node ZoneBC_t.
        Each child node of ZoneBC_t is a BC_t node, which contains
        the boundary condition information. The boundary condition
        information is stored in the following format:
        self.data = BCType string
        self.children = {GridLocation: GridLocation string,
                         Optional[PointList]: PointList array,
                         Optional[PointRange]: PointRange array}
        The BC patch may be specified by PointRange if it constitutes
        a logically rectangular region.
        In all other cases, PointList should be used to list the vertices
        or cell edges/faces making up the BC patch.
        When GridLocation is set to Vertex, then PointList or PointRange
        refer to vertex indices, for both structured and unstructured grids.
        When GridLocation is set to EdgeCenter, then PointRange/List refer to
        edge elements. For 3-D grids, when GridLocation is set to FaceCenter,
        IFaceCenter, etc., then PointRange/List refer to face elements.
        More details can be found in the CGNS official documentation,
        https://cgns.github.io/CGNS_docs_current/sids/bc.html#BC.

        Returns:
            dict: The boundary condition information.

        Raises:
            ValueError: if the BC infos are corrupted.
            NotImplementedError: if called by an invalid node.
        """

        if self.is_node_instance(CGNSNodeType_S.Zone) or \
           self.is_node_instance(CGNSNodeType_S.ZoneBC):
            # Handles the data or control-flow case represented by this statement.
            pass
        else:
            raise NotImplementedError

        # 缺少 BC_t 在很多真实样本中是正常情况，这里直接返回空字典，
        # 避免把“无边界条件数据”当成异常路径刷日志。
        bc_list = self._search_node_by_label(CGNSNodeType_S.BC)
        # Initializes the `bc_info` as a collection for values from this processing stage.
        bc_info = {}

        if len(bc_list) == 0:
            return bc_info

        def decode_node_text(node: Optional['Node']) -> Optional[str]:
            # Returns early when `node is None or node.data is None` makes the remaining work inapplicable.
            if node is None or node.data is None:
                return None
            value = np.asarray(node.data).reshape(-1)
            if value.size == 0:
                # Returns `None` to the caller for the next stage.
                return None
            raw_value = value[0]
            if isinstance(raw_value, bytes):
                return raw_value.decode(errors="replace").rstrip("\x00")
            # Returns `str(raw_value).rstrip("\x00")` to the caller for the next stage.
            return str(raw_value).rstrip("\x00")

        resolved_family_types = family_bc_types or {}

        # loop over the bc nodes
        for bc in bc_list:
            bc_name = bc.name
            # Stores the `raw_bc_type` for use by the remaining steps in this flow.
            raw_bc_type = decode_node_text(bc)
            if not raw_bc_type:
                raise ValueError(f"边界条件解析失败: BC_t '{bc_name}' 缺少 BCType")

            family_name = decode_node_text(bc.children.get("FamilyName"))
            # Stores the `bc_type` for use by the remaining steps in this flow.
            bc_type = raw_bc_type
            if raw_bc_type == "FamilySpecified":
                if not family_name:
                    raise ValueError(
                        # Completes the multi-line `ValueError` arguments or elements.
                        f"边界条件解析失败: BC_t '{bc_name}' 为 FamilySpecified 但缺少 FamilyName"
                    )
                bc_type = resolved_family_types.get(family_name)
                if not bc_type or bc_type == "FamilySpecified":
                    # Raises `ValueError` to report why this path cannot continue.
                    raise ValueError(
                        f"边界条件解析失败: BC_t '{bc_name}' 的 Family '{family_name}' 未提供有效 FamilyBC"
                    )

            grid_loc = decode_node_text(bc.children.get("GridLocation")) or "Vertex"

            # Reads `PointList` from the mapping into the `point_list_node`.
            point_list_node = bc.children.get("PointList")
            point_range_node = bc.children.get("PointRange")
            if point_list_node is not None:
                point_list = point_list_node.data
                # Stores the `point_range` for use by the remaining steps in this flow.
                point_range = None
                point_set_type = "PointList"
            elif point_range_node is not None:
                point_list = None
                # Stores the `point_range` for use by the remaining steps in this flow.
                point_range = point_range_node.data
                point_set_type = "PointRange"
            else:
                raise ValueError(
                    # Completes the multi-line `ValueError` arguments or elements.
                    "边界条件解析失败: "
                    f"BC_t '{bc_name}' 的 GridLocation={grid_loc} 缺少 PointList 或 PointRange"
                )

            bc_info[bc_name] = {
                # Completes the multi-line `collection` arguments or elements.
                "type": bc_type,
                "raw_type": raw_bc_type,
                "family_name": family_name,
                "grid_loc": grid_loc,
                # Completes the multi-line `collection` arguments or elements.
                "point_set_type": point_set_type,
                "point_list": point_list,
                "point_range": point_range,
            }

        # Returns `bc_info` to the caller for the next stage.
        return bc_info

    def get_grid_connectivities(self) -> dict:
        """提取当前 Zone 的网格连接区域，供 interface 局部特征使用。"""
        if not self.is_node_instance(CGNSNodeType_S.Zone):
            raise NotImplementedError

        # Stores the `connectivity_nodes` for use by the remaining steps in this flow.
        connectivity_nodes = self._search_node_by_label("GridConnectivity_t")
        connectivity_nodes.extend(self._search_node_by_label("GridConnectivity1to1_t"))
        connectivity_info = {}
        for connectivity in connectivity_nodes:
            # Reads `PointList` from the mapping into the `point_list_node`.
            point_list_node = connectivity.children.get("PointList")
            point_range_node = connectivity.children.get("PointRange")
            if point_list_node is not None:
                point_list = point_list_node.data
                # Stores the `point_range` for use by the remaining steps in this flow.
                point_range = None
                point_set_type = "PointList"
            elif point_range_node is not None:
                point_list = None
                # Stores the `point_range` for use by the remaining steps in this flow.
                point_range = point_range_node.data
                point_set_type = "PointRange"
            else:
                continue

            # Reads `PointListDonor` from the mapping into the `donor_point_list_node`.
            donor_point_list_node = connectivity.children.get("PointListDonor")
            donor_point_range_node = connectivity.children.get("PointRangeDonor")
            if donor_point_list_node is not None:
                donor_point_list = donor_point_list_node.data
                # Stores the `donor_point_range` for use by the remaining steps in this flow.
                donor_point_range = None
                donor_point_set_type = "PointList"
            elif donor_point_range_node is not None:
                donor_point_list = None
                # Stores the `donor_point_range` for use by the remaining steps in this flow.
                donor_point_range = donor_point_range_node.data
                donor_point_set_type = "PointRange"
            else:
                donor_point_list = None
                # Stores the `donor_point_range` for use by the remaining steps in this flow.
                donor_point_range = None
                donor_point_set_type = None

            grid_location_node = connectivity.children.get("GridLocation")
            grid_location = "Vertex"
            # Checks `` before entering this conditional branch.
            if grid_location_node is not None and grid_location_node.data is not None:
                raw_value = np.asarray(grid_location_node.data).reshape(-1)
                if raw_value.size > 0:
                    grid_location = (
                        # Completes the multi-line `raw_value[0].decode(errors='replace').rstrip` arguments or elements.
                        raw_value[0].decode(errors="replace").rstrip("\x00")
                        if isinstance(raw_value[0], bytes)
                        else str(raw_value[0]).rstrip("\x00")
                    )

            # Stores the `donor_zone` for use by the remaining steps in this flow.
            donor_zone = None
            if connectivity.data is not None:
                raw_donor = np.asarray(connectivity.data).reshape(-1)
                if raw_donor.size > 0:
                    # Stores the `donor_zone` for use by the remaining steps in this flow.
                    donor_zone = (
                        raw_donor[0].decode(errors="replace").rstrip("\x00")
                        if isinstance(raw_donor[0], bytes)
                        else str(raw_donor[0]).rstrip("\x00")
                    # Continues the surrounding statement or collection definition.
                    )
            connectivity_info[connectivity.name] = {
                "type": "interface",
                "raw_type": connectivity.label,
                # Completes the multi-line `collection` arguments or elements.
                "record_kind": "grid_connectivity",
                "donor_zone": donor_zone,
                "grid_loc": grid_location,
                "point_set_type": point_set_type,
                # Completes the multi-line `collection` arguments or elements.
                "point_list": point_list,
                "point_range": point_range,
                "donor_point_set_type": donor_point_set_type,
                "donor_point_list": donor_point_list,
                # Completes the multi-line `collection` arguments or elements.
                "donor_point_range": donor_point_range,
            }
        return connectivity_info

    def get_boundary_flags(self) -> np.ndarray:
        """
        Flag the boundary nodes, which are the nodes that are
        specified by the boundary conditions. The boundary nodes
        are flagged according to the CGNSBCType enum defined in
        `tf_cgns.py`.

        Returns:
            np.ndarray: The boundary node flags.

        Raises:
            NotImplementedError: if called by an invalid node.
                Since the flagging needs both the bc infos and the
                mesh connectivity, it should be called by the Zone_t node.
            NotImplementedError: if the bc type is not supported.
        Shape:
            returned ndarray: (nnode, )
        """

        # Validates `not self.is_node_instance(CGNSNodeType_S.Zone)` and raises `exception` when the input cannot be processed safely.
        if not self.is_node_instance(CGNSNodeType_S.Zone):
            raise NotImplementedError

        # get the number of vertex nodes
        nnode = self.get_mesh_size()["VertexSize"]

        # get the bc infos
        bc_info = self.get_boundary_conditions()

        # initialize the bc flags
        # bc_flags = np.zeros(nnode, dtype=np.int32)
        # Stores the `bc_flags` for use by the remaining steps in this flow.
        bc_flags = np.zeros(nnode)

        # loop over the bc group
        # for bc_name, bc in bc_info.items():
        #     # get the bc type
        #     bc_type = bc["type"]
        #     # get the bc grid location
        #     bc_grid_loc = bc["grid_loc"]
        #     # get the bc point list
        #     bc_point_list = bc["point_list"]
        #     # get the bc point range
        #     bc_point_range = bc["point_range"]

        #     # get the bc flag
        #     bc_flag = None
        #     for bc_type_key, bc_type_val in CGNSBCType_S2E.items():
        #         if bc_type_key in bc_type:
        #             bc_flag = bc_type_val
        #             break
        #     # check if the bc is in tower
        #     if "tower" in bc_name.lower():
        #         bc_type_key = "Tower_" + bc_type_key
        #         bc_flag = CGNSBCType_S2E[bc_type_key]

        #     # check if the bc flag is found
        #     if bc_flag is None:
        #         emsg = f"Boundary condition flagging failed: "\
        #             f"BC type {bc_type} not supported"
        #         logger.error(emsg)
        #         raise NotImplementedError(emsg)

        #     if bc_grid_loc == "Vertex":
        #         bc_nodes = self._cgns_idx_to_numpy_idx(bc_point_list)
        #         bc_flags[bc_nodes] = bc_flag
        #     elif bc_grid_loc == "FaceCenter":
        #         face2node = self.get_mesh_face2node()
        #         offset = face2node["offset"]
        #         face2node = face2node["connectivity"]
        #         sface, eface = self._cgns_idx_to_numpy_idx(bc_point_range.flatten())
        #         for iface in range(sface, eface+1):
        #             snode, nnode = offset[iface]
        #             nodes = face2node[snode:snode+nnode]
        #             nodes = self._cgns_idx_to_numpy_idx(nodes)
        #             bc_flags[nodes] = bc_flag
        #     else:
        #         emsg = f"Boundary condition flagging failed: "\
        #             f"GridLocation {bc_grid_loc} not supported"
        #         logger.error(emsg)
        #         raise NotImplementedError(emsg)

        # loop over the CGNSBCType enum, as the strong bc flagging
        # will overwrite the weak bc flagging
        for bc_type_key, bc_type_val in CGNSBCType_S2E.items():
            # loop over the bc group
            for bc_name, bc in bc_info.items():
                # get the bc type
                bc_type = bc["type"]
                # get the bc grid location
                # Selects the required item from `bc` as the `bc_grid_loc`.
                bc_grid_loc = bc["grid_loc"]
                # get the bc point list
                bc_point_list = bc["point_list"]
                # get the bc point range
                bc_point_range = bc["point_range"]

                bc_type = "Tower_" + bc_type if "tower" in bc_name.lower() \
                          else bc_type
                # add for chimney case
                # Stores the `bc_type` for use by the remaining steps in this flow.
                bc_type = "Chimney_" + bc_type if "chimney" in bc_name.lower() \
                          else bc_type
                # added for interface case
                bc_type = CGNSBCType_S.Tower_BCInterface_In if \
                    "tower" in bc_name.lower() and "_in" in bc_name.lower() \
                    else bc_type
                bc_type = CGNSBCType_S.Tower_BCInterface_Out if \
                    "tower" in bc_name.lower() and "_out" in bc_name.lower() \
                    else bc_type

                # check if the bc type is the current bc type
                if bc_type_key in bc_type:
                    # get the bc flag
                    # Stores the `bc_flag` for use by the remaining steps in this flow.
                    bc_flag = bc_type_val

                    # check if the bc grid location is vertex
                    if bc_grid_loc == "Vertex":
                        bc_nodes = self._cgns_idx_to_numpy_idx(bc_point_list)
                        bc_flags[bc_nodes] = bc_flag
                    # check if the bc grid location is face center
                    # Checks `bc_grid_loc == "FaceCenter"` before entering this conditional branch.
                    elif bc_grid_loc == "FaceCenter":
                        face2node = self.get_mesh_face2node()
                        offset = face2node["offset"]
                        face_ranges = face2node["face_ranges"]
                        # Selects the required item from `face2node` as the `face2node`.
                        face2node = face2node["connectivity"]
                        sface, eface = [int(v) for v in bc_point_range.flatten()]
                        if sface <= eface:
                            face_ids = range(sface, eface + 1)
                        # Continues the surrounding statement or collection definition.
                        else:
                            face_ids = range(eface, sface + 1)

                        for face_id in face_ids:
                            iface = self._face_id_to_local_idx(face_id, face_ranges)
                            # Selects the required item from `offset` as the `(snode, nnode)`.
                            snode, nnode = offset[iface]
                            nodes = face2node[snode:snode+nnode]
                            nodes = self._cgns_idx_to_numpy_idx(nodes)
                            bc_flags[nodes] = bc_flag
                    # Continues the surrounding statement or collection definition.
                    else:
                        emsg = f"Boundary condition flagging failed: "\
                            f"GridLocation {bc_grid_loc} not supported"
                        logger.error(emsg)
                        raise NotImplementedError(emsg)

        # Returns `bc_flags` to the caller for the next stage.
        return bc_flags

    # helper function for boundary node flagging
    # TODO: should add request param for the get_mesh_connectivity function
    #       to achieve this functionality
    def get_mesh_face2node(self) -> np.ndarray:
        """
        Extracts the face2node mesh connectivity from the polyhedral mesh,
        which is stored in the Elements_t node with element type NGON_n and
        NFACE_n. The implementation is taken from the `get_mesh_connectivity`
        function in `tf_cgns.py`.

        Returns:
            np.ndarray: The mesh connectivity.
        """

        # find all the mesh entities
        entity_list = self.get_node_by_label(CGNSNodeType_S.Elements)

        # extract the interior and boundary connectivity
        interior_elements = [e for e in entity_list if
                             # Continues the surrounding statement or collection definition.
                             e.data[0] == cgnskeywords.ElementType["NFACE_n"]]
        boundary_elements = [e for e in entity_list if
                             e.data[0] == cgnskeywords.ElementType["NGON_n"]]

        if len(boundary_elements) == 0:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = f"No boundary mesh entity found."
            logger.error(emsg)
            raise NotImplementedError(emsg)

        return self._build_face2node_poly(boundary_elements)

    # ==================== #
    # 4. Solution section  #
    # ==================== #

    # The following functions are dealing with the solution data
    # The solution data extraction, manipulation, and writing,
    # should all be done on the Solution_t node.

    # As the python variable only has the reference to the node,
    # we need to get the parent node, which is the Solution_t node,
    # in order to manipulate the solution data, such as adding new
    # solution data, deleting existing solution data, etc.

    # Get the Solution_t node, which is the parent node of the
    def get_solution_node(self) -> 'Node':
        """
        Get the Solution_t node, which is the parent node of the
        solution data. Currently, we only support one solution nodea.

        Returns:
            Node: The single Solution_t node if found, None otherwise.

        Raises:
            NotImplementedError: if called by an invalid node.
        """

        if self.is_node_instance(CGNSNodeType_S.Zone) or\
           self.is_node_instance(CGNSNodeType_S.FlowSolution):
            pass
        else:
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError

        # find the solution node list
        sol_list = self.get_node_by_label(CGNSNodeType_S.FlowSolution)

        if sol_list is None:
            emsg = f"Cannot find the solution node"
            # Records the current execution state to make progress and failures traceable.
            logger.warning(emsg)
            return None

        # check if there is more than one solution node
        if len(sol_list) > 1:
            emsg = f"Multiple solution nodes found, only one solution node is supported"
            # Records the current execution state to make progress and failures traceable.
            logger.warning(emsg)

        # return the solution node
        return sol_list[0]

    #该函数读取cgns结果文件，并把求解场的数据和场名称放到字典变量sol_data中
    def get_solution_datadict(self) -> Dict[str, np.ndarray]:
        """
        Extracts the solution data in dictionary format.

        Returns:
            Dict[str, np.ndarray]: The solution data in dictionary format.
                The key is the solution data name, and the value is the
                solution data array.

        Raises:
            NotImplementedError: if called by an invalid node.
        """
        # since the solution data are general data arrays,
        # we cannot directly find them by the label "DataArray_t",
        # other data, such as Coordinates, Connectivity, etc. are also
        # stored in the DataArray_t node.
        # Therefore, we need to find the node with the label "FlowSolution_t"
        # and then extract the solution data from that parent node.

        if self.is_node_instance(CGNSNodeType_S.Zone) or\
           self.is_node_instance(CGNSNodeType_S.FlowSolution):
            # Handles the data or control-flow case represented by this statement.
            pass
        else:
            raise NotImplementedError

        # find the solution node,这里我感觉有些重复
        sol_node = self.get_solution_node()

        # Checks `sol_node is None` before entering this conditional branch.
        if sol_node is None:
            emsg = f"Solution data extraction failed as no solution node found"
            logger.warning(emsg)
            return None

        # create a dictionary to store the solution data
        # Initializes the `sol_data` as a collection for values from this processing stage.
        sol_data = {}
        # fill in the dictionary
        for child in sol_node.children.values():
            sol_data[child.name] = child.data

        return sol_data

    # Extracts the solution data from the solution node with the given
    def get_solution_data_by_name(self, comp_name: str) -> Union[np.ndarray, None]:
        """
        Extracts the solution data from the solution node with the given
        component name.

        Args:
            comp_name (str): The component name of the solution data.

        Returns:
            np.ndarray: The solution data array.

        Raises:
            NotImplementedError: if called by an invalid node.
        """
        if self.is_node_instance(CGNSNodeType_S.Zone) or\
           self.is_node_instance(CGNSNodeType_S.FlowSolution):
            pass
        else:
            # Raises `NotImplementedError` to report why this path cannot continue.
            raise NotImplementedError

        # locate the solution node
        sol_node = self.get_solution_node()

        # extract the solution data by the component name
        try:
            sol_data = sol_node.children[comp_name].data
        # Continues the surrounding statement or collection definition.
        except KeyError:
            emsg = f"Solution data extraction failed: " +\
                    "component {comp_name} not found"
            logger.warning(emsg)
            return None

        # Returns `sol_data` to the caller for the next stage.
        return sol_data

    # Besides solution data extraction, we also need more data manipulation
    # functions, such as data deletion, addition, and replacement.
    def delete_sol_data(self, comp_name: str) -> bool:
        """
        Deletes the data from the solution node with the given component name.

        Args:
            comp_name (str): The component name of the solution data.

        Returns:
            bool: True if the data is deleted, False otherwise.

        Raises:
            NotImplementedError: if called by an invalid node.
        """
        if not self.is_node_instance(CGNSNodeType_S.FlowSolution):
            raise NotImplementedError
        # Stores the `sol_node` for use by the remaining steps in this flow.
        sol_node = self.get_solution_node()

        # delete the data by the component name
        deleted = False
        try:
            del sol_node.children[comp_name]
            # Stores the `deleted` for use by the remaining steps in this flow.
            deleted = True
        except KeyError:
            emsg = f"Solution data deletion failed: component {comp_name} not found"
            logger.warning(emsg)

        # Returns `deleted` to the caller for the next stage.
        return deleted

    def add_sol_data(self, comp_name: str, data: np.ndarray) -> bool:
        """
        Adds the data to the solution node with the given component name.

        Args:
            comp_name (str): The component name of the solution data.
            data (np.ndarray): The solution data array.

        Returns:
            bool: True if the data is added, False otherwise.

        Raises:
            NotImplementedError: if called by an invalid node.
        """
        if not self.is_node_instance(CGNSNodeType_S.FlowSolution):
            raise NotImplementedError
        # Stores the `sol_node` for use by the remaining steps in this flow.
        sol_node = self.get_solution_node()

        # ----------- add the data by the component name -------------------#
        # currently we only support the data type of float32 and float64.
        # node flags are default to 1 in both Ansys Fluent and STAR-CCM,
        # while in pyCGNS it is 0 by default. Does not seem to matter in
        # reading and writing for now.
        flags = np.array([1], dtype=np.int32)
        data_node = Node(name=comp_name, label="DataArray_t",
                         flags=flags, data=data)

        # check the data type
        # Checks `data.dtype not in [np.float32, np.float64]` before entering this conditional branch.
        if data.dtype not in [np.float32, np.float64]:
            emsg = f"Solution data addition failed: " +\
                    "data type {data.dtype} not supported."
            logger.warning(emsg)
            return False

        # check the data dimension by comparing the data shape with the
        # child node shape get the first key
        # Selects the required item from `list(sol_node.children.keys())` as the `first_sol_key`.
        first_sol_key = list(sol_node.children.keys())[0]
        if data.shape != sol_node.children[first_sol_key].data.shape:
            emsg = f"Solution data addition failed: " +\
                    "data dimension {data.shape} not matching"
            logger.warning(emsg)
            # Calls `sol_node.add_child` to perform the operation required at this stage.
            sol_node.add_child(data_node)
            return False
        else:
            # add the data node to the solution node
            sol_node.add_child(data_node)

    # Replaces the data in the solution node with the given component name.
    def replace_sol_data(self, comp_name: str, data: np.ndarray,
                         inplace: Optional[bool] = True) -> None:
        """
        Replaces the data in the solution node with the given component name.

        Args:
            comp_name (str): The component name of the solution data.
            data (np.ndarray): The solution data array.
            inplace (bool, optional): If True, replace the data in place.

        Raises:
            NotImplementedError: if called by an invalid node.
        """
        if not self.is_node_instance(CGNSNodeType_S.FlowSolution):
            raise NotImplementedError
        # locate the solution node
        # Stores the `sol_node` for use by the remaining steps in this flow.
        sol_node = self.get_solution_node()

        # replace the data by the component name
        # If inplace is true, we replace the data in place
        # otherwise, we delete the data and then add the new data
        if inplace:
            try:
                # check the dimension of the data
                if data.shape != sol_node.children[comp_name].data.shape:
                    # Derives the `emsg` from the current values for downstream processing.
                    emsg = f"Solution data replacement failed: " +\
                           "data dimension {data.shape} not matching"
                    logger.warning(emsg)
                else:
                    sol_node.children[comp_name].data = data
            # Continues the surrounding statement or collection definition.
            except KeyError:
                emsg = f"Solution data replacement failed: " +\
                        "component {comp_name} not found"
                logger.warning(emsg)
        else:
            # delete the data and then add the new data
            # Calls `self.delete_sol_data` to perform the operation required at this stage.
            self.delete_sol_data(comp_name)
            self.add_sol_data(comp_name, data)

    # check the data data compatibility of the input data and the data of
    # the current node
    def _check_data_comp(data: np.ndarray, node) -> bool:
        """
        This function checks if the input data is compatible for
        replacing the data of the current node. The compatibility is checked
        based on the data type and shape of the data.
        """

        # initialize the compatibility flag
        compat = True

        # check the data
        # Checks `data is not None and node.data is not None` before entering this conditional branch.
        if data is not None and node.data is not None:
            # check the data type and data shapes
            if data.dtype != node.data.dtype:
                emsg = f"Node data type do not match: {data.dtype} != {node.data.dtype}"
                logger.warning(emsg)
                # Stores the `compat` for use by the remaining steps in this flow.
                compat = False
            if data.shape != node.data.shape:
                emsg = f"Node data shape do not match: {data.shape} != {node.data.shape}"
                logger.warning(emsg)
                # Stores the `compat` for use by the remaining steps in this flow.
                compat = False
        elif data is None and node.data is None:
            pass
        else:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = f"Either of the node data is None but not both"
            logger.warning(emsg)
            compat = False

        return compat

    # check if the input node is compatible for replacing the current node
    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _check_node_comp(in_node, c_node, check_children: bool = False) -> bool:
        """
        This function checks if the input node is compatible for
        replacing the current node. The compatibility is checked
        based on the string attributes and the data type and shape of
        the data. The compatibility is checked recursively for the
        children of the nodes if the check_children flag is set to True,
        default is False.
        """

        # check the attrs of the node.
        # the names are not checked as renamed nodes are allowed
        attrs = ["label", "flags"]
        for attr in attrs:
            # Checks `getattr(in_node, attr, None) != getattr(c_node, attr, None)` before entering this conditional branch.
            if getattr(in_node, attr, None) != getattr(c_node, attr, None):
                # None is used as the default value for the getattr
                # in case that flags are missing in the node
                emsg = f"Node {attr} do not match: {getattr(in_node, attr, None)} != {getattr(c_node, attr, None)}"
                logger.warning(emsg)
                return False

        # check the data
        # Checks `in_node.data is not None and c_node.data is not None` before entering this conditional branch.
        if in_node.data is not None and c_node.data is not None:
            # check the data type and data shapes
            if in_node.data.dtype != c_node.data.dtype:
                emsg = f"Node data type do not match: {in_node.data.dtype} != {c_node.data.dtype}"
                logger.warning(emsg)
                # Returns `False` to the caller for the next stage.
                return False
            if in_node.data.shape != c_node.data.shape:
                emsg = f"Node data shape do not match: {in_node.data.shape} != {c_node.data.shape}"
                logger.warning(emsg)
                # Returns `False` to the caller for the next stage.
                return False
        elif in_node.data is None and c_node.data is None:
            pass
        else:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = f"Either of the node data is None but not both"
            logger.warning(emsg)
            return False

        # check if the node children are the same
        if check_children:
            # Checks `in_node.children is not None and c_node.children is not None` before entering this conditional branch.
            if in_node.children is not None and c_node.children is not None:
                if len(in_node.children) != len(c_node.children):
                    emsg = f"Node children length do not match: {in_node.children} != {c_node.children}"
                    logger.warning(emsg)
                    # Returns `False` to the caller for the next stage.
                    return False
                else:
                    for cname, child in in_node.children.items():
                        if cname not in c_node.children:
                            # Stores the `emsg` for use by the remaining steps in this flow.
                            emsg = f"Node children names do not match: {in_node.children} != {c_node.children}"
                            logger.warning(emsg)
                            return False
                        else:
                            # Stores the `compat` for use by the remaining steps in this flow.
                            compat = Node._check_node_comp(child, c_node.children[cname], check_children=True)
                            if not compat:
                                emsg = f"Node children do not match: {in_node.children} != {c_node.children}"
                                logger.warning(emsg)
                                # Returns `False` to the caller for the next stage.
                                return False
            elif in_node.children is None and c_node.children is None:
                pass
            else:
                # Stores the `emsg` for use by the remaining steps in this flow.
                emsg = f"Node children do not match: {in_node.children} != {c_node.children}"
                logger.warning(emsg)
                return False

        return True

    # replace the node data with the input data given a node name
    # This function replaces the data of the node with the given name
    def replace_data_by_name(self, node_name: str, data: np.ndarray) -> None:
        """
        This function replaces the data of the node with the given name
        with the input data. The data is replaced only if the data is
        compatible with the current data of the node, including the
        data type and shape.
        """

        # find the node with the given name
        node = self.get_node_by_name(node_name)
        if node is None:
            emsg = f"data replacement failed: node {node_name} not found"
            # Records the current execution state to make progress and failures traceable.
            logger.warning(emsg)
        else:
            # check the data compatibility
            compat = Node._check_data_comp(data, node)
            if compat:
                # Stores the current data for use by the remaining steps in this flow.
                node.data = data
            else:
                emsg = f"data replacement failed: data not compatible"
                logger.warning(emsg)

        # Ends this function and returns control to its caller.
        return

    # replace the node with the input node given a node name
    def replace_node_by_name(self, node_name: str, in_node, replace_children: bool = False) -> None:
        """
        This function replaces the node with the given name
        with the input node. The node is replaced only if the node is
        compatible with the current node, including the
        data type and shape.
        """

        # find the node with the given name
        node = self.get_node_by_name(node_name)
        if node is None:
            # Stores the `emsg` for use by the remaining steps in this flow.
            emsg = f"node replacement failed: node {node_name} not found"
            logger.warning(emsg)
        else:
            # check the node compatibility
            compat = Node._check_node_comp(in_node, node, replace_children)
            # Checks `compat` before entering this conditional branch.
            if compat:
                # replace the node
                node.label = in_node.label
                node.flags = in_node.flags
                node.data = in_node.data
                # Checks `replace_children` before entering this conditional branch.
                if replace_children:
                    emsg = f"node children replacement not implemented"
                    logger.error(emsg)
                    raise NotImplementedError(emsg)

            # Continues the surrounding statement or collection definition.
            else:
                emsg = f"node replacement failed: node not compatible"
                logger.warning(emsg)

        return

    # ----------------------------------------------------------------------- #
    # The following codes are for cell-centered data manipulation, which is   #
    # the default CGNS data format for STAR-CCM+. To make the data            #
    # vertex-centered, we need to convert the data format.                    #
    # ----------------------------------------------------------------------- #

    # "
    def _get_cell_center_coords(self):
        """"
        This function gets the cell-centered coordinates from the node coords.

        Args:
            None

        Returns:
            cell_center_coords: the cell-centered coordinates
        """

        # get the node coords
        coords_node = self.get_mesh_coords()

        # get the connectivity
        domain_list = self.get_node_by_label("Elements_t")
        elem_connectivity = None
        # Visits each item in `domain_list` and applies the current processing step.
        for domain in domain_list:
            if elem_connectivity is None:
                elem_connectivity = domain.get_node_by_name("ElementConnectivity").data
            else:
                # elem_connectivity = np.vstack((elem_connectivity,
                #                                domain.get_node_by_name("ElementConnectivity").data))
                # Stores the `elem_connectivity` for use by the remaining steps in this flow.
                elem_connectivity = np.concatenate((elem_connectivity,
                                                    domain.get_node_by_name("ElementConnectivity").data))

        return coords_node, elem_connectivity


# define a CGNS data class to hold the data and the CGNS header
class CGNSData(object):
    # Defines `__init__` for   init   logic.
    def __init__(self, header: dict = None, root: Node = None) -> None:
        # if header is None:
        #    header = get_default_cgns_header()
        # since the write function in the CGNS IO provides default header option
        # we do not need to provide default header here. Just for now.
        # TODO: need a clearer way to handle the default header
        self.header = header
        self.root = root

    def __str__(self) -> str:
        # get the hdf5 format and version
        # Stores the `h5_format` for use by the remaining steps in this flow.
        h5_format = self.header[" format"].decode()  # .rstrip("\x00")
        version = self.header[" hdf5version"].decode()  # .rstrip()
        out_str = f"CGNS format: {h5_format}\n"
        out_str += f"HDF5 version: {version}\n"

        # Stores the `out_str` for use by the remaining steps in this flow.
        out_str += self.root.__str__()

        return out_str

    def __repr__(self) -> str:
        return self.__str__()

    # wrap some useful member functions from the Node class
    # to make the CGNS data class more user-friendly
    # Since the root attribute is a Node object, we can directly
    # call the member functions of the Node class.

    # So far, we only need the following functions:
    # 1. check_data: check if the CGNS data is valid or supported
    # 2. get_solution_data: get the solution data from the CGNS data
    # 3. get_solution_data_by_name: get the solution data by the component name

    # This function extracts the basic infos from the root data node:
    def get_basic_info(self) -> None:
        """
        This function extracts the basic infos from the root data node:

        """
        self.dim_info = self.root.get_dimension_info()
        self.zone_type = self.root.get_zone_type()
        self.zone_info = self.root.get_zone_info()

    # This function extracts the solution data from the root data node
    def get_solution_data(self) -> dict:
        """
        This function extracts the solution data from the root data node
        and returns a dictionary of the solution data.
        """
        return self.root.get_solution_data()

    def get_solution_data_by_name(self, comp_name: str) -> np.ndarray:
        """
        This function extracts the solution data from the root data node
        by the given component name and returns the data.
        """
        return self.root.get_solution_data_by_name(comp_name)

    # This function extracts the mesh coordinates from the root data node
    def get_mesh_coords(self) -> np.ndarray:
        """
        This function extracts the mesh coordinates from the root data node
        and returns the data.
        """
        return self.root.get_mesh_coords()

    def check_data(self) -> bool:
        """
        This function checks if the CGNS data is valid or supported.
        """

        self.get_basic_info()
        # Checks `self.zone_type != "Unstructured"` before entering this conditional branch.
        if self.zone_type != "Unstructured":
            emsg = f"Currently only support unstructured zone type, but {self.zone_type} found"
            logger.warning(emsg)
            return False

        # Checks `self.get_mesh_coords().shape[0] != self.zone_info["VertexSize"]` before entering this conditional branch.
        if self.get_mesh_coords().shape[0] != self.zone_info["VertexSize"]:
            emsg = "Mesh coordinates dimension not matching"
            logger.warning(emsg)
            return False

        # Stores the `sol_data` for use by the remaining steps in this flow.
        sol_data = self.get_solution_data()
        if sol_data is None:
            emsg = "No solution data found"
            logger.warning(emsg)
            # Returns `False` to the caller for the next stage.
            return False
        else:
            # check the dimension of the solution data
            for comp_name, data in sol_data.items():
                if data.shape[0] != self.zone_info["VertexSize"]:
                    # Derives the `emsg` from the current values for downstream processing.
                    emsg = f"Solution data dimension not matching for {comp_name}, " +\
                            "check if the solution is stored at vertices"
                    logger.warning(emsg)
                    return False

        return True


# Defines `main` for main logic.
def main():
    pass


if __name__ == "__main__":
    main()
