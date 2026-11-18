"""
This file contains the IO class for CGNS files.
Currently, only the hdf5 implementation is supported.
ADF format can be read by first converting it to hdf5 using the adf2hdf5 tool.
The CGNS file is read using the h5py library, closely following
the implementation of the pyCGNS library.

"""
import logging
import sys

import h5py  # used for reading and writing hdf5 files
# Imports `numpy` for reuse in this module.
import numpy as np

from .tf_cgns import CGNSData, Node

# get the module-specific logger inherited from the root logger
logger = logging.getLogger(__name__)


class CGNSReader(object):
    """ This class reads the CGNS file and returns the data in a dictionary. """

    # Initialize the reader class.
    def __init__(self, filename):
        """
        Initialize the reader class.
        The filename is the path to the CGNS file.
        """
        self._filename = filename

    @property
    def filename(self):
        # Returns `self._filename` to the caller for the next stage.
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _load_header(group):
        """
        Load the header information from the hdf5 file.
        """
        header = dict()
        format = group[" format"][()]
        # Selects the required item from `group[" hdf5version"]` as the `version`.
        version = group[" hdf5version"][()]
        # convert int8 to byte string
        header[" format"] = np.bytes_(format)
        header[" hdf5version"] = np.bytes_(version)

        return header

    # Continues the surrounding statement or collection definition.
    @staticmethod
    def _load_groups(group, load_children=True):
        """
        Load the groups in the hdf5 file as a tree structure.
        """
        # create the node
        # node = Node(str(group.attrs["name"]), str(group.attrs["label"]), None, None)
        # print(group.attrs["name"], group.attrs["label"], group.attrs.get("flags", None))
        node = Node(group.attrs["name"].decode("utf-8"),
                    group.attrs["label"].decode("ascii"),  # name and label should always exist
                    # Completes the multi-line `group.attrs.get` arguments or elements.
                    group.attrs.get("flags", None),        # flag may not exist
                    None, None)

        # if the group has data, load it
        if " data" in group:  # the extra space is intentional for some reason
            data_from = group[" data"]
            # Stores the `n` for use by the remaining steps in this flow.
            n = data_from.size
            # load the data
            node.data = np.copy(data_from[()].transpose(), order="F")
            # if the data is of type C1, convert it to a string
            if group.attrs["type"] == b"C1":
                # node.data = np.vectorize(chr)(node.data).astype(np.dtype("c"))
                node.shape = node.data.shape
                # Stores the current data for use by the remaining steps in this flow.
                node.data = np.bytes_(node.data)  # stores the bytes for easier manipulation
        # if the children should be loaded, load them
        if load_children:
            for child_name in group.keys():
                if child_name[0] != " ":
                    # Stores the `child` for use by the remaining steps in this flow.
                    child = CGNSReader._load_groups(group[child_name])
                    node.add_child(child)

        return node

    def read(self, load_children=True):
        """
        Read the CGNS file and return the data in a tree structure.
        """
        # open the file
        # Scopes resource use so the context manager can release it on exit.
        with h5py.File(self.filename, "r") as f:

            # load the header
            header = CGNSReader._load_header(f)
            root = CGNSReader._load_groups(f, load_children)

        return CGNSData(header, root)


# This class writes the data to the CGNS file.
class CGNSWriter(object):
    """ This class writes the data to the CGNS file. """

    def __init__(self, filename):
        """
        Initialize the writer class.
        The filename is the path to the CGNS file.
        """
        self._filename = filename

    @property
    # Defines `filename` for filename logic.
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        # Stores the `_filename` on the instance so other methods can reuse it.
        self._filename = filename

    @staticmethod
    def _write_header(group, header: dict) -> None:
        """
        Write the header information to the hdf5 file as dataset.
        """
        # convert np.bytes_ to int8
        format = np.frombuffer(header[" format"], dtype=np.int8)
        # Stores the `version` for use by the remaining steps in this flow.
        version = np.frombuffer(header[" hdf5version"], dtype=np.int8)

        group.create_dataset(" format", data=format)
        group.create_dataset(" hdf5version", data=version)

    @staticmethod
    # Get the default header information.
    def _get_default_header() -> dict:
        """
        Get the default header information.
        """
        header = dict()
        header[" format"] = np.bytes_(b"NATIVE\x00")
        header[" hdf5version"] = np.bytes_(b"HDF5 Version 1.8.15".ljust(33, b"\x00"))

        # Returns `header` to the caller for the next stage.
        return header

    @staticmethod
    def _write_groups(group, node: Node) -> None:
        """
        Write the tree structure to the hdf5 file.
        """

        def _str_to_const_size_bytes(s, n):
            """
            Convert ascii string to an numpy byte array of constant size n.
            """
            # Returns `np.bytes_(bytes(s, "utf-8").ljust(n, b"\x00"))` to the caller for the next stage.
            return np.bytes_(bytes(s, "utf-8").ljust(n, b"\x00"))

        # start writing the node to the group
        # if node.label == "CGNSTree_t":
        #    group.attrs["name"] = _str_to_const_size_bytes("HDF5 MotherNode", 33)
        #    group.attrs["label"] = _str_to_const_size_bytes("Root Node of HDF5 File", 33)
        # else:
        #    group.attrs["name"] = _str_to_const_size_bytes(node.name, 33)
        #    group.attrs["label"] = _str_to_const_size_bytes(node.label, 33)
        #    group.attrs["flag"] = np.array([0], dtype=np.int32)
        group.attrs["name"] = _str_to_const_size_bytes(node.name, 33)
        group.attrs["label"] = _str_to_const_size_bytes(node.label, 33)
        if node.flags is not None:
            # Stores the `attrs['flags']` for use by the remaining steps in this flow.
            group.attrs["flags"] = node.flags  # np.array([node.flags], dtype=np.int32)

        # determine the data type
        if node.data is None:
            group.attrs["type"] = _str_to_const_size_bytes("MT", 3)
        else:
            # Stores the `ndata_type` for use by the remaining steps in this flow.
            ndata_type = node.data.dtype
            if ndata_type == np.float32:
                group.attrs["type"] = _str_to_const_size_bytes("R4", 3)
            elif ndata_type == np.float64:
                # Stores the `attrs['type']` for use by the remaining steps in this flow.
                group.attrs["type"] = _str_to_const_size_bytes("R8", 3)
            elif ndata_type == np.int32:
                group.attrs["type"] = _str_to_const_size_bytes("I4", 3)
            elif ndata_type == np.int64:
                # Stores the `attrs['type']` for use by the remaining steps in this flow.
                group.attrs["type"] = _str_to_const_size_bytes("I8", 3)
            # check if it's a byte string
            elif ndata_type.kind == "S":
                group.attrs["type"] = _str_to_const_size_bytes("C1", 3)
            else:
                # Stores the `emsg` for use by the remaining steps in this flow.
                emsg = f"Data type {ndata_type} not supported at node {node.name}."
                logger.error(emsg)
                raise NotImplementedError(emsg)

            # write the data
            data = node.data
            # if ndata_type == np.dtype("|S1"):
            #    idx1 = np.where(data != b"")
            #    idx2 = np.where(data == b"")
            #
            #    data[idx1] = np.vectorize(ord)(data[idx1])
            #    data[idx2] = np.zeros(idx2[0].size, dtype=np.int8)
            #    data = data.astype(np.int8)
            # convert byte string to int8
            # Checks `ndata_type.kind == "S"` before entering this conditional branch.
            if ndata_type.kind == "S":
                data = np.frombuffer(data, dtype=np.int8).reshape(node.shape)

            # transpose the data
            data = data.transpose()
            group.create_dataset(" data", data=data)

        # Checks `node.children is not None` before entering this conditional branch.
        if node.children is not None:
            for cname, child in node.children.items():
                cgroup = group.create_group(cname)
                CGNSWriter._write_groups(cgroup, child)

    # Write the data to the CGNS file.
    def write(self, data: CGNSData, use_default_header=False) -> None:
        """ Write the data to the CGNS file. """
        # open the file
        with h5py.File(self._filename, "w") as f:
            # write the data
            # f.create_dataset(
            #    " format", data=np.array([78, 65, 84, 73, 86, 69, 0], dtype=np.int8)
            # )
            # f.create_dataset(
            #    " hdf5version",
            #    data=np.array(
            #        [
            #            72,
            #            68,
            #            70,
            #            53,
            #            32,
            #            86,
            #            101,
            #            114,
            #            115,
            #            105,
            #            111,
            #            110,
            #            32,
            #            49,
            #            46,
            #            56,
            #            46,
            #            49,
            #            53,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #            0,
            #        ],
            #        dtype=np.int8,
            #    ),
            # )

            header = data.header

            if use_default_header:
                # Stores the `header` for use by the remaining steps in this flow.
                header = CGNSWriter._get_default_header()

            # write the header
            CGNSWriter._write_header(f, header)
            CGNSWriter._write_groups(f, data.root)


def main():
    # create the reader
    # Stores the data reader for use by the remaining steps in this flow.
    reader = CGNSReader("h5.cgns")
    # load the data
    data = reader.read()
    # get the data
    writer = CGNSWriter("h5_out.cgns")
    writer.write(data)


# Checks `__name__ == "__main__"` before entering this conditional branch.
if __name__ == "__main__":
    main()
