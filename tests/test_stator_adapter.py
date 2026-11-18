import importlib.util
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np

spec=importlib.util.spec_from_file_location("stator",Path(__file__).resolve().parents[1]/"scripts/prepare_stator_case.py")
stator=importlib.util.module_from_spec(spec);spec.loader.exec_module(stator)

class StatorAdapterTests(unittest.TestCase):
    def test_solver_boundary_semantics_and_arrays_are_preserved(self):
        text="&boundary\nname='inlet'\nkind='stagnation'\np=2.8D6\ntotalT=.true.\naks=1,1\n/\n"
        values=stator.namelists(text)[0]["values"]
        self.assertEqual(values["kind"],"stagnation")
        self.assertEqual(values["p"],2800000.)
        self.assertTrue(values["totalT"])
        self.assertEqual(values["aks"],[1,1])

    def test_mixed_boundary_faces_keep_exact_node_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory)/"test.h5","w") as f:
                section=f.create_group("boundary");section.create_dataset(" data",data=[20,0])
                section.create_dataset("ElementConnectivity/ data",data=[5,1,2,3,7,1,3,4,2])
                nodes,offsets=stator.boundary_faces(section)
                np.testing.assert_array_equal(nodes,[0,1,2,0,2,3,1])
                np.testing.assert_array_equal(offsets,[0,3,7])
