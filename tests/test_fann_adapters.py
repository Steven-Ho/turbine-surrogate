import numpy as np
import unittest
import torch
from tfai.benchmarks.fann import FannAdapter,Normalizer,FIELDS,load_sample


def sample():
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.float32)
    return dict(coordinates=xyz,hexahedra=np.arange(8).reshape(1,8),original_node_ids=np.arange(8),
                targets=np.arange(48,dtype=np.float32).reshape(8,6),operating_conditions=np.array([900,1e5],dtype=np.float32),output_names=np.array(FIELDS))


class FannTests(unittest.TestCase):
    def test_targets_never_enter_graph_or_pod_input(self):
        s=sample();norm=Normalizer([s]);g=norm.graph(s,'cpu')
        other={**s,'targets':s['targets']+10000}
        torch.testing.assert_close(g.x,norm.graph(other,'cpu').x)
        model=FannAdapter('POD',[s],norm)
        with self.assertRaisesRegex(ValueError,'ordering'):
            model.prepare(g,s['original_node_ids'][::-1].copy())
        assert model(model.prepare(g,s['original_node_ids'])).shape==(8,6)


    def test_hex_edges_use_true_topology(self):
        g=Normalizer([sample()]).graph(sample(),'cpu')
        assert g.edge_index.shape==(2,24)
        assert torch.allclose(g.edge_attr[:,-1],torch.ones(24))


    def test_invalid_connectivity_rejected(self):
        import tempfile
        from pathlib import Path
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        tmp_path=Path(directory.name)
        s=sample();s['hexahedra'][0,0]=8
        p=tmp_path/'bad.npz';np.savez(p,**s)
        with self.assertRaisesRegex(ValueError,'connectivity'):load_sample(p)

if __name__ == "__main__": unittest.main()
