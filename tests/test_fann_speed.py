import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial import cKDTree
from tfai.benchmarks.fann import FannAdapter,Normalizer
from tfai.benchmarks.fann_unet_cache import nearest_mapping
from tests.test_fann_bounded_graphs import lattice


class SpeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);torch.set_num_threads(2)
    def test_nearest_mapping_exact_reuse_and_geometry_grid_invalidation(self):
        xyz=np.random.default_rng(4).normal(size=(200,3)).astype(np.float32)
        positions=xyz[:23]+.01
        expected=cKDTree(xyz).query(positions)
        first=nearest_mapping(xyz,positions,self.root)
        for a,b in zip(first,expected):np.testing.assert_array_equal(a,b)
        with patch('tfai.benchmarks.fann_unet_cache.cKDTree',side_effect=AssertionError('Cache should avoid rebuilding KDTree')):
            second=nearest_mapping(xyz,positions,self.root)
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
        changed=xyz.copy();changed[0]+=1
        nearest_mapping(changed,positions,self.root)
        nearest_mapping(xyz,positions[:17],self.root)
        self.assertEqual(len(list(self.root.glob('*.npz'))),3)
    def test_cached_unet_input_predictions_and_parameter_gradients(self):
        s=lattice();norm=Normalizer([s]);torch.manual_seed(7)
        dense=FannAdapter('UNet',[s],norm);cached=copy.deepcopy(dense)
        cached.unet_mapping_cache=self.root
        graph=norm.graph(s,'cpu',with_edges=False)
        a=dense.prepare(graph,s['original_node_ids']);b=cached.prepare(graph,s['original_node_ids'])
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        expected=dense(a);actual=cached(b)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        target=norm.target(s,'cpu')
        (expected-target).square().mean().backward();(actual-target).square().mean().backward()
        for (key,p),(_,q) in zip(dense.named_parameters(),cached.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=key)
    def test_transolver_chunk_and_checkpoint_changes_preserve_global_gradient(self):
        s=lattice();norm=Normalizer([s]);torch.manual_seed(4)
        dense=FannAdapter('Transolver_3',[s],norm);dense.net.chunk_size=7
        graph=norm.graph(s,'cpu',with_edges=False);target=norm.target(s,'cpu')
        expected=dense(graph);(expected-target).square().mean().backward()
        for checkpoint in [True,False]:
            other=copy.deepcopy(dense);other.zero_grad(set_to_none=True)
            other.net.chunk_size=23;other.net.use_checkpoint=checkpoint
            actual=other(graph);torch.testing.assert_close(actual,expected,rtol=1e-4,atol=2e-6)
            (actual-target).square().mean().backward()
            for (key,p),(_,q) in zip(dense.named_parameters(),other.named_parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-3,atol=2e-6,msg=key)

if __name__=='__main__':unittest.main()
