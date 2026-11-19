"""Small CPU-only correctness checks; no training or GPU benchmark."""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('_geo_spatial_unit')
package.__path__ = [str(ROOT / 'tfai/models/_vendor/geotransolver2')]
sys.modules[package.__name__] = package
BQ = importlib.import_module('_geo_spatial_unit.ball_query').BQWarp


class SpatialCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'TURBINE_GEO_CACHE_DIR': self.temp.name,
                                          'TURBINE_GEO_MEMORY_CACHE': '4'})
        self.env.start()

    def tearDown(self):
        self.env.stop(); self.temp.cleanup()

    def compare(self, points, queries, radius, k, chunk=8):
        reference = BQ(radius, k, chunk, backend='torch')
        fast = BQ(radius, k, chunk, backend='kdtree')
        a = points.clone().requires_grad_(); b = points.clone().requires_grad_()
        ia, ya = reference(a.unsqueeze(0), queries.unsqueeze(0))
        ib, yb = fast(b.unsqueeze(0), queries.unsqueeze(0))
        torch.testing.assert_close(ia, ib, rtol=0, atol=0)
        torch.testing.assert_close(ya, yb, rtol=0, atol=0)
        ya.square().sum().backward(); yb.square().sum().backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertEqual(fast.state_dict(), {})
        return fast

    def test_random_float32_and_float64(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(3)
            p = torch.randn(71, 3, dtype=dtype)
            q = torch.randn(29, 3, dtype=dtype)
            self.compare(p, q, .9, 7)

    def test_ties_boundaries_duplicates_and_padding(self):
        p = torch.tensor([[0.,0,0],[.25,0,0],[-.25,0,0],[0,.25,0],[0,0,0]])
        q = torch.tensor([[0.,0,0],[1.,0,0]])
        for k in (1, 3, 8):
            fast = self.compare(p, q, .25, k, 1)
            self.assertGreater(fast.reference_chunks, 0)

    def test_full_domain_ties_never_call_dense_cdist(self):
        torch.manual_seed(23)
        points=torch.randn(40001,3);points[:4]=0
        points.requires_grad_();queries=torch.tensor([[0.,0,0],[50.,50,50]])
        layer=BQ(.1,3,backend='kdtree')
        with patch('torch.cdist',side_effect=AssertionError('Full-domain dense fallback')):
            indices,neighbors=layer(points[None],queries[None])
        self.assertGreater(layer.large_query_ambiguity_rows,0)
        self.assertEqual(layer.reference_chunks,0)
        self.assertEqual(tuple(neighbors.shape),(1,2,3,3))
        torch.testing.assert_close(neighbors,torch.zeros_like(neighbors))
        neighbors.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertAlmostEqual(float(points.grad.sum()),9.)

    def test_dense_reference_rejects_oversized_source_before_kernel(self):
        with patch.dict(os.environ,{'TURBINE_GEO_REFERENCE_MAX_POINTS':'4'}):
            layer=BQ(.5,2,backend='torch')
        with patch('torch.cdist',side_effect=AssertionError('Kernel should not run')):
            with self.assertRaisesRegex(ValueError,'Dense radius-query reference'):
                layer(torch.zeros(1,5,3),torch.zeros(1,2,3))

    def test_tree_query_temporaries_are_chunk_bounded(self):
        module=importlib.import_module('_geo_spatial_unit.spatial_neighbors')
        original=module.cKDTree;calls=[]
        class TreeSpy:
            def __init__(self,*args,**kwargs):self.tree=original(*args,**kwargs)
            def query(self,queries,**kwargs):
                calls.append(len(queries));return self.tree.query(queries,**kwargs)
        p=torch.randn(31,3).numpy();q=torch.randn(17,3).numpy()
        baseline=module.tree_candidates(p,q,1.,4)
        with patch.dict(os.environ,{'TURBINE_GEO_TREE_QUERY_CHUNK':'3'}),patch.object(module,'cKDTree',TreeSpy):
            bounded=module.tree_candidates(p,q,1.,4)
        self.assertEqual(max(calls),3)
        import numpy as np
        np.testing.assert_array_equal(baseline[0],bounded[0])
        np.testing.assert_array_equal(baseline[1],bounded[1])

    def test_memory_cache_has_a_byte_budget(self):
        import numpy as np
        module=importlib.import_module('_geo_spatial_unit.spatial_neighbors')
        with patch.dict(os.environ,{'TURBINE_GEO_MEMORY_CACHE_MB':'0'}):
            cache=module.NeighborIndexCache(None,32)
        cache.put('key',np.zeros((100,4),dtype=np.int32))
        self.assertEqual(len(cache.memory),0);self.assertEqual(cache.memory_bytes,0)

    def test_reference_policy_changes_cache_identity(self):
        x=torch.zeros(1,9,3)
        with patch.dict(os.environ,{'TURBINE_GEO_REFERENCE_MAX_POINTS':'32'}):
            original=BQ(.5,3,backend='kdtree');original(x,x)
        with patch.dict(os.environ,{'TURBINE_GEO_REFERENCE_MAX_POINTS':'0'}):
            bounded=BQ(.5,3,backend='kdtree');bounded(x,x)
        self.assertEqual(bounded.disk_cache_hits,0)
        self.assertEqual(bounded.tree_builds,1)
        self.assertEqual(bounded.reference_chunks,0)

    def test_A_B_A_uses_memory_without_rebuild(self):
        torch.manual_seed(5)
        a = torch.randn(1, 19, 3); b = a + .01
        layer = BQ(.5, 4, backend='kdtree')
        original = layer(a, a)
        layer(b, b)
        actual = layer(a.clone(), a.clone())
        self.assertEqual(layer.tree_builds, 2)
        self.assertEqual(layer.memory_cache_hits, 1)
        torch.testing.assert_close(original[0], actual[0])
        torch.testing.assert_close(original[1], actual[1])

    def test_fresh_module_uses_persistent_cache(self):
        torch.manual_seed(7); x = torch.randn(1, 13, 3)
        first = BQ(1., 4, backend='kdtree'); expected = first(x, x)
        second = BQ(1., 4, backend='kdtree'); actual = second(x, x)
        self.assertEqual(second.disk_cache_hits, 1)
        self.assertEqual(second.tree_builds, 0)
        torch.testing.assert_close(actual[0], expected[0])
        self.assertFalse(list(Path(self.temp.name).rglob('*.tmp')))

    def test_corrupt_cache_rebuilds_and_coordinate_change_invalidates(self):
        torch.manual_seed(11); x = torch.randn(1, 13, 3)
        first = BQ(1., 4, backend='kdtree'); first(x, x)
        path = next(Path(self.temp.name).rglob('*.npz')); path.write_bytes(b'truncated')
        second = BQ(1., 4, backend='kdtree'); second(x, x)
        self.assertEqual(second.index_cache.corruptions, 1)
        self.assertEqual(second.tree_builds, 1)
        changed = x.clone(); changed[0,0,0] += .001
        third = BQ(1., 4, backend='kdtree'); third(changed, changed)
        self.assertEqual(third.tree_builds, 1)
        self.assertEqual(third.disk_cache_hits, 0)


if __name__ == '__main__':
    unittest.main()
