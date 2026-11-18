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
