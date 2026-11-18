import unittest
import numpy as np
from scipy.spatial import cKDTree
from types import SimpleNamespace
from tfai.processors.field.image_voxel_processor import ImageVoxelProcessor, _compute_voxel_mask

class SurfaceUNetTests(unittest.TestCase):
    def test_curved_surface_grid_band_covers_original_nodes(self):
        x, y = np.meshgrid(np.linspace(0, 1, 31), np.linspace(0, 1, 31), indexing="ij")
        nodes = np.column_stack([x.ravel(), y.ravel(), (.2 * x * x).ravel()])
        grid = np.stack(np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.linspace(0, .2, 4), indexing="ij"),axis=-1).reshape(-1,3)
        mask = _compute_voxel_mask(grid, nodes, [], show_progress=False, mesh_dimension=2)
        self.assertGreater(mask.sum(), 20)
        spacing = np.array([1/9, 1/9, .2/3])
        distance, _ = cKDTree(grid[mask == 1]).query(nodes)
        self.assertLessEqual(distance.max(), .5 * np.linalg.norm(spacing) + 1e-12)

    def test_single_sample_and_batch_restore_pressure_units_identically(self):
        processor = SimpleNamespace(mode="steady")
        processor.denormalize_predictions = lambda v,n: ImageVoxelProcessor.denormalize_predictions(processor,v,n)
        stats = {"norm_type": "min_max", "y": {"min": [100000.], "max": [200000.]}}
        values = np.full((1, 3, 4, 5), .5)
        single = processor.denormalize_predictions({7: values}, stats)[7]
        batch = processor.denormalize_predictions(values[None], stats)[0]
        np.testing.assert_array_equal(single, batch)
        np.testing.assert_allclose(single, 150000.)

    def test_invalid_voxels_do_not_keep_raw_parameter_units(self):
        import torch
        processor = ImageVoxelProcessor({"task": {"mode": "steady", "is_geometry_varying": False},
                                         "processor": {"base_resolution": 50}})
        values = torch.zeros(5, 2, 2, 2)
        values[3] = 50000.
        values[4, 0, 0, 0] = 1.
        stats = {"geometry": {"coord_min": [0,0,0], "coord_max": [1,1,1]},
                 "x": {"mask_channel": 4, "feature_channels": [3], "min": [0.],
                       "max": [100000.], "mean": [50000.], "std": [10000.]}}
        result = processor.normalize_processed_sample({"x": values}, stats, "min_max", normalize_y=False)
        self.assertAlmostEqual(result["x"][3,0,0,0].item(), .5)
        self.assertTrue(torch.all(result["x"][3][values[4] == 0] == 0))
        self.assertEqual(values[3,0,0,1].item(), 50000.)
