import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from tfai.prediction.adapters.base_adapter import BasePredictAdapter

class WritebackGeometryTests(unittest.TestCase):
    def test_explicit_sample_mesh_overrides_fixed_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.vtu"; source.touch()
            adapter = SimpleNamespace(is_geometry_varying=False,
                processor=SimpleNamespace(requires_fixed_geometry_predict_writeback_reference=lambda: True),
                _resolve_writeback_template_context=lambda source, sample_id: {"template_file": source},
                _augment_motion_writeback_context=lambda sample_id, context: context,
                _sample_writeback_contexts={})
            actual = BasePredictAdapter._get_sample_file(adapter, 0, 7, [str(source)])
            self.assertEqual(actual, str(source))

    def test_inverse_interpolation_cache_is_keyed_by_source_geometry(self):
        import numpy as np
        from tfai.prediction.adapters.image_voxel_adapter import ImageVoxelPredictAdapter
        first = np.zeros((2, 3)); second = np.ones((2, 3))
        def read(source, **kwargs):
            coords = first if source == "a.vtu" else second
            return SimpleNamespace(point_or_cell="point", get_data=lambda: {"PointCoordinates": coords})
        adapter = SimpleNamespace(is_geometry_varying=False, _cached_original_coords=None,
            _cached_original_source=None, _cached_interpolation_plan="old", reader=SimpleNamespace(read=read),
            _resolve_writeback_template_source=lambda source, sample_id: source,
            config={"task": {"point_or_cell": "point"}})
        a = ImageVoxelPredictAdapter._get_original_coords(adapter, 1, "a.vtu")
        adapter._cached_interpolation_plan = "a-plan"
        b = ImageVoxelPredictAdapter._get_original_coords(adapter, 2, "b.vtu")
        np.testing.assert_array_equal(a, first)
        np.testing.assert_array_equal(b, second)
        self.assertIsNone(adapter._cached_interpolation_plan)
        adapter._cached_interpolation_plan = "b-plan"
        ImageVoxelPredictAdapter._get_original_coords(adapter, 2, "b.vtu")
        self.assertEqual(adapter._cached_interpolation_plan, "b-plan")
