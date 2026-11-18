import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).resolve().parents[1] / "scripts/compare_models.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)

class OriginalNodeMetricTests(unittest.TestCase):
    def test_perfect_original_node_predictions(self):
        truth = np.array([[1., 2., 3.], [2., 3., 4.]])
        result = comparison.metrics(truth, truth, .001)
        self.assertEqual(result["r2"], 1.0)
        self.assertEqual(result["rmse_pa"], 0.0)
        self.assertEqual(result["mape_percent"], 0.0)
        self.assertEqual(result["node_count"], 6)

    def test_global_weighting_and_shared_mape_mask(self):
        result = comparison.metrics(np.array([0., 2., 4.]), np.array([1., 3., 5.]), .1)
        self.assertAlmostEqual(result["rmse_pa"], 1.0)
        self.assertAlmostEqual(result["mae_pa"], 1.0)
        self.assertAlmostEqual(result["mape_percent"], 37.5)
        self.assertEqual(result["mape_valid_count"], 2)
        self.assertAlmostEqual(result["r2"], .625)

    def test_invalid_predictions_do_not_silently_improve_scores(self):
        with self.assertRaises(ValueError):
            comparison.metrics(np.array([1., 2.]), np.array([1., np.nan]), .001)

    def test_training_duration_in_seconds_minutes_and_hours(self):
        self.assertEqual(comparison.training_loop_seconds("模型训练耗时: 55.44 秒"), 55.44)
        self.assertEqual(comparison.training_loop_seconds("模型训练耗时: 5 分 43.24 秒"), 343.24)
        self.assertEqual(comparison.training_loop_seconds("模型训练耗时: 1 小时 2 分 3.5 秒"), 3723.5)
