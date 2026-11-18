import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tfai.data.data_pipeline import DataPipeline


class TargetScaleTests(unittest.TestCase):
    def compute(self, target):
        processor = SimpleNamespace(config={"task": {"variables": ["Pressure"]}})
        with patch("tfai.data.data_pipeline.load_torch", return_value={"y": target}):
            return DataPipeline._compute_mape_target_scales(["sample.pt"], processor, {})

    def test_small_finite_absolute_targets(self):
        result = self.compute(torch.tensor([-4., -2., 0., 2., 4., float("nan"), float("inf")]))
        self.assertEqual(result, {"quantile": 0.95, "by_variable": {"Pressure": 4.0}})

    def test_full_dataset_size_exceeds_torch_quantile_limit(self):
        target = torch.full((2**24 + 1,), 7.0)
        target[:100] = 0.0
        self.assertEqual(self.compute(target)["by_variable"]["Pressure"], 7.0)


if __name__ == "__main__":
    unittest.main()
