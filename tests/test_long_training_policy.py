import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from run_long_training import extend_training
from report_long_training import paired_rmse_ratio_ci


class LongTrainingPolicyTests(unittest.TestCase):
    def history(self, losses):
        return {"val_loss":losses,"val_units":[(i+1)*800 for i in range(len(losses))]}

    def test_extend_only_completed_and_still_improving_stage(self):
        self.assertTrue(extend_training(self.history([1.]*160+[.95]*40)))
        self.assertFalse(extend_training(self.history([1.]*160+[.99]*40)))
        self.assertFalse(extend_training(self.history([1.]*160+[.5]*39)))
        self.assertFalse(extend_training(self.history([1.]*200),200,200))

    def test_extension_uses_global_best_not_an_arbitrary_bad_recent_epoch(self):
        self.assertFalse(extend_training(self.history([.5]+[1.]*159+[.6]*40)))

    def test_paired_bootstrap_preserves_case_matching(self):
        target=np.zeros((100,3))
        reference=np.arange(1.,101.)[:,None]*np.ones((1,3))
        low,high=paired_rmse_ratio_ci(.5*reference,reference,target)
        self.assertAlmostEqual(low,.5);self.assertAlmostEqual(high,.5)
        low,high=paired_rmse_ratio_ci(2*reference,reference,target)
        self.assertAlmostEqual(low,2.);self.assertAlmostEqual(high,2.)
