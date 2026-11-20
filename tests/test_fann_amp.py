"""CPU-only coverage for Geo BF16 complete-case DDP and precision identity."""
import os
import unittest
import torch
from tests.test_fann_ddp import DDPTests
from tfai.benchmarks.fann_training import execution_config,assert_resume_config


class AmpTests(unittest.TestCase):
    def test_precision_identity(self):
        self.assertEqual(execution_config('GeoTransolver',{}),execution_config('GeoTransolver',{'amp':'off'}))
        with self.assertRaises(ValueError):assert_resume_config({}, {'amp':'bf16'})
        with self.assertRaises(ValueError):execution_config('UNet',{'amp':'bf16'})

    def test_bf16_ddp_tail_resume(self):
        case=DDPTests();case.setUp()
        try:
            case.env.update(CUDA_VISIBLE_DEVICES='',TURBINE_GEO_NODE_CHUNK='7',
                            TURBINE_GEO_ACTIVATION_OFFLOAD='0',TURBINE_GEO_REFERENCE_MAX_POINTS='0',
                            TURBINE_GEO_CACHE_DIR=str(case.root/'neighbors'))
            root=case.root/'amp'
            first=case.run_job('GeoTransolver',root,['--amp','bf16','--max-updates','1'])
            self.assertIn('amp=bf16',first.stdout)
            c=torch.load(root/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(c['config']['amp'],'bf16')
            self.assertTrue(all(v.dtype==torch.float32 for v in c['state_dict'].values() if v.is_floating_point()))
            self.assertTrue(all(torch.isfinite(v).all() for v in c['state_dict'].values() if v.is_floating_point()))
            case.run_job('GeoTransolver',root,['--amp','bf16','--resume'])
            c=torch.load(root/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(c['global_step'],4)
            self.assertEqual(c['completed_epochs'],2)
        finally:case.doCleanups()
