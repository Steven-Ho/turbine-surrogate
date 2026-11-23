import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import torch
from tests.test_fann_ddp import DDPTests
from tests.test_fann_formal import ROOT


class ScalingTests(unittest.TestCase):
    def test_cpu_suite_fixed_workload_and_report(self):
        fixture=DDPTests();fixture.setUp()
        try:
            # Four unique training cases allow 2 warm-up + 2 measured cases.
            with np.load(fixture.root/'case_1.npz') as data:sample={key:data[key] for key in data.files}
            sample['coordinates']=sample['coordinates']+3
            sample['geometry_id']=np.array([5]);extra=fixture.root/'case_5.npz'
            np.savez_compressed(extra,**sample)
            split=json.loads(fixture.split.read_text());split['train'].append(str(extra));fixture.split.write_text(json.dumps(split))
            env=dict(fixture.env,CUDA_VISIBLE_DEVICES='',FANN_DDP_SAMPLER='geometry_affinity',FANN_PROFILE_STEPS='20',
                TURBINE_GEO_MEMORY_CACHE_MB='4096',TURBINE_GEO_CACHE_DIR=str(fixture.root/'neighbors'))
            output=fixture.root/'scaling'
            result=subprocess.run([sys.executable,'scripts/run_fann_geo_scaling.py','--splits',str(fixture.split),
                '--output',str(output),'--gpu-counts','1','2','--warmup-cases','2','--measure-cases','2',
                '--device','cpu','--backend','gloo','--threads','1'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['state'],'complete');self.assertEqual(len(report['results']),2)
            self.assertEqual(report['results'][0]['speedup'],1.)
            for row in report['results']:
                self.assertEqual(row['measured_cases'],2)
                self.assertGreater(row['cases_per_second'],0)
                p=row['gpus'];records=json.loads((output/f'gpus_{p}/steps.json').read_text())
                self.assertEqual(len(records),4//p)
                self.assertEqual(sum(r['cases'] for r in records if r['measured']),2)
                c=torch.load(output/f'gpus_{p}/GeoTransolver/last.pt',map_location='cpu',weights_only=False)
                self.assertNotIn('sampler',c['config']);self.assertEqual(c['config']['amp'],'bf16')
                self.assertFalse(c['execution_runtime']['geo_activation_offload'])
            self.assertTrue((output/'summary.csv').is_file())
        finally:fixture.doCleanups()
