import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from tests.test_fann_ddp import DDPTests
from tests.test_fann_formal import ROOT


class StabilityTests(unittest.TestCase):
    def test_epoch_budget_includes_validation_and_resumes_exactly(self):
        case=DDPTests();case.setUp()
        try:
            case.env.update(CUDA_VISIBLE_DEVICES='',FANN_PROFILE_STEPS='0',TURBINE_GEO_ACTIVATION_OFFLOAD='0',
                            TURBINE_GEO_CACHE_DIR=str(case.root/'neighbors'))
            case.env.pop('FANN_MAX_EPOCHS_PER_RUN',None)
            full=case.root/'full';resume=case.root/'resume'
            case.run_job('GeoTransolver',full,['--amp','bf16'])
            case.env['FANN_MAX_EPOCHS_PER_RUN']='1'
            case.env['FANN_RUNTIME_SNAPSHOT_DIR']=str(case.root/'runtime')
            case.env.pop('FANN_COMM_DIAGNOSTICS_DIR',None)
            case.run_job('GeoTransolver',resume,['--amp','bf16'])
            folder=resume/'GeoTransolver';status=json.loads((folder/'status.json').read_text())
            self.assertEqual(status['completed_epochs'],1);self.assertEqual(status['stop_reason'],'epoch_budget')
            self.assertEqual(status['global_step'],2)
            self.assertEqual(len(json.loads((folder/'history.json').read_text())),1)
            self.assertTrue((folder/'best.pt').exists());self.assertFalse((folder/'predictions').exists())
            snapshots=list((case.root/'runtime').glob('runtime_*.json'));self.assertEqual(len(snapshots),4)
            for path in snapshots:
                info=json.loads(path.read_text());self.assertIn('cpu_affinity',info)
                self.assertNotIn('NCCL_DEBUG',info['environment'])
            case.env.pop('FANN_MAX_EPOCHS_PER_RUN')
            case.run_job('GeoTransolver',resume,['--amp','bf16','--resume'])
            a=torch.load(full/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
            b=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(a['history'],b['history'])
            for key,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
        finally:case.doCleanups()

    def test_submit_ignores_stale_experiment_settings(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);trace=root/'record';stub=root/'sbatch'
            stub.write_text('#!'+sys.executable+'\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ["SUBMIT_TRACE"]).write_text(json.dumps({"args":sys.argv[1:],"env":dict(os.environ)}))\nprint(12345)\n');stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],SUBMIT_TRACE=str(trace),
                FANN_MAX_UPDATES='20',FANN_INTEROP_THREADS='1',FANN_DDP_SAMPLER='geometry_affinity',
                FANN_COMM_DIAGNOSTICS_DIR='old',NCCL_DEBUG='INFO',MCCL_DEBUG='INFO',FANN_PROFILE_STEPS='5')
            result=subprocess.run(['bash','scripts/submit_fann_geo_stability.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(trace.read_text());e=data['env']
            for key in ['FANN_MAX_UPDATES','FANN_INTEROP_THREADS','FANN_COMM_DIAGNOSTICS_DIR','NCCL_DEBUG','MCCL_DEBUG']:
                self.assertNotIn(key,e)
            self.assertEqual(e['FANN_MAX_EPOCHS_PER_RUN'],'5');self.assertEqual(e['FANN_EPOCHS'],'400')
            self.assertEqual(e['FANN_PROFILE_STEPS'],'0');self.assertEqual(e['FANN_DDP_SAMPLER'],'random')
            self.assertIn('--exclusive',data['args']);self.assertIn('--nodes=2',data['args'])
