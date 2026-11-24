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


class SixNodeStabilityTests(unittest.TestCase):
    def test_submission_is_exclusive_48_gpu_with_independent_output(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);stub=root/'sbatch';trace=root/'record'
            stub.write_text('#!'+sys.executable+'\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ["SUBMIT_TRACE"]).write_text(json.dumps({"args":sys.argv[1:],"env":dict(os.environ)}))\nprint(12345)\n');stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],SUBMIT_TRACE=str(trace),FANN_STABILITY_RUN_ROOT='old_two_node')
            result=subprocess.run(['bash','scripts/submit_fann_geo_stability_6nodes.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(trace.read_text());args=report['args'];e=report['env']
            self.assertIn('--nodes=6',args);self.assertIn('--exclusive',args);self.assertIn('--gres=gpu:8',args)
            self.assertEqual(e['FANN_DDP_GPUS_PER_NODE'],'8');self.assertEqual(e['FANN_MAX_EPOCHS_PER_RUN'],'5')
            self.assertNotEqual(e['FANN_RUN_ROOT'],'old_two_node');self.assertEqual(e['FANN_AMP'],'bf16')
            self.assertIn('48_%A_%a.out',' '.join(args))

    def test_many_inactive_ranks_train_and_validate(self):
        case=DDPTests();case.setUp()
        try:
            env=dict(case.env,CUDA_VISIBLE_DEVICES='',FANN_INTEROP_THREADS='1',FANN_MAX_EPOCHS_PER_RUN='1',
                     TURBINE_GEO_ACTIVATION_OFFLOAD='0',TURBINE_GEO_CACHE_DIR=str(case.root/'neighbors'))
            output=case.root/'many_ranks';cmd=case.command('GeoTransolver',output,['--amp','bf16'])
            cmd[cmd.index('--nproc_per_node=2')]='--nproc_per_node=6'
            result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            folder=output/'GeoTransolver';status=json.loads((folder/'status.json').read_text())
            self.assertEqual(status['completed_epochs'],1);self.assertEqual(status['world_size'],6)
            self.assertEqual(status['stop_reason'],'epoch_budget');self.assertEqual(status['global_step'],1)
            saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['pending_elements'],0);self.assertEqual(len(saved['history']),1)
        finally:case.doCleanups()
