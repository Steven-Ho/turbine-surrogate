"""CPU regressions for node-domain SUM autograd and hybrid case accounting."""
import json
import os
import subprocess
import unittest
import torch
from tests.test_fann_ddp import DDPTests
from tests.test_fann_formal import ROOT


class GeoDomainTests(unittest.TestCase):
    def setUp(self):
        self.case=DDPTests();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.case.env.update(CUDA_VISIBLE_DEVICES='',FANN_INTEROP_THREADS='1',
            TURBINE_GEO_ACTIVATION_OFFLOAD='0',TURBINE_GEO_NODE_CHUNK='5',
            TURBINE_GEO_QUERY_BACKEND='kdtree',TURBINE_GEO_REFERENCE_MAX_POINTS='0',
            TURBINE_GEO_CACHE_DIR=str(self.case.root/'neighbors'))

    def run_mode(self,root,domain=1,extra=()):
        command=self.case.command('GeoTransolver',root,['--domain-size',str(domain),*extra])
        if domain>1:command[command.index('--nproc_per_node=2')]='--nproc_per_node=4'
        result=subprocess.run(command,cwd=ROOT,env=self.case.env,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return torch.load(root/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)

    def test_global_gradient_matches_two_complete_cases(self):
        a=self.run_mode(self.case.root/'native',extra=['--max-updates','1'])
        b=self.run_mode(self.case.root/'domain',domain=2,extra=['--max-updates','1'])
        self.assertEqual(b['distributed_state']['global_batch_cases'],2)
        self.assertEqual(b['pending_elements'],a['pending_elements'])
        for key,state in a['optimizer_state_dict']['state'].items():
            for field in ['exp_avg','exp_avg_sq']:
                torch.testing.assert_close(state[field],b['optimizer_state_dict']['state'][key][field],rtol=2e-3,atol=2e-7,msg=lambda m:f'{key}/{field}: {m}')
        for key,value in a['state_dict'].items():
            torch.testing.assert_close(value,b['state_dict'][key],rtol=2e-3,atol=3e-6,msg=lambda m:f'{key}: {m}')

    def test_tail_validation_and_exact_resume(self):
        self.case.env['FANN_MAX_EPOCHS_PER_RUN']='1'
        root=self.case.root/'resume'
        a=self.run_mode(root,domain=2)
        self.assertEqual(a['completed_epochs'],1);self.assertEqual(a['global_step'],2)
        self.assertEqual(a['pending_elements'],0)
        self.case.env.pop('FANN_MAX_EPOCHS_PER_RUN')
        resumed=self.run_mode(root,domain=2,extra=['--resume'])
        fullroot=self.case.root/'full';full=self.run_mode(fullroot,domain=2)
        self.assertEqual(resumed['history'],full['history'])
        for key,value in full['state_dict'].items():
            torch.testing.assert_close(value,resumed['state_dict'][key],rtol=0,atol=0)
        report=json.loads((root/'GeoTransolver/report.json').read_text())
        self.assertEqual(report['test']['cases'],1)
        self.assertEqual(report['test']['nodes'],8)

    def test_fixed_batch_scaling_controller(self):
        import sys
        output=self.case.root/'scaling'
        command=[sys.executable,'scripts/run_fann_geo_scaling.py','--splits',str(self.case.split),
            '--output',str(output),'--gpu-counts','1','4','--domain-sizes','1','4',
            '--warmup-cases','1','--measure-cases','1','--device','cpu','--backend','gloo','--threads','1']
        result=subprocess.run(command,cwd=ROOT,env=self.case.env,capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        report=json.loads((output/'report.json').read_text())
        self.assertEqual(report['state'],'complete')
        for row in report['results']:
            self.assertEqual(row['data_replicas'],1)
            saved=torch.load(output/f'gpus_{row["gpus"]}/GeoTransolver/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['global_step'],2);self.assertEqual(saved['position'],2)
            self.assertEqual(saved['distributed_state']['global_batch_cases'],1)
