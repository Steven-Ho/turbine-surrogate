import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tfai.benchmarks.fann import FannAdapter,Normalizer
from tfai.benchmarks.fann_training import inspect_splits,store_for,prepare_input,train_one,evaluate_frozen
from tests import test_fann_ddp as ddp_helpers
from tests.test_fann_formal import ROOT,config

MODELS=['GCN','Graph_UNet','Graph_Transformer','GeoTransolver']


class GraphDDPTests(unittest.TestCase):
    def setUp(self):
        ddp_helpers.DDPTests.setUp(self)
        self.env.update(TURBINE_FANN_GRAPH_CACHE=str(self.root/'graph_cache'),
                        TURBINE_GEO_QUERY_BACKEND='kdtree',TURBINE_GEO_REFERENCE_MAX_POINTS='0',
                        TURBINE_GEO_NODE_CHUNK='7',TURBINE_GEO_ACTIVATION_OFFLOAD='1',
                        TURBINE_GEO_CACHE_DIR=str(self.root/'neighbors'))
    def command(self,name,output,extra=(),backend='gloo',device='cpu'):
        return ddp_helpers.DDPTests.command(self,name,output,(),backend,device)+[
            '--graph-owner-nodes','4','--graph-edge-chunk','7','--graph-cache-mb','1',*extra]
    run_job=ddp_helpers.DDPTests.run_job
    def test_all_four_tail_coverage_resume_and_frozen_metrics(self):
        for name in MODELS:
            with self.subTest(model=name):
                root=self.root/name
                result=self.run_job(name,root,['--max-updates','2'])
                saved=torch.load(root/name/'last.pt',map_location='cpu',weights_only=False)
                self.assertEqual(saved['position'],3);self.assertEqual(saved['pending_elements'],(8+8+16)*6)
                self.assertEqual(saved['global_step'],2);self.assertIn('batch_cases=1',result.stdout)
                self.assertIn('DDP capacity per rank',result.stdout)
                self.run_job(name,root,['--resume'])
                report=json.loads((root/name/'report.json').read_text())
                self.assertEqual(report['optimizer_steps'],4);self.assertEqual(report['test']['nodes'],8)
                with patch.dict(os.environ,self.env):
                    frozen=evaluate_frozen(root/name/'best.pt',inspect_splits(self.split),'test','cpu',self.root/(name+'_frozen'))
                np.testing.assert_allclose(report['test']['rmse'],frozen['rmse'],rtol=3e-5,atol=3e-5)
                engine='native_ddp' if name=='GeoTransolver' else 'case_allreduce'
                self.assertEqual(saved['execution_runtime']['gradient_sync'],engine)
    def test_all_four_first_step_matches_original_dense_global_objective(self):
        for name in MODELS:
            with self.subTest(model=name):
                root=self.root/(name+'_gradient');self.run_job(name,root,['--max-updates','1'])
                saved=torch.load(root/name/'last.pt',map_location='cpu',weights_only=False)
                manifest=inspect_splits(self.split);norm=Normalizer(store_for(manifest,'train'))
                torch.manual_seed(42)
                model=FannAdapter(name,[],norm)
                optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
                store=store_for(manifest,'train')
                order=torch.randperm(3,generator=torch.Generator().manual_seed(42)).tolist()[:2]
                samples=[store[i] for i in order];denominator=sum(s['targets'].size for s in samples)
                for sample in samples:
                    prediction=model(prepare_input(model,norm,sample,'cpu'))
                    ((prediction-norm.target(sample,'cpu')).square().sum()/denominator).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                for key,state in optimizer.state_dict()['state'].items():
                    torch.testing.assert_close(state['exp_avg'],saved['optimizer_state_dict']['state'][key]['exp_avg'],rtol=8e-4,atol=3e-6)
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,saved['state_dict'][key],rtol=8e-4,atol=8e-6,msg=key)
    def test_all_four_same_world_resume_exact_parameters_optimizer_and_history(self):
        for name in MODELS:
            with self.subTest(model=name):
                full=self.root/(name+'_full');resume=self.root/(name+'_resume')
                self.run_job(name,full);self.run_job(name,resume,['--max-updates','1'])
                self.run_job(name,resume,['--resume'])
                a=torch.load(full/name/'last.pt',map_location='cpu',weights_only=False)
                b=torch.load(resume/name/'last.pt',map_location='cpu',weights_only=False)
                self.assertEqual(a['history'],b['history'])
                for key,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
                for key,state in a['optimizer_state_dict']['state'].items():
                    for field,value in state.items():torch.testing.assert_close(value,b['optimizer_state_dict']['state'][key][field],rtol=0,atol=0)
    def test_all_four_old_optimized_checkpoints_import(self):
        cfg=config(2);cfg.update(rank=32,pod_geometry_bins=256,lr_patience=8,lr_threshold=.005,lr_cooldown=2,
                                graph_execution='bounded',graph_owner_nodes=4,graph_edge_chunk=7,graph_cache_mb=1)
        for name in MODELS:
            with self.subTest(model=name):
                root=self.root/(name+'_old')
                with patch.dict(os.environ,self.env):train_one(name,inspect_splits(self.split),cfg,root/name,'cpu',max_updates=1)
                self.run_job(name,root,['--resume','--max-updates','1'])
                saved=torch.load(root/name/'last.pt',map_location='cpu',weights_only=False)
                self.assertEqual(saved['global_step'],2);self.assertEqual(saved['position'],3)
    def test_heavy_slurm_uses_confirmed_full_node_and_preserves_owner_setting(self):
        bindir=self.root/'bin';bindir.mkdir();calls=self.root/'calls'
        for name,body in [('module','exit 0'),('scontrol','echo localhost'),('srun','printf "%s\\n" "$*" > "$CALL_LOG"; exit 0')]:
            command=bindir/name;command.write_text('#!/bin/bash\n'+body+'\n');command.chmod(0o755)
        venv=self.root/'venv/bin';venv.mkdir(parents=True);(venv/'activate').touch()
        runtime=self.root/'runtime';runtime.mkdir();(runtime/'load.sh').touch()
        env=dict(self.env,PATH=str(bindir)+os.pathsep+os.environ['PATH'],TURBINE_VENV=str(venv.parent),
                 TURBINE_OPENGL_RUNTIME=str(runtime),SLURM_SUBMIT_DIR=str(ROOT),SLURM_ARRAY_TASK_ID='6',
                 SLURM_JOB_ID='181920',SLURM_ARRAY_JOB_ID='181920',SLURM_JOB_NODELIST='localhost',SLURM_NNODES='2',
                 FANN_SPLITS=str(self.split),FANN_RUN_ROOT=str(self.root/'slurm'),FANN_DDP_GPUS_PER_NODE='2',
                 FANN_GRAPH_OWNER_NODES='8192',CALL_LOG=str(calls))
        env.pop('FANN_HEAVY_GPUS_PER_NODE',None);env.pop('FANN_STEPS',None)
        result=subprocess.run(['bash','scripts/slurm_fann_ddp_heavy.sh'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('model=GeoTransolver',result.stdout);self.assertIn('GPUs/node=8',result.stdout)
        self.assertIn('--graph-owner-nodes 8192',calls.read_text())
        self.assertIn('--checkpoint-every 1',calls.read_text())

    def test_interrupt_mid_owner_halo_discards_partial_global_batch(self):
        for name in ['GCN','Graph_Transformer']:
            with self.subTest(model=name):
                root=self.root/(name+'_abort');worker=self.root/(name+'_worker.py')
                worker.write_text('''from pathlib import Path
import os
import torch.distributed as dist
from tfai.benchmarks.fann_graph_execution import BoundedGraphExecution
from scripts.train_fann_ddp import main
original=BoundedGraphExecution.pieces
def pieces(self,*args,**kwargs):
    for i,piece in enumerate(original(self,*args,**kwargs)):
        yield piece
        if i==0 and dist.get_rank()==0:Path(os.environ['TEST_STOP_FILE']).touch()
BoundedGraphExecution.pieces=pieces
if __name__=='__main__':main()
''')
                command=self.command(name,root);command[5]=str(worker)
                env=dict(self.env,TEST_STOP_FILE=str(root/name/'STOP'))
                result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                saved=torch.load(root/name/'last.pt',map_location='cpu',weights_only=False)
                self.assertEqual(saved['global_step'],0);self.assertEqual(saved['position'],0)
                self.assertEqual(saved['pending_elements'],0);self.assertEqual(saved['optimizer_state_dict']['state'],{})
                torch.manual_seed(42);norm=Normalizer.from_state(saved['normalization']);model=FannAdapter(name,[],norm)
                for key,value in model.state_dict().items():torch.testing.assert_close(value,saved['state_dict'][key],rtol=0,atol=0)
                (root/name/'STOP').unlink();self.run_job(name,root,['--resume'])
                self.assertEqual(json.loads((root/name/'report.json').read_text())['state'],'complete')

if __name__=='__main__':unittest.main()
