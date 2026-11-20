import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
from tfai.benchmarks.fann import Normalizer,load_sample
from tfai.benchmarks.fann_training import build_model,inspect_splits,prepare_input,store_for,train_one,evaluate_frozen
from tfai.benchmarks.fann_startup_cache import load_startup
from tests.test_fann_formal import ROOT,fixture,config


class DDPTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.split=fixture(self.root);torch.set_num_threads(1)
        # Three training cases, unequal sizes, two ranks: exercise the short tail.
        sample=load_sample(self.root/'case_0.npz')
        sample={k:v for k,v in sample.items() if k not in ('source','fingerprint')}
        sample['coordinates']=np.concatenate([sample['coordinates'],sample['coordinates']+2])
        sample['targets']=np.concatenate([sample['targets']+1,sample['targets']+2])
        sample['hexahedra']=np.arange(16).reshape(2,8);sample['original_node_ids']=np.arange(16)
        sample['original_cell_ids']=np.arange(2);sample['wall_distance']=np.tile(sample['wall_distance'],2)
        sample['geometry_id']=np.array([4]);sample['full_domain_nodes']=np.array(16)
        path=self.root/'case_4.npz';np.savez_compressed(path,**sample)
        raw=json.loads(self.split.read_text());raw['train'].append(str(path));self.split.write_text(json.dumps(raw))
        self.env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      TURBINE_FANN_PIPELINE_GPUS='1',TURBINE_FANN_STARTUP_CACHE=str(self.root/'cache'),
                      TURBINE_FANN_UNET_CACHE=str(self.root/'unet_cache'),TURBINE_FANN_TRANSOLVER_CHUNK_SIZE='7')
    def command(self,name,output,extra=(),backend='gloo',device='cpu'):
        return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                'scripts/train_fann_ddp.py','--models',name,'--splits',str(self.split),'--output',str(output),
                '--device',device,'--backend',backend,'--epochs','2','--threads','1','--checkpoint-every','1',*extra]
    def run_job(self,name,output,extra=()):
        result=subprocess.run(self.command(name,output,extra),cwd=ROOT,env=self.env,capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return result
    def test_both_models_tail_coverage_resume_and_frozen_metrics(self):
        for name in ['UNet','Transolver_3']:
            with self.subTest(model=name):
                root=self.root/name
                out=self.run_job(name,root,['--max-updates','2'])
                checkpoint=torch.load(root/name/'last.pt',map_location='cpu',weights_only=False)
                self.assertEqual(checkpoint['position'],3)
                self.assertEqual(checkpoint['pending_elements'],(8+8+16)*6)
                self.assertEqual(checkpoint['global_step'],2)
                self.assertIn('batch_cases=1',out.stdout)
                self.run_job(name,root,['--resume'])
                report=json.loads((root/name/'report.json').read_text())
                self.assertEqual(report['optimizer_steps'],4);self.assertEqual(report['test']['cases'],1)
                self.assertEqual(report['test']['nodes'],8)
                manifest=inspect_splits(self.split)
                frozen=evaluate_frozen(root/name/'best.pt',manifest,'test','cpu',self.root/(name+'_frozen'))
                np.testing.assert_allclose(report['test']['rmse'],frozen['rmse'],rtol=2e-5,atol=2e-5)
                self.assertEqual(len(list((root/name/'predictions/test').rglob('prediction_*.npz'))),1)
    def test_same_world_resume_exact_weights_optimizer_and_history(self):
        name='Transolver_3';full=self.root/'full';resume=self.root/'resume'
        self.run_job(name,full)
        self.run_job(name,resume,['--max-updates','1'])
        self.run_job(name,resume,['--resume'])
        a=torch.load(full/name/'last.pt',map_location='cpu',weights_only=False)
        b=torch.load(resume/name/'last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(a['history'],b['history']);self.assertEqual(a['scheduler_state_dict'],b['scheduler_state_dict'])
        for key,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
        for key,state in a['optimizer_state_dict']['state'].items():
            for field,value in state.items():torch.testing.assert_close(value,b['optimizer_state_dict']['state'][key][field],rtol=0,atol=0)
    def test_first_step_matches_node_weighted_global_batch_reference(self):
        name='Transolver_3';root=self.root/'gradient'
        self.run_job(name,root,['--max-updates','1'])
        checkpoint=torch.load(root/name/'last.pt',map_location='cpu',weights_only=False)
        manifest=inspect_splits(self.split);norm=Normalizer(store_for(manifest,'train'))
        torch.manual_seed(42)
        with unittest.mock.patch.dict(os.environ,{'TURBINE_FANN_TRANSOLVER_CHUNK_SIZE':'7'}):
            model=build_model(name,[],norm,checkpoint['config'])
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        store=store_for(manifest,'train');order=torch.randperm(3,generator=torch.Generator().manual_seed(42)).tolist()[:2]
        samples=[store[i] for i in order];denominator=sum(s['targets'].size for s in samples)
        for sample in samples:
            p=model(prepare_input(model,norm,sample,'cpu'))
            ((p-norm.target(sample,'cpu')).square().sum()/denominator).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        for key,state in optimizer.state_dict()['state'].items():
            torch.testing.assert_close(state['exp_avg'],checkpoint['optimizer_state_dict']['state'][key]['exp_avg'],rtol=3e-4,atol=2e-7)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,checkpoint['state_dict'][key],rtol=3e-4,atol=2e-6)
    def test_old_single_gpu_checkpoint_import(self):
        name='Transolver_3';root=self.root/'legacy';cfg=config(2)
        cfg.update(rank=32,pod_geometry_bins=256,lr_patience=8,lr_threshold=.005,lr_cooldown=2)
        train_one(name,inspect_splits(self.split),cfg,root/name,'cpu',max_updates=1)
        self.run_job(name,root,['--resume','--max-updates','1'])
        checkpoint=torch.load(root/name/'last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(checkpoint['global_step'],2);self.assertEqual(checkpoint['position'],3)
        self.assertEqual(checkpoint['distributed_state']['world_size'],2)
    def test_cooperative_stop_file_and_resume(self):
        root=self.root/'stop';folder=root/'Transolver_3';folder.mkdir(parents=True)
        (folder/'STOP').touch()
        result=self.run_job('Transolver_3',root)
        self.assertIn('Safe DDP stop',result.stdout)
        checkpoint=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(checkpoint['global_step'],0)
        self.assertEqual(json.loads((folder/'status.json').read_text())['state'],'interrupted')
        (folder/'STOP').unlink();self.run_job('Transolver_3',root,['--resume'])
        self.assertEqual(json.loads((folder/'report.json').read_text())['optimizer_steps'],4)

    def test_slurm_prewalltime_signal_requests_stop_without_killing_launcher(self):
        bindir=self.root/'bin';bindir.mkdir()
        for name,body in [('module','exit 0'),('scontrol','echo localhost'),('srun','sleep 0.5; exit 0')]:
            command=bindir/name;command.write_text('#!/bin/bash\n'+body+'\n');command.chmod(0o755)
        venv=self.root/'venv/bin';venv.mkdir(parents=True);(venv/'activate').touch()
        runtime=self.root/'runtime';runtime.mkdir();(runtime/'load.sh').touch()
        output=self.root/'slurm'
        env=dict(self.env,PATH=str(bindir)+os.pathsep+os.environ['PATH'],TURBINE_VENV=str(venv.parent),
                 TURBINE_OPENGL_RUNTIME=str(runtime),SLURM_SUBMIT_DIR=str(ROOT),SLURM_ARRAY_TASK_ID='5',
                 SLURM_JOB_ID='181919',SLURM_ARRAY_JOB_ID='181919',SLURM_JOB_NODELIST='localhost',SLURM_NNODES='2',
                 FANN_SPLITS=str(self.split),FANN_RUN_ROOT=str(output))
        env.pop('FANN_STEPS',None)
        process=subprocess.Popen(['bash','scripts/slurm_fann_ddp.sh'],cwd=ROOT,env=env,
                                 stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        first=process.stdout.readline();self.assertIn('DDP model=',first)
        process.send_signal(signal.SIGUSR1)
        out,err=process.communicate(timeout=20);self.assertEqual(process.returncode,0,first+out+err)
        self.assertTrue((output/'.ddp_stop_181919_5').exists())

    def test_ddp_lock_does_not_modify_live_job_status(self):
        import fcntl
        root=self.root/'locked';folder=root/'Transolver_3';folder.mkdir(parents=True)
        status=folder/'status.json';status.write_text('{"state":"training","owner":"existing"}')
        with (folder/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=subprocess.run(self.command('Transolver_3',root,['--max-updates','1']),cwd=ROOT,env=self.env,
                                  capture_output=True,text=True,timeout=90)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Another process holds',result.stdout+result.stderr)
        self.assertEqual(json.loads(status.read_text()),{'state':'training','owner':'existing'})

    def test_two_torchrun_agents_rendezvous(self):
        with socket.socket() as stream:stream.bind(('127.0.0.1',0));port=stream.getsockname()[1]
        output=self.root/'agents';processes=[]
        for rank in range(2):
            command=[sys.executable,'-m','torch.distributed.run','--nnodes=2','--nproc_per_node=1',f'--node_rank={rank}',
                     '--master_addr=127.0.0.1',f'--master_port={port}',*self.command('Transolver_3',output,['--max-updates','1'])[5:]]
            processes.append(subprocess.Popen(command,cwd=ROOT,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
        for process in processes:
            out,err=process.communicate(timeout=180);self.assertEqual(process.returncode,0,out+err)
        checkpoint=torch.load(output/'Transolver_3/last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(checkpoint['distributed_state']['world_size'],2)

if __name__=='__main__':unittest.main()
