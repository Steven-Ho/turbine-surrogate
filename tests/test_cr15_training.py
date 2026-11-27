"""Five-field cache, accumulation/DDP parity, exact resume and frozen test."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['VelocityX','VelocityY','VelocityZ','Pressure','Temperature']


def fixture(root):
    prepared=root/'prepared';(prepared/'samples').mkdir(parents=True);(prepared/'metadata').mkdir()
    contract=json.loads((ROOT/'configs/datasets/cr15_map_v1.json').read_text())
    splits=dict(train=[],valid=[],test=[],baseline=[]);training=[];coords=[]
    rng=np.random.default_rng(42)
    for i,(split,n) in enumerate([('train',9),('train',12),('train',16),('valid',7),('test',10),('baseline',8)]):
        xyz=(.4+rng.random((n,3))*.1+i*.001).astype('float32')
        target=np.column_stack([xyz[:,0]*10+i,xyz[:,1]*20-i,xyz[:,2]*30,1e6+xyz[:,0]*1e5+i*1e4,1800+xyz[:,1]*10+i]).astype('float32')
        name=f'CR15S_{i:04d}';path=prepared/'samples'/f'{name}.npz'
        np.savez(path,coordinates=xyz,targets=target,global_conditions=np.array(contract['boundary_values'],dtype='float32'),
                 output_names=np.array(FIELDS),original_node_ids=np.arange(n))
        splits[split].append(str(path.relative_to(prepared)))
        (prepared/'metadata'/f'{name}.json').write_text(json.dumps(dict(split=split,grade='A',nodes=n)))
        if split=='train':training.append(target);coords.append(xyz)
    values=np.concatenate(training).astype('float64');xyz=np.concatenate(coords)
    stats=dict(source_split='train',nodes=len(values),mean=values.mean(0).tolist(),std=values.std(0).tolist(),
               coordinate_minimum=xyz.min(0).tolist(),coordinate_maximum=xyz.max(0).tolist())
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    summary=dict(state='complete',cache_ready=True,schema='cr15-map-five-fields-v1',contract_signature=signature,
                 splits={key:len(paths) for key,paths in splits.items()})
    for name,value in [('summary',summary),('contract',contract),('splits',splits),('train_statistics',stats)]:
        (prepared/(name+'.json')).write_text(json.dumps(value))
    return prepared


class CR15TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.prepared=fixture(self.root)
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MPLCONFIGDIR=str(self.root/'mpl'),TURBINE_GEO_QUERY_BACKEND='kdtree',TURBINE_GEO_REFERENCE_MAX_POINTS='0',
            TURBINE_GEO_CACHE_DIR=str(self.root/'neighbors'))

    def run_mode(self,output,extra=(),ddp=False,success=True):
        command=[sys.executable]
        if ddp:command+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
        command+=['scripts/train_cr15.py','--prepared',str(self.prepared),'--output',str(output),
                  '--device','cpu','--scale','small','--amp','off','--epochs','2','--batch-cases','2',
                  '--chunk','5','--threads','1','--checkpoint-every','1',*extra]
        result=subprocess.run(command,cwd=ROOT,env=self.env,capture_output=True,text=True,timeout=120)
        if success:self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return result

    def test_single_accumulation_matches_ddp_global_case_gradient(self):
        a=self.root/'single';b=self.root/'ddp'
        self.run_mode(a,['--max-updates','1']);self.run_mode(b,['--max-updates','1'],ddp=True)
        first=torch.load(a/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
        second=torch.load(b/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(first['position'],2);self.assertEqual(second['position'],2)
        self.assertEqual(first['pending_elements'],second['pending_elements'])
        for key,state in first['optimizer_state_dict']['state'].items():
            torch.testing.assert_close(state['exp_avg'],second['optimizer_state_dict']['state'][key]['exp_avg'],rtol=2e-3,atol=2e-7)
        for key,value in first['state_dict'].items():torch.testing.assert_close(value,second['state_dict'][key],rtol=2e-3,atol=3e-6)

    def test_exact_resume_and_five_field_final_metrics(self):
        full=self.root/'full';resume=self.root/'resume'
        self.run_mode(full)
        self.run_mode(resume,['--epoch-budget','1'])
        status=json.loads((resume/'GeoTransolver/status.json').read_text());self.assertEqual(status['completed_epochs'],1)
        self.assertFalse((resume/'GeoTransolver/report.json').exists())
        self.run_mode(resume,['--resume'])
        a=torch.load(full/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
        b=torch.load(resume/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(a['history'],b['history'])
        for key,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
        report=json.loads((resume/'GeoTransolver/report.json').read_text())
        self.assertEqual(report['fields'],FIELDS);self.assertEqual(report['test']['nodes'],10)
        self.assertEqual(report['test']['cases'],1);self.assertEqual(report['baseline']['cases'],1)
        self.assertEqual(len(report['test']['rmse']),5);self.assertFalse(report['near_wall_metrics_available'])
        self.assertNotIn('near_wall',report['test'])
        self.assertEqual(report['best_epoch'],min(a['history'],key=lambda row:row['val_loss'])['epoch'])
        row=report['test']['samples'][0]
        with np.load(row['prediction_file']) as predicted,np.load(row['source']) as source:
            error=predicted['prediction'].astype('float64')-source['targets']
            np.testing.assert_allclose(np.sqrt((error**2).mean(0)),report['test']['rmse'],rtol=1e-10)
            self.assertEqual(predicted['prediction'].shape,(10,5))
        visual=self.root/'visual'
        result=subprocess.run([sys.executable,'scripts/summarize_fann_run.py','--run-root',str(resume),
            '--output',str(visual),'--cases','1','--fields','Pressure','Temperature',
            '--max-points','10','--slice-points','8'],cwd=ROOT,env=self.env,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        summary=json.loads((visual/'summary.json').read_text())
        self.assertEqual(len(summary['metrics']),10)
        self.assertTrue((visual/'case_00_Pressure.png').is_file())
        rejected=self.run_mode(resume,['--resume','--learning-rate','0.0002'],success=False)
        self.assertNotEqual(rejected.returncode,0);self.assertIn('Resume configuration changed',rejected.stderr)

    def test_largest_case_capacity_does_not_read_validation_or_test(self):
        for relative in json.loads((self.prepared/'splits.json').read_text())['test']:
            (self.prepared/relative).unlink()
        output=self.root/'capacity';self.run_mode(output,['--batch-cases','1','--largest-first','--max-updates','1','--amp','bf16'])
        saved=torch.load(output/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(saved['pending_elements'],16*5)
        self.assertEqual(saved['completed_epochs'],0)
        self.assertFalse((output/'GeoTransolver/predictions').exists())

    def test_ddp_tail_has_no_duplicate_test_counts(self):
        output=self.root/'ddp_full';self.run_mode(output,ddp=True)
        report=json.loads((output/'GeoTransolver/report.json').read_text())
        self.assertEqual(report['optimizer_steps'],4)
        self.assertEqual(report['test']['nodes'],10);self.assertEqual(report['test']['cases'],1)
        self.assertEqual(report['validation']['nodes'],7);self.assertEqual(report['baseline']['nodes'],8)

    def test_capacity_and_short_submission_are_independent_of_fann(self):
        stub=self.root/'sbatch';trace=self.root/'trace'
        stub.write_text('#!'+sys.executable+'\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ["TRACE"]).write_text(json.dumps({"env":dict(os.environ),"args":sys.argv[1:]}))\n');stub.chmod(0o755)
        env=dict(self.env,PATH=str(self.root)+':'+os.environ['PATH'],TRACE=str(trace),FANN_SPLITS='untouched',
                 CR15_MAX_UPDATES='999',CR15_RESUME='1')
        for mode,batch,budget in [('capacity','1','0'),('short','16','5')]:
            result=subprocess.run(['bash',f'scripts/submit_cr15_{mode}.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            data=json.loads(trace.read_text());e=data['env']
            self.assertEqual(e['CR15_BATCH_CASES'],batch);self.assertEqual(e['CR15_EPOCH_BUDGET'],budget)
            self.assertEqual(e['CR15_RESUME'],'0');self.assertEqual(e['CR15_SCALE'],'base')
            self.assertEqual(e['FANN_SPLITS'],'untouched');self.assertIn('--gres=gpu:1',data['args'])
            if mode=='capacity':self.assertEqual(e['CR15_MAX_UPDATES'],'2')
            else:self.assertNotIn('CR15_MAX_UPDATES',e)

    def test_short_is_submitted_after_successful_capacity_job(self):
        stub=self.root/'sbatch';trace=self.root/'chain_trace'
        stub.write_text('#!'+sys.executable+'\nimport os,json,sys\nfrom pathlib import Path\np=Path(os.environ["TRACE"])\nrows=p.read_text().splitlines() if p.exists() else []\nwith p.open("a") as f:f.write(json.dumps({"env":dict(os.environ),"args":sys.argv[1:]})+"\\n")\nprint(str(2001+len(rows))+";cluster")\n');stub.chmod(0o755)
        env=dict(self.env,PATH=str(self.root)+':'+os.environ['PATH'],TRACE=str(trace))
        result=subprocess.run(['bash','scripts/submit_cr15_check_and_short.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        rows=[json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(len(rows),2);self.assertIn('--dependency=afterok:2001',rows[1]['args'])
        self.assertEqual(rows[0]['env']['CR15_MAX_UPDATES'],'2')
        self.assertEqual(rows[1]['env']['CR15_BATCH_CASES'],'16')

    def test_targets_are_not_input_features(self):
        from tfai.benchmarks.cr15 import load_prepared,CR15Normalizer,CR15Store
        prepared=load_prepared(self.prepared);norm=CR15Normalizer(prepared)
        sample=CR15Store(prepared['paths']['train'])[0]
        first=norm.input(sample,'cpu').x
        changed=dict(sample,targets=sample['targets']+1000000)
        second=norm.input(changed,'cpu').x
        self.assertEqual(first.shape,(9,9));torch.testing.assert_close(first,second,rtol=0,atol=0)
