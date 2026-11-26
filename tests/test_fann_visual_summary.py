"""CPU-only real NPZ/plot/ZIP smoke test; no model/runtime dependencies."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['VelocityX','VelocityY','VelocityZ','Pressure','Temperature','Density']


class VisualSummaryTests(unittest.TestCase):
    def test_compact_report_reads_shared_ids_and_remaps_moved_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'run/GeoTransolver';folder.mkdir(parents=True)
            prepared=root/'prepared';prepared.mkdir()
            n=401;rng=np.random.default_rng(42)
            xyz=rng.random((n,3)).astype('float32');truth=rng.normal(size=(n,6)).astype('float32')
            truth[:,3]=1e5+truth[:,3]*1000;truth[:,4]=300+truth[:,4];truth[:,5]=1+truth[:,5]*.01
            predicted=truth+np.array([.1,.2,.3,10,.5,.001],dtype='float32')
            ids=np.arange(n,dtype='int64')
            np.savez(prepared/'mesh.npz',original_node_ids=ids,hexahedra=np.zeros((10,8),dtype='int64'))
            np.savez(prepared/'case.npz',coordinates=xyz,targets=truth,mesh_file=np.array('mesh.npz'))
            p=folder/'predictions/test/rank_00000/case_00000';p.mkdir(parents=True)
            prediction=p/'prediction_00000.npz'
            np.savez_compressed(prediction,prediction=predicted,original_node_ids=ids,output_names=np.array(FIELDS))
            error=predicted.astype('float64')-truth;rmse=np.sqrt(np.mean(error**2,axis=0)).tolist()
            row=dict(source='/old/prepared/case.npz',prediction_file='/old/run/GeoTransolver/predictions/test/rank_00000/case_00000/prediction_00000.npz',rmse=rmse,mae=np.abs(error).mean(0).tolist())
            metric=dict(fields=FIELDS,rmse=rmse,mae=row['mae'],r2=[.9]*6,normalized_mse=.4,nodes=n,cases=1,samples=[row])
            report=dict(state='complete',model='GeoTransolver',completed_epochs=3,best_epoch=2,fields=FIELDS,validation=metric,test=metric,training_seconds=20)
            (folder/'report.json').write_text(json.dumps(report))
            history=[dict(epoch=i,global_step=i*16,train_loss=1/i,val_loss=1/i+.01,lr=1e-4,validation_rmse=rmse) for i in range(1,4)]
            (folder/'history.json').write_text(json.dumps(history))
            # A large run member must not be included in the compact ZIP.
            (folder/'last.pt').write_bytes(b'x'*1024*1024)
            (root/'torch.py').write_text('raise OSError("No accelerator runtime")\n')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root),MPLCONFIGDIR=str(root/'mpl'))
            out=root/'summary';command=[sys.executable,'scripts/summarize_fann_run.py','--run-root',str(root/'run'),
                '--output',str(out),'--cases','1','--fields','Pressure','Temperature','--max-points','100',
                '--slice-points','80','--path-map',f'/old/prepared={prepared}']
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            summary=json.loads((out/'summary.json').read_text());case=summary['cases'][0]
            self.assertEqual(case['sampled_nodes'],100);self.assertLessEqual(case['slice_nodes'],80)
            self.assertEqual(case['nodes'],n)
            with (out/'field_metrics.csv').open() as f:metrics=list(csv.DictReader(f))
            self.assertEqual(len(metrics),12);self.assertEqual(float(metrics[3]['rmse']),rmse[3])
            with np.load(out/'case_00_sample.npz') as f:self.assertEqual(f['prediction'].shape,(100,6))
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                names=z.namelist();self.assertTrue(any(n.endswith('index.html') for n in names))
                self.assertFalse(any('last.pt' in n or 'prediction_00000.npz' in n for n in names))
            self.assertTrue((out/'case_00_Pressure.png').is_file())
            self.assertTrue((out/'field_validation_rmse.png').is_file())
            # Misordered node IDs must fail before exporting deceptive cloud plots.
            np.savez_compressed(prediction,prediction=predicted,original_node_ids=ids[::-1],output_names=np.array(FIELDS))
            bad=command.copy();bad[bad.index(str(out))]=str(root/'bad')
            result=subprocess.run(bad,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertNotEqual(result.returncode,0);self.assertIn('node IDs differ',result.stderr)
