import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tfai.benchmarks.fann import MODELS,Normalizer
from tfai.benchmarks.fann_startup_cache import load_startup
from tfai.benchmarks.fann_training import store_for,build_model,prepare_input,train_one
from tfai.benchmarks.fann_precompute import precompute
from tfai.benchmarks.fann_pod_cache import shared_pod_basis,basis_key
from tfai.benchmarks.fann_graph_execution import BoundedGraphExecution
from tfai.models._vendor.geotransolver2.spatial_neighbors import tree_candidates
from tests.test_fann_formal import fixture,config,ROOT


class PrecomputeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.split=fixture(self.root);torch.set_num_threads(2)
        self.environment={'TURBINE_FANN_STARTUP_CACHE':str(self.root/'startup'),
            'TURBINE_FANN_CACHE_WORK_ROOT':str(self.root/'work'),'TURBINE_FANN_POD_CACHE':str(self.root/'pod'),
            'TURBINE_FANN_UNET_CACHE':str(self.root/'unet'),'TURBINE_FANN_GRAPH_CACHE':str(self.root/'graphs'),
            'TURBINE_GEO_CACHE_DIR':str(self.root/'geo'),'TURBINE_GEO_REFERENCE_MAX_POINTS':'0',
            'TURBINE_GEO_QUERY_BACKEND':'kdtree','TURBINE_FANN_PIPELINE_GPUS':'1'}
        scope=patch.dict(os.environ,self.environment);scope.start();self.addCleanup(scope.stop)
        self.manifest=load_startup(self.split);self.norm=Normalizer.from_state(self.manifest['training_normalization'])
        self.cfg={'scale':'small','rank':2,'grid':32,'owner_nodes':[4],'graph_edge_chunk':7,'graph_cache_mb':0}
    def prepare(self,models=MODELS,name='report',**kwargs):
        return precompute(self.manifest,list(models),self.cfg,self.root/name,'cpu',**kwargs)
    def test_all_seven_cold_warm_reports_and_no_rebuild(self):
        cold=self.prepare()
        self.assertEqual(cold['state'],'complete');self.assertEqual(set(cold['models']),set(MODELS))
        self.assertGreater(cold['unique_artifact_bytes'],0)
        self.assertEqual(cold['models']['POD']['built_tasks'],1)
        self.assertEqual(cold['models']['GeoTransolver']['built_tasks'],8)
        self.assertIn('Graph_UNet deeper pooled adjacency depends on weights and must be recomputed',cold['not_cacheable'])
        with (patch('tfai.benchmarks.fann_training.fit_streaming_pod',side_effect=AssertionError('POD refit')),
              patch('tfai.benchmarks.fann_graph_execution.csr_from_hexahedra',side_effect=AssertionError('CSR rebuild')),
              patch.object(BoundedGraphExecution,'square_adjacency',side_effect=AssertionError('Static square rebuild')),
              patch('tfai.benchmarks.fann_unet_cache.cKDTree',side_effect=AssertionError('UNet tree rebuild')),
              patch('tfai.models._vendor.geotransolver2.ball_query.tree_candidates',side_effect=AssertionError('Geo tree rebuild'))):
            warm=self.prepare(name='warm')
        self.assertEqual(warm['unique_artifact_bytes'],cold['unique_artifact_bytes'])
        self.assertTrue(all(entry['built_tasks']==0 for entry in warm['models'].values()))
        rows=list(csv.DictReader((self.root/'warm/summary.csv').read_text().splitlines()))
        self.assertEqual(len(rows),7)
        events=[json.loads(line) for line in (self.root/'warm/timings.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['seconds']>=0 for row in events))
        self.assertEqual(sum(row['status']=='not_required' for row in events),1)
    def test_runtime_models_consume_the_prepared_artifacts(self):
        self.prepare();sample=store_for(self.manifest,'train')[0]
        cfg=config();cfg.update(graph_execution='bounded',graph_owner_nodes=4,graph_edge_chunk=7,graph_cache_mb=1)
        with (patch('tfai.benchmarks.fann_training.fit_streaming_pod',side_effect=AssertionError('Runtime POD refit')),
              patch('tfai.benchmarks.fann_graph_execution.csr_from_hexahedra',side_effect=AssertionError('Runtime CSR rebuild')),
              patch('tfai.benchmarks.fann_unet_cache.cKDTree',side_effect=AssertionError('Runtime UNet tree rebuild')),
              patch('tfai.models._vendor.geotransolver2.ball_query.tree_candidates',side_effect=AssertionError('Runtime Geo tree rebuild'))):
            for name in MODELS:
                model=build_model(name,store_for(self.manifest,'train'),self.norm,cfg,scratch=self.root/'unused_scratch')
                model.eval();execution=getattr(model,'graph_executor',None)
                with torch.no_grad():
                    if name=='Graph_UNet':
                        with patch.object(BoundedGraphExecution,'square_adjacency',side_effect=AssertionError('Runtime static rebuild')):
                            execution.prepare_unet_static(sample)
                        original=BoundedGraphExecution.square_adjacency
                        with patch.object(BoundedGraphExecution,'square_adjacency',wraps=original) as dynamic:
                            prediction=execution.graph_unet(model,self.norm,sample,'cpu')
                            self.assertEqual(dynamic.call_count,model.net.net.depth-1)
                    elif execution is not None:
                        with patch('tfai.benchmarks.fann_graph_execution.np.union1d',side_effect=AssertionError('Runtime halo rebuild')):
                            prediction=execution.predict_cpu(model,self.norm,sample,'cpu')
                    else:prediction=model(prepare_input(model,self.norm,sample,'cpu'))
                self.assertEqual(tuple(prediction.shape),(8,6));self.assertTrue(torch.isfinite(prediction).all())
            for trial in ['new_trial_a','new_trial_b']:
                result=train_one('POD',self.manifest,cfg,self.root/trial,'cpu',max_updates=1)
                self.assertEqual(result['state'],'interrupted')
    def test_pod_cache_is_train_only_and_rank_sensitive(self):
        train=store_for(self.manifest,'train');first=shared_pod_basis(train,self.norm,2,self.root/'pod')
        old_key=basis_key(train,self.norm,2)
        path=self.root/'case_3.npz'
        with np.load(path) as data:sample={key:data[key] for key in data.files}
        sample['targets']+=1000;np.savez_compressed(path,**sample)
        updated=load_startup(self.split);norm=Normalizer.from_state(updated['training_normalization'])
        self.assertEqual(basis_key(store_for(updated,'train'),norm,2),old_key)
        with patch('tfai.benchmarks.fann_training.fit_streaming_pod',side_effect=AssertionError('Unexpected refit')):
            second=shared_pod_basis(store_for(updated,'train'),norm,2,self.root/'pod')
        torch.testing.assert_close(first['basis'],second['basis'],rtol=0,atol=0)
        self.assertNotEqual(basis_key(train,self.norm,1),old_key)
        path=self.root/'case_0.npz'
        with np.load(path) as data:sample={key:data[key] for key in data.files}
        sample['targets']+=10;np.savez_compressed(path,**sample)
        changed=load_startup(self.split);norm=Normalizer.from_state(changed['training_normalization'])
        self.assertNotEqual(basis_key(store_for(changed,'train'),norm,2),old_key)
    def test_can_import_matching_old_pod_checkpoint(self):
        train_one('POD',self.manifest,config(),self.root/'old','cpu',max_updates=1)
        checkpoint=self.root/'old/last.pt'
        with patch('tfai.benchmarks.fann_training.fit_streaming_pod',side_effect=AssertionError('Old POD must not refit')):
            state=shared_pod_basis(store_for(self.manifest,'train'),self.norm,2,self.root/'import',checkpoint=checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        torch.testing.assert_close(state['basis'],saved['state_dict']['basis'],rtol=0,atol=0)
    def test_parallel_spatial_queries_keep_identical_indices(self):
        points=np.random.default_rng(4).normal(size=(60,3)).astype(np.float32)
        with patch.dict(os.environ,{'TURBINE_GEO_TREE_WORKERS':'1'}):a,_=tree_candidates(points,points,2.,8)
        with patch.dict(os.environ,{'TURBINE_GEO_TREE_WORKERS':'2'}):b,_=tree_candidates(points,points,2.,8)
        np.testing.assert_array_equal(a,b)
    def test_two_fresh_pod_runs_fit_basis_only_once(self):
        marker=self.root/'fit_calls';worker=self.root/'worker.py'
        worker.write_text('''from pathlib import Path
import os
import tfai.benchmarks.fann_training as training
original=training.fit_streaming_pod
def fit(*args,**kwargs):
    with open(os.environ['FIT_CALLS'],'a') as stream:stream.write('fit\\n')
    return original(*args,**kwargs)
training.fit_streaming_pod=fit
from scripts.train_fann import main
if __name__=='__main__':main()
''')
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',FIT_CALLS=str(marker),OMP_NUM_THREADS='1')
        processes=[]
        for name in ['a','b']:
            command=[sys.executable,str(worker),'--splits',str(self.split),'--models','POD','--output',str(self.root/name),
                     '--device','cpu','--rank','2','--pod-geometry-bins','4','--threads','1','--max-updates','1']
            processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
        for process in processes:
            out,err=process.communicate(timeout=90);self.assertEqual(process.returncode,130,out+err)
        self.assertEqual(marker.read_text().splitlines(),['fit'])
    def test_slurm_cache_job_passes_configured_models_and_owner_variants(self):
        bindir=self.root/'bin';bindir.mkdir();calls=self.root/'argv.json'
        module=bindir/'module';module.write_text('#!/bin/bash\nexit 0\n');module.chmod(0o755)
        python=bindir/'python';python.write_text('#!'+sys.executable+'\nimport json,os,sys\nopen(os.environ["CALL_LOG"],"w").write(json.dumps(sys.argv[1:]))\n');python.chmod(0o755)
        venv=self.root/'venv/bin';venv.mkdir(parents=True);(venv/'activate').touch()
        runtime=self.root/'runtime';runtime.mkdir();(runtime/'load.sh').touch()
        env=dict(os.environ,PATH=str(bindir)+os.pathsep+os.environ['PATH'],TURBINE_VENV=str(venv.parent),
                 TURBINE_OPENGL_RUNTIME=str(runtime),SLURM_SUBMIT_DIR=str(ROOT),SLURM_JOB_ID='1900',
                 SLURM_CPUS_PER_TASK='8',FANN_SPLITS=str(self.split),FANN_CACHE_MODELS='UNet GeoTransolver',
                 FANN_CACHE_OWNER_NODES='8192 32768',FANN_SCALE='base',CALL_LOG=str(calls))
        result=subprocess.run(['bash','scripts/slurm_fann_precompute.sh'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        argv=json.loads(calls.read_text());self.assertEqual(argv[argv.index('--device')+1],'cuda')
        self.assertEqual(argv[argv.index('--scale')+1],'base')
        self.assertEqual(argv[argv.index('--models')+1:argv.index('--models')+3],['UNet','GeoTransolver'])
        owner=max(i for i,value in enumerate(argv) if value=='--owner-nodes')
        self.assertEqual(argv[owner+1:owner+3],['8192','32768'])

    def test_cli_uses_fixed_data_and_writes_failed_report_on_bad_geometry(self):
        # Geometry preprocessing never marks an incomplete task set as complete.
        with patch('tfai.benchmarks.fann_precompute.read_coordinates',side_effect=RuntimeError('simulated IO failure')):
            with self.assertRaisesRegex(RuntimeError,'simulated IO'):self.prepare(['UNet'],name='failed')
        self.assertEqual(json.loads((self.root/'failed/report.json').read_text())['state'],'failed')

if __name__=='__main__':unittest.main()
