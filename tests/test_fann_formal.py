import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from tfai.benchmarks.fann import FIELDS, MODELS, load_sample, Normalizer
from tfai.benchmarks.fann_training import inspect_splits, train_one, evaluate_frozen, fit_streaming_pod
from scripts.fann_format import select_structured_patches

ROOT=Path(__file__).resolve().parents[1]


def fixture(root, protocol='full'):
    paths=[]
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=np.float32)
    for i in range(4):
        coordinates=xyz+np.array([i*.03,0,0],dtype=np.float32)
        target=np.column_stack([coordinates[:,0]+i,coordinates[:,1]*2-i,coordinates[:,2]+i*.1,
                               1e5+coordinates[:,0]*300+i*100,290+coordinates[:,1]+i,np.ones(8)*(1+i*.01)]).astype(np.float32)
        path=root/f'case_{i}.npz'
        np.savez_compressed(path,coordinates=coordinates,targets=target,hexahedra=np.arange(8).reshape(1,8),
            original_node_ids=np.arange(8),original_cell_ids=np.array([0]),operating_conditions=np.array([900+i,1e5+i*100],np.float32),
            output_names=np.array(FIELDS),wall_distance=np.linspace(0,.01,8,dtype=np.float32),geometry_id=np.array([i]),
            selection_protocol=np.array(protocol),full_domain_nodes=np.array(8))
        paths.append(str(path))
    split=root/'splits.json';split.write_text(json.dumps({'train':paths[:2],'valid':paths[2:3],'test':paths[3:]}))
    return split


def config(epochs=2):
    return dict(epochs=epochs,seed=42,learning_rate=1e-4,lr_patience=0,lr_factor=.5,lr_threshold=.9,
                lr_cooldown=0,min_lr=1e-6,patience=60,grid=32,rank=2,scale='small',pod_geometry_bins=4,
                streaming_pod=True,grad_clip=1.,near_wall=.001,cache_samples=1,threads=2)


class FormalFannTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.split=fixture(self.root)
        torch.set_num_threads(2)

    @unittest.skipUnless(torch.cuda.device_count()>=2,'Two GPUs required for formal model-parallel recovery')
    def test_four_models_pipeline_interruption_resume_and_frozen_test(self):
        from unittest.mock import patch
        manifest=inspect_splits(self.split);cfg=config(1)
        cfg.update(graph_execution='bounded',graph_owner_nodes=4,graph_edge_chunk=7,graph_cache_mb=1)
        with patch.dict(os.environ,{'TURBINE_FANN_PIPELINE_GPUS':'2','TURBINE_GEO_NODE_CHUNK':'3',
                                    'TURBINE_FANN_GRAPH_CACHE':str(self.root/'topology')}):
            for name in ['GCN','Graph_UNet','Graph_Transformer','GeoTransolver']:
                with self.subTest(model=name):
                    folder=self.root/name
                    result=train_one(name,manifest,cfg,folder,'cuda:0',max_updates=1)
                    self.assertEqual(result['state'],'interrupted')
                    saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
                    self.assertEqual(saved['global_step'],1)
                    report=train_one(name,manifest,cfg,folder,'cuda:0',resume=True)
                    self.assertEqual(report['state'],'complete');self.assertEqual(report['optimizer_steps'],2)
                    frozen=evaluate_frozen(folder/'best.pt',manifest,'test','cuda:0',self.root/(name+'_frozen'))
                    np.testing.assert_allclose(frozen['rmse'],report['test']['rmse'],rtol=1e-5,atol=1e-5)

    def test_all_seven_train_select_restore_and_test(self):
        manifest=inspect_splits(self.split)
        for name in MODELS:
            with self.subTest(model=name):
                folder=self.root/'runs'/name
                report=train_one(name,manifest,config(),folder,'cpu')
                self.assertEqual(report['state'],'complete')
                self.assertEqual(report['optimizer_steps'],4)
                self.assertEqual(report['test']['nodes'],8)
                self.assertEqual(report['test']['cases'],1)
                self.assertEqual(len(report['test']['rmse']),6)
                self.assertTrue((folder/'predictions/test/prediction_00000.npz').exists())
                best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
                self.assertEqual(best['completed_epochs'],report['best_epoch'])
                self.assertEqual(best['manifest']['scope'],'full-domain nodal evaluation')
                with np.load(folder/'predictions/test/prediction_00000.npz') as data:
                    self.assertEqual(data['prediction'].shape,(8,6))
                    np.testing.assert_array_equal(data['original_node_ids'],np.arange(8))

        subprocess.run([os.sys.executable,'scripts/compare_fann.py','--root',str(self.root/'runs')],cwd=ROOT,check=True,capture_output=True,text=True)
        comparison=json.loads((self.root/'runs/comparison.json').read_text())
        self.assertTrue(comparison['complete']);self.assertEqual(len(comparison['models']),7)
        merged=self.root/'merged'
        subprocess.run([os.sys.executable,'scripts/compare_fann.py','--root',str(merged),'--reference-root',str(self.root/'runs')],cwd=ROOT,check=True,capture_output=True,text=True)
        self.assertTrue(json.loads((merged/'comparison.json').read_text())['complete'])
        (merged/'GCN').mkdir();(merged/'GCN/status.json').write_text('{"state":"failed"}')
        result=subprocess.run([os.sys.executable,'scripts/compare_fann.py','--root',str(merged),'--reference-root',str(self.root/'runs')],cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(json.loads((merged/'comparison.json').read_text())['states']['GCN']['state'],'failed')

    def test_interrupted_resume_matches_uninterrupted_optimizer_and_weights(self):
        manifest=inspect_splits(self.split);cfg=config(3)
        uninterrupted=self.root/'uninterrupted';resumed=self.root/'resumed'
        train_one('GCN',manifest,cfg,uninterrupted,'cpu')
        result=train_one('GCN',manifest,cfg,resumed,'cpu',max_updates=1)
        self.assertEqual(result['state'],'interrupted')
        self.assertFalse((resumed/'predictions/test').exists())
        train_one('GCN',manifest,cfg,resumed,'cpu',resume=True)
        left=torch.load(uninterrupted/'last.pt',map_location='cpu',weights_only=False)
        right=torch.load(resumed/'last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(left['history'],right['history'])
        self.assertEqual(left['scheduler_state_dict'],right['scheduler_state_dict'])
        for key,value in left['state_dict'].items():torch.testing.assert_close(value,right['state_dict'][key],rtol=0,atol=0)
        for index,state in left['optimizer_state_dict']['state'].items():
            for key,value in state.items():torch.testing.assert_close(value,right['optimizer_state_dict']['state'][index][key],rtol=0,atol=0)

    def test_signal_during_validation_preserves_pending_epoch(self):
        import signal
        from unittest.mock import patch
        from tfai.benchmarks import fann_training as module
        original=module.evaluate
        def terminate(*args,**kwargs):
            os.kill(os.getpid(),signal.SIGTERM)
            return original(*args,**kwargs)
        folder=self.root/'signal'
        manifest=inspect_splits(self.split)
        with patch.object(module,'evaluate',side_effect=terminate):
            result=train_one('GCN',manifest,config(),folder,'cpu')
        self.assertEqual(result['state'],'interrupted')
        checkpoint=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(checkpoint['completed_epochs'],0)
        self.assertEqual(checkpoint['position'],2)
        self.assertFalse((folder/'predictions/test').exists())
        report=train_one('GCN',manifest,config(),folder,'cpu',resume=True)
        self.assertEqual(report['optimizer_steps'],4)
        self.assertEqual(report['state'],'complete')

    def test_bounded_graph_formal_resume_and_frozen_evaluation(self):
        from unittest.mock import patch
        manifest=inspect_splits(self.split)
        cfg={**config(),'graph_execution':'bounded','graph_owner_nodes':4,'graph_edge_chunk':3,'graph_cache_mb':1}
        with patch.dict(os.environ,{'TURBINE_FANN_GRAPH_CACHE':str(self.root/'graph_cache')}):
            for name in ['GCN','Graph_UNet','Graph_Transformer']:
                with self.subTest(model=name):
                    folder=self.root/'bounded'/name
                    result=train_one(name,manifest,cfg,folder,'cpu',max_updates=1)
                    self.assertEqual(result['state'],'interrupted')
                    report=train_one(name,manifest,cfg,folder,'cpu',resume=True)
                    self.assertEqual(report['optimizer_steps'],4)
                    frozen=evaluate_frozen(folder/'best.pt',manifest,'test','cpu',self.root/'frozen_bounded'/name)
                    np.testing.assert_allclose(frozen['rmse'],report['test']['rmse'],atol=0,rtol=0)

    def test_halo_interruption_discards_partial_case_gradients(self):
        import signal
        from unittest.mock import patch
        from tfai.benchmarks.fann_graph_execution import BoundedGraphExecution
        manifest=inspect_splits(self.split)
        cfg={**config(),'graph_execution':'bounded','graph_owner_nodes':4,'graph_edge_chunk':3,'graph_cache_mb':1}
        original=BoundedGraphExecution.pieces
        def interrupted_pieces(*args,**kwargs):
            for i,item in enumerate(original(*args,**kwargs)):
                yield item
                if i==0:os.kill(os.getpid(),signal.SIGTERM)
        folder=self.root/'halo_signal'
        with patch.dict(os.environ,{'TURBINE_FANN_GRAPH_CACHE':str(self.root/'signal_cache')}):
            with patch.object(BoundedGraphExecution,'pieces',interrupted_pieces):
                result=train_one('GCN',manifest,cfg,folder,'cpu')
            self.assertEqual(result['global_step'],0)
            saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['position'],0)
            self.assertEqual(saved['optimizer_state_dict']['state'],{})
            report=train_one('GCN',manifest,cfg,folder,'cpu',resume=True)
            self.assertEqual(report['optimizer_steps'],4)

    def test_geometry_leakage_and_legacy_rejected(self):
        raw=json.loads(self.split.read_text());raw['valid']=raw['train'][:1]
        self.split.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,'Duplicate|Geometry leakage'):inspect_splits(self.split)
        with np.load(self.root/'case_0.npz') as data:sample={k:data[k] for k in data.files}
        sample['selection_protocol']=np.array('first');np.savez_compressed(self.root/'case_0.npz',**sample)
        with self.assertRaisesRegex(ValueError,'Formal protocol'):inspect_splits(self.split)

    def test_duplicate_run_lock_does_not_modify_active_status(self):
        import fcntl
        from tfai.benchmarks.fann_training import RunLockedError
        folder=self.root/'locked';folder.mkdir()
        status=folder/'status.json';status.write_text('{"state":"training"}')
        with (folder/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(RunLockedError):
                train_one('GCN',inspect_splits(self.split),config(),folder,'cpu')
        self.assertEqual(json.loads(status.read_text())['state'],'training')

    def test_modified_dataset_or_optimizer_config_cannot_resume(self):
        manifest=inspect_splits(self.split);folder=self.root/'resume_guard';cfg=config()
        train_one('GCN',manifest,cfg,folder,'cpu',max_updates=1)
        changed=copy.deepcopy(manifest);changed['signature']='changed'
        with self.assertRaisesRegex(ValueError,'Dataset/split'):train_one('GCN',changed,cfg,folder,'cpu',resume=True)
        with self.assertRaisesRegex(ValueError,'Resume configuration'):train_one('GCN',manifest,{**cfg,'learning_rate':.1},folder,'cpu',resume=True)

    def test_frozen_pod_inference_does_not_fit_basis_again(self):
        manifest=inspect_splits(self.split);folder=self.root/'pod'
        report=train_one('POD',manifest,config(),folder,'cpu')
        from unittest.mock import patch
        with patch('tfai.benchmarks.fann_training.fit_streaming_pod',side_effect=AssertionError('refit')):
            result=evaluate_frozen(folder/'best.pt',manifest,'test','cpu',self.root/'frozen')
        np.testing.assert_allclose(result['rmse'],report['test']['rmse'],rtol=0,atol=0)

    def test_early_stopping_and_learning_rate_are_saved(self):
        manifest=inspect_splits(self.split);cfg=config(10);cfg.update(learning_rate=1e-30,min_lr=1e-30,patience=2)
        report=train_one('GCN',manifest,cfg,self.root/'early','cpu')
        self.assertTrue(report['early_stopped']);self.assertEqual(report['completed_epochs'],3)
        last=torch.load(self.root/'early/last.pt',map_location='cpu',weights_only=False)
        self.assertEqual(last['early_stopping_counter'],2)
        self.assertIn('scheduler_state_dict',last)

    def test_test_labels_do_not_change_selected_weights(self):
        manifest=inspect_splits(self.split)
        train_one('GCN',manifest,config(),self.root/'first','cpu')
        path=self.root/'case_3.npz'
        with np.load(path) as data:sample={key:data[key] for key in data.files}
        sample['targets']=sample['targets']+1000;np.savez_compressed(path,**sample)
        train_one('GCN',inspect_splits(self.split),config(),self.root/'second','cpu')
        a=torch.load(self.root/'first/best.pt',map_location='cpu',weights_only=False)
        b=torch.load(self.root/'second/best.pt',map_location='cpu',weights_only=False)
        self.assertEqual(a['completed_epochs'],b['completed_epochs'])
        for key,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)

    def test_streaming_pod_matches_snapshot_projection(self):
        train=[load_sample(self.root/f'case_{i}.npz') for i in range(2)];norm=Normalizer(train)
        state=fit_streaming_pod(train,norm,2,self.root/'basis',chunk=3)
        basis=state['basis'].numpy()
        snapshots=np.stack([((s['targets']-norm.mean)/norm.scale).reshape(-1) for s in train])
        np.testing.assert_allclose(basis.T@basis,np.eye(2),atol=1e-5)
        np.testing.assert_allclose(snapshots@basis@basis.T,snapshots,atol=1e-5)

    def test_structured_selection_spans_blocks_and_is_deterministic(self):
        count=120
        conn=np.arange(1,count*8+1).reshape(count,8)
        block=np.repeat(np.repeat([0,1],60),8)
        i=np.repeat(np.tile(np.arange(10),12),8)
        j=np.repeat(np.tile(np.repeat([0,1,2],20),2),8)
        k=np.zeros(count*8)
        a=select_structured_patches(conn,[i,j,k],block,24,42)
        b=select_structured_patches(conn,[i,j,k],block,24,42)
        np.testing.assert_array_equal(a,b);self.assertEqual(len(a),24)
        self.assertEqual(set(block[conn[a,0]-1]),{0,1})
        self.assertEqual(set(j[conn[a,0]-1]),{0,1,2})

    def test_slurm_dependency_and_single_model_output(self):
        fake=self.root/'sbatch';log=self.root/'calls'
        fake.write_text('#!/bin/bash\necho "$*" >> "$CALL_LOG"\necho 12345\n');fake.chmod(0o755)
        env=dict(os.environ,PATH=str(self.root)+os.pathsep+os.environ['PATH'],CALL_LOG=str(log),FANN_RUN_ROOT=str(self.root/'submitted'))
        env.pop('FANN_SPLITS',None);env.pop('FANN_STEPS',None)
        subprocess.run(['bash','scripts/submit_fann_formal.sh'],cwd=ROOT,env=env,check=True,capture_output=True,text=True)
        calls=log.read_text().splitlines();self.assertEqual(len(calls),4)
        self.assertIn('afterok:12345',calls[1]);self.assertIn('afterok:12345',calls[2]);self.assertIn('afterany:12345',calls[3])

if __name__=='__main__':unittest.main()
