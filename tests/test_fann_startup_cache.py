import copy
import hashlib
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
from tfai.benchmarks.fann import load_sample,Normalizer
from tfai.benchmarks.fann_training import inspect_splits,store_for,train_one
from tfai.benchmarks.fann_startup_cache import load_startup
from tests.test_fann_formal import fixture,config,ROOT


class StartupCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.split=fixture(self.root);self.cache=self.root/'cache'
        torch.set_num_threads(2)
    def start(self,**kwargs):return load_startup(self.split,cache_dir=self.cache,**kwargs)
    def rewrite(self,index,**values):
        path=self.root/f'case_{index}.npz'
        with np.load(path) as data:sample={key:data[key] for key in data.files}
        sample.update(values);np.savez_compressed(path,**sample)
    def test_signature_and_normalization_match_strict_old_path_and_warm_hit_skips_reads(self):
        strict=inspect_splits(self.split)
        expected=Normalizer(store_for(strict,'train'))
        registered=self.start()
        self.assertEqual(registered['signature'],strict['signature'])
        actual=Normalizer.from_state(registered['training_normalization'])
        for key,value in vars(expected).items():np.testing.assert_array_equal(value,vars(actual)[key])
        with patch('tfai.benchmarks.fann_training.load_sample',side_effect=AssertionError('No full content reread on startup cache hit')):
            warm=self.start()
        self.assertEqual(warm['signature'],strict['signature'])
        with patch('tfai.benchmarks.fann_training.load_sample',wraps=load_sample) as loader:
            sample=store_for(warm,'train')[0]
        self.assertEqual(loader.call_args.kwargs,{'validate':False})
        self.assertEqual(sample['fingerprint'],strict['splits']['train'][0]['fingerprint'])
    def test_fingerprint_bytes_unchanged_by_zero_copy_hashing(self):
        sample=load_sample(self.root/'case_0.npz');digest=hashlib.sha256()
        for key in ('coordinates','targets','operating_conditions','original_node_ids','hexahedra','wall_distance','fixed_conditions'):
            if key not in sample:continue
            digest.update((key+str(sample[key].shape)+str(sample[key].dtype)).encode())
            digest.update(np.ascontiguousarray(sample[key]).tobytes())
        self.assertEqual(sample['fingerprint'],digest.hexdigest())
    def test_new_data_invalidates_cache_and_active_store_rejects_changes(self):
        initial=self.start();store=store_for(initial,'train');store[0]
        target=load_sample(self.root/'case_0.npz')['targets']+1
        self.rewrite(0,targets=target)
        with self.assertRaisesRegex(ValueError,'data changed'):store[0]
        changed=self.start()
        self.assertNotEqual(initial['signature'],changed['signature'])
    def test_validation_targets_never_enter_training_statistics(self):
        a=self.start();sample=load_sample(self.root/'case_3.npz')
        self.rewrite(3,targets=sample['targets']+1e6)
        b=self.start()
        self.assertNotEqual(a['signature'],b['signature'])
        for key,value in a['training_normalization'].items():np.testing.assert_array_equal(value,b['training_normalization'][key])
    def test_split_change_and_legacy_cannot_reuse_registration(self):
        self.start();raw=json.loads(self.split.read_text());raw['valid']=raw['train'][:1]
        self.split.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,'Duplicate|Geometry leakage'):self.start()
        self.split=fixture(self.root);self.rewrite(0,selection_protocol=np.array('first'))
        with self.assertRaisesRegex(ValueError,'Formal protocol'):self.start()
    def test_shared_mesh_change_invalidates_cache_even_with_preserved_mtime(self):
        shared=self.root/'mesh.npz'
        np.savez(shared,hexahedra=np.arange(8).reshape(1,8),original_node_ids=np.arange(8),original_cell_ids=np.array([0]))
        for i in range(4):
            with np.load(self.root/f'case_{i}.npz') as data:sample={key:data[key] for key in data.files}
            for key in ['hexahedra','original_node_ids','original_cell_ids']:del sample[key]
            sample['mesh_file']=np.array('mesh.npz');np.savez_compressed(self.root/f'case_{i}.npz',**sample)
        a=self.start();stamp=shared.stat()
        np.savez(shared,hexahedra=np.arange(8)[::-1].reshape(1,8),original_node_ids=np.arange(8),original_cell_ids=np.array([0]))
        os.utime(shared,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        b=self.start();self.assertNotEqual(a['signature'],b['signature'])
    def test_changed_data_never_reimports_stale_checkpoint(self):
        initial=self.start();norm=Normalizer.from_state(initial['training_normalization'])
        checkpoint=self.root/'old.pt';torch.save({'manifest':initial,'normalization':vars(norm)},checkpoint)
        target=load_sample(self.root/'case_0.npz')['targets']+10
        self.rewrite(0,targets=target)
        changed=self.start(from_checkpoint=checkpoint)
        self.assertNotEqual(changed['signature'],initial['signature'])
        self.assertFalse(np.array_equal(changed['training_normalization']['mean'],initial['training_normalization']['mean']))

    def test_multiworker_matches_serial_in_original_sample_order(self):
        serial=inspect_splits(self.split,collect_normalization=True)
        parallel=inspect_splits(self.split,collect_normalization=True,workers=2)
        self.assertEqual(serial['signature'],parallel['signature'])
        for key,value in serial['training_normalization'].items():np.testing.assert_array_equal(value,parallel['training_normalization'][key])
    def test_four_processes_register_once_and_then_share_result(self):
        script=self.root/'register.py'
        script.write_text("from tfai.benchmarks.fann_startup_cache import load_startup\nif __name__=='__main__':\n load_startup("+repr(str(self.split))+",cache_dir="+repr(str(self.cache))+",progress=lambda s:print(s,flush=True))\n")
        env=dict(os.environ,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        processes=[subprocess.Popen([sys.executable,str(script)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
        outputs=[]
        for process in processes:
            out,err=process.communicate(timeout=90);self.assertEqual(process.returncode,0,err);outputs.append(out)
        combined=''.join(outputs)
        self.assertEqual(combined.count('registering dataset once:'),1)
        self.assertEqual(combined.count('startup cache hit:'),3)
    def test_cached_normalization_no_second_pass_and_resumes_old_checkpoint(self):
        strict=inspect_splits(self.split);cfg=config(1);folder=self.root/'run'
        train_one('GCN',strict,cfg,folder,'cpu',max_updates=1)
        cached=self.start()
        with patch('tfai.benchmarks.fann_training.Normalizer',wraps=Normalizer) as constructor:
            train_one('GCN',cached,cfg,folder,'cpu',resume=True)
            constructor.assert_not_called()
        # New models also restore the registered train-only statistics.
        with patch('tfai.benchmarks.fann_training.Normalizer',wraps=Normalizer) as constructor:
            train_one('GCN',cached,cfg,self.root/'new','cpu',max_updates=1)
            constructor.assert_not_called()
    def test_import_existing_checkpoint_only_for_explicit_unchanged_same_split(self):
        strict=inspect_splits(self.split);norm=Normalizer(store_for(strict,'train'))
        checkpoint=self.root/'last.pt';torch.save({'manifest':strict,'normalization':vars(norm)},checkpoint)
        with patch('tfai.benchmarks.fann_training.inspect_splits',side_effect=AssertionError('Import must avoid rereading all cases')):
            imported=self.start(from_checkpoint=checkpoint)
        self.assertEqual(imported['signature'],strict['signature'])
        raw=json.loads(self.split.read_text());raw['train'].reverse();self.split.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError,'paths/order differ'):
            load_startup(self.split,cache_dir=self.root/'other_cache',from_checkpoint=checkpoint)

if __name__=='__main__':unittest.main()
