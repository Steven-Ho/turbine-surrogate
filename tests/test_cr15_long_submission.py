"""No-Torch formal resume submission settings and refusal of wrong topology."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class CR15LongSubmissionTests(unittest.TestCase):
    def test_resumes_two_node_checkpoint_without_short_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'experiment/GeoTransolver';folder.mkdir(parents=True)
            for name in ['last.pt','best.pt']:(folder/name).touch()
            status=dict(dataset='CR15',state='interrupted',world_size=16,global_batch_cases=16,completed_epochs=1)
            (folder/'status.json').write_text(json.dumps(status))
            (folder/'history.json').write_text(json.dumps([dict(epoch=1)]))
            config=dict(scale='base',amp='bf16',batch_cases=16,chunk=32768,largest_first=False,offload=False,
                beta1=.9,beta2=.999,seed=42,grad_clip=1.,checkpoint_every=10,log_every=1,
                learning_rate=1e-4,lr_patience=20,lr_threshold=.001,lr_cooldown=5,lr_factor=.5,min_lr=1e-6,patience=60)
            (folder/'configuration.json').write_text(json.dumps(dict(config=config,prepared_root=str(root/'prepared'))))
            stub=root/'sbatch';stub.write_text('#!/usr/bin/env python3\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ["TRACE"]).write_text(json.dumps({"env":dict(os.environ),"args":sys.argv[1:]}))\nprint(12345)\n');stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],TRACE=str(root/'trace'),CR15_MAX_UPDATES='2',
                CR15_EPOCH_BUDGET='1',CR15_GPUS_PER_NODE='1',CR15_LARGEST_FIRST='1',CR15_RESUME='0')
            command=['bash','scripts/submit_cr15_2nodes_long.sh','--run-root',str(root/'experiment')]
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'trace').read_text());e=report['env'];a=report['args']
            self.assertEqual(e['CR15_EPOCHS'],'400');self.assertEqual(e['CR15_EPOCH_BUDGET'],'0')
            self.assertNotIn('CR15_MAX_UPDATES',e);self.assertEqual(e['CR15_RESUME'],'1')
            self.assertEqual(e['CR15_GPUS_PER_NODE'],'8');self.assertEqual(e['CR15_BATCH_CASES'],'16')
            self.assertEqual(e['CR15_LARGEST_FIRST'],'0');self.assertEqual(e['CR15_SCALE'],'base')
            self.assertEqual(e['CR15_PREPARED'],str(root/'prepared'))
            self.assertIn('--nodes=2',a);self.assertIn('--gres=gpu:8',a);self.assertIn('--exclusive',a)
            self.assertIn('--time=1-00:00:00',a);self.assertIn('remaining<=399',result.stdout)
            status['world_size']=1;(folder/'status.json').write_text(json.dumps(status));(root/'trace').unlink()
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertFalse((root/'trace').exists())
